"""Run the frozen existing predictors with explicit training/evaluation accounting."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from common import ROOT, OUT, PY, TEMBERTURE_PY, dump, job, read, rows, seqid, tsv, status

SCORERS = ROOT / "experiments/prodipa/conditional_evaluation"
FIELDS = {
    "structure": {"fold_plddt_100": "plddt", "fold_mean_pae": "pae"},
    "sol": {"solubility_scaled_01": "sol"},
    "thermo": {"thermophilicity_score_01": "thermo"},
    "ppl": {"ppl": "ppl"},
    "protrek": {"protrek_percent": "protrek_percent"},
    "novelty_u": {"novelty_percent": "novelty_u_percent"},
    "novelty_t": {"novelty_percent": "novelty_t_percent"},
}


def score(input_path, directory, paradigm, method, phase, node):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "sequence_scores.tsv"
    if (directory / "COMPLETE.json").exists():
        return rows(final)
    inputs = rows(input_path)
    for r in inputs:
        r["sequence_id"] = seqid(r["sequence"])
        r["length"] = len(r["sequence"])
    unique = {r["sequence_id"]: r for r in inputs if r["valid_complete"].lower() == "true"}
    # Only reuse score evidence from this frozen protocol and paradigm.
    store_file = OUT / "score_store" / paradigm / f"{phase}.json"
    store = read(store_file) if store_file.exists() else {}
    required = ["plddt", "pae", "sol", "thermo", "novelty_u_percent", "novelty_t_percent"]
    if phase == "evaluation":
        required += ["ppl"] + (["protrek_percent"] if paradigm == "conditional" else [])
    missing = [r for sid, r in unique.items() if not all(k in store.get(sid, {}) for k in required)]
    prepared = directory / "unique_valid_sequences.tsv"
    tsv(prepared, missing, ["sequence_id", "sequence", "length", "valid_complete"])
    config = read(OUT / "protocol" / f"{paradigm}_evaluation.json")
    files = {}
    tasks = []
    prefix = f"{paradigm}_{method}_{phase}_{node}_{directory.name}"
    def add(label, command, gpus=0, memory=8000):
        tasks.append((label, command, gpus, memory))
    if missing:
        for shard in range(4):
            target = directory / f"structure_{shard}.tsv"
            files[f"structure_{shard}"] = target
            add(f"structure_{shard}", [PY, ROOT / "scripts/run_esmfold_properties.py", "--input", prepared,
                "--out", target, "--model", config["scorers"]["structure"]["model"], "--chunk-size", "32",
                "--num-recycles", "4", "--num-shards", "4", "--shard-index", str(shard),
                "--device", "cuda:0"], 1, 28000)
        files["sol"] = directory / "sol.tsv"
        add("sol", [PY, ROOT / "scripts/run_protein_sol.py", "--input", prepared, "--out", files["sol"], "--work", directory / "sol_work"])
        files["thermo"] = directory / "thermo.tsv"
        add("thermo", [TEMBERTURE_PY, ROOT / "scripts/run_temBERTure_cls.py", "--input", prepared, "--out", files["thermo"],
            "--repo", ROOT / "tools/TemBERTure", "--batch-size", "8", "--device", "cuda"], 1)
        for kind in ("novelty_u", "novelty_t"):
            files[kind] = directory / f"{kind}.tsv"
            ref = (["--reference-db", config["references"]["novelty_u_database"], "--reference-label", config["references"]["novelty_u_label"]]
                   if kind == "novelty_u" else ["--reference", ROOT / config["references"]["novelty_t_fasta"], "--reference-label", paradigm + "_teacher_train_union_300x3"])
            add(kind, [PY, SCORERS / "score_mmseqs_novelty.py", "--input", prepared, "--out", files[kind],
                "--mmseqs", ROOT / "tools/mmseqs/bin/mmseqs", "--threads", "16", "--split-memory-limit", "150G",
                "--tmp-dir", directory / "mmseqs_tmp", *ref])
        if phase == "evaluation":
            files["ppl"] = directory / "ppl.tsv"
            add("ppl", [PY, SCORERS / "score_protgpt2_ppl.py", "--input", prepared, "--out", files["ppl"],
                "--model", config["references"]["ppl_model"]], 1)
            if paradigm == "conditional":
                files["protrek"] = directory / "protrek.tsv"
                add("protrek", [PY, SCORERS / "score_protrek_similarity.py", "--input", prepared, "--output", files["protrek"],
                    "--repo", ROOT / "third_party/ProTrek", "--model", "__PRODIPA_MODELS__/ProTrek_35M", "--batch-size", "32", "--device", "cuda:0"], 1)
        def execute(task):
            label, command, gpus, memory = task
            job(prefix + "_" + label, command, paradigm, method, phase, node, gpus, memory)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(execute, task): task[0] for task in tasks}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    # Report failure immediately while other running scorers drain.
                    status(f"{paradigm}/{method}/{phase}/{node}/scoring", "FAILED",
                           failed_task=futures[future], error=repr(error))
                    raise
        for label, file in files.items():
            mapping = FIELDS["structure" if label.startswith("structure_") else label]
            for r in rows(file):
                if r.get("status", "ok") not in ("ok", "no_hit"):
                    raise RuntimeError(f"retryable scoring failure in {file}: {r}")
                dst = store.setdefault(r["sequence_id"], {})
                for source, field in mapping.items():
                    value = float(r[source])
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite {source} in {file}")
                    dst[field] = value
        dump(store_file, store)
    for r in inputs:
        if r["valid_complete"].lower() == "true":
            r.update(store.get(r["sequence_id"], {}))
            if not all(k in r for k in required):
                raise RuntimeError(f"missing score for {r['sequence_id']}")
    fields = list(dict.fromkeys([k for r in inputs for k in r] + required))
    tsv(final, inputs, fields)
    dump(directory / "COMPLETE.json", {"requests": len(inputs), "unique_valid": len(unique),
                                      "newly_scored": len(missing), "phase": phase})
    return inputs
