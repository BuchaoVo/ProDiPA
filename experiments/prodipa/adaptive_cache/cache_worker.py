"""Generate candidates or annotate selected trajectories using the existing PoE."""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM
from prodipa.unconditional.length_control import ProteinLengthLogitsProcessor

from common import BASES, CACHES, CONFIG, OLD, OUT, TEACHERS, dump, read, seqid, tsv
sys.path.insert(0, str(OLD))
from build_cache import canonicalize, seed_all
from prodipa.conditional.offline_opd import cache_record_path, load_cache_record


def loaders(paradigm, output):
    if paradigm == "conditional":
        from prollama_opd_config import load_opd_config
        from prollama_opd_data import ProLLaMAOpdDataCollator
        from prollama_opd_train import OpdTrainingArguments, _load_tokenizer, _load_teacher_model_and_specs
        config = load_opd_config(str(CONFIG))
        args = OpdTrainingArguments(output_dir=str(output), bf16=True, load_in_kbits=16, report_to=[])
        tok = _load_tokenizer(config.student.backbone_path)
        prompt = ProLLaMAOpdDataCollator(tok, config.prompt).prompt_ids
        kwargs = dict(max_new_tokens=config.generation.max_new_tokens, temperature=config.generation.student_temperature,
                      repetition_penalty=config.generation.repetition_penalty, top_p=config.generation.top_p, top_k=config.generation.top_k)
    else:
        from prodipa.unconditional.train import UnconditionalTrainingArguments, _load_tokenizer, _load_teacher_model_and_specs, _load_teacher_config
        config = _load_teacher_config(str(TEACHERS), 1.0)
        args = UnconditionalTrainingArguments(output_dir=str(output), bf16=True, report_to=[])
        tok = _load_tokenizer(BASES[paradigm])
        prompt = [tok.eos_token_id]
        kwargs = dict(max_new_tokens=512, temperature=1.0, repetition_penalty=1.2, top_p=0.95, top_k=500)
    return tok, prompt, kwargs, lambda: _load_teacher_model_and_specs(config, args)


def metadata(ids, prompt_length, tok, paradigm, index, seed, raw=None,
             natural_terminal=None):
    raw = ids if raw is None else raw
    completion = raw[prompt_length:].tolist()
    eos = tok.eos_token_id in completion
    if eos:
        completion = completion[:completion.index(tok.eos_token_id)]
    text = tok.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if paradigm == "conditional":
        match = re.search(r"Seq\s*=\s*<([^>]*)>", text)
        sequence = "".join(match.group(1).split()).upper() if match else ""
        closed = match is not None
    else:
        sequence = "".join(text.split()).upper()
        closed = True
    natural = eos if natural_terminal is None else bool(natural_terminal)
    valid = (50 <= len(sequence) <= 512
             and not (set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
             and closed and natural)
    return dict(sample_id=index, sequence_id=seqid(sequence), sequence=sequence, length=len(sequence),
                valid_complete=str(valid).lower(), seed=seed, wrapper_closed=closed,
                eos_reached=eos, natural_eos_reached=natural)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["decode", "rollout", "teacher"])
    p.add_argument("--paradigm", required=True, choices=list(BASES))
    p.add_argument("--directory", type=__import__("pathlib").Path, required=True)
    p.add_argument("--adapter")
    p.add_argument("--seed", type=int, default=900001)
    args = p.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    tok, prompt, generation, teacher_loader = loaders(args.paradigm, directory)
    if args.action == "decode":
        entries = []
        for index in range(512):
            r = load_cache_record(cache_record_path(CACHES[args.paradigm], index))
            entries.append(metadata(
                r["input_ids"], r["prompt_length"], tok, args.paradigm, index, 42,
                natural_terminal=r.get("supervise_terminal", True),
            ))
        tsv(directory / "candidates.tsv", entries)
        return
    if args.action == "rollout":
        student = AutoModelForCausalLM.from_pretrained(BASES[args.paradigm], torch_dtype=torch.bfloat16, local_files_only=True)
        student = PeftModel.from_pretrained(student, args.adapter, is_trainable=False).eval().to("cuda")
        entries = []
        (directory / "raw").mkdir(exist_ok=True)
        for start in range(0, 96, 8):
            seed_all(args.seed + start)
            prompts = torch.tensor([prompt] * 8, device="cuda")
            processors = []
            if args.paradigm == "unconditional":
                processors.append(ProteinLengthLogitsProcessor(tok, 512, len(prompt)))
            with torch.inference_mode():
                raw = student.generate(input_ids=prompts, attention_mask=torch.ones_like(prompts),
                                       **generation, do_sample=True, use_cache=True,
                                       eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                       **({"logits_processor": processors} if processors else {}))
            forced_flags = processors[0].forced_flags() if processors else [False] * len(raw)
            ids, _, lengths = canonicalize(raw, len(prompt), tok.eos_token_id, tok.pad_token_id)
            for j, length in enumerate(lengths):
                index = start + j
                completion = raw[j, len(prompt):].tolist()
                natural_terminal = tok.eos_token_id in completion and not forced_flags[j]
                item = dict(sample_id=index, input_ids=ids[j, :length].cpu(),
                            prompt_length=len(prompt),
                            supervise_terminal=bool(natural_terminal))
                torch.save(item, directory / "raw" / f"{index:06d}.pt")
                entries.append(metadata(
                    item["input_ids"], len(prompt), tok, args.paradigm, index,
                    args.seed + start, raw[j], natural_terminal=natural_terminal,
                ))
            print(f"candidate requests {start + 8}/96", flush=True)
        tsv(directory / "candidates.tsv", entries)
        return
    selection = read(directory / "selection.json")
    accepted = selection["accepted"]
    if not accepted:
        dump(directory / "TEACHER_COMPLETE.json", dict(accepted=0, teacher_scoring_tokens=0))
        return
    teacher, specs = teacher_loader()
    teacher.eval().to("cuda")
    (directory / "records").mkdir(exist_ok=True)
    token_count = 0
    for index in accepted:
        r = torch.load(directory / "raw" / f"{index:06d}.pt", map_location="cpu", weights_only=True)
        ids = r["input_ids"].unsqueeze(0).to("cuda")
        with torch.inference_mode():
            logits_list = []
            for spec in specs:
                teacher.set_adapter(spec.adapter_name)
                logits = teacher(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, r["prompt_length"] - 1:-1]
                logits_list.append(F.log_softmax(logits / spec.temperature, dim=-1))
            weights = torch.tensor([s.weight for s in specs], dtype=logits_list[0].dtype, device="cuda")
            target = F.log_softmax((weights.view(1, 1, -1, 1) * torch.stack(logits_list, dim=2)).sum(dim=2), dim=-1)
        r["target_log_probs"] = target[0].to(torch.bfloat16).cpu()
        path = directory / "records" / f"{index:06d}.pt"
        temp = path.with_suffix(".tmp")
        torch.save(r, temp)
        os.replace(temp, path)
        restored = load_cache_record(path)
        if not torch.equal(restored["target_log_probs"].to(target.device, target.dtype), target[0]):
            raise RuntimeError("serialized PoE differs from live full-vocabulary target")
        token_count += len(specs) * ids.numel()
        print(f"teacher annotated {index}", flush=True)
    dump(directory / "TEACHER_COMPLETE.json", dict(accepted=len(accepted), teacher_scoring_tokens=token_count,
                                                   consistency="exact serialized/live target equality"))


if __name__ == "__main__":
    main()
