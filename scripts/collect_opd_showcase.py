#!/usr/bin/env python3
"""Collect traced ProDiPA showcase candidates; export ESMFold structures separately.

This is post-hoc showcase selection, not an unbiased evaluation or a new training run.
Historical scores are immutable; regenerated structural scores live in a separate file.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/showcase/multiteacher_opd_20261110'
FULL = ROOT / 'outputs/experiments/offline_multiteacher_opd_full'


def read_tsv(path):
    with path.open() as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f, delimiter='\t')]


def write_tsv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def dump(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def prepare(out):
    if (out / 'selection.json').exists():
        raise FileExistsError('Selection already frozen; use --export or a new --out')
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('fasta', 'pdb', 'confidence', 'figures', 'pymol'):
        (out / folder).mkdir(exist_ok=True)
    selected, eligible, cohorts, seen = [], [], {}, set()
    for par in ('conditional', 'unconditional'):
        for mode in ('online', 'offline'):
            group = par + '_' + mode
            model = ('prollama' if par == 'conditional' else 'protgpt2') + '_multiteacher_student'
            source = (ROOT / f'outputs/evaluation/{par}/methods/{model}/results/sequence_scores.tsv'
                      if mode == 'online' else FULL / f'evaluation/methods/{par}/prodipa_offline/results/sequence_scores.tsv')
            novelty_path = (ROOT / f'outputs/evaluation/{par}/scores/novelty_u.tsv'
                            if mode == 'online' else FULL / f'evaluation/novelty/{par}_novelty_u.tsv')
            novelty = {r['sequence_id']: r for r in read_tsv(novelty_path)}
            candidates = []
            records = read_tsv(source)
            for row in records:
                seq = row['sequence']
                if row['valid_complete'].lower() != 'true' or not 80 <= len(seq) <= 400:
                    continue
                try:
                    plddt, pae, sol, thermo = (float(row[k]) for k in ('plddt', 'pae', 'sol', 'thermo'))
                except (ValueError, KeyError):
                    continue
                if not all(map(math.isfinite, (plddt, pae, sol, thermo))):
                    continue
                if plddt < 80 or pae > 8 or sol < .7 or thermo < .7:
                    continue
                if max(Counter(seq).values())/len(seq) > .4:
                    continue
                if max(Counter(seq[i:i+3] for i in range(len(seq)-2)).values())/(len(seq)-2) > .2:
                    continue
                digest = hashlib.sha256(seq.encode()).hexdigest()
                assert row['sequence_id'] == 'sha256:' + digest
                nov = novelty.get(row['sequence_id'], {})
                candidates.append(dict(candidate_id=f'{group}_{int(row["sample_id"]):04d}',
                    group=group, sample_id=row['sample_id'], request_id=row['request_id'],
                    sequence_id=row['sequence_id'], seed=row['seed'], length_aa=len(seq),
                    plddt=plddt, pae=pae, sol=sol, thermo=thermo, ppl=row['ppl'],
                    protrek_percent=row.get('protrek_percent', ''),
                    novelty_u_percent=nov.get('novelty_percent', row.get('novelty_u_percent', '')),
                    novelty_t_percent=row.get('novelty_t_percent', ''),
                    uniprot_best_hit=nov.get('best_reference_id', ''),
                    novelty_search_status=nov.get('status', ''),
                    balance_min=min(plddt/100, sol, thermo),
                    balance_geomean=(plddt/100*sol*thermo)**(1/3),
                    source_scores=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    novelty_source=str(novelty_path), sequence=seq))
            candidates.sort(key=lambda r: (-r['balance_min'], -r['balance_geomean'], r['pae'], r['sample_id']))
            cohorts[group] = dict(requests=len(records), eligible=len(candidates))
            eligible.extend(candidates)
            count = 0
            for row in candidates:
                if row['sequence_id'] in seen:
                    continue
                selected.append(row)
                seen.add(row['sequence_id'])
                count += 1
                if count == 2:
                    break
    assert len(selected) == 8, 'Fewer than eight unique qualifying candidates; do not relax gates silently'
    write_tsv(out / 'eligible_candidates.tsv', eligible)
    write_tsv(out / 'selected_candidates.tsv', selected)
    fasta = ''
    for row in selected:
        record = f'>{row["candidate_id"]} {row["sequence_id"]}\n{row["sequence"]}\n'
        (out / 'fasta' / (row['candidate_id'] + '.fasta')).write_text(record)
        fasta += record
    (out / 'selected.fasta').write_text(fasta)
    dump(out / 'selection.json', dict(created_at=time.strftime('%F %T %z'), cohorts=cohorts,
        gates=dict(valid_complete=True, length_aa=[80,400], plddt_min=80, pae_max=8,
                   sol_min=.7, thermo_min=.7, max_residue_fraction=.4, max_trimer_fraction=.2),
        ranking='descending min(pLDDT/100,Sol,Thermo), geometric mean tie-break, then lower pAE; top two unique sequences per group',
        caveat='Post-hoc showcase, not representative averages, not wet-lab validated; no claim of Pareto optimality or causal mechanism',
        candidates=selected))
    dump(out / 'STATUS.json', dict(stage='SEQUENCES_READY_PDB_PENDING', selected=len(selected)))
    print(json.dumps(cohorts, indent=2), flush=True)


def export(out):
    # Reserve through both existing locks; never interrupt evaluation workers.
    import fcntl
    own_lock = (out / '.export.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from types import SimpleNamespace
    from run_esmfold_properties import reserve_gpus, load_model, extract_metrics
    args = SimpleNamespace(min_free_memory_gib=24, max_gpu_utilization=95,
                           model=Path('__PRODIPA_MODELS__/esmfold_v1_hf'),
                           chunk_size=32, num_recycles=4, tf32=None, device='cuda:0')
    dump(out / 'STATUS.json', dict(stage='WAITING_FOR_GPU_0_3', updated_at=time.time()))
    while True:
        devices, locks = reserve_gpus(args, 1)
        if devices:
            break
        print('Waiting for a lock-free GPU 0-3 with >=24 GiB free and <=95% utilization', flush=True)
        time.sleep(30)
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    started = time.time()
    dump(out / 'STATUS.json', dict(stage='EXPORTING_PDB', gpu=devices[0], started_at=started))
    tokenizer, model, device = load_model(args)
    records = json.loads((out / 'selection.json').read_text())['candidates']
    summaries = []
    for row in records:
        name, seq = row['candidate_id'], row['sequence']
        inputs = tokenizer([seq], return_tensors='pt', add_special_tokens=False).to(device)
        with torch.inference_mode():
            prediction = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], num_recycles=4)
        metrics = extract_metrics(prediction, [len(seq)])[0]
        # This local checkpoint emits confidence in [0,1]; PDB/PyMOL uses [0,100].
        pdb_output = dict(prediction)
        if float(prediction.plddt.detach().float().mean()) <= 1.0:
            pdb_output['plddt'] = prediction.plddt * 100.0
        pdb_text = model.output_to_pdb(pdb_output)[0]
        pdb_path = out / 'pdb' / (name + '.pdb')
        pdb_path.write_text(pdb_text)
        structure = PDBParser(QUIET=True).get_structure(name, str(pdb_path))
        residues = list(structure.get_residues())
        assert ''.join(seq1(r.resname) for r in residues) == seq, 'PDB sequence mismatch'
        ca = np.array([r['CA'].bfactor for r in residues])
        assert np.isfinite(ca).all() and 1 < ca.mean() <= 100, 'PDB confidence scale mismatch'
        pae = prediction.predicted_aligned_error[0].detach().float().cpu().numpy()
        np.save(out / 'confidence' / (name + '_pae.npy'), pae)
        np.save(out / 'confidence' / (name + '_ca_plddt.npy'), ca)
        write_tsv(out / 'confidence' / (name + '_residues.tsv'), [
            dict(residue_index=i+1, amino_acid=aa, ca_plddt=float(ca[i])) for i, aa in enumerate(seq)])
        # Geometric charged contacts only, NOT an energy calculation or predicted Tm.
        acidic = [a for r in residues if r.resname in ('ASP','GLU') for a in r
                  if a.name in ('OD1','OD2','OE1','OE2') and r['CA'].bfactor >= 70]
        basic = [a for r in residues if r.resname in ('LYS','ARG') for a in r
                 if a.name in ('NZ','NE','NH1','NH2') and r['CA'].bfactor >= 70]
        pairs = {}
        for a in acidic:
            for b in basic:
                dist = float(a-b)
                if dist <= 4:
                    key = (a.parent.id[1], b.parent.id[1])
                    if key not in pairs or dist < pairs[key]['distance_A']:
                        pairs[key] = dict(acid_residue=key[0], acid_atom=a.name,
                            base_residue=key[1], base_atom=b.name, distance_A=dist)
        contacts = list(pairs.values())
        dump(out / 'confidence' / (name + '_charged_contacts.json'), contacts)
        fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
        axes[0].plot(np.arange(1,len(seq)+1), ca, color='#2469a5')
        axes[0].set(xlabel='Residue', ylabel='C-alpha pLDDT', ylim=(0,100), title='Local model confidence')
        im = axes[1].imshow(pae, vmin=0, vmax=30, cmap='viridis_r', origin='lower')
        axes[1].set(xlabel='Residue index (0-based)', ylabel='Residue index (0-based)', title='Predicted aligned error')
        fig.colorbar(im, ax=axes[1], label='PAE (angstrom)')
        fig.suptitle(name)
        fig.savefig(out / 'figures' / (name + '_confidence.png'), dpi=200)
        plt.close(fig)
        plddt = float(metrics['fold_plddt_100'])
        ptm = float(metrics['fold_ptm_01'])
        mean_pae = float(metrics['fold_mean_pae'])
        dump(out / 'confidence' / (name + '_structure_metrics.json'), dict(
            candidate_id=name, sequence_id=row['sequence_id'], length_aa=len(seq),
            fold_plddt_100=plddt, fold_ptm_01=ptm, fold_mean_pae=mean_pae,
            pae_matrix_shape=list(pae.shape), ca_plddt_shape=list(ca.shape),
            confidence_source='local Hugging Face ESMFold; one independent structure inference',
            pdb_b_factor_definition='C-alpha pLDDT on a 0-100 scale'))
        summary = dict(candidate_id=name, pdb=str(pdb_path), sequence_verified=True,
            pdb_sha256=hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
            historical_plddt=row['plddt'], regenerated_plddt=plddt,
            regenerated_ptm=ptm,
            historical_pae=row['pae'], regenerated_pae=mean_pae,
            delta_plddt=plddt-row['plddt'], delta_pae=mean_pae-row['pae'],
            ca_plddt_mean=float(ca.mean()), confident_charged_contact_pairs=len(contacts),
            structural_gate_reconfirmed=plddt>=80 and mean_pae<=8,
            review_needed=abs(plddt-row['plddt'])>2 or abs(mean_pae-row['pae'])>1)
        summaries.append(summary)
        write_tsv(out / 'structure_audit.tsv', summaries)
        dump(out / 'STATUS.json', dict(stage='EXPORTING_PDB', completed=len(summaries), total=len(records), gpu=devices[0]))
        print(f'{name} PDB sequence verified; pLDDT={plddt:.3f}; pAE={mean_pae:.3f}; contacts={len(contacts)}', flush=True)
        del prediction, pdb_output
    elapsed = time.time()-started
    dump(out / 'export_protocol.json', dict(model=str(args.model), num_recycles=4, chunk_size=32,
        gpu_uuid=devices[0], gpu_hours=elapsed/3600, scope='showcase structure re-export, excluded from experiment costs',
        torch=torch.__version__, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        sequence_verified_count=len(summaries), historical_scores_unchanged=True,
        pdb_b_factor_definition='pLDDT scaled to 0-100, not experimental B-factor'))
    dump(out / 'STATUS.json', dict(stage='PDB_COMPLETE', completed=len(summaries),
        review_needed=[r['candidate_id'] for r in summaries if r['review_needed'] or not r['structural_gate_reconfirmed']],
        pymol_rendering='NOT_RUN; run render_pymol.py in your PyMOL installation', completed_at=time.time()))
    print('PDB export COMPLETE', flush=True)
    for lock in locks:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    try:
        export(args.out) if args.export else prepare(args.out)
    except Exception as error:
        if args.export:
            dump(args.out / 'STATUS.json', dict(stage='FAILED', error=repr(error)))
        raise
