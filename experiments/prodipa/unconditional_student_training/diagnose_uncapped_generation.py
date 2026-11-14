#!/usr/bin/env python3
"""ProDiPA uncapped student diagnostic; isolated from official evaluation tables."""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/prodipa/adaptive_cache'))
import common

SOURCE = ROOT/'outputs/experiments/unconditional_length_control_20261111'
OUT = ROOT/'outputs/experiments/unconditional_uncapped_diagnostic_20261111'
common.OUT = OUT
from common import dump, read, rows, tsv, sha, seqid, job, PY

AA = set('ACDEFGHIKLMNPQRSTVWY')
GENERATOR = ROOT/'experiments/prodipa/conditional_evaluation/generate_model.py'
BINS = [(0,49),(50,256),(257,512),(513,768),(769,1024),(1025,2048),(2049,10**9)]


def state(stage, **extra):
    payload = dict(stage=stage, updated_at=time.strftime('%FT%T%z'), **extra)
    dump(OUT/'STATUS.json', payload)
    print(payload, flush=True)


def repetition(sequence):
    n = len(sequence)
    counts = collections.Counter(sequence)
    kmers = collections.Counter(sequence[i:i+3] for i in range(max(0,n-2)))
    return dict(max_homopolymer_run=max((sum(1 for _ in group) for _,group in itertools.groupby(sequence)),default=0),
                dominant_aa_fraction=max(counts.values(),default=0)/n if n else None,
                aa_entropy_bits=-sum((v/n)*math.log2(v/n) for v in counts.values()) if n else None,
                repeated_3mer_fraction=1-len(kmers)/(n-2) if n>=3 else None)


def annotate(record):
    r = dict(record)
    sequence = r['sequence']
    natural = str(r['eos_reached']).lower() == 'true' and str(r['truncated']).lower() == 'false'
    canonical = bool(sequence) and set(sequence) <= AA
    # Do not use the formal generator's 50–512-aa validity gate in this diagnostic.
    r.update(sequence_id=seqid(sequence), length_aa=len(sequence),
             diagnostic_natural_eos=natural, diagnostic_canonical=canonical,
             diagnostic_structure_eligible=natural and canonical and len(sequence)>=50,
             **repetition(sequence))
    return r


def mean(values):
    values = [float(x) for x in values if x is not None and str(x) != '' and math.isfinite(float(x))]
    return statistics.fmean(values) if values else None


def summarize(records):
    lengths = sorted(int(r['length_aa']) for r in records)
    natural = [r for r in records if r['diagnostic_natural_eos']]
    return dict(requests=len(records), natural_eos=len(natural),
                token_budget_censored=len(records)-len(natural),
                natural_over_512=sum(int(r['length_aa'])>512 for r in natural),
                all_requests_over_512=sum(n>512 for n in lengths),
                mean_length=mean(lengths), median_length=statistics.median(lengths) if lengths else None,
                p90_length=lengths[math.ceil(.9*len(lengths))-1] if lengths else None,
                max_length=max(lengths,default=0),
                unique_sequences=len({r['sequence_id'] for r in records}),
                mean_repeated_3mer_fraction=mean(r['repeated_3mer_fraction'] for r in records),
                mean_aa_entropy_bits=mean(r['aa_entropy_bits'] for r in records))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'.pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=read(SOURCE/'evaluation/protocol/unconditional_evaluation.json')
    protocol=dict(model=str(SOURCE/'student'), adapter_sha256=sha(SOURCE/'student/adapter_model.safetensors'),
        backbone='__PRODIPA_MODELS__/ProtGPT2',requests=512,generation_max_aa=0,max_new_tokens=512,
        seed_rule='42 + 1009 * sample_id',temperature=.7,top_p=.9,top_k=200,repetition_penalty=1.2,
        structure_model=config['scorers']['structure']['model'],structure_recycles=4,structure_chunk_size=32,
        structure_eligibility='Natural EOS, canonical amino acids, length >= 50; no upper-aa filter; no cropping',
        structure_failures='Retain OOM/errors as missing, never assign zero or silently exclude from coverage',
        reference=str(SOURCE/'generation/opd.tsv'),
        reference_sha256=sha(SOURCE/'generation/opd.tsv'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [GENERATOR,ROOT/'scripts/run_esmfold_properties.py',Path(__file__)]},
        repetition_definition='Repeated 3-mer fraction = 1 - distinct overlapping 3-mers / (length - 2); length dependent, not a validated quality gate',
        confidence_gate='pLDDT >= 70 and pAE <= 10; prediction confidence only, not experimental validity',
        official_tables_changed=False)
    if (OUT/'PROTOCOL.json').exists() and read(OUT/'PROTOCOL.json')!=protocol:
        raise ValueError('Frozen diagnostic protocol or model changed')
    dump(OUT/'PROTOCOL.json',protocol)
    state('GENERATION_QUEUED_OR_RUNNING',requests=512,gpus='0,1,2,3',note='Wait for available GPU; no preemption')
    def generate(shard):
        target=OUT/f'generation/shard_{shard}.tsv'
        job(f'generation_{shard}',[PY,GENERATOR,'--paradigm','unconditional',
            '--model-name','protgpt2_student_uncapped_diagnostic','--base-model',protocol['backbone'],
            '--adapter',protocol['model'],'--out',target,'--num-sequences','512',
            '--sample-id-start',str(shard*128),'--sample-id-stop',str((shard+1)*128),
            '--generation-max-aa','0','--max-new-tokens','512'],
            'unconditional','student_uncapped','diagnostic',0,gpus=1,free_mib=16000)
        return rows(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=[annotate(r) for part in pool.map(generate,range(4)) for r in part]
    records.sort(key=lambda r:int(r['sample_id']))
    if {int(r['sample_id']) for r in records} != set(range(512)) or len(records)!=512:
        raise ValueError('Missing/duplicate generation requests')
    tsv(OUT/'sequence_diagnostics.tsv',records)
    # Capped reference only supports a policy comparison, not a natural-length baseline.
    capped=rows(SOURCE/'generation/opd.tsv')
    lookup={int(r['sample_id']):r for r in capped}
    paired=[]
    for r in records:
        old=lookup[int(r['sample_id'])]
        paired.append(dict(sample_id=r['sample_id'],capped_length=old['length_aa'],
            capped_forced_eos=old['length_forced_eos'],uncapped_length=r['length_aa'],
            uncapped_natural_eos=r['diagnostic_natural_eos'],identical_sequence=old['sequence']==r['sequence']))
    tsv(OUT/'paired_length_changes.tsv',paired)
    generation_summary=summarize(records)
    dump(OUT/'GENERATION_SUMMARY.json',generation_summary)
    # Reuse structure scores only for identical sequences from the same model/scoring run.
    reused={}
    for r in rows(SOURCE/'evaluation/methods/opd/scores/sequence_scores.tsv'):
        if r['valid_complete'].lower()=='true' and all(r.get(k) and math.isfinite(float(r[k])) for k in ['plddt','pae']):
            reused[seqid(r['sequence'])]=dict(sequence_id=seqid(r['sequence']),status='ok',
                fold_plddt_100=r['plddt'],fold_mean_pae=r['pae'],score_source='exact_sequence_previous_student_evaluation')
    unique={r['sequence_id']:r for r in records if r['diagnostic_structure_eligible']}
    pending=[dict(sequence_id=sid,sequence=r['sequence'],valid_complete='true') for sid,r in unique.items() if sid not in reused]
    tsv(OUT/'structure_input.tsv',pending,['sequence_id','sequence','valid_complete'])
    dump(OUT/'STRUCTURE_REUSE.json',dict(source=str(SOURCE/'evaluation/methods/opd/scores/sequence_scores.tsv'),
        source_sha256=sha(SOURCE/'evaluation/methods/opd/scores/sequence_scores.tsv'),
        reused_unique=sum(sid in reused for sid in unique),pending_unique=len(pending)))
    state('STRUCTURE_QUEUED_OR_RUNNING',**generation_summary,structure_pending_unique=len(pending))
    def structure(shard):
        if not pending or shard>=len(pending):return []
        target=OUT/f'structure/shard_{shard}.tsv'
        job(f'structure_{shard}',[PY,ROOT/'scripts/run_esmfold_properties.py','--input',OUT/'structure_input.tsv',
            '--out',target,'--model',protocol['structure_model'],'--chunk-size','32','--num-recycles','4',
            '--batch-size','1','--num-shards','4','--shard-index',str(shard),'--device','cuda:0'],
            'unconditional','student_uncapped','diagnostic',0,gpus=1,free_mib=48000)
        return rows(target)
    scores=dict(reused)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(structure,range(4)):
            scores.update({r['sequence_id']:dict(r,score_source='uncapped_full_sequence_esmfold') for r in part})
    for r in records:
        s=scores.get(r['sequence_id'],{}) if r['diagnostic_structure_eligible'] else {}
        r.update(structure_status=s.get('status','not_eligible'),plddt=s.get('fold_plddt_100',''),
            pae=s.get('fold_mean_pae',''),structure_error=s.get('error_reason',''),structure_score_source=s.get('score_source',''))
    tsv(OUT/'sequence_diagnostics.tsv',records)
    bins=[]
    for lo,hi in BINS:
        for ending in ['natural','token_budget']:
            subset=[r for r in records if lo<=int(r['length_aa'])<=hi and r['diagnostic_natural_eos']==(ending=='natural')]
            ok=[r for r in subset if r['structure_status']=='ok']
            bins.append(dict(length_bin=f'{lo}-{hi}' if hi<10**9 else f'>={lo}',ending=ending,
                requests=len(subset),structure_eligible=sum(r['diagnostic_structure_eligible'] for r in subset),
                structure_scored=len(ok),structure_failed=sum(r['structure_status']=='failed' for r in subset),
                mean_plddt=mean(r['plddt'] for r in ok),mean_pae=mean(r['pae'] for r in ok),
                confident_structure_count=sum(float(r['plddt'])>=70 and float(r['pae'])<=10 for r in ok),
                mean_repeated_3mer_fraction=mean(r['repeated_3mer_fraction'] for r in subset),
                mean_aa_entropy_bits=mean(r['aa_entropy_bits'] for r in subset)))
    tsv(OUT/'length_bin_summary.tsv',bins)
    failures=sum(r['diagnostic_structure_eligible'] and r['structure_status']!='ok' for r in records)
    dump(OUT/'SUMMARY.json',dict(**generation_summary,length_bins=bins,structure_missing_requests=failures,
        interpretation='Compare natural >512-aa structure confidence and repetition by length; token-budget outputs are right-censored, not proof of nontermination. Structure confidence is not function or wet-lab validation.',
        official_tables_changed=False))
    state('COMPLETE_WITH_STRUCTURE_FAILURES' if failures else 'COMPLETE',**generation_summary,structure_missing_requests=failures)


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('FAILED',error=repr(error));raise
