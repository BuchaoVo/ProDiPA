#!/usr/bin/env python3
"""Frozen original ProtGPT2/Prefix teacher distribution and termination audit."""
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT/'experiments/prodipa/conditional_multiteacher_student/diagnose_distributions.py'
# Load just the reusable, tested full-vocabulary metric helpers; do not run its main.
import importlib.util
spec = importlib.util.spec_from_file_location('conditional_distribution_metrics', SHARED)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
OUT = ROOT/'outputs/diagnostics/unconditional_multiteacher_distributions'
NAMES = ['fold','sol','thermo','student']


def dump(name, value):
    temporary = OUT/(name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(OUT/name)


def positions(tokenizer, ids, eos, censored):
    result = []
    previous = ''
    for i,target in enumerate(ids):
        decoded = tokenizer.decode(ids[:i+1],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        before = len(''.join(previous.split()))
        after = len(''.join(decoded.split()))
        region = 'eos' if target==eos else ('protein' if after>before else 'whitespace')
        result.append(dict(position=i,aa_before=before,region=region,target_id=target,
                           target_text=tokenizer.decode([target]),context_tail=previous[-48:],observed_target=True))
        previous=decoded
    if censored:
        result.append(dict(position=len(ids),aa_before=len(''.join(previous.split())),region='budget_endpoint',
                           target_id='',target_text='',context_tail=previous[-48:],observed_target=False))
    total = max(1,len(''.join(previous.split())))
    for row in result:
        row['position_decile']=min(9,int(10*row['aa_before']/total))
        row['aa_bin']=next((label for limit,label in [(50,'000-049'),(128,'050-127'),(256,'128-255'),
                               (384,'256-383'),(512,'384-511')] if row['aa_before']<limit), '512+')
    return result


def summarize(rows, trajectories):
    frame=pd.DataFrame(rows)
    frame['intervention_candidate']=((frame.distill_teacher_js>=.05)&(frame.distill_student_poe_js>=.05)).astype(float)
    body=frame[frame.region=='protein']
    keys=[k for k in rows[0] if k.endswith(('_js','_entropy','_prob','_nll','_rank')) or k in
          ('teacher_top1_disagreement','confident_teacher_conflict','student_poe_top1_disagreement',
           'teacher_sampled_logp_range','poe_suppresses_teacher_top1_10x','poe_eos_vs_teacher_mean_ratio',
           'student_eos_vs_poe_ratio')] + ['intervention_candidate']
    seq=body.groupby(['producer','sample_id'])[keys].mean().reset_index()
    seq.to_csv(OUT/'sequence_summary.tsv',sep='\t',index=False)
    rng=np.random.default_rng(20261111)
    summary=[]
    for producer,part in seq.groupby('producer'):
        for key in keys:
            values=part[key].to_numpy()
            boots=rng.choice(values,size=(1000,len(values)),replace=True).mean(1)
            summary.append(dict(producer=producer,metric=key,sequences=len(values),mean=float(values.mean()),
                                ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975))))
    shared.write_table(OUT/'summary.tsv',summary)
    for group,filename in [('position_decile','position_summary.tsv'),('aa_bin','absolute_position_summary.tsv'),
                           ('region','region_summary.tsv'),('length_group','length_group_summary.tsv')]:
        source=frame if group=='region' else body
        per=source.groupby(['producer','sample_id',group])[keys].mean().reset_index()
        means=per.groupby(['producer',group])[keys].mean()
        means['sequences']=per.groupby(['producer',group]).size()
        means.to_csv(OUT/filename,sep='\t')
    boundary=[]
    persistent=[]
    for (producer,sample),part in frame.groupby(['producer','sample_id']):
        if part.length_aa.iloc[0]>512:
            choices=part[part.aa_before<=512].sort_values('position')
            if len(choices):
                boundary.append(choices.iloc[-1].to_dict())
        longest=streak=0
        previous=-2
        for row in part[part.region=='protein'].sort_values('position').itertuples():
            streak=(streak+1 if row.position==previous+1 else 1) if row.intervention_candidate else 0
            longest=max(streak,longest);previous=row.position
        persistent.append(dict(producer=producer,sample_id=int(sample),max_candidate_run=longest))
    shared.write_table(OUT/'boundary_512.tsv',boundary)
    shared.write_table(OUT/'persistent_states.tsv',persistent)
    body.sort_values(['distill_teacher_js','distill_student_poe_js'],ascending=False).head(100).to_csv(OUT/'hotspots.tsv',sep='\t',index=False)
    frame[frame.region.isin(['eos','budget_endpoint'])].to_csv(OUT/'endpoints.tsv',sep='\t',index=False)
    lines=['# Original Unconditional multi-teacher distribution diagnostic','',
           'Frozen original main-table student and its original three Prefix teachers; no training, sampling, '
           'weight updates or official table edits. Reconstructed text prefixes, not original sampling token IDs.', '',
           'Four producers × 64 randomly chosen recorded requests (seed 20261111), without length/validity filtering. '
           'Natural EOS is appended only when recorded; truncated trajectories receive a separately labeled last-context '
           'probe with no observed target and blank sampled-token metrics. No artificial EOS supervision is applied.', '',
           'Full-vocabulary FP32 metrics, original teacher/student temperatures 1.0, original equal-weight PoE. '
           'No top-k/top-p/repetition penalty is applied to diagnostic probabilities. '
           'Protein rows include residue-bearing BPE tokens; whitespace and EOS are separate. '
           'Relative positions use observed length; truncated trajectories are censored, not completed proteins.', '',
           '| Producer | Body sequences | Teacher JS | Student–PoE JS | Top-1 disagreement | Confident conflict | Candidate states |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,part in seq.groupby('producer'):
        mean=part.mean(numeric_only=True)
        lines.append(f'| {name} | {len(part)} | {mean.distill_teacher_js:.6f} | {mean.distill_student_poe_js:.6f} | '
                     f'{100*mean.teacher_top1_disagreement:.3f}% | {100*mean.confident_teacher_conflict:.3f}% | {100*mean.intervention_candidate:.3f}% |')
    lines += ['', 'JS/entropy units: nats. Primary means are averaged within each sequence then across sequences. '
              'summary.tsv includes sequence-bootstrap 95% intervals (1000 resamples). '
              'Confident conflict means two teachers have different top tokens, each with probability ≥0.5. '
              'Candidate screening uses teacher JS ≥0.05 and student–PoE JS ≥0.05, as in the Conditional diagnostic. '
              'These thresholds are descriptive, not calibrated biological-quality or intervention gates.', '',
              'Low next-token EOS probability by itself does not prove runaway generation. Interpret absolute length, '
              'natural termination versus token-budget censoring, teacher EOS targets and student alignment jointly. '
              'Distribution conflict is not gradient conflict; this run computes no gradients and cannot establish '
              'that a proposed intervention improves Fold/Sol/Thermo.', '',
              f'Positions: {len(frame)}; protein-body positions: {len(body)}. See trajectories.tsv, summary.tsv, '
              'region_summary.tsv, absolute_position_summary.tsv, position_summary.tsv, boundary_512.tsv, '
              'endpoints.tsv, persistent_states.tsv and per_token.tsv.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def decompose(log_probs, poe, weights, groups):
    """Separate residue-content distributions from EOS/whitespace event choices."""
    result={}
    all_distributions=log_probs+[poe]
    residue=[F.log_softmax(p[:,groups[0]],-1) for p in all_distributions]
    categorical=[torch.stack([torch.logsumexp(p[:,indices],-1) for indices in groups],-1)
                 for p in all_distributions]
    w=torch.tensor(weights,device=poe.device).view(3,1,1)
    teacher_residue=torch.stack(residue[:3])
    mixture=torch.logsumexp(teacher_residue+w.log(),0)
    result['residue_teacher_js']=(w.squeeze(-1)*(teacher_residue.exp()*(teacher_residue-mixture)).sum(-1)).sum(0)
    result['residue_student_poe_js']=shared.js(residue[3],residue[4])
    result['boundary_student_poe_js']=shared.js(categorical[3],categorical[4])
    for name,cp in zip(NAMES+['poe'],categorical):
        result[name+'_residue_mass_prob']=cp[:,0].exp()
        result[name+'_whitespace_prob']=cp[:,1].exp()
    return {k:v.cpu().tolist() for k,v in result.items()}


def execute():
    started=time.time();gpu_lock=None
    status=dict(state='PREPARING',pid=os.getpid(),completed=0,total=256)
    dump('STATUS.json',status)
    try:
        student_path=ROOT/'outputs/unconditional/protgpt2_multiteacher_student'
        manifest=json.loads((student_path/'RUN_MANIFEST.json').read_text())
        config_path=ROOT/'src/prodipa/unconditional/teachers.yaml'
        config=yaml.safe_load(config_path.read_text());specs=config['teachers']
        assert [s['name'] for s in specs]==NAMES[:3]
        adapters={s['name']:Path(s['adapter_path']) for s in specs};adapters['student']=student_path
        hashes={k:shared.sha(v/'adapter_model.safetensors') for k,v in adapters.items()}
        for teacher in manifest['teacher_adapters']:
            if hashes[teacher['name']]!=teacher['adapter_weights_sha256']:
                raise ValueError('Current teacher weights do not match original student training')
        sources={name:ROOT/f'outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers/generation/unconditional/{name}_teacher.tsv'
                 for name in NAMES[:3]};sources['student']=student_path/'generation.tsv'
        rng=random.Random(20261111);records=[];source_metadata={}
        for name,path in sources.items():
            metadata=json.loads(path.with_suffix('.metadata.json').read_text())
            if Path(metadata['adapter']).resolve()!=adapters[name].resolve():
                raise ValueError(f'Adapter mismatch: {name}')
            with path.open() as f:
                cohort=list(csv.DictReader(f,delimiter='\t'))
            records.extend(dict(row,producer=name) for row in sorted(rng.sample(cohort,64),key=lambda r:int(r['sample_id'])))
            source_metadata[name]=dict(path=str(path),sha256=shared.sha(path),metadata=metadata)
        weights=[float(s['weight']) for s in specs]
        temps=[float(s['temperature']) for s in specs]+[float(manifest['script_arguments']['temperature'])]
        if abs(sum(weights)-1)>1e-8:
            raise ValueError('Teacher weights must sum to one')
        dump('PROTOCOL.json',dict(no_training=True,student=str(student_path),weights=weights,temperatures=temps,
             adapters={k:dict(path=str(v),sha256=hashes[k]) for k,v in adapters.items()},sources=source_metadata,
             seed=20261111,samples_per_producer=64,config_sha256=shared.sha(config_path),
             source_sha256=shared.sha(Path(__file__)),shared_metric_sha256=shared.sha(SHARED),
             representation='re-encoded recorded text; special BOS/EOS/PAD ID equality is handled by explicit lengths',
             vocabulary_partition='canonical-AA-bearing (after removing whitespace), pure whitespace, EOS, other; residue JS conditions each distribution on first group',
             selection='all requests eligible, including overlong and token-budget-censored',
             gate=dict(teacher_js_min=.05,student_poe_js_min=.05,region='protein')))
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        _,uuid,free,util=next([x.strip() for x in line.split(',')] for line in query.splitlines() if line.startswith('0,'))
        if int(free)<16000 or int(util)>60:
            raise RuntimeError('GPU 0 has insufficient free headroom; existing jobs are untouched')
        gpu_lock=open(f'/tmp/prodipa_gpu_{uuid}.lock','a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        torch.set_num_threads(4)
        base=config['teacher_backbone_path']
        tok=AutoTokenizer.from_pretrained(base,local_files_only=True);eos=tok.eos_token_id
        groups=[[],[],[eos],[]]
        for token in range(len(tok)):
            if token==eos:continue
            decoded=tok.decode([token]);compact=''.join(decoded.split())
            group=0 if compact and not(set(compact)-set('ACDEFGHIKLMNPQRSTVWY')) else (1 if decoded and not compact else 3)
            groups[group].append(token)
        if any(not group for group in groups):raise ValueError('Unexpected empty token category')
        teacher=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,local_files_only=True),
                     str(adapters['fold']),adapter_name='fold',is_trainable=False).cuda().eval()
        for name in NAMES[1:3]:
            teacher.load_adapter(str(adapters[name]),adapter_name=name,is_trainable=False)
        student=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,local_files_only=True),
                     str(student_path),is_trainable=False).cuda().eval()
        teacher.requires_grad_(False);student.requires_grad_(False)
        all_rows=[];trajectories=[]
        with (OUT/'per_token.tsv').open('w',newline='') as f:
            writer=None
            for number,record in enumerate(records):
                text=record['raw_completion'].replace('\\n','\n')
                ids=tok.encode(text,add_special_tokens=False)
                if tok.decode(ids,clean_up_tokenization_spaces=False)!=text or eos in ids:
                    raise ValueError('Raw trajectory must round-trip without embedded EOS')
                natural=record['eos_reached']=='true'
                if natural: ids.append(eos)
                locs=positions(tok,ids,eos,not natural)
                if 1+len(ids)+100>teacher.config.n_positions:
                    raise ValueError('Prefix-aware context limit exceeded; refusing truncation')
                input_ids=torch.tensor([[eos]+ids],device='cuda')
                points=len(locs)
                targets=ids if natural else ids+[eos]  # dummy probe target is removed from reported metrics
                with torch.inference_mode():
                    logits=[]
                    for name in NAMES[:3]:
                        teacher.set_adapter(name);teacher.requires_grad_(False).eval()
                        logits.append(teacher(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False).logits[0,:points].float())
                    logits.append(student(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False).logits[0,:points].float())
                    metrics=shared.measures(logits,torch.tensor(targets,device='cuda'),temps,weights,{'eos':eos,'close':[]})
                    for j,name in enumerate(NAMES):
                        metrics[name+'_eos_rank']=(1+(logits[j]>logits[j][:,eos,None]).sum(-1)).cpu().tolist()
                    lp=[F.log_softmax(logits[j]/temps[j],-1) for j in range(4)]
                    poe=F.log_softmax(sum(w*p for w,p in zip(weights,lp)),-1)
                    metrics.update(decompose(lp,poe,weights,groups))
                    metrics['poe_eos_rank']=(1+(poe>poe[:,eos,None]).sum(-1)).cpu().tolist()
                for key in list(metrics):
                    if key.endswith('_close_prob'):del metrics[key]
                rows=[]
                for pos,loc in enumerate(locs):
                    values={k:v[pos] for k,v in metrics.items()}
                    if not loc['observed_target']:
                        for key in values:
                            if key.endswith('_nll') or key=='teacher_sampled_logp_range':values[key]=float('nan')
                    values['teacher_mean_eos_prob']=sum(weights[i]*values[NAMES[i]+'_eos_prob'] for i in range(3))
                    values['poe_eos_vs_teacher_mean_ratio']=values['poe_eos_prob']/max(values['teacher_mean_eos_prob'],1e-30)
                    values['student_eos_vs_poe_ratio']=values['student_eos_prob']/max(values['poe_eos_prob'],1e-30)
                    rows.append(dict(producer=record['producer'],sample_id=int(record['sample_id']),length_aa=int(record['length_aa']),
                                     length_group='>512' if int(record['length_aa'])>512 else '<=512',**loc,**values))
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader()
                writer.writerows(rows);f.flush();all_rows.extend(rows)
                trajectories.append(dict(producer=record['producer'],sample_id=int(record['sample_id']),sequence=record['sequence'],
                    length_aa=int(record['length_aa']),natural_eos=natural,truncated=record['truncated'],valid_complete=record['valid_complete'],
                    completion_tokens=len(ids),protein_tokens=sum(r['region']=='protein' for r in rows)))
                del logits,lp,poe,input_ids
                status.update(state='RUNNING',completed=number+1,elapsed_seconds=time.time()-started,updated_at_unix=time.time(),gpu=0)
                dump('STATUS.json',status)
                print(f'{number+1}/256 {record["producer"]}/{record["sample_id"]} aa={record["length_aa"]} elapsed={time.time()-started:.1f}s',flush=True)
        shared.write_table(OUT/'trajectories.tsv',trajectories)
        summarize(all_rows,trajectories);status.update(state='COMPLETE')
    except Exception as exc:
        status.update(state='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        status.update(updated_at_unix=time.time(),elapsed_seconds=time.time()-started);dump('STATUS.json',status)
        if gpu_lock is not None:gpu_lock.close()


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'STATUS.json').exists() and json.loads((OUT/'STATUS.json').read_text())['state']=='COMPLETE':
            raise SystemExit('Already complete')
        execute()
