#!/usr/bin/env python3
"""Read-only model diagnostic on balanced, recorded Conditional generation paths."""
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/diagnostics/conditional_multiteacher_distributions'
NAMES = ['fold', 'sol', 'thermo', 'student']
PAIRS = list(itertools.combinations(range(4), 2))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def write_table(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def js(a, b):
    mixture = torch.logaddexp(a, b) - math.log(2)
    return .5 * ((a.exp() * (a-mixture)).sum(-1) + (b.exp() * (b-mixture)).sum(-1))


def measures(logits, targets, temperatures, weights, closing_ids):
    """Full-vocabulary FP32 metrics; no top-k approximation or gradients."""
    raw = [F.log_softmax(x.float(), -1) for x in logits]
    scaled = [F.log_softmax(x.float()/t, -1) for x, t in zip(logits, temperatures)]
    result = {}
    for label, dist in [('raw', raw), ('distill', scaled)]:
        teachers = torch.stack(dist[:3])
        w = torch.tensor(weights, device=teachers.device).view(3, 1, 1)
        mean_log = torch.logsumexp(teachers + w.log(), 0)
        poe = F.log_softmax((teachers*w).sum(0), -1)
        result[f'{label}_teacher_js'] = (w.squeeze(-1) * (teachers.exp()*(teachers-mean_log)).sum(-1)).sum(0)
        result[f'{label}_student_poe_js'] = js(dist[3], poe)
        result[f'{label}_poe_entropy'] = -(poe.exp()*poe).sum(-1)
        for i, name in enumerate(NAMES):
            result[f'{label}_{name}_entropy'] = -(dist[i].exp()*dist[i]).sum(-1)
        for i, j in PAIRS:
            result[f'{label}_{NAMES[i]}_{NAMES[j]}_js'] = js(dist[i], dist[j])
        if label == 'raw':
            continue
        tops = torch.stack([p.argmax(-1) for p in dist])
        maxima = torch.stack([p.exp().amax(-1) for p in dist])
        disagreement = torch.stack([tops[i] != tops[j] for i,j in PAIRS if j < 3])
        confident = torch.stack([(tops[i]!=tops[j]) & (maxima[i]>=.5) & (maxima[j]>=.5)
                                 for i,j in PAIRS if j < 3])
        result['teacher_top1_disagreement'] = disagreement.any(0).float()
        result['confident_teacher_conflict'] = confident.any(0).float()
        result['student_poe_top1_disagreement'] = (tops[3] != poe.argmax(-1)).float()
        result['poe_suppresses_teacher_top1_10x'] = torch.stack([
            poe.gather(-1, tops[i,:,None]).squeeze(-1) < dist[i].amax(-1)-math.log(10)
            for i in range(3)]).any(0).float()
        selected = torch.stack([p.gather(-1, targets[:,None]).squeeze(-1) for p in dist])
        result['teacher_sampled_logp_range'] = selected[:3].amax(0)-selected[:3].amin(0)
        result['poe_sampled_nll'] = -poe.gather(-1, targets[:,None]).squeeze(-1)
        for i, name in enumerate(NAMES):
            result[f'{name}_sampled_nll'] = -selected[i]
            result[f'{name}_top1_id'] = tops[i]
            result[f'{name}_top1_prob'] = maxima[i]
            result[f'{name}_eos_prob'] = dist[i][:, closing_ids['eos']].exp()
            result[f'{name}_close_prob'] = dist[i][:, closing_ids['close']].exp().sum(-1)
        result['poe_eos_prob'] = poe[:, closing_ids['eos']].exp()
        result['poe_close_prob'] = poe[:, closing_ids['close']].exp().sum(-1)
    return {k: v.detach().cpu().tolist() for k,v in result.items()}


def locations(tokenizer, ids, text, eos):
    match = re.search(r'Seq\s*=\s*<([^>]*)', text)
    begin, end = (match.start(1), match.end(1)) if match else (-1, -1)
    previous = ''
    result = []
    for pos, target in enumerate(ids):
        prefix = tokenizer.decode(ids[:pos+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        a, b = len(previous), len(prefix)
        if target == eos:
            kind = 'eos'
        elif begin >= 0 and a >= begin and b <= end and b > a:
            kind = 'protein'
        elif '>' in prefix[a:]:
            kind = 'closing_or_mixed'
        else:
            kind = 'format_or_other'
        aa_before = len(''.join(text[begin:min(a,end)].split())) if begin >= 0 and a >= begin else 0
        length = len(''.join(match.group(1).split())) if match else 0
        result.append(dict(position=pos, target_id=target,
                           target_text=tokenizer.decode([target], clean_up_tokenization_spaces=False),
                           region=kind, aa_before=aa_before,
                           position_decile=min(9, int(10*aa_before/max(1,length))),
                           context_tail=previous[-48:]))
        previous = prefix
    return result


def summarize(rows, trajectories):
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame['intervention_candidate'] = ((frame.distill_teacher_js >= .05)
                                      & (frame.distill_student_poe_js >= .05)).astype(float)
    metric_names = [k for k in rows[0] if (k.endswith(('_js', '_entropy', '_prob', '_nll'))
                   or k in ('teacher_top1_disagreement', 'confident_teacher_conflict',
                            'student_poe_top1_disagreement', 'poe_suppresses_teacher_top1_10x',
                            'teacher_sampled_logp_range'))] + ['intervention_candidate']
    body = frame[frame.region == 'protein']
    # Sequence-level means prevent long proteins from dominating the primary summary.
    seq = body.groupby(['producer','sample_id'])[metric_names].mean().reset_index()
    seq.to_csv(OUT/'sequence_summary.tsv', sep='\t', index=False)
    rng = np.random.default_rng(20261111)
    summary = []
    for producer, part in seq.groupby('producer'):
        for key in metric_names:
            values = part[key].to_numpy()
            boots = rng.choice(values, size=(1000,len(values)), replace=True).mean(1)
            summary.append(dict(producer=producer, metric=key, sequences=len(values),
                                mean=float(values.mean()), ci_low=float(np.quantile(boots,.025)),
                                ci_high=float(np.quantile(boots,.975))))
    write_table(OUT/'summary.tsv', summary)
    for keys, name, subset in [(['producer','position_decile'], 'position_summary.tsv', body),
                               (['producer','region'], 'region_summary.tsv', frame)]:
        per_seq = subset.groupby(['sample_id', *keys])[metric_names].mean().reset_index()
        means = per_seq.groupby(keys)[metric_names].mean()
        means['sequences'] = per_seq.groupby(keys).size()
        means.to_csv(OUT/name, sep='\t')
    # Rank diagnostic states, not proteins' biological quality.
    hotspots = body.sort_values(['distill_teacher_js','distill_student_poe_js'], ascending=False)
    hotspots.head(100).to_csv(OUT/'hotspots.tsv', sep='\t', index=False)
    persistence = []
    for (producer,sample), part in body.groupby(['producer','sample_id']):
        longest = streak = 0
        prior = -2
        for row in part.sort_values('position').itertuples():
            streak = (streak+1 if row.position == prior+1 else 1) if row.intervention_candidate else 0
            longest = max(longest,streak)
            prior = row.position
        persistence.append(dict(producer=producer,sample_id=int(sample),max_candidate_run=longest))
    write_table(OUT/'persistent_states.tsv', persistence)
    header = ['# Conditional multi-teacher distribution diagnostic', '',
              'No training, weight changes, new protein sampling, or main-table writes were performed.', '',
              '256 recorded generation requests: 64 per producer, seed 20261111, sampled without validity filtering. '
              'Text was re-tokenized; original generation token IDs were not saved. These are reconstructed prefixes, '
              'not guaranteed bit-identical historical token paths. Only recorded natural EOS is appended. '
              'All four models see the same prefixes in eval/inference mode.', '',
              'Primary statistics below average protein-token metrics within each sequence, then across sequences. '
              'summary.tsv contains sequence-bootstrap 95% intervals (1000 resamples). '
              'Missing/empty protein bodies are counted in trajectories.tsv and omitted from body-only means.', '',
              'Raw distributions use T=1 for all models. Distillation distributions use teacher T=0.7, student T=1, '
              'equal-weight normalized geometric PoE, FP32 full-vocabulary probabilities. '
              'No repetition penalty, top-k or top-p is applied to these diagnostic distributions; '
              'they describe model/OPD targets, not the processed sampling policy. Entropies and JS are in nats.', '',
              '| Trajectory producer | Body sequences | Teacher JS | Student–PoE JS | Teacher top-1 disagreement | Confident conflict | Candidate states | Sequences with ≥3 consecutive candidate states |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for producer, part in seq.groupby('producer'):
        means = part.mean(numeric_only=True)
        persistent = sum(r['max_candidate_run']>=3 for r in persistence if r['producer']==producer)
        header.append(f"| {producer} | {len(part)} | {means.distill_teacher_js:.4f} | {means.distill_student_poe_js:.4f} | "
                      f"{100*means.teacher_top1_disagreement:.1f}% | {100*means.confident_teacher_conflict:.1f}% | "
                      f"{100*means.intervention_candidate:.1f}% | {persistent} |")
    header += ['', '## Interpretation limits', '',
               'Teacher top-1 disagreement alone is not strong conflict: near-tied diffuse distributions can disagree. '
               'Confident conflict requires two teachers to prefer different top tokens, each with probability ≥0.5. '
               'Candidate states use predeclared teacher JS ≥0.05 AND student–PoE JS ≥0.05 on protein-body tokens. '
               'These are descriptive screening thresholds, not calibrated intervention or biological quality gates.', '',
               'A persistent candidate region is a location worth investigating, not evidence that intervention improves '
               'Fold/Sol/Thermo, nor evidence of conflicting training gradients. No gradients were computed. '
               'Regions/positions are reported separately so formatting and natural termination do not masquerade as '
               'protein-body conflicts. Only this TPR condition and the original main-table student are covered; '
               'do not generalize directly to later Offline/Adaptive checkpoints.', '',
               'See summary.tsv, position_summary.tsv, region_summary.tsv, persistent_states.tsv, '
               'hotspots.tsv and per_token.tsv for supporting data. PROTOCOL.json records sources, weights and hashes.']
    (OUT/'REPORT.md').write_text('\n'.join(header)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'STATUS.json').exists() and json.loads((OUT/'STATUS.json').read_text())['state']=='COMPLETE':
            print('Already complete.', flush=True)
            return
        execute()


def execute():
    started = time.time()
    status = dict(state='PREPARING', pid=os.getpid(), completed=0, total=256)
    dump(OUT/'STATUS.json', status)
    gpu_lock = None
    try:
        config_path = ROOT/'outputs/conditional/prollama_multiteacher_student/opd_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        base = config['student']['backbone_path']
        adapters = {s['name']: Path(s['adapter_path']) for s in config['teachers']['adapters']}
        adapters['student'] = ROOT/'outputs/conditional/prollama_multiteacher_student'
        sources = {name: ROOT/f'outputs/evaluation/conditional/generation/conditional/{name}_teacher.tsv' for name in NAMES[:3]}
        sources['student'] = adapters['student']/'generation/conditional/prollama_multiteacher_student.tsv'
        records = []
        source_meta = {}
        rng = random.Random(20261111)
        for name,path in sources.items():
            meta = json.loads(path.with_suffix('.metadata.json').read_text())
            if Path(meta['adapter']).resolve() != adapters[name].resolve():
                raise ValueError(f'Generation adapter mismatch: {name}')
            source_meta[name] = meta
            with path.open() as f:
                cohort = list(csv.DictReader(f,delimiter='\t'))
            for row in sorted(rng.sample(cohort,64),key=lambda r:int(r['sample_id'])):
                records.append(dict(row,producer=name))
        weights = [float(s['weight']) for s in config['teachers']['adapters']]
        temperatures = [float(s['temperature']) for s in config['teachers']['adapters']] + [float(config['generation']['student_temperature'])]
        if list(adapters)!=NAMES or abs(sum(weights)-1)>1e-8:
            raise ValueError('Unexpected teacher order/weights')
        weight_paths = {name: next(p for p in [path/'adapter_model.safetensors',path/'adapter_model.bin'] if p.exists())
                        for name,path in adapters.items()}
        protocol = dict(student=str(adapters['student']), config=str(config_path),
                        config_sha256=sha(config_path), source_code_sha256=sha(Path(__file__)),
                        weights=weights, temperatures=temperatures, sample_seed=20261111,
                        samples_per_producer=64, total=256, no_training=True,
                        trajectory_representation='decoded generation text re-encoded, original token IDs unavailable',
                        sources={k:dict(path=str(v),sha256=sha(v),metadata=source_meta[k]) for k,v in sources.items()},
                        adapters={k:dict(path=str(v),sha256=sha(v)) for k,v in weight_paths.items()},
                        candidate_gate=dict(teacher_js_min=.05,student_poe_js_min=.05,region='protein'),
                        note='Distribution conflicts are not gradient conflicts or demonstrated quality loss.')
        dump(OUT/'PROTOCOL.json',protocol)
        query = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
        index,uuid,free,util = next([x.strip() for x in line.split(',')] for line in query.splitlines() if line.startswith('0,'))
        if int(free)<30000 or int(util)>50:
            raise RuntimeError('GPU 0 lacks headroom; other training is left untouched')
        gpu_lock = open(f'/tmp/prodipa_gpu_{uuid}.lock','a')
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        tokenizer = AutoTokenizer.from_pretrained(base,local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,
                       attn_implementation='sdpa',local_files_only=True).to('cuda')
        model = PeftModel.from_pretrained(base_model,str(adapters['fold']),adapter_name='fold',is_trainable=False)
        for name in NAMES[1:]:
            model.load_adapter(str(adapters[name]),adapter_name=name,is_trainable=False)
        model.requires_grad_(False).eval()
        eos = tokenizer.eos_token_id
        # EOS text is </s>, which contains '>' but is not the Seq=<...> closing marker.
        specials = set(tokenizer.all_special_ids)
        closing_ids = {'eos':eos,'close':[i for i in range(len(tokenizer))
                                         if i not in specials and '>' in tokenizer.decode([i])]}
        all_rows, trajectories = [], []
        with (OUT/'per_token.tsv').open('w',newline='') as f:
            writer = None
            for number, record in enumerate(records):
                name = record['producer']
                prompt = '[Generate by superfamily] Superfamily=<' + source_meta[name]['superfamily'] + '>'
                prompt_ids = tokenizer.encode(prompt)
                text = record['raw_completion'].replace('\\n','\n')
                ids = tokenizer.encode(text,add_special_tokens=False)
                decoded = tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if decoded.strip() != text.strip():
                    raise ValueError(f'Text round-trip mismatch: {name}/{record["sample_id"]}')
                if record['eos_reached']=='true':
                    ids.append(eos)
                if not ids:
                    raise ValueError('No observed token or EOS to evaluate')
                input_ids = torch.tensor([prompt_ids+ids],device='cuda')
                if input_ids.shape[1]>model.config.max_position_embeddings:
                    raise ValueError('Trajectory exceeds context; refusing silent truncation')
                logits = []
                with torch.inference_mode():
                    for adapter in NAMES:
                        model.set_adapter(adapter)
                        model.requires_grad_(False).eval()
                        output = model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False)
                        logits.append(output.logits[0,len(prompt_ids)-1:-1].float())
                        del output
                    metrics = measures(logits,torch.tensor(ids,device='cuda'),temperatures,weights,closing_ids)
                del logits, input_ids
                rows = []
                for pos, loc in enumerate(locations(tokenizer,ids,decoded,eos)):
                    row = dict(producer=name,sample_id=int(record['sample_id']),**loc,
                               **{key:value[pos] for key,value in metrics.items()})
                    rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t')
                    writer.writeheader()
                writer.writerows(rows)
                f.flush()
                all_rows.extend(rows)
                trajectories.append(dict(producer=name,sample_id=int(record['sample_id']),sequence=record['sequence'],
                                         length_aa=int(record['length_aa']),valid_complete=record['valid_complete'],
                                         eos_reached=record['eos_reached'],truncated=record['truncated'],
                                         retokenized_completion_tokens=len(ids),protein_tokens=sum(r['region']=='protein' for r in rows)))
                status.update(state='RUNNING',completed=number+1,total=len(records),gpu=0,
                              elapsed_seconds=time.time()-started,updated_at_unix=time.time())
                dump(OUT/'STATUS.json',status)
                print(f'{number+1}/{len(records)} {name}/{record["sample_id"]} tokens={len(ids)} elapsed={time.time()-started:.1f}s',flush=True)
        write_table(OUT/'trajectories.tsv',trajectories)
        summarize(all_rows,trajectories)
        status.update(state='COMPLETE')
    except Exception as exc:
        status.update(state='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status.update(updated_at_unix=time.time(),elapsed_seconds=time.time()-started)
        dump(OUT/'STATUS.json',status)
        if gpu_lock is not None:
            gpu_lock.close()


if __name__=='__main__':
    main()
