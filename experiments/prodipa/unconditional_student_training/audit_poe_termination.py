#!/usr/bin/env python3
"""Read-only model audit on retained student sequences and teacher training data.

Inputs are re-tokenized contexts, not reconstructed original sampling token IDs.
No training, generation, score recomputation, or official table publication.
"""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'outputs/experiments/unconditional_length_control_20261111'
OUT = ROOT / 'outputs/experiments/unconditional_poe_termination'
BASE = '__PRODIPA_MODELS__/ProtGPT2'


def read(path, delimiter='\t'):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def dump(name, data):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dump('STATUS.json', dict(state='LOADING', started_at=started))
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    config = yaml.safe_load((SOURCE / 'configs/teachers.yaml').read_text())
    specs = config['teachers']
    teacher = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True),
        specs[0]['adapter_path'], adapter_name=specs[0]['name']).cuda().eval()
    for spec in specs[1:]:
        teacher.load_adapter(spec['adapter_path'], adapter_name=spec['name'])
    student = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True),
        str(SOURCE / 'student')).cuda().eval()
    weights = [float(s['weight']) for s in specs]
    weights = [w / sum(weights) for w in weights]
    contexts = []
    retained = ROOT / 'outputs/experiments/unconditional_uncapped_diagnostic_20261111/sequence_diagnostics.tsv'
    for row in read(retained):
        contexts.append(('student_uncapped', row['sample_id'], row['raw_completion'].replace('\\n', '\n')))
    seen = set()
    for attribute, key in [('fold', 'plddt'), ('sol', 'sol'), ('thermo', 'thermo')]:
        for row in read(ROOT / f'data/unconditional/{attribute}/train_{key}_300.csv', ','):
            seq = row['sequence']
            if seq not in seen:
                seen.add(seq)
                contexts.append(('teacher_train_union', hashlib.sha256(seq.encode()).hexdigest(), seq))
    dump('PROTOCOL.json', dict(
        teachers=specs, student=str(SOURCE / 'student'), contexts=len(contexts),
        retained_student_sha256=hashlib.sha256(retained.read_bytes()).hexdigest(),
        context_rule='Re-tokenized saved raw completion (newlines restored); clean teacher train sequence. No synthetic terminal EOS in input.',
        endpoints='Final context and last token boundary <=512 non-whitespace characters on long sequences.',
        probabilities='FP32 normalized full-vocabulary weighted-log-prob PoE; teacher temperatures unchanged; raw distributions without sampling processors.',
        official_tables_changed=False))
    records = []
    with (OUT / 'context_metrics.tsv').open('w') as handle, torch.inference_mode():
        writer = None
        for index, (kind, identifier, text) in enumerate(contexts):
            tokens = tok.encode(text, add_special_tokens=False)
            ids = torch.tensor([[eos] + tokens], device='cuda')
            if len(tokens) + 1 + 100 > teacher.config.n_positions:
                raise ValueError(f'Context exceeds prefix-aware model capacity: {identifier}')
            # The decoder uses ASCII protein/whitespace tokens. Count the decoded
            # prefix, not token count, for the 512-aa observation point.
            lens = [0] + [len(''.join(tok.decode(tokens[:i], skip_special_tokens=True).split()))
                          for i in range(1, len(tokens) + 1)]
            points = [('end', len(tokens))]
            if lens[-1] > 512:
                points.append(('boundary_512', max(i for i, n in enumerate(lens) if n <= 512)))
            attention = torch.ones_like(ids)
            positions = [i for _, i in points]
            distributions = {}
            for spec in specs:
                teacher.set_adapter(spec['name'])
                logits = teacher(input_ids=ids, attention_mask=attention, use_cache=False).logits[0, positions].float()
                distributions[spec['name']] = F.log_softmax(logits / float(spec['temperature']), -1)
            pooled = sum(weights[i] * distributions[s['name']] for i, s in enumerate(specs))
            distributions['poe'] = F.log_softmax(pooled, -1)
            distributions['student'] = F.log_softmax(student(input_ids=ids, attention_mask=attention, use_cache=False).logits[0, positions].float(), -1)
            with student.disable_adapter():
                distributions['base'] = F.log_softmax(student(input_ids=ids, attention_mask=attention, use_cache=False).logits[0, positions].float(), -1)
            for j, (point, pos) in enumerate(points):
                row = dict(source=kind, sample_id=identifier, point=point,
                           total_aa=lens[-1], context_aa=lens[pos], context_tokens=pos)
                for name, values in distributions.items():
                    row[name + '_p_eos'] = values[j, eos].exp().item()
                    row[name + '_eos_rank'] = int((values[j] > values[j, eos]).sum().item()) + 1
                row['poe_vs_student_eos_logodds'] = (distributions['poe'][j,eos] - distributions['student'][j,eos]).item()
                # Positive means the fixed teacher target wants more EOS probability.
                for name in ['student', 'base']:
                    p = distributions[name][j]
                    q = distributions['poe'][j]
                    m = torch.logaddexp(p, q) - math.log(2)
                    row[name + '_js_to_poe'] = (.5 * (p.exp() * (p-m) + q.exp() * (q-m)).sum()).item()
                records.append(row)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row), delimiter='\t')
                    writer.writeheader()
                writer.writerow(row)
            handle.flush()
            if (index + 1) % 16 == 0 or index + 1 == len(contexts):
                state = dict(state='RUNNING', completed=index+1, total=len(contexts), elapsed_seconds=time.time()-started)
                dump('STATUS.json', state)
                print(state, flush=True)
    import statistics
    summary = []
    for source in ['student_uncapped', 'teacher_train_union']:
        for point in ['end', 'boundary_512']:
            for length_group in ['<=512', '>512']:
                group = [r for r in records if r['source'] == source and r['point'] == point and
                         (r['total_aa'] > 512) == (length_group == '>512')]
                if not group:
                    continue
                row = dict(source=source, point=point, length_group=length_group, n=len(group))
                for name in ['base', 'fold', 'sol', 'thermo', 'poe', 'student']:
                    row[name + '_mean_p_eos'] = statistics.mean(r[name+'_p_eos'] for r in group)
                    row[name + '_median_p_eos'] = statistics.median(r[name+'_p_eos'] for r in group)
                row['fraction_poe_eos_above_student'] = statistics.mean(r['poe_p_eos'] > r['student_p_eos'] for r in group)
                summary.append(row)
    dump('SUMMARY.json', summary)
    with (OUT / 'summary.tsv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), delimiter='\t')
        w.writeheader(); w.writerows(summary)
    dump('STATUS.json', dict(state='COMPLETE', completed=len(contexts), elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        dump('STATUS.json', dict(state='FAILED', error=repr(error)))
        raise
