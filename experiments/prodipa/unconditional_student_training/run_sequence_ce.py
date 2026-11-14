#!/usr/bin/env python3
"""Launch the complete-sequence CE student repair without legacy EOS-only assets."""
import csv
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/experiments/unconditional_student_sequence_ce'
PREVIOUS = ROOT / 'outputs/experiments/unconditional_length_control_20261111'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--gpus', default='auto', help='auto selects two available GPUs from 0-3')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(lock, args.gpus)


def run(controller_lock, requested_gpus='auto'):
    if (OUT / 'student/DONE.json').exists():
        print('Training already complete; no evaluation launched.', flush=True)
        return
    (OUT / 'logs').mkdir(exist_ok=True)
    (OUT / 'protocol').mkdir(exist_ok=True)
    sources = [ROOT / 'data/unconditional' / name for name in (
        'fold/train_plddt_300.csv', 'sol/train_sol_300.csv', 'thermo/train_thermo_300.csv')]
    sequences = []
    seen = set()
    for source in sources:
        with source.open() as stream:
            for row in csv.DictReader(stream):
                seq = row['sequence']
                if not 50 <= len(seq) <= 512 or set(seq) - set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError(f'Invalid complete training sequence in {source}; refusing truncation')
                if seq not in seen:
                    sequences.append(seq)
                    seen.add(seq)
    anchor = OUT / 'protocol/training_anchors.csv'
    if anchor.exists():
        with anchor.open() as stream:
            if [r['sequence'] for r in csv.DictReader(stream)] != sequences:
                raise ValueError('Existing anchors differ; use a new run directory')
    else:
        with anchor.open('x', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['sequence'])
            writer.writerows((seq,) for seq in sequences)
    teacher_config = PREVIOUS / 'configs/teachers.yaml'
    warmstart = PREVIOUS / 'student'
    env = os.environ.copy()
    env.update(MAIN_PROCESS_PORT='29631', SEQUENCE_FORMAT='unwrapped',
               OUTPUT_DIR=str(OUT / 'student'), TEACHER_CONFIG=str(teacher_config),
               STUDENT_LORA_ADAPTER_PATH=str(warmstart), EOS_ANCHOR_PATH=str(anchor),
               EOS_ANCHOR_WEIGHT='4', EOS_ANCHOR_LOSS_WEIGHT='0.01', MAX_LENGTH_AA='0',
               PYTHON_BIN=sys.executable, ACCELERATE_BIN=str(Path(sys.executable).with_name('accelerate')),
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1')
    code_paths = [ROOT / name for name in (
        'src/prodipa/unconditional/eos_anchor.py', 'src/prodipa/unconditional/train.py',
        'src/prodipa/unconditional/trainer.py', 'src/prodipa/unconditional/data.py',
        'src/prodipa/unconditional/sequence_format.py',
        'experiments/prodipa/unconditional_student_training/run_training.sh',
        'experiments/prodipa/unconditional_student_training/run_sequence_ce.py')]
    protocol = dict(objective='full_sequence_weighted_ce_v2', terminal_weight=4,
                    auxiliary_coefficient=0.01, anchor_count=len(sequences),
                    anchor_sha256=digest(anchor), sources={str(p): digest(p) for p in sources},
                    warmstart=str(warmstart), warmstart_sha256=digest(warmstart / 'adapter_model.safetensors'),
                    teacher_config=str(teacher_config), teacher_config_sha256=digest(teacher_config),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                    optimizer_initialization='fresh; pretrained student weights only, not optimizer resume',
                    updates=256, num_train_samples=8192, per_device_batch=8, accumulation=2,
                    learning_rate=2e-5, warmup_ratio=0.1, max_new_tokens=512, max_length_aa=0,
                    sequence_format='unwrapped', artificial_terminal_loss='masked_independently_of_aa_cap',
                    automatic_evaluation=False, automatic_promotion=False,
                    coefficient_rationale='Fixed conservative full-token CE scale; not a validated optimum.')
    protocol_path = OUT / 'PROTOCOL.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new run directory')
    dump(protocol_path, protocol)
    allocation = []
    started = None
    status = dict(state='PREPARING', updated_at_unix=time.time(), pid=os.getpid(),
                  training_log=str(OUT / 'logs/student_training.log'))
    dump(OUT / 'STATUS.json', status)
    try:
        query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        devices = {parts[0]: parts for line in query.splitlines()
                   if (parts := [part.strip() for part in line.split(',')])}
        candidates = sorted(('0','1','2','3'), key=lambda i: int(devices[i][2]), reverse=True) if requested_gpus == 'auto' else requested_gpus.split(',')
        if requested_gpus != 'auto' and (len(candidates) != 2 or len(set(candidates)) != 2 or not set(candidates) <= {'0','1','2','3'}):
            raise ValueError('Select exactly two distinct GPUs within 0-3')
        chosen = []
        for index in candidates:
            _, uuid, free, util = devices[index]
            if int(free) < 64000 or int(util) > 30:
                continue
            handle = open(f'/tmp/prodipa_gpu_{uuid}.lock', 'a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            allocation.append(handle)
            chosen.append(index)
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise RuntimeError('Two GPUs within 0-3 with >=64000 MiB free and <=30% utilization are required')
        env['TRAINING_GPUS'] = ','.join(chosen)
        started = time.time()
        status.update(state='RUNNING', started_at_unix=started, gpu_indices=list(map(int, chosen)),
                      gpu_uuids=[devices[i][1] for i in chosen])
        dump(OUT / 'STATUS.json', status)
        print(f'Starting 256 updates on GPUs {env["TRAINING_GPUS"]}; {len(sequences)} complete training anchors; unwrapped protocol.', flush=True)
        with (OUT / 'logs/student_training.log').open('a') as log:
            subprocess.run(['bash', str(ROOT / 'experiments/prodipa/unconditional_student_training/run_training.sh')],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        done = json.loads((OUT / 'student/DONE.json').read_text())
        if done['global_step'] != 256:
            raise RuntimeError(f'Unexpected training endpoint: {done}')
        status.update(state='COMPLETE', global_step=done['global_step'])
        print('Training complete; evaluation and promotion were not launched.', flush=True)
    except Exception as exc:
        status.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        finished = time.time()
        status.update(updated_at_unix=finished)
        if started is not None:
            status.update(allocated_seconds=finished-started, gpu_hours=2*(finished-started)/3600,
                          cost_scope='This student repair job only; excludes prior training and evaluation.')
        dump(OUT / 'STATUS.json', status)
        for handle in allocation:
            handle.close()


if __name__ == '__main__':
    main()
