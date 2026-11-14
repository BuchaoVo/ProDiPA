#!/usr/bin/env python3
"""Fresh ProDiPA Prefix teachers and OPD with explicit 512-aa rollout control."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/experiments/unconditional_length_control_20261111'
BASE = Path('__PRODIPA_MODELS__/ProtGPT2')
LOCK = threading.RLock()
STATE = {'status': 'STARTING', 'jobs': {}}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def status(name, **values):
    with LOCK:
        STATE['jobs'].setdefault(name, {}).update(values)
        STATE['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(OUT / 'STATUS.json', STATE)
        print(name, values, flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    from prodipa.unconditional.train_teacher import ProteinCsvDataset
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    audit = {}
    for attribute in ('fold', 'sol', 'thermo'):
        source = ROOT / f'outputs/unconditional/protgpt2/configs/{attribute}.yaml'
        config = yaml.safe_load(source.read_text())
        audit[attribute] = {'source_config_sha256': sha(source)}
        for split in ('train', 'validation'):
            path = config[f'{split}_path']
            data = ProteinCsvDataset(path, tokenizer, config['text_column'], config['id_column'], config['max_length'])
            lengths = [row['sequence_length_aa'] for row in data.records]
            if min(lengths) < 50 or max(lengths) > 512:
                raise ValueError(f'{attribute}/{split}: out-of-range training data; no automatic truncation')
            audit[attribute][split] = {'count': len(data), 'min_aa': min(lengths),
                                     'max_aa': max(lengths), 'max_tokens_with_eos': max(data.token_lengths),
                                     'sha256': sha(path)}
        config.update(output_dir=str(OUT / f'teachers/{attribute}'),
                      generation_enforce_aa_limit=True,
                      source_code_revision='sha256:' + sha(ROOT / 'src/prodipa/unconditional/train_teacher.py'))
        path = OUT / f'configs/{attribute}.yaml'
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content and (OUT / f'teachers/{attribute}').exists():
            raise ValueError(f'Resume config changed: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    teachers = yaml.safe_load((ROOT / 'src/prodipa/unconditional/teachers.yaml').read_text())
    for spec in teachers['teachers']:
        spec['adapter_path'] = str(OUT / f"teachers/{spec['name']}/best_adapter")
    (OUT / 'configs/teachers.yaml').write_text(yaml.safe_dump(teachers, sort_keys=False))
    write_json(OUT / 'DATA_AUDIT.json', audit)
    write_json(OUT / 'PROTOCOL.json', {
        'training_initialization': 'Fresh Base ProtGPT2 for every teacher and the student; no old OPD resume',
        'max_new_tokens': 512, 'max_generated_aa': 512,
        'cap_rule': 'Mask next tokens exceeding remaining residue budget; no post-hoc slicing',
        'termination': 'Forced boundary EOS is invalid_complete=false and its OPD loss position is masked',
        'teacher_protocol': 'Original 300 train / 100 validation; Prefix100; max50 epochs, early-stop patience5',
        'student_protocol': '8192 prompts, 2 GPUs, batch8/GPU, accumulation2, 1 epoch = 256 updates',
        'loss': 'Original full-vocabulary PoE/JS beta0.5, equal teachers; no termination reweighting',
        'generation_validation': '512 independent requests each: Base, Fold, Sol, Thermo, OPD; same 512-aa cap',
        'historical_outputs': 'Not overwritten; current adaptive-cache experiment remains opt-out',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / 'src/prodipa/unconditional/length_control.py',
            ROOT / 'src/prodipa/unconditional/trainer.py',
            ROOT / 'src/prodipa/unconditional/train.py',
            ROOT / 'src/prodipa/unconditional/train_teacher.py',
            ROOT / 'experiments/prodipa/conditional_evaluation/generate_model.py',
            ROOT / 'experiments/prodipa/unconditional_student_training/run_training.sh']}})


def acquire(name, count, memory):
    status(name, status='WAITING_GPU', required_gpus=count, min_free_mib=memory)
    while True:
        selected = []
        try:
            listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True)
            rows = [row.split(', ') for row in listing.strip().splitlines()]
            for index, uuid, free, utilization in sorted(rows, key=lambda row: -int(row[2])):
                if index not in {'0', '1', '2', '3'} or int(free) < memory or int(utilization) > 90:
                    continue
                handle = open(f'/tmp/prodipa_gpu_{uuid}.lock', 'a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                selected.append((uuid, handle))
                if len(selected) == count:
                    return selected
        finally:
            if len(selected) != count:
                for _, handle in selected:
                    handle.close()
        time.sleep(30)


def job(name, command, count=1, memory=20000, extra_env=None):
    selected = acquire(name, count, memory)
    gpus = ','.join(uuid for uuid, _ in selected)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, TRAINING_GPUS=gpus,
               PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
    env.update(extra_env or {})
    log = OUT / f'logs/{name}.log'
    started = time.time()
    status(name, status='RUNNING', gpus=gpus, started_at=started, log=str(log), command=command)
    try:
        with log.open('a') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    pass_fds=tuple(handle.fileno() for _, handle in selected))
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {log}')
        status(name, status='COMPLETE', elapsed_seconds=time.time()-started,
               gpu_hours=count*(time.time()-started)/3600)
    except Exception as error:
        status(name, status='FAILED', error=str(error), elapsed_seconds=time.time()-started)
        raise
    finally:
        for _, handle in selected:
            handle.close()


def teacher(attribute):
    if (OUT / f'teachers/{attribute}/DONE.json').exists():
        status(f'teacher_{attribute}', status='ALREADY_COMPLETE')
        return
    job(f'teacher_{attribute}', [sys.executable, '-m', 'prodipa.unconditional.train_teacher',
                              '--config', str(OUT / f'configs/{attribute}.yaml')])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    with (OUT / '.pipeline.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'STATUS.json').exists():
            STATE.update(json.loads((OUT / 'STATUS.json').read_text()))
        STATE['status'] = 'RUNNING'
        prepare()
        status('data_audit', status='PASS')
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(teacher, ('fold', 'sol', 'thermo')))
        if not (OUT / 'student/DONE.json').exists():
            with socket.socket() as sock:
                sock.bind(('', 0))
                port = sock.getsockname()[1]
            job('opd_training', ['bash', str(ROOT / 'experiments/prodipa/unconditional_student_training/run_training.sh')],
                count=2, memory=40000, extra_env={
                    'OUTPUT_DIR': str(OUT / 'student'), 'TEACHER_CONFIG': str(OUT / 'configs/teachers.yaml'),
                    'MAX_LENGTH_AA': '512', 'PYTHON_BIN': sys.executable,
                    'MAIN_PROCESS_PORT': str(port),
                    'ACCELERATE_BIN': str(Path(sys.executable).with_name('accelerate'))})
        else:
            status('opd_training', status='ALREADY_COMPLETE')
        for name in ('base', 'fold', 'sol', 'thermo', 'opd'):
            command = [sys.executable, str(ROOT / 'experiments/prodipa/conditional_evaluation/generate_model.py'),
                       '--paradigm', 'unconditional', '--model-name', f'protgpt2_aa512_{name}',
                       '--base-model', str(BASE), '--out', str(OUT / f'generation/{name}.tsv'),
                       '--num-sequences', '512', '--generation-max-aa', '512']
            if name != 'base':
                adapter = OUT / ('student' if name == 'opd' else f'teachers/{name}/best_adapter')
                command += ['--adapter', str(adapter)]
            job(f'generation_{name}', command, memory=16000)
        audit = {}
        for path in (OUT / 'generation').glob('*.tsv'):
            with path.open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            lengths = [int(row['length_aa']) for row in rows]
            assert len(rows) == 512 and max(lengths) <= 512, path
            audit[path.stem] = {'requests': len(rows), 'max_aa': max(lengths),
                                'natural_eos': sum(row['natural_eos_reached'] == 'true' for row in rows),
                                'forced_eos': sum(row['length_forced_eos'] == 'true' for row in rows),
                                'valid_complete': sum(row['valid_complete'] == 'true' for row in rows)}
        write_json(OUT / 'LENGTH_VALIDATION.json', audit)
        STATE['status'] = 'TRAINING_AND_LENGTH_VALIDATION_COMPLETE'
        status('pipeline', status='COMPLETE', note='Attribute scoring not yet run; no claim of quality improvement')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        STATE['status'] = 'FAILED'
        status('pipeline', status='FAILED', error=str(error))
        raise
