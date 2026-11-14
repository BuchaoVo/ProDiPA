#!/usr/bin/env python3
"""Configure only the extracted public copy; never launch GPU workloads."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEXT={'.py','.sh','.json','.yaml','.yml','.toml','.md','.txt'}


def check_data():
    source=ROOT/'data/release/prodipa_dataset/train_25000_complete.tsv.gz'
    with gzip.open(source,'rt') as f:
        records=list(csv.DictReader(f,delimiter='\t'))
    assert len(records)==25000
    assert len({r['sequence_id'] for r in records})==25000
    assert sum(r['in_prior_15k'].lower()=='true' for r in records)==15000
    assert all(r['split'].lower()=='train' for r in records)
    assert all(r[k].lower()=='true' for r in records for k in ['fold_present','sol_present','thermo_present'])
    inventory=json.loads((ROOT/'configs/reproduction/dataset_inventory.json').read_text())
    for relative,expected in inventory['current_teachers'].items():
        path=ROOT/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],relative
    print('PASS: 25K complete Train, nested 15K, and unchanged current teacher data')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path)
    p.add_argument('--databases',type=Path)
    p.add_argument('--envs',type=Path)
    p.add_argument('--conda',type=Path)
    p.add_argument('--check-data',action='store_true')
    a=p.parse_args()
    if a.check_data:
        check_data();return
    if not all([a.models,a.databases,a.envs,a.conda]):
        p.error('--models, --databases, --envs and --conda are required (no downloads performed)')
    if (ROOT/'LOCAL_PATHS.json').exists():
        p.error('Already configured. Re-extract a fresh copy to change paths safely.')
    mapping={'ROOT':ROOT,'MODELS':a.models,'DATABASES':a.databases,'ENVS':a.envs,'CONDA':a.conda}
    replacements={('__PRODIPA_'+k+'__'):str(v.resolve()) for k,v in mapping.items()}
    if any(any(c in v for c in ['\n','\r','"',"'",'$','`','\\',' ']) for v in replacements.values()):
        p.error('Use shell-safe absolute paths without spaces/quotes/metacharacters')
    changed=0
    for base in ['src','scripts','experiments','configs','third_party','tools']:
        for path in (ROOT/base).rglob('*'):
            if not path.is_file() or path.suffix not in TEXT or path.resolve()==Path(__file__).resolve():continue
            text=path.read_text()
            updated=text
            for old,new in replacements.items():updated=updated.replace(old,new)
            if updated!=text:path.write_text(updated);changed+=1
    (ROOT/'LOCAL_PATHS.json').write_text(json.dumps({k:str(v.resolve()) for k,v in mapping.items()},indent=2)+'\n')
    print(f'Configured {changed} source/config files; frozen data untouched.')
    print('Check model weights, independent predictor environments, MMseqs2 and reference database before any GPU command.')


if __name__=='__main__':main()
