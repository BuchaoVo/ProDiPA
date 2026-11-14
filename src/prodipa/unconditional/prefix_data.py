#!/usr/bin/env python3
"""Fail-closed preflight for the approved ProDiPA ProtGPT2 teacher data.

This module imports only Python's standard library and can run before a GPU is free.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ATTRS = {'plddt': ('pLDDT', .85), 'sol': ('sol', .7), 'thermo': ('thermo', .6)}
ATTR_DIRS = {'plddt': 'fold', 'sol': 'sol', 'thermo': 'thermo'}
AA = set('ACDEFGHIKLMNPQRSTVWY')
BINS = ((50,149),(150,249),(250,349),(350,512))
TRAIN_QUOTAS = (15,90,135,60)
EXPECTED_QUOTAS = {
    ('train','plddt'):(1,104,135,60), ('train','sol'):TRAIN_QUOTAS,
    ('train','thermo'):TRAIN_QUOTAS, ('test','plddt'):(0,32,47,21),
    ('test','sol'):(24,50,20,6), ('test','thermo'):(30,35,19,16),
}


def expected_quotas(data_dir: Path) -> dict[tuple[str, str], tuple[int, ...]]:
    """Use staged audited quotas when present; retain frozen historical defaults."""

    report_path = data_dir / 'build_report.json'
    if not report_path.is_file():
        return EXPECTED_QUOTAS
    with report_path.open(encoding='utf-8') as handle:
        report = json.load(handle)
    staged = report.get('quotas')
    if not isinstance(staged, dict):
        return EXPECTED_QUOTAS
    resolved = {}
    for attr in ATTRS:
        item = staged.get(attr)
        if not isinstance(item, dict):
            raise ValueError(f'{report_path}: missing quota mapping for {attr}')
        for split in ('train', 'test'):
            values = item.get(split)
            if not isinstance(values, list) or len(values) != len(BINS):
                raise ValueError(f'{report_path}: invalid {split}/{attr} quotas')
            resolved[(split, attr)] = tuple(int(value) for value in values)
    return resolved


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode('utf-8')).hexdigest()


def file_for(data_dir: Path, split: str, attr: str) -> Path:
    n=300 if split=='train' else 100
    name=f'{split}_{attr}_{n}.csv'
    flat=data_dir/name
    nested=data_dir/ATTR_DIRS[attr]/name
    if flat.is_file():
        return flat
    return nested


def _open_manifest(path: Path):
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8-sig', newline='')
    return path.open('r', encoding='utf-8-sig', newline='')


def _resolve_provenance(data_dir: Path) -> tuple[Path, str]:
    selected=data_dir/'selected_sequence_manifest.tsv'
    if selected.is_file():
        return selected, 'selected_manifest'
    global_membership=data_dir.parent/'provenance'/'split_membership.tsv.gz'
    if global_membership.is_file():
        return global_membership, 'global_split_membership'
    raise ValueError(
        'Missing homology provenance: expected selected_sequence_manifest.tsv or '
        f'{global_membership}'
    )


def scan_dataset(data_dir: str | Path) -> dict:
    data_dir=Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError(f'Data directory not found: {data_dir}')
    groups_index, provenance_kind=_resolve_provenance(data_dir)
    quotas=expected_quotas(data_dir)
    hashes={}; seen={}; summary={}; source={}; group_by_split={'train':set(),'test':set()}
    for split in ('train','test'):
        for attr,(column,threshold) in ATTRS.items():
            file=file_for(data_dir,split,attr)
            if not file.is_file():
                raise ValueError(f'Missing required dataset: {file}')
            with file.open('r', newline='', encoding='utf-8-sig') as fh:
                reader=csv.DictReader(fh)
                if reader.fieldnames!=['sequence',column]:
                    raise ValueError(f'Unexpected columns {reader.fieldnames} in {file}; expected sequence,{column}')
                rows=list(reader)
            required=300 if split=='train' else 100
            if len(rows)!=required:
                raise ValueError(f'{file}: expected {required} rows, found {len(rows)}')
            group=set(); counts=[0,0,0,0]; scores=[]
            for row in rows:
                seq=row['sequence'].strip()
                if not set(seq)<=AA or not (50<=len(seq)<=512):
                    raise ValueError(f'Invalid AA/length: {file} ({len(seq)} aa)')
                h=sequence_hash(seq)
                if h in group:
                    raise ValueError(f'Duplicate sequence within {file}')
                group.add(h)
                try:score=float(row[column])
                except (TypeError,ValueError) as exc:raise ValueError(f'Non-numeric score in {file}') from exc
                if not (threshold-1e-7 <= score <= 1.0 + 1e-7):
                    raise ValueError(f'Strict threshold failed in {file}: score={score}, threshold={threshold}')
                scores.append(score)
                counts[next(i for i,(a,b) in enumerate(BINS) if a<=len(seq)<=b)]+=1
            expected=quotas[(split,attr)]
            if tuple(counts)!=expected:
                raise ValueError(f'{file}: expected audited length quotas {expected}, found {counts}')
            relative=str(file.relative_to(data_dir))
            hashes[relative]=sha256(file)
            seen[(split,attr)]=group
            summary[f'{split}_{attr}']={'path':relative,'n':len(rows),'length_bins':counts,'mean_property':sum(scores)/len(scores),'sha256':hashes[relative]}
    for a in ATTRS:
        for b in ATTRS:
            if seen[('train',a)] & seen[('test',b)]:
                raise ValueError(f'Train/test exact sequence overlap: {a}/{b}')
    with _open_manifest(groups_index) as fh:
        reader=csv.DictReader(fh,delimiter='\t')
        rows=list(reader)
    manifest_index={}
    if provenance_kind=='selected_manifest':
        required={'dataset','attribute','sequence_sha256','homology_group'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{groups_index} lacks columns {sorted(required-set(reader.fieldnames or []))}')
        for r in rows:
            key=(r['dataset'],r['attribute'],r['sequence_sha256'])
            if key in manifest_index:
                raise ValueError('Duplicate provenance key: '+str(key))
            manifest_index[key]=r['homology_group']
    else:
        required={'sequence','homology_group','split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{groups_index} lacks columns {sorted(required-set(reader.fieldnames or []))}')
        by_hash={}
        for r in rows:
            sequence=r['sequence'].strip().upper()
            h=sequence_hash(sequence)
            if h in by_hash:
                raise ValueError(f'Duplicate exact sequence in global membership: {h}')
            by_hash[h]=r
        for (split,attr),seq_hashes in seen.items():
            expected_source_split='train' if split=='train' else 'validation'
            for h in seq_hashes:
                source=by_hash.get(h)
                if source is None:
                    raise ValueError(f'Sequence not present in global membership: {h}')
                if source['split'] != expected_source_split:
                    raise ValueError(
                        f'Unexpected source split for {split}/{attr}/{h}: {source["split"]}'
                    )
                manifest_index[(split,attr,h)]=source['homology_group']
    if any(not value for value in manifest_index.values()):
        raise ValueError('Blank homology_group in provenance manifest')
    for (split,attr),seq_hashes in seen.items():
        present_groups=set()
        for h in seq_hashes:
            key=(split,attr,h)
            if key not in manifest_index:
                raise ValueError(f'Sequence not matched to frozen provenance: {key}')
            hg=manifest_index[key]
            if hg in present_groups:
                raise ValueError(f'Repeated homology group within {split} {attr}: {hg}')
            present_groups.add(hg)
            group_by_split[split].add(hg)
        if len(present_groups)!=len(seq_hashes):
            raise ValueError('Internal homology audit mismatch')
    leakage=group_by_split['train'] & group_by_split['test']
    if leakage:
        raise ValueError(f'Train/test homology-group leakage, examples: {sorted(leakage)[:5]}')
    return {'data_dir':str(data_dir),'provenance_path':str(groups_index),
            'provenance_kind':provenance_kind,'provenance_sha256':sha256(groups_index),
            'source':'Train25K training / Validation1K used for checkpoint selection; not blind test',
            'thresholds':{k:v[1] for k,v in ATTRS.items()},'split_homology_overlap':0,
            'counts':summary,'dataset_sha256':hashes}


def run_preflight(data_dir: str | Path, output_path: str | Path | None = None) -> dict:
    result=scan_dataset(data_dir)
    if output_path is not None:
        path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',required=True)
    p.add_argument('--write-report',default=None)
    args=p.parse_args()
    print(json.dumps(run_preflight(args.data_dir,args.write_report),ensure_ascii=False,indent=2))
