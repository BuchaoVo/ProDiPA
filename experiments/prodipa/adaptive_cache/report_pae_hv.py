#!/usr/bin/env python3
"""Supplementary five-objective HV: replace PPL with reversed mean pAE."""
import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
from common import ROOT, OUT, FULL, METHODS, read, rows, dump, tsv, sha
from report import exact_hv, normalized_auc, num, fmt, eligible, IDS, DISPLAY
from report import point as report_point, hv as report_hv, archive_stats

DEST = ROOT/'outputs/evaluation/paper/hv5_pae_no_ppl'
HV = 'HV-5D (PAE,noPPL)'
RET = 'HV Ret. (%) [PAE,noPPL]'
AUC = 'HV-AUC [PAE,noPPL]'
HQ = 'HQ Disc. [PAE,noPPL]'
ALIASES = {v: IDS[k] for k,v in DISPLAY.items()} | {'ProDiPA (Offline)': 'prodipa_offline'}

SIX_FIELDS = ['novelty_u_percent', 'novelty_t_percent', 'plddt', 'pae', 'sol', 'thermo']
FOUR_FIELDS = ['plddt', 'pae', 'sol', 'thermo']


def six_oriented(row):
    return oriented_values(row, SIX_FIELDS)


def oriented_values(row, fields):
    values = [num(row.get(k)) for k in fields]
    if any(v is None for v in values):
        raise ValueError(f'Missing objective evidence: {row.get("model")} {row.get("sequence_id")}')
    values[fields.index('pae')] *= -1
    return values


def project_four_spec(previous):
    """Drop novelty dimensions without refitting any retained scale."""
    fields = [o['field'] for o in previous['objectives']]
    indices = [fields.index(k) for k in FOUR_FIELDS]
    spec = dict(previous)
    for key in ['objectives', 'raw_directions', 'oriented_min', 'oriented_max', 'reference_point']:
        spec[key] = [previous[key][i] for i in indices]
    spec.update(definition_source='User-requested post-hoc four-objective Conditional main-table revision',
                excluded_from_hv=['PPL', 'ProTrek %', 'Novelty-U', 'Novelty-T'],
                missing_policy='Abort if any valid sequence lacks any of the four objectives',
                scale_projection='Inherited pLDDT, -pAE, Sol and Thermo ranges from the published six-objective definition; no refitting',
                interpretation='Post-hoc descriptive comparison; not directly comparable with five- or six-objective HV')
    return spec


def publish_conditional(objective_fields=SIX_FIELDS):
    """Post-hoc main-table revision; no training, scoring or unconditional writes."""
    import moocore
    paper = ROOT / 'outputs/evaluation/paper'
    canonical = ROOT / 'outputs/evaluation/conditional/results/conditional_prodipa_main.tsv'
    paths = [paper / 'conditional_main.tsv', canonical]
    costs_path = paper / 'adaptive_cache_hv_cost.tsv'
    metrics_path = paper / 'adaptive_cache_metrics.json'
    definition_path = paper / 'conditional_hv_definition.json'
    audit_path = paper / 'conditional_hv_recalculation.json'
    dimensions = len(objective_fields)
    label = 'four' if dimensions == 4 else 'six'
    backup = paper / ('conditional_hv_previous_six.json' if dimensions == 4 else 'conditional_hv_previous.json')
    protected = [paper / 'unconditional_main.tsv',
                 ROOT / 'outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv']
    protected_hashes = {str(p): sha(p) for p in protected if p.exists()}
    tables = {str(p): rows(p) for p in paths}
    wanted = {ALIASES.get(r['Model'], r['Model']) for table in tables.values() for r in table} - {'Semi-OPD'}
    source = ROOT / 'outputs/evaluation/conditional/results/conditional_sequence_scores.tsv'
    data = rows(source)
    sources = {str(source): sha(source)}
    cohorts = {m: [r for r in data if r['model'] == m] for m in wanted}
    for method, model in IDS.items():
        node = 0 if method == 'reference' else 4
        path = OUT / f'evaluation/conditional/{method}/node_{node}/scores/sequence_scores.tsv'
        sources[str(path)] = sha(path)
        cohorts[model] = rows(path)
    for model, records in cohorts.items():
        if len(records) != 512 or len({r['sample_id'] for r in records}) != 512:
            raise ValueError(f'{model}: expected 512 distinct requests, got {len(records)}')
        for r in records:
            if r['valid_complete'].lower() == 'true':
                oriented_values(r, objective_fields)  # Never impute missing values.
    pooled = np.asarray([oriented_values(r, objective_fields) for model in sorted(cohorts) for r in cohorts[model]
                         if r['valid_complete'].lower() == 'true'])
    lower, upper = pooled.min(axis=0), pooled.max(axis=0)
    if np.any(upper == lower):
        raise ValueError('An objective dimension has zero pooled range')
    spec = dict(
        definition_source=f'User-requested post-hoc {label}-objective Conditional main-table revision',
        objectives=[dict(field=k, transform='negate' if k == 'pae' else 'identity',
                         direction='maximize') for k in objective_fields],
        raw_directions=['minimize' if k == 'pae' else 'maximize' for k in objective_fields],
        excluded_from_hv=['PPL', 'ProTrek %'] + (['Novelty-U', 'Novelty-T'] if dimensions == 4 else []), oriented_min=lower.tolist(),
        oriented_max=upper.tolist(), reference_point=[0.0]*dimensions,
        normalization_scope='Shared pooled min-max across valid complete sequence-level final cohorts of all current Conditional main-table models; not per-model',
        fit_models=sorted(cohorts), fit_complete_records=len(pooled),
        archive_policy='Use the same final-cohort scale for intermediate archives without refitting or clipping',
        invalid_policy='Invalid requests retained in request counts; contribute no HV box',
        missing_policy=f'Abort if any valid sequence lacks any of the {label} objectives',
        interpretation='Post-hoc descriptive comparison, not the pre-training frozen evaluation protocol; not directly comparable with old five-objective HV',
        algorithm=f'moocore {moocore.__version__} exact hypervolume, maximise=True',
        source_files_sha256=dict(sources), scoring_rerun=False,
    )
    # Once published, keep this shared scale fixed on subsequent reruns.
    if definition_path.exists():
        previous = read(definition_path)
        if dimensions == 4 and len(previous['objectives']) == 6:
            spec = project_four_spec(previous)
            spec['inherited_definition_sha256'] = sha(definition_path)
        elif previous['objectives'] == spec['objectives']:
            spec = previous
        else:
            raise ValueError('Existing objective definition differs; refusing an implicit scale change')
    results = {}
    for model, records in sorted(cohorts.items()):
        print(f'Computing {label}-objective Conditional HV: {model}', flush=True)
        results[model] = dict(HV=report_hv(records, 'conditional', spec),
                             requests=len(records), complete=sum(r['valid_complete'].lower() == 'true' for r in records))
    reference = results['reference_theta0']['HV']
    gain = results['online_opd_paired']['HV'] - reference
    protocol = read(OUT / 'PROTOCOL.json')
    meaningful = gain > max(protocol['meaningful_online_gain_absolute_min'],
                           abs(reference)*protocol['meaningful_online_gain_relative_min'])
    for model in IDS.values():
        results[model]['HV Ret. (%)'] = 100*(results[model]['HV']-reference)/gain if meaningful else None
    old_costs = rows(costs_path)
    curves = []
    for method in METHODS:
        archive_data = list(cohorts['reference_theta0'])
        curve = []
        for row in [r for r in old_costs if r['paradigm'] == 'conditional' and r['method'] == method]:
            node = row['node']
            if node != 'cache_ready':
                if int(node) > 0:
                    path = OUT / f'evaluation/conditional/{method}/node_{node}/scores/sequence_scores.tsv'
                    sources[str(path)] = sha(path)
                    archive_data.extend(rows(path))
                for r in archive_data:
                    if r['valid_complete'].lower() == 'true':
                        oriented_values(r, objective_fields)
                h, count, qualified = archive_stats(archive_data, 'conditional', spec, protocol['evaluation_gate'])
            curve.append(dict(row, training_gpu_hours=float(row['training_gpu_hours']),
                              trusted_archive_hv=h, hq_archive_size=count, qualified_unique=qualified))
        if not curve or curve[-1]['node'] != '4':
            raise ValueError(f'Missing complete cost curve: {method}')
        total = curve[-1]['training_gpu_hours']
        results[IDS[method]].update({'HQ Disc.':100*count/total, 'hq_unique_final_nd':count,
                                    'training_gpu_hours':total})
        curves.extend(curve)
    cstar = min(r['training_gpu_hours'] for r in curves if r['node'] == '4')
    for method in METHODS:
        curve = [r for r in curves if r['method'] == method]
        count = next(r['hq_archive_size'] for r in reversed(curve) if r['training_gpu_hours'] <= cstar)
        results[IDS[method]].update({'HV-AUC':normalized_auc(curve, cstar),
                                    'common_cost_gpu_hours':cstar, 'hq_at_common_cost':count,
                                    'hq_discovery_common_cost':100*count/cstar})
    # All computations pass before any published table is replaced.
    if not backup.exists():
        saved = {str(p): p.read_text() for p in [*paths, costs_path, metrics_path, definition_path, audit_path] if p.exists()}
        dump(backup, dict(reason=f'Prior values before requested {label}-objective revision', files=saved))
    changes = []
    for path in paths:
        table = tables[str(path)]
        fields = list(dict.fromkeys(list(table[0])+['HV Comparable']))
        for row in table:
            model = ALIASES.get(row['Model'], row['Model'])
            if model not in results:
                continue
            result = results[model]
            changes.append(dict(table=str(path), model=model, old_hv=row['HV'], new_hv=result['HV']))
            row['HV'] = fmt(result['HV'])
            row['HV Comparable'] = f'posthoc_{label}_objective_scale'
            for key in ['HV Ret. (%)', 'HV-AUC', 'HQ Disc.']:
                if key in fields:
                    row[key] = fmt(result.get(key)) if model in IDS.values() else '--'
        tsv(path, table, fields)
    curve_map = {(r['paradigm'],r['method'],r['node']):r for r in curves}
    updated_curves = [curve_map.get((r['paradigm'],r['method'],r['node']),r) for r in old_costs]
    tsv(costs_path, updated_curves)
    if metrics_path.exists():
        metrics = read(metrics_path)
        conditional = metrics['results']['conditional']
        conditional.update(online_hv_gain=gain, retention_applicable=meaningful,
                           hv_definition=str(definition_path))
        mapping = {'HV-AUC':'hv_auc', 'HQ Disc.':'hq_discovery_per_100_gpu_hours',
                   'HV Ret. (%)':'hv_retention_percent'}
        for method, model in IDS.items():
            target = conditional[method]
            for key, value in results[model].items():
                if key not in ('requests','complete'):
                    target[mapping.get(key,key)] = value
            if method in METHODS:
                target['hv_minus_online'] = results[model]['HV'] - results['online_opd_paired']['HV']
        dump(metrics_path, metrics)
    dump(definition_path, spec)
    dump(audit_path, dict(state='COMPLETE', results=results, changes=changes,
                         sources_sha256=sources, definition=str(definition_path),
                         command=f'python experiments/prodipa/adaptive_cache/report_pae_hv.py --publish-conditional-{label}',
                         prior_tables=str(backup),
                         historical_report_note='The original adaptive_cache_core_results.md and original frozen protocol remain historical five-objective analyses; current values are in the main TSV, metrics JSON and this audit.',
                         unconditional_sha256=protected_hashes))
    assert all(sha(Path(p)) == digest for p,digest in protected_hashes.items())
    print('Updated Conditional main tables and dependent HV columns; Unconditional tables unchanged', flush=True)


def point(row, paradigm, spec):
    keys = ['protrek_percent' if paradigm=='conditional' else 'novelty_u_percent',
            'plddt','sol','thermo','pae']
    values = [num(row.get(k)) for k in keys]
    if any(v is None for v in values):
        raise ValueError(f'Missing objective: {paradigm} {row.get("model")} {row.get("sequence_id")}')
    if not 0 <= values[-1] <= 31.75 + 1e-6:
        raise ValueError(f'pAE outside the scorer range: {values[-1]}')
    values[-1] = -values[-1]
    lo, hi = np.array(spec['oriented_min']), np.array(spec['oriented_max'])
    return (np.array(values)-lo)/(hi-lo)-np.array(spec['reference_point'])


def valid_points(data, paradigm, spec):
    return np.array([point(r,paradigm,spec) for r in data
                     if r['valid_complete'].lower()=='true']).reshape(-1,5)


def archive(data, paradigm, spec, gate):
    unique = {r['sequence_id']:r for r in data if eligible(r,gate)}
    points = np.array([point(r,paradigm,spec) for r in unique.values()]).reshape(-1,5)
    nd = sum(not np.any(np.all(points>=p,axis=1)&np.any(points>p,axis=1)) for p in points)
    return exact_hv(points),int(nd),len(unique)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    started=time.time()
    dump(DEST/'STATUS.json',dict(state='RUNNING',started_at_unix=started))
    specs={}
    sources={}
    table_paths={}
    cohorts={}
    for paradigm in ('conditional','unconditional'):
        manifest=(OUT/'protocol/conditional_hv.json' if paradigm=='conditional' else
                  ROOT/'outputs/evaluation/unconditional/results/unconditional_hv_manifest.json')
        prior=read(manifest)
        specs[paradigm]=dict(objectives=[o['field'] for o in prior['objectives'][1:]]+['negative_pae'],
            oriented_min=prior['oriented_min'][1:]+[-31.75],
            oriented_max=prior['oriented_max'][1:]+[0.0],
            reference_point=prior['reference_point'][1:]+[0.0],
            inherited_manifest=str(manifest),inherited_manifest_sha256=sha(manifest))
        table_paths[paradigm]=[
            ROOT/f'outputs/evaluation/{paradigm}/results/{paradigm}_prodipa_main.tsv',
            ROOT/f'outputs/evaluation/paper/{paradigm}_main.tsv']
        wanted={ALIASES.get(r['Model'],r['Model']) for path in table_paths[paradigm] for r in rows(path)}-{'Semi-OPD'}
        source=ROOT/f'outputs/evaluation/{paradigm}/results/{paradigm}_sequence_scores.tsv'
        sources[str(source)]=sha(source)
        data=rows(source)
        cohorts[paradigm]={m:[r for r in data if r['model']==m] for m in wanted}
        if paradigm=='conditional':
            for method,model in IDS.items():
                node=0 if method=='reference' else 4
                source=OUT/f'evaluation/conditional/{method}/node_{node}/scores/sequence_scores.tsv'
                sources[str(source)]=sha(source)
                cohorts[paradigm][model]=rows(source)
        else:
            novelty_file=FULL/'evaluation/novelty/unconditional_novelty_u.tsv'
            sources[str(novelty_file)]=sha(novelty_file)
            novelty={r['sequence_id']:r['novelty_percent'] for r in rows(novelty_file)}
            for model in ('reference_theta0','online_opd_paired','prodipa_offline'):
                source=FULL/f'evaluation/methods/unconditional/{model}/results/sequence_scores.tsv'
                sources[str(source)]=sha(source)
                records=rows(source)
                for r in records:
                    if r['sequence_id'] in novelty:
                        r['novelty_u_percent']=novelty[r['sequence_id']]
                cohorts[paradigm][model]=records
        for model,records in cohorts[paradigm].items():
            if len(records)!=512 or len({r['sample_id'] for r in records})!=512:
                raise ValueError(f'{paradigm}/{model}: incomplete request evidence ({len(records)})')
            valid_points(records,paradigm,specs[paradigm])

    protocol=dict(name='HV-5D (PAE,noPPL)',paradigms=specs,pae_range_angstrom=[0,31.75],
        pae_range_basis='ESMFold max_bin=31, no_bins=64; bin step=31/62=0.5; final center=31+0.25+0.5=31.75; lower reference bound=0.',
        pae_transform='1 - mean_pae / 31.75',fitting='No new cohort min-max; other dimensions inherited unchanged',
        clipping=False,scope='Supplementary post-hoc sensitivity analysis, not a replacement for the original frozen protocol',
        original_columns_preserved=True,training_and_scoring_rerun=False,
        unconditional_adaptive_evaluation_remains_paused=True)
    frozen=DEST/'PROTOCOL.json'
    if frozen.exists() and read(frozen)!=protocol:
        raise ValueError('Supplementary protocol changed; use a new versioned output directory')
    dump(frozen,protocol)
    dump(DEST/'INPUTS.json',sources)
    results={}
    for paradigm,models in cohorts.items():
        results[paradigm]={}
        for model,data in models.items():
            print(f'Computing {paradigm}/{model}',flush=True)
            points=valid_points(data,paradigm,specs[paradigm])
            results[paradigm][model]={HV:exact_hv(points),RET:None,AUC:None,HQ:None,
                                       'requests':len(data),'complete':len(points)}
        ref=results[paradigm]['reference_theta0'][HV]
        gain=results[paradigm]['online_opd_paired'][HV]-ref
        meaningful=gain>max(1e-6,abs(ref)*0.01)
        for model in IDS.values():
            if model in results[paradigm]:
                m=results[paradigm][model]
                m[RET]=100*(m[HV]-ref)/gain if meaningful else None
                m['online_hv_gain']=gain
                m['retention_applicable']=meaningful
        dump(DEST/'metrics.json',results)

    print('Recomputing Conditional trusted archive curves and final nondominance',flush=True)
    old_curves=rows(ROOT/'outputs/evaluation/paper/adaptive_cache_hv_cost.tsv')
    gate=read(OUT/'PROTOCOL.json')['evaluation_gate']
    all_curves=[]
    for method in METHODS:
        data=list(cohorts['conditional']['reference_theta0'])
        curve=[]
        for row in [r for r in old_curves if r['paradigm']=='conditional' and r['method']==method]:
            node=row['node']
            if node!='cache_ready':
                if int(node)>0:
                    source=OUT/f'evaluation/conditional/{method}/node_{node}/scores/sequence_scores.tsv'
                    sources[str(source)]=sha(source)
                    data.extend(rows(source))
                h,count,qualified=archive(data,'conditional',specs['conditional'],gate)
            curve.append(dict(row,training_gpu_hours=float(row['training_gpu_hours']),
                 trusted_archive_hv=h,hq_archive_size=count,qualified_unique=qualified))
        total=curve[-1]['training_gpu_hours']
        m=results['conditional'][IDS[method]]
        m.update(training_gpu_hours=total,hq_unique_final_nd=count)
        m[HQ]=100*count/total
        all_curves.extend(curve)
    cstar=min(r['training_gpu_hours'] for r in all_curves if r['node']=='4')
    for method in METHODS:
        curve=[r for r in all_curves if r['method']==method]
        m=results['conditional'][IDS[method]]
        m[AUC]=normalized_auc(curve,cstar)
        m['common_cost_gpu_hours']=cstar
        m['hq_discovery_common_cost']=100*next(r['hq_archive_size'] for r in reversed(curve) if r['training_gpu_hours']<=cstar)/cstar
    dump(DEST/'metrics.json',results)
    dump(DEST/'INPUTS.json',sources)
    tsv(DEST/'conditional_hv_cost.tsv',all_curves)
    # Append clearly named supplementary columns only, retaining every old cell.
    for paradigm,paths in table_paths.items():
        for path in paths:
            table=rows(path)
            fields=list(dict.fromkeys(list(table[0])+[HV,RET,AUC,HQ]))
            for row in table:
                model=ALIASES.get(row['Model'],row['Model'])
                m=results[paradigm].get(model)
                for key in (HV,RET,AUC,HQ):
                    row[key]=fmt(m[key]) if m and (key==HV or model in IDS.values()) else '--'
            backup=DEST/'original_tables'/path.relative_to(ROOT/'outputs/evaluation')
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():shutil.copy2(path,backup)
            tsv(path,table,fields)
        tsv(DEST/f'{paradigm}_hv.tsv',[dict(Model=model,**{key:fmt(m[key]) for key in (HV,RET,AUC,HQ)},
                   requests=m['requests'],complete=m['complete']) for model,m in results[paradigm].items()])
    dump(DEST/'STATUS.json',dict(state='COMPLETE',started_at_unix=started,finished_at_unix=time.time(),
         updated_tables=[str(p) for paths in table_paths.values() for p in paths],latex_generated=False))
    print('Complete: original HV columns retained; supplementary columns published to four tables',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--publish-conditional-six', action='store_true',
                        help='Replace Conditional main HV with Novelty-U/T, pLDDT, -pAE, Sol, Thermo')
    group.add_argument('--publish-conditional-four', action='store_true',
                       help='Use only pLDDT, -pAE, Sol, Thermo; inherit the published six-objective scale')
    args = parser.parse_args()
    if args.publish_conditional_six or args.publish_conditional_four:
        publish_conditional(FOUR_FIELDS if args.publish_conditional_four else SIX_FIELDS)
        raise SystemExit(0)
    try:
        main()
    except BaseException as error:
        dump(DEST/'STATUS.json',dict(state='FAILED',error=repr(error),failed_at_unix=time.time()))
        raise
