#!/usr/bin/env python3
"""Evaluate the new ProDiPA model set and promote results only after frozen gates."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'outputs/experiments/unconditional_length_control_20261111'
EVAL = RUN / 'evaluation'
sys.path.insert(0, str(ROOT / 'experiments/prodipa/adaptive_cache'))
import common
# Reuse the established predictors/accounting, but never the paused experiment's state.
common.OUT = EVAL
from common import dump, read, rows, tsv, sha, job, PY
from scoring import score
from report import summary, hv, fmt

CANONICAL = ROOT / 'outputs/evaluation/unconditional'
CONFIG = ROOT / 'configs/evaluation/unconditional.json'
PAPER = ROOT / 'outputs/evaluation/paper/unconditional_main.tsv'
IDS = {'base': 'base', 'fold': 'fold_teacher', 'sol': 'sol_teacher',
       'thermo': 'thermo_teacher', 'opd': 'protgpt2_multiteacher_student'}
METRICS = ['ppl', 'novelty_u_percent', 'novelty_t_percent', 'plddt', 'pae', 'sol', 'thermo']


def state(stage, **extra):
    payload = dict(stage=stage, updated_at=time.strftime('%FT%T%z'), **extra)
    dump(RUN / 'EVALUATION_STATUS.json', payload)
    print(payload, flush=True)


def weight(path):
    return next(p for p in [path/'adapter_model.safetensors', path/'adapter_model.bin'] if p.is_file())


def freeze():
    manifest = RUN / 'EVALUATION_PROTOCOL.json'
    if manifest.exists():
        return read(manifest)
    old = {p: str(ROOT/f'outputs/unconditional/protgpt2/{p}/best_adapter') for p in ['fold', 'sol', 'thermo']}
    old['opd'] = str(ROOT / 'outputs/unconditional/protgpt2_multiteacher_student')
    for name in ['unconditional_evaluation', 'unconditional_hv']:
        source = CONFIG if name.endswith('evaluation') else CANONICAL/'results/unconditional_hv_manifest.json'
        dump(EVAL / 'protocol' / f'{name}.json', read(source))
    protocol = dict(version=1, frozen_at=time.time(), requests=512, generation_max_aa=512,
                    old_adapters=old, old_weights_sha256={p: sha(weight(Path(v))) for p,v in old.items()},
                    hv_manifest_sha256=sha(EVAL/'protocol/unconditional_hv.json'),
                    scoring_config_sha256=sha(EVAL/'protocol/unconditional_evaluation.json'),
                    sampling=dict(seed=42, temperature=0.7, top_p=0.9, top_k=200, repetition_penalty=1.2, max_new_tokens=512),
                    gates=dict(student_min_relative_hv_gain=0.01, absolute_hv_epsilon=1e-6,
                               student_nonregression=['pLDDT','Sol','Thermo','complete'],
                               teacher_nonregression={'fold':'pLDDT','sol':'Sol','thermo':'Thermo'},
                               teacher_valid_count_nonregression=True,
                               require_student_hv_above_base=True,
                               require_teacher_target_above_base=True),
                    decision='All four models must pass together; failures preserve all official results',
                    interpretation='Single 512-request comparison; not a statistical significance claim',
                    promotion_scope='Result tables, sequence evidence and score registry only; no checkpoint deletion or physical weight overwrite',
                    old_innovation_ii='Remains paused. New lineage caches and matched comparisons must be built separately.')
    dump(manifest, protocol)
    return protocol


def validate(records):
    if len(records) != 512 or {int(r['sample_id']) for r in records} != set(range(512)):
        raise ValueError('Evaluation must contain all 512 distinct requests, including invalid outputs')
    for r in records:
        sid = int(r['sample_id'])
        if int(r['seed']) != 42 + 1009*sid or int(r['generation_max_aa']) != 512:
            raise ValueError('Sampling seed or residue-cap mismatch')
        for key, expected in [('temperature', .7), ('top_p', .9), ('top_k', 200), ('repetition_penalty', 1.2), ('max_new_tokens', 512)]:
            if float(r[key]) != expected:
                raise ValueError(f'Sampling parameter mismatch: {key}')
        seq = r['sequence']
        if len(seq) != int(r['length_aa']) or len(seq) > 512:
            raise ValueError('Decoded sequence exceeds residue cap or has inconsistent length')
        complete = str(r['valid_complete']).lower() == 'true'
        forced = str(r['length_forced_eos']).lower() == 'true'
        if complete:
            if forced or str(r['truncated']).lower() != 'false' or str(r['natural_eos_reached']).lower() != 'true':
                raise ValueError('Artificial termination incorrectly marked complete')
            if not 50 <= len(seq) <= 512 or set(seq) - set('ACDEFGHIKLMNPQRSTVWY'):
                raise ValueError('Invalid complete sequence')


def gates(metrics, protocol):
    checks = {}
    base, old, new = metrics['base'], metrics['old_opd'], metrics['opd']
    for name in ['fold','sol','thermo','opd']:
        checks[f'{name}_valid_count_nonregression'] = metrics[name]['complete'] >= metrics['old_'+name]['complete']
    for name, metric in protocol['gates']['teacher_nonregression'].items():
        checks[f'{name}_{metric}_vs_old'] = metrics[name][metric] >= metrics['old_'+name][metric]
        checks[f'{name}_{metric}_vs_base'] = metrics[name][metric] > base[metric]
    for metric in ['pLDDT','Sol','Thermo']:
        checks[f'opd_{metric}_nonregression'] = new[metric] >= old[metric]
    for name, comparator in [('old',old), ('base',base)]:
        margin = max(protocol['gates']['absolute_hv_epsilon'], abs(comparator['HV'])*protocol['gates']['student_min_relative_hv_gain'])
        checks[f'opd_hv_gain_vs_{name}'] = new['HV'] - comparator['HV'] > margin
    if old.get('comparison_scope') == 'historical_unmatched_generation':
        # Historical scores are useful context, not a matched length-policy control.
        for key in ['opd_valid_count_nonregression', 'opd_hv_gain_vs_old',
                    'opd_pLDDT_nonregression', 'opd_Sol_nonregression', 'opd_Thermo_nonregression']:
            checks[key] = None
        checks['old_opd_matched_protocol_available'] = False
    return dict(pass_all=all(v is True for v in checks.values()), checks=checks,
                failed=[k for k,v in checks.items() if v is False],
                not_applicable=[k for k,v in checks.items() if v is None])


def reuse_historical_old_opd(manifest):
    source = CANONICAL/'results/unconditional_sequence_scores.tsv'
    records = [r for r in rows(source) if r['model'] == IDS['opd']]
    if len(records) != 512 or {int(r['sample_id']) for r in records} != set(range(512)):
        raise ValueError('Historical old OPD must retain all 512 requests')
    valid = [r for r in records if r['valid_complete'].lower() == 'true']
    if not valid or any(not math.isfinite(float(r.get(k, 'nan'))) for r in valid for k in METRICS):
        raise ValueError('Incomplete historical old OPD scores')
    destination = EVAL/'historical_old_opd/sequence_scores.tsv'
    tsv(destination, records)
    evidence = dict(source=str(source), source_sha256=sha(source), snapshot=str(destination),
                    snapshot_sha256=sha(destination), reused_at=time.time(), requests=len(records),
                    max_observed_aa=max(int(r['length_aa']) for r in records),
                    comparison_scope='historical_unmatched_generation',
                    reason='User requested historical score reuse. Original sampling lacks the new 512-aa logits cap.',
                    hv_scale=str(EVAL/'protocol/unconditional_hv.json'),
                    hv_recomputed=True, eligible_for_automatic_promotion=False)
    dump(RUN/'HISTORICAL_OLD_OPD_REUSE.json', evidence)
    result = summary(records, 'unconditional', manifest)
    result.update(comparison_scope=evidence['comparison_scope'], score_source=str(destination),
                  forced_eos=None, natural_eos=None)
    return records, result


def summarize_checked(records, manifest):
    validate(records)
    valid = [r for r in records if str(r['valid_complete']).lower() == 'true']
    if not valid:
        raise ValueError('No valid complete sequences; automatic promotion prohibited')
    for r in valid:
        if any(not math.isfinite(float(r.get(k, 'nan'))) for k in METRICS):
            raise ValueError('Missing/nonfinite score; automatic promotion prohibited')
        if float(r['ppl']) <= 0 or float(r['pae']) < 0:
            raise ValueError('Out-of-range PPL/pAE')
        if any(not 0 <= float(r[k]) <= top for k,top in [('plddt',100),('sol',1),('thermo',1),('novelty_u_percent',100),('novelty_t_percent',100)]):
            raise ValueError('Out-of-range quality score')
    result = summary(records, 'unconditional', manifest)
    result['forced_eos'] = sum(str(r['length_forced_eos']).lower() == 'true' for r in records)
    result['natural_eos'] = sum(str(r['natural_eos_reached']).lower() == 'true' for r in records)
    return result


def export_group(directory, records, manifest):
    """Write standard evidence files consumed by the existing cohort registry."""
    grouped = {}
    for r in records:
        grouped.setdefault(r['model'], []).append(r)
    main, supplements, completeness = [], [], []
    config = read(EVAL/'protocol/unconditional_evaluation.json')
    thresholds = config['supplementary_thresholds']
    for model, rs in grouped.items():
        m = summary(rs, 'unconditional', manifest)
        main.append({key: model if key=='Model' else fmt(m.get(key)) for key in config['main_table_fields']})
        valid = [r for r in rs if str(r['valid_complete']).lower()=='true']
        lengths = sorted(int(r['length_aa']) for r in valid)
        joint = sum(float(r['plddt']) >= thresholds['fold_plddt_min'] and float(r['sol']) >= thresholds['sol_min'] and float(r['thermo']) >= thresholds['thermo_min'] for r in valid)
        supplements.append({'Paradigm':'unconditional','Model':model,'Requests':len(rs),
            'Valid Rate':len(valid)/len(rs), 'EOS Rate':sum(str(r.get('natural_eos_reached',r['eos_reached'])).lower()=='true' for r in rs)/len(rs),
            'Strict JSR':joint/len(rs), 'Mean Length':sum(lengths)/len(lengths),
            'Median Length':(lengths[(len(lengths)-1)//2]+lengths[len(lengths)//2])/2})
        completeness.append({'Paradigm':'unconditional','Model':model,'Requests':len(rs),'Valid complete':len(valid),
                             **{f'N {k}':len(valid) for k in METRICS}, 'N protrek_percent':0})
    tsv(directory/'results/sequence_scores.tsv', records, list(dict.fromkeys(k for r in records for k in r)))
    tsv(directory/'results/unconditional_prodipa_main.tsv', main, config['main_table_fields'])
    tsv(directory/'results/supplementary_metrics.tsv', supplements)
    tsv(directory/'results/scoring_completeness.tsv', completeness)
    unique = {r['sequence_id']:dict(r, length=r['length_aa']) for r in records if str(r['valid_complete']).lower()=='true'}
    tsv(directory/'prepared/unique_valid_sequences.tsv', list(unique.values()), ['sequence_id','sequence','length','valid_complete'])
    dump(directory/'SCORING_STATUS.json', {'state':'COMPLETE','requests':len(records),'protocol':str(RUN/'EVALUATION_PROTOCOL.json')})
    return main, supplements, completeness


def promote(scored, protocol, decision):
    if not decision['pass_all']:
        raise ValueError('Fail-closed promotion: not all acceptance gates passed')
    manifest = read(EVAL/'protocol/unconditional_hv.json')
    hashes = RUN/'EVALUATION_MODEL_HASHES.json'
    if hashes.exists():
        for path, expected in read(hashes).items():
            if sha(path) != expected:
                raise ValueError('Model weights changed after evaluation; refusing promotion')
    # Coordinate with standard registry/publication and preserve other models.
    with (CONFIG.parent/'.unconditional.json.register.lock').open('a') as registry_lock, (CANONICAL/'.completion.lock').open('a') as cohort_lock:
        fcntl.flock(registry_lock, fcntl.LOCK_EX)
        fcntl.flock(cohort_lock, fcntl.LOCK_EX)
        current = read(CONFIG)
        frozen = read(EVAL/'protocol/unconditional_evaluation.json')
        for key in ['requests_per_model','generation_protocol','references','scorers','supplementary_thresholds']:
            if current[key] != frozen[key]:
                raise ValueError(f'Official evaluation protocol changed during run: {key}')
        current_hv = read(CANONICAL/'results/unconditional_hv_manifest.json')
        for key in ['objectives','oriented_min','oriented_max','reference_point']:
            if current_hv[key] != manifest[key]:
                raise ValueError('Official HV scale changed during run; refusing silent replacement')
        records = []
        for name, model in IDS.items():
            for r in scored[name]:
                r = dict(r, model=model, request_id=f'unconditional:{model}:{int(r["sample_id"]):04d}')
                records.append(r)
        group = EVAL/'promoted_cohort'
        export_group(group, records, manifest)
        groups = []
        for g in current['score_groups']:
            remaining = [m for m in g['models'] if m not in IDS.values()]
            if remaining and remaining != g['models']:
                raise ValueError('A mixed registry group requires manual splitting; no publication attempted')
            if remaining:
                groups.append(g)
        groups.append({'name':'protgpt2_aa512_verified','models':list(IDS.values()),'evaluation_root':str(group.relative_to(ROOT))})
        current['score_groups'] = groups
        current['hypervolume']['frozen_manifest'] = str(EVAL/'protocol/unconditional_hv.json')
        current['hypervolume']['normalization'] = 'Frozen pre-run scale; do not refit on promoted models'
        current['model_result_revisions'] = {model:dict(run=str(RUN), generation_max_aa=512, adapter=str(RUN/('student' if name=='opd' else f'teachers/{name}/best_adapter')) if name!='base' else 'BASE') for name,model in IDS.items()}
        old_records = rows(CANONICAL/'results/unconditional_sequence_scores.tsv')
        combined = [r for r in old_records if r['model'] not in IDS.values()] + records
        stage = EVAL/'publication_stage'
        main, supplements, completeness = export_group(stage, combined, manifest)
        tsv(stage/'results/unconditional_sequence_scores.tsv', combined, list(dict.fromkeys(k for r in combined for k in r)))
        tsv(stage/'results/unconditional_supplementary_metrics.tsv', supplements)
        tsv(stage/'results/unconditional_scoring_completeness.tsv', completeness)
        unique = {r['sequence_id']:r for r in combined if str(r['valid_complete']).lower()=='true'}
        tsv(stage/'scores/novelty_u.tsv', [{'sequence_id':sid,'novelty_percent':r['novelty_u_percent']} for sid,r in unique.items()])
        manifest.update(hypervolume_by_model={r['Model']:float(r['HV']) for r in main},
                        complete_points_by_model={r['Model']:int(r['Valid complete']) for r in completeness},
                        complete_points=sum(int(r['Valid complete']) for r in completeness),
                        normalization_scope='Frozen before 512-aa model-set evaluation; no refitting',
                        sequence_score_files=[str(ROOT/g['evaluation_root']/'results/sequence_scores.tsv') for g in groups],
                        novelty_u_file=str(CANONICAL/'scores/novelty_u.tsv'), promotion_run=str(RUN))
        manifest.pop('nondominated_points_by_model', None)  # Historical counts no longer describe this cohort.
        dump(stage/'results/unconditional_hv_manifest.json', manifest)
        # Recompute paper HV on the same frozen scale; mark unrecoverable historical rows.
        paper = rows(PAPER)
        main_map = {r['Model']:r for r in main}
        aliases = {'Reference θ0':'reference_theta0','Online OPD (paired)':'online_opd_paired','ProDiPA (Offline)':'prodipa_offline'}
        seen = set()
        for r in paper:
            model = r['Model']; seen.add(model)
            r['HV Ret. (%)'] = 'N/A'
            if model in main_map:
                r.update({k:v for k,v in main_map[model].items() if k in r})
                r['HV Comparable'] = 'frozen_pre_aa512_scale'
                r['Result revision'] = 'aa512_verified' if model in IDS.values() else 'historical_external'
            elif model in aliases:
                source = ROOT/f'outputs/experiments/offline_multiteacher_opd_full/evaluation/methods/unconditional/{aliases[model]}/results/sequence_scores.tsv'
                evidence = rows(source) if source.exists() else []
                valid = [q for q in evidence if q.get('valid_complete')=='true']
                if len(evidence)==512 and valid and all(all(q.get(k,'') not in ('','N/A') and math.isfinite(float(q[k])) for k in METRICS) for q in valid):
                    r['HV'] = fmt(hv(evidence,'unconditional',manifest)); r['HV Comparable']='frozen_pre_aa512_scale'
                else:
                    r['HV']='N/A'; r['HV Comparable']='historical_scale_not_convertible'
                r['Result revision']='historical_old_weights_not_new_paired_control'
            else:
                r['HV Comparable']='not_run_or_unconverted'
        for model,r in main_map.items():
            if model not in seen:
                paper.append(dict(r, Group='External Protein Design Baselines', **{'HV Ret. (%)':'N/A','HV Comparable':'frozen_pre_aa512_scale','Result revision':'historical_external'}))
        tsv(stage/'paper_unconditional_main.tsv',paper, list(dict.fromkeys(k for r in paper for k in r)))
        dump(stage/'config.json',dict(current,output_root=str(stage)))
        subprocess.run([PY,str(ROOT/'experiments/prodipa/unconditional_evaluation/audit_results.py'),
                        '--config',str(stage/'config.json'),'--project-root',str(ROOT)],check=True)
        destinations = [(stage/'results'/n,CANONICAL/'results'/n) for n in ['unconditional_prodipa_main.tsv','unconditional_sequence_scores.tsv','unconditional_supplementary_metrics.tsv','unconditional_scoring_completeness.tsv','unconditional_hv_manifest.json']]
        destinations += [(stage/'scores/novelty_u.tsv',CANONICAL/'scores/novelty_u.tsv'),(stage/'paper_unconditional_main.tsv',PAPER)]
        dump(stage/'live_config.json',current)
        destinations.append((stage/'live_config.json',CONFIG))
        destinations.append((stage/'prepared/unique_valid_sequences.tsv',CANONICAL/'prepared/cohort_unique_valid_sequences.tsv'))
        dump(stage/'EVALUATION_PROTOCOL.json',dict(
            **{k:current[k] for k in ['schema_version','requests_per_model','generation_protocol','references','scorers','resources','hypervolume','supplementary_thresholds']},
            config=str(CONFIG),config_sha256=sha(stage/'live_config.json'),
            model_result_revisions=current['model_result_revisions'],promotion_protocol=str(RUN/'EVALUATION_PROTOCOL.json')))
        dump(stage/'STATUS.json',dict(state='COMPLETE',stage='verified_model_result_promotion',promotion_run=str(RUN),updated_at_unix=time.time()))
        destinations += [(stage/name,CANONICAL/name) for name in ['EVALUATION_PROTOCOL.json','STATUS.json']]
        backup = RUN/'publication_backup'
        if backup.exists():
            raise ValueError('Existing publication backup: inspect previous transaction before retry')
        backup.mkdir()
        saved = []
        created = []
        for _,target in destinations:
            if not target.exists():
                created.append(target)
                continue
            copy = backup/target.relative_to(ROOT); copy.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,copy); saved.append((copy,target))
        for name in ['RESULT_AUDIT.json','COMPLETE.json']:
            target=CANONICAL/name
            if target.exists():
                copy=backup/target.relative_to(ROOT);copy.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,copy);saved.append((copy,target))
        dump(backup/'MANIFEST.json',{str(t):sha(c) for c,t in saved})
        try:
            for source,target in destinations:
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix(target.suffix+'.promotion.tmp');shutil.copy2(source,tmp);os.replace(tmp,target)
            subprocess.run([PY,str(ROOT/'experiments/prodipa/unconditional_evaluation/audit_results.py'), '--config',str(CONFIG),'--project-root',str(ROOT)],check=True)
            dump(CANONICAL/'COMPLETE.json',dict(status='COMPLETE',config=str(CONFIG),config_sha256=sha(CONFIG),models=[r['Model'] for r in main],promotion_run=str(RUN),completed_at=time.time()))
        except BaseException:
            for copy,target in saved:
                tmp=target.with_suffix(target.suffix+'.rollback.tmp');shutil.copy2(copy,tmp);os.replace(tmp,target)
            for target in created:
                if target.exists():target.unlink()
            raise
        dump(RUN/'PROMOTION_COMPLETE.json',dict(status='COMPLETE',promoted_at=time.time(),decision=decision,backup=str(backup),scope='results_and_evidence',physical_weights_overwritten=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--reuse-historical-old-opd',action='store_true',
                        help='Reuse historical old-student scores as unmatched context; prohibit promotion')
    args=parser.parse_args()
    EVAL.mkdir(parents=True,exist_ok=True)
    with (RUN/'.post_training_evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze()
        if sha(EVAL/'protocol/unconditional_hv.json') != protocol['hv_manifest_sha256'] or sha(EVAL/'protocol/unconditional_evaluation.json') != protocol['scoring_config_sha256']:
            raise ValueError('Frozen scoring/HV files changed')
        if args.freeze_only:
            print(json.dumps(protocol,indent=2));return
        if (RUN/'PROMOTION_COMPLETE.json').exists():
            state('ALREADY_PROMOTED');return
        state('WAITING_FOR_TRAINING_AND_GENERATION')
        while not (RUN/'LENGTH_VALIDATION.json').exists():
            if read(RUN/'STATUS.json').get('status')=='FAILED':
                raise RuntimeError('Training/generation pipeline failed; evaluation not launched')
            time.sleep(30)
        if read(RUN/'student/trainer_state.json')['global_step'] != 256 or not (RUN/'student/DONE.json').exists():
            raise ValueError('Full student training endpoint not verified')
        for attribute in ['fold','sol','thermo']:
            if read(RUN/f'teachers/{attribute}/DONE.json').get('smoke_run') is not False:
                raise ValueError('Teacher formal-training completion not verified')
        teacher_hashes = read(RUN/'student/PREFLIGHT.json')['teachers']
        for spec in teacher_hashes:
            if sha(weight(Path(spec['path']))) != spec['adapter_weights_sha256']:
                raise ValueError('Teacher weights changed since OPD initialization')
        for p,path in protocol['old_adapters'].items():
            if sha(weight(Path(path))) != protocol['old_weights_sha256'][p]:
                raise ValueError('Old comparison model changed since protocol freeze')
        model_paths = [weight(Path(p)) for p in protocol['old_adapters'].values()]
        model_paths += [weight(RUN/'student')] + [weight(RUN/f'teachers/{p}/best_adapter') for p in ['fold','sol','thermo']]
        current_hashes={str(p):sha(p) for p in model_paths}
        hashes_path=RUN/'EVALUATION_MODEL_HASHES.json'
        if hashes_path.exists() and read(hashes_path)!=current_hashes:
            raise ValueError('Models differ from existing evaluation evidence')
        dump(hashes_path,current_hashes)
        manifest=read(EVAL/'protocol/unconditional_hv.json')
        reuse_old = args.reuse_historical_old_opd or (RUN/'HISTORICAL_OLD_OPD_REUSE.json').exists()
        cached_metrics = read(RUN/'COMPARISON_METRICS.json') if reuse_old and (RUN/'COMPARISON_METRICS.json').exists() else {}
        scored={}; metrics={}
        for name in ['base','fold','sol','thermo','opd','old_fold','old_sol','old_thermo','old_opd']:
            if name == 'old_opd' and reuse_old:
                state('REUSING_HISTORICAL_OLD_OPD', comparison_scope='historical_unmatched_generation')
                scored[name], metrics[name] = reuse_historical_old_opd(manifest)
                dump(RUN/'COMPARISON_METRICS.json',metrics)
                continue
            completed = EVAL/'methods'/name/'scores'
            if reuse_old and name in cached_metrics and (completed/'COMPLETE.json').exists():
                scored[name] = rows(completed/'sequence_scores.tsv')
                validate(scored[name])
                if cached_metrics[name]['complete'] != sum(r['valid_complete'].lower() == 'true' for r in scored[name]):
                    raise ValueError('Cached comparison summary count mismatch')
                metrics[name] = cached_metrics[name]
                continue
            generation=RUN/'generation'/f'{name}.tsv'
            if name.startswith('old_'):
                state('OLD_MODEL_MATCHED_GENERATION',model=name)
                job(f'{name}_matched_generation',[PY,ROOT/'experiments/prodipa/conditional_evaluation/generate_model.py',
                    '--paradigm','unconditional','--model-name',f'protgpt2_aa512_{name}', '--base-model','__PRODIPA_MODELS__/ProtGPT2',
                    '--adapter',protocol['old_adapters'][name[4:]],'--out',generation,'--num-sequences','512','--generation-max-aa','512'],
                    'unconditional',name,'evaluation',0,gpus=1,free_mib=16000)
            validate(rows(generation))
            state('SCORING',model=name)
            scored[name]=score(generation,EVAL/'methods'/name/'scores','unconditional',name,'evaluation',0)
            metrics[name]=summarize_checked(scored[name],manifest)
            dump(RUN/'COMPARISON_METRICS.json',metrics)
        decision=gates(metrics,protocol)
        dump(RUN/'PROMOTION_DECISION.json',decision)
        tsv(RUN/'comparison.tsv',[dict(v, Model=k,
            comparison_scope=v.get('comparison_scope','matched_512aa_generation'),
            score_source=v.get('score_source',str(EVAL/'methods'/k/'scores/sequence_scores.tsv')))
            for k,v in metrics.items()])
        if decision['pass_all']:
            state('PUBLISHING_VERIFIED_RESULTS');promote(scored,protocol,decision)
            state('PROMOTED',backup=str(RUN/'publication_backup'))
        else:
            state('COMPLETE_NOT_PROMOTED',failed_gates=decision['failed'],note='Official results and weights unchanged')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        state('FAILED',error=repr(error));raise
