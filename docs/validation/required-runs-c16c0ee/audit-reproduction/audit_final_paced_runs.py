from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from collections import Counter
import h5py
import numpy as np

SAMPLE_RATE=20_000
BLOCK=100
LEAD=2_000
SAMPLES=36_000_000
BLOCKS=SAMPLES//BLOCK
REV='c16c0eee0174c7c016d05a5ce617186a3c4c42d8'
CONTROLLERS=('rule','pid','esn')


def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dec(x):return x.decode('utf-8',errors='replace').rstrip('\0') if isinstance(x,bytes) else str(x)
def json_load(path):return json.loads(path.read_text(encoding='utf-8'))

def check_h5(path:Path,env:dict,case:dict,source_h5_sha:str,abf_path:Path,abf_sha:str,progress_path:Path):
 m=case; controller=m['controller']; failures=[]
 if not path.is_file():raise FileNotFoundError(path)
 committed=int(m.get('recording_committed_samples',-1));expected_source=int(m.get('source_samples',-1))
 progress_lines=[json.loads(s) for s in progress_path.read_text(encoding='utf-8').splitlines() if s.strip()]
 caseid=m['case_id'];
 with h5py.File(path,'r') as h:
  md=json.loads(dec(h.attrs['metadata_json'])); cfg=md.get('configuration',{}); src=md.get('source',{}); imported=src.get('metadata',{}).get('source',{})
  attrs={k:dec(h.attrs.get(k,'')) for k in ('schema','format_tag')}
  committed_attr=int(h.attrs.get('committed_samples',-1));complete=int(h.attrs.get('complete',0))
  _expected_shapes={'signals/ai_raw_V':(2,committed_attr),'signals/electrode_mV':(2,committed_attr),'signals/ao_command_V':(1,committed_attr)}
  shapes={k:tuple(h[k].shape) for k in _expected_shapes}
  if attrs['schema']!='echo-chamber-recording':failures.append('wrong H5 schema')
  if attrs['format_tag']!='echoChamber_H5_v5':failures.append('wrong H5 format tag')
  if complete!=1:failures.append('H5 complete flag not set')
  if committed_attr!=committed:failures.append(f'H5 committed_samples {committed_attr} != report {committed}')
  if int(h.attrs.get('sample_rate_hz',-1))!=SAMPLE_RATE:failures.append('H5 sample rate mismatch')
  if dec(md.get('data_file',''))!=path.name:failures.append('metadata data_file does not match recording filename')
  if md.get('git_revision')!=REV or env.get('git_revision')!=REV:failures.append('revision provenance mismatch')
  if cfg.get('controller')!=controller:failures.append('controller configuration provenance mismatch')
  for k,v in [('sample_rate',SAMPLE_RATE),('chunk_size',BLOCK),('ao_target_lead_ms',100.0),('electrode_labels',['CA3','EC']),('ctx_index',1)]:
   if cfg.get(k)!=v:failures.append(f'config mismatch: {k}')
  if cfg.get('sample_rate')!=SAMPLE_RATE or cfg.get('chunk_size')!=BLOCK or cfg.get('ao_target_lead_ms')!=100.0:failures.append('report config mismatch')
  for k,shape in _expected_shapes.items():
   if shapes[k]!=shape:failures.append(f'{k} shape mismatch {shapes[k]} != {shape}')
  if src.get('path')!=str(Path(env['source_path']).resolve()) or env.get('source_path')!=str(Path(env['source_path']).resolve()):
   # Windows report paths are already canonical; compare resolved paths below.
   if Path(src.get('path','')).resolve()!=Path(env.get('source_path','')).resolve():failures.append('H5 source path differs from report source path')
  if src.get('sha256')!=source_h5_sha or env.get('source_sha256')!=source_h5_sha:failures.append('source H5 checksum mismatch')
  if int(src.get('sample_count',-1))!=SAMPLES or float(src.get('sample_rate_hz',-1))!=SAMPLE_RATE:failures.append('source geometry metadata mismatch')
  if src.get('labels')!=['CA3','EC'] or src.get('roles')!=['CA3','Cortex'] or int(src.get('ctx_index',-1))!=1:failures.append('source channel-role metadata mismatch')
  if imported.get('filename')!=abf_path.name or imported.get('sha256')!=abf_sha:failures.append('embedded raw ABF provenance mismatch')
  gap_rows=int(h['events/sample_discontinuities'].shape[0])
  if gap_rows:failures.append(f'{gap_rows} H5 source discontinuity rows')
  blocks=h['diagnostics/blocks']; controllers=h['diagnostics/controller_blocks']
  block_rows=int(blocks.shape[0]);controller_rows=int(controllers.shape[0])
  if block_rows!=committed_attr//BLOCK or controller_rows!=committed_attr//BLOCK:failures.append('diagnostic row count does not match committed source extent')
  gap_index_count=0; bad_block_count=0; bad_controller_index_count=0; target_mismatch=0; feature_out_of_bounds=0; invalid_total=0; stimulated_total=0;ready=calibrated=feature_valid=0; generations=Counter(); controller_ids=Counter()
  max_rows=16_384
  for start in range(0,block_rows,max_rows):
   stop=min(start+max_rows,block_rows);b=blocks[start:stop];c=controllers[start:stop]
   expected=np.arange(start,stop,dtype=np.uint64)*BLOCK
   bad_block_count+=int(np.count_nonzero(b['sample_offset']!=expected))+int(np.count_nonzero(b['ai_sample_index']!=expected))+int(np.count_nonzero(b['sample_count']!=BLOCK))
   target_mismatch+=int(np.count_nonzero(b['ao_target_sample_index']!=expected+LEAD))
   bad_controller_index_count+=int(np.count_nonzero(c['sample_offset']!=expected))
   if 'feature_invalid_sample_count' in c.dtype.names:
    inv=c['feature_invalid_sample_count'].astype(np.uint64);stim=c['stimulated_sample_count'].astype(np.uint64);elig=c['feature_eligible_sample_count'].astype(np.uint64)
    invalid_total+=int(inv.sum());stimulated_total+=int(stim.sum());feature_out_of_bounds+=int(np.count_nonzero((inv>BLOCK)|(stim>BLOCK)|(elig>BLOCK)))
   ready+=int(np.count_nonzero(c['ready']));calibrated+=int(np.count_nonzero(c['calibrated']));feature_valid+=int(np.count_nonzero(c['feature_valid']))
   generations.update(int(x) for x in np.unique(c['config_generation']))
   controller_ids.update(dec(x) for x in np.unique(c['controller_id']))
  if bad_block_count:failures.append(f'{bad_block_count} block source index/count deviations')
  if bad_controller_index_count:failures.append(f'{bad_controller_index_count} controller index deviations')
  if feature_out_of_bounds:failures.append(f'{feature_out_of_bounds} feature count fields exceeded block size')
  ao=h['signals/ao_command_V'];ao_nonzero=0
  for start in range(0,committed_attr,1_048_576):ao_nonzero+=int(np.count_nonzero(ao[:,start:min(start+1_048_576,committed_attr)]))
  actions=h['events/action_stages'][:]
  stage_counts=Counter(dec(x) for x in actions['stage']);reason_counts=Counter(dec(x) for x in actions['reason'])
  ids_by_stage={}
  rows_by_pulse=[]
  ao_commit_range_samples=0;commit_range_invalid=0;ao_commit_min=None;ao_commit_max=-1;zero_range_min=None;zero_range_max=-1;max_ao_stage_end=-1;min_ao_stage_start=None;fault_zero_count=0;fault_zero_samples=0
  for row in actions:
   pid=dec(row['request_id']);stage=dec(row['stage']);start=int(row['sample_index']);end=int(row['range_end_sample_index'])
   if pid:ids_by_stage.setdefault(stage,set()).add(pid);rows_by_pulse.append((pid,stage,start,end,int(row['generation']),dec(row['controller'])))
   if stage=='ao_commit' and pid:
    if end<=start:commit_range_invalid+=1
    else:
     ao_commit_range_samples+=end-start
     ao_commit_min=start if ao_commit_min is None else min(ao_commit_min,start)
     ao_commit_max=max(ao_commit_max,end)
   if stage=='zero_substituted':
    fault_zero_count+=1
    if end>start:
     fault_zero_samples+=end-start
     zero_range_min=start if zero_range_min is None else min(zero_range_min,start)
     zero_range_max=max(zero_range_max,end)
   if end>start:
    min_ao_stage_start=start if min_ao_stage_start is None else min(min_ao_stage_start,start);max_ao_stage_end=max(max_ao_stage_end,end)
  h5_requests=set(dec(x) for x in m.get('pulse_request_ids',[]));h5_rejected=set(dec(x) for x in m.get('pulse_rejected_ids',[]));h5_committed=set(dec(x) for x in m.get('pulse_commit_ids',[]))
  seen_request=ids_by_stage.get('rendered_for_ao',set())|ids_by_stage.get('rejected_refractory',set())|ids_by_stage.get('rejected',set())
  seen_committed=ids_by_stage.get('ao_commit',set())
  seen_rejected=ids_by_stage.get('rejected_refractory',set())|ids_by_stage.get('rejected',set())
  id_checks={'requested_ids_match_H5':seen_request==h5_requests,'committed_ids_match_H5':seen_committed==h5_committed,'rejected_ids_match_H5':seen_rejected==h5_rejected,'request_partition_complete':h5_requests==h5_committed|h5_rejected and not (h5_committed & h5_rejected)}
  if not all(id_checks.values()):failures.extend([f'pulse identity mismatch: {k}' for k,v in id_checks.items() if not v])
  if commit_range_invalid:failures.append(f'{commit_range_invalid} invalid AO commit ranges')
  if ao_commit_range_samples!=int(m.get('pulse_commit_samples',-1)):failures.append(f'AO commit range total {ao_commit_range_samples} != report pulse_commit_samples')
  if ao_nonzero!=int(m.get('ao_nonzero_generated_samples',-1)):failures.append(f'H5 AO nonzero count {ao_nonzero} != report {m.get("ao_nonzero_generated_samples")}')
  if target_mismatch:failures.append(f'{target_mismatch} diagnostic AO targets do not equal source offset + 2000')
  stage_generation_unique=sorted(set(int(x) for x in actions['generation']))
  controller_detail={'id_checks':id_checks,'pulse_ids_by_stage':{k:len(v) for k,v in ids_by_stage.items()},'ao_commit_ranges':stage_counts.get('ao_commit',0),'ao_commit_range_samples':ao_commit_range_samples,'ao_commit_ao_start_min':ao_commit_min,'ao_commit_ao_end_max_exclusive':ao_commit_max,'rejected_stage_rows':stage_counts.get('rejected_refractory',0)+stage_counts.get('rejected',0),'fault_zero_stage_rows':fault_zero_count,'fault_zero_stage_samples':fault_zero_samples,'fault_zero_ao_start_min':zero_range_min,'fault_zero_ao_end_max_exclusive':zero_range_max,'action_stage_counts':dict(stage_counts),'reason_counts':dict(reason_counts),'action_generation_values':stage_generation_unique,'action_ao_range_min':min_ao_stage_start,'action_ao_range_max_exclusive':max_ao_stage_end}
  row={'controller':controller,'case_id':caseid,'h5_path':str(path),'h5_sha256':sha256(path),'h5_size_bytes':path.stat().st_size,'h5_schema':attrs['schema'],'h5_format_tag':attrs['format_tag'],'h5_complete':complete,'recorded_samples':committed_attr,'expected_source_samples':SAMPLES,'recording_fraction':committed_attr/SAMPLES,'signal_shapes':{k:list(v) for k,v in shapes.items()},'block_rows':block_rows,'controller_rows':controller_rows,'source_gaps':gap_rows,'block_index_count_errors':bad_block_count,'controller_index_count_errors':bad_controller_index_count,'ao_target_offset_mismatch_rows':target_mismatch,'controller_ids':dict(controller_ids),'config_generations':dict(generations),'ready_rows':ready,'calibrated_rows':calibrated,'feature_valid_rows':feature_valid,'feature_invalid_sample_sum':invalid_total,'stimulated_sample_sum':stimulated_total,'feature_count_out_of_bounds':feature_out_of_bounds,'ao_recorded_nonzero_samples':ao_nonzero,'report_metrics':{k:m.get(k) for k in ('source_samples','processed_blocks','duration_source_s','recording_completed','recording_committed_samples','runtime_fault','process_p99_ms','process_max_ms','deadline_exceeded_blocks','ao_active_demand_misses','ao_late_commands','ao_drain_complete','ao_final_queue_depth','requested_rate_blocks','requested_pulse_events','rendered_pulse_events','rejected_pulse_events','unclassified_pulse_events','pulse_commit_samples','pulse_request_ids','pulse_rejected_ids','pulse_commit_ids','ao_nonzero_generated_samples','max_requested_rate_hz','actuator_rate_phase_high_water','actuator_rate_phase_reset_count','actuator_rate_phase_final','browser_ui_required','browser_connected','browser_disconnect_count','browser_disconnected_duration_ms','browser_connected_at_finish','browser_disconnect_open_at_finish','ui_transport_kind','progress_events_accepted','progress_events_written','progress_events_dropped','progress_events_rejected','progress_events_pending','progress_writer_close_timed_out','progress_writer_owner_alive','recorder_max_backlog')},'browser_transport_environment':env.get('ui_transport'),'channel_mapping':env.get('ui_channel_mapping'),'recording_file_retained':env.get('recording_file_retained'),'source_path_matches':Path(env.get('source_path','')).resolve()==Path(src.get('path','')).resolve(),'source_sha_matches':env.get('source_sha256')==source_h5_sha==src.get('sha256'),'embedded_abf_sha_matches':imported.get('sha256')==abf_sha,'metadata_configuration':{k:cfg.get(k) for k in ('controller','sample_rate','chunk_size','ao_target_lead_ms','electrode_labels','ctx_index')},'event_summary':controller_detail,'verification_failures':failures}
 return row, progress_lines

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,action='append',required=True);ap.add_argument('--source-h5',type=Path,required=True);ap.add_argument('--raw-abf',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 source_h5=a.source_h5.resolve(strict=True);abf=a.raw_abf.resolve(strict=True);source_sha=sha256(source_h5);abf_sha=sha256(abf)
 summaries=[];progress_summaries=[];errors=[];case_envs=[]
 for report_path in a.report:
  doc=json_load(report_path);case=doc['cases'][0];env=doc['environments'][0]
  if doc.get('schema')!='echo-chamber-controller-benchmark-v1':errors.append(f'{report_path}: wrong report schema')
  if doc.get('git_revision')!=REV or env.get('git_revision')!=REV:errors.append(f'{report_path}: wrong revision')
  if len(doc.get('cases',[]))!=1 or len(doc.get('environments',[]))!=1:errors.append(f'{report_path}: expected one case/environment')
  if doc.get('physical_pulse_claims') is not False or env.get('physical_pulse_claims') is not False:errors.append(f'{report_path}: physical pulse claim not false')
  acceptance=doc.get('acceptance',[{}])[0];coverage=acceptance.get('coverage',{})
  if acceptance.get('scope')!='paced_30min_case' or acceptance.get('full_acceptance_status')!='FAIL' or acceptance.get('full_acceptance_passed') is not False:errors.append(f'{report_path}: expected full-intent FAIL classification')
  for k,v in [('duration_seconds',1800.0),('processed_blocks',BLOCKS),('source_samples',SAMPLES),('sample_rate_hz',SAMPLE_RATE),('block_samples',BLOCK),('block_ms',5.0),('deadline_ms',5.0)]:
   if coverage.get('observed',{}).get(k)!=v:errors.append(f'{report_path}: coverage {k} differs from full-run target')
  if not coverage.get('complete'):errors.append(f'{report_path}: report does not record full processing coverage')
  if case.get('controller') not in CONTROLLERS:errors.append(f'{report_path}: unexpected controller')
  if case.get('ao_lead_ms')!=100.0:errors.append(f'{report_path}: AO lead differs from 100ms default')
  progress_path=report_path.parent/'progress.jsonl'
  if not progress_path.is_file():raise FileNotFoundError(progress_path)
  result,progress=check_h5(Path(env['recording_file']),env,case,source_sha,abf,abf_sha,progress_path)
  case_envs.append(env);summaries.append(result)
  progress_complete=sum(1 for x in progress if x.get('type')=='validation_complete' and (x.get('json') or x.get('result')) and Path(x.get('json') or x.get('result')).resolve()==report_path.resolve())
  progress_types=Counter(x.get('type','') for x in progress)
  progress_sequences=[int(x['progress_sequence']) for x in progress if isinstance(x.get('progress_sequence'),int)]
  progress_summaries.append({'controller':case['controller'],'progress_log':str(progress_path),'event_count':len(progress),'event_types':dict(progress_types),'validation_complete_references_report':progress_complete==1,'progress_sequence_rows':len(progress_sequences),'progress_sequence_unique':len(set(progress_sequences))==len(progress_sequences),'progress_sequence_min':min(progress_sequences) if progress_sequences else None,'progress_sequence_max':max(progress_sequences) if progress_sequences else None,'browser_events':[{'type':x.get('type'),'connected':x.get('connected'),'disconnect_count':x.get('browser_disconnect_count'),'timestamp_utc':x.get('timestamp_utc')} for x in progress if 'browser' in x.get('type','')],'first_fault_events':[{'type':x.get('type'),'source_sample_index':x.get('source_sample_index'),'fault':x.get('fault'),'generated_sample_count':x.get('generated_sample_count'),'next_write_block':x.get('next_write_block'),'ao_service_gap_ms':x.get('ao_service_gap_ms'),'observation_is_fault_onset':x.get('observation_is_fault_onset')} for x in progress if x.get('type') in ('ao_clock_fault','first_active_demand_miss','first_late_command')], 'ao_progress_totals':[(x.get('type'),x.get('ao',{}).get('zero_substitutions'),x.get('ao',{}).get('pulse_requests'),x.get('ao',{}).get('pulse_committed_samples'),x.get('ao',{}).get('generated_sample_count')) for x in progress if x.get('type') in ('run_complete','run_progress')]})
 if {r['controller'] for r in summaries}!=set(CONTROLLERS):errors.append('reports do not cover each requested controller exactly once')
 if len({r['controller'] for r in summaries})!=len(summaries):errors.append('duplicate controller in evidence reports')
 # Document the aggregate timing fail without converting three individual reports into a successful collection.
 summary={'schema':'paced-30min-three-controller-evidence-audit-v1','source_reports':[str(p.resolve()) for p in a.report],'git_revision':REV,'source_h5':str(source_h5),'source_h5_sha256':source_sha,'raw_abf':str(abf),'raw_abf_sha256':abf_sha,'full_intent_scope':'three separate 30-minute paced cases; each full-duration case fails unchanged processing timing gate; this is not a passed three-case collection','all_recordings_metadata_and_index_checks_passed':all(not r['verification_failures'] for r in summaries),'all_three_individual_full_acceptance_passed':False,'physical_pulse_claims':False,'global_issues':errors,'cases':summaries,'progress':progress_summaries}
 out=a.output_dir
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 (out/'paced_30min_evidence.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 fields=['controller','case_id','recorded_samples','expected_source_samples','recording_fraction','block_rows','controller_rows','source_gaps','ao_recorded_nonzero_samples','runtime_fault','process_p99_ms','process_max_ms','deadline_exceeded_blocks','ao_active_demand_misses','ao_late_commands','requested_rate_blocks','requested_pulse_events','rendered_pulse_events','rejected_pulse_events','pulse_commit_samples','ao_nonzero_generated_samples','browser_connected','browser_disconnect_count','browser_connected_at_finish','progress_events_accepted','progress_events_written','progress_events_dropped','progress_events_pending','all_index_event_provenance_checks_clear']
 with (out/'paced_30min_segments.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in summaries:
   m=r['report_metrics'];w.writerow({'controller':r['controller'],'case_id':r['case_id'],'recorded_samples':r['recorded_samples'],'expected_source_samples':r['expected_source_samples'],'recording_fraction':r['recording_fraction'],'block_rows':r['block_rows'],'controller_rows':r['controller_rows'],'source_gaps':r['source_gaps'],'ao_recorded_nonzero_samples':r['ao_recorded_nonzero_samples'],'runtime_fault':m['runtime_fault'],'process_p99_ms':m['process_p99_ms'],'process_max_ms':m['process_max_ms'],'deadline_exceeded_blocks':m['deadline_exceeded_blocks'],'ao_active_demand_misses':m['ao_active_demand_misses'],'ao_late_commands':m['ao_late_commands'],'requested_rate_blocks':m['requested_rate_blocks'],'requested_pulse_events':m['requested_pulse_events'],'rendered_pulse_events':m['rendered_pulse_events'],'rejected_pulse_events':m['rejected_pulse_events'],'pulse_commit_samples':m['pulse_commit_samples'],'ao_nonzero_generated_samples':m['ao_nonzero_generated_samples'],'browser_connected':m['browser_connected'],'browser_disconnect_count':m['browser_disconnect_count'],'browser_connected_at_finish':m['browser_connected_at_finish'],'progress_events_accepted':m['progress_events_accepted'],'progress_events_written':m['progress_events_written'],'progress_events_dropped':m['progress_events_dropped'],'progress_events_pending':m['progress_events_pending'],'all_index_event_provenance_checks_clear':not r['verification_failures']})
 lines=['# Final paced 30-minute evidence audit','',f"Source revision: `{REV}`. Provenance H5 and ABF hashes are recorded in `paced_30min_evidence.json`.",'All three reports are full-intent paced cases. Each report classifies full acceptance as **FAIL** because the unchanged strict timing gate is not met; this bundle does not combine them into a passing collection.','', '| Controller | Recorded samples / 36M | p99 ms | Max ms | Deadline misses | AO miss / late | Browser disconnects | Pulse requests / commits / nonzero samples |', '|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in summaries:
  m=r['report_metrics'];lines.append(f"| {r['controller']} | {r['recorded_samples']:,} | {m['process_p99_ms']} | {m['process_max_ms']} | {m['deadline_exceeded_blocks']} | {m['ao_active_demand_misses']} / {m['ao_late_commands']} | {m['browser_disconnect_count']} | {m['requested_pulse_events']} / {m['pulse_commit_samples']} samples / {r['ao_recorded_nonzero_samples']} samples |")
 lines+=['','## Recording and event verification','',f"All three H5 files were checked in bounded chunks for report/controller/source metadata, v5 schema, committed extent, shapes, every block offset/AI index/sample count, controller diagnostic offsets, generation IDs, per-block feature count bounds, source discontinuities, AO command nonzero sample counts, pulse-ID partitions, and committed AO pulse ranges. Canonical H5 source checksum: `{source_sha}`; underlying ABF checksum: `{abf_sha}`.",'','## Browser and writer health','','The per-case metrics and progress logs preserve browser connection/disconnect state, progress accepted/written/dropped/pending counters, scheduler drain, and first AO fault events. See JSON for exact records.','', 'No claim of physical output is made.']
 (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({'output_dir':str(out.resolve()),'cases':len(summaries),'all_h5_metadata_and_index_checks_passed':summary['all_recordings_metadata_and_index_checks_passed'],'all_three_full_acceptance_passed':False,'global_issues':errors,'recorded_samples':{r['controller']:r['recorded_samples'] for r in summaries}},indent=2))
if __name__=='__main__':main()
