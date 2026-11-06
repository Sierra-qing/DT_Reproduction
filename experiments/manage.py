'''
统一实验入口。list/status只读；run执行计划；smoke只做明确标记的短验证。
单GPU最多两路队列，各队列内部串行；检查计划交集，防止重复任务。
'''
import argparse
from datetime import datetime,timezone
import fcntl
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments'
_ATARI_PROBE=None

def atari_runtime():
    path=HERE/'runtime.json'
    return load(path)['atari_python'] if path.exists() else sys.executable

def probe_atari_runtime():
    global _ATARI_PROBE
    if _ATARI_PROBE is not None:return _ATARI_PROBE
    python=atari_runtime()
    if not Path(python).exists():
        _ATARI_PROBE={'error':'missing Atari Python: '+python};return _ATARI_PROBE
    code="import importlib.util,json; names=['torch','atari_py','dopamine','tensorflow','blosc','cv2']; print(json.dumps({n:(importlib.util.find_spec(n).origin if importlib.util.find_spec(n) else None) for n in names}))"
    result=subprocess.run([python,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
    _ATARI_PROBE=json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr[-300:]}
    return _ATARI_PROBE

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(path.read_text())
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False
    except PermissionError:return True

def register_lane(lane,chosen,batch):
    '''同一时刻只能注册互不重叠的计划；正在运行的旧队列也写入了lane1登记。'''
    with (HERE/'state/scheduler.lock').open('a') as scheduler:
        fcntl.flock(scheduler,fcntl.LOCK_EX)
        ids={j['id'] for j in chosen}
        for other in [1,2]:
            if other==lane:continue
            p=HERE/'state'/f'lane{other}.json'
            if p.exists():
                record=load(p)
                overlap=ids.intersection(record['job_ids'])
                if alive(record.get('driver_pid')) and overlap:
                    raise SystemExit('计划与另一运行队列重叠，拒绝启动：'+', '.join(sorted(overlap)))
        save(HERE/'state'/f'lane{lane}.json',{'driver_pid':os.getpid(),'lane':lane,
             'job_ids':sorted(ids),'batch':str(batch),'registered_utc':now()})
def catalog():return {j['id']:j for j in load(HERE/'configs/catalog.json')['jobs']}
def state(job):
    p=HERE/'state'/ (job['id']+'.json')
    return load(p) if p.exists() else None

def completed_run(job):
    saved=state(job)
    if saved and saved.get('state')=='complete' and job['backend']!='gym':
        return ROOT/saved['run_dir'] if saved.get('run_dir') else Path(saved['log']).parent
    candidates=([saved['run_dir']] if saved and saved.get('state')=='complete' and saved.get('run_dir') else [])+job.get('completed_runs',[])
    if saved and saved.get('state')=='complete' and job['backend']=='gym':
        # 兼容整理前已启动的串行控制器；并行期间它可能未能唯一推断结果目录。
        candidates+=sorted((HERE/'runs').glob('*-'+job['id']),reverse=True)
    for value in candidates:
        p=Path(value);p=p if p.is_absolute() else ROOT/p
        if (p/'checkpoint_latest.pt').exists() and (p/'metrics.jsonl').exists():
            rows=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines() if line.strip()]
            expected=job['params']['max_iters']*job['params']['num_steps_per_iter']
            if rows and rows[-1]['global_step']==expected:return p
    return None

def problems(job,jobs):
    if job.get('blocked_reason'):return [job['blocked_reason']]
    backend=job['backend'];issues=[]
    if backend=='gym':
        p=job['params'];path=ROOT/f"gym/data/{p['env']}-{p['dataset']}-v2.pkl"
        if not path.exists():issues.append('missing dataset: '+str(path.relative_to(ROOT)))
        for mod in ['torch','gym','mujoco_py','swanlab']:
            if importlib.util.find_spec(mod) is None:issues.append('missing Python module: '+mod)
    elif backend in ['sweep','atari-sweep']:
        if completed_run(jobs[job['dependency']]) is None:issues.append('dependency incomplete: '+job['dependency'])
    elif backend=='atari':
        runtime=probe_atari_runtime()
        if runtime.get('error'):issues.append(runtime['error'])
        for mod in ['torch','atari_py','dopamine','tensorflow','blosc','cv2']:
            if not runtime.get(mod):issues.append('missing Atari module: '+mod)
        replay=ROOT/'atari/dqn_replay'/job['params']['game']/'1/replay_logs'
        if not replay.is_dir():
            issues.append('missing DQN replay: '+job['params']['game'])
        else:
            suffixes=Counter(p.name.split('.')[-2] for p in replay.glob('*.gz'))
            missing=[str(i) for i in range(50-job['params']['num_buffers'],50) if suffixes[str(i)] not in (6,7)]
            if missing:issues.append('incomplete replay buffer suffixes: '+','.join(missing))
        if runtime.get('atari_py'):
            rom=Path(runtime['atari_py']).parent/'atari_roms'/(job['params']['game'].lower()+'.bin')
            if not rom.exists():issues.append('missing Atari ROM: '+str(rom))
    elif backend=='graph':
        if importlib.util.find_spec('torch') is None:issues.append('missing torch')
    return issues

def command_for(job,mode='online',smoke=False):
    backend=job['backend']
    if backend=='gym':
        p=dict(job['params'])
        if smoke:p.update(max_iters=1,num_steps_per_iter=2,num_eval_episodes=1)
        flags=[]
        for key,value in p.items():
            flags+=['--'+key]+([str(x) for x in value] if isinstance(value,list) else [str(value)])
        output=HERE/('smoke_runs' if smoke else 'runs')
        return [sys.executable,'-u',str(ROOT/'gym/experiment_observed.py'),*flags,
                '--output_dir',str(output),'--run_name',('SMOKE-' if smoke else '')+job['id'],
                '--log_interval','100','--eval_log_interval','10','--batch_snapshot_interval','100',
                '--log_to_swanlab','--swanlab_mode','offline' if smoke else mode,
                '--swanlab_project','decision-transformer-reproduction'],ROOT/'gym'
    if backend=='atari':
        flags=[]
        for key,value in job['params'].items():flags+=['--'+key,str(value)]
        return [atari_runtime(),'-u',str(HERE/'runners/atari.py'),*flags,'--job_id',job['id'],'--swanlab_mode',mode],ROOT/'atari'
    if backend in ['sweep','atari-sweep']:
        runner='atari_sweep.py' if backend=='atari-sweep' else 'return_sweep.py'
        python=atari_runtime() if backend=='atari-sweep' else sys.executable
        return [python,'-u',str(HERE/'runners'/runner),'--job_id',job['id'],'--swanlab_mode',mode],ROOT
    if backend=='graph':
        params=dict(job['params'])
        if smoke:params.update(updates=2,eval_interval=2)
        flags=[]
        for key,value in params.items():flags+=['--'+key,str(value)]
        return [sys.executable,'-u',str(HERE/'reimplementations/graph_shortest_path.py'),*flags,
                '--output_dir',str(HERE/('smoke_runs' if smoke else 'runs')),
                '--swanlab_mode','offline' if smoke else mode],ROOT
    raise ValueError('No verified implementation: '+job['id'])

def selected(args,jobs):
    if args.plan:return [jobs[x] for x in load(HERE/'configs'/(args.plan+'.json'))['job_ids']]
    return [jobs[x] for x in args.ids] if args.ids else list(jobs.values())

def execute(job,jobs,batch,mode,smoke=False):
    before=set((HERE/('smoke_runs' if smoke else 'runs')).glob('*'))
    command,cwd=command_for(job,mode,smoke)
    record={'job_id':job['id'],'state':'running','started_utc':now(),'command':command,
            'execution_lane':os.getenv('DT_EXPERIMENT_LANE','1'),'concurrency_limit':2,
            'cwd':str(cwd),'smoke':smoke,'log':str(batch/(job['id']+'.log'))}
    sources=[ROOT/'gym/experiment.py',ROOT/'gym/experiment_observed.py',
             ROOT/'gym/experiment_trace.py',HERE/'configs/catalog.json',HERE/'manage.py',
             *sorted((HERE/'runners').glob('*.py')),*sorted((HERE/'reimplementations').glob('*.py'))]
    record['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    path=(batch/(job['id']+'.json')) if smoke else HERE/'state'/(job['id']+'.json')
    save(path,record)
    environment=dict(os.environ,PYTHONUNBUFFERED='1')
    environment['LD_LIBRARY_PATH']='/home/ubuntu/.mujoco/mujoco200/bin:'+environment.get('LD_LIBRARY_PATH','')
    environment['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
    interrupted=False
    with Path(record['log']).open('wb') as log:
      with subprocess.Popen(command,cwd=cwd,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as child:
        record['pid']=child.pid;save(path,record)
        try:
          for line in child.stdout:
            log.write(line);log.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
          code=child.wait()
        except KeyboardInterrupt:
          interrupted=True;child.send_signal(signal.SIGINT)
          try:code=child.wait(timeout=30)
          except subprocess.TimeoutExpired:child.terminate();code=child.wait()
    after=set((HERE/('smoke_runs' if smoke else 'runs')).glob('*'))
    candidates=sorted(p for p in after-before if p.name.endswith('-'+job['id']) or p.name.startswith(job['id']+'-'))
    record.update(state='interrupted' if interrupted else 'complete' if code==0 else 'failed',
                  returncode=code,finished_utc=now())
    if len(candidates)==1:record['run_dir']=str(candidates[0].relative_to(ROOT))
    save(path,record)
    if interrupted:raise KeyboardInterrupt
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','status','run','smoke'])
    parser.add_argument('ids',nargs='*')
    parser.add_argument('--plan',choices=[p.stem for p in (HERE/'configs').glob('*.json') if p.stem!='catalog'])
    parser.add_argument('--swanlab_mode',choices=['online','offline'],default='online')
    parser.add_argument('--lane',type=int,choices=[1,2],default=1,
                        help='最多两个互不重叠队列；每个队列内部串行，默认1')
    args=parser.parse_args();jobs=catalog();chosen=selected(args,jobs)
    if args.action in ['list','status']:
      rows=[]
      for job in chosen:
        issues=problems(job,jobs);saved=state(job)
        status='complete' if completed_run(job) else (saved['state'] if saved and saved['state'] in ['running','failed','interrupted'] else 'blocked' if issues else 'ready')
        rows.append({'id':job['id'],'state':status,'reason':'; '.join(issues),'backend':job['backend']})
      print(f'{"ID":58} {"STATE":12} DETAILS')
      for row in rows:print(f'{row["id"]:58} {row["state"]:12} {row["reason"]}')
      save(HERE/'reports/status.json',{'generated_utc':now(),'jobs':rows})
      return
    if not args.ids and not args.plan:parser.error('执行必须显式指定实验ID或 --plan，避免误启动全部实验')
    HERE.joinpath('state').mkdir(exist_ok=True)
    lock_name='gpu_queue.lock' if args.lane==1 else 'gpu_queue_lane2.lock'
    with (HERE/'state'/lock_name).open('a') as lock:
      try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:raise SystemExit('该运行通道已有队列；不重复启动')
      import torch
      if not torch.cuda.is_available():raise SystemExit('CUDA不可用，不回退CPU')
      assert (torch.ones(4,device='cuda')*2).sum().item()==8.
      if args.swanlab_mode=='online' and args.action!='smoke':
        from swanlab.package import has_api_key
        if not has_api_key():raise SystemExit('请先 swanlab login；不会静默切换离线')
      stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
      batch=HERE/'batches'/stamp;batch.mkdir(parents=True)
      register_lane(args.lane,chosen,batch)
      os.environ['DT_EXPERIMENT_LANE']=str(args.lane)
      save(batch/'plan.json',{'created_utc':now(),'jobs':chosen,'mode':args.swanlab_mode,'action':args.action})
      (HERE/'latest_batch.txt').write_text(str(batch)+'\n')
      results=[]
      for job in chosen:
        if args.action!='smoke' and completed_run(job):
            results.append({'job_id':job['id'],'state':'skipped_complete'});continue
        issues=problems(job,jobs)
        if issues:
            result={'job_id':job['id'],'state':'blocked','reasons':issues}
            results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        else:
            results.append(execute(job,jobs,batch,args.swanlab_mode,args.action=='smoke'))
        save(batch/'status.json',{'updated_utc':now(),'state':'running','results':results})
      save(batch/'status.json',{'updated_utc':now(),'state':'finished','results':results})
      print('Batch finished:',batch,flush=True)
      if any(x['state'] in ['failed','blocked'] for x in results):raise SystemExit(1)

if __name__=='__main__':main()
