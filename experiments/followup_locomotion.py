'''
等待现有第一通道完整释放，再短验证HalfCheetah/Walker并接续正式任务。
不停止现有任务，不在评估间隙插入第三个实验。
'''
import fcntl,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'experiments/state/followup_locomotion.json'
def status(phase,**extra):
 STATE.write_text(json.dumps({'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def free():
 with (ROOT/'experiments/state/gpu_queue.lock').open('a') as handle:
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  fcntl.flock(handle,fcntl.LOCK_UN);return True

def main():
 status('waiting_for_lane1')
 print('Waiting for existing lane1 queue to finish; current training is preserved.',flush=True)
 while not free():time.sleep(30)
 checks=[sys.executable,'-u',str(ROOT/'experiments/manage.py'),'smoke',
         'dt-halfcheetah-medium-s0','dt-walker2d-medium-s0','--lane','1']
 status('checking_environments',command=checks)
 code=subprocess.call(checks,cwd=ROOT)
 if code:
  status('failed_smoke',returncode=code);raise SystemExit(code)
 command=[sys.executable,'-u',str(ROOT/'experiments/manage.py'),'run','--plan','locomotion-next','--lane','1']
 status('running_locomotion_plan',command=command)
 code=subprocess.call(command,cwd=ROOT)
 status('complete' if code==0 else 'failed',returncode=code)
 raise SystemExit(code)
if __name__=='__main__':main()
