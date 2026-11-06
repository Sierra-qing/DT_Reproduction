'''等待独立Atari的Torch安装，再验证软件依赖。数据/ROM缺失不会伪称可训练。'''
import argparse,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
STATUS=ROOT/'experiments/setup/atari_status.json'
PYTHON='/home/ubuntu/miniconda3/envs/decision-transformer-atari/bin/python'
def save(state,**extra):
 STATUS.write_text(json.dumps({'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2)+'\n')
def installer_running(pid):
 try:return 'pip' in (Path('/proc')/str(pid)/'cmdline').read_text() and 'torch==' in (Path('/proc')/str(pid)/'cmdline').read_text()
 except FileNotFoundError:return False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--wait_pid',type=int,default=0);args=parser.parse_args()
 save('installing_torch',installer_pid=args.wait_pid)
 while args.wait_pid and installer_running(args.wait_pid):time.sleep(15)
 code="""import sys,json,numpy,torch,tensorflow,atari_py,cv2,blosc,gin
from dopamine.replay_memory import circular_replay_buffer
sys.path.insert(0,'atari')
from fixed_replay_buffer import FixedReplayBuffer
from mingpt.model_atari import GPT,GPTConfig
print(json.dumps({'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
 'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
 'tensorflow':tensorflow.__version__,'numpy':numpy.__version__,'roms':atari_py.list_games(),
 'original_replay_import':'passed','original_model_import':'passed',
 'gpu_policy_forward_backward':'deferred until a training lane is available'}))
"""
 result=subprocess.run([PYTHON,'-c',code],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 if result.returncode:
  save('software_check_failed',error=result.stderr[-4000:]);raise SystemExit(result.returncode)
 report=json.loads(result.stdout.splitlines()[-1])
 (ROOT/'experiments/reports/atari_runtime_check.json').write_text(json.dumps(report,indent=2)+'\n')
 missing=[]
 if 'breakout' not in report['roms']:missing.append('Breakout ROM not installed')
 if not (ROOT/'atari/dqn_replay/Breakout/1/replay_logs').exists():missing.append('Original DQN replay unavailable from upstream (401/403 observed)')
 save('software_ready_inputs_blocked' if missing else 'software_ready_pending_gpu_check',runtime=report,blockers=missing)
 print(STATUS.read_text(),flush=True)
if __name__=='__main__':main()
