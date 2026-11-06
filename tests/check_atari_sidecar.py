'''只验证跨Python版本日志通道，不启动策略训练，不上传数据。'''
import sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/runners'))
from swanlab_sidecar import SwanLabSidecar
out=Path(tempfile.mkdtemp(prefix='atari-log-sidecar-',dir=str(ROOT/'experiments/smoke_runs')))
args=SimpleNamespace(game='Breakout',model_type='reward_conditioned',pct=100,context_length=30,seed=123,
 swanlab_project='decision-transformer-reproduction',swanlab_workspace=None,swanlab_mode='offline')
logger=SwanLabSidecar(args,out);logger.scalar('test/action_cross_entropy',.5,1,1);logger.close()
assert (out/'swanlab_sidecar_ready.json').exists()
print('PASS Python3.7 client -> Python3.8 SwanLab worker:',out)
