'''从原版Atari最终checkpoint做目标回报扫描，不再读取训练replay或更新参数。'''
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'gym'),str(ROOT/'atari')]
from manage import catalog,completed_run

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job_id',required=True)
 p.add_argument('--swanlab_mode',choices=['online','offline'],default='online');args=p.parse_args()
 import torch
 if not torch.cuda.is_available():raise SystemExit('CUDA unavailable, no CPU fallback')
 from mingpt.model_atari import GPT,GPTConfig
 from mingpt.trainer_atari import Trainer,TrainerConfig
 from mingpt.utils import set_seed
 from swanlab_sidecar import SwanLabSidecar
 jobs=catalog();job=jobs[args.job_id];source=completed_run(jobs[job['dependency']])
 if source is None:raise SystemExit('Missing trained model: '+job['dependency'])
 checkpoint=source/'checkpoints/epoch_05.pt';data=torch.load(checkpoint,map_location='cpu')
 config=dict(data['model_config']);vocab=config.pop('vocab_size');block=config.pop('block_size')
 model=GPT(GPTConfig(vocab,block,**config));model.load_state_dict(data['model_state_dict'])
 trainer=Trainer(model,None,None,TrainerConfig(game=data['config']['game'],seed=job['seed'],max_timestep=config['max_timestep']))
 out=ROOT/'experiments/runs'/(args.job_id+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True)
 args.swanlab_project='decision-transformer-reproduction';args.swanlab_workspace=None
 args.training_python=sys.executable
 args.logging_note='Separate Python3.8 SwanLab process; process hardware metrics refer to logger'
 logger=SwanLabSidecar(args,out);rows=[]
 try:
  for index,target in enumerate(job['targets']):
   set_seed(job['seed'])
   with torch.no_grad():score=float(trainer.get_returns(target))
   rows.append({'target':target,'return_mean':score,'episodes':10,'seed':job['seed']})
   logger.scalar('sweep/raw_return_mean',score,index,index)
   logger.scalar('sweep/target_rtg',target,index,index)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots();ax.plot([r['target'] for r in rows],[r['return_mean'] for r in rows],'o-')
  ax.set(xlabel='Requested RTG',ylabel='Raw episode return',title=args.job_id);fig.tight_layout()
  fig.savefig(out/'return_sweep.png',dpi=150);logger.media('sweep/requested_vs_achieved',out/'return_sweep.png',0)
  (out/'summary.json').write_text(json.dumps({'job':job,'checkpoint':str(checkpoint),'results':rows},indent=2))
 finally:logger.close()
if __name__=='__main__':main()
