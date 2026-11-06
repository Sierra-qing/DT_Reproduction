'''
在原版Atari入口外包装百分位筛选、论文明确差异和SwanLab观测。
依赖原版DQN replay、Atari ROM及原始依赖。不会伪造数据或回退CPU。
'''
import argparse,json,runpy,sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]

def select_percentile(data,pct):
    '''按完整轨迹回报排序，在pct%的时间步预算内保留高回报轨迹。'''
    import numpy as np
    if pct==100:return data
    observations,actions,returns,ends,rtg,timesteps=data
    ends=np.asarray(ends,dtype=int);starts=np.r_[0,ends[:-1]];lengths=ends-starts
    scores=np.asarray([rtg[start] for start in starts]);order=np.argsort(scores)
    chosen=[int(order[-1])];used=lengths[chosen[0]];budget=max(1,int(lengths.sum()*pct/100.))
    for index in order[-2::-1]:
        if used+lengths[index]>budget:break
        chosen.append(int(index));used+=lengths[index]
    chosen=sorted(chosen);indices=np.concatenate([np.arange(starts[i],ends[i]) for i in chosen])
    new_ends=np.cumsum(lengths[chosen]);new_obs=[observations[i] for i in indices]
    new_times=np.concatenate([np.asarray(timesteps)[starts[i]:ends[i]] for i in chosen]+[np.array([0])])
    return new_obs,np.asarray(actions)[indices],np.r_[scores[chosen],0],new_ends,np.asarray(rtg)[indices],new_times

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key,default in [('game','Breakout'),('seed',123),('context_length',30),('batch_size',128),('epochs',5),
                     ('num_steps',500000),('num_buffers',50),('model_type','reward_conditioned'),('pct',100)]:
  p.add_argument('--'+key,type=type(default),default=default)
 p.add_argument('--job_id',required=True);p.add_argument('--swanlab_mode',default='online',choices=['online','offline'])
 args=p.parse_args()
 import torch
 if not torch.cuda.is_available():raise SystemExit('CUDA unavailable; CPU fallback prohibited')
 sys.path[:0]=[str(ROOT/'atari'),str(ROOT/'gym')]
 import create_dataset
 import mingpt.trainer_atari as training
 from mingpt.model_atari import GPT
 from mingpt.utils import set_seed
 from swanlab_sidecar import SwanLabSidecar
 out=ROOT/'experiments/runs'/(args.job_id+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
 out.mkdir(parents=True)
 config=SimpleNamespace(**{**vars(args),'swanlab_project':'decision-transformer-reproduction','swanlab_workspace':None})
 config.training_python=sys.executable
 config.logging_note='SwanLab runs in a separate Python3.8 process; process hardware metrics refer to the logger'
 logger=SwanLabSidecar(config,out)
 (out/'config.json').write_text(json.dumps(vars(args),indent=2))
 (out/'training_runtime.json').write_text(json.dumps({'python':sys.executable,'python_version':sys.version,
    'torch':torch.__version__,'torch_cuda':torch.version.cuda,
    'logging_process':'separate Python3.8 SwanLab worker'},indent=2))
 original_dataset=create_dataset.create_dataset;original_config=training.TrainerConfig
 original_returns=training.Trainer.get_returns;original_forward=GPT.forward
 original_step=torch.optim.AdamW.step
 counter={'updates':0,'epoch':0,'loss':None,'model':None}
 def dataset(*values,**options):
  result=select_percentile(original_dataset(*values,**options),args.pct)
  (out/'dataset.json').write_text(json.dumps({'sampled_timesteps_after_filter':len(result[0]),
    'trajectories':len(result[3]),'pct':args.pct,'provenance':'original replay sampler then whole-trajectory percentile selection'},indent=2))
  return result
 class Config(original_config):
  def __init__(self,**kwargs):
   # Table8明确final_tokens；官方入口多乘3，因此只在此包装中覆盖并留痕。
   kwargs['final_tokens']=2*args.num_steps*args.context_length
   super().__init__(**kwargs)
 def forward(model,*values,**options):
  result=original_forward(model,*values,**options)
  if model.training and result[1] is not None:counter['loss']=float(result[1].detach().mean().cpu())
  return result
 def step(optimizer,*values,**options):
  result=original_step(optimizer,*values,**options);counter['updates']+=1
  if counter['loss'] is not None:logger.scalar('train/action_cross_entropy',counter['loss'],counter['updates'],counter['updates'])
  logger.scalar('train/learning_rate',optimizer.param_groups[0]['lr'],counter['updates'],counter['updates'])
  return result
 def evaluate(trainer,target):
  # Table8明确Qbert=2500、Seaquest=1450；覆盖官方硬编码差异。
  if args.model_type=='reward_conditioned':
   target={'Breakout':90,'Qbert':2500,'Pong':20,'Seaquest':1450}[args.game]
  result=original_returns(trainer,target);counter['epoch']+=1
  logger.scalar('eval/episode_return_mean',result,counter['updates'],counter['updates'])
  model=trainer.model.module if hasattr(trainer.model,'module') else trainer.model
  folder=out/'checkpoints';folder.mkdir(exist_ok=True)
  torch.save({'model_state_dict':model.state_dict(),'model_config':vars(model.config),'config':vars(args),'epoch':counter['epoch'],
              'global_step':counter['updates'],'target_return':target},folder/f'epoch_{counter["epoch"]:02d}.pt')
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'epoch':counter['epoch'],'global_step':counter['updates'],'return':float(result),'target':target})+'\n')
  return result
 flags=[]
 for key in ['game','seed','context_length','batch_size','epochs','num_steps','num_buffers','model_type']:
  flags+=['--'+key,str(getattr(args,key))]
 flags+=['--data_dir_prefix',str(ROOT/'atari/dqn_replay')+'/']
 try:
  with patch.object(create_dataset,'create_dataset',dataset),patch.object(training,'TrainerConfig',Config),\
       patch.object(training.Trainer,'get_returns',evaluate),patch.object(GPT,'forward',forward),\
       patch.object(torch.optim.AdamW,'step',step),patch.object(sys,'argv',['run_dt_atari.py',*flags]):
    runpy.run_path(str(ROOT/'atari/run_dt_atari.py'),run_name='__main__')
 finally:logger.close()
if __name__=='__main__':main()
