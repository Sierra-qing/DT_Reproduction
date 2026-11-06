'''
按DT附录A.3机制重实现图任务。不是原作者未公开代码的逐字复刻。
明示实现选择：state/RTG/action顺序以便R条件包含当前状态；分类RTG为剩余代价d=-R。
论文未报告的模型和优化参数借用Gym默认并写入配置。最短路只用于评估。
'''
import argparse,json,random,sys
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

class Graph:
 def __init__(self,edges,goal):self.edges=np.asarray(edges,dtype=bool);self.goal=int(goal)
 def step(self,state,action):
  if state==self.goal:return self.goal,0.,True
  next_state=int(action) if self.edges[state,action] else int(state)
  return next_state,-1.,next_state==self.goal

def distances(graph):
 '''反向BFS仅作事后指标；策略推断不调用本函数。'''
 d=np.full(len(graph.edges),np.inf);d[graph.goal]=0;queue=deque([graph.goal])
 while queue:
  node=queue.popleft()
  for predecessor in np.flatnonzero(graph.edges[:,node]):
   if not np.isfinite(d[predecessor]):d[predecessor]=d[node]+1;queue.append(int(predecessor))
 return d

def random_walks(graph,count,horizon,rng):
 paths=[]
 for _ in range(count):
  state=int(rng.randint(len(graph.edges)));states=[];actions=[];rewards=[]
  for t in range(horizon):
   action=int(rng.randint(len(graph.edges)));states.append(state);actions.append(action)
   state,reward,done=graph.step(state,action);rewards.append(reward)
   if done:break
  rtg=np.cumsum(np.asarray(rewards)[::-1])[::-1]
  paths.append({'states':states,'actions':actions,'rewards':rewards,'costs':(-rtg).astype(int).tolist()})
 return paths

def return_posterior(logits,horizon,exponent=10):
 '''用正代价d表示负回报，先验T+1-d偏向短路径；在推断时合并概率。'''
 import torch
 costs=torch.arange(horizon+1,device=logits.device,dtype=logits.dtype)
 log_prior=(horizon+1-costs).log()
 return torch.softmax(torch.log_softmax(logits,dim=-1)+exponent*log_prior,dim=-1)

def make_model(nodes,horizon,hidden,layers,heads,dropout):
 import torch
 from torch import nn
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.nodes=nodes;self.horizon=horizon
   self.embedding=nn.Embedding(2*nodes+horizon+1,hidden)
   self.timestep=nn.Embedding(horizon+1,hidden);self.norm=nn.LayerNorm(hidden)
   self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,heads,4*hidden,dropout,'relu'),layers)
   self.return_head=nn.Linear(hidden,horizon+1);self.action_head=nn.Linear(hidden,nodes)
  def encode(self,tokens,padding=None):
   positions=torch.arange(tokens.shape[1],device=tokens.device)//3
   values=self.norm(self.embedding(tokens)+self.timestep(positions)[None,:,:])
   mask=torch.triu(torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool),diagonal=1)
   return self.transformer(values.transpose(0,1),mask=mask,src_key_padding_mask=padding).transpose(0,1)
 return Model()

def batch(paths,size,nodes,horizon,rng,device):
 import torch
 tokens=np.zeros((size,3*horizon),dtype=np.int64);valid=np.zeros((size,horizon),bool)
 cost_targets=np.zeros((size,horizon),np.int64);action_targets=np.zeros((size,horizon),np.int64)
 for i in range(size):
  path=paths[int(rng.randint(len(paths)))];n=len(path['states']);valid[i,:n]=True
  tokens[i,:3*n:3]=path['states'];tokens[i,1:3*n:3]=nodes+np.asarray(path['costs'])
  tokens[i,2:3*n:3]=nodes+horizon+1+np.asarray(path['actions'])
  cost_targets[i,:n]=path['costs'];action_targets[i,:n]=path['actions']
 return tuple(torch.as_tensor(x,device=device) for x in [tokens,valid,cost_targets,action_targets])

def rollout(model,graph,start,horizon,device):
 '''策略仅接收已知当前状态和历史；没有BFS距离或最优RTG参数。'''
 import torch
 model.eval();state=int(start);prefix=[];trace=[]
 if state==graph.goal:return {'start':state,'success':True,'steps':0,'trace':[]}
 with torch.no_grad():
  for t in range(horizon):
   prefix.append(state)
   states=model.encode(torch.tensor([prefix],device=device))
   posterior=return_posterior(model.return_head(states[0,-1]),horizon)
   cost=int(posterior.argmax());prefix.append(model.nodes+cost)
   hidden=model.encode(torch.tensor([prefix],device=device))
   action=int(model.action_head(hidden[0,-1]).argmax());prefix.append(model.nodes+horizon+1+action)
   next_state,reward,done=graph.step(state,action)
   trace.append({'state':state,'requested_rtg':-cost,'action':action,'reward':reward,'next_state':next_state})
   state=next_state
   if done:break
 return {'start':int(start),'success':state==graph.goal,'steps':len(trace),'trace':trace}

def evaluate(model,graph,horizon,device,out,step):
 shortest=distances(graph);results=[rollout(model,graph,start,horizon,device) for start in range(len(graph.edges)) if start!=graph.goal]
 for result in results:
  best=shortest[result['start']]
  result['shortest_steps']=int(best) if np.isfinite(best) else None
  result['optimal']=bool(result['success'] and result['steps']==best)
 eligible=[r for r in results if r['shortest_steps'] is not None and r['shortest_steps']<=horizon]
 summary={'update':step,'success_all_starts':float(np.mean([r['success'] for r in results])),
          'optimal_all_starts':float(np.mean([r['optimal'] for r in results])),
          'reachable_within_horizon':len(eligible),'total_starts':len(results),
          'success_reachable':float(np.mean([r['success'] for r in eligible])) if eligible else None}
 (out/f'evaluation_{step:06d}.json').write_text(json.dumps({'summary':summary,'rollouts':results},indent=2)+'\n')
 return summary,results

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--seed',type=int,default=0);p.add_argument('--updates',type=int,default=10000)
 p.add_argument('--batch_size',type=int,default=64);p.add_argument('--eval_interval',type=int,default=1000)
 p.add_argument('--output_dir',type=Path,default=ROOT/'experiments/runs')
 p.add_argument('--swanlab_mode',choices=['online','offline'],default='online')
 args=p.parse_args()
 if min(args.updates,args.batch_size,args.eval_interval)<1:p.error('更新数、batch和间隔必须为正')
 import torch
 from torch.nn import functional as F
 if not torch.cuda.is_available():raise SystemExit('CUDA unavailable; no CPU fallback')
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 rng=np.random.RandomState(args.seed);nodes=20;horizon=10
 edges=rng.rand(nodes,nodes)<.1;np.fill_diagonal(edges,False);graph=Graph(edges,goal=nodes-1)
 paths=random_walks(graph,1000,horizon,rng)
 out=args.output_dir/('graph-reimplementation-s'+str(args.seed)+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
 out.mkdir(parents=True)
 config={**vars(args),'output_dir':str(args.output_dir),'nodes':nodes,'horizon':horizon,'trajectories':1000,'edge_probability':.1,
         'hidden':128,'layers':3,'heads':1,'dropout':.1,'lr':1e-4,'warmup':100000,
         'classification':'paper-based reimplementation, NOT recovered author protocol',
         'implementation_choices':['directed Bernoulli graph, no self edges, fixed goal19',
           'uniform node actions and starts, stop at goal or T10',
           'state/return/action tokens support p(R_t | current state, history)',
           'RTG categorical cost d=-R; inference prior proportional to T+1-d, exponent10',
           'greedy posterior RTG and greedy action',
           'unreported model/optimizer settings borrowed from Gym; updates10000 explicit engineering choice']}
 (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 np.save(out/'graph_edges.npy',edges);(out/'dataset.json').write_text(json.dumps(paths))
 device='cuda';model=make_model(nodes,horizon,128,3,1,.1).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/100000,1.))
 sys.path.insert(0,str(ROOT/'gym'));from swanlab_observer import SwanLabObserver
 from types import SimpleNamespace
 logargs=SimpleNamespace(**{**vars(args),'job_id':'Graph-DT-Reimpl-s'+str(args.seed),
                            'swanlab_project':'decision-transformer-reimplementations','swanlab_workspace':None})
 logger=SwanLabObserver(logargs,out)
 try:
  for step in range(1,args.updates+1):
   model.train();tokens,valid,costs,actions=batch(paths,args.batch_size,nodes,horizon,rng,device)
   hidden=model.encode(tokens,~valid.repeat_interleave(3,dim=1))
   r_loss=F.cross_entropy(model.return_head(hidden[:,0::3])[valid],costs[valid])
   a_loss=F.cross_entropy(model.action_head(hidden[:,1::3])[valid],actions[valid]);loss=r_loss+a_loss
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.25);optimizer.step();scheduler.step()
   if step==1 or step%100==0:
    for name,value in [('action_ce',a_loss.item()),('return_ce',r_loss.item())]:logger.scalar('train/'+name,value,step,step)
    print(f'update {step}/{args.updates} action_ce={a_loss.item():.4f} return_ce={r_loss.item():.4f}',flush=True)
   if step%args.eval_interval==0 or step==args.updates:
    summary,results=evaluate(model,graph,horizon,device,out,step)
    for name,value in summary.items():
     if value is not None:logger.scalar('eval/'+name,value,step,step)
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'config':config,'update':step},out/'checkpoint_latest.pt')
    print(summary,flush=True)
 finally:logger.close()
 print('Graph result:',out,flush=True)
if __name__=='__main__':main()
