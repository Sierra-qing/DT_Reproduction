'''
将论文实验组织成显式配置；缺失实现/数据不生成伪结果。
运行本脚本只写配置，不启动训练。所有配方以论文明确值优先。
'''
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/configs'
PAPER='https://arxiv.org/html/2106.01345v2'
CORE={'mode':'normal','K':20,'pct_traj':1.,'batch_size':64,'model_type':'dt',
      'embed_dim':128,'n_layer':3,'n_head':1,'activation_function':'relu','dropout':.1,
      'learning_rate':1e-4,'weight_decay':1e-4,'warmup_steps':100000,
      'num_eval_episodes':100,'max_iters':10,'num_steps_per_iter':10000,'device':'cuda'}
TARGETS={'hopper':[3600,1800],'halfcheetah':[6000,12000],'walker2d':[5000,2500],'reacher2d':[50,76]}
OLD='paper-overnight-20261105-134746'

def generate():
    jobs=[]
    def add(name,backend,group,**extra):
        jobs.append({'id':name,'backend':backend,'group':group,'paper':PAPER,**extra})
    for env in TARGETS:
      for dataset in ['medium','medium-replay','medium-expert']:
       for seed in [0,1,2]:
        params={**CORE,'env':env,'dataset':dataset,'seed':seed,'eval_targets':TARGETS[env]}
        if env=='reacher2d':params['K']=5
        name=f'dt-{env}-{dataset}-s{seed}'
        completed=[]
        if env=='hopper' and dataset=='medium':
            completed=[str(p.relative_to(ROOT)) for p in (ROOT/'gym/runs_observed').glob(f'hopper-medium-dt-seed{seed}-*{OLD}')]
        add(name,'gym','gym-dt',params=params,completed_runs=completed,
            source='Paper Table9 + official sampler/trainer; extra evaluation target retained')
        add(f'bc-{env}-{dataset}-p100-s{seed}','gym','gym-bc',
            params={**params,'model_type':'bc','K':20,'embed_dim':256,'max_iters':5,'num_steps_per_iter':5000},
            source='Paper A.2.2/Table10: 25000 updates; unspecified K follows repository default20')
        if dataset in ['medium','medium-replay']:
          for pct in [10,25,40]:
            add(f'bc-{env}-{dataset}-p{pct}-s{seed}','gym','gym-percentile-bc',
                params={**params,'model_type':'bc','K':20,'embed_dim':256,'max_iters':5,'num_steps_per_iter':5000,'pct_traj':pct/100.},
                source='Paper Table3; top percentile is timestep budget ranked by trajectory return; official selector')
        if env=='hopper':
          add(f'delayed-{env}-{dataset}-s{seed}','gym','gym-delayed',
              params={**params,'mode':'delayed'},source='Paper delayed-return Hopper comparison; same100kDTupdates')
        if dataset=='medium-replay':
          add(f'sweep-{env}-{dataset}-s{seed}','sweep','return-sweep',dependency=name,
              targets={'hopper':[0,600,1200,1800,2400,3000,3600],
                       'halfcheetah':[0,1000,2000,3000,4000,5000,6000],
                       'walker2d':[0,1000,2000,3000,4000,5000],
                       'reacher2d':[0,10,20,30,40,50]}[env],episodes=100,seed=seed,
              source='Paper Figure4 medium-replay; grid chosen for reproduction, exact original grid not specified')
    for seed in [0,1,2]:
      add(f'diagnostic-sweep-hopper-medium-s{seed}','sweep','diagnostic',
          dependency=f'dt-hopper-medium-s{seed}',targets=[0,600,1200,1800,2400,3000,3600],episodes=100,seed=seed,
          source='Diagnostic extension on existing medium model; NOT a direct reproduction of Figure4')
    for game in ['Breakout','Qbert','Pong','Seaquest']:
      for seed in [123,231,312]:
       for pct in [None,100,40,25,10]:
        bc=pct is not None
        add(f'atari-{game.lower()}-'+(f'bc-p{pct}' if bc else 'dt')+f'-s{seed}',
            'atari','atari-bc' if bc else 'atari-dt',
            params={'game':game,'seed':seed,'context_length':50 if game=='Pong' else 30,
                    'batch_size':512 if game=='Pong' else 128,'epochs':5,'num_steps':500000,
                    'num_buffers':50,'model_type':'naive' if bc else 'reward_conditioned',
                    'pct':100 if pct is None else pct},
            source='Paper Table8 and official run.sh; original Atari seeds retained')
       add(f'atari-{game.lower()}-k1-s{seed}','atari','atari-context',
           params={'game':game,'seed':seed,'context_length':1,'batch_size':512 if game=='Pong' else 128,
                   'epochs':5,'num_steps':500000,'num_buffers':50,'model_type':'reward_conditioned','pct':100},
           source='Paper context ablation; same Atari training settings except K=1')
       add(f'atari-sweep-{game.lower()}-s{seed}','atari-sweep','atari-return-sweep',
           dependency=f'atari-{game.lower()}-dt-s{seed}',seed=seed,
           targets={'Breakout':[0,30,60,90,120],'Qbert':[0,500,1000,1500,2000,2500],
                    'Pong':[-20,-10,0,10,20],'Seaquest':[0,300,600,900,1200,1450]}[game],
           source='Paper Figure4 Atari; explicit reproduction grid, original exact grid unspecified')
    for size in [1000,10000]:
      for method in ['dt','bc','percentile-bc','cql','random']:
       add(f'keydoor-{size}-{method}','external','keydoor',
           blocked_reason='Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction',
           specification={'random_trajectories':size,'three_phases':['key','distractor','door'],
                          'reward':'binary success, requires key picked up in phase1',
                          'context':'full episode; not fixed Hopper K20','method':method})
    add('keydoor-return-prediction-attention','external','keydoor-analysis',
        blocked_reason='Requires paper-specific Key-to-Door implementation and return-prediction model, absent from pinned repository',
        specification={'outputs':['actions','returns'],'analysis':['predicted success probability','attention to key/door events']})
    add('graph-shortest-path','external','graph',
        blocked_reason='Appendix specifies environment but pinned repository lacks experiment implementation and complete training hyperparameters; avoid inventing official settings',
        specification={'nodes':20,'edge_sparsity':.1,'random_walks':1000,'horizon':10,
                       'reward':'0 at goal, -1 otherwise','tokens':['rtg','state','action'],
                       'generation':'learned return likelihood times prior^10, then action'})
    for seed in [0,1,2]:
      add(f'graph-reimplementation-s{seed}','graph','graph-reimplementation',
          params={'seed':seed,'updates':10000,'batch_size':64,'eval_interval':1000},
          source='Paper A.3 mechanism reconstruction; unreported training choices explicit, not exact author-code reproduction')
    catalog={'schema_version':1,'paper':PAPER,'official_commit':'e2d82e68f330c00f763507b3b01d774740bee53f',
             'rules':['paper explicit values override repository','missing code is blocked, not claimed implemented',
                      'no CPU fallback','do not rerun completed jobs automatically','no implicit SDK or environment upgrades'],
             'jobs':jobs}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    plans={
      'hopper-ready':[j['id'] for j in jobs if j['backend']=='gym' and j['params']['env']=='hopper' and j['params']['dataset']=='medium' and j['group']!='gym-dt'],
      'hopper-next':[j['id'] for j in jobs if j['backend']=='gym' and j['params']['env']=='hopper' and not j.get('completed_runs')],
      'gym-all':[j['id'] for j in jobs if j['backend']=='gym' and not j.get('completed_runs')],
      'return-sweeps':[j['id'] for j in jobs if j['backend']=='sweep'],
      'atari-all':[j['id'] for j in jobs if j['backend']=='atari'],
      'atari-sweeps':[j['id'] for j in jobs if j['backend']=='atari-sweep'],
      'graph-reimplementation':[j['id'] for j in jobs if j['backend']=='graph'],
      'locomotion-next':[j['id'] for j in jobs if j['backend']=='gym' and j['params']['env'] in ['halfcheetah','walker2d']],
    }
    # 先完成普通BC，再比较百分位BC，最后跑delayed，方便按任务复盘。
    plans['hopper-ready'].sort(key=lambda name:(0 if '-p100-' in name else 2 if name.startswith('delayed') else 1,name))
    dataset_order={'medium':0,'medium-replay':1,'medium-expert':2}
    by_id={j['id']:j for j in jobs}
    plans['locomotion-next'].sort(key=lambda name:(0 if by_id[name]['group']=='gym-dt' else 1 if by_id[name]['group']=='gym-bc' else 2,
          dataset_order[by_id[name]['params']['dataset']],by_id[name]['params']['seed'],by_id[name]['params']['env']))
    ready=set(plans['hopper-ready'])
    plans['hopper-remainder']=[x for x in plans['hopper-next'] if x not in ready]
    plans['hopper-remainder'].sort(key=lambda name:(0 if name.startswith('dt-') else 1 if '-p100-' in name else 2 if name.startswith('bc-') else 3,
                                                  0 if 'medium-replay' in name else 1,name))
    for name,ids in plans.items():
       (OUT/(name+'.json')).write_text(json.dumps({'name':name,'job_ids':ids},indent=2)+'\n')
    print('catalog jobs:',len(jobs),'plans:',{k:len(v) for k,v in plans.items()})
    return catalog

if __name__=='__main__':generate()
