'''从已落盘结果生成统计与图，不执行训练或评估，不选最高分模型。'''
from pathlib import Path
import json,csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
runs=sorted((ROOT/'gym/runs_observed').glob('*paper-overnight-20261105-134746'))
records=[]; curves=[]
for run in runs:
 config=json.loads((run/'config.json').read_text())
 metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
 final=metrics[-1]
 assert len(metrics)==10 and final['global_step']==100000
 rec={'seed':config['seed'],'run':str(run.relative_to(ROOT)),
      'raw_return_3600':final['evaluation/target_3600_return_mean'],
      'score_3600':final['evaluation/target_3600_normalized_score_mean'],
      'length_3600':final['evaluation/target_3600_length_mean'],
      'score_1800':final['evaluation/target_1800_normalized_score_mean'],
      'final_iteration_loss':final['training/train_loss_mean'],
      'checkpoints':len(list((run/'checkpoints').glob('iteration_*.pt'))),
      'trajectories':len(list((run/'trajectories').glob('*.npz'))),
      'videos':len(list((run/'videos').glob('*.gif')))}
 records.append(rec)
 curves.append([m['evaluation/target_3600_normalized_score_mean'] for m in metrics])
values=np.array([r['score_3600'] for r in records]);curves=np.array(curves)
summary={'selection':'last checkpoint at 100000 updates, target RTG=3600; no best-checkpoint selection',
         'seeds':records,'mean_score':float(values.mean()),'population_std_ddof0':float(values.std()),
         'sample_std_ddof1':float(values.std(ddof=1)),'paper_score':67.6,'gap_points':float(67.6-values.mean()),
         'paper_url':'https://arxiv.org/html/2106.01345v2',
         'limitations':['paper dataset identity and checkpoint aggregation protocol not fully established',
                        'CUDA build and experiment tracking differ from original dependency declaration']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (OUT/'per_seed.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
steps=np.arange(1,11)*10000
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for seed,curve in enumerate(curves):axes[0].plot(steps/1000,curve,marker='o',label=f'Seed {seed}')
axes[0].plot(steps/1000,curves.mean(axis=0),color='black',linewidth=2,label='Across-seed mean')
axes[0].axhline(67.6,color='firebrick',linestyle='--',label='Paper: 67.6')
axes[0].set(xlabel='Training updates (thousands)',ylabel='D4RL normalized score',title='Hopper-medium: target RTG = 3600')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
axes[1].bar(['Seed 0','Seed 1','Seed 2','Mean'],list(values)+[values.mean()],color=['C0','C1','C2','gray'])
axes[1].errorbar(3,values.mean(),yerr=values.std(),fmt='none',color='black',capsize=5)
axes[1].axhline(67.6,color='firebrick',linestyle='--');axes[1].set(ylabel='D4RL normalized score',title='Final checkpoints at 100k updates',ylim=(0,80))
for i,v in enumerate(list(values)+[values.mean()]):axes[1].text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
fig.savefig(OUT/'learning_curves.png',dpi=170)
print(json.dumps({k:v for k,v in summary.items() if k!='seeds'},indent=2))
