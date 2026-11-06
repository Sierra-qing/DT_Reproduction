'''固定最终模型，只改变评估RTG；每个目标使用相同评估seed，不更新参数。'''
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import subprocess,sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'gym')]
from manage import catalog,completed_run
from swanlab_observer import SwanLabObserver

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job_id',required=True)
 parser.add_argument('--swanlab_mode',choices=['online','offline'],default='online');args=parser.parse_args()
 jobs=catalog();job=jobs[args.job_id];source=completed_run(jobs[job['dependency']])
 if source is None:raise SystemExit('先完成依赖训练：'+job['dependency'])
 checkpoint=source/'checkpoint_latest.pt'
 out=ROOT/'experiments/runs'/(args.job_id+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
 out.mkdir(parents=True)
 config=SimpleNamespace(**{**vars(args),'swanlab_project':'decision-transformer-reproduction','swanlab_workspace':None,
                           'source_checkpoint':str(checkpoint),'source_note':job['source']})
 logger=SwanLabObserver(config,out);rows=[]
 try:
  for target in job['targets']:
   folder=out/f'target_{target}'
   subprocess.run([sys.executable,'-u',str(ROOT/'gym/evaluate_checkpoint.py'),str(checkpoint),
                   '--target',str(target),'--episodes',str(job['episodes']),'--seed',str(job['seed']),
                   '--output_dir',str(folder)],check=True,cwd=ROOT/'gym')
   summary=json.loads((folder/'summary.json').read_text());rows.append(summary)
   for key in ['return_mean','return_std','d4rl_score_mean','d4rl_score_std','length_mean']:
    if key in summary:logger.scalar('sweep/'+key,summary[key],target,target)
   for gif in (folder/'videos').glob('*.gif'):logger.media('sweep/first_episode',gif,target,video=True)
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(6,5),constrained_layout=True)
  ax.errorbar([r['target_return'] for r in rows],[r['return_mean'] for r in rows],
              yerr=[r['return_std'] for r in rows],marker='o',label='Mean +/- episode std')
  ax.plot(job['targets'],job['targets'],'--',color='gray',label='Requested = achieved')
  ax.set(xlabel='Requested initial RTG (raw)',ylabel='Actual episode return',title=args.job_id);ax.legend();ax.grid(alpha=.2)
  fig.savefig(out/'return_sweep.png',dpi=150);logger.media('sweep/requested_vs_achieved',out/'return_sweep.png',0)
  with (out/'summary.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  (out/'summary.json').write_text(json.dumps({'job':job,'source_checkpoint':str(checkpoint),'targets':rows},ensure_ascii=False,indent=2))
 finally:logger.close()
 print('Sweep result:',out,flush=True)
if __name__=='__main__':main()
