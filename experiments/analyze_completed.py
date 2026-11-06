'''只读分析第一次overnight的最终轨迹，不执行模型或仿真。'''
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/reports/completed_hopper';OUT.mkdir(parents=True,exist_ok=True)
records=[]
for run in sorted((ROOT/'gym/runs_observed').glob('*paper-overnight-20261105-134746')):
 seed=json.loads((run/'config.json').read_text())['seed']
 for target in [3600,1800]:
  rows=[]
  for file in sorted((run/'trajectories').glob(f'iteration_0010_target_{target}_episode_*.npz')):
   with np.load(file,allow_pickle=False) as d:
    reward=d['reward'];qpos=d['physics_qpos'];times=d['physics_time']
    duration=float(times[-1]-times[0]);dx=float(qpos[-1,0]-qpos[0,0])
    rows.append({'return':float(reward.sum()),'steps':len(reward),'height':float(qpos[-1,1]),
                 'angle':float(qpos[-1,2]),'dx':dx,'speed':dx/duration,'reward_per_step':float(reward.mean())})
  record={'seed':seed,'target':target,'episodes':len(rows),
          'mean_return':float(np.mean([x['return'] for x in rows])),
          'mean_steps':float(np.mean([x['steps'] for x in rows])),
          'mean_reward_per_step':float(np.mean([x['reward_per_step'] for x in rows])),
          'mean_forward_velocity':float(np.mean([x['speed'] for x in rows])),
          'height_failure_count':sum(x['height']<=.7 for x in rows),
          'angle_failure_count':sum(abs(x['angle'])>=.2 for x in rows),
          'return_quantiles_10_50_90':np.percentile([x['return'] for x in rows],[10,50,90]).tolist()}
  records.append(record)
(OUT/'trajectory_summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
lines=['# 已完成Hopper实验的行为复盘','',
       '数据来自第一次overnight的100000更新最终模型，读取已记录物理轨迹，不新评估、不改训练。',
       '', '| seed | RTG目标 | 回报均值 | 平均步数 | 每步奖励均值 | 前进速度均值 | 高度终止计数 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in records:
 lines.append(f"| {r['seed']} | {r['target']} | {r['mean_return']:.2f} | {r['mean_steps']:.2f} | {r['mean_reward_per_step']:.3f} | {r['mean_forward_velocity']:.3f} | {r['height_failure_count']}/{r['episodes']} |")
lines+=['','观察：先比较每步奖励与持续步数，再判断较低累计回报是否主要来自提前结束。终止时满足高度/角度阈值是可观测事实，不等于已定位哪个动作导致失败。',
        '', '两个RTG目标使用不同评估回合初始条件，不能将均值差直接当成受控的目标条件因果效应；进一步扫描应固定评估seed并记录相同初始条件。',
        '', '本分析不证明过拟合，也不把较低loss当作控制性能保证。完整正式统计见 reproduction/results/overnight_20261105/REPORT.md。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
