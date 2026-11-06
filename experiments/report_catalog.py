'''生成供人工复盘的实验总表；不启动任务或访问GPU。'''
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from manage import catalog,completed_run,problems,state,now
jobs=catalog();rows=[]
for job in jobs.values():
 saved=state(job);issues=problems(job,jobs)
 status='complete' if completed_run(job) else saved['state'] if saved and saved.get('state') in ['running','failed','interrupted'] else 'blocked' if issues else 'ready'
 implementation='仅论文规格，未实现' if job['backend']=='external' else 'Atari入口已写，未端到端验证' if job['backend'].startswith('atari') else '附录机制重实现，GPU训练待验证' if job['backend']=='graph' else 'Gym原版训练/评估包装'
 rows.append({'id':job['id'],'group':job['group'],'state':status,'implementation':implementation,'reason':'; '.join(issues)})
out=ROOT/'experiments/reports';out.mkdir(exist_ok=True)
lines=['# 实验覆盖与准备状态','', '生成时间（UTC）：'+now(),'',
       'ready只表示静态准备检查通过，不等于已做真实短验证或复现成功；external为明确缺失的实现，不是可运行代码。',
       '', '| 状态 | 配置数 |','|---|---:|']
for status,count in sorted(Counter(x['state'] for x in rows).items()):lines.append(f'| {status} | {count} |')
lines+=['','| 实验ID | 组别 | 状态 | 实现范围 | 缺项 |','|---|---|---|---|---|']
for r in rows:lines.append('| '+' | '.join(r[k] for k in ['id','group','state','implementation','reason'])+' |')
(out/'CATALOG.md').write_text('\n'.join(lines)+'\n')
print('Generated',out/'CATALOG.md',dict(Counter(x['state'] for x in rows)))
