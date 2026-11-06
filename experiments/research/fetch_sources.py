'''只下载公开代码供审阅，不执行第三方代码，不安装依赖。固定commit并记录SHA256。'''
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
OUT=Path(__file__).resolve().parent/'sources_20261106';OUT.mkdir(exist_ok=True)
REPOS=['courtneyma/Improving-Credit-Assignment-with-Sequence-Models','srzer/LaMo-2023','tsmatz/decision-transformer','Jaron-U/decision-transformer-atari']

def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'DT-reproduction-source-audit'})
 return urllib.request.urlopen(req,timeout=30).read()
def fetch(repo):
 record={'repo':repo,'files':[]}
 try:
  meta=json.loads(get('https://api.github.com/repos/'+repo))
  branch=meta['default_branch'];commit=json.loads(get('https://api.github.com/repos/'+repo+'/commits/'+branch))['sha']
  record.update(commit=commit,default_branch=branch)
  tree=json.loads(get('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1'))['tree']
  paths=[x['path'] for x in tree if x['type']=='blob']
  (OUT/(repo.split('/')[0]+'_tree.json')).write_text(json.dumps(paths,indent=2))
  if repo.startswith('courtneyma'):selected=[p for p in paths if p.endswith('.ipynb') and ('keytodoor' in p.lower() or 'random_walk' in p.lower())]
  elif repo.startswith('srzer'):selected=[p for p in paths if 'data_generation_PPO' in p and p.endswith(('.py','.md','.sh'))]
  elif repo.startswith('tsmatz'):selected=[p for p in paths if p.endswith('.ipynb')]
  else:selected=[p for p in paths if p in ['run_dt_atari.py','data/create_dataset.py','environment.yml']]
  for path in selected:
   url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+urllib.parse.quote(path)
   try:
    blob=get(url);target=OUT/repo.split('/')[0]/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    record['files'].append({'path':path,'url':url,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)})
   except Exception as error:record['files'].append({'path':path,'error':str(error)})
 except Exception as error:record['error']=str(error)
 return record

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,REPOS))
(OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:print(json.dumps(r),flush=True)
