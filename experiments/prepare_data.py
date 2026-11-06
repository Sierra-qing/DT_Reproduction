'''
下载官方 D4RL v2 HDF5，并严格按官方下载脚本分割完整轨迹。
只补缺失文件，不覆盖已跑实验的数据。官方HTTP地址先尝试同主机HTTPS。
'''
import argparse,collections,hashlib,importlib.util,json,pickle,runpy,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def convert(dataset):
 import numpy as np
 keys=['observations','next_observations','actions','rewards','terminals']
 n=len(dataset['rewards']);pending=collections.defaultdict(list);paths=[];episode_step=0
 for i in range(n):
  terminal=bool(dataset['terminals'][i])
  timeout=bool(dataset['timeouts'][i]) if 'timeouts' in dataset else episode_step==999
  for key in keys:pending[key].append(dataset[key][i])
  if terminal or timeout:
   episode_step=0;paths.append({k:np.asarray(v) for k,v in pending.items()});pending=collections.defaultdict(list)
  episode_step+=1
 return paths

def prepare(env,dataset):
 import h5py,numpy as np
 name=f'{env}-{dataset}-v2';folder=ROOT/'gym/data';out=folder/(name+'.pkl')
 if out.exists():print('EXISTS, unchanged:',out,flush=True);return
 spec=importlib.util.find_spec('d4rl')
 if spec is None:raise RuntimeError('Missing installed D4RL metadata')
 urls=runpy.run_path(str(Path(spec.origin).with_name('infos.py')))['DATASET_URLS']
 url=urls[name];raw=folder/(name+'.hdf5');used_url=None
 if not raw.exists():
  partial=raw.with_suffix('.download')
  attempts=[url.replace('http://','https://',1),url]
  failures=[]
  for candidate in attempts:
   try:
    print('DOWNLOAD:',candidate,flush=True)
    request=urllib.request.Request(candidate,headers={'User-Agent':'DecisionTransformerReproduction/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response,partial.open('wb') as handle:
     while True:
      block=response.read(1024*1024)
      if not block:break
      handle.write(block)
    with h5py.File(partial,'r') as h:assert 'rewards' in h and 'observations' in h
    partial.replace(raw);used_url=candidate;break
   except Exception as error:failures.append(candidate+': '+str(error))
  else:raise RuntimeError('Official download failed; no unverified mirror used: '+'; '.join(failures))
 with h5py.File(raw,'r') as h:
  data={key:h[key][:] for key in ['observations','next_observations','actions','rewards','terminals','timeouts'] if key in h}
 paths=convert(data)
 if not paths:raise ValueError('No complete trajectories')
 temp=out.with_suffix('.tmp')
 with temp.open('wb') as f:pickle.dump(paths,f)
 temp.replace(out)
 manifest={'dataset':name,'official_url':url,'download_url':used_url,'hdf5_sha256':digest(raw),
           'pickle_sha256':digest(out),'input_steps':len(data['rewards']),
           'saved_steps':sum(len(p['rewards']) for p in paths),'trajectories':len(paths),
           'return_mean':float(np.mean([p['rewards'].sum() for p in paths])),
           'conversion':'official terminal/timeout split; incomplete trailing trajectory omitted',
           'upstream_digest':'not published/verified; hashes record local provenance only'}
 (folder/(name+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
 print('READY:',json.dumps(manifest),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--envs',nargs='+',choices=['hopper','halfcheetah','walker2d'],default=['hopper'])
 p.add_argument('--datasets',nargs='+',choices=['medium','medium-replay','medium-expert'],default=['medium-replay','medium-expert'])
 args=p.parse_args();failed=[]
 for env in args.envs:
  for dataset in args.datasets:
   try:prepare(env,dataset)
   except Exception as error:failed.append({'env':env,'dataset':dataset,'error':str(error)});print('BLOCKED:',failed[-1],flush=True)
 if failed:raise SystemExit(1)
if __name__=='__main__':main()
