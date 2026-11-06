'''读取官方PyPI元数据与Dopamine wheel目录，选择与旧接口匹配的独立环境。'''
import io,json,urllib.request,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent/'atari_package_probe.json'
packages={'tensorflow-cpu':'2.8.4','dopamine-rl':'3.0.1','atari-py':'0.2.9','opencv-python':'4.5.5.64','blosc':'1.10.6'}
records={}
for package,version in packages.items():
 with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30) as r:d=json.load(r)
 records[package]={'version':version,'python':d['info']['requires_python'],'requires':d['info']['requires_dist']}
 if package=='dopamine-rl':
  wheel=next(x for x in d['urls'] if x['filename'].endswith('.whl'))
  with urllib.request.urlopen(wheel['url'],timeout=30) as r:z=zipfile.ZipFile(io.BytesIO(r.read()))
  paths=[x for x in z.namelist() if 'circular_replay_buffer.py' in x]
  records[package]['replay_paths']=paths
  for path in paths:
   records[package]['replay_imports']=[line for line in z.read(path).decode().splitlines() if line.startswith(('import ','from '))]
OUT.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
