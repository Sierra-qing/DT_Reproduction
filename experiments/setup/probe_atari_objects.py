'''检查官方存储桶的单个公开对象是否可读；不下载数据内容。'''
import concurrent.futures,json,urllib.request,urllib.error,urllib.parse
from pathlib import Path
names=['$store$_observation_ckpt.0.gz','$store$_action_ckpt.0.gz','$store$_reward_ckpt.0.gz',
       '$store$_terminal_ckpt.0.gz','add_count_ckpt.0.gz','invalid_range_ckpt.0.gz']
def check(name):
 url='https://storage.googleapis.com/atari-replay-datasets/dqn/Breakout/1/replay_logs/'+urllib.parse.quote(name)
 try:
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
   return {'name':name,'status':r.status,'bytes':r.headers.get('Content-Length'),'hash':r.headers.get('x-goog-hash')}
 except urllib.error.HTTPError as error:return {'name':name,'status':error.code,'reason':str(error)}
 except Exception as error:return {'name':name,'error':str(error)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(check,names))
Path(__file__).with_name('breakout_object_probe.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
