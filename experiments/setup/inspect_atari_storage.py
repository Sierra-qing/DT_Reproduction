'''只列官方公开GCS对象元数据，不下载replay。'''
import json,urllib.parse,urllib.request
from pathlib import Path
items=[];token=None
while True:
 query={'prefix':'dqn/Breakout/1/replay_logs/','maxResults':1000}
 if token:query['pageToken']=token
 url='https://storage.googleapis.com/storage/v1/b/atari-replay-datasets/o?'+urllib.parse.urlencode(query)
 with urllib.request.urlopen(url,timeout=40) as r:data=json.load(r)
 items.extend(data.get('items',[]));token=data.get('nextPageToken')
 if not token:break
selected=[]
for item in items:
 parts=item['name'].split('.')
 if len(parts)>=3 and parts[-1]=='gz' and parts[-2].isdigit() and 0<=int(parts[-2])<50:
  selected.append({k:item.get(k) for k in ['name','size','md5Hash','generation']})
record={'bucket':'atari-replay-datasets','prefix':'dqn/Breakout/1/replay_logs/','objects':selected,
        'bytes':sum(int(x['size']) for x in selected),'count':len(selected)}
out=Path(__file__).resolve().parent/'breakout_storage_manifest.json';out.write_text(json.dumps(record,indent=2)+'\n')
print('Breakout run1 buffers0..49:',record['count'],'objects;',round(record['bytes']/1024**3,2),'GiB')
