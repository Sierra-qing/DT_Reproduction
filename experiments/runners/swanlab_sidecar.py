'''
让官方Atari Python3.7训练进程通过独立Python3.8日志进程使用SwanLab。
只传标量和媒体路径，不共享模型/梯度，不改变策略设备或训练计算。
'''
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
LOGGING_PYTHON='/home/ubuntu/miniconda3/envs/decision-transformer-gym/bin/python'

class SwanLabSidecar:
 def __init__(self,args,run_dir):
  self.folder=Path(run_dir);self.ready=self.folder/'swanlab_sidecar_ready.json'
  config=self.folder/'swanlab_sidecar_config.json'
  config.write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+'\n')
  runtime=ROOT/'experiments/runtime.json'
  logging_python=json.loads(runtime.read_text()).get('atari_logging_python',LOGGING_PYTHON) if runtime.exists() else LOGGING_PYTHON
  self.process=subprocess.Popen([logging_python,'-u',str(Path(__file__).resolve()),'--worker',str(config),str(self.folder)],
                                stdin=subprocess.PIPE,universal_newlines=True,bufsize=1)
  started=time.monotonic()
  while not self.ready.exists():
   if self.process.poll() is not None:raise RuntimeError('SwanLab日志进程初始化失败，见控制台')
   if time.monotonic()-started>120:
    self.process.terminate();self.process.wait();raise RuntimeError('SwanLab日志进程初始化超时')
   time.sleep(.1)
 def send(self,data):
  if self.process.poll() is not None:raise RuntimeError('SwanLab日志进程已退出')
  self.process.stdin.write(json.dumps(data)+'\n');self.process.stdin.flush()
 def scalar(self,tag,value,update,axis_step):self.send({'op':'scalar','tag':tag,'value':float(value),'update':int(update),'step':int(axis_step)})
 def media(self,tag,path,update,video=False):self.send({'op':'media','tag':tag,'path':str(Path(path).resolve()),'update':int(update),'video':video})
 def close(self):
  if self.process.poll() is None:
   self.send({'op':'finish'});self.process.stdin.close()
  code=self.process.wait(timeout=120)
  if code:raise RuntimeError('SwanLab日志进程失败，退出码='+str(code))

def worker(config,folder):
 sys.path.insert(0,str(ROOT/'gym'))
 from swanlab_observer import SwanLabObserver
 logger=SwanLabObserver(SimpleNamespace(**json.loads(config.read_text())),folder)
 (folder/'swanlab_sidecar_ready.json').write_text(json.dumps({'pid':os.getpid(),'python':sys.executable})+'\n')
 try:
  for line in sys.stdin:
   record=json.loads(line)
   if record['op']=='finish':break
   if record['op']=='scalar':logger.scalar(record['tag'],record['value'],record['update'],record['step'])
   elif record['op']=='media':logger.media(record['tag'],Path(record['path']),record['update'],record['video'])
   else:raise ValueError('Unknown log operation')
 finally:logger.close()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True);parser.add_argument('folder',type=Path)
 args=parser.parse_args();worker(args.worker,args.folder)
