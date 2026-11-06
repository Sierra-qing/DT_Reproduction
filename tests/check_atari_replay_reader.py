'''验证原Dopamine replay落盘/加载/帧堆叠一致，不使用真实数据或训练网络。'''
from pathlib import Path
import tempfile,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'atari'))
from dopamine.replay_memory import circular_replay_buffer
from fixed_replay_buffer import FixedReplayBuffer
kwargs=dict(observation_shape=(84,84),stack_size=4,replay_capacity=64,batch_size=1,update_horizon=1,gamma=.99,observation_dtype=np.uint8)
source=circular_replay_buffer.OutOfGraphReplayBuffer(**kwargs)
for index in range(40):source.add(np.full((84,84),index,dtype=np.uint8),index%4,float(index),index%17==16)
with tempfile.TemporaryDirectory(prefix='atari-replay-fixture-',dir=str(ROOT/'experiments/smoke_runs')) as folder:
 source.save(folder,0)
 loaded=FixedReplayBuffer(data_dir=folder,replay_suffix=0,**kwargs)
 assert loaded._loaded_buffers
 expected=source.sample_transition_batch(batch_size=1,indices=[20])
 actual=loaded.sample_transition_batch(batch_size=1,indices=[20])
 for a,b in zip(actual,expected):np.testing.assert_equal(a,b)
 assert actual[0].shape==(1,84,84,4)
 files=sorted(p.name for p in Path(folder).glob('*.gz'))
 report={'result':'passed','scope':'synthetic original replay save/load equality, not real-data verification',
         'state_shape':list(actual[0].shape),'files':files,'python':sys.executable}
 (ROOT/'experiments/reports/atari_replay_reader_check.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
