'''
真实 Hopper + CUDA 小规模对照：逐步记录不得改变模型权重、动作、环境状态和回报。
并核查 RTG 对齐、物理状态、每步指标、训练输入还原、轨迹 CSV/PNG。
'''
import contextlib
import io
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gym'))
import experiment as official
import experiment_observed as observed
from review_experiment import load_episode, export_episode, reconstruct_batch


class CaptureTrainer(official.SequenceTrainer):
    current=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        CaptureTrainer.current=self


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


if __name__=='__main__':
    records={'baseline':[],'observed':[]}
    label='baseline'
    original_make=official.gym.make
    created=[]
    def factory(*args,**kwargs):
        env=original_make(*args,**kwargs)
        env.seed(73);env.action_space.seed(73)
        reset_original,step_original=env.reset,env.step
        previous=[None]
        def reset(*values,**options):
            result=reset_original(*values,**options)
            records[label].append([])
            previous[0]=result.copy()
            return result
        def step(action):
            result=step_original(action)
            records[label][-1].append((previous[0].copy(),action.copy(),float(result[1]),result[0].copy(),bool(result[2])))
            previous[0]=result[0].copy()
            return result
        env.reset,env.step=reset,step
        created.append(env)
        return env

    args=observed.build_parser().parse_args(['--seed','73','--device','cuda','--embed_dim','16','--n_layer','1',
        '--batch_size','4','--max_iters','2','--num_steps_per_iter','2','--num_eval_episodes','1',
        '--log_interval','2','--eval_log_interval','1','--batch_snapshot_interval','1',
        '--output_dir',str(ROOT/'reproduction/trajectory_validation_runs'),'--run_name','real-hop-test-not-paper-result'])
    config={key:getattr(args,key) for key in observed.CORE_DEFAULTS}
    with patch.object(official.gym,'make',factory),patch.object(official,'SequenceTrainer',CaptureTrainer):
        seed(73)
        with contextlib.redirect_stdout(io.StringIO()):
            official.experiment('test',dict(config))
        baseline={key:value.detach().cpu().clone() for key,value in CaptureTrainer.current.model.state_dict().items()}
        label='observed'
        seed(73)
        with contextlib.redirect_stdout(io.StringIO()):
            run=observed.run_observed(args)
    for env in created: env.close()
    checkpoint=torch.load(run/'checkpoint_latest.pt',map_location='cpu')
    for key,value in baseline.items():
        assert torch.equal(value,checkpoint['model_state_dict'][key]),key
    assert len(records['baseline'])==len(records['observed'])==4
    for baseline_episode,observed_episode in zip(records['baseline'],records['observed']):
        assert len(baseline_episode)==len(observed_episode)
        for left,right in zip(baseline_episode,observed_episode):
            for a,b in zip(left,right):
                assert np.array_equal(a,b)
    print('PASS: real Hopper states/actions/rewards and CUDA parameter updates unchanged',flush=True)
    episodes=sorted((run/'trajectories').glob('*.npz'))
    assert len(episodes)==4
    for path in episodes:
        arrays,metadata=load_episode(path)
        position=(metadata['iteration']-1)*2 + (0 if metadata['target_return']==3600 else 1)
        expected=records['observed'][position]
        count=len(expected)
        assert metadata['complete'] and count==metadata['steps']
        assert (run/metadata['checkpoint']).is_file()
        assert np.array_equal(arrays['environment_timestep'],np.arange(count))
        assert np.array_equal(arrays['action'],arrays['action_predicted'])
        for name,column in [('observation',0),('action',1),('reward',2),('next_observation',3),('done',4)]:
            assert np.array_equal(arrays[name],np.array([row[column] for row in expected])),name
        assert np.allclose(arrays['target_rtg_raw'],arrays['target_rtg_scaled']*metadata['rtg_scale'])
        assert np.allclose(arrays['realized_rtg_raw'],np.cumsum(arrays['reward'][::-1])[::-1])
        assert np.allclose(arrays['target_rtg_scaled'][1:],
            arrays['target_rtg_scaled'][:-1].astype(np.float32)-(arrays['reward'][:-1]/metadata['rtg_scale']).astype(np.float32),atol=1e-6)
        assert np.isnan(arrays['next_target_rtg_scaled_observed'][-1])
        assert not arrays['next_target_rtg_observed_mask'][-1]
        assert arrays['physics_qpos'].shape[0]==count+1
        assert arrays['physics_qvel'].shape[0]==count+1
        assert arrays['terminated'][-1] or arrays['truncated'][-1] or metadata['horizon_cutoff']
        normalized=(arrays['observation'].astype(np.float32)-np.array(metadata['state_mean'],dtype=np.float32))/np.array(metadata['state_std'],dtype=np.float32)
        assert np.array_equal(normalized,arrays['observation_normalized'])
    print('PASS: per-step RTG/state alignment, realized RTG, termination flags, physical states, checkpoint linkage',flush=True)
    for update in range(1,5):
        inputs=reconstruct_batch(run,update)
        with np.load(run/'training_batches'/f'update_{update:07d}.npz',allow_pickle=False) as snapshot:
            for key,value in inputs.items():
                assert np.array_equal(value,snapshot[key]),key
    export_episode(episodes[0],run/'review')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0,'images':0})
    accumulator.Reload()
    assert len(accumulator.Scalars('train/action_mse_step'))==4
    assert len(accumulator.Images('rollout/target_3600/first_episode'))==2
    assert len(list((run/'checkpoints').glob('iteration_*.pt')))==2
    import io
    from PIL import Image
    videos=list((run/'videos').glob('*.gif'))
    assert len(videos)==4
    for video in videos:
        with Image.open(video) as image:
            assert image.n_frames>=2
    video_summaries=accumulator.Images('simulation/target_3600/first_episode')
    assert len(video_summaries)==2
    with Image.open(io.BytesIO(video_summaries[-1].encoded_image_string)) as image:
        assert image.n_frames>=2
    print('PASS: animated MuJoCo GIFs and TensorBoard animated image summaries',flush=True)
    print('PASS: every-update input reconstruction, dense TensorBoard scalars, trajectory images and CSV export',flush=True)
    print('Validation run:',run,flush=True)
