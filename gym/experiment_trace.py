'''
逐步复盘数据记录工具，不实现新的策略、损失或环境动力学。
所有评估时间步保留到 NPZ；所有训练更新保留实际采样索引，按需还原完整 batch。
'''
import json
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def info_json(info):
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    return json.dumps(convert(info), ensure_ascii=False)


def capture_physics(env):
    '''Hopper 观测不含完整全局位置，因此额外保存 MuJoCo 的 qpos/qvel 等物理状态。'''
    try:
        sim = env.unwrapped.sim
        state = sim.get_state()
        return {
            'time': np.array(state.time), 'qpos': state.qpos.copy(), 'qvel': state.qvel.copy(),
            'act': np.asarray(state.act if state.act is not None else [], dtype=np.float64).copy(), 'qacc_warmstart': sim.data.qacc_warmstart.copy(),
            'ctrl': sim.data.ctrl.copy(),
        }
    except AttributeError:
        return None


class EpisodeTrace:
    '''一次实际评估回合；由原版 reset/get_action/step 的包装收集，不重复调用模型或环境。'''
    def __init__(self, observer, episode_number):
        self.observer = observer
        self.episode_number = episode_number
        self.rows = []
        self.physics = []
        self.previous_observation = None
        self.policy_input = None
        self.env = None
        self.rendered_actions = []
        self.path = observer.run_dir / 'trajectories' / (
            f'iteration_{observer.iteration:04d}_target_{observer.active_target}_episode_{episode_number:04d}.npz'
        )

    def on_reset(self, env, observation):
        self.env = env
        self.previous_observation = np.asarray(observation).copy()
        self.physics = [capture_physics(env)]

    def get_action(self, function):
        def record(*args, **kwargs):
            output = function(*args, **kwargs)
            states = args[0]
            if self.observer.args.model_type == 'dt':
                rtg = args[3] if len(args) > 3 else kwargs['returns_to_go']
                timesteps = args[4] if len(args) > 4 else kwargs['timesteps']
                condition = float(rtg.reshape(-1)[-1].detach().cpu().item())
                timestep = int(timesteps.reshape(-1)[-1].detach().cpu().item())
            else:
                condition, timestep = np.nan, len(self.rows)
            self.policy_input = {
                'observation_normalized': states.reshape(-1, self.observer.dataset['state_dim'])[-1].detach().cpu().numpy().copy(),
                'target_rtg_scaled': condition,
                'model_timestep': timestep,
                'action_predicted': output.detach().cpu().numpy().copy(),
            }
            return output
        return record

    def on_step(self, env, action, result):
        next_observation, reward, done, info = result
        if self.previous_observation is None or self.policy_input is None:
            raise RuntimeError('Trace missing reset or model input; refusing to fabricate a trajectory')
        truncated = bool(info.get('TimeLimit.truncated', False))
        row = {
            'observation': self.previous_observation.copy(),
            'next_observation': np.asarray(next_observation).copy(),
            'action': np.asarray(action).copy(), 'reward': float(reward), 'info_json': info_json(info),
            'done': bool(done), 'terminated': bool(done and not truncated), 'truncated': truncated,
            'environment_timestep': len(self.rows), **self.policy_input,
        }
        self.rows.append(row)
        self.physics.append(capture_physics(env))
        self.previous_observation = np.asarray(next_observation).copy()
        self.policy_input = None

    def finish(self, complete, error=None):
        if not self.rows:
            return None
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        scale = self.observer.dataset['scale']
        arrays['target_rtg_raw'] = arrays['target_rtg_scaled'] * scale
        arrays['cumulative_reward'] = np.cumsum(arrays['reward'])
        arrays['cumulative_reward_before'] = np.concatenate([[0.], arrays['cumulative_reward'][:-1]])
        arrays['realized_rtg_raw'] = np.cumsum(arrays['reward'][::-1])[::-1]
        arrays['next_target_rtg_scaled_observed'] = np.concatenate([arrays['target_rtg_scaled'][1:], [np.nan]])
        arrays['next_target_rtg_observed_mask'] = np.arange(len(self.rows)) < len(self.rows) - 1
        cutoff = bool(complete and not arrays['done'][-1])
        metadata = {
            'schema_version': 1, 'iteration': self.observer.iteration, 'global_step': self.observer.global_step,
            'target_return': self.observer.active_target, 'episode': self.episode_number,
            'context_length': self.observer.args.K,
            'mode': self.observer.args.mode, 'model_type': self.observer.args.model_type,
            'seed': self.observer.args.seed, 'steps': len(self.rows), 'complete': complete,
            'total_return': float(arrays['reward'].sum()), 'rtg_scale': scale,
            'target_rtg_used_by_policy': self.observer.args.model_type == 'dt',
            'horizon_cutoff': cutoff, 'error': error,
            'checkpoint': getattr(self.observer, 'checkpoint_reference', f'checkpoints/iteration_{self.observer.iteration:04d}.pt'),
            'state_mean': self.observer.dataset['state_mean'].tolist(),
            'state_std': self.observer.dataset['state_std'].tolist(),
            'realized_rtg_scope': 'full_episode' if complete else 'observed_prefix_only',
            'last_next_rtg': 'not observed; intentionally stored as NaN',
            'physics_note': 'qpos/qvel/time/act/ctrl/warmstart are diagnostics, not a bitwise simulator replay guarantee',
        }
        if self.env is not None:
            metadata['environment_id'] = getattr(getattr(self.env, 'spec', None), 'id', None)
            try:
                model = self.env.unwrapped.sim.model
                metadata['joint_names'] = list(model.joint_names)
                metadata['actuator_names'] = list(model.actuator_names)
                metadata['actuator_gear'] = model.actuator_gear.tolist()
            except AttributeError:
                pass
        if all(state is not None for state in self.physics):
            for key in self.physics[0]:
                arrays['physics_' + key] = np.stack([state[key] for state in self.physics])
        arrays['metadata_json'] = np.array(json.dumps(metadata, ensure_ascii=False))
        atomic_npz(self.path, **arrays)
        self.observer.csv_row('trajectory_index.csv', {
            'iteration': self.observer.iteration, 'global_step': self.observer.global_step,
            'target_return': self.observer.active_target, 'episode': self.episode_number,
            'context_length': self.observer.args.K,
            'steps': len(self.rows), 'total_return': metadata['total_return'], 'complete': complete,
            'terminated': bool(arrays['terminated'][-1]), 'truncated': bool(arrays['truncated'][-1]),
            'horizon_cutoff': cutoff, 'path': str(self.path.relative_to(self.observer.run_dir)),
            'checkpoint': metadata['checkpoint'],
        })
        return arrays, metadata


def attach_environment_trace(env, observer):
    '''只观察真实返回值；保持原环境对象、类型和 reset/step 返回对象不变。'''
    original_reset, original_step = env.reset, env.step
    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        if observer.active_trace is not None:
            observer.active_trace.on_reset(env, result)
        return result
    def step(action):
        result = original_step(action)
        if observer.active_trace is not None:
            observer.active_trace.on_step(env, action, result)
        return result
    env.reset, env.step = reset, step
    return env


def capture_batch(get_batch, observer, *args, **kwargs):
    '''
    临时替换原 experiment 模块的随机数命名空间，记录它真正抽中的轨迹与起点。
    不重新抽样，也不修改 NumPy/Python 的全局随机函数；后台日志线程不受影响。
    '''
    import experiment as official
    numpy_module, random_module = official.np, official.random
    numpy_proxy = types.SimpleNamespace(**vars(numpy_module))
    numpy_random_proxy = types.SimpleNamespace(**vars(numpy_module.random))
    random_proxy = types.SimpleNamespace(**vars(random_module))
    positions, starts = [], []
    def choice(*values, **options):
        result = numpy_module.random.choice(*values, **options)
        positions.append(np.asarray(result).copy())
        return result
    def randint(*values, **options):
        result = random_module.randint(*values, **options)
        starts.append(result)
        return result
    numpy_random_proxy.choice = choice
    numpy_proxy.random = numpy_random_proxy
    random_proxy.randint = randint
    with patch.object(official, 'np', numpy_proxy), patch.object(official, 'random', random_proxy):
        batch = get_batch(*args, **kwargs)
    if len(positions) != 1 or len(starts) != len(positions[0]):
        raise RuntimeError('Official sampler structure differs; cannot reliably record provenance')
    ids = observer.dataset['sorted_inds'][positions[0]].astype(np.int64)
    provenance = {
        'global_step': observer.global_step + 1, 'iteration': observer.iteration,
        'iteration_step': observer.completed + 1, 'trajectory_indices': ids.tolist(),
        'start_indices': starts, 'max_length': int(batch[0].shape[1]),
        'batch_size': int(batch[0].shape[0]),
    }
    handle = observer.provenance_file
    handle.write(json.dumps(provenance, separators=(',', ':')) + '\n')
    handle.flush()
    observer.current_provenance = provenance
    return batch


def save_batch_snapshot(observer):
    '''保存日志采样点的实际输入和预测；其余更新可根据 provenance 和原始数据还原输入。'''
    if observer.trace_batch is None:
        return
    names = ['states_normalized', 'actions_data', 'rewards', 'dones', 'rtg_scaled', 'timesteps', 'attention_mask']
    arrays = {name: tensor.detach().cpu().numpy().copy() for name, tensor in zip(names, observer.trace_batch)}
    if observer.trace_predictions is not None:
        arrays['action_predictions'] = observer.trace_predictions
    arrays['metadata_json'] = np.array(json.dumps({
        **observer.current_provenance, 'mode': observer.args.mode,
        'rtg_scale': observer.dataset['scale'], 'checkpoint_scope': 'iteration checkpoint; not this exact optimizer update',
    }))
    path = observer.run_dir / 'training_batches' / f'update_{observer.global_step:07d}.npz'
    atomic_npz(path, **arrays)
    observer.trace_batch = observer.trace_predictions = None


def trajectory_figure(arrays, metadata):
    '''离线生成四联图；不用 env.render，不改变环境或模型状态。'''
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1, sharex=True)
    t = arrays['environment_timestep']
    axes[0].plot(t, arrays['target_rtg_raw'], label='Target RTG (policy condition)')
    axes[0].plot(t, arrays['realized_rtg_raw'], label='Realized future return (hindsight)')
    axes[0].set_ylabel('Return'); axes[0].legend(loc='best')
    axes[1].plot(t, arrays['reward'], label='Reward')
    axes[1].set_ylabel('Step reward'); axes[1].legend(loc='upper left')
    right = axes[1].twinx()
    right.plot(t, arrays['cumulative_reward'], color='tab:orange', label='Cumulative reward')
    right.set_ylabel('Cumulative reward'); right.legend(loc='upper right')
    for index in range(arrays['action'].shape[1]):
        axes[2].plot(t, arrays['action'][:, index], label=f'action[{index}]')
    axes[2].set_ylabel('Action'); axes[2].legend(loc='best')
    for index in range(min(3, arrays['observation'].shape[1])):
        axes[3].plot(t, arrays['observation'][:, index], label=f'observation[{index}]')
    axes[3].set_ylabel('Raw observation'); axes[3].set_xlabel('Environment timestep'); axes[3].legend(loc='best')
    for axis in axes:
        axis.grid(alpha=.25)
    figure.suptitle(f"Iteration {metadata['iteration']} | target {metadata['target_return']} | episode {metadata['episode']} | return {metadata['total_return']:.2f}")
    return figure
