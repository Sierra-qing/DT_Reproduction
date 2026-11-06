'''
离线复盘工具：列出回合、导出逐步 CSV/PNG、按更新编号还原实际训练输入。
只读取已记录数据；不运行策略、不与环境交互、不要求 CUDA。
'''
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from experiment_trace import atomic_npz, trajectory_figure


def load_episode(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data['metadata_json'].item())
        arrays = {name: data[name].copy() for name in data.files if name != 'metadata_json'}
    return arrays, metadata


def export_episode(path, output):
    arrays, metadata = load_episode(path)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(arrays['reward'])):
        row = {}
        for name, values in arrays.items():
            if name.startswith('physics_'):
                continue
            value = values[index]
            if np.ndim(value) == 0:
                scalar = value.item() if hasattr(value, 'item') else value
                row[name] = '' if isinstance(scalar, float) and not np.isfinite(scalar) else scalar
            else:
                for dimension, scalar in enumerate(np.asarray(value).reshape(-1)):
                    row[f'{name}[{dimension}]'] = scalar
        for name in ['physics_time', 'physics_qpos', 'physics_qvel', 'physics_act', 'physics_ctrl', 'physics_qacc_warmstart']:
            if name not in arrays:
                continue
            for label, position in [('before', index), ('after', index + 1)]:
                value = arrays[name][position]
                if np.ndim(value) == 0:
                    row[f'{name}_{label}'] = value.item()
                else:
                    for dimension, scalar in enumerate(value):
                        row[f'{name}_{label}[{dimension}]'] = scalar
        rows.append(row)
    csv_path = output / (path.stem + '.csv')
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    figure = trajectory_figure(arrays, metadata)
    png_path = output / (path.stem + '.png')
    figure.savefig(png_path, dpi=130)
    (output / (path.stem + '.json')).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('CSV:', csv_path)
    print('Figure:', png_path)
    return csv_path, png_path


def discounted_returns(rewards):
    result = np.zeros_like(rewards)
    result[-1] = rewards[-1]
    for index in reversed(range(len(rewards) - 1)):
        result[index] = rewards[index] + result[index + 1]
    return result


def reconstruct_inputs(trajectories, provenance, dataset, config):
    '''按官方采样代码的运算顺序还原输入，不重新抽随机数。'''
    fields = {name: [] for name in ['states_normalized', 'actions_data', 'rewards', 'dones', 'rtg_scaled', 'timesteps', 'attention_mask']}
    max_length = provenance['max_length']
    state_mean, state_std = np.array(dataset['state_mean']), np.array(dataset['state_std'])
    max_episode_length = dataset['max_episode_length']
    scale = dataset['rtg_scale']
    for trajectory_id, start in zip(provenance['trajectory_indices'], provenance['start_indices']):
        trajectory = trajectories[trajectory_id]
        rewards = trajectory['rewards'].copy()
        if config['mode'] == 'delayed':
            rewards[-1] = rewards.sum(); rewards[:-1] = 0.
        stop = start + max_length
        states = trajectory['observations'][start:stop][None]
        actions = trajectory['actions'][start:stop][None]
        sequence_rewards = rewards[start:stop].reshape(1, -1, 1)
        key = 'terminals' if 'terminals' in trajectory else 'dones'
        dones = trajectory[key][start:stop].reshape(1, -1)
        length = states.shape[1]
        timesteps = np.arange(start, start + length).reshape(1, -1)
        timesteps[timesteps >= max_episode_length] = max_episode_length - 1
        rtg = discounted_returns(rewards[start:])[:length + 1].reshape(1, -1, 1)
        if rtg.shape[1] <= length:
            rtg = np.concatenate([rtg, np.zeros((1, 1, 1))], axis=1)
        padding = max_length - length
        states = np.concatenate([np.zeros((1, padding, states.shape[2])), states], axis=1)
        states = (states - state_mean) / state_std
        actions = np.concatenate([np.ones((1, padding, actions.shape[2])) * -10., actions], axis=1)
        sequence_rewards = np.concatenate([np.zeros((1, padding, 1)), sequence_rewards], axis=1)
        dones = np.concatenate([np.ones((1, padding)) * 2, dones], axis=1)
        rtg = np.concatenate([np.zeros((1, padding, 1)), rtg], axis=1) / scale
        timesteps = np.concatenate([np.zeros((1, padding)), timesteps], axis=1)
        mask = np.concatenate([np.zeros((1, padding)), np.ones((1, length))], axis=1)
        values = [states, actions, sequence_rewards, dones, rtg, timesteps, mask]
        for name, value in zip(fields, values):
            fields[name].append(value)
    dtypes = [np.float32, np.float32, np.float32, np.int64, np.float32, np.int64, np.float64]
    return {name: np.concatenate(values, axis=0).astype(dtype) for (name, values), dtype in zip(fields.items(), dtypes)}


def reconstruct_batch(run_dir, update, output=None, dataset_path=None):
    dataset = json.loads((run_dir / 'dataset.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    provenance = None
    with (run_dir / 'training_provenance.jsonl').open() as handle:
        for line in handle:
            item = json.loads(line)
            if item['global_step'] == update:
                provenance = item; break
    if provenance is None:
        raise ValueError(f'没有记录到更新 {update} 的采样来源')
    source = dataset_path or Path(dataset['path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != dataset['sha256']:
        raise ValueError('数据文件哈希不匹配，不能声称还原了当时输入')
    with source.open('rb') as handle:
        trajectories = pickle.load(handle)
    inputs = reconstruct_inputs(trajectories, provenance, dataset, config)
    destination = output or run_dir / 'review' / f'batch_update_{update:07d}.npz'
    atomic_npz(destination, **inputs, metadata_json=np.array(json.dumps(provenance)))
    print('Reconstructed batch:', destination)
    print('只还原实际输入；未记录的中间更新预测不能事后从当前模型伪造。')
    return inputs


def policy_context(path, timestep, output=None):
    '''
    还原预测指定动作时进入 DT 的 K 步序列；当前动作是 0 占位，不能把答案泄露给模型。
    评估的状态已经标准化，再左补 0；注意它与训练中“先补 0 再标准化”的处理不同。
    '''
    arrays, metadata = load_episode(path)
    if metadata['model_type'] != 'dt':
        raise ValueError('此上下文导出针对 DT；BC 的 MLP 输入方式不同')
    if timestep < 0 or timestep >= len(arrays['reward']):
        raise ValueError('环境时间步超出范围')
    context_length = metadata['context_length']
    start = max(0, timestep + 1 - context_length)
    states = arrays['observation_normalized'][start:timestep + 1].astype(np.float32)
    actions = arrays['action'][start:timestep + 1].astype(np.float32).copy()
    actions[-1] = 0.
    rtg = arrays['target_rtg_scaled'][start:timestep + 1].astype(np.float32).reshape(-1, 1)
    timesteps = arrays['model_timestep'][start:timestep + 1].astype(np.int64)
    length = len(states)
    padding = context_length - length
    inputs = {
        'states_normalized': np.concatenate([np.zeros((padding, states.shape[1]), dtype=np.float32), states])[None],
        'actions_history': np.concatenate([np.zeros((padding, actions.shape[1]), dtype=np.float32), actions])[None],
        'rtg_scaled': np.concatenate([np.zeros((padding, 1), dtype=np.float32), rtg])[None],
        'timesteps': np.concatenate([np.zeros(padding, dtype=np.int64), timesteps])[None],
        'attention_mask': np.concatenate([np.zeros(padding, dtype=np.int64), np.ones(length, dtype=np.int64)])[None],
    }
    destination = output or path.parent.parent / 'review' / f'{path.stem}_context_t{timestep:04d}.npz'
    atomic_npz(destination, **inputs, metadata_json=np.array(json.dumps({**metadata, 'selected_environment_timestep': timestep})))
    print('Policy input context:', destination)
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='列出已记录评估回合')
    listing.add_argument('run_dir', type=Path)
    episode = commands.add_parser('episode', help='导出指定回合的 CSV 和诊断图')
    episode.add_argument('npz', type=Path)
    episode.add_argument('--output_dir', type=Path)
    batch = commands.add_parser('batch', help='按更新数还原完整训练输入')
    batch.add_argument('run_dir', type=Path)
    batch.add_argument('--update', type=int, required=True)
    batch.add_argument('--output', type=Path)
    batch.add_argument('--dataset', type=Path)
    context = commands.add_parser('context', help='还原某个评估动作预测时的完整 K 步模型输入')
    context.add_argument('npz', type=Path)
    context.add_argument('--timestep', type=int, required=True)
    context.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'list':
        index = args.run_dir / 'trajectory_index.csv'
        if not index.is_file():
            parser.error('此运行尚未保存逐步轨迹；旧日志导入运行无法补回这些数据')
        print(index.read_text(), end='')
    elif args.command == 'episode':
        export_episode(args.npz, args.output_dir or args.npz.parent.parent / 'review')
    elif args.command == 'context':
        policy_context(args.npz, args.timestep, args.output)
    else:
        reconstruct_batch(args.run_dir, args.update, args.output, args.dataset)


if __name__ == '__main__':
    main()
