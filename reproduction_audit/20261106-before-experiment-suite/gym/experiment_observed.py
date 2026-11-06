'''
原版 Decision Transformer 的独立观测入口。

保留 experiment.py、模型和训练器源码，通过包装原训练器记录日志；参数更新仍调用原方法。
训练默认参数遵循已确定的论文协议：尤其 warmup_steps=100000。
从 gym 目录运行：python -u experiment_observed.py --seed 0

新增功能：终端进度表、TensorBoard、CSV/JSONL、环境和数据快照、每轮最新 checkpoint。
这些记录用于观察单次实验；论文分数仍需对应任务归一化及三个种子的统计。
'''
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from unittest.mock import patch

import numpy as np
import torch

import experiment as official
from experiment_trace import (
    EpisodeTrace, attach_environment_trace, capture_batch, save_batch_snapshot, trajectory_figure,
)


CORE_DEFAULTS = {
    'env': 'hopper', 'dataset': 'medium', 'mode': 'normal', 'K': 20,
    'pct_traj': 1., 'batch_size': 64, 'model_type': 'dt', 'embed_dim': 128,
    'n_layer': 3, 'n_head': 1, 'activation_function': 'relu', 'dropout': .1,
    'learning_rate': 1e-4, 'weight_decay': 1e-4, 'warmup_steps': 100000,
    'num_eval_episodes': 100, 'max_iters': 10, 'num_steps_per_iter': 10000,
    'device': 'cuda', 'log_to_swanlab': False,
}


def json_value(value):
    '''JSON 不写入非标准的 NaN/Infinity，异常数值另以 loss_finite 标记。'''
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class Tee:
    '''同时保留终端输出和原始 console.log；不依赖 shell 的 tee。'''
    def __init__(self, terminal, logfile, lock):
        self.terminal, self.logfile, self.lock = terminal, logfile, lock

    def write(self, text):
        with self.lock:
            self.terminal.write(text)
            self.logfile.write(text)
            self.logfile.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.logfile.flush()

    def isatty(self):
        return False


class Tables:
    '''按中文终端显示宽度对齐；连续进度共用表头，长路径会折行。'''
    def __init__(self):
        self.key = None
        self.border = None

    @staticmethod
    def width(text):
        return sum(0 if unicodedata.combining(c) else
                   2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(text))

    @classmethod
    def wrap(cls, text, width):
        result, line = [], ''
        for char in str(text):
            if char == '\n' or (line and cls.width(line + char) > width):
                result.append(line)
                line = ''
            if char != '\n':
                line += char
        return result + [line]

    def cells(self, values, widths):
        cells = [self.wrap(value, width) for value, width in zip(values, widths)]
        for index in range(max(map(len, cells))):
            parts = []
            for lines, width in zip(cells, widths):
                text = lines[index] if index < len(lines) else ''
                parts.append(' ' + text + ' ' * (width - self.width(text)) + ' ')
            print('|' + '|'.join(parts) + '|', flush=True)

    def row(self, key, title, headers, values, widths):
        if key != self.key:
            self.close()
            print('\n' + title, flush=True)
            self.border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
            print(self.border, flush=True)
            self.cells(headers, widths)
            print(self.border, flush=True)
            self.key = key
        self.cells(values, widths)

    def close(self):
        if self.key is not None:
            print(self.border, flush=True)
        self.key = None

    def info(self, title, values):
        for key, value in values:
            self.row(title, title, ['项目', '内容'], [key, value], [24, 76])
        self.close()


def duration(seconds):
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes}m{seconds:02d}s'


def reference_scores(env_name, dataset):
    '''直接读取已安装 D4RL 的官方评分常量，不导入其他任务或更换模拟器。'''
    spec = importlib.util.find_spec('d4rl')
    if spec is None or spec.origin is None:
        return None
    info = runpy.run_path(str(Path(spec.origin).with_name('infos.py')))
    key = f'{env_name}-{dataset}-v2'
    if key not in info['REF_MIN_SCORE'] or key not in info['REF_MAX_SCORE']:
        return None
    return float(info['REF_MIN_SCORE'][key]), float(info['REF_MAX_SCORE'][key])


class Observer:
    '''
    只持有观测状态：全局更新数、区间损失、评估记录和输出文件。
    TensorBoard 的横轴：训练和汇总评估使用全局参数更新数，单回合曲线使用已完成回合数。
    '''
    def __init__(self, args, writer_factory=None):
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            from matplotlib.figure import Figure
            writer_factory = SummaryWriter
        self.args = args
        self.config = {name: getattr(args, name) for name in CORE_DEFAULTS}
        suffix = re.sub(r'[^\w.-]+', '_', args.run_name or '')
        seed_label = f'seed{args.seed}' if args.seed is not None else 'seed-unset'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        name = f'{args.env}-{args.dataset}-{args.model_type}-{seed_label}-{stamp}'
        if suffix:
            name += '-' + suffix
        self.run_dir = Path(args.output_dir).resolve() / name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.swanlab = None
        if args.log_to_swanlab:
            from swanlab_observer import SwanLabObserver
            self.swanlab = SwanLabObserver(args, self.run_dir)
        self.writer = writer_factory(log_dir=str(self.run_dir / 'tensorboard'), flush_secs=5)
        self.tables = Tables()
        self.started = time.perf_counter()
        self.global_step = 0
        self.iteration = 0
        self.total_steps = 0
        self.completed = 0
        self.batch_seconds = 0.
        self.interval_losses = []
        self.pending_batch = None
        self.active_target = None
        self.active_trace = None
        self.current_provenance = None
        self.trace_batch = None
        self.trace_predictions = None
        self.provenance_file = (self.run_dir / 'training_provenance.jsonl').open('w')
        self.checkpoint_saved_iteration = None
        self.episode_counts = {}
        self.episode_returns = []
        self.episode_lengths = []
        self.dataset = None
        self.refs = reference_scores(args.env, args.dataset)
        self.gpu = args.device.startswith('cuda')
        self.files = {}
        self.csv_writers = {}
        dump_json(self.run_dir / 'config.json', vars(args))
        dump_json(self.run_dir / 'recording_schema.json', {
            'schema_version': 1,
            'training_updates': 'train_steps.csv 每次参数更新一行；global_step 不是环境时间步',
            'training_inputs': 'training_provenance.jsonl 每次更新的真实轨迹编号和起点；配合哈希一致的数据可精确还原输入',
            'training_predictions': 'training_batches/*.npz 仅在 batch_snapshot_interval 指定更新保留；设为1可保存每次预测',
            'evaluation_coverage': '全部评估回合的全部环境时间步，正常结束或异常时分别保存 complete 状态',
            'trajectory_axes': 'observation[t]、action[t]、target_rtg_scaled[t] 在执行动作前；reward[t]、next_observation[t] 在执行后',
            'target_rtg_scaled': '原版模型实际收到的 float32 目标 RTG，不是事后奖励和',
            'target_rtg_raw': '实际目标 RTG 乘 scale，可能含 float32 舍入误差',
            'realized_rtg_raw': '从该步至回合末的实际奖励和；partial 回合只有已记录片段的奖励和',
            'next_target_rtg_scaled_observed': '取下一次实际模型调用的条件，最后一步未知，存NaN并配mask，不伪造',
            'observations': '原始环境观测，observation_normalized 是实际模型调用的最后一个标准化状态',
            'actions': 'action 是传给 env.step 的控制量，action_predicted 是模型返回值；不是已换算的物理力矩',
            'physics': 'T+1 个物理状态，对应初始状态和每次执行后的状态；不承诺位级物理回放',
            'checkpoints': 'checkpoints/iteration_XXXX.pt 保留每轮；评估前先保存，轮末更新随机数状态；没有每次参数更新的模型快照',
            'tensorboard': 'loss每次更新记录；终端每log_interval次显示；每轮每个目标第一回合的诊断图见Images',
            'overhead': '逐步复制、压缩和画图增加耗时；时间指标是启用记录后的运行耗时',
        })
        self.write_metadata()
        self.writer.add_text('说明/阅读指南',
            '训练 loss 是有效动作位置的 MSE；回报越高通常越好，但 loss 降低不保证回报提高。\n\n'
            '评估按目标 RTG 分开；normalized_score 是 D4RL 参考分数 ×100。\n\n'
            'reference/paper_hopper_medium 是论文 67.6 的参考线，不是本次运行的实测分数。\n\n'
            '每次运行是一个种子，评估回合间标准差不等于论文跨三个种子的标准差。', 0)
        self.writer.add_custom_scalars({
            '训练': {'动作MSE': ['Multiline', ['train/action_mse_step', 'train/action_mse_mean']]},
            '评估': {
                '两个RTG条件的原始回报': ['Multiline', ['eval/target_3600/return_mean', 'eval/target_1800/return_mean']],
                'Hopper评分与论文参考': ['Multiline', ['eval/target_3600/normalized_score_mean',
                    'eval/target_1800/normalized_score_mean', 'reference/paper_hopper_medium']],
            },
        } if args.env == 'hopper' else {
            '训练': {'动作MSE': ['Multiline', ['train/action_mse_step', 'train/action_mse_mean']]},
        })
        self.writer.flush()

    def write_metadata(self):
        workspace = Path(__file__).resolve().parents[1]
        paths = [workspace / 'gym/experiment_observed.py',
                 workspace / 'gym/swanlab_observer.py',
                 workspace / 'gym/experiment_trace.py',
                 workspace / 'gym/review_experiment.py',
                 workspace / 'gym/render_trajectory.py',
                 workspace / 'gym/experiment.py',
                 workspace / 'gym/decision_transformer/models/decision_transformer.py',
                 workspace / 'gym/decision_transformer/models/trajectory_gpt2.py',
                 workspace / 'gym/decision_transformer/training/seq_trainer.py',
                 workspace / 'gym/decision_transformer/training/trainer.py',
                 workspace / 'gym/decision_transformer/evaluation/evaluate_episodes.py']
        hashes = {str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                    if dist.metadata.get('Name')}
        dump_json(self.run_dir / 'metadata.json', {
            'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'executable': sys.executable, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'device': self.args.device,
            'runtime_flags': {
                'torch_num_threads': torch.get_num_threads(),
                'torch_num_interop_threads': torch.get_num_interop_threads(),
                'default_dtype': str(torch.get_default_dtype()),
                'cudnn_version': torch.backends.cudnn.version(),
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cuda_matmul_allow_tf32': getattr(torch.backends.cuda.matmul, 'allow_tf32', None),
                'cudnn_allow_tf32': getattr(torch.backends.cudnn, 'allow_tf32', None),
            },
            'gpu': torch.cuda.get_device_name(self.args.device) if self.gpu else None,
            'source_sha256': hashes, 'versions': versions,
            'd4rl_reference_scores': self.refs,
            'checkpoint_resume_supported': False,
            'notes': '独立观测包装，不改原版更新规则；warmup 默认按论文；seed 仅在显式指定时设置。',
        })
        self.startup_info = [
            ('结果目录', self.run_dir), ('设备', self.args.device),
            ('PyTorch / CUDA', f'{torch.__version__} / {torch.version.cuda}'),
            ('GPU', torch.cuda.get_device_name(self.args.device) if self.gpu else '未使用GPU'),
            ('随机种子', self.args.seed if self.args.seed is not None else '未显式设置'),
            ('训练规模', f'{self.args.max_iters} 轮 × {self.args.num_steps_per_iter} 次更新'),
            ('学习率 / 预热', f'{self.args.learning_rate} / {self.args.warmup_steps} 次更新'),
            ('终端 / 曲线间隔', f'表格每 {self.args.log_interval} 次；loss 曲线每次参数更新'),
            ('逐步评估记录', '全部回合的每个环境时间步，含状态、动作、奖励、RTG、物理状态'),
            ('仿真动画', '关闭' if self.args.no_video else '每轮每个目标的首回合；videos/ 和 TensorBoard Images'),
            ('训练输入复盘', f'每次更新保存采样索引；每 {self.args.batch_snapshot_interval} 次保存实际输入/预测快照'),
            ('TensorBoard目录', self.run_dir.parent),
            ('SwanLab', ('启用 / ' + self.args.swanlab_mode) if self.swanlab is not None else '关闭'),
        ]

    def event(self, event, **values):
        record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                  'elapsed_seconds': time.perf_counter() - self.started,
                  'event': event, 'global_step': self.global_step, 'iteration': self.iteration, **values}
        with (self.run_dir / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(json_value(record), ensure_ascii=False, allow_nan=False) + '\n')

    def csv_row(self, name, values):
        if name not in self.files:
            self.files[name] = (self.run_dir / name).open('w', newline='')
            self.csv_writers[name] = csv.DictWriter(self.files[name], fieldnames=list(values))
            self.csv_writers[name].writeheader()
        self.csv_writers[name].writerow(values)
        self.files[name].flush()

    def scalar(self, tag, value, step=None):
        self.writer.add_scalar(tag, float(value), self.global_step if step is None else step)
        if self.swanlab is not None:
            self.swanlab.scalar(tag, value, self.global_step, self.global_step if step is None else step)

    def attach_dataset(self, get_batch):
        '''从原版采样闭包读取统计量；不更改轨迹、筛选索引或采样概率。'''
        values = inspect.getclosurevars(get_batch).nonlocals
        trajectories = values['trajectories']
        lengths = np.array([len(t['rewards']) for t in trajectories])
        returns = np.array([t['rewards'].sum() for t in trajectories])
        self.dataset = values
        path = Path(f'data/{self.args.env}-{self.args.dataset}-v2.pkl')
        summary = {
            'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'trajectories': len(trajectories), 'timesteps': int(lengths.sum()),
            'selected_trajectories': len(values['sorted_inds']),
            'selected_timesteps': int(lengths[values['sorted_inds']].sum()),
            'return_mean': float(returns.mean()), 'return_std': float(returns.std()),
            'return_min': float(returns.min()), 'return_max': float(returns.max()),
            'return_quantiles_5_50_95': np.quantile(returns, [.05, .5, .95]).tolist(),
            'state_mean': values['state_mean'].tolist(), 'state_std': values['state_std'].tolist(),
            'rtg_scale': values['scale'], 'max_episode_length': values['max_ep_len'],
        }
        dump_json(self.run_dir / 'dataset.json', summary)
        self.writer.add_histogram('data/trajectory_return', returns, 0)
        self.writer.add_histogram('data/trajectory_length', lengths, 0)
        self.tables.info('数据集', [
            ('轨迹数 / 总步数', f'{len(trajectories)} / {lengths.sum()}'),
            ('保留轨迹 / 步数', f'{summary["selected_trajectories"]} / {summary["selected_timesteps"]}'),
            ('回报均值±标准差', f'{returns.mean():.2f} ± {returns.std():.2f}'),
            ('回报最小 / 最大', f'{returns.min():.2f} / {returns.max():.2f}'),
            ('模型参数量', sum(p.numel() for p in self.trainer.model.parameters())),
            ('归一化评分参考', self.refs if self.refs is not None else '无；只记录原始回报'),
        ])
        self.writer.flush()

    def logging_due(self):
        step = self.completed + 1
        return step == 1 or step % self.args.log_interval == 0 or step == self.total_steps

    def batch_trace_due(self):
        interval = self.args.batch_snapshot_interval
        return interval > 0 and ((self.global_step + 1) == 1 or (self.global_step + 1) % interval == 0)

    def wrap_batch(self, get_batch):
        def measured(*args, **kwargs):
            started = time.perf_counter()
            batch = capture_batch(get_batch, self, *args, **kwargs)
            if self.batch_trace_due():
                self.trace_batch = batch
            self.batch_seconds += time.perf_counter() - started
            if self.logging_due():
                self.pending_batch = batch
            return batch
        return measured

    def start_iteration(self, number, total_steps):
        self.iteration, self.total_steps, self.completed = number, total_steps, 0
        self.batch_seconds = 0.
        self.interval_losses = []
        self.iteration_started = self.interval_started = time.perf_counter()
        self.interval_sampling_start = 0.
        if self.gpu:
            torch.cuda.reset_peak_memory_stats(self.args.device)
        self.tables.info('阶段', [('当前阶段', f'第 {number}/{self.args.max_iters} 轮训练开始')])
        self.event('training_start', total_steps=total_steps)

    def after_step(self, loss, step_seconds=0.):
        self.completed += 1
        self.global_step += 1
        self.interval_losses.append(float(loss))
        learning_rate = self.trainer.optimizer.param_groups[0]['lr']
        self.csv_row('train_steps.csv', {
            'global_step': self.global_step, 'iteration': self.iteration, 'iteration_step': self.completed,
            'action_mse': float(loss), 'learning_rate': learning_rate, 'step_seconds': step_seconds,
        })
        self.scalar('train/action_mse_step', loss)
        self.scalar('train/learning_rate', learning_rate)
        save_batch_snapshot(self)
        if self.completed != 1 and self.completed % self.args.log_interval and self.completed != self.total_steps:
            return
        interval_elapsed = time.perf_counter() - self.interval_started
        elapsed = time.perf_counter() - self.iteration_started
        speed = len(self.interval_losses) / max(interval_elapsed, 1e-9)
        sampling_percent = 100 * (self.batch_seconds - self.interval_sampling_start) / max(interval_elapsed, 1e-9)
        learning_rate = self.trainer.optimizer.param_groups[0]['lr']
        loss_mean = float(np.mean(self.interval_losses))
        record = {
            'global_step': self.global_step, 'iteration': self.iteration, 'iteration_step': self.completed,
            'interval_steps': len(self.interval_losses), 'action_mse_mean': loss_mean,
            'action_mse_std': float(np.std(self.interval_losses)), 'action_mse_last': float(loss),
            'loss_finite': bool(np.isfinite(self.interval_losses).all()), 'learning_rate': learning_rate,
            'steps_per_second': speed, 'sampling_percent': sampling_percent,
            'training_elapsed_seconds': elapsed, 'estimated_training_remaining_seconds': (self.total_steps - self.completed) / max(speed, 1e-9),
            'sampling_seconds': self.batch_seconds,
        }
        self.csv_row('train.csv', record)
        self.event('training_progress', **record)
        for field in ['action_mse_mean', 'action_mse_std', 'action_mse_last']:
            self.scalar('train/' + field, record[field])
        self.scalar('performance/updates_per_second', speed)
        self.scalar('performance/sampling_percent', sampling_percent)
        self.scalar('diagnostics/loss_finite', record['loss_finite'])

        '''只在日志点读取 batch 和梯度，避免每一步都加入额外的 GPU 同步。'''
        if self.pending_batch is not None:
            _, actions, _, _, rtg, _, mask = self.pending_batch
            valid = mask > 0
            raw_rtg = rtg[:, :-1, 0][valid].detach().cpu().numpy() * self.dataset['scale']
            action_data = actions[valid].detach().cpu().numpy()
            self.scalar('batch/padding_fraction', 1 - float(valid.float().mean().item()))
            self.writer.add_histogram('batch/rtg_raw', raw_rtg, self.global_step)
            self.writer.add_histogram('batch/action_data', action_data, self.global_step)
            self.pending_batch = None
        gradients = [p.grad.detach().norm(2) for p in self.trainer.model.parameters() if p.grad is not None]
        if gradients:
            tag = 'diagnostics/gradient_norm_after_clip' if self.args.model_type == 'dt' else 'diagnostics/gradient_norm'
            self.scalar(tag, torch.stack(gradients).norm(2).item())
        if self.gpu:
            self.scalar('system/gpu_memory_mib', torch.cuda.memory_allocated(self.args.device) / 1024**2)
            self.scalar('system/gpu_peak_memory_mib', torch.cuda.max_memory_allocated(self.args.device) / 1024**2)
        self.tables.row(('train', self.iteration), '训练进度（最近日志区间；采样耗时包含数据处理和传输）',
            ['轮次', '更新步数', '全局步', '平均loss', '本步LR', '步/s', '采样%', '已耗时', '训练ETA'],
            [self.iteration, f'{self.completed}/{self.total_steps}', self.global_step,
             f'{loss_mean:.5f}', f'{learning_rate:.2e}', f'{speed:.2f}', f'{sampling_percent:.1f}',
             duration(elapsed), duration(record['estimated_training_remaining_seconds'])],
            [4, 11, 8, 10, 10, 7, 7, 10, 10])
        if not record['loss_finite']:
            self.tables.close()
            print('警告：动作损失出现 NaN/Inf，详见 train.csv 和 events.jsonl。', flush=True)
        self.writer.flush()
        self.interval_losses = []
        self.interval_sampling_start = self.batch_seconds
        self.interval_started = time.perf_counter()

    def wrap_evaluation(self, fn):
        target = inspect.getclosurevars(fn).nonlocals['target_rew']
        def evaluate(model):
            if self.checkpoint_saved_iteration != self.iteration:
                self.save_checkpoint('before_evaluation')
            self.active_target = target
            self.episode_returns, self.episode_lengths = [], []
            started = self.evaluation_started = time.perf_counter()
            self.show_evaluation(0, 0.)
            self.event('evaluation_start', target_return=target)
            result = fn(model)
            elapsed = time.perf_counter() - started
            self.scalar(f'time/eval_target_{target}_seconds', elapsed)
            for key, value in result.items():
                metric = key[len(f'target_{target}_'):]
                self.scalar(f'eval/target_{target}/{metric}', value)
            if self.refs is not None:
                low, high = self.refs
                score_mean = 100 * (result[f'target_{target}_return_mean'] - low) / (high - low)
                score_std = 100 * result[f'target_{target}_return_std'] / (high - low)
                result[f'target_{target}_normalized_score_mean'] = score_mean
                result[f'target_{target}_normalized_score_std'] = score_std
                self.scalar(f'eval/target_{target}/normalized_score_mean', score_mean)
                self.scalar(f'eval/target_{target}/normalized_score_std', score_std)
                if self.args.env == 'hopper' and self.args.dataset == 'medium':
                    self.scalar('reference/paper_hopper_medium', 67.6)
            self.event('evaluation_done', target_return=target, duration_seconds=elapsed, metrics=result)
            self.writer.flush()
            self.active_target = None
            return result
        return evaluate

    def wrap_episode(self, function):
        def evaluate(*args, **kwargs):
            started = time.perf_counter()
            episode_number = len(self.episode_returns) + 1
            trace = EpisodeTrace(self, episode_number)
            model = args[3] if len(args) > 3 else kwargs['model']
            self.active_trace = trace
            complete, error = False, None
            try:
                with patch.object(model, 'get_action', trace.get_action(model.get_action)):
                    result = function(*args, **kwargs)
                complete = True
            except BaseException as failure:
                error = type(failure).__name__ + ': ' + str(failure)
                raise
            finally:
                self.active_trace = None
                saved = trace.finish(complete, error)
                if saved is not None and complete and episode_number == 1:
                    arrays, metadata = saved
                    figure = trajectory_figure(arrays, metadata)
                    figure_path = self.run_dir / 'trajectory_figures' / (trace.path.stem + '.png')
                    figure_path.parent.mkdir(exist_ok=True)
                    figure.savefig(figure_path, dpi=110)
                    if self.swanlab is not None:
                        self.swanlab.media(f'rollout/target_{self.active_target}/first_episode', figure_path, self.global_step)
                    self.writer.add_figure(f'rollout/target_{self.active_target}/first_episode', figure,
                                           global_step=self.global_step, close=True)
                    if not self.args.no_video:
                        self.record_video(trace.path)
            ret, length = result
            target = self.active_target
            self.episode_returns.append(float(ret))
            self.episode_lengths.append(int(length))
            self.episode_counts[target] = self.episode_counts.get(target, 0) + 1
            episode = len(self.episode_returns)
            self.csv_row('eval_episodes.csv', {
                'global_step': self.global_step, 'iteration': self.iteration,
                'target_return': target, 'episode': episode, 'return': float(ret),
                'length': int(length), 'duration_seconds': time.perf_counter() - started,
            })
            self.scalar(f'episode/target_{target}/return', ret, self.episode_counts[target])
            self.scalar(f'episode/target_{target}/length', length, self.episode_counts[target])
            if episode == 1 or episode % self.args.eval_log_interval == 0 or episode == self.args.num_eval_episodes:
                self.show_evaluation(episode, time.perf_counter() - self.evaluation_started)
                self.writer.flush()
            return result
        return evaluate

    def record_video(self, trajectory):
        '''
        子进程根据已记录状态渲染，父进程的环境、策略和随机数序列不变。
        GIF 是 TensorBoard add_video 使用的同类图像格式，无须另装 moviepy 或 FFmpeg。
        '''
        destination = self.run_dir / 'videos' / (trajectory.stem + '.gif')
        command = [sys.executable, str(Path(__file__).with_name('render_trajectory.py')), str(trajectory),
                   '--output', str(destination), '--stride', str(self.args.video_stride)]
        log = self.run_dir / 'video_render.log'
        try:
            with log.open('a') as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode != 0 or not destination.is_file():
                self.event('video_error', trajectory=str(trajectory), log=str(log))
                print(f'动画生成失败，训练数据已保留，详见 {log}', flush=True)
                return
            from PIL import Image
            from tensorboard.compat.proto.summary_pb2 import Summary
            with Image.open(destination) as image:
                width, height = image.size
            image = Summary.Image(height=height, width=width, colorspace=3,
                                  encoded_image_string=destination.read_bytes())
            summary = Summary(value=[Summary.Value(tag=f'simulation/target_{self.active_target}/first_episode', image=image)])
            self.writer.file_writer.add_summary(summary, self.global_step)
            self.writer.flush()
            self.event('video_saved', trajectory=str(trajectory), video=str(destination))
            if self.swanlab is not None:
                self.swanlab.media(f'simulation/target_{self.active_target}/first_episode', destination, self.global_step, video=True)
            self.tables.info('仿真动画', [('目标 RTG', self.active_target), ('动画文件', destination)])
        except subprocess.TimeoutExpired:
            self.event('video_error', reason='renderer timeout', trajectory=str(trajectory))
            print(f'动画渲染超时，轨迹保留在 {trajectory}', flush=True)

    def show_evaluation(self, episode, elapsed):
        mean_return = f'{np.mean(self.episode_returns):.2f} ± {np.std(self.episode_returns):.2f}' if episode else '-'
        score = '-'
        if episode and self.refs is not None:
            low, high = self.refs
            score = f'{100 * (np.mean(self.episode_returns) - low) / (high - low):.2f}'
        self.tables.row(('eval', self.iteration, self.active_target), '评估进度（± 为本次评估回合间标准差，评分为 D4RL ×100）',
            ['轮次', '目标RTG', '已评估回合', '原始回报均值±标准差', '归一化评分', '平均长度', '耗时'],
            [self.iteration, self.active_target, f'{episode}/{self.args.num_eval_episodes}', mean_return, score,
             f'{np.mean(self.episode_lengths):.1f}' if episode else '-', duration(elapsed)],
            [4, 8, 12, 23, 12, 8, 10])

    def save_checkpoint(self, phase):
        '''保留每轮模型，以便将轨迹对应到当时的策略；评估前也先保存，支持中断后的复盘。'''
        checkpoint = {
            'iteration': self.iteration, 'global_step': self.global_step, 'phase': phase,
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
            'variant': self.config, 'observer_config': vars(self.args),
            'state_mean': self.dataset['state_mean'], 'state_std': self.dataset['state_std'],
            'scale': self.dataset['scale'], 'max_ep_len': self.dataset['max_ep_len'],
            'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if self.gpu else None,
        }
        folder = self.run_dir / 'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder / f'iteration_{self.iteration:04d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        latest = self.run_dir / 'checkpoint_latest.tmp'
        shutil.copyfile(path, latest)
        latest.replace(self.run_dir / 'checkpoint_latest.pt')
        self.checkpoint_saved_iteration = self.iteration

    def finish_iteration(self, metrics):
        '''每轮保存完整指标和模型，不按最高分选优。'''
        self.tables.close()
        save_started = time.perf_counter()
        self.save_checkpoint('after_evaluation')
        save_seconds = time.perf_counter() - save_started
        metrics = {**metrics, 'time/sampling': self.batch_seconds, 'time/saving': save_seconds,
                   'time/iteration': time.perf_counter() - self.iteration_started}
        with (self.run_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(json_value({'iteration': self.iteration, 'global_step': self.global_step, **metrics}), allow_nan=False) + '\n')
        for key in ['time/training', 'time/evaluation', 'time/sampling', 'time/saving']:
            self.scalar(key, metrics[key])
        self.tables.info('本轮汇总', [
            ('轮次 / 全局更新数', f'{self.iteration} / {self.global_step}'),
            ('平均动作MSE', f'{metrics["training/train_loss_mean"]:.6f}'),
            ('训练 / 其中采样', f'{duration(metrics["time/training"])} / {duration(self.batch_seconds)}'),
            ('评估 / 保存', f'{duration(metrics["time/evaluation"])} / {duration(save_seconds)}'),
            ('最新checkpoint', self.run_dir / 'checkpoint_latest.pt'),
        ])
        self.event('iteration_done', metrics=metrics)
        self.writer.flush()

    def close(self):
        self.tables.close()
        if self.swanlab is not None:
            self.swanlab.close()
        self.writer.flush()
        self.writer.close()
        self.provenance_file.flush()
        self.provenance_file.close()
        for handle in self.files.values():
            handle.close()


def observed_trainer(base, observer):
    '''继承原训练器，super() 保留原始损失、反向、梯度裁剪、优化和学习率调度顺序。'''
    class ObservedTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observer.trainer = self
            observer.attach_dataset(self.get_batch)
            self.get_batch = observer.wrap_batch(self.get_batch)
            self.eval_fns = [observer.wrap_evaluation(fn) for fn in self.eval_fns]

        def train_step(self):
            step_started = time.perf_counter()
            if observer.batch_trace_due():
                original_forward = self.model.forward
                def forward(*values, **options):
                    result = original_forward(*values, **options)
                    observer.trace_predictions = result[1].detach().cpu().numpy().copy()
                    return result
                with patch.object(self.model, 'forward', forward):
                    loss = super().train_step()
            else:
                loss = super().train_step()
            observer.after_step(loss, time.perf_counter() - step_started)
            return loss

        def train_iteration(self, num_steps, iter_num=0, print_logs=False):
            observer.start_iteration(iter_num, num_steps)
            metrics = super().train_iteration(num_steps, iter_num, print_logs=False)
            observer.finish_iteration(metrics)
            return metrics
    return ObservedTrainer


def run_observed(args, writer_factory=None):
    '''
    设置 seed 是独立、显式的可重复性控制；不传 --seed 时保留官方未设置种子的行为。
    包装仅在本次调用内生效，结束或异常时恢复模块中的原函数和类。
    '''
    observer = Observer(args, writer_factory)
    original_make = official.gym.make
    envs = []

    def make_env(*values, **kwargs):
        env = original_make(*values, **kwargs)
        if args.seed is not None:
            env.seed(args.seed)
            env.action_space.seed(args.seed)
        env = attach_environment_trace(env, observer)
        envs.append(env)
        return env

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device.startswith('cuda'):
            torch.cuda.manual_seed_all(args.seed)
    lock = threading.RLock()
    try:
        with (observer.run_dir / 'console.log').open('w') as console:
            with contextlib.redirect_stdout(Tee(sys.stdout, console, lock)), contextlib.redirect_stderr(Tee(sys.stderr, console, lock)):
                observer.tables.info('实验与观测面板', observer.startup_info)
                with patch.object(official, 'SequenceTrainer', observed_trainer(official.SequenceTrainer, observer)), \
                     patch.object(official, 'ActTrainer', observed_trainer(official.ActTrainer, observer)), \
                     patch.object(official.gym, 'make', make_env), \
                     patch.object(official, 'evaluate_episode_rtg', observer.wrap_episode(official.evaluate_episode_rtg)), \
                     patch.object(official, 'evaluate_episode', observer.wrap_episode(official.evaluate_episode)):
                    observer.event('experiment_start')
                    try:
                        '''由观测适配器统一记录 SwanLab，避免官方入口再次初始化同一运行。'''
                        official.experiment('gym-experiment', {**observer.config, 'log_to_swanlab': False})
                        observer.event('experiment_done')
                        observer.tables.info('实验结束', [('总耗时', duration(time.perf_counter() - observer.started)),
                                                        ('结果目录', observer.run_dir)])
                    except BaseException as error:
                        observer.event('experiment_interrupted' if isinstance(error, KeyboardInterrupt) else 'experiment_error',
                                       error_type=type(error).__name__, error=str(error))
                        import traceback
                        traceback.print_exc()
                        raise
    finally:
        for env in envs:
            env.close()
        observer.close()
    return observer.run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in CORE_DEFAULTS.items():
        if name == 'log_to_swanlab':
            parser.add_argument('--log_to_swanlab', '-w', action='store_true')
        else:
            aliases = {'learning_rate': '-lr', 'weight_decay': '-wd'}
            flags = ['--' + name] + ([aliases[name]] if name in aliases else [])
            parser.add_argument(*flags, type=type(default), default=default)
    parser.add_argument('--seed', type=int, default=None, help='显式固定种子；不传则保留原入口行为')
    parser.add_argument('--output_dir', default='runs_observed')
    parser.add_argument('--run_name', default='')
    parser.add_argument('--swanlab_project', default='decision-transformer-reproduction')
    parser.add_argument('--swanlab_workspace', default=None, help='SwanLab 用户或团队；默认使用登录账号')
    parser.add_argument('--swanlab_mode', choices=['offline', 'online'], default='offline',
                        help='默认本地记录，登录后可显式选择 online 查看网页面板')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_log_interval', type=int, default=10)
    parser.add_argument('--no_video', action='store_true', help='关闭自动仿真动画；仍保存所有原始轨迹')
    parser.add_argument('--video_stride', type=int, default=4, help='动画每多少个已记录环境状态取一帧')
    parser.add_argument('--batch_snapshot_interval', type=int, default=100,
                        help='每多少次更新保存实际训练输入与预测，1=全部，0=只保存采样索引')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.log_interval < 1 or args.eval_log_interval < 1:
        parser.error('日志间隔必须大于 0')
    if args.video_stride < 1:
        parser.error('video_stride 必须大于 0')
    if args.batch_snapshot_interval < 0:
        parser.error('训练 batch 快照间隔不能小于 0')
    if args.num_steps_per_iter < 1 or args.num_eval_episodes < 1 or args.max_iters < 1:
        parser.error('更新次数、评估回合和轮数必须大于 0')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA 不可用，不自动回退 CPU')
    if not Path(f'data/{args.env}-{args.dataset}-v2.pkl').is_file():
        parser.error('未找到数据文件；请先 cd 到仓库 gym 目录，再运行此入口')
    run_observed(args)
