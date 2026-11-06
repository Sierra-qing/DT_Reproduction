'''
Decision Transformer 现代适配实验的可读版本（学习参考，非官方基线）。

从文件末尾的 experiment() 开始阅读，可以先看到完整实验流程，再进入各个函数。
本文件对应回退前的现代适配版流程，包含 Gymnasium、显式随机种子和 checkpoint 保存。
当前 experiment.py 已恢复官方旧 Gym 基线，两者不再是同一套运行入口。
正式复现前请阅读仓库根目录 REPRODUCTION_AUDIT.md；此处保留用于理解数据流。

从 gym 目录运行：
    python experiment_readable.py --env hopper --dataset medium --model_type dt

训练只读取离线数据；环境交互用于评估，不会把评估轨迹加入训练集。
维度约定：B 为 batch 大小，K 为上下文步数，S 为观测维度，A 为动作维度。
'''

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


@dataclass
class EnvironmentSettings:
    '''
    将同一环境的实验设置放在一起，避免在不同函数间传递含义不清的元组。
    target_returns 是期望回报；return_scale 只缩放 RTG，不是 D4RL 评分公式。
    max_episode_length 是回合时间步上限，不是 Transformer 的 token 数。
    '''
    env_id: str
    max_episode_length: int
    target_returns: list
    return_scale: float


@dataclass
class PreparedDataset:
    '''
    保存数据和训练所需统计量。
    trajectory_returns 是每条完整轨迹的总回报；采样时还会计算每步的 RTG。
    selected_indices 指向原 trajectories 列表；sampling_probabilities 与该索引数组一一对应。
    状态均值和标准差仍从完整数据集计算，与原版保持一致。
    '''
    trajectories: list
    trajectory_lengths: np.ndarray
    trajectory_returns: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    selected_indices: np.ndarray
    sampling_probabilities: np.ndarray


def configure_randomness(seed, device, num_threads):
    '''设置随机数与 CPU 线程数。指定 CUDA 时若不可用则报错，不自动切换设备。'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check GPU access and the PyTorch installation.')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)


def create_run_directory(config):
    '''每次运行创建独立目录，并保存原始命令行配置；路径相对于当前工作目录。'''
    run_name = (
        f"{config['env']}-{config['dataset']}-{config['model_type']}-seed{config['seed']}-"
        + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    )
    run_directory = Path(config['output_dir']) / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Run directory: {run_directory}', flush=True)
    return run_directory


def create_environment(env_name, model_type, seed):
    '''
    环境用于评估；训练数据稍后从 pkl 文件加载。
    使用与 experiment.py 相同的 v4 环境，仍需注意其与原始 D4RL 模拟器的差异。
    自定义 Reacher 分支照原版保留，其旧环境 API 的兼容问题也没有在这里额外修改。
    '''
    settings_by_name = {
        'hopper': EnvironmentSettings('Hopper-v4', 1000, [3600, 1800], 1000.),
        'halfcheetah': EnvironmentSettings('HalfCheetah-v4', 1000, [12000, 6000], 1000.),
        'walker2d': EnvironmentSettings('Walker2d-v4', 1000, [5000, 2500], 1000.),
        'reacher2d': EnvironmentSettings('', 100, [76, 40], 10.),
    }
    if env_name not in settings_by_name:
        raise NotImplementedError
    settings = settings_by_name[env_name]

    if env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
    else:
        env = gym.make(settings.env_id)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    if model_type == 'bc':
        settings.target_returns = settings.target_returns[:1]
    return env, settings


def select_training_trajectories(trajectory_lengths, trajectory_returns, fraction):
    '''
    按总回报从高到低保留完整轨迹，预算是完整数据的 fraction 比例的时间步数。
    至少保留最高回报的一条；随后在下一条放不下时停止，与原版的边界规则相同。
    返回的索引仍按回报从低到高排列，以保持相同种子下的抽样映射。
    '''
    timestep_budget = max(int(fraction * sum(trajectory_lengths)), 1)
    ranked_indices = np.argsort(trajectory_returns)
    selected_count = 1
    selected_timesteps = trajectory_lengths[ranked_indices[-1]]
    candidate_position = len(trajectory_lengths) - 2

    while candidate_position >= 0:
        candidate_length = trajectory_lengths[ranked_indices[candidate_position]]
        if selected_timesteps + candidate_length > timestep_budget:
            break
        selected_timesteps += candidate_length
        selected_count += 1
        candidate_position -= 1

    return ranked_indices[-selected_count:]


def prepare_dataset(config, state_dim, action_dim):
    '''
    读取轨迹、处理奖励模式、计算统计量，再选取训练轨迹。
    每条轨迹包含 observations[T,S]、actions[T,A]、rewards[T] 和终止标记[T]。
    delayed 模式只修改内存：把全部奖励移到最后一步，保持总回报不变。
    '''
    dataset_path = f"data/{config['env']}-{config['dataset']}-v2.pkl"
    with open(dataset_path, 'rb') as dataset_file:
        trajectories = pickle.load(dataset_file)

    if not trajectories:
        raise ValueError('Dataset contains no trajectories')
    for trajectory in trajectories:
        if (trajectory['observations'].shape[1] != state_dim
                or trajectory['actions'].shape[1] != action_dim):
            raise ValueError('Dataset dimensions do not match the evaluation environment')

    observations = []
    trajectory_lengths = []
    trajectory_returns = []
    for trajectory in trajectories:
        if config.get('mode', 'normal') == 'delayed':
            trajectory['rewards'][-1] = trajectory['rewards'].sum()
            trajectory['rewards'][:-1] = 0.
        observations.append(trajectory['observations'])
        trajectory_lengths.append(len(trajectory['observations']))
        trajectory_returns.append(trajectory['rewards'].sum())

    trajectory_lengths = np.array(trajectory_lengths)
    trajectory_returns = np.array(trajectory_returns)
    all_states = np.concatenate(observations, axis=0)
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0) + 1e-6

    print('=' * 50)
    print(f"Starting new experiment: {config['env']} {config['dataset']}")
    print(f'{len(trajectory_lengths)} trajectories, {sum(trajectory_lengths)} timesteps found')
    print(f'Average return: {np.mean(trajectory_returns):.2f}, std: {np.std(trajectory_returns):.2f}')
    print(f'Max return: {np.max(trajectory_returns):.2f}, min: {np.min(trajectory_returns):.2f}')
    print('=' * 50)

    selected_indices = select_training_trajectories(
        trajectory_lengths, trajectory_returns, config.get('pct_traj', 1.),
    )
    selected_lengths = trajectory_lengths[selected_indices]
    sampling_probabilities = selected_lengths / sum(selected_lengths)
    return PreparedDataset(
        trajectories, trajectory_lengths, trajectory_returns,
        state_mean, state_std, selected_indices, sampling_probabilities,
    )


def discount_cumsum(rewards, gamma):
    '''从后向前求 G_t = r_t + gamma * G_(t+1)；gamma=1 时 [1,2,3] → [6,5,3]。'''
    returns_to_go = np.zeros_like(rewards)
    returns_to_go[-1] = rewards[-1]
    for timestep in reversed(range(rewards.shape[0] - 1)):
        returns_to_go[timestep] = rewards[timestep] + gamma * returns_to_go[timestep + 1]
    return returns_to_go


def left_pad(sequence, padding_length, value=0.):
    '''
    sequence 的形状为 [1,时间长度,...]，在时间维左侧填充值。
    padding 默认使用 NumPy float64，保持原版拼接时的数值类型提升行为。
    例如长度 3、补齐 2 步：[pad,pad,x0,x1,x2]。
    '''
    padding_shape = (1, padding_length, *sequence.shape[2:])
    padding = np.ones(padding_shape) * value
    return np.concatenate([padding, sequence], axis=1)


class TrajectoryBatchSampler:
    '''
    将原先嵌套在 experiment() 内的 get_batch 拆成独立采样器。
    对外仍提供 get_batch(batch_size)，可直接交给原来的 Trainer。
    数据、维度、归一化参数都有明确来源，不再依赖大函数里的隐式外层变量。
    '''

    def __init__(self, dataset, state_dim, action_dim, context_length, settings, device):
        self.dataset = dataset
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.settings = settings
        self.device = device

    def sample_sequence(self, trajectory, start, max_length):
        '''
        从指定轨迹和起点构造一个样本：截取、计算 RTG、左侧补齐、标准化。
        此函数不抽随机数，便于单独观察某个指定片段的数据变化。
        '''
        stop = start + max_length
        states = trajectory['observations'][start:stop].reshape(1, -1, self.state_dim)
        actions = trajectory['actions'][start:stop].reshape(1, -1, self.action_dim)
        rewards = trajectory['rewards'][start:stop].reshape(1, -1, 1)
        terminal_key = 'terminals' if 'terminals' in trajectory else 'dones'
        dones = trajectory[terminal_key][start:stop].reshape(1, -1)
        sequence_length = states.shape[1]

        '''
        时间索引仍是原回合内的绝对时间，不从片段起点重新编号为 0。
        RTG 先对原轨迹余下的所有奖励求和，再截取片段，避免漏掉片段之后的奖励。
        RTG 多留一项是原训练器接口约定；训练前使用 rtg[:, :-1] 对齐状态序列。
        '''
        timesteps = np.arange(start, start + sequence_length).reshape(1, -1)
        timesteps[timesteps >= self.settings.max_episode_length] = self.settings.max_episode_length - 1
        returns_to_go = discount_cumsum(trajectory['rewards'][start:], gamma=1.)
        returns_to_go = returns_to_go[:sequence_length + 1].reshape(1, -1, 1)
        if returns_to_go.shape[1] <= sequence_length:
            returns_to_go = np.concatenate([returns_to_go, np.zeros((1, 1, 1))], axis=1)

        '''
        真实片段不足 K 步时左侧补齐。状态先补 0 再标准化，与原版顺序相同。
        动作 -10、终止标记 2 都是占位值；当前 DT 损失会按 mask 排除补位动作。
        r 保持原奖励数值，只有 RTG 除以 return_scale。
        mask=0 表示补位，mask=1 表示真实时间步；因果注意力遮罩由模型另外处理。
        '''
        padding_length = max_length - sequence_length
        states = left_pad(states, padding_length)
        states = (states - self.dataset.state_mean) / self.dataset.state_std
        actions = left_pad(actions, padding_length, value=-10.)
        rewards = left_pad(rewards, padding_length)
        dones = left_pad(dones, padding_length, value=2.)
        returns_to_go = left_pad(returns_to_go, padding_length) / self.settings.return_scale
        timesteps = left_pad(timesteps, padding_length)
        attention_mask = np.concatenate([
            np.zeros((1, padding_length)), np.ones((1, sequence_length)),
        ], axis=1)
        return states, actions, rewards, dones, returns_to_go, timesteps, attention_mask

    def get_batch(self, batch_size=256, max_len=None):
        '''
        先按轨迹长度有放回抽取轨迹，再在各轨迹内均匀抽起点。
        每个时间步被当作起点的概率相同；同一条轨迹可以在一个 batch 中出现多次。
        返回顺序和 dtype 与原版一致：
        states[B,K,S]、actions[B,K,A]、rewards[B,K,1]、dones[B,K]、
        RTG[B,K+1,1]、timesteps[B,K]、attention_mask[B,K]。
        '''
        if max_len is None:
            max_len = self.context_length
        dataset = self.dataset
        sampled_positions = np.random.choice(
            np.arange(len(dataset.selected_indices)),
            size=batch_size,
            replace=True,
            p=dataset.sampling_probabilities,
        )
        samples = []
        for position in sampled_positions:
            trajectory_index = int(dataset.selected_indices[position])
            trajectory = dataset.trajectories[trajectory_index]
            start = random.randint(0, trajectory['rewards'].shape[0] - 1)
            samples.append(self.sample_sequence(trajectory, start, max_len))

        '''
        zip(*samples) 将“按样本排列”转为“按字段排列”，分别拼出 B 个状态序列、动作序列等。
        mask 保留原版 NumPy 推断出的浮点类型，其余字段显式转换为 float32 或 long。
        '''
        dtypes = (
            torch.float32, torch.float32, torch.float32, torch.long,
            torch.float32, torch.long, None,
        )
        batch = []
        for sample_field, dtype in zip(zip(*samples), dtypes):
            tensor = torch.from_numpy(np.concatenate(sample_field, axis=0))
            tensor = tensor.to(device=self.device) if dtype is None else tensor.to(dtype=dtype, device=self.device)
            batch.append(tensor)
        return tuple(batch)


def make_evaluation_function(env, settings, dataset, config, target_return):
    '''
    返回接收 model 的评估函数，供 Trainer 在每轮训练后调用。
    目标回报先缩放到训练时的量级；输出指标仍是环境产生的原始回报。
    '''
    def evaluate_model(model):
        episode_returns = []
        episode_lengths = []
        for _ in range(config['num_eval_episodes']):
            with torch.no_grad():
                evaluation_kwargs = dict(
                    max_ep_len=settings.max_episode_length,
                    target_return=target_return / settings.return_scale,
                    mode=config.get('mode', 'normal'),
                    state_mean=dataset.state_mean,
                    state_std=dataset.state_std,
                    device=config.get('device', 'cuda'),
                )
                if config['model_type'] == 'dt':
                    episode_return, episode_length = evaluate_episode_rtg(
                        env, env.observation_space.shape[0], env.action_space.shape[0], model,
                        scale=settings.return_scale, **evaluation_kwargs,
                    )
                else:
                    episode_return, episode_length = evaluate_episode(
                        env, env.observation_space.shape[0], env.action_space.shape[0], model,
                        **evaluation_kwargs,
                    )
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        return {
            f'target_{target_return}_return_mean': np.mean(episode_returns),
            f'target_{target_return}_return_std': np.std(episode_returns),
            f'target_{target_return}_length_mean': np.mean(episode_lengths),
            f'target_{target_return}_length_std': np.std(episode_lengths),
        }
    return evaluate_model


def create_model(config, state_dim, action_dim, settings):
    '''
    复用原仓库模型，不重写 Transformer。
    DT 将每步组织成 (RTG,state,action) 三个 token；BC 将 K 个状态拼接后输入 MLP。
    n_inner 是前馈网络中间维度；n_positions=1024 是底层注意力的 token 容量。
    '''
    common_kwargs = dict(
        state_dim=state_dim,
        act_dim=action_dim,
        max_length=config['K'],
        hidden_size=config['embed_dim'],
        n_layer=config['n_layer'],
    )
    if config['model_type'] == 'dt':
        model = DecisionTransformer(
            **common_kwargs,
            max_ep_len=settings.max_episode_length,
            n_head=config['n_head'],
            n_inner=4 * config['embed_dim'],
            activation_function=config['activation_function'],
            n_positions=1024,
            resid_pdrop=config['dropout'],
            attn_pdrop=config['dropout'],
        )
    elif config['model_type'] == 'bc':
        model = MLPBCModel(**common_kwargs)
    else:
        raise NotImplementedError
    return model.to(device=config.get('device', 'cuda'))


def action_prediction_loss(state_prediction, action_prediction, reward_prediction,
                           state_target, action_target, reward_target):
    '''
    训练器要求六个参数，但当前实验只使用动作 MSE，不监督状态和回报预测。
    SequenceTrainer 在调用这里之前已排除 padding，并负责反向传播和梯度裁剪。
    '''
    return torch.mean((action_prediction - action_target) ** 2)


def create_trainer(model, sampler, evaluation_functions, config):
    '''
    将训练相关组件集中组装。不同模型使用各自原有的 Trainer 子类。
    学习率按参数更新次数线性预热，达到基础学习率后保持不变。
    '''
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config['warmup_steps'], 1),
    )
    trainer_class = SequenceTrainer if config['model_type'] == 'dt' else ActTrainer
    return trainer_class(
        model=model,
        optimizer=optimizer,
        batch_size=config['batch_size'],
        get_batch=sampler.get_batch,
        scheduler=scheduler,
        loss_fn=action_prediction_loss,
        eval_fns=evaluation_functions,
    )


def save_iteration(run_directory, iteration, metrics, trainer, dataset, settings, config):
    '''
    每轮追加一行指标，并保存最新 checkpoint，字段名与原版一致。
    先写临时文件再替换正式模型；目前仍没有命令行断点续训入口。
    '''
    with (run_directory / 'metrics.jsonl').open('a') as metrics_file:
        metrics_file.write(json.dumps({'iteration': iteration, **metrics}) + '\n')
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'variant': config,
        'state_mean': dataset.state_mean,
        'state_std': dataset.state_std,
        'scale': settings.return_scale,
        'max_ep_len': settings.max_episode_length,
    }
    torch.save(checkpoint, run_directory / 'checkpoint.tmp')
    (run_directory / 'checkpoint.tmp').replace(run_directory / 'checkpoint.pt')


def experiment(exp_prefix, variant):
    '''
    实验总流程：先从这里阅读，再按需进入各个辅助函数。
    保留原版 experiment(exp_prefix, variant) 调用方式，内部把 variant 称为 config。
    随机实验名称仍在创建环境之前生成，避免改变后续 Python 随机采样的序列。
    '''
    config = variant
    swanlab_logger = None
    if config.get('log_to_swanlab', False):
        import swanlab
        swanlab_logger = swanlab

    configure_randomness(config['seed'], config.get('device', 'cuda'), config['num_threads'])
    run_directory = create_run_directory(config)
    group_name = f"{exp_prefix}-{config['env']}-{config['dataset']}"
    experiment_name = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    env, settings = create_environment(config['env'], config['model_type'], config['seed'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = prepare_dataset(config, state_dim, action_dim)
    sampler = TrajectoryBatchSampler(
        dataset, state_dim, action_dim, config['K'], settings, config.get('device', 'cuda'),
    )
    evaluation_functions = [
        make_evaluation_function(env, settings, dataset, config, target)
        for target in settings.target_returns
    ]
    model = create_model(config, state_dim, action_dim, settings)
    trainer = create_trainer(model, sampler, evaluation_functions, config)

    if swanlab_logger is not None:
        swanlab_logger.init(
            experiment_name=experiment_name, project='decision-transformer', config=config,
            mode=config.get('swanlab_mode', 'offline'),
        )

    '''
    一轮是 num_steps_per_iter 次梯度更新，加上所有目标的评估，然后保存结果。
    因为随机有放回抽样，这里的轮次并不表示把整个数据集遍历一遍。
    '''
    for iteration in range(1, config['max_iters'] + 1):
        metrics = trainer.train_iteration(
            num_steps=config['num_steps_per_iter'], iter_num=iteration, print_logs=True,
        )
        save_iteration(run_directory, iteration, metrics, trainer, dataset, settings, config)
        if swanlab_logger is not None:
            swanlab_logger.log(metrics)
    if swanlab_logger is not None:
        swanlab_logger.finish()
    env.close()


def build_argument_parser():
    '''命令行参数名、别名、类型和默认值均与 experiment.py 一致，分组仅方便阅读。'''
    parser = argparse.ArgumentParser()

    runtime = parser.add_argument_group('运行和记录')
    runtime.add_argument('--seed', type=int, default=0)
    runtime.add_argument('--num_threads', type=int, default=4)
    runtime.add_argument('--output_dir', type=str, default='runs')
    runtime.add_argument('--device', type=str, default='cuda')
    runtime.add_argument('--log_to_swanlab', '-w', action='store_true')
    runtime.add_argument('--swanlab_mode', choices=['offline', 'online'], default='offline')

    data = parser.add_argument_group('环境和离线数据')
    data.add_argument('--env', type=str, default='hopper')
    data.add_argument('--dataset', type=str, default='medium')
    data.add_argument('--mode', type=str, default='normal')
    data.add_argument('--K', type=int, default=20)
    data.add_argument('--pct_traj', type=float, default=1.)
    data.add_argument('--batch_size', type=int, default=64)

    architecture = parser.add_argument_group('模型结构')
    architecture.add_argument('--model_type', type=str, default='dt')
    architecture.add_argument('--embed_dim', type=int, default=128)
    architecture.add_argument('--n_layer', type=int, default=3)
    architecture.add_argument('--n_head', type=int, default=1)
    architecture.add_argument('--activation_function', type=str, default='relu')
    architecture.add_argument('--dropout', type=float, default=0.1)

    optimization = parser.add_argument_group('优化和实验规模')
    optimization.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    optimization.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    optimization.add_argument('--warmup_steps', type=int, default=10000)
    optimization.add_argument('--num_eval_episodes', type=int, default=100)
    optimization.add_argument('--max_iters', type=int, default=10)
    optimization.add_argument('--num_steps_per_iter', type=int, default=10000)
    return parser


if __name__ == '__main__':
    experiment('gym-experiment', variant=vars(build_argument_parser().parse_args()))
