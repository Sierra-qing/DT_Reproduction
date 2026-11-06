'''
本文件是 Gym 实验的总入口：读取配置 → 加载离线轨迹 → 抽取训练序列 → 创建模型 → 监督学习动作 → 在环境中评估 → 保存日志和模型。
训练数据来自已有数据集；环境交互发生在评估阶段，评估轨迹不会加入训练数据。
建议阅读顺序：experiment() 中的数据加载 → get_batch() → 模型/训练器 → eval_episodes()。
配合阅读：models/decision_transformer.py、training/seq_trainer.py、
training/trainer.py、evaluation/evaluate_episodes.py（均在 decision_transformer/ 下）。
默认从 gym 目录运行：python experiment.py --env hopper --dataset medium --model_type dt
'''

'''
创建环境；
本脚本使用 Gymnasium 的 reset/step 接口
'''
import gymnasium as gym
import numpy as np
import torch

'''
可选的在线实验记录工具；
启用 --log_to_wandb 时才导入
'''
wandb = None

import argparse
import pickle
import random
import sys
import json
import time
import unicodedata
from pathlib import Path
from datetime import datetime

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


class ExperimentTables:
    '''
    只使用标准库输出文本表格，并同步保存到 console.log，无须安装表格依赖。
    中文按终端显示宽度对齐；长路径自动换行。切换表格时才重新打印表头。
    不使用覆盖屏幕或颜色控制符，因此 tmux、重定向和日志文件中都能清楚阅读。
    '''
    def __init__(self, log_path):
        self.log_path = log_path
        self.current_table = None
        self.border = None

    @staticmethod
    def display_width(text):
        return sum(
            0 if unicodedata.combining(char) else
            2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
            for char in str(text)
        )

    @classmethod
    def wrap_cell(cls, value, width):
        lines = []
        for paragraph in str(value).split('\n'):
            current = ''
            for char in paragraph:
                if current and cls.display_width(current + char) > width:
                    lines.append(current)
                    current = ''
                current += char
            lines.append(current)
        return lines

    def emit(self, line):
        print(line, flush=True)
        with self.log_path.open('a') as log_file:
            log_file.write(line + '\n')

    def close(self):
        if self.current_table is not None:
            self.emit(self.border)
        self.current_table = None

    def row(self, key, title, headers, values, widths):
        if self.current_table != key:
            self.close()
            self.emit('')
            self.emit(title)
            self.border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'
            self.emit(self.border)
            self.write_cells(headers, widths)
            self.emit(self.border)
            self.current_table = key
        self.write_cells(values, widths)

    def write_cells(self, values, widths):
        cells = [self.wrap_cell(value, width) for value, width in zip(values, widths)]
        for line_index in range(max(map(len, cells))):
            parts = []
            for cell, width in zip(cells, widths):
                text = cell[line_index] if line_index < len(cell) else ''
                parts.append(' ' + text + ' ' * (width - self.display_width(text)) + ' ')
            self.emit('|' + '|'.join(parts) + '|')


def format_duration(seconds):
    '''短耗时显示秒，长耗时显示分秒或时分秒；机器可读日志中始终保留原始秒数。'''
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes}m{seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m{seconds:02d}s'


'''
计算每个时刻起直到轨迹末尾的折扣累计奖励：
G_t = r_t + gamma * G_{t+1}，最后一步 G_{T-1} = r_{T-1}。
逆序递推只需遍历一次。这里传入 gamma=1，因此 RTG（return-to-go）是不折扣的未来奖励和。

例：奖励 [1, 2, 3] 对应 RTG [6, 5, 3]；它包含当前步奖励。
'''
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    '''
    一、实验配置和运行记录
    variant 是命令行参数转成的字典，例如 variant['K'] 默认为 20。
    exp_prefix 用于组织实验名称；运行结果目录另外由环境、模型、种子和时间命名。
    '''
    experiment_started = time.perf_counter()
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    if log_to_wandb:
        global wandb
        import wandb
    '''
    随机性来自：序列采样、模型初始化、dropout、环境初始状态等。
    同时设置各随机数生成器，便于重复实验；仍不保证 GPU 计算逐位一致。
    '''
    seed = variant['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    '''
    明确要求 CUDA 时提前检查；如果不可用就报错，不自动改用 CPU。
    '''
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check GPU access and the PyTorch installation.')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    '''
    这是 PyTorch 的 CPU 运算线程数，不是 GPU 数量或 CUDA 线程数。
    '''
    torch.set_num_threads(variant['num_threads'])
    '''
    每次启动新建一个目录，防止不同实验的结果混在一起。
    output_dir 为相对路径时，相对于执行命令时的工作目录。
    '''
    run_dir = Path(variant['output_dir']) / (
        f"{variant['env']}-{variant['dataset']}-{variant['model_type']}-seed{seed}-"
        + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'config.json').write_text(json.dumps(variant, indent=2))
    tables = ExperimentTables(run_dir / 'console.log')

    '''
    耗时使用单调时钟 perf_counter；每条进度立即打印，并追加到 timing.jsonl。
    elapsed_seconds 是本次实验从启动至今的墙钟时间，所有 duration_seconds 单位均为秒。
    不在每步额外同步 CUDA，以免计时改变训练速度；这些耗时不是纯 GPU kernel 时间。
    '''
    def log_timing(event, message, **measurements):
        record = {
            'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
            'event': event,
            'elapsed_seconds': time.perf_counter() - experiment_started,
            **measurements,
        }
        record['message'] = message
        render_timing(record)
        with (run_dir / 'timing.jsonl').open('a') as timing_file:
            timing_file.write(json.dumps(record) + '\n')

    def render_timing(record):
        '''
        将不同事件映射到各自的固定列。终端侧重易读，timing.jsonl 保留全部详细字段。
        训练表的 LR 是刚完成的参数更新所使用的学习率；采样百分比按本日志区间计算。
        '''
        event = record['event']
        clock = record['timestamp'][11:19]
        if event == 'training_progress':
            tables.row(
                ('training', record['iteration']), '训练进度（loss、速度、采样占比均按最近日志区间统计）',
                ['时间', '轮次', '更新步数', '平均loss', '本步LR', '步/s', '采样%', '已耗时', '训练ETA', '状态'],
                [clock, record['iteration'], f"{record['completed_steps']}/{variant['num_steps_per_iter']}",
                 f"{record['interval_loss_mean']:.5f}", f"{record['learning_rate']:.2e}",
                 f"{record['interval_steps_per_second']:.2f}", f"{record['interval_sampling_percent']:.1f}",
                 format_duration(record['duration_seconds']),
                 format_duration(record['estimated_remaining_training_seconds']),
                 '正常' if record['loss_is_finite'] else '异常'],
                [8, 4, 11, 10, 9, 7, 6, 8, 9, 4],
            )
        elif event == 'evaluation_start':
            tables.row(
                ('evaluation', record['iteration'], record['target_return']), '评估进度（原始回报，± 表示标准差）',
                ['时间', '轮次', '目标回报', '回合', '回报均值±标准差', '平均步数', '耗时'],
                [clock, record['iteration'], record['target_return'], f"0/{num_eval_episodes}", '-', '-', '-'],
                [8, 4, 8, 9, 22, 8, 9],
            )
        elif event == 'evaluation_progress':
            tables.row(
                ('evaluation', record['iteration'], record['target_return']), '评估进度（原始回报，± 表示标准差）',
                ['时间', '轮次', '目标回报', '回合', '回报均值±标准差', '平均步数', '耗时'],
                [clock, record['iteration'], record['target_return'],
                 f"{record['completed_episodes']}/{num_eval_episodes}",
                 f"{record['return_mean']:.2f} ± {record['return_std']:.2f}",
                 f"{record['length_mean']:.1f}", format_duration(record['duration_seconds'])],
                [8, 4, 8, 9, 22, 8, 9],
            )
        elif event == 'evaluation_done':
            tables.close()
        elif event == 'iteration_done':
            tables.row(
                ('summary', record['iteration']), '本轮汇总（采样耗时已包含在训练耗时内，不能重复相加）',
                ['轮次', '训练', '其中采样', '评估', '保存', '本轮总计', '显存MiB', '峰值MiB'],
                [record['iteration'], format_duration(record['training_seconds']),
                 format_duration(record['sampling_seconds']), format_duration(record['evaluation_seconds']),
                 format_duration(record['saving_seconds']), format_duration(record['duration_seconds']),
                 record['gpu_memory_mib'], record['gpu_peak_memory_mib']],
                [4, 9, 9, 9, 8, 10, 9, 9],
            )
            tables.row('checkpoint', '模型保存', ['项目', '内容'],
                       ['最新模型', run_dir / 'checkpoint.pt'], [16, 76])
            tables.close()
        else:
            duration = format_duration(record['duration_seconds']) if 'duration_seconds' in record else '-'
            tables.row('stages', '阶段状态', ['时间', '累计耗时', '阶段耗时', '说明'],
                       [clock, format_duration(record['elapsed_seconds']), duration, record['message']],
                       [8, 10, 10, 60])

    runtime_info = [
        ('结果目录', run_dir), ('Python', sys.version.split()[0]), ('Python路径', sys.executable),
        ('PyTorch', torch.__version__), ('CUDA构建版本', torch.version.cuda or '无'),
        ('请求设备', device), ('随机种子', seed),
        ('环境 / 数据', f"{variant['env']} / {variant['dataset']}"),
        ('模型 / K / batch', f"{variant['model_type']} / {variant['K']} / {variant['batch_size']}"),
        ('更新规模', f"{variant['max_iters']} 轮，每轮 {variant['num_steps_per_iter']} 步"),
        ('评估规模', f"每个目标每轮 {variant['num_eval_episodes']} 回合"),
        ('计时说明', '墙钟耗时；采样包含数据处理和传输，不代表纯GPU计算耗时'),
    ]
    for item, value in runtime_info:
        tables.row('runtime', '实验启动信息', ['项目', '内容'], [item, value], [18, 74])
    tables.close()

    log_interval = variant.get('log_interval', 100)
    stage_started = time.perf_counter()
    log_timing('environment_start', '开始创建评估环境')


    '''
    二、创建评估环境，定义轨迹上限、目标回报和回报缩放系数
    '''
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    '''
    这些 MuJoCo 环境用于测试策略；训练样本仍从后面的 pkl 数据集读取。
    v4 是当前适配环境，与原始 D4RL 的旧模拟器环境存在差异，比较论文分数时需记录。
    env_targets 是评估时输入模型的期望总回报，不是模型实际已经取得的回报。
    scale 用于缩小 RTG 数值，例如 3600 / 1000 = 3.6；不是 D4RL 归一化分数。
    '''
    if env_name == 'hopper':
        env = gym.make('Hopper-v4')
        max_ep_len = 1000

        '''
        两种期望总回报，分别评估
        '''
        env_targets = [3600, 1800]

        '''
        RTG 的缩放系数；本文件中的原始奖励 r 不做此缩放
        '''
        scale = 1000.
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v4')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v4')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        '''
        自定义 Reacher 是仓库保留的旧分支；其环境 API 需单独核对，当前适配重点是 Hopper。
        '''
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    '''
    初始化环境随机数。
    评估中继续使用该随机数序列，各回合并非强制从同一状态出发。
    '''
    env.reset(seed=seed)
    env.action_space.seed(seed)

    if model_type == 'bc':

        '''
        BC 不使用 RTG 条件，因此只需保留一组评估
        '''
        env_targets = env_targets[:1]

    '''
    state 在这里指智能体获得的 observation（观测），不一定是模拟器完整内部状态。
    Hopper-v4 默认 state_dim=11、act_dim=3；连续动作向量由模型预测。
    '''
    log_timing(
        'environment_done', f'环境准备完成，耗时 {time.perf_counter() - stage_started:.2f}s',
        duration_seconds=time.perf_counter() - stage_started,
    )
    stage_started = time.perf_counter()
    log_timing('dataset_start', '开始加载数据、计算统计量和选择轨迹')
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    '''
    三、加载离线数据并计算统计量
    trajectories 是 list，每一项是一个完整回合（或按终止/超时切出的轨迹）的 dict。
    一条长度为 T 的轨迹通常包含：
        observations: [T, state_dim]，
        actions: [T, act_dim]，
        rewards: [T]，
        terminals: [T]。
    也可能有 next_observations；当前 DT 的动作监督训练不需要读取它。
    下面是相对路径，因此通常要先 cd 到 gym 目录再执行脚本。
    '''
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    '''
    基本完整性检查：数据非空，且观测/动作维度与评估环境匹配。
    维度相同并不能保证两个版本环境的动力学完全一致。
    '''
    if not trajectories:
        raise ValueError('Dataset contains no trajectories')
    for trajectory in trajectories:
        if trajectory['observations'].shape[1] != state_dim or trajectory['actions'].shape[1] != act_dim:
            raise ValueError('Dataset dimensions do not match the evaluation environment')

    '''
    汇总所有轨迹的观测、长度和整条轨迹的总回报。
    returns[i] 是第 i 条轨迹的总回报；后面 rtg 是轨迹中每个时刻的剩余回报。
    两者不同：总回报是一个数，RTG 是随时间变化的序列。
    '''
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:

        '''
        延迟奖励实验：把全部奖励移到最后一步，总回报保持不变
        '''
        if mode == 'delayed':
            '''
            这里直接修改内存中的数据，不会回写原始 pkl 文件。
            '''
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    '''
    把所有轨迹的观测拼成 [数据集总步数, state_dim]，逐个观测维度计算均值和标准差。
    标准化公式：(state - state_mean) / state_std；加 1e-6 避免除零。
    注意：统计量取自完整数据集，在后面的 pct_traj 筛选之前计算，评估也使用同一组统计量。
    '''
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    dataset_summary = [
        ('轨迹数 / 总步数', f'{len(traj_lens)} / {num_timesteps}'),
        ('回报均值 / 标准差', f'{np.mean(returns):.2f} / {np.std(returns):.2f}'),
        ('回报最小 / 最大', f'{np.min(returns):.2f} / {np.max(returns):.2f}'),
        ('观测 / 动作维度', f'{state_dim} / {act_dim}'),
        ('评估目标 / RTG缩放', f'{env_targets} / {scale}'),
    ]

    '''
    四、确定训练时使用哪些轨迹
    K 是每个样本的环境步数；DT 每步有 RTG、state、action 三个 token，故 token 数为 3*K。
    batch_size 是一次参数更新使用的序列数量；不是环境回合数。
    '''
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    '''
    pct_traj 控制按回报从高到低选取的数据量，预算按时间步数计算，而非轨迹条数。
    例如 pct_traj=0.1：优先保留高回报轨迹，使总步数尽量不超过完整数据的 10%。
    至少保留回报最高的一条完整轨迹，因此预算很小时可能超过预算。
    pct_traj=1.0 时保留所有轨迹。此筛选对 DT 和 BC 都生效。
    '''
    num_timesteps = max(int(pct_traj*num_timesteps), 1)

    '''
    轨迹总回报从低到高排列的索引
    '''
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    '''
    此后 sorted_inds 只包含保留下来的轨迹在原 trajectories 列表中的索引。
    '''
    sorted_inds = sorted_inds[-num_trajectories:]

    '''
    以轨迹长度作为抽样权重：较长的轨迹被选中的概率更大。
    随后在选中轨迹内均匀抽起点，使保留数据中的每个时间步作为起点的概率相同。
    这里不再按照回报加权；回报只用于上面的轨迹筛选。
    '''
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    dataset_summary.extend([
        ('保留轨迹数 / 步数', f'{num_trajectories} / {sum(traj_lens[sorted_inds])}'),
        ('时间步预算比例', f'{pct_traj:.2%}'),
    ])
    for item, value in dataset_summary:
        tables.row('dataset', '数据集信息', ['项目', '内容'], [item, value], [22, 70])
    tables.close()
    log_timing(
        'dataset_done', f'数据准备完成，耗时 {time.perf_counter() - stage_started:.2f}s',
        duration_seconds=time.perf_counter() - stage_started,
    )

    '''
    每轮分别累计采样和 train_step 的墙钟耗时。
    采样包含 NumPy 处理及张量传输；train_step 还包含前向、反向、优化与损失取回。
    两者之差只能称为其余训练耗时，不能直接解释成纯 GPU 计算时间。
    '''
    step_timing = {
        'sampling_seconds': 0., 'step_seconds': 0., 'completed_steps': 0,
        'interval_loss_sum': 0., 'interval_loss_count': 0,
        'interval_sampling_start': 0., 'interval_started': 0.,
    }
    current_iteration = 0

    '''
    五、训练数据采样器：训练器每次更新参数时调用一次
    这是嵌套函数，可以直接读取外层的 trajectories、state_mean、device 等变量。
    函数签名的 batch_size=256 是备用默认值；训练器实际传入命令行的 batch_size（默认 64）。
    记 B=batch_size、K=max_len、S=state_dim、A=act_dim。
    '''
    def get_batch(batch_size=256, max_len=K):
        batch_started = time.perf_counter()
        '''
        有放回抽取 B 次，同一轨迹可以在一个 batch 中出现多次。
        batch_inds 是保留轨迹集合内的编号，还需用 sorted_inds 映射回原始轨迹编号。
        '''

        '''
        按轨迹长度分配抽样概率
        '''
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        '''
        先为每个样本构造 NumPy 数组，再沿 batch 维拼接并转成 PyTorch 张量。
        d 是终止标记；它随 batch 返回，但当前 SequenceTrainer 的动作损失不使用 d。
        '''
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            '''
            si 是轨迹内部的随机起点，范围为 [0, T-1]；不是每次从回合第一步训练。
            '''
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            '''
            从 si 起最多截取 K 步；如果接近轨迹末尾，实际长度 L 会小于 K。
            reshape 中前面的 1 是临时 batch 维，-1 让 NumPy 自动推断实际长度 L。
            此时 s:[1,L,S]，a:[1,L,A]，r:[1,L,1]，d:[1,L]。
            s[-1] 表示列表中刚加入的一个样本，不是该样本的最后一个时间步。
            '''
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            '''
            使用原回合中的绝对时间 si, si+1, ...，而不是把每个采样片段重新编号为 0。
            这些整数会在模型中查 timestep embedding，必须落在嵌入表允许的范围内。
            '''
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))

            '''
            将过大的索引截到 max_ep_len-1，避免嵌入表越界
            '''
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1
            '''
            先用“从 si 到原轨迹结尾”的全部奖励计算 RTG，再截取前 L+1 个值。
            不能先把奖励截成 K 步再求和，否则会漏掉片段之后的奖励。
            多留一个 RTG 是原仓库的 batch 约定；SequenceTrainer 实际用 rtg[:, :-1]
            与 K 个状态对齐，当前动作预测损失并不使用额外的末尾 RTG。
            '''
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            '''
            若片段已经到轨迹末尾，就没有额外的下一时刻 RTG，此时补 0。
            '''
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            '''
            左侧 padding：将不同实际长度 L 的样本都补成 K 步，便于组成 batch。
            例 K=5、L=3： [pad, pad, x0, x1, x2]，mask 为 [0,0,1,1,1]。
            状态先补 0 再标准化，所以补位处最终通常不是 0；靠 mask 识别补位。
            动作补 -10、终止标记补 2 是占位值，不是真实动作/终止类别。
            SequenceTrainer 计算动作 MSE 前会用 mask 排除补位，避免学习这些占位值。
            '''
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            '''
            RTG 的最终长度为 K+1，并除以 scale；r 仍保留原始奖励数值。
            '''
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            '''
            这个 mask 标识真实时间步；模型会把它扩展到每步的三个 token。
            Transformer 内部另有因果 mask，用来限制读取未来 token；两种 mask 作用不同。
            '''
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        '''
        沿 axis=0 将 B 个 [1,...] 数组合成一个 batch，并送到指定设备。
        连续数值使用 float32；终止标记和时间索引使用整数 long。
        最终形状：s [B,K,S]；a [B,K,A]；r [B,K,1]；d [B,K]；
                  rtg [B,K+1,1]；timesteps [B,K]；mask [B,K]。
        Hopper 默认 B=64、K=20 时，s 是 [64,20,11]，a 是 [64,20,3]。
        '''
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        step_timing['sampling_seconds'] += time.perf_counter() - batch_started
        return s, a, r, d, rtg, timesteps, mask

    '''
    六、构造评估函数（闭包）
    eval_episodes(3600) 返回一个“接收 model、评估目标回报 3600”的函数 fn。
    这里是创建函数，不是立即运行评估；训练器在每轮训练后再调用 fn(model)。
    '''
    def eval_episodes(target_rew):
        def fn(model):
            evaluation_started = time.perf_counter()
            log_timing(
                'evaluation_start', f'第 {current_iteration} 轮：开始评估目标 {target_rew}，共 {num_eval_episodes} 回合',
                iteration=current_iteration, target_return=target_rew,
            )
            returns, lengths = [], []
            for episode_index in range(num_eval_episodes):
                episode_started = time.perf_counter()
                '''
                评估不计算梯度、不更新参数；模型 eval() 模式由 Trainer 设置。
                '''
                with torch.no_grad():
                    '''
                    DT 根据目标 RTG、历史状态和动作逐步决策。
                    评估函数在每步后更新 RTG：下一目标 = 当前目标 - 实际奖励 / scale
                    （delayed 模式例外，保持 RTG 条件不变）。
                    '''
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                '''
                ret 是环境实际产生的原始奖励总和，不是输入的目标，也不是归一化分数。
                '''
                returns.append(ret)
                lengths.append(length)
                completed_episodes = episode_index + 1
                if completed_episodes == 1 or completed_episodes % 10 == 0 or completed_episodes == num_eval_episodes:
                    elapsed = time.perf_counter() - evaluation_started
                    log_timing(
                        'evaluation_progress',
                        f'第 {current_iteration} 轮：目标 {target_rew}，评估 {completed_episodes}/{num_eval_episodes}，'
                        f'累计 {elapsed:.2f}s，本回合 {time.perf_counter() - episode_started:.2f}s，回报 {ret:.2f}',
                        iteration=current_iteration, target_return=target_rew,
                        completed_episodes=completed_episodes, duration_seconds=elapsed,
                        return_mean=float(np.mean(returns)), return_std=float(np.std(returns)),
                        length_mean=float(np.mean(lengths)),
                    )
            '''
            汇总多次回合的回报和长度，降低只评估一次所带来的偶然性。
            '''
            elapsed = time.perf_counter() - evaluation_started
            log_timing(
                'evaluation_done', f'目标 {target_rew} 评估完成，耗时 {elapsed:.2f}s',
                iteration=current_iteration, target_return=target_rew, duration_seconds=elapsed,
            )
            return {
                f'target_{target_rew}_evaluation_seconds': elapsed,
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    '''
    七、创建策略模型
    DT 在模型内部把每个时间步排列为 (RTG_t, state_t, action_t)，使用因果注意力。
    用 state_t 对应的输出预测 action_t；因果遮罩避免该位置读取动作标签 action_t。
    '''
    stage_started = time.perf_counter()
    log_timing('model_start', '开始创建模型、优化器和训练器')
    if model_type == 'dt':

        '''
        Transformer 每层前馈网络的中间维度，设为隐藏维度的 4 倍。
        底层 GPT-2 因果注意力所用的序列容量，按 token 数计算，需容纳 3*K。
        DT 用自定义的环境时间嵌入，不能把这个参数等同于 max_ep_len。
        '''
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],

            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],


            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        '''
        BC 基线：把最近 K 个状态拼接后输入 MLP，预测最后一步的动作，不使用 RTG。
        '''
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    '''
    八、优化器和学习率调度
    AdamW 根据动作预测损失更新参数，weight_decay 为权重衰减系数。
    '''
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    '''
    warmup：学习率从较小值线性增加到设定的 learning_rate，之后保持不变。
    steps 计的是参数更新次数，不是环境步数或外层迭代次数。
    lambda 返回学习率倍率；训练器每完成一次更新，调用一次 scheduler.step()。
    '''
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    '''
    九、把模型、采样器、损失函数和评估函数交给训练器
    SequenceTrainer.train_step() 的关键流程：
    get_batch → model.forward(..., rtg[:, :-1], ...) → 按 mask 选出真实动作
    → 动作 MSE → 清梯度 → 反向传播 → 梯度裁剪 → optimizer.step()。
    本实验只监督动作，不训练状态/回报预测损失，也没有 TD 目标或策略梯度更新。
    lambda 的六个形参是训练器统一接口，实际仅使用 a_hat 和 a。
    '''
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    '''
    可选的在线日志；不启用时仍会打印日志并保存本地 metrics.jsonl。
    '''
    log_timing(
        'model_done', f'模型及训练器准备完成，耗时 {time.perf_counter() - stage_started:.2f}s',
        duration_seconds=time.perf_counter() - stage_started,
    )

    '''
    包装原 train_step，只测量并输出进度；梯度计算、优化器更新仍由原训练器执行。
    每轮第 1 步、每 log_interval 步以及最后一步输出；ETA 仅估计本轮剩余训练时间。
    原 train_step 返回损失标量时已将结果取回 CPU，计时包含对应的 CUDA 等待。
    '''
    model_device = next(model.parameters()).device
    gpu_enabled = model_device.type == 'cuda'
    model_summary = [
        ('实际模型设备', str(model_device)),
        ('GPU名称', torch.cuda.get_device_name(model_device) if gpu_enabled else '未使用GPU'),
        ('参数总数 / 可训练', f'{sum(p.numel() for p in model.parameters()):,} / '
                           f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}'),
        ('基础学习率 / 预热步数', f"{variant['learning_rate']:.2e} / {variant['warmup_steps']}"),
        ('显存计量', '仅PyTorch已分配显存；峰值每轮重置，不含其他进程和CUDA上下文'),
    ]
    for item, value in model_summary:
        tables.row('model', '模型与设备', ['项目', '内容'], [item, value], [22, 70])
    tables.close()
    original_train_step = trainer.train_step

    def timed_train_step():
        step_started = time.perf_counter()
        loss = original_train_step()
        step_timing['step_seconds'] += time.perf_counter() - step_started
        step_timing['completed_steps'] += 1
        step_timing['interval_loss_sum'] += float(loss)
        step_timing['interval_loss_count'] += 1
        completed = step_timing['completed_steps']
        total = variant['num_steps_per_iter']
        if completed == 1 or completed % log_interval == 0 or completed == total:
            elapsed = time.perf_counter() - iteration_started
            interval_elapsed = time.perf_counter() - step_timing['interval_started']
            interval_steps = step_timing['interval_loss_count']
            interval_speed = interval_steps / max(interval_elapsed, 1e-9)
            remaining = (total - completed) / max(interval_speed, 1e-9)
            interval_loss = step_timing['interval_loss_sum'] / interval_steps
            interval_sampling = step_timing['sampling_seconds'] - step_timing['interval_sampling_start']
            log_timing(
                'training_progress',
                f'第 {current_iteration} 轮：训练 {completed}/{total}，耗时 {elapsed:.2f}s，'
                f'{completed / max(elapsed, 1e-9):.2f} 步/s，预计剩余训练 {remaining:.2f}s，loss={loss:.6f}',
                iteration=current_iteration, completed_steps=completed, loss=float(loss),
                duration_seconds=elapsed, estimated_remaining_training_seconds=remaining,
                sampling_seconds=step_timing['sampling_seconds'],
                train_step_seconds=step_timing['step_seconds'],
                interval_loss_mean=interval_loss, interval_steps=interval_steps,
                interval_steps_per_second=interval_speed,
                interval_sampling_percent=100 * interval_sampling / max(interval_elapsed, 1e-9),
                learning_rate=float(optimizer.param_groups[0]['lr']),
                loss_is_finite=bool(np.isfinite(loss) and np.isfinite(interval_loss)),
            )
            step_timing.update(
                interval_loss_sum=0., interval_loss_count=0,
                interval_sampling_start=step_timing['sampling_seconds'],
                interval_started=time.perf_counter(),
            )
        return loss

    trainer.train_step = timed_train_step

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    '''
    十、主循环：一轮 = 多次参数更新 + 一次完整评估 + 保存结果
    这里的 iter 不是“遍历完整数据集一次”的 epoch，因为每个 batch 都是随机有放回抽样。
    默认 10 轮 × 每轮 10000 次更新 = 100000 次参数更新。
    默认每轮每个目标评估 100 回合；Hopper 有两个目标，即每轮共评估 200 回合。
    '''
    for iter in range(variant['max_iters']):
        current_iteration = iter + 1
        step_timing.update(sampling_seconds=0., step_seconds=0., completed_steps=0)
        if gpu_enabled:
            torch.cuda.reset_peak_memory_stats(model_device)
        iteration_started = time.perf_counter()
        step_timing.update(
            interval_loss_sum=0., interval_loss_count=0,
            interval_sampling_start=0., interval_started=iteration_started,
        )
        log_timing('iteration_start', f'开始第 {current_iteration}/{variant["max_iters"]} 轮训练', iteration=current_iteration)
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=False)
        '''
        JSONL 每行是一轮指标，包含训练损失、评估回报及时间，便于后续画学习曲线。
        '''
        outputs['time/sampling'] = step_timing['sampling_seconds']
        outputs['time/train_steps'] = step_timing['step_seconds']
        outputs['time/train_steps_excluding_sampling'] = step_timing['step_seconds'] - step_timing['sampling_seconds']
        gpu_memory_mib = round(torch.cuda.memory_allocated(model_device) / 1024**2, 1) if gpu_enabled else None
        gpu_peak_memory_mib = round(torch.cuda.max_memory_allocated(model_device) / 1024**2, 1) if gpu_enabled else None
        outputs['system/gpu_memory_mib'] = gpu_memory_mib
        outputs['system/gpu_peak_memory_mib'] = gpu_peak_memory_mib
        for name, value in [
            ('平均动作MSE', f'{outputs["training/train_loss_mean"]:.6f}'),
            ('动作MSE标准差', f'{outputs["training/train_loss_std"]:.6f}'),
            ('当前学习率', f'{optimizer.param_groups[0]["lr"]:.2e}'),
        ]:
            tables.row(('loss_summary', current_iteration), '本轮训练指标',
                       ['项目', '数值'], [name, value], [22, 30])
        save_started = time.perf_counter()
        with (run_dir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'iteration': iter + 1, **outputs}) + '\n')
        '''
        推理需要模型参数及训练时的状态归一化统计；优化器/调度器状态便于后续实现续训。
        当前脚本只实现保存，还没有从 checkpoint 恢复训练的命令行入口。
        '''
        checkpoint = {
            'iteration': iter + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'variant': variant,
            'state_mean': state_mean,
            'state_std': state_std,
            'scale': scale,
            'max_ep_len': max_ep_len,
        }
        '''
        先写临时文件，再替换正式文件；保存成功前保留上一轮的正式 checkpoint。
        checkpoint.pt 每轮覆盖，只保留最新一轮，不是每一轮都生成一个历史模型文件。
        '''
        torch.save(checkpoint, run_dir / 'checkpoint.tmp')
        (run_dir / 'checkpoint.tmp').replace(run_dir / 'checkpoint.pt')
        save_seconds = time.perf_counter() - save_started
        iteration_seconds = time.perf_counter() - iteration_started
        log_timing(
            'iteration_done',
            f'第 {current_iteration} 轮完成：训练 {outputs["time/training"]:.2f}s，'
            f'评估 {outputs["time/evaluation"]:.2f}s，保存 {save_seconds:.2f}s，'
            f'本轮总计 {iteration_seconds:.2f}s；采样累计 {step_timing["sampling_seconds"]:.2f}s',
            iteration=current_iteration, duration_seconds=iteration_seconds,
            training_seconds=outputs['time/training'], evaluation_seconds=outputs['time/evaluation'],
            saving_seconds=save_seconds, sampling_seconds=step_timing['sampling_seconds'],
            gpu_memory_mib=gpu_memory_mib if gpu_enabled else '-',
            gpu_peak_memory_mib=gpu_peak_memory_mib if gpu_enabled else '-',
        )
        if log_to_wandb:
            wandb.log(outputs)
    '''
    所有轮次完成后释放环境资源。
    '''
    trainer.train_step = original_train_step
    env.close()
    log_timing('experiment_done', f'实验结束，总耗时 {format_duration(time.perf_counter() - experiment_started)}')
    tables.close()


'''
十一、命令行入口：直接运行本文件时执行；被其他模块 import 时不自动启动实验。
下列 default 是未提供对应命令行参数时采用的值。
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    实验记录：随机种子、CPU 运算线程数、结果目录。
    '''
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100, help='每多少次参数更新输出一次耗时进度，必须大于 0')
    parser.add_argument('--output_dir', type=str, default='runs')
    '''
    数据选择：--env hopper --dataset medium 对应 data/hopper-medium-v2.pkl。
    '''
    parser.add_argument('--env', type=str, default='hopper')

    '''
    对应数据文件必须已经存在
    '''
    parser.add_argument('--dataset', type=str, default='medium')
    '''
    normal 使用原始奖励；delayed 把整条轨迹奖励移到末尾，用于延迟奖励实验。
    '''

    '''
    原始奖励 / 延迟奖励
    '''
    parser.add_argument('--mode', type=str, default='normal')
    '''
    序列采样：K 为上下文环境步数，pct_traj 为高回报轨迹的时间步预算比例。
    '''
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    '''
    模型结构：dt 为 Decision Transformer，bc 为 MLP 行为克隆基线。
    '''

    '''
    DT / 行为克隆
    '''
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    '''
    优化设置：基础学习率、权重衰减、按更新次数计的预热长度。
    '''
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    '''
    运行规模：每个目标评估多少回合、训练多少轮、每轮更新多少次参数。
    '''
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    '''
    cuda 使用 GPU；日志开关写 --log_to_wandb 或 -w 即可，不需要附加 True。
    '''
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', action='store_true')
    
    '''
    parse_args() 返回 Namespace；vars(args) 将其转成 experiment 使用的配置字典。
    '''
    args = parser.parse_args()
    if args.log_interval <= 0:
        parser.error('--log_interval 必须大于 0')

    experiment('gym-experiment', variant=vars(args))
