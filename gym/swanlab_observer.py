'''
SwanLab 观测适配器：记录同一实验的标量、诊断图和仿真 GIF。
训练指标横轴为参数更新数，episode 指标横轴为该目标累计评估回合数。
'''
from pathlib import Path


def display_name(args, run_dir):
    '''网页名称只保留辨识实验所需字段；本地目录和稳定实验ID保持原样。'''
    env = getattr(args, 'env', None)
    if env is not None:
        task = {'hopper': 'Hopper', 'halfcheetah': 'Cheetah', 'walker2d': 'Walker',
                'reacher2d': 'Reacher'}.get(env, env)
        dataset = {'medium': 'Medium', 'medium-replay': 'Replay',
                   'medium-expert': 'Mix'}.get(args.dataset, args.dataset)
        model = args.model_type.upper()
        if args.model_type == 'bc' and args.pct_traj < 1:
            model = f'BC{args.pct_traj * 100:g}%'
        mode = '-Delayed' if args.mode == 'delayed' else ''
        seed = f's{args.seed}' if args.seed is not None else 'seed-unset'
        return f'{task}-{model}-{dataset}{mode}-{seed}'
    game = getattr(args, 'game', None)
    if game is not None:
        model = 'DT' if args.model_type == 'reward_conditioned' else f'BC{args.pct}%'
        return f'{game}-{model}-K{args.context_length}-s{args.seed}'
    return getattr(args, 'job_id', None) or Path(run_dir).name


class SwanLabObserver:
    def __init__(self, args, run_dir):
        import swanlab
        self.sdk = swanlab
        self.folder = Path(run_dir) / 'swanlog'
        name = display_name(args, run_dir)
        self.run = swanlab.init(
            project=args.swanlab_project, workspace=args.swanlab_workspace,
            experiment_name=name, logdir=str(self.folder),
            config={key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()},
            mode=args.swanlab_mode,
        )
        (Path(run_dir) / 'swanlab_location.txt').write_text(
            'mode=' + args.swanlab_mode + '\ndisplay_name=' + name + '\nlocal_dir=' + str(self.folder) + '\n')

    def scalar(self, tag, value, update, axis_step):
        '''SwanLab 按指标维护 step；两个目标的回合轴可以分别累计。'''
        self.sdk.log({tag: float(value)}, step=int(axis_step))

    def media(self, tag, path, update, video=False):
        value = self.sdk.Video(str(path)) if video else self.sdk.Image(str(path))
        self.sdk.log({tag: value}, step=int(update))

    def flush(self):
        '''log 由 SDK 管理持久化，finish 等待剩余记录完成。'''

    def close(self):
        self.sdk.finish()
