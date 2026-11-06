'''
随时把已保存的模型放入原版仿真系统评估，并保存真实回合轨迹及首回合动画。
独立进程运行，不更新参数、不把评估数据加入训练集。
'''
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import gym
import numpy as np
import torch

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from experiment_observed import reference_scores
from experiment_trace import EpisodeTrace, attach_environment_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--target', type=float)
    parser.add_argument('--episodes', type=int, default=1, help='1回合用于演示；统计效果需多个回合')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes 必须大于0')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA 不可用；不会自动改用 CPU')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['variant']
    env_ids = {'hopper': 'Hopper-v3', 'halfcheetah': 'HalfCheetah-v3', 'walker2d': 'Walker2d-v3'}
    if config['env'] not in env_ids:
        parser.error('此独立评估入口目前支持三个官方 MuJoCo v3 任务')
    targets = {'hopper': 3600, 'halfcheetah': 6000, 'walker2d': 5000}
    target = args.target if args.target is not None else targets[config['env']]
    if float(target).is_integer():
        target = int(target)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    env = gym.make(env_ids[config['env']])
    env.seed(args.seed); env.action_space.seed(args.seed)
    common = dict(state_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                  max_length=config['K'], hidden_size=config['embed_dim'], n_layer=config['n_layer'])
    if config['model_type'] == 'dt':
        model = DecisionTransformer(**common, max_ep_len=checkpoint['max_ep_len'], n_head=config['n_head'],
            n_inner=4*config['embed_dim'], activation_function=config['activation_function'], n_positions=1024,
            resid_pdrop=config['dropout'], attn_pdrop=config['dropout'])
    else:
        model = MLPBCModel(**common)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device); model.eval()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    checkpoint_root = args.checkpoint.resolve().parent
    if checkpoint_root.name == 'checkpoints':
        checkpoint_root = checkpoint_root.parent
    output = args.output_dir or checkpoint_root / 'evaluations' / f'{stamp}-seed{args.seed}-target{target:g}'
    output.mkdir(parents=True, exist_ok=False)
    files, writers = {}, {}
    def csv_row(name, values):
        if name not in files:
            files[name] = (output / name).open('w', newline='')
            writers[name] = csv.DictWriter(files[name], fieldnames=list(values))
            writers[name].writeheader()
        writers[name].writerow(values); files[name].flush()
    observer = SimpleNamespace(
        run_dir=output, iteration=checkpoint['iteration'], global_step=checkpoint.get('global_step', 0),
        active_target=target, active_trace=None, checkpoint_reference=str(args.checkpoint.resolve()),
        args=SimpleNamespace(**{**config, 'seed':args.seed}), csv_row=csv_row,
        dataset={'state_dim':env.observation_space.shape[0], 'scale':checkpoint['scale'],
                 'state_mean':checkpoint['state_mean'], 'state_std':checkpoint['state_std']},
    )
    env = attach_environment_trace(env, observer)
    returns, lengths = [], []
    print('Independent evaluation:', output, flush=True)
    try:
        for episode in range(1,args.episodes+1):
            trace = EpisodeTrace(observer,episode)
            observer.active_trace = trace
            complete,error=False,None
            try:
                with torch.no_grad(), patch.object(model,'get_action',trace.get_action(model.get_action)):
                    kwargs=dict(max_ep_len=checkpoint['max_ep_len'],device=args.device,target_return=target/checkpoint['scale'],
                                mode=config.get('mode','normal'),state_mean=checkpoint['state_mean'],state_std=checkpoint['state_std'])
                    if config['model_type']=='dt':
                        ret,length=evaluate_episode_rtg(env,env.observation_space.shape[0],env.action_space.shape[0],model,
                                                        scale=checkpoint['scale'],**kwargs)
                    else:
                        ret,length=evaluate_episode(env,env.observation_space.shape[0],env.action_space.shape[0],model,**kwargs)
                complete=True
            except BaseException as failure:
                error=type(failure).__name__+': '+str(failure)
                raise
            finally:
                observer.active_trace=None
                trace.finish(complete,error)
            returns.append(float(ret)); lengths.append(int(length))
            print(f'Episode {episode}/{args.episodes}: return={ret:.2f}, length={length}',flush=True)
            if episode==1 and not args.no_video:
                subprocess.run([sys.executable,str(Path(__file__).with_name('render_trajectory.py')),str(trace.path)],check=True)
        refs=reference_scores(config['env'],config['dataset'])
        summary={
            'checkpoint':str(args.checkpoint.resolve()), 'training_global_step':observer.global_step,
            'target_return':target,'evaluation_seed':args.seed,'episodes':args.episodes,
            'return_mean':float(np.mean(returns)),'return_std':float(np.std(returns)),
            'length_mean':float(np.mean(lengths)), 'note':'独立诊断评估；单回合动画不能证明策略稳定性',
        }
        if refs is not None:
            low,high=refs
            summary['d4rl_score_mean']=100*(summary['return_mean']-low)/(high-low)
            summary['d4rl_score_std']=100*summary['return_std']/(high-low)
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    finally:
        env.close()
        for handle in files.values(): handle.close()


if __name__=='__main__':
    main()
