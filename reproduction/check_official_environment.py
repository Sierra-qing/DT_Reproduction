'''
检查官方依赖环境，不修改原始 DT 源码，不回退到 CPU，也不执行完整训练。
在 decision-transformer-gym 环境中运行，可用 --check 分阶段定位问题。
'''
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

EXPECTED = {
    'gym': '0.18.3', 'mujoco-py': '2.0.2.13', 'numpy': '1.20.3',
    'torch': '1.8.1', 'transformers': '4.5.1', 'swanlab': '0.7.20',
}


def check_packages():
    print('Python:', sys.version, flush=True)
    assert sys.version_info[:3] == (3, 8, 5), 'Python version differs from official configuration'
    for name, expected in EXPECTED.items():
        installed = importlib.metadata.version(name)
        print(f'{name}: {installed} (expected {expected})', flush=True)
        assert installed.split('+')[0] == expected, name


def check_mujoco():
    import gym
    import mujoco_py
    engine_version = mujoco_py.functions.mj_version()
    print('MuJoCo engine version number:', engine_version, flush=True)
    assert engine_version == 200, 'Unexpected MuJoCo engine version'
    env = gym.make('Hopper-v3')
    env.seed(0)
    observation = env.reset()
    assert observation.shape == (11,)
    for _ in range(10):
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            observation = env.reset()
    env.close()
    print('Hopper-v3 reset/step: PASS', flush=True)


def check_cuda():
    import torch
    print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
    assert torch.__version__ == '1.8.1+cu111', 'Expected the user-approved CUDA 11.1 build'
    assert torch.version.cuda == '11.1', 'Unexpected CUDA runtime build'
    assert torch.cuda.is_available(), 'CUDA unavailable; do not switch to CPU'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Compiled architectures:', torch.cuda.get_arch_list(), flush=True)
    print('CUDA elementwise result:', (torch.ones(4, device='cuda') * 2).sum().item(), flush=True)
    matrix = torch.randn(128, 128, device='cuda', requires_grad=True)
    loss = (matrix @ matrix.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(matrix.grad).all().item()
    print('CUDA matrix multiplication/backward: PASS', flush=True)


def check_model():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gym'))
    from decision_transformer.models.decision_transformer import DecisionTransformer
    model = DecisionTransformer(
        state_dim=11, act_dim=3, max_length=20, max_ep_len=1000, hidden_size=128,
        n_layer=3, n_head=1, n_inner=512, activation_function='relu', n_positions=1024,
        resid_pdrop=0.1, attn_pdrop=0.1,
    ).to('cuda')
    states = torch.randn(2, 20, 11, device='cuda')
    actions = torch.randn(2, 20, 3, device='cuda')
    rewards = torch.randn(2, 20, 1, device='cuda')
    returns_to_go = torch.randn(2, 20, 1, device='cuda')
    timesteps = torch.arange(20, device='cuda').repeat(2, 1)
    mask = torch.ones(2, 20, device='cuda')
    _, predictions, _ = model(states, actions, rewards, returns_to_go, timesteps, attention_mask=mask)
    loss = (predictions - actions).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert predictions.shape == (2, 20, 3)
    assert torch.isfinite(loss).item()
    print('Original DT model CUDA forward/backward: PASS; test loss=', loss.item(), flush=True)


def check_d4rl():
    import gym
    import d4rl
    import d4rl.gym_mujoco
    import mujoco_py
    print('D4RL:', importlib.metadata.version('D4RL'), flush=True)
    env = gym.make('hopper-medium-v2')
    observation = env.reset()
    assert observation.shape == (11,)
    assert mujoco_py.functions.mj_version() == 200
    reference_min = env.ref_min_score
    reference_max = env.ref_max_score
    assert reference_max > reference_min
    assert abs(env.get_normalized_score(reference_min)) < 1e-6
    assert abs(env.get_normalized_score(reference_max) - 1.) < 1e-6
    env.close()
    print('D4RL Hopper registry and normalized score: PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', choices=['packages', 'mujoco', 'cuda', 'model', 'd4rl', 'all'], default='all')
    args = parser.parse_args()
    checks = {'packages': check_packages, 'mujoco': check_mujoco, 'cuda': check_cuda, 'model': check_model, 'd4rl': check_d4rl}
    for name, check in checks.items():
        if args.check in (name, 'all'):
            print(f'\nCHECK {name}', flush=True)
            check()
