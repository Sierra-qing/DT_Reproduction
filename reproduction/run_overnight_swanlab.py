'''
在 tmux 中按论文协议依次运行三个独立种子。在线登录和 CUDA 检查失败时直接退出。
每个种子固定100000次更新，不按墙钟时间加训，不按最好回报选择模型。
'''
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('种子不能重复')
    import torch
    from swanlab.package import has_api_key
    if not has_api_key():
        raise RuntimeError('请先在服务器终端执行 swanlab login；不会自动转为离线记录')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，不启动实验，不回退 CPU')
    assert (torch.ones(4, device='cuda') * 2).sum().item() == 8.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    batch = ROOT / 'reproduction/overnight_runs' / stamp
    batch.mkdir(parents=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'gym').rglob('*.py') if 'runs_' not in str(p)}
    status = {'started_utc': stamp, 'state': 'starting', 'seeds': args.seeds,
              'python': sys.executable, 'gpu': torch.cuda.get_device_name(0),
              'source_sha256': hashes, 'experiments': []}
    save(batch / 'status.json', status)
    (ROOT / 'reproduction/overnight_latest.txt').write_text(str(batch) + '\n')
    print('Overnight batch:', batch, flush=True)
    for seed in args.seeds:
        command = [sys.executable, '-u', 'experiment_observed.py',
            '--env', 'hopper', '--dataset', 'medium', '--model_type', 'dt', '--mode', 'normal',
            '--device', 'cuda', '--seed', str(seed), '--K', '20', '--pct_traj', '1.0',
            '--batch_size', '64', '--embed_dim', '128', '--n_layer', '3', '--n_head', '1',
            '--activation_function', 'relu', '--dropout', '0.1', '--learning_rate', '1e-4',
            '--weight_decay', '1e-4', '--warmup_steps', '100000', '--max_iters', '10',
            '--num_steps_per_iter', '10000', '--num_eval_episodes', '100',
            '--log_interval', '100', '--eval_log_interval', '10', '--batch_snapshot_interval', '100',
            '--run_name', 'paper-overnight-' + stamp, '--log_to_swanlab',
            '--swanlab_mode', 'online', '--swanlab_project', 'decision-transformer-reproduction']
        record = {'seed': seed, 'command': command, 'state': 'running',
                  'started_utc': datetime.now(timezone.utc).isoformat(),
                  'console_log': str(batch / f'seed_{seed}.log')}
        status['experiments'].append(record)
        status.update(state='running', active_seed=seed)
        save(batch / 'status.json', status)
        print(f'Starting seed {seed}; log: {record["console_log"]}', flush=True)
        environment = dict(os.environ, PYTHONUNBUFFERED='1')
        environment['LD_LIBRARY_PATH'] = '/home/ubuntu/.mujoco/mujoco200/bin:' + environment.get('LD_LIBRARY_PATH', '')
        with Path(record['console_log']).open('wb') as log:
            with subprocess.Popen(command, cwd=ROOT / 'gym', env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                record['pid'] = process.pid
                save(batch / 'status.json', status)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = process.wait()
        record.update(returncode=code, state='complete' if code == 0 else 'failed',
                      finished_utc=datetime.now(timezone.utc).isoformat())
        status['state'] = 'between_seeds' if code == 0 else 'failed'
        save(batch / 'status.json', status)
        if code:
            raise SystemExit(code)
    status.update(state='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    status.pop('active_seed', None)
    save(batch / 'status.json', status)
    print('All seeds complete:', batch, flush=True)


if __name__ == '__main__':
    main()
