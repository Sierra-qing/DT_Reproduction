'''
等待 CUDA wheel 安装完成，检查环境，再运行原版入口的 10 步流程测试。
测试输出和退出码单独保存；这不是论文正式实验，不修改原始入口。
'''
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'reproduction/setup_logs'
PYTHON = '/home/ubuntu/miniconda3/envs/decision-transformer-gym/bin/python'

if __name__ == '__main__':
    status_file = LOGS / 'validation_status.json'
    started = datetime.now(timezone.utc).isoformat()
    status_file.write_text(json.dumps({'state': 'waiting_for_install', 'started': started}, indent=2))
    while True:
        installation = json.loads((LOGS / 'cuda_compatibility_install_status.json').read_text())
        if installation['state'] == 'finished':
            break
        time.sleep(5)
    if installation['returncode'] != 0:
        status_file.write_text(json.dumps({'state': 'finished', 'stage': 'install_failed', 'returncode': installation['returncode']}, indent=2))
        raise SystemExit(installation['returncode'])

    environment = os.environ.copy()
    environment['PATH'] = str(Path(PYTHON).parent) + os.pathsep + environment.get('PATH', '')
    environment['LD_LIBRARY_PATH'] = '/home/ubuntu/.mujoco/mujoco200/bin' + (
        os.pathsep + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else ''
    )
    commands = [
        ('preflight', [PYTHON, '-u', str(ROOT / 'reproduction/check_official_environment.py'), '--check', 'all'], ROOT),
        ('smoke_test', [
            PYTHON, '-u', 'experiment.py', '--env', 'hopper', '--dataset', 'medium', '--model_type', 'dt',
            '--device', 'cuda', '--warmup_steps', '100000',
            '--max_iters', '1', '--num_steps_per_iter', '10', '--num_eval_episodes', '1',
        ], ROOT / 'gym'),
    ]
    results = []
    for stage, command, cwd in commands:
        status_file.write_text(json.dumps({'state': 'running', 'stage': stage, 'started': started, 'command': command}, indent=2))
        with (LOGS / (stage + '_cu111.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=cwd)
        results.append({'stage': stage, 'returncode': result.returncode, 'command': command})
        if result.returncode:
            break
    with (LOGS / 'installed_packages_freeze.txt').open('w') as lockfile:
        subprocess.run([PYTHON, '-m', 'pip', 'freeze'], stdout=lockfile, stderr=subprocess.STDOUT, env=environment)
    status_file.write_text(json.dumps({
        'state': 'finished', 'started': started, 'finished': datetime.now(timezone.utc).isoformat(),
        'returncode': result.returncode, 'results': results,
    }, indent=2))
