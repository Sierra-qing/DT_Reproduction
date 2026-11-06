'''
用户已授权的硬件兼容调整：保留 PyTorch 1.8.1，仅更换为官方 CUDA 11.1 wheel。
不安装 torchvision/torchaudio，不改动原版源码和其余包。
'''
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'reproduction' / 'setup_logs'
PYTHON = '/home/ubuntu/miniconda3/envs/decision-transformer-gym/bin/python'
COMMAND = [
    PYTHON, '-u', '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
    'torch==1.8.1+cu111', '-f', 'https://download.pytorch.org/whl/torch_stable.html',
]

if __name__ == '__main__':
    LOGS.mkdir(parents=True, exist_ok=True)
    status_file = LOGS / 'cuda_compatibility_install_status.json'
    started = datetime.now(timezone.utc).isoformat()
    status_file.write_text(json.dumps({'state': 'running', 'started': started, 'command': COMMAND}, indent=2))
    environment = os.environ.copy()
    environment['PIP_DEFAULT_TIMEOUT'] = '120'
    with (LOGS / 'cuda_compatibility_install.log').open('w') as log:
        result = subprocess.run(COMMAND, stdout=log, stderr=subprocess.STDOUT, env=environment)
    status_file.write_text(json.dumps({
        'state': 'finished', 'started': started,
        'finished': datetime.now(timezone.utc).isoformat(), 'returncode': result.returncode,
        'command': COMMAND,
    }, indent=2))
