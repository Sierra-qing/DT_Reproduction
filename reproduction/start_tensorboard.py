'''
启动本机 TensorBoard 和原始日志同步器，只绑定 127.0.0.1。
不启动或停止训练；Mac 通过 SSH 端口转发访问。进程及日志位置写入状态文件。
'''
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/ubuntu/miniconda3/envs/decision-transformer-gym/bin/python'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original_log', type=Path)
    parser.add_argument('--port', type=int, default=6006)
    args = parser.parse_args()
    logs = ROOT / 'reproduction/setup_logs'
    logs.mkdir(parents=True, exist_ok=True)
    status_file = logs / 'dashboard_status.json'
    if status_file.is_file():
        previous = json.loads(status_file.read_text())
        if alive(previous['tensorboard_pid']):
            print('TensorBoard already running:', previous['url'])
            raise SystemExit(0)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    logdir = ROOT / 'gym/runs_observed'
    logdir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['LD_LIBRARY_PATH'] = '/home/ubuntu/.mujoco/mujoco200/bin' + (
        os.pathsep + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else ''
    )
    watcher_pid = None
    if args.original_log is not None:
        source = args.original_log.resolve()
        if not source.is_file():
            parser.error('原始日志不存在')
        with (logs / 'original_log_watcher.log').open('a') as handle:
            watcher = subprocess.Popen([
                PYTHON, '-u', str(ROOT / 'gym/watch_original_log.py'), str(source),
                '--output_dir', str(logdir),
            ], cwd=ROOT / 'gym', env=environment, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        watcher_pid = watcher.pid
    command = [PYTHON, '-u', '-m', 'tensorboard.main', '--logdir', str(logdir),
               '--host', '127.0.0.1', '--port', str(args.port), '--reload_interval', '5', '--load_fast=false']
    with (logs / 'tensorboard_server.log').open('a') as handle:
        server = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    url = f'http://127.0.0.1:{args.port}'
    status = {'url': url, 'logdir': str(logdir), 'tensorboard_pid': server.pid,
              'watcher_pid': watcher_pid, 'command': command, 'http_ready': False}
    status_file.write_text(json.dumps(status, indent=2))
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                status['http_status'] = response.status
                status['http_ready'] = response.status == 200
            break
        except Exception:
            if server.poll() is not None:
                break
            time.sleep(.5)
    status_file.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status['http_ready']:
        raise SystemExit('TensorBoard 尚未就绪，查看 reproduction/setup_logs/tensorboard_server.log')
