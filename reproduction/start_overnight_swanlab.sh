#!/usr/bin/env bash
set -eo pipefail
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
if tmux has-session -t dt-swanlab-night 2>/dev/null; then
  echo 'tmux 会话 dt-swanlab-night 已存在；请先查看，避免重复启动。'
  exit 1
fi
python -c "from swanlab.package import has_api_key; assert has_api_key(), '请先执行 swanlab login，再运行此脚本'"
tmux new-session -d -s dt-swanlab-night -c /home/ubuntu/weight/decision-transformer-master \
  'bash -lc "source reproduction/activate_official.sh; python -u reproduction/run_overnight_swanlab.py; result=\$?; echo Batch_exit_code=\$result; exec bash"'
echo '已创建 tmux 会话。查看：tmux attach -t dt-swanlab-night'
echo '离开会话但保留运行：先按 Ctrl+b，松开后按 d。'
