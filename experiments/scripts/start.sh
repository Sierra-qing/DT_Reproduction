#!/usr/bin/env bash
set -eo pipefail
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
plan=${1:-hopper-ready}
lane=${2:-1}
case "$lane" in 1|2) ;; *) echo 'lane只能是1或2'; exit 1;; esac
session=dt-paper-suite
if [ "$lane" = 2 ]; then session=dt-paper-suite-2; fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo 'dt-paper-suite 已存在；使用 tmux attach -t dt-paper-suite 查看，避免重复启动。'
  exit 1
fi
case "$plan" in hopper-ready|hopper-remainder|hopper-next|locomotion-next|gym-all|return-sweeps|atari-all|atari-sweeps|graph-reimplementation) ;; *) echo '未知计划'; exit 1;; esac
python -c "from swanlab.package import has_api_key; assert has_api_key(), '请先执行 swanlab login'"
tmux new-session -d -s "$session" -c /home/ubuntu/weight/decision-transformer-master \
  "bash -lc 'source reproduction/activate_official.sh; python -u experiments/manage.py run --plan $plan --lane $lane; echo Queue_exit_code=\$?; exec bash'"
echo "已启动 $plan；查看：tmux attach -t $session"
