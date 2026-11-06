#!/usr/bin/env bash
set -eo pipefail
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
if tmux has-session -t dt-locomotion-next 2>/dev/null; then
 echo 'dt-locomotion-next 已存在，不重复排队';exit 1
fi
tmux new-session -d -s dt-locomotion-next -c /home/ubuntu/weight/decision-transformer-master \
 'bash -lc "source reproduction/activate_official.sh; python -u experiments/followup_locomotion.py; result=\$?; echo Followup_exit_code=\$result; exec bash"'
echo '已排队：等待现有lane1完成后验证并接续HalfCheetah/Walker，不增加当前GPU并发数。'
