#!/usr/bin/env bash
# 仅供已准备好独立Atari环境和磁盘空间后手动使用；本次未自动下载。
set -eo pipefail
cd /home/ubuntu/weight/decision-transformer-master
if ! command -v gsutil >/dev/null 2>&1; then
  echo '缺少gsutil；请先按Atari官方README准备下载工具，不向Gym环境自动安装。'
  exit 1
fi
game=${1:?请指定Breakout、Qbert、Pong或Seaquest}
case "$game" in Breakout|Qbert|Pong|Seaquest) ;; *) echo '未知Atari游戏';exit 1;; esac
mkdir -p atari/dqn_replay
gsutil -m cp -R "gs://atari-replay-datasets/dqn/$game" atari/dqn_replay/
