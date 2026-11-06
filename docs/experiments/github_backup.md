# GitHub 与实验备份

目标仓库：https://github.com/Sierra-qing/DT_Reproduction

GitHub 保存代码、配置、环境说明、论文协议、审计记录和结果摘要。
SwanLab 保存已记录的指标与图表；它不能替代本地完整训练目录。

## 本次代码快照的范围

保留原作者许可证和固定官方版本信息。数据集、模型权重、逐步轨迹、
TensorBoard 事件、大型训练日志和正在变化的队列锁文件不随代码上传。
实验覆盖表是生成时的状态快照，不表示远端持续同步服务器上的进度。

## 换机器后继续实验

1. 克隆仓库，按环境说明建立环境；修改 `experiments/runtime.json` 中的本机 Python 路径。
2. 按数据准备说明恢复数据，核对数据来源和校验值。
3. 从独立备份恢复 `gym/runs_observed/`、`experiments/state/` 中的任务 JSON
   及 `experiments/batches/`；队列锁、lane 登记和进程 PID 不能直接沿用。
4. 核对旧记录中的绝对路径，检查 `python experiments/manage.py status`，
   确认已完成实验识别正确后，再启动剩余计划，避免重复训练。

目前 checkpoint 包含模型、优化器、调度器及随机数状态，但训练入口尚未实现
恢复训练。因此可以加载模型评估，不能直接保证从中断步数无缝续训。

## 完整结果另行备份

至少保留每个已完成实验的完整目录，包括配置、数据元信息、指标文件、
checkpoints、评估轨迹和动画，并保留队列任务记录及环境版本清单。
备份完成后生成 SHA256 清单并核验；仅保存 GitHub 代码或 SwanLab 曲线不足以恢复完整实验。
训练中的文件持续变化，应在一轮评估和 checkpoint 写入结束后制作一致快照，
或待实验完成后归档。

本次整理和上传代码不停止、重启或修改正在运行的训练。

## 从服务器带走完整文件

在服务器项目目录运行 `python -u experiments/backup_project.py`。
产物位于 `backups/日期时间/`，包括完整项目压缩包、`SHA256SUMS` 和
`summary.json`。脚本逐文件复制快照后归档，校验归档内每个项目文件，
并导出两个训练环境的 pip 版本清单。运行中的实验只保存当前部分结果，
不是事务一致的恢复训练快照。安装好的 Conda 环境二进制、项目外的 MuJoCo
和登录凭据不包含在项目压缩包中。

在 **Mac 本地终端**（不是 SSH 登录后的服务器终端）执行：

```bash
mkdir -p ~/Downloads/DT_Backup
rsync -avP ubuntu@服务器地址:/home/ubuntu/weight/decision-transformer-master/backups/日期时间/ ~/Downloads/DT_Backup/
cd ~/Downloads/DT_Backup
shasum -a 256 -c SHA256SUMS
```

将服务器地址和日期时间替换为真实值；SSH 使用非默认端口时为 rsync
增加 `-e 'ssh -p 端口'`。中断后重复 rsync 命令可继续传输。
校验通过后用 `tar -xzf DT_Reproduction-日期时间.tar.gz` 解压。
确认本地备份完整且训练全部结束前，不删除服务器上的原文件。
