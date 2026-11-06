# 实验观测：终端表格、实时曲线和复盘文件

## 当前原版实验已经接入面板

本次没有停止或重新启动原版训练进程。
TensorBoard 已在服务器的 http://127.0.0.1:6006 启动，只监听本机。
同步器每 5 秒读取 gym/logs/hopper-paper-20261105-090102.log，将完整的每轮指标加入面板。

在 Mac 本地终端执行（把“服务器IP或SSH别名”替换为平时连接服务器用的地址，保留原有 SSH 端口设置）：

```bash
ssh -N -L 6006:127.0.0.1:6006 ubuntu@服务器IP或SSH别名
```

保持这个 SSH 转发窗口打开，在 Mac 浏览器访问 http://127.0.0.1:6006 。
如果使用 VS Code Remote，也可以在“端口”面板转发服务器的 6006，再打开本地地址。
若 Mac 的 6006 已被占用，可改成 `-L 16006:127.0.0.1:6006`，浏览器访问本地 16006。

Scalars 查看常规曲线；Custom Scalars 将两种 RTG 条件放在一张图里比较。
第一次只有一个点是正常的：当前原版入口只有完成一整轮训练和评估后才输出。
下一轮结束后，同步器会自动增加下一个点。服务每 5 秒重读事件；页面可开启自动刷新或手动刷新。
当前导入运行名称以 imported-official- 开头，种子未知；它不是新启动的一次实验。

首轮已读取到的真实结果：

| 指标 | 值 |
|---|---:|
| 参数更新数 | 10000 |
| 训练动作 MSE 均值 | 0.17095 |
| 目标 3600 的原始回报均值 | 1325.20 |
| 对应 D4RL 评分（乘 100） | 41.34 |
| 论文 Hopper-medium DT 参考分数 | 67.6 |

这只是训练中间的一轮，不能直接判定最终是否复现成功。论文参考来自多种子结果，当前单次运行不能替代相同统计口径。

## 下一次训练使用完整观测入口

独立入口：gym/experiment_observed.py。它调用原版 experiment()、模型和 Trainer，官方源码不改动。
预热默认采用已确认的论文值 100000，其余核心默认值保持官方入口设置。
`--seed` 是显式的可重复性控制；不传时保留官方未显式设种子的行为。本次建议显式传入并在比较时记录。

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
cd gym
python -u experiment_observed.py \
  --env hopper --dataset medium --model_type dt --device cuda \
  --seed 0 --warmup_steps 100000 \
  --max_iters 10 --num_steps_per_iter 10000 --num_eval_episodes 100 \
  --log_interval 100 --eval_log_interval 10
```

这个命令会新建一次训练，不会接管当前原版进程。不要为了查看现有曲线而提前停止当前实验。
旧进程未记录的逐步指标、checkpoint 和种子不能事后补回；新入口从启动时开始记录。
已有进程继续运行，新的完整观测实验何时启动由用户决定。

终端：第 1 步、每 100 步及每轮最后一步输出训练表；评估第 1 回合、每 10 回合及最后一回合输出评估表。
曲线：loss 和学习率每次更新写入 TensorBoard，默认每 5 秒刷新事件文件；终端仍每 100 步显示区间汇总。每轮评估汇总单独记录。单回合曲线以回合数为横轴。
每次运行建立独立目录，之后用不同的 --seed 启动可以在 TensorBoard 中对比，不覆盖旧结果。

## 先看哪些图

| 图 / 标签 | 含义与读法 |
|---|---|
| train/action_mse_step | 每一次参数更新的动作 MSE，提供全量逐步曲线 |
| train/action_mse_mean | 最近日志区间的动作 MSE；看趋势，不要求单调下降 |
| train/action_mse_last | 单步误差，显示 batch 间波动 |
| train/learning_rate | 本次参数更新实际使用的学习率，论文配置下前 100000 步预热 |
| eval/target_3600/return_mean | 期望 RTG=3600 时真正拿到的原始回报 |
| eval/target_1800/return_mean | 同一模型在另一个回报条件下的表现，用于比较条件作用 |
| eval/target_3600/normalized_score_mean | D4RL 官方参考分数转换后 ×100，用于与论文分数量级比较 |
| reference/paper_hopper_medium | 67.6 的固定参考线；不代表本次实测结果 |
| episode/target_3600/return | 各个评估回合的原始回报；横轴是累计评估回合数 |
| performance/updates_per_second | 最近日志区间的参数更新速度 |
| performance/sampling_percent | 准备 batch 的墙钟耗时占比，包含 NumPy 计算和张量传输，不是 GPU 利用率 |
| batch/rtg_raw（Histograms） | 训练采样片段里的实际 RTG 分布，可对照评估时指定的目标 |
| batch/padding_fraction | 日志点采样 batch 中 padding 的比例 |
| diagnostics/gradient_norm_after_clip | DT 裁剪后的梯度范数，不能据此还原裁剪前的梯度大小 |
| system/gpu_memory_mib | 当前 PyTorch 已分配显存，不含 CUDA 上下文和其他进程 |
| time/training、time/evaluation | 每轮训练、评估耗时；采样已包含在训练时间内，不重复相加 |

建议先将平滑系数调为 0 看原始波动，再适度平滑看趋势。
loss 下降不等于策略回报必然上升。评估回合间标准差也不等于论文跨三个随机种子的标准差。
单次运行即使超过参考线，也还需要检查完整配置、数据来源及多种子统计。

## 保存的复盘文件

新入口结果：gym/runs_observed/<本次运行名>/。

| 文件 | 内容 |
|---|---|
| config.json | 原训练参数及观测参数，包括 seed 和日志间隔 |
| metadata.json | 实际依赖版本、CUDA 构建、源码哈希、D4RL 评分常量 |
| dataset.json | 数据文件 SHA-256、轨迹数、筛选情况、回报分布、状态归一化统计 |
| console.log | 终端原始输出与表格 |
| train_steps.csv | 每次参数更新的 loss、学习率和更新耗时 |
| train.csv | 每个日志区间的 loss、学习率、速度、采样占比和时间 |
| training_provenance.jsonl | 每次更新真实抽中的轨迹编号与起点，可还原所有训练输入 |
| training_batches/ | 默认每 100 次更新保留实际完整输入与动作预测，可设为每次更新 |
| trajectories/ | 全部评估回合、全部环境时间步的状态/动作/奖励/RTG/物理状态 |
| trajectory_index.csv | 回合列表与对应模型路径 |
| recording_schema.json | 数据字段的含义、时间对齐和记录范围 |
| eval_episodes.csv | 每个评估回合的目标、回报、长度和耗时 |
| metrics.jsonl | 每轮汇总，含原始回报、D4RL 评分和时间 |
| events.jsonl | 阶段、进度、异常或中断事件 |
| tensorboard/ | 实时曲线和直方图事件文件 |
| checkpoints/iteration_XXXX.pt | 每轮策略、优化器、调度器、归一化统计与随机数状态，供轨迹对应 |
| checkpoint_latest.pt | 最新模型副本，评估前先保存、轮末更新 |

每轮 checkpoint 都保留，latest 副本会覆盖，不自动挑最高分模型；目前没有断点续训命令行入口。
评估前先保存本轮策略；若评估中断，已执行的回合片段也会以 complete=false 保存。若训练中途被强制杀进程，模型仍只保证保留已保存轮次。
导入的旧原版运行只有原日志提供的每轮汇总，不会伪造逐步指标或 checkpoint。

## 已完成验证与服务管理

- 同种子小规模 GPU 对照：观测版与原版参数及优化器状态逐项完全一致。
- 真实观测入口：6 次 GPU 更新、两轮少量评估、TensorBoard 事件、CSV 与 checkpoint 保存检查通过。
- 真实 TensorBoard HTTP 页面及 scalar 接口检查通过，已读取当前原版实验的第一轮结果。
- 官方 14 个 gym 文件仍逐字节一致。
- 测试运行存放在 reproduction/observer_validation_runs，不混入正式面板目录。
- TensorBoard 2.13.0 已安装；protobuf 仍为 3.20.3，仅将 Conda 构建重装为相同版本 pip wheel，解决 C++ ABI 导入问题。

服务状态与 PID：reproduction/setup_logs/dashboard_status.json。
服务日志：reproduction/setup_logs/tensorboard_server.log。
旧日志同步器日志：reproduction/setup_logs/original_log_watcher.log。

服务器重启后，在仓库根目录重新启动面板与旧日志同步器：

```bash
source reproduction/activate_official.sh
python reproduction/start_tensorboard.py \
  --original_log gym/logs/hopper-paper-20261105-090102.log --port 6006
```

不会自动启动或停止任何训练。若要关闭面板，只停止状态文件中的 tensorboard_pid 和 watcher_pid；不要停止训练 PID。

## 逐步复盘增强

完整字段、每个回合 CSV/图导出、还原某次参数更新的 batch，以及还原某个动作预测的 K 步上下文，见 [逐步复盘说明](TRACE_REVIEW.md)。


动画、独立 checkpoint 评估和 Hopper 指标解释见 [SIMULATION_AND_EVALUATION.md](SIMULATION_AND_EVALUATION.md)。新入口默认保存每轮首回合 GIF，并写入 TensorBoard Images。


2026-11-05 更新：活动代码的实验追踪改为 SwanLab 0.7.20；历史 W&B 依赖说明保留用于审计。当前登录、导入和训练命令见 [SWANLAB.md](SWANLAB.md)。
