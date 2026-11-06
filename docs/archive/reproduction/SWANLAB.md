# SwanLab 与 TensorBoard 对照

## 过夜实验

登录后执行 `bash reproduction/start_overnight_swanlab.sh`（从仓库根目录）。脚本创建 `dt-swanlab-night` tmux 会话，依次运行种子0、1、2，每次100000更新、预热100000，SwanLab在线模式，TensorBoard同步本地记录。重复执行会拒绝创建同名会话，避免意外重复训练。未登录或CUDA不可用时拒绝训练。

查看终端：`tmux attach -t dt-swanlab-night`。按 Ctrl+b 后按 d 脱离会话，断开SSH后继续运行。每个种子结束后自动开始下一个，失败时停止队列并保留日志，不自动改变配置或重试。

队列路径记录在 `reproduction/overnight_latest.txt`，对应目录下 `status.json` 保存命令、种子、PID与退出状态，`seed_0.log` 等为完整输出。模型、轨迹、动画仍在 `gym/runs_observed/` 每个种子的独立目录。三个种子预计需十余小时，明早可能仍在运行，不承诺指定时间完成。

按本次要求，活动代码中的 W&B 已替换为 SwanLab：experiment.py、experiment_readable.py、experiment_observed.py、导入脚本、依赖声明和检查脚本均已更新。旧参数 --log_to_wandb 不再使用，新参数为 --log_to_swanlab。

SwanLab 0.7.20 支持本项目 Python 3.8。PyTorch、CUDA、Gym、MuJoCo、NumPy、Transformers、TensorBoard 的版本未改变。安装新增 SwanLab 依赖，并将辅助包 platformdirs 更新到 4.3.6、wrapt 更新到 1.17.3；安装前版本和约束文件保存在本目录。原有 W&B 安装包没有卸载，活动训练代码已不再调用它。

## 登录并立即查看已有结果

在服务器自己的 Bash/tmux 中执行：

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
swanlab login
```

根据提示在终端输入 SwanLab API key，不要发送到聊天。注册与管理账号：https://swanlab.cn 。

无须重跑训练，就能把已有真实轮级指标导入在线项目：

```bash
cd /home/ubuntu/weight/decision-transformer-master/gym
python import_swanlab.py \
  runs_observed/imported-official-hopper-paper-20261105-090102 \
  --swanlab_mode online
```

终端会打印项目/实验链接，使用浏览器打开即可。该命令导入命令执行时已有的数据，不持续监听；后续新增的轮次需要再导入，形成新记录。旧运行只有轮级指标，没有逐步 loss 和动画，导入不会补造这些数据。旧运行 warmup=10000，与论文约定100000不同。

## 下一次正式训练同时记录两个面板

以下命令会启动一场新实验；不要只为查看面板而重复启动训练。

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
cd gym
python -u experiment_observed.py \
  --env hopper --dataset medium --model_type dt --device cuda \
  --seed 0 --warmup_steps 100000 \
  --max_iters 10 --num_steps_per_iter 10000 --num_eval_episodes 100 \
  --log_interval 100 --eval_log_interval 10 \
  --log_to_swanlab --swanlab_mode online \
  --swanlab_project decision-transformer-reproduction
```

TensorBoard 继续写入相同运行目录的 tensorboard/。SwanLab 写入 swanlog/，在线模式同步到账号。--swanlab_workspace 可指定团队空间，默认个人空间。默认 mode=offline 只记录本地文件，不能据此声称已经打开在线网页。

## 面板如何对应

| 记录 | 横轴 | 看什么 |
|---|---|---|
| train/action_mse_step | 参数更新数 | 每次更新的实际动作 MSE |
| train/learning_rate | 参数更新数 | 学习率预热 |
| eval/target_3600/return_mean | 参数更新数 | 同一策略评估后的回报均值 |
| eval/target_3600/normalized_score_mean | 参数更新数 | 与论文标准化分数比较 |
| episode/target_3600/return | 该目标累计评估回合数 | 不同回合的波动 |
| rollout/target_3600/first_episode | 参数更新数 | 每轮首回合逐步诊断图 |
| simulation/target_3600/first_episode | 参数更新数 | 每轮首回合 Hopper 动画 |

新观测入口同时保留 TensorBoard 和 SwanLab，核心指标名称、值与坐标语义相同，可并排查看。TensorBoard 的 Images 对应 SwanLab 的图像/视频面板。直方图和 TensorBoard 自定义布局目前未同步，全部原始轨迹仍在本地 NPZ 中。

experiment.py 的 --log_to_swanlab 仍只记录官方输出的轮级汇总；逐步曲线和动画请用 experiment_observed.py。可读版同样换成 SwanLab，但不作为正式论文复现入口。

## 保留审计历史

reproduction_audit/ 中官方原始快照、已有实验配置/日志和审计文档中的历史依赖信息保留原文。它们可能出现 W&B 名称，不是当前活动代码。experiment.py 的本次差异仅为日志接口替换，模型、数据采样、损失、优化器和评估规则未改。以原始快照为依据比较后续变动。
