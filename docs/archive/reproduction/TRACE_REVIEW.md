# 逐步记录与实验复盘

## 生效范围

使用 `gym/experiment_observed.py` 新启动的实验会保存下面的记录。
正在运行的旧 `experiment.py` 进程不能补回未记录的中间动作、状态和 RTG；面板中的旧 imported-official 运行仍只有每轮汇总。
原始官方源码不改动。评估频率仍是每轮一次，不额外插入评估来改变官方实验流程。

## 新实验命令

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
cd gym
python -u experiment_observed.py \
  --seed 0 --warmup_steps 100000 \
  --log_interval 100 --eval_log_interval 10 \
  --batch_snapshot_interval 100
```

这会新建一次完整实验，不接管旧进程。默认仍为 10 轮、每轮 10000 次更新，每个目标 100 个评估回合。
终端每 100 次更新显示区间平均；TensorBoard 中 `train/action_mse_step` 每次更新都有一个点。
默认事件文件最多约 5 秒刷新一次，面板还受页面刷新周期影响。TensorBoard 可能抽样显示点，CSV 才是逐步记录的完整依据。

## 实际记录的范围

| 内容 | 频率 | 文件 |
|---|---|---|
| 动作 MSE、学习率、更新耗时 | 每次参数更新 | train_steps.csv |
| 区间 loss、速度、采样占比 | 默认每 100 次更新 | train.csv |
| 真实采样轨迹编号与片段起点 | 每次参数更新、每个 batch 样本 | training_provenance.jsonl |
| 完整训练输入与当时的动作预测 | 第一次更新及默认每 100 次更新 | training_batches/update_XXXXXXX.npz |
| 评估状态、动作、奖励、RTG | 全部回合的全部环境时间步 | trajectories/*.npz |
| 原始环境 info、终止/超时标记 | 每个已执行环境时间步 | 同一轨迹 NPZ |
| MuJoCo qpos/qvel/time/act/ctrl/warmstart | 初始及每步执行后，共 T+1 个状态 | 同一轨迹 NPZ |
| 策略及优化器等状态 | 每轮，评估前先保存，轮末更新 | checkpoints/iteration_XXXX.pt |
| 回合列表、回报、长度、对应策略 | 每回合 | trajectory_index.csv |
| 每目标每轮的第一个回合诊断图 | 每次评估 | trajectory_figures/*.png 与 TensorBoard Images |

所有训练输入均能根据实际采样位置、配置和哈希一致的原始数据还原，避免把重复输入在每次更新都存一份。
未保存预测的更新不能靠最后的模型“补算”成当时的预测；若确实需要每次更新的完整输入和预测，设置 `--batch_snapshot_interval 1`。
`--batch_snapshot_interval 0` 关闭完整 batch 快照，但保留全部采样来源和训练损失。
记录会增加数据复制、磁盘写入与压缩耗时；不能把开启详细记录后的速度当作纯算法或纯 GPU 性能。
没有默认保存所有网络隐藏激活或完整梯度张量。

## 分清两个时间轴

- `global_step`：参数更新次数。例如 20000 表示训练进行了两万次更新。
- `environment_timestep`：一个评估回合内的环境时间，从 0 开始。
- `model_timestep`：真正交给模型时间嵌入的索引；正常 Hopper 评估中与回合时间相同。
- 训练 batch 中的 `timesteps`：来自离线原轨迹的绝对时间，不是片段内从 0 重编号。

## 一个评估时间步的字段对齐

预测动作之前：

```text
observation[t]              原始环境观测
observation_normalized[t]   本次 get_action 真正收到的最后一个标准化状态
 target_rtg_scaled[t]       本次 get_action 真正收到的目标 RTG（模型数值尺度）
 target_rtg_raw[t]          上述值 × scale，含 float32 舍入误差
 action_predicted[t]        模型实际返回的动作
```

执行动作之后：

```text
action[t]                   真正传给 env.step 的控制向量
reward[t]                   执行动作后获得的奖励
next_observation[t]         执行动作后的观测
cumulative_reward_before[t] 执行前已经得到的奖励总和
cumulative_reward[t]        执行后已经得到的奖励总和
terminated[t]               Gym done 且不是 TimeLimit.truncated
truncated[t]                原始 info 中的 TimeLimit.truncated
info_json[t]                原始 info 的 JSON 文本
```

`action` 是控制向量，不应直接当作已换算的物理力矩。元数据保留 actuator_names、actuator_gear、joint_names。
原始观测不等于完整模拟器状态；Hopper 的观测省略全局 x 位置，且部分速度被裁剪，因此额外保存物理状态用于判断运动过程。
这些物理字段有助于复盘，但不承诺仅凭它们就能位级一致地恢复模拟器。

### 两种 RTG 必须区分

- `target_rtg_raw`：当时要求模型尝试达到的剩余回报，是策略的条件。
- `realized_rtg_raw`：回合结束后，按真实奖励从后向前求和得到的实际剩余回报，只用于事后分析。

后者绝不反向送入当时的评估策略，不能作为它“提前知道未来奖励”的证据。
`next_target_rtg_scaled_observed[t]` 来自下一次真实模型调用；最后一步没有下一次调用，因此存 NaN，并用 mask 标记未知，不能伪造为已观察值。
BC 不使用 RTG 条件，相关字段标记为未使用。

## 从一个回合导出 CSV 与图

在 gym 目录，用实际运行目录代替下例路径：

```bash
python review_experiment.py list runs_observed/你的运行目录

python review_experiment.py episode \
  runs_observed/你的运行目录/trajectories/iteration_0001_target_3600_episode_0001.npz
```

输出到该运行目录的 review/：

- CSV：每行一个环境时间步，状态/动作每个维度分别展开，含 RTG、奖励、终止标记和物理状态。
- PNG：目标/实际剩余回报、逐步/累计奖励、所有动作分量，以及前 3 个观测分量。其余观测维度仍完整保存在 CSV/NPZ 中。
- JSON：回合元数据和对应 checkpoint 路径。

TensorBoard 的 Images 标签中也可用训练步数滑块对比各轮每个目标的第一个回合。

## 还原某次动作预测时的全部输入

例如还原第 20 个环境时间步（索引从 0 开始）：

```bash
python review_experiment.py context \
  runs_observed/你的运行目录/trajectories/iteration_0001_target_3600_episode_0001.npz \
  --timestep 20
```

输出一个 context NPZ，包含 K 步的标准化状态、历史动作、RTG、时间索引和 attention_mask。
当前动作位置使用 0 占位，不能把已知的真实预测动作作为它自己的输入。
评估中，状态先标准化，再由模型左补 0；训练中则先补原始状态 0，再标准化。这是官方代码原有差别，复盘时不能混淆。
回合少于 21 步时，请选合法的时间索引。

## 还原任何一次训练更新的完整输入

```bash
python review_experiment.py batch runs_observed/你的运行目录 --update 1234
```

工具检查原始数据的 SHA-256，然后按当时记录的真实轨迹编号和起点还原：

- states_normalized [B,K,S]
- actions_data [B,K,A]
- rewards [B,K,1]
- dones [B,K]
- rtg_scaled [B,K+1,1]
- timesteps [B,K]
- attention_mask [B,K]

若数据移到了其他位置，可加 `--dataset /新的路径/hopper-medium-v2.pkl`，哈希仍必须一致。
在实际 batch 快照存在的更新上，还可以直接比较动作预测、动作标签以及 mask 后的 MSE。
不能把某轮 checkpoint 误认为每一次参数更新时的模型；我们只保留每轮模型，并明确记录 phase。

## 中断与数据边界

普通异常或 Ctrl+C 时，已记录的当前评估片段以 `complete=false` 保存；其 realized_rtg_raw 只是已观察前缀内的奖励和，不能称为完整未来回报。
训练和索引行会及时 flush，但强制 kill -9 或系统掉电无法保证正在写入的最后一行/最后一个回合被保存。
每轮评估前先保存对应模型，因此中途评估片段也能找到当时的策略文件。
目前仍没有断点续训命令行入口；保存这些状态用于审查，不等于已经实现恢复训练。

## 验证结果

- 相同种子的 CUDA 更新对照，新增记录不改变原版模型参数和优化器状态。
- 真实 Hopper 对照，逐步状态、动作、奖励与原版逐项一致。
- 检查了 RTG、标准化状态、终止标记、T+1 物理状态和对应轮次模型。
- 按采样索引还原的训练 batch 与实际输入快照逐项一致，预测 MSE 与记录损失一致。
- 检查了每次更新的 TensorBoard scalar、每轮轨迹 Images、CSV/PNG 导出。
- 验证测试单独放在 reproduction/trajectory_validation_runs，不作为论文实验分数。
- 绘图依赖最终仍为原版本 matplotlib 3.7.2、contourpy 1.0.5、kiwisolver 1.4.4，仅替换成兼容的 pip 二进制构建。


动画、独立 checkpoint 评估和 Hopper 指标解释见 [SIMULATION_AND_EVALUATION.md](SIMULATION_AND_EVALUATION.md)。新入口默认保存每轮首回合 GIF，并写入 TensorBoard Images。


2026-11-05 更新：活动代码的实验追踪改为 SwanLab 0.7.20；历史 W&B 依赖说明保留用于审计。当前登录、导入和训练命令见 [SWANLAB.md](SWANLAB.md)。
