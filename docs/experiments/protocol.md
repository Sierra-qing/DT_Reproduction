# 论文协议与实现边界

优先级：论文明确设置 > 固定官方仓库（commit e2d82e68f330c00f763507b3b01d774740bee53f）。
来源：[论文正文与附录](https://arxiv.org/html/2106.01345v2)。用户粘贴版本与arXiv v2的表号、章节号有差异，按实验内容对应。

| 参数 | Gym DT | Gym BC / %BC |
|---|---|---|
| 模型 | GPT，3层/1头/hidden128 | 官方MLP，3层/hidden256 |
| 更新总数 | 100000 | 25000 |
| 每轮更新/轮数 | 10000×10 | 5000×5（评估间隔为本次明确配置） |
| batch | 64 | 64 |
| 优化器/lr/wd | AdamW / 1e-4 / 1e-4 | 同左 |
| warmup | 100000次，线性 | 100000次，遵循Table10 |
| K | 运动任务20，Reacher5 | 论文未指定，按官方默认20 |
| 激活/dropout | ReLU / 0.1 | ReLU / 0.1 |
| 梯度裁剪 | 0.25，原seq_trainer | 原ActTrainer未裁剪，不自行添加 |
| 训练seed | 0、1、2；各自独立 | 同左 |
| 每目标评估回合 | 100 | 100，BC只评估一个标签，策略不使用RTG |

DT主目标：Hopper3600、HalfCheetah6000、Walker5000、Reacher50。HalfCheetah/Reacher的官方入口默认首目标为12000/76，与论文不同，统一配方通过`eval_targets`显式覆盖并保留第二个辅助条件。Hopper/Walker原目标不变。主结果只用论文目标，不从多个条件中挑最好分数。

%BC按整条轨迹回报排序，在时间步预算内保留top10%、25%、40%；100%与普通BC共用一项，避免重复训练。delayed将轨迹总奖励放在最后一步，其余为0，使用官方实现；评估也保留官方delayed条件更新方式。

Atari：6层、8头、hidden128，5epochs，DQN replay取500000步/50buffers；Pong K50/batch512，其余K30/batch128；种子123/231/312沿用run.sh。Table8目标为Breakout90、Qbert2500、Pong20、Seaquest1450。独立包装纠正仓库Qbert14000/Seaquest1150硬编码，并按论文设置final_tokens=2×500000×K，而非仓库入口多乘3。官方Atari文件不改写。Atari配方尚缺数据/依赖，未作真实训练验证。

RTG扫描的网格是显式复现选择，论文未给完整原始网格；不能假称逐点与图4完全一致。Gym图4使用medium-replay，在已有medium模型上的扫描标为diagnostic。

Key-to-Door及图最短路径：固定官方仓库没有完整实验实现，目录记录论文已知规格与缺项，**未写虚假的可运行替代实现**。Reacher有环境代码，但缺作者生成的三种训练数据，不拿别的Reacher数据顶替。

CQL/BEAR/BRAC/AWR若引用论文对照值，必须标明引用；本套件不把他人的报告分数标成本机实测。

原始快照`reproduction_audit/`保留不动。当前`experiment.py`的修改限于日志接口与显式评估目标覆盖，采样、模型、损失、优化器和环境动力学不因本次整理改变。BC/Atari包装尚需更多真实对照，准备状态不能等同复现成功。


图任务另有明确分组的机制重实现，代码与假设见 [图重实现说明](graph_reimplementation.md)；不替代缺原作者完整配置的正式条目。Atari准备状态见 [专用环境](atari_setup.md)。
