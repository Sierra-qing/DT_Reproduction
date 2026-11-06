# Hopper-medium 复现实验协议

2026-11-05 用户确定的优先级：**论文明确规定的严格执行；论文未规定的，使用固定官方代码。**

论文：Decision Transformer，arXiv:2106.01345v2，第 3 节、第 4.2 节、附录 A.2.1、表 9。
https://arxiv.org/html/2106.01345v2

官方代码固定提交：e2d82e68f330c00f763507b3b01d774740bee53f。
https://github.com/kzl/decision-transformer/tree/e2d82e68f330c00f763507b3b01d774740bee53f

## 参数与来源

| 项目 | 本次采用值 | 来源 |
|---|---|---|
| 实验对象 | Hopper-medium，DT | 论文表 2 |
| 序列表示 | RTG、状态、动作；因果注意力 | 论文第 3 节 |
| 训练目标 | 对有效时间步的连续动作 MSE | 论文第 3 节；mask 实现按官方代码 |
| RTG | 当前至轨迹结束的奖励和，不折扣 | 论文第 3 节 |
| 模型层数 / 注意力头 / 隐藏维度 | 3 / 1 / 128 | 论文表 9 |
| K / batch | 20 / 64 | 论文表 9 |
| 激活 / dropout | ReLU / 0.1 | 论文表 9 |
| 优化器 / 学习率 / 权重衰减 | AdamW / 1e-4 / 1e-4 | 论文附录 A.2.1 和表 9 |
| 梯度裁剪 | 0.25 | 论文表 9；官方 seq_trainer.py 已符合 |
| 更新总次数 | 100000 | 论文附录 A.2.1 |
| 学习率预热 | **100000 次更新** | **论文表 9 优先于代码默认的 10000** |
| 主评估目标 | 3600 | 论文表 9 |
| 多次运行 | 3 个随机种子 | 论文第 4.2 节的结果汇总设置 |
| 评估环境 | Gym Hopper-v3 | 论文未固定此软件版本，按官方入口 |
| 数据文件 | data/hopper-medium-v2.pkl | 论文未给出文件哈希或明确此文件版本，按官方入口；来源完整性另需核查 |
| 每轮更新 / 轮数 | 10000 / 10 | 代码默认，总更新数满足论文 |
| 每个目标每轮评估回合数 | 100 | 代码默认 |
| RTG 数值缩放 | 除以 1000 | 代码 |
| 状态归一化 / 轨迹采样 / padding | 原样采用官方实现 | 代码 |
| 数据筛选比例 | pct_traj=1.0 | 代码默认，使用全部轨迹 |
| Python 与依赖版本 | 先以 gym/conda_env.yml 为依据 | 论文未列出完整依赖版本，按仓库 |

论文规定了三种子的数量，但这里不伪造作者使用的具体种子编号。
官方入口没有 seed 参数或完整的设种子流程；后续需要独立、可审查的启动包装来固定和记录种子，不能把启动三次直接称为受控的三种子复现。

官方入口还评估 1800。保留此附加评估为代码行为，但它仅作为辅助指标；正式对比使用论文规定的 3600，不能取两者的最好分数。
论文结果使用 D4RL 归一化分数；官方脚本输出原始回报，后续必须按对应数据/任务版本的参考分数转换。
跨回合标准差和跨三个种子的标准差必须分开记录，不能相互替代。

## 完整训练参数命令

以下仅固定参数，不表示当前软件环境已经准备好。完成旧环境的 CUDA、MuJoCo、数据与短流程检查后再运行。

```bash
cd /home/ubuntu/weight/decision-transformer-master/gym
python -u experiment.py \
  --env hopper --dataset medium --model_type dt --mode normal \
  --K 20 --pct_traj 1.0 --batch_size 64 \
  --embed_dim 128 --n_layer 3 --n_head 1 \
  --activation_function relu --dropout 0.1 \
  --learning_rate 1e-4 --weight_decay 1e-4 \
  --warmup_steps 100000 \
  --max_iters 10 --num_steps_per_iter 10000 \
  --num_eval_episodes 100 --device cuda
```

原始源码保持固定提交一致，通过已有命令行参数覆盖预热步数。当前不为日志、进度、兼容性等重新改动核心源码。
若旧依赖无法安装或无法运行于 4090，应记录具体报错，再确定最小兼容方案；不能自动换为 Gymnasium/v4 或修改算法。

## 当前状态

- 已恢复并核对官方 gym 文件；本次仅新增协议和参数清单，没有改动官方源码。
- 当前 dt_raw 为此前现代适配环境，不代表已满足上述依赖。
- 尚未建立并验证本次正式实验环境，也未启动正式实验。
- 过去的 v4 实验保留归档，不作为本次复现结果。

## 2026-11-05 已授权的 GPU 兼容差异

用户同意 CUDA/PyTorch 层面的硬件兼容调整。本次先保留 PyTorch 主版本 1.8.1，仅从 cu102 换到官方 cu111 wheel，验证后才确认可运行。
原 cu102 构建已在沙箱外和用户 Bash 中共同验证：4090 可检测，但 CUDA 运算报 no kernel image。
此差异不修改 DT 源码、Hopper-v3/MuJoCo 2.0 或论文训练参数；但不再声称 CUDA 运行时与官方 conda 配置完全相同。
官方 conda_env.yml 保持原样；实际运行时以 torch.version.cuda 和环境快照为准，不能用系统 nvcc 或 nvidia-smi 的版本栏代替。
验证结果与剩余限制见 OFFICIAL_ENVIRONMENT_SETUP.md。

## 当前验证结果

2026-11-05：用户授权的 cu111 构建已安装，原始入口 10 步训练和少量评估通过。正式实验尚未启动。
环境版本、验证日志及可选依赖警告见 [环境结果](OFFICIAL_ENVIRONMENT_SETUP.md)。
