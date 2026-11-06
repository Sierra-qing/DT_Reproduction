# 实验复盘

先看[首轮三种子报告](../../reproduction/results/overnight_20261105/REPORT.md)，再从实验ID查状态文件的run_dir。全部结果通过experiments/runs/访问，旧路径保持有效。

| 文件 | 保存内容 |
|---|---|
| config.json / metadata.json | 实际参数、依赖、硬件、源码哈希 |
| dataset.json | 文件哈希、数据统计、状态归一化、筛选情况 |
| train_steps.csv | 每次更新loss、lr、耗时 |
| training_provenance.jsonl | 每次更新抽中的轨迹ID、起点 |
| training_batches/ | 首步及每100次更新的实际输入和预测 |
| metrics.jsonl | 每轮汇总 |
| eval_episodes.csv | 每个评估回合回报、长度、耗时 |
| trajectories/*.npz | 每个回合每一步原始/归一化状态、动作、reward、目标与实际RTG、结束标记、物理状态 |
| trajectory_index.csv | 轨迹到对应checkpoint的映射 |
| checkpoints/iteration_XXXX.pt | 每轮模型、优化器、归一化统计和随机数状态 |
| videos/ / trajectory_figures/ | 动画与四联图 |

## 先回答什么

1. 仿真回报是否提高、回合间是否稳定？不要只看loss。
2. 同一模型的不同RTG目标是否产生不同动作和回报？目标不等于保证实现的成绩。
3. 失败发生在哪一步、因何条件结束？对照动作、躯干高度/角度、奖励及动画。
4. 跨种子统计是否与单次回合标准差混淆？比较时统一目标、训练预算、checkpoint口径。

Hopper每步奖励约为前进速度+1-0.001×动作平方和，不能只看站了多久。最长1000步约8秒仿真时间，实际墙钟耗时是另一概念。

## 现有复盘工具

进入gym目录后，替换真实路径：

```bash
python review_experiment.py list runs_observed/运行目录
python review_experiment.py episode runs_observed/运行目录/trajectories/轨迹文件.npz
python review_experiment.py batch runs_observed/运行目录 --update 20000
python review_experiment.py context runs_observed/运行目录/trajectories/轨迹文件.npz --timestep 20
python evaluate_checkpoint.py runs_observed/运行目录/checkpoints/iteration_0010.pt --target 3600 --episodes 100 --seed 0
```

batch可依据哈希一致的数据及采样索引还原实际输入；未保存的当时预测不能从后来的模型伪造。context还原模型当时实际能看到的K步窗口。独立评估不更新参数，但会占计算资源；两路训练繁忙时先分析已有轨迹。

## RTG扫描

先 `python experiments/manage.py status --plan return-sweeps`。扫描使用指定训练依赖的最终模型，保持每个目标相同评估seed，生成请求回报与实际回报图、summary.csv和逐步轨迹。要等GPU通道空闲再运行，不能与另一个重叠计划重复启动。

`diagnostic-sweep-hopper-medium-s0`等使用已有medium模型，无需重训；`sweep-hopper-medium-replay-s0`对应论文图4的数据设定。扫描网格为明确记录的复现选择，原文未列逐点网格。

首次结果统计固定用100000更新后的最终模型、主目标3600，三个种子均值及标准差；不挑中间最好分数。本轮61.24±4.40比论文67.6±1.0低6.36分，不能称已匹配论文。下一步应核查数据、评估口径并做行为分析，而非为追分随意调参。

新增只读行为分析：[最终轨迹汇总](../../experiments/reports/completed_hopper/REPORT.md)。复算：`python experiments/analyze_completed.py`。
