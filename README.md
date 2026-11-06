# Decision Transformer 论文复现

本项目以 [Decision Transformer](https://arxiv.org/abs/2106.01345) 及 [官方代码](https://github.com/kzl/decision-transformer) 为依据。**论文明确设置优先，未规定的采用固定官方代码；所有兼容与观测差异留痕。**

先读 [实验导航](docs/README.md)，全部实验通过 [experiments/](experiments/README.md) 管理。

| 你要做什么 | 唯一当前入口 |
|---|---|
| 查论文参数、复现边界 | [实验协议](docs/experiments/protocol.md) |
| 查哪些完成、哪些能跑、缺什么 | `python experiments/manage.py status` |
| 启动/查看实验队列 | [运行指南](docs/experiments/running.md) |
| 看SwanLab、TensorBoard、动画 | [监控说明](docs/experiments/monitoring.md) |
| 分析已有权重和逐步轨迹 | [复盘指南](docs/experiments/review.md) |
| 查环境和已授权变更 | [环境说明](docs/experiments/environment.md) |
| 看第一次三种子结果 | [Hopper-medium结果](reproduction/results/overnight_20261105/REPORT.md) |

```text
experiments/            配置清单、准备数据、队列、各实验运行包装
  configs/              论文实验目录及可选择的计划JSON
  runners/              Atari、回报扫描等独立入口
  state/                逐任务状态、两路队列登记与互斥锁
  batches/              每次队列的计划、命令、控制台日志和退出状态
  runs/                 训练结果入口（兼容链接到 gym/runs_observed/）
  reports/              实验状态总表
  smoke_runs/           短验证结果，绝不作为正式论文结果
 gym/                   官方Gym算法与观测/复盘工具；data/保存数据
 atari/                 固定官方Atari代码与数据入口
 docs/                  当前文档；archive/只保存历史，不照旧命令启动
 tests/                 验证代码
 reproduction/          已有环境工具、安装日志、第一次overnight和结果归档
 reproduction_audit/    官方原始源码及变更前快照，不改写
```

已完成的 Hopper-medium DT 三种子最终分数为 **61.24±4.40**（跨种子总体标准差），论文参考67.6±1.0；尚未匹配论文成绩。模型、数据采样和训练目标并未因为整理目录而替换。

SwanLab/RTG目标参数为可审查的日志与论文协议适配，`gym/experiment.py` 因此不再与官方字节一致。未提供作者实现的实验不会以自造替代品标为复现完成。

[原始英文README与论文引用](docs/archive/official_README.md) · [MIT许可证](LICENSE.md)
