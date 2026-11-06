# 实验工作区

从仓库根目录运行，先 `source reproduction/activate_official.sh`。

```bash
python experiments/manage.py list
python experiments/manage.py status --plan hopper-ready
python experiments/manage.py status --plan hopper-remainder
```

目录清单覆盖 Gym DT/BC/%BC/delayed、Gym与Atari RTG扫描、Atari DT/BC/%BC/K=1、Key-to-Door及图示例。**配方存在不等于实现完整或环境可用**：`blocked`显示缺失数据、依赖或作者代码。

| 计划 | 内容 |
|---|---|
| hopper-ready | 15项：已有medium上的BC/%BC/delayed |
| hopper-remainder | 27项：medium-replay/medium-expert上的Hopper实验；与前者不重叠 |
| hopper-next | 上述两组共42项；不能与其中任一正在运行的计划重复启动 |
| locomotion-next | 72项HalfCheetah/Walker实验，已安排等待通道后短验证并接续 |
| graph-reimplementation | 3项按附录机制重实现，GPU训练尚待验证，独立标注 |
| gym-all | 150项尚未完成的Gym训练配置，含当前有数据或代码缺口的任务 |
| return-sweeps | Gym medium-replay扫描及现有medium模型诊断 |
| atari-all | Atari训练与K=1消融；需单独依赖和DQN replay准备 |

每项有稳定ID，例如 `dt-hopper-medium-replay-s0`、`bc-hopper-medium-p10-s0`。失败时保留原因与日志；已完成项自动跳过，失败不会自动修改参数或换CPU。运行整个计划之前必须先查看status。

结果入口`runs/`是兼容链接，实际文件仍在`gym/runs_observed/`，旧报告、checkpoint路径和TensorBoard不因整理而断开。测试结果只进入`smoke_runs/`。

人工总表：[实验覆盖与准备状态](reports/CATALOG.md)。刷新：`python experiments/report_catalog.py`。

详细操作：[运行指南](../docs/experiments/running.md)。
