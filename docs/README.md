# 文档导航

仅以下文档为当前使用说明；archive/是历史记录，部分内容反映当时状态，不再作为操作步骤。

1. [实验协议](experiments/protocol.md)：论文参数、代码默认值、明确差异。
2. [运行指南](experiments/running.md)：查看目录、准备数据、两路队列与故障处理。
3. [监控](experiments/monitoring.md)：SwanLab/TensorBoard、横轴、动画、记录频率。
4. [复盘](experiments/review.md)：结果统计、权重、逐步CSV、上下文与RTG扫描。
5. [环境](experiments/environment.md)：使用哪个Python，哪些依赖不应动。

实验清单唯一来源：`experiments/configs/catalog.json`；状态以本地文件核实，不以文档中的手写“完成”替代。

[首轮三种子报告](../reproduction/results/overnight_20261105/REPORT.md) · [历史审计](archive/REPRODUCTION_AUDIT.md)

公开复现调查：[Atari、Reacher、Key-to-Door与图任务](research/other_experiments.md)。

准备进度：[Atari独立环境与数据阻塞](experiments/atari_setup.md) · [图任务重实现边界](experiments/graph_reimplementation.md)。
