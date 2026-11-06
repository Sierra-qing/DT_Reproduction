# Atari 实验入口

固定官方源代码保留在本目录。当前配方、论文差异与阻塞项统一见：

- [论文协议](../docs/experiments/protocol.md)
- [实验工作区](../experiments/README.md)
- [运行与依赖准备](../docs/experiments/running.md)

`experiments/runners/atari.py` 包装原版入口，支持DT、BC/%BC、K=1消融和SwanLab；`atari_sweep.py`使用已保存最终模型扫描RTG。这些入口尚未在当前服务器完成真实Atari验证，缺少DQN replay、ROM与相应依赖，不应直接启动长期任务。

数据布局：`atari/dqn_replay/<Game>/1/replay_logs/`。官方源为 `gs://atari-replay-datasets/dqn/<Game>`，不要拿另一套Atari离线数据替代。下载前确认磁盘容量和准备好的独立环境；本轮未下载大型Atari replay。

[原始英文说明](../docs/archive/official_readme-atari.md)仅作历史记录。
