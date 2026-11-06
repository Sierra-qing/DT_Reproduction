# 运行环境

```bash
source /home/ubuntu/weight/decision-transformer-master/reproduction/activate_official.sh
which python
```

应为 `/home/ubuntu/miniconda3/envs/decision-transformer-gym/bin/python`。

| 项目 | 当前版本 |
|---|---|
| Python | 3.8.5 |
| PyTorch / CUDA wheel | 1.8.1+cu111 / CUDA11.1 |
| Gym | 0.18.3 |
| mujoco-py / 实际MuJoCo引擎 | 2.0.2.13 / 2.0 |
| NumPy | 1.20.3 |
| Transformers | 4.5.1 |
| D4RL | 1.1，固定源码89141a689b0353b0dac3da5cba60da4b1b16254d |
| SwanLab / TensorBoard | 0.7.20 / 2.13.0 |

cu102在4090上出现no kernel image，用户授权更换为同版torch的cu111构建并已验证真实CUDA运算。不要用nvidia-smi显示的最高CUDA兼容版本代替torch实际构建版本。不要换Gymnasium/v4来掩盖旧依赖问题。

SwanLab迁移仅新增其依赖，并更新辅助包platformdirs4.3.6、wrapt1.17.3；训练相关版本不变。历史W&B包仍安装但活动训练代码不调用。安装前快照/约束及完整日志在reproduction/。

环境较老且包含Anaconda额外工具，部分与packaging20.9有既有冲突；不能声称整个环境pip check零问题。训练路径已单独验证，不为可选工具升级核心依赖。

Atari当前缺失的模块与replay由status列出；不向本Gym环境随意安装tensorflow/新torch去满足Atari，应单独建立并验证环境。Reacher环境存在，但论文作者生成的数据缺失。Key-to-Door原始代码缺失，不安装一个同名环境后假称论文设置相同。

测试入口在tests/；旧reproduction/test_*.py仅为兼容转发。短流程真实GPU验证结果位于experiments/smoke_runs/，不与正式训练混合。


Atari专用Python3.7.9环境已建立，详见 [准备状态](atari_setup.md)。当前原始数据HTTP401/403和目标游戏ROM仍是阻塞，不能称已可正式训练。
