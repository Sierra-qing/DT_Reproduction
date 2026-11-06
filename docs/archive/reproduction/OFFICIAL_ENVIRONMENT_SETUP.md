# 官方基线环境安装与验证结果

日期：2026-11-05。环境：`decision-transformer-gym`。

## 当前结论

4090 上的原版 Hopper 实验链路已实际通过验证：CUDA 运算、原版 DT 前向/反向、Hopper-v3、D4RL Hopper 注册和评分函数，以及原始 experiment.py 的 10 步训练和评估。
这不是完整论文实验；正式 100000 步训练和跨三个随机种子的统计尚未进行。
官方源码保持固定提交一致。现有 dt_raw 未改动。

## 版本和唯一已授权的核心差异

| 项目 | 当前值 | 说明 |
|---|---|---|
| Python | 3.8.5 | 官方依赖文件 |
| Gym | 0.18.3 | 官方依赖文件 |
| mujoco-py | 2.0.2.13 | 官方依赖文件 |
| Hopper 使用的 MuJoCo 引擎 | 2.0，C API 版本号 200 | 实测 |
| NumPy | 1.20.3 | 官方依赖文件 |
| PyTorch | 1.8.1+cu111 | 用户授权从 cu102 换到 cu111；保留 1.8.1 主版本 |
| PyTorch CUDA 运行时 | 11.1 | 官方 cu111 wheel，自带运行库，不修改 NVIDIA 驱动 |
| Transformers | 4.5.1 | 官方依赖文件 |
| wandb | 0.9.1 | 官方依赖文件 |
| D4RL | 1.1 | 按官方 README 获取源码；提交见 vendor/d4rl_commit.json |

原版 cu102 的实际运算在 4090 上报 no kernel image，用户 Bash 和沙箱外检查结果一致。
cu111 编译架构包含 sm_80、sm_86，已在 4090 上实际通过本次检查。
官方 conda_env.yml 仍保留 CUDA 10.x 的原文；本环境有明确记录的 CUDA 构建兼容差异，不能称为所有软件层完全一致。

## 验证证据

| 检查 | 实际结果 |
|---|---|
| 固定核心包版本 | 通过 |
| MuJoCo 版本号 | 200 |
| Hopper-v3 reset/step | 通过 |
| CUDA 逐元素计算 | 8.0，正确 |
| CUDA 矩阵乘法和反向 | 通过 |
| 原版 DT 模型 CUDA 前向/反向 | 通过 |
| D4RL Hopper 注册和归一化评分端点 | 通过 |
| 原版 experiment.py，10 次更新＋两个目标各 1 回合 | 退出码 0 |

流程测试日志显示训练约 1.52 秒，评估约 0.19 秒，训练动作 MSE 均值约 0.633。
这只是未充分训练的流程检查，回报不能作为论文复现分数。

- `setup_logs/preflight_cu111.log`：完整依赖、模拟器、GPU、模型和 D4RL 检查输出。
- `setup_logs/smoke_test_cu111.log`：原始入口实际训练与评估输出。
- `setup_logs/validation_status.json`：命令与退出码。
- `setup_logs/official_environment_check_cu102.log`：旧 CUDA 构建失败证据。
- `setup_logs/cuda_compatibility_install.log`：CUDA 构建替换日志。
- `setup_logs/installed_versions.json`：实际包名、版本和 Python 路径快照。
- `setup_logs/installed_packages_freeze.txt`：pip 原始输出，包含 Conda 构建路径，不作为跨机器一键安装文件。

## 构建兼容处理与限制

为了安装官方指定的旧包，新环境使用 pip 21.3.1、setuptools 65.5.0、wheel 0.38.4、Cython 0.29.37。
旧版 Transformers 的版本解析需要旧 packaging，实际使用 20.9；protobuf 为 3.20.3，imageio 为 2.9.0。
这些属于官方未锁定的构建/间接依赖，版本已记录。

当前 `pip check` 仍报告 Anaconda 带入的 6 个现代工具与 packaging 20.9 的声明冲突：
xarray、statsmodels、pytoolconfig、jupyterlab-server、jupyter-server、black。
它们未进入本次 Hopper 训练链路；不能把当前环境描述为全套 Anaconda 应用都验证可用。
具体输出见 setup_logs/pip_check.log。本次没有为清理无关工具而修改核心版本。

D4RL 会尝试导入很多非 Hopper 任务，因此存在 Flow/CARLA 缺失、无 DISPLAY，以及可选 PyBullet 的 NumPy C API 不匹配警告。
没有隐藏这些警告或为修复可选任务而升级 NumPy；本次不把这些任务列为已支持。
D4RL 的 dm_control 依赖额外安装了现代 `mujoco` 3.2.3，供其其他任务使用；
它与 `mujoco-py` 是不同的 Python 包。本次 Hopper-v3 及 D4RL Hopper 使用的仍是 MuJoCo 2.0，已断言确认版本号 200。

系统层安装了 libosmesa6-dev、libglfw3、patchelf、git 及其系统依赖；没有修改 NVIDIA 驱动或设备权限。
MuJoCo 官方二进制位于 `reproduction/vendor/mujoco200_linux`，经 `~/.mujoco/mujoco200` 链接发现。
官方公开激活文件经 `~/.mujoco/mjkey.txt` 链接发现。

## 在用户 Bash / tmux 中运行

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
python reproduction/check_official_environment.py --check all
```

短流程测试已经由本次配置工作执行，无需为了重复确认再跑一次。
准备正式实验时，使用 PAPER_PROTOCOL.md 的参数；预热为 100000 次更新。
原始入口没有 --seed、--output_dir、--log_interval 或本地 checkpoint 功能，不能传入此前适配版参数。
完整三种子实验仍需单独明确并记录种子设置；当前不把重复启动三次等同于受控的三种子实验。

## 2026-11-05 观测功能补充

新增独立观测入口和 TensorBoard；原始源码仍保持不变。TensorBoard=2.13.0，protobuf=3.20.3（同版本改用 pip wheel 修复 C++ ABI）。
使用方法、验证和限制见 [观测说明](OBSERVABILITY.md)。


2026-11-05 更新：活动代码的实验追踪改为 SwanLab 0.7.20；历史 W&B 依赖说明保留用于审计。当前登录、导入和训练命令见 [SWANLAB.md](SWANLAB.md)。
