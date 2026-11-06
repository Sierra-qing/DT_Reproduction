# Atari 准备状态与独立环境

2026-11-06：保留官方Python3.7.9，专用环境路径：

`/home/ubuntu/miniconda3/envs/decision-transformer-atari/bin/python`

不修改正在运行的Gym环境。为兼容4090，策略Torch采用已验证系列的1.8.1+cu111构建；原官方1.2/CUDA10不能直接用于该GPU。SwanLab0.7.20要求Python3.8，因此日志使用Gym环境的独立进程，不将SwanLab强塞进Python3.7，也不更换训练算法。

## 软件组件

原始replay依赖TensorFlow1.15.5、Dopamine3.0.1、NumPy1.18.5、gin-config0.3.0；图像环境使用atari-py0.2.9、OpenCV4.5.5.64。明确版本在`experiments/setup/atari_requirements.txt`。

已核查原版FixedReplayBuffer可导入，合成replay保存/加载及4帧堆叠与原Dopamine一致。Python3.7到Python3.8的离线SwanLab通道也已验证。这些检查不代表真实Breakout数据与模型训练通过。

Torch安装后的完整软件检查在tmux `dt-atari-setup`中自动继续；实时状态见 `experiments/setup/atari_status.json`、`reproduction/setup_logs/atari_finish_setup.log`。最终版本核查写入`experiments/reports/atari_runtime_check.json`。模型GPU前后向短验证仍需空闲通道，未伪称已完成。

注意：SwanLab自身的进程CPU/内存图反映日志进程；实际训练Python、Torch和CUDA版本另写`training_runtime.json`，配置也注明了独立日志进程，不能混淆硬件耗时统计。

## 数据阻塞

实际探测官方 `gs://atari-replay-datasets`：列目录API返回401，Breakout run1的6个原始对象HEAD均返回403。不是因为未安装gsutil；单个对象也不可公开访问。本次没有下载到真实Atari训练数据，没有用5buffers、其他数据集或自行生成数据冒充原始50buffers。

[公开维护者问题记录](https://github.com/takuseno/d4rl-atari/issues/19)中也记录访问失效。评论给出的其他Atari数据来源尚未证明与原DQN replay相同，不能直接替换。

探测记录：`experiments/setup/breakout_object_probe.json`。如果以后找到具有可验证来源的原始副本，再校验buffer范围、状态帧、动作、终止标记、奖励和采样协议，方可解除阻塞。

## ROM

当前atari-py仅发现其自带Tetris ROM，四个论文任务的ROM未准备。真实Breakout reset/step仍待ROM。不能将Tetris测试当成论文任务验证。数据和ROM齐备后仍需GPU短流程，而不是马上启动72项长期训练。
