# 在 RTX 4090 上运行 Hopper-medium

这是现代依赖适配版：保留仓库的 DT 网络、采样和损失，评估使用
Gymnasium Hopper-v4 / MuJoCo 2.3.7，而原始 D4RL 使用旧 MuJoCo 环境。
因此先验证训练和评估，再比较论文指标；不要把这里的原始回报直接当作 D4RL 归一化分数。
尚未在 GPU 上验证此依赖组合。原始 conda_env.yml 保留供参考。

## 安装（在能够运行 nvidia-smi 的服务器终端）

```bash
cd /home/ubuntu/weight/decision-transformer-master
conda create -n dt4090 python=3.10 -y
conda activate dt4090
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
python -m pip install -r gym/requirements-4090.txt
python -m pip check
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(4, device='cuda') * 2).sum().item())"
```

不需要安装 torchvision、torchaudio 或下载 GPT-2 权重；模型从随机初始化开始训练。
不需要激活仓库里原有的 .venv。默认不启用 wandb。

## 先跑通训练和评估

```bash
cd /home/ubuntu/weight/decision-transformer-master/gym
python -u experiment.py --env hopper --dataset medium --model_type dt --device cuda --seed 0 --max_iters 1 --num_steps_per_iter 10 --num_eval_episodes 1
```

应加载 data/hopper-medium-v2.pkl，完成 10 次梯度更新以及两个目标回报各 1 次评估。
预期日志包含 training/train_loss_mean 和 evaluation/target_3600_return_mean。
这一步只检查流程，不判断收敛。
每次启动建立独立 runs 子目录，其中 config.json 保存参数、metrics.jsonl 保存每轮指标、
checkpoint.pt 保存最新一轮模型、优化器、学习率调度器和状态归一化统计。
checkpoint 每轮覆盖；目前没有命令行断点续训或单独评估入口。

## 完整训练

```bash
python -u experiment.py --env hopper --dataset medium --model_type dt --device cuda --seed 0 --K 20 --batch_size 64 --embed_dim 128 --n_layer 3 --n_head 1 --max_iters 10 --num_steps_per_iter 10000 --num_eval_episodes 100
```

保持原默认训练规模：总计 100000 次更新，每轮两个目标回报各评估 100 回合。
评估和 Python 数据采样也会占用时间，GPU 利用率未必一直很高。
先保持 batch_size=64，避免同时改变复现实验的训练设置。
跑通后可分别用 --seed 1、--seed 2 重复，并汇总多次运行结果。
可在另一个终端用 nvidia-smi 检查显存和利用率。
在对应运行目录保存环境清单：python -m pip freeze > runs/<本次运行目录>/pip-freeze.txt。

## 参考

- PyTorch 2.5.1 官方安装命令：https://pytorch.org/get-started/previous-versions/
- Hopper 环境版本差异：https://gymnasium.farama.org/v1.0.0a2/environments/mujoco/hopper/
