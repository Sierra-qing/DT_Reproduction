# 运行与队列

所有命令从仓库根目录执行：

```bash
cd /home/ubuntu/weight/decision-transformer-master
source reproduction/activate_official.sh
python experiments/manage.py status
```

状态说明：complete=核实完成（或明确导入的旧结果）；ready=静态检查未发现缺项，仍应先短验证；running=控制器登记运行中；failed/interrupted=保留退出原因；blocked=缺数据、依赖或作者实现。ready不保证未验证任务一定能跑通。

## 当前两路队列

- `dt-paper-suite`：15项Hopper-medium普通BC、%BC、delayed。
- `dt-paper-suite-2`：27项Hopper medium-replay/medium-expert DT、BC/%BC、delayed。
- 每路内部串行，两路最多同时两个训练进程。计划互不重叠，使用GPU通道锁与计划交集检查。
- 当前42项是长队列，可能持续数天，并不承诺一晚结束。已完成的3个medium DT不会重跑。
- 第二路保留当前第一路任务，不重启正在训练的模型。所有核心训练预算、batch和学习率不因并行而变。

```bash
tmux attach -t dt-paper-suite
tmux attach -t dt-paper-suite-2
python experiments/manage.py status --plan hopper-ready
python experiments/manage.py status --plan hopper-remainder
```

离开tmux：Ctrl+b后按d。Ctrl+C会中断前台程序。失败任务保留日志，队列继续其他独立任务；不会自动调参、改CPU或反复重试。

## 新启动与短验证

当前两路已有任务时不要重复执行启动命令。今后空闲时：

```bash
bash experiments/scripts/start.sh hopper-ready 1
bash experiments/scripts/start.sh hopper-remainder 2
```

单项短验证（每项2次更新、每目标1回合，offline，不进入正式结果状态）：

```bash
python experiments/manage.py smoke bc-hopper-medium-p100-s0
```

单项正式实验：`python experiments/manage.py run 实验ID --lane 1`。运行整个清单必须显式指定`--plan`，不允许无参数误启动所有实验。

## 数据准备

```bash
python experiments/prepare_data.py --envs hopper --datasets medium-replay medium-expert
python experiments/prepare_data.py --envs halfcheetah walker2d --datasets medium medium-replay medium-expert
```

下载只用安装的D4RL元数据指向的官方主机，优先HTTPS；不覆盖已有pickle。按官方terminal/timeout分割完整轨迹，保留原脚本舍弃末尾不完整轨迹的行为。HDF5/pickle SHA256、输入和输出步数写入`gym/data/*.manifest.json`。本地哈希是溯源记录，不代表上游有发布校验值。Reacher训练数据没有验证来源，不能用普通Reacher替代。

## 状态与日志落点

- `experiments/configs/catalog.json`：所有配方、论文依据、依赖。
- `experiments/state/<ID>.json`：精确命令、PID、退出码、日志、结果目录。
- `experiments/state/lane{1,2}.json`：通道控制器PID与计划。
- `experiments/batches/<时间>/`：队列快照、每任务完整stdout、完成/失败列表。
- `experiments/reports/status.json`：最近一次status生成的总表。
- `experiments/runs/`：结果入口，兼容链接到旧实际目录。

检查running状态时还应核对PID和新日志，不能只信历史JSON。若服务器重启导致旧状态残留，先确认无进程，再处理；不要删除日志和checkpoint。

## 并行的统计限制

独立进程各自设种子、持有模型和优化器，不共享梯度。并行仍不能保证GPU计算逐位相同。CPU、磁盘和GPU会竞争，因此墙钟耗时、吞吐量、显存指标不能直接当串行基准。保持训练配置不变，记录通道与批次；不为提高利用率擅自增大batch。

## 尚未可运行的部分

Atari需独立依赖、ROM和DQN replay；入口与配方已有，但未做真实端到端验证，不把ready/配置数当完成数。Key-to-Door和图示例仅有规格与明确阻塞原因，缺官方实现/完整参数。详见实验协议，不将自造环境称为严格复现。


HalfCheetah/Walker接续队列已排入tmux `dt-locomotion-next`，等待lane1整组释放后，先短验证两个环境，再运行`locomotion-next`计划。状态见`experiments/state/followup_locomotion.json`；该等待进程不启动第三个训练。
