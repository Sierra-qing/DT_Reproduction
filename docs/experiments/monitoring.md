# SwanLab、TensorBoard与动画

训练使用SwanLab0.7.20在线模式和TensorBoard同时记录。不升级当前项目到要求Python3.9+的新看板；本环境Python3.8.5。

服务器登录只需一次：`source reproduction/activate_official.sh` 后 `swanlab login`。API key只在终端输入，不放配置或聊天。

在线项目：Sierra-qing/decision-transformer-reproduction。每个任务一个独立运行，名称包含任务/数据/算法/种子/时间与实验ID；日志会打印链接。不要把历史导入运行与新训练运行混为一组。

| 指标/面板 | 横轴 | 含义 |
|---|---|---|
| train/action_mse_step | 参数更新数 | 每次更新动作误差，越低不保证仿真回报越高 |
| train/learning_rate | 参数更新数 | 预热按既定公式增加，不由成绩自动调整 |
| eval/target_3600/return_mean | 参数更新数 | 本轮100个回合的实际回报均值 |
| eval/target_3600/normalized_score_mean | 参数更新数 | D4RL参考转换×100，便于对照论文 |
| episode/target_3600/return | 该目标累计评估回合数 | 每个回合表现 |
| rollout/target_3600/first_episode | 参数更新数 | RTG/奖励/动作/状态四联诊断图 |
| simulation/target_3600/first_episode | 参数更新数 | 同一回合的Hopper GIF |
| performance/sampling_percent | 参数更新数 | batch准备占训练区间墙钟时间比例；不是GPU利用率 |

DT每10000次更新后评估，BC每5000次更新后评估。没有到第一次评估时看不到rollout/回报/动画是正常的。BC没有RTG条件，日志沿用目标标签，但不能由此声称BC使用RTG。

终端每100次更新显示汇总，loss曲线每次更新记录。所有评估回合的每步轨迹都落盘，但每轮每目标只显示首回合图与动画，避免面板膨胀。训练RTG直方图目前只写入TensorBoard，未同步SwanLab；这不影响模型输入和本地完整记录。

TensorBoard在服务器127.0.0.1:6006。Mac转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 ubuntu@服务器地址
```

浏览器打开http://127.0.0.1:6006。Scalars看曲线，Images看rollout和动画。新结果仍位于gym/runs_observed，旧服务可以发现；先确认该运行被勾选并刷新页面。

原版`experiment.py`只记录轮级指标；完整观测统一入口由experiments/manage.py调用experiment_observed.py。旧日志导入脚本`gym/import_swanlab.py`只导入已有数据，不会补造逐步曲线或动画。


## 简短显示名

后续新建的运行使用 `Hopper-DT-Replay-s0`、`Hopper-BC10%-Medium-s0`、`Hopper-DT-Medium-Delayed-s0` 等简名；Medium=medium，Replay=medium-replay，Mix=medium-expert。本地长目录名与任务ID保持原样，`swanlab_location.txt`记录初始化时的显示名。已启动运行不会自动改名，可以在网页手动改；网页改名不会重写本地console.log、config.json或目录。
