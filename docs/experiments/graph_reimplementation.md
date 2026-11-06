# 图最短路：按论文机制重实现

入口：`experiments/reimplementations/graph_shortest_path.py`。目录ID为`graph-reimplementation-s0/s1/s2`，与缺作者源码的正式`graph-shortest-path`分开。**这不是已恢复作者完整训练配置的声明。**

已落实附录A.3机制：固定20节点图、边概率0.1、1000条最长10步随机游走、无效动作原地不动、目标状态0奖励/其他状态-1、RTG不折扣、同时预测回报与动作。模型推断时结合倾向短路径的回报先验（指数10），最短路BFS只用于事后评估，不给策略最优RTG或动作。

明示的重建选择：有向无自环Bernoulli图、固定goal19、均匀起点/动作、到目标停止、state/RTG/action token顺序支持当前状态条件下回报预测、使用代价d=-RTG作为分类索引，先验权重T+1-d。原文公式符号需结合其偏好短路径的语义理解，此映射作为明确实现选择留痕。

作者未列出的训练参数采用Gym参考值：3层/1头/hidden128/batch64/AdamW1e-4/wd1e-4/warmup100000，训练10000更新是本实现显式选择。不能据此宣称严格复现原作者图6数值。初版选择最大后验回报和最大概率动作，采样方式差异同样在config中记录。

已通过6项单元检查，包括非法动作、最短路评估、真实奖励和RTG、数据生成/策略不调用oracle、先验方向、因果mask。检查未训练模型；GPU短流程待空闲通道。当前两路训练不受影响，没有额外启动第三个实验。

空闲通道先做：

```bash
source reproduction/activate_official.sh
python experiments/manage.py smoke graph-reimplementation-s0 --lane 1
```

检查结果后再考虑`--plan graph-reimplementation`。该组在单独SwanLab项目`decision-transformer-reimplementations`中记录，避免与原协议主表混合。
