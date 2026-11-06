# Atari、Reacher、Key-to-Door与图最短路公开实现调查

核查日期：2026-11-06。只检索、下载及静态阅读，没有执行第三方代码、安装依赖或改动运行中的训练。

## 证据标准

README声称成功、notebook保存过输出、原论文严格可复现是不同层次的证据。第三方自行生成的数据与修改后的环境只能先标为重实现，不能直接拿原论文数字验收。以下是本次找到并检查的来源，不是“互联网上绝不存在其他实现”的断言。

## Atari

1. [官方Atari说明](https://github.com/kzl/decision-transformer/blob/master/atari/readme-atari.md)：原始DQN replay、minGPT和Dopamine读数据路径，最接近原始复现路线。
2. [tsmatz Pong教学实现](https://github.com/tsmatz/decision-transformer)：公开完整notebook，直接处理原始gzip/NumPy数据，将轨迹切段后分块加载到GPU，避免整套像素数据同时放入显存。模型部分有6层、8头、hidden128；当前训练单元Pong batch128，与论文Pong batch512不同，不能直接当成原参数复现。commit bcdcbbd87d3b7a3ce65d9fdfc1f82a73daed26af。
3. [Jaron-U Pong实现](https://github.com/Jaron-U/decision-transformer-atari)：基于官方/minGPT/tsmatz，独立预处理后训练，有权重和loss图。README明确演示只加载5/50个数据文件，不能认为与原始采样覆盖相同。commit bcc5e2002883b96d6cc348b5201cf98c71cbacd4。项目提到的完整数据存储量是项目作者的描述，本机未下载实测。

可借鉴点：原始文件读取和分块数据管道可减少旧Dopamine/TensorFlow依赖；但更换读取器前需要验证帧堆叠、边界、RTG、采样及500000步的覆盖，不直接复制教学超参。

## Reacher2d

[LaMo作者仓库](https://github.com/srzer/LaMo-2023)公开 `data/data_generation_PPO/`，并链接其预处理数据。其[论文补充材料](https://proceedings.iclr.cc/paper_files/paper/2024/file/c7793beb7f32b559df55d48370f2b8ae-Supplementary-Conference.pdf)说明用中等水平PPO生成50回合Reacher数据，这是LaMo自己的设置，不是原DT训练数据的证明。

静态代码核查（commit 3e4fcc8a7e68916882033a0cef110370c0144022）：

- `envs/reacher_2d.py` 与本地固定官方文件字节一致，SHA256=894ead0c4a1a7687476d34119844890459b49b452949a80cc5e08da892cf1646。
- 有PPO训练、权重加载和轨迹采集流程，可以作为重建数据的起点。
- `test.py`设置50，但从episode1开始循环到小于50，且只接受回报>20的回合，当前代码可能收集49条满足筛选的轨迹。
- pickle保存语句被注释，不能原封不动执行后就声称已生成数据。
- README有Google Drive链接，本次未验证其中文件就是原DT的medium/replay/expert三套原始数据。

结论：找到具体的环境和数据生成参考；原论文Reacher数据身份仍未核实，不能把LaMo的50回合数据当原DT的实验输入。

## Key-to-Door

MIT 6.8200课程项目 [Improving Credit Assignment with Sequence Models](https://github.com/courtneyma/Improving-Credit-Assignment-with-Sequence-Models)，commit 389f3ec4f5ad9c2cc1e677e74f03bb3ae183d501，公开 `6_8200_DT_S4_keytodoor.ipynb`。

实际实现基于MiniGrid，自定义三个阶段；按时间步将agent传送到钥匙房、干扰房、门房，只有携带钥匙开门才得到奖励。代码包含环境、数据生成、模型训练、评估和录像。

与严格复现有关的差异：

- 当前可见配置为225步回合、batch1、20000次更新、1659条数据，而非完整复用原论文1K/10K协议。
- 部分随机采集禁用drop动作；后续还从私有路径加载额外成功轨迹并合并。
- 数据与checkpoint路径依赖Google Drive，本次仓库树中没有这些pkl/权重文件。
- README声称DT可以解决任务；但下载版本的notebook保存了58个训练评估均值，全部为0。此观察只说明该文件保存的那次执行没有展示成功，不能推断整个项目从未有过成功运行。

结论：可作为环境与训练重实现的参考，不能据README声称已经复现原论文71.8%/94.6%。还需统一环境、纯随机数据比例、训练预算和受控多种子评估。

## 图最短路

官方仓库 [Issue83：Missing Graph Shortest Path Code](https://github.com/kzl/decision-transformer/issues/83) 也有人报告找不到图实验代码，页面显示仍开放。

同一MIT项目公开 `6_8200_decision_transformer_random_walk.ipynb`，使用NetworkX和walker生成轨迹，增加动作与回报预测，并将生成路径长度与NetworkX最短路比较。包含允许无效动作停留和不含此类停留的不同实验变体。

静态核查发现：

- notebook混合多个数据版本，部分图、数据和checkpoint来自私有Google Drive路径；并非当前服务器可以顺序执行的独立脚本。
- 有1000条轨迹训练片段，也有8000条随机游走准备及T50数据路径，不能仅看到一个1000就认定全套协议一致。
- 一个训练片段在return logits上加10倍先验；原论文说明的是推断时将回报概率与先验的10次幂结合，两者不应未经论证视为相同。
- 一个评估片段固定初始RTG再递减，而非展示完整的模型回报分布加先验采样流程。
- 因此它能说明别人怎样搭建图任务，但不能直接作为已通过原论文图6复现的证据。

## 对当前项目的影响

- 保持当前Hopper队列不动。
- Atari优先保留原论文数据/采样协议，参考教学工程的数据读取方案，单独环境验证。
- Reacher从已确认相同的环境与PPO生成参考继续审查；自生成数据需明确标注。
- Key-to-Door和图任务可从第三方notebook重构，但仍需审查补齐，不解除正式目录中的阻塞状态，不伪称官方实现。

下载与固定提交清单：`experiments/research/sources_20261106/manifest.json`。`.review.txt`是从notebook提取的代码文字，仅供审阅，从未执行。

## 2026-11-06 按论文重新审查后的选择

这是路线决定，尚未实施新的第三方训练；当前两路Hopper任务保持不变。

| 实验 | 选择 | 验收边界 |
|---|---|---|
| Atari | 固定官方实现为主体；从Breakout开始，第三方仅参考读取工程 | 500000步/50buffers、3seeds、Table8参数；不采用5buffers或教学版Pong batch128替代 |
| 图最短路 | 按附录A.3重写独立模块，第三方仅作参考 | 20节点、边稀疏系数0.1、1000条10步随机游走、无效动作停留、同时预测动作和回报；推断时使用回报先验，不由最短路oracle给策略提示 |
| Key-to-Door | 根据DT三阶段变体重建；参考MIT环境结构，不照搬其数据加工 | DT使用完整随机数据，1K/10K；仅%BC筛成功轨迹；全episode上下文、末段开门且已拿钥匙才奖励1。未报告的环境尺寸/阶段时长/训练设置列为重建假设 |
| Reacher | 官方Reacher2d环境保留；原始数据继续寻找，自生成另列重建实验 | 可以参考LaMo PPO代码，但不把其50回合、回报筛选数据当原DT的medium/replay/expert；未核实身份前不直接与原论文分数验收 |

Atari先选Breakout是工程排序决定：其K30/batch128比Pong K50/batch512更便于先验证原协议，未改变论文给该游戏的设置。GPU当前两路已占用，准备依赖/数据不等于再启动第三个训练。

图实验强调的是从随机游走生成更短路径，最短路算法只用于评估，不能参与动作选择或提供真实最优RTG。原文回报先验的索引/符号约定需要明确记录，不能为追求结果偷偷借用最优路径长度。未报告训练超参会显式列为实现选择，因此初版标为“按论文机制重实现”，不声称逐项恢复作者训练。

Key-to-Door原始参考Mesnard的环境中第二阶段有苹果奖励；DT使用的变体描述为空房。不能把Mesnard的全部奖励机制原样搬入DT实验，也不能把普通MiniGrid DoorKey视为已经相同。参考：[DT第5.4节](https://arxiv.org/html/2106.01345v2#S5.SS4)、[Mesnard原文](https://arxiv.org/html/2011.09464)。

优先顺序：继续现有Gym队列，同时准备Atari官方路线；随后实现低成本图任务，再重建Key-to-Door。Reacher自生成数据留在明确标注的重建组。按用户要求追求忠实度，不以能运行或高分替代协议核对。
