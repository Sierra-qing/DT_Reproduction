# 实验覆盖与准备状态

生成时间（UTC）：2026-11-06T11:40:44.257559+00:00

ready只表示静态准备检查通过，不等于已做真实短验证或复现成功；external为明确缺失的实现，不是可运行代码。

| 状态 | 配置数 |
|---|---:|
| blocked | 144 |
| complete | 6 |
| ready | 115 |
| running | 2 |

| 实验ID | 组别 | 状态 | 实现范围 | 缺项 |
|---|---|---|---|---|
| dt-hopper-medium-s0 | gym-dt | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p100-s0 | gym-bc | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p10-s0 | gym-percentile-bc | running | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-s0 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-hopper-medium-s1 | gym-dt | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p100-s1 | gym-bc | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-s1 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-hopper-medium-s2 | gym-dt | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p100-s2 | gym-bc | complete | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-s2 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-hopper-medium-replay-s0 | gym-dt | running | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p10-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-replay-s0 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| sweep-hopper-medium-replay-s0 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-hopper-medium-replay-s0 |
| dt-hopper-medium-replay-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-replay-s1 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| sweep-hopper-medium-replay-s1 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-hopper-medium-replay-s1 |
| dt-hopper-medium-replay-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-replay-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-replay-s2 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| sweep-hopper-medium-replay-s2 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-hopper-medium-replay-s2 |
| dt-hopper-medium-expert-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-expert-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-expert-s0 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-hopper-medium-expert-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-expert-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-expert-s1 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-hopper-medium-expert-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-hopper-medium-expert-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| delayed-hopper-medium-expert-s2 | gym-delayed | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p10-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-replay-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p10-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-halfcheetah-medium-replay-s0 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-halfcheetah-medium-replay-s0 |
| dt-halfcheetah-medium-replay-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-halfcheetah-medium-replay-s1 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-halfcheetah-medium-replay-s1 |
| dt-halfcheetah-medium-replay-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-replay-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-halfcheetah-medium-replay-s2 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-halfcheetah-medium-replay-s2 |
| dt-halfcheetah-medium-expert-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-expert-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-expert-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-expert-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-halfcheetah-medium-expert-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-halfcheetah-medium-expert-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p10-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-replay-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p10-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p25-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p40-s0 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-walker2d-medium-replay-s0 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-walker2d-medium-replay-s0 |
| dt-walker2d-medium-replay-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p10-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p25-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p40-s1 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-walker2d-medium-replay-s1 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-walker2d-medium-replay-s1 |
| dt-walker2d-medium-replay-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p10-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p25-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-replay-p40-s2 | gym-percentile-bc | ready | Gym原版训练/评估包装 |  |
| sweep-walker2d-medium-replay-s2 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-walker2d-medium-replay-s2 |
| dt-walker2d-medium-expert-s0 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-expert-p100-s0 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-expert-s1 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-expert-p100-s1 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-walker2d-medium-expert-s2 | gym-dt | ready | Gym原版训练/评估包装 |  |
| bc-walker2d-medium-expert-p100-s2 | gym-bc | ready | Gym原版训练/评估包装 |  |
| dt-reacher2d-medium-s0 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p100-s0 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p10-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p25-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p40-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| dt-reacher2d-medium-s1 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p100-s1 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p10-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p25-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p40-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| dt-reacher2d-medium-s2 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p100-s2 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p10-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p25-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| bc-reacher2d-medium-p40-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-v2.pkl |
| dt-reacher2d-medium-replay-s0 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p100-s0 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p10-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p25-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p40-s0 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| sweep-reacher2d-medium-replay-s0 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-reacher2d-medium-replay-s0 |
| dt-reacher2d-medium-replay-s1 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p100-s1 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p10-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p25-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p40-s1 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| sweep-reacher2d-medium-replay-s1 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-reacher2d-medium-replay-s1 |
| dt-reacher2d-medium-replay-s2 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p100-s2 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p10-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p25-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| bc-reacher2d-medium-replay-p40-s2 | gym-percentile-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-replay-v2.pkl |
| sweep-reacher2d-medium-replay-s2 | return-sweep | blocked | Gym原版训练/评估包装 | dependency incomplete: dt-reacher2d-medium-replay-s2 |
| dt-reacher2d-medium-expert-s0 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| bc-reacher2d-medium-expert-p100-s0 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| dt-reacher2d-medium-expert-s1 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| bc-reacher2d-medium-expert-p100-s1 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| dt-reacher2d-medium-expert-s2 | gym-dt | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| bc-reacher2d-medium-expert-p100-s2 | gym-bc | blocked | Gym原版训练/评估包装 | missing dataset: gym/data/reacher2d-medium-expert-v2.pkl |
| diagnostic-sweep-hopper-medium-s0 | diagnostic | ready | Gym原版训练/评估包装 |  |
| diagnostic-sweep-hopper-medium-s1 | diagnostic | ready | Gym原版训练/评估包装 |  |
| diagnostic-sweep-hopper-medium-s2 | diagnostic | ready | Gym原版训练/评估包装 |  |
| atari-breakout-dt-s123 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p100-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p40-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p25-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p10-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-k1-s123 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-sweep-breakout-s123 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-breakout-dt-s123 |
| atari-breakout-dt-s231 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p100-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p40-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p25-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p10-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-k1-s231 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-sweep-breakout-s231 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-breakout-dt-s231 |
| atari-breakout-dt-s312 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p100-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p40-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p25-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-bc-p10-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-breakout-k1-s312 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Breakout; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/breakout.bin |
| atari-sweep-breakout-s312 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-breakout-dt-s312 |
| atari-qbert-dt-s123 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p100-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p40-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p25-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p10-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-k1-s123 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-sweep-qbert-s123 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-qbert-dt-s123 |
| atari-qbert-dt-s231 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p100-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p40-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p25-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p10-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-k1-s231 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-sweep-qbert-s231 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-qbert-dt-s231 |
| atari-qbert-dt-s312 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p100-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p40-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p25-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-bc-p10-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-qbert-k1-s312 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Qbert; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/qbert.bin |
| atari-sweep-qbert-s312 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-qbert-dt-s312 |
| atari-pong-dt-s123 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p100-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p40-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p25-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p10-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-k1-s123 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-sweep-pong-s123 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-pong-dt-s123 |
| atari-pong-dt-s231 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p100-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p40-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p25-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p10-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-k1-s231 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-sweep-pong-s231 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-pong-dt-s231 |
| atari-pong-dt-s312 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p100-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p40-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p25-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-bc-p10-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-pong-k1-s312 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Pong; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/pong.bin |
| atari-sweep-pong-s312 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-pong-dt-s312 |
| atari-seaquest-dt-s123 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p100-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p40-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p25-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p10-s123 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-k1-s123 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-sweep-seaquest-s123 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-seaquest-dt-s123 |
| atari-seaquest-dt-s231 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p100-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p40-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p25-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p10-s231 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-k1-s231 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-sweep-seaquest-s231 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-seaquest-dt-s231 |
| atari-seaquest-dt-s312 | atari-dt | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p100-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p40-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p25-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-bc-p10-s312 | atari-bc | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-seaquest-k1-s312 | atari-context | blocked | Atari入口已写，未端到端验证 | missing DQN replay: Seaquest; missing Atari ROM: /home/ubuntu/miniconda3/envs/decision-transformer-atari/lib/python3.7/site-packages/atari_py/atari_roms/seaquest.bin |
| atari-sweep-seaquest-s312 | atari-return-sweep | blocked | Atari入口已写，未端到端验证 | dependency incomplete: atari-seaquest-dt-s312 |
| keydoor-1000-dt | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-1000-bc | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-1000-percentile-bc | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-1000-cql | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-1000-random | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-10000-dt | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-10000-bc | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-10000-percentile-bc | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-10000-cql | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-10000-random | keydoor | blocked | 仅论文规格，未实现 | Pinned official repository does not contain the paper Key-to-Door implementation/data/config; require author source before faithful reproduction |
| keydoor-return-prediction-attention | keydoor-analysis | blocked | 仅论文规格，未实现 | Requires paper-specific Key-to-Door implementation and return-prediction model, absent from pinned repository |
| graph-shortest-path | graph | blocked | 仅论文规格，未实现 | Appendix specifies environment but pinned repository lacks experiment implementation and complete training hyperparameters; avoid inventing official settings |
| graph-reimplementation-s0 | graph-reimplementation | ready | 附录机制重实现，GPU训练待验证 |  |
| graph-reimplementation-s1 | graph-reimplementation | ready | 附录机制重实现，GPU训练待验证 |  |
| graph-reimplementation-s2 | graph-reimplementation | ready | 附录机制重实现，GPU训练待验证 |  |
