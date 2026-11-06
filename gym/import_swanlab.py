'''
将已有 metrics.jsonl 导入 SwanLab，便于直接比较已有 TensorBoard 曲线。
只导入实际存在的轮级指标，不伪造逐步 loss、动画或模型。
'''
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from swanlab_observer import SwanLabObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--swanlab_project', default='decision-transformer-reproduction')
    parser.add_argument('--swanlab_workspace', default=None)
    parser.add_argument('--swanlab_mode', choices=['online', 'offline'], default='offline')
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run_dir / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    output = args.run_dir / 'swanlab_import'
    output.mkdir(exist_ok=True)
    config = {**vars(args), 'run_dir': str(args.run_dir.resolve()),
              'data_scope': 'existing iteration summaries only',
              'warmup_note': '旧 imported-official 运行的实际 warmup 为10000，不是论文100000'}
    logger = SwanLabObserver(SimpleNamespace(**config), output.resolve())
    try:
        for record in records:
            update = record['global_step']
            for tag, value in record.items():
                if isinstance(value, (int, float)):
                    logger.scalar(tag, value, update, update)
            logger.flush()
    finally:
        logger.close()
    print('Imported actual iteration summaries:', len(records), 'Output:', output)


if __name__ == '__main__':
    main()
