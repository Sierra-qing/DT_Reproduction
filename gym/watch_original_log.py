'''
将原版 experiment.py 已输出的每轮日志同步到 TensorBoard。
不会注入/停止训练进程，不能恢复未记录的逐步指标、种子或模型参数。
'''
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from torch.utils.tensorboard import SummaryWriter
from experiment_observed import reference_scores

REQUIRED = {'time/training', 'time/evaluation', 'training/train_loss_mean', 'training/train_loss_std'}


def parse_iterations(text):
    '''仅返回必要字段齐全的迭代，忽略尚未写完的日志块。'''
    result = {}
    current = None
    for line in text.splitlines():
        match = re.fullmatch(r'Iteration\s+(\d+)', line.strip())
        if match:
            current = int(match.group(1))
            result[current] = {}
        elif current is not None and ':' in line:
            key, value = line.split(':', 1)
            if key.startswith(('time/', 'training/', 'evaluation/')):
                try:
                    result[current][key.strip()] = float(value.strip())
                except ValueError:
                    pass
    return {iteration: values for iteration, values in result.items() if REQUIRED <= values.keys()}


def sync_once(args, writer, seen, output):
    records = parse_iterations(args.log.read_text(errors='replace'))
    references = reference_scores(args.env, args.dataset)
    for iteration, values in sorted(records.items()):
        if iteration in seen:
            continue
        step = iteration * args.steps_per_iter
        writer.add_scalar('train/action_mse_mean', values['training/train_loss_mean'], step)
        writer.add_scalar('train/action_mse_std', values['training/train_loss_std'], step)
        writer.add_scalar('performance/updates_per_second', args.steps_per_iter / max(values['time/training'], 1e-9), step)
        for key, value in values.items():
            if key.startswith('time/'):
                writer.add_scalar(key, value, step)
            match = re.fullmatch(r'evaluation/target_(\d+)_(.+)', key)
            if match:
                target, metric = match.groups()
                writer.add_scalar(f'eval/target_{target}/{metric}', value, step)
                if references is not None and metric in ('return_mean', 'return_std'):
                    low, high = references
                    score = 100 * (value - low) / (high - low) if metric == 'return_mean' else 100 * value / (high - low)
                    writer.add_scalar(f'eval/target_{target}/normalized_score_' + metric[7:], score, step)
        if args.env == 'hopper' and args.dataset == 'medium':
            writer.add_scalar('reference/paper_hopper_medium', 67.6, step)
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({'iteration': iteration, 'global_step': step, **values}) + '\n')
        seen.add(iteration)
        writer.flush()
        print(f'[{datetime.now(timezone.utc).isoformat()}] 已同步第 {iteration} 轮，更新步数 {step}', flush=True)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output_dir', type=Path, default=Path('runs_observed'))
    parser.add_argument('--env', default='hopper')
    parser.add_argument('--dataset', default='medium')
    parser.add_argument('--steps_per_iter', type=int, default=10000)
    parser.add_argument('--poll_seconds', type=float, default=5.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error('日志文件不存在')
    if args.steps_per_iter < 1 or args.poll_seconds <= 0:
        parser.error('步数和轮询时间必须大于 0')
    output = args.output_dir.resolve() / ('imported-official-' + args.log.stem)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'source.json').write_text(json.dumps({
        'source_log': str(args.log.resolve()), 'steps_per_iter': args.steps_per_iter,
        'env': args.env, 'dataset': args.dataset, 'seed': None,
        'scope': '仅每轮聚合；不包含逐步指标或checkpoint；不是单独启动的新实验',
    }, ensure_ascii=False, indent=2))
    seen = set()
    previous = output / 'metrics.jsonl'
    if previous.is_file():
        seen = {json.loads(line)['iteration'] for line in previous.read_text().splitlines() if line.strip()}
    writer = SummaryWriter(log_dir=str(output / 'tensorboard'), flush_secs=5)
    writer.add_text('说明/当前原版进程',
        '此面板来自正在运行的原版日志，只能在一轮训练和评估完成后更新。\n\n'
        '没有逐步 loss、数据采样占比、已知种子或 checkpoint。\n\n'
        'Global step = 日志轮数 × 用户指定的每轮更新次数。\n\n'
        '论文 67.6 是跨种子结果参考线，本次单次评估不能与其标准差直接比较。', 0)
    writer.add_custom_scalars({'评估': {
        '两个RTG条件的回报': ['Multiline', ['eval/target_3600/return_mean', 'eval/target_1800/return_mean']],
        'Hopper评分与论文参考': ['Multiline', ['eval/target_3600/normalized_score_mean',
            'eval/target_1800/normalized_score_mean', 'reference/paper_hopper_medium']],
    }})
    writer.flush()
    try:
        while True:
            sync_once(args, writer, seen, output)
            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        writer.close()


if __name__ == '__main__':
    main()
