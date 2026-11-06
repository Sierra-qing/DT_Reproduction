'''验证真实 SwanLab SDK 的离线标量、图像和 GIF；不训练、不上传。'''
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gym'))
from experiment_observed import build_parser
from swanlab_observer import SwanLabObserver

args = build_parser().parse_args(['--log_to_swanlab'])
out = ROOT / 'reproduction/swanlab_validation'
out.mkdir(exist_ok=True)
logger = SwanLabObserver(args, out)
try:
    logger.scalar('train/action_mse_step', .5, 1, 1)
    logger.scalar('train/learning_rate', .000001, 1, 1)
    logger.scalar('train/action_mse_step', .4, 2, 2)
    logger.scalar('episode/target_3600/return', 12., 2, 1)
    prior = ROOT / 'reproduction/trajectory_validation_runs/hopper-medium-dt-seed73-20261105-113216-980488-real-hop-test-not-paper-result'
    gif = next((prior / 'videos').glob('*.gif'))
    png = next((prior / 'trajectory_figures').glob('*.png'))
    logger.media('simulation/test', gif, 2, video=True)
    logger.media('rollout/test', png, 2)
finally:
    logger.close()
assert list(logger.folder.rglob('*.gif')), 'GIF was not persisted'
assert list(logger.folder.rglob('*.png')), 'Diagnostic image was not persisted'
print('PASS real SwanLab offline scalars, separate episode axis, GIF and diagnostic image:', logger.folder)
