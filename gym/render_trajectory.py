'''
将已记录的真实 MuJoCo 物理状态离屏渲染为 GIF 动画。
不调用策略，不执行 env.step，不生成新的评估结果；动画对应原回合的实际状态。
'''
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_trajectory(source, output, stride=4, width=480, height=360):
    import gym
    import mujoco_py
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(data['metadata_json'].item())
        required = ['physics_qpos', 'physics_qvel', 'physics_time', 'physics_act']
        if not all(key in data.files for key in required):
            raise ValueError('此轨迹没有完整物理状态，不能从观测向量伪造动画')
        arrays = {key: data[key].copy() for key in data.files if key != 'metadata_json'}
    env_id = metadata.get('environment_id')
    if not env_id:
        raise ValueError('轨迹未记录环境 ID')
    env = gym.make(env_id)
    base = env.unwrapped
    indices = list(range(0, len(arrays['physics_time']), stride))
    if indices[-1] != len(arrays['physics_time']) - 1:
        indices.append(len(arrays['physics_time']) - 1)
    frames = []
    try:
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 13)
        except OSError:
            font = ImageFont.load_default()
        for index in indices:
            state = mujoco_py.MjSimState(
                float(arrays['physics_time'][index]), arrays['physics_qpos'][index],
                arrays['physics_qvel'][index], arrays['physics_act'][index], base.sim.get_state().udd_state,
            )
            base.sim.set_state(state)
            if 'physics_ctrl' in arrays:
                base.sim.data.ctrl[:] = arrays['physics_ctrl'][index]
            if 'physics_qacc_warmstart' in arrays:
                base.sim.data.qacc_warmstart[:] = arrays['physics_qacc_warmstart'][index]
            base.sim.forward()
            pixels = env.render(mode='rgb_array', width=width, height=height)
            frame = Image.fromarray(pixels)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, 44), fill=(0, 0, 0))
            reward_so_far = float(arrays['cumulative_reward'][index - 1]) if index else 0.
            draw.text((6, 3), f"iteration {metadata['iteration']} | target {metadata['target_return']} | episode {metadata['episode']}", font=font, fill=(255, 255, 255))
            draw.text((6, 23), f'state step {index} | return so far {reward_so_far:.2f}', font=font, fill=(255, 255, 255))
            frames.append(frame)
    finally:
        env.close()
    times = arrays['physics_time'][indices]
    intervals = np.diff(times)
    final_interval = float(intervals[-1]) if len(intervals) else .032
    milliseconds = [max(10, int(round(float(value) * 100)) * 10) for value in [*intervals, final_interval]]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:],
                   duration=milliseconds, loop=0, optimize=False, disposal=2)
    details = {
        'source_trajectory': str(source.resolve()), 'output': str(output.resolve()),
        'environment_id': env_id, 'frames': len(frames), 'frame_stride': stride,
        'width': width, 'height': height, 'simulation_duration_seconds': float(times[-1] - times[0]),
        'playback_duration_seconds': sum(milliseconds) / 1000.,
        'playback_note': 'GIF 时间以 10ms 量化，近似实时；评估时间请看原始物理时间数组',
        'rendering_only_no_env_step': True, 'episode_return': metadata['total_return'],
        'complete_episode': metadata['complete'], 'training_step': metadata['global_step'],
    }
    output.with_suffix('.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(json.dumps(details, ensure_ascii=False, indent=2), flush=True)
    return details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()
    if args.stride < 1 or args.width < 64 or args.height < 64:
        parser.error('stride 至少为1，宽高至少64')
    destination = args.output or args.trajectory.parent.parent / 'videos' / (args.trajectory.stem + '.gif')
    render_trajectory(args.trajectory, destination, args.stride, args.width, args.height)
