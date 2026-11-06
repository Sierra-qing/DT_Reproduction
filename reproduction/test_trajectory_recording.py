'''兼容旧测试路径；实现已整理到 tests/。'''
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tests' / 'test_trajectory_recording.py'), run_name='__main__')
