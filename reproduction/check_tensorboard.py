'''
检查本地 TensorBoard HTTP 页面、插件和当前原版实验曲线。
仅访问 127.0.0.1，不修改训练进程。
'''
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
status = json.loads((ROOT / 'reproduction/setup_logs/dashboard_status.json').read_text())
base = status['url']


def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        assert response.status == 200
        return json.load(response)


if __name__ == '__main__':
    plugins = get('/data/plugins_listing')
    assert plugins['scalars']['enabled']
    tags = get('/data/plugin/scalars/tags')
    run = next(name for name in tags if 'imported-official-hopper-paper-20261105-090102' in name)
    selected = ['train/action_mse_mean', 'eval/target_3600/return_mean',
                'eval/target_3600/normalized_score_mean', 'reference/paper_hopper_medium']
    records = {}
    for tag in selected:
        assert tag in tags[run], tag
        query = urllib.parse.urlencode({'run': run, 'tag': tag})
        samples = get('/data/plugin/scalars/scalars?' + query)
        assert samples and samples[-1][1] >= 10000
        records[tag] = samples[-1]
    result = {'url': base, 'run': run, 'last_samples_walltime_step_value': records, 'http_and_scalars': 'PASS'}
    (ROOT / 'reproduction/setup_logs/dashboard_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
