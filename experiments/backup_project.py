"""制作完整、可校验的项目备份；运行中的文件仅代表逐文件快照。无需导入 torch。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.codex', '.agents', '.aws', 'backups', '__pycache__', '.pytest_cache'}
PRIVATE = {'.env', '.netrc', '.npmrc', '.pypirc', 'credentials', 'credentials.json'}


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'backups' / stamp
    out.mkdir(parents=True, exist_ok=False)
    final = out / ('DT_Reproduction-' + stamp + '.tar.gz')
    partial = final.with_suffix(final.suffix + '.partial')
    files, skipped, changed = [], [], []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED)
        for name in sorted(names):
            p = Path(directory) / name
            if name in PRIVATE or name.endswith('.lock'):
                skipped.append(str(p.relative_to(ROOT)))
            elif not p.is_symlink() and p.is_file():
                files.append(p)
    entries = []
    prefix = 'decision-transformer-master/'
    with tempfile.TemporaryDirectory(prefix='snapshot-', dir=out) as tmp:
        temp = Path(tmp)
        with tarfile.open(partial, 'w:gz', compresslevel=1) as archive:
            for index, p in enumerate(files):
                rel = str(p.relative_to(ROOT))
                snapshot = temp / 'file'
                digest = hashlib.sha256()
                try:
                    # 固定打开文件的初始长度，避免不断追加的日志使复制无限延长。
                    with p.open('rb') as source, snapshot.open('wb') as target:
                        before = os.fstat(source.fileno())
                        remaining = before.st_size
                        while remaining:
                            data = source.read(min(4 * 1024 * 1024, remaining))
                            if not data:
                                raise RuntimeError('file truncated during backup: ' + rel)
                            target.write(data)
                            digest.update(data)
                            remaining -= len(data)
                        after = os.fstat(source.fileno())
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        changed.append(rel)
                    archive.add(snapshot, arcname=prefix + rel, recursive=False)
                    entries.append({'path': rel, 'size': before.st_size, 'sha256': digest.hexdigest()})
                except FileNotFoundError:
                    skipped.append(rel + ' (disappeared during snapshot)')
                if index % 1000 == 0:
                    print('Snapshot files: %d/%d' % (index + 1, len(files)), flush=True)
            # Python/Conda 清单只读取已安装元信息，不初始化 GPU。
            for env in ['decision-transformer-gym', 'decision-transformer-atari']:
                python = Path('/home/ubuntu/miniconda3/envs') / env / 'bin/python'
                if python.exists():
                    result = subprocess.run([str(python), '-m', 'pip', 'freeze'],
                                            capture_output=True, text=True, timeout=90)
                    if result.returncode:
                        raise RuntimeError('pip freeze failed: ' + env)
                    export = temp / (env + '-pip-freeze.txt')
                    export.write_text(result.stdout)
                    archive.add(export, arcname=prefix + 'backup_metadata/' + export.name)
            manifest = {
                'started_utc': stamp, 'scope': 'project files including datasets and training artifacts',
                'live_snapshot': True,
                'limitation': 'Files captured individually; active experiments are incomplete. Not an atomic training resume snapshot.',
                'not_included': ['installed Conda environment binaries', 'external MuJoCo installation', 'credentials'],
                'changed_during_copy': changed, 'skipped': skipped, 'files': entries,
            }
            manifest_path = temp / 'MANIFEST.json'
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.add(manifest_path, arcname=prefix + 'backup_metadata/MANIFEST.json')
            # 保留项目的结果入口链接，不跟随链接重复归档。
            link = ROOT / 'experiments/runs'
            if link.is_symlink():
                archive.add(link, arcname=prefix + 'experiments/runs', recursive=False)
        partial.replace(final)
    print('Verifying archive and every captured file...', flush=True)
    expected = {prefix + e['path']: e for e in entries}
    with tarfile.open(final, 'r|gz') as archive:
        for member in archive:
            if member.name not in expected:
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                for data in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    digest.update(data)
            if digest.hexdigest() != expected.pop(member.name)['sha256']:
                raise RuntimeError('Archive verification failed: ' + member.name)
    if expected:
        raise RuntimeError('Archive is missing files')
    digest = hashlib.sha256()
    with final.open('rb') as source:
        for data in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(data)
    (out / 'SHA256SUMS').write_text(digest.hexdigest() + '  ' + final.name + '\n')
    summary = {'archive': str(final), 'bytes': final.stat().st_size, 'file_count': len(entries),
               'sha256': digest.hexdigest(), 'verified': True, 'live_snapshot': True,
               'changed_during_copy': changed, 'skipped': skipped}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
