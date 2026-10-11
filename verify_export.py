"""Verify an extracted project against its SHA-256 export manifest (stdlib)."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = ROOT / 'EXPORT_MANIFEST.json'
    if not manifest_path.exists():
        print('Run this checker inside the extracted full export, next to EXPORT_MANIFEST.json.')
        return 1
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    for index, row in enumerate(manifest['files'], 1):
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT):
            errors.append(f'Unsafe path: {row["path"]}')
        elif not path.is_file():
            errors.append(f'Missing: {row["path"]}')
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append(f'Changed: {row["path"]}')
        if index % 500 == 0:
            print(f'Checked {index}/{len(manifest["files"])} files', flush=True)
    result = {'files_checked':len(manifest['files']),'all_file_hashes_match':not errors,
              'errors':errors,'accuracy_validated':False}
    (ROOT / 'export_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
