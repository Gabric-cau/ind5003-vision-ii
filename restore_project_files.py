"""Restore Git LFS project assets using Python's standard library."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
BUFFER = 4 * 1024 * 1024
PAYLOAD_PREFIXES = ('models/weights/', 'outputs/', 'notebooks/archive/',
                    '.deps_goalkeeper_segmentation/')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(BUFFER), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative, payload=False):
    parts = PurePosixPath(relative).parts
    if not parts or '\\' in relative or ':' in relative or relative.startswith('/') or '..' in parts:
        raise ValueError(f'Unsafe manifest path: {relative}')
    if payload and not relative.startswith(PAYLOAD_PREFIXES):
        raise ValueError(f'Unexpected payload location: {relative}')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves the project directory: {relative}')
    return target


def file_matches(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and sha256(path) == record['sha256']


def restore(root, check_only=False, overwrite=False):
    root = root.resolve()
    manifest = json.loads((root / 'docs/git_supplement_manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    if manifest['schema_version'] != 1 or len(files) != manifest['file_count']:
        raise ValueError('Invalid supplement manifest')
    if len(set(row['path'] for row in files)) != len(files):
        raise ValueError('Duplicate payload paths')
    archive_paths = {row['path'] for row in manifest['archives']}
    if len(archive_paths) != len(manifest['archives']):
        raise ValueError('Duplicate archive paths')
    targets = {row['path']: safe_path(root, row['path'], payload=True) for row in files}
    if any(row['archive'] not in archive_paths for row in files):
        raise ValueError('Manifest refers to an unknown archive')
    if not check_only:
        for archive in manifest['archives']:
            if not archive['path'].startswith('project_files/'):
                raise ValueError('Unexpected archive location')
            path = safe_path(root, archive['path'])
            if not path.is_file():
                raise FileNotFoundError(f'Missing {archive["path"]}. Run git lfs pull first.')
            with path.open('rb') as reader:
                pointer = reader.read(150)
            if pointer.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('Only Git LFS pointers were downloaded. Install Git LFS, then run git lfs install and git lfs pull.')
            if not file_matches(path, archive):
                raise ValueError(f'Archive checksum mismatch: {archive["path"]}. Download it again with Git LFS.')
        conflicts = [r['path'] for r in files if targets[r['path']].exists() and not file_matches(targets[r['path']], r)]
        if conflicts and not overwrite:
            raise FileExistsError(f'{len(conflicts)} existing files differ from the snapshot. First: {conflicts[0]}. Use a fresh clone, or --overwrite if you intend to replace these files.')
        needed = sum(r['bytes'] for r in files if not file_matches(targets[r['path']], r))
        if shutil.disk_usage(root).free < needed + 256 * 1024**2:
            raise RuntimeError(f'Need {needed / 1024**3:.2f} GiB plus 256 MiB free to restore this snapshot.')
        restored, reused = 0, 0
        for archive in manifest['archives']:
            members = [r for r in files if r['archive'] == archive['path']]
            with zipfile.ZipFile(safe_path(root, archive['path'])) as package:
                names = package.namelist()
                if len(names) != len(set(names)) or set(names) != {r['path'] for r in members}:
                    raise ValueError('Archive entries do not match the manifest: ' + archive['path'])
                for row in members:
                    target = safe_path(root, row['path'], payload=True)
                    if file_matches(target, row):
                        reused += 1
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(prefix='.visionii_restore_', dir=target.parent, delete=False) as writer:
                            temporary = Path(writer.name)
                            digest, count = hashlib.sha256(), 0
                            with package.open(row['path']) as reader:
                                for block in iter(lambda: reader.read(BUFFER), b''):
                                    writer.write(block)
                                    digest.update(block)
                                    count += len(block)
                        if count != row['bytes'] or digest.hexdigest() != row['sha256']:
                            raise ValueError('Extracted file hash mismatch: ' + row['path'])
                        os.replace(temporary, target)
                        restored += 1
                    finally:
                        if temporary is not None and temporary.exists():
                            temporary.unlink()
                    if (restored + reused) % 3000 == 0:
                        print(f'Restored/verified {restored + reused}/{len(files)} files', flush=True)
            print(f'Finished {archive["path"]}', flush=True)
    else:
        restored, reused = 0, 0
    # Verify the final files, including any existing files reused above.
    errors = [r['path'] for r in files if not file_matches(targets[r['path']], r)]
    result = {
        'files_checked': len(files), 'all_file_hashes_match': not errors,
        'files_restored': restored, 'matching_existing_files_reused': reused,
        'errors': errors, 'raw_clips_included_in_supplement': False,
        'fresh_environment_all_modules_validated': False,
    }
    if not errors:
        output = safe_path(root, 'outputs/github_supplement_restoration.json', payload=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**result, 'errors': errors[:10]}, ensure_ascii=False, indent=2))
    return bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the restored files without extracting archives')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files that differ from the archived snapshot')
    args = parser.parse_args()
    return restore(ROOT, check_only=args.check, overwrite=args.overwrite)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
