"""Export the complete project, its videos, weights and results; verify ZIP bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ['assets','configs','data','docs','models','notebooks','output','outputs',
               'scripts','src','tests','.deps_goalkeeper_segmentation']
EXCLUDED_PARTS = {'__pycache__','.pytest_cache','.ipynb_checkpoints'}
STORED_SUFFIXES = {'.mp4','.mov','.avi','.mkv','.pt','.png','.jpg','.jpeg','.webp','.zip','.exe','.dll','.pyd','.pdf'}


def select_files():
    files = [p for p in ROOT.iterdir() if p.is_file() and
             (p.name in {'README.md','run_project.py','verify_export.py'} or p.name.startswith('requirements'))]
    for name in DIRECTORIES:
        folder = ROOT / name
        if folder.exists():
            for base, dirs, names in os.walk(folder):
                dirs[:] = sorted(n for n in dirs if n not in EXCLUDED_PARTS)
                for name in names:
                    p = Path(base)/name
                    if p.suffix.lower() not in {'.pyc','.pyo'}:
                        files.append(p)
    return sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',default='VisionII_Full_Project_2026-10-11')
    args=p.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'deliverables'):
        raise ValueError('Export must be saved under the project deliverables directory')
    if output.exists():
        raise FileExistsError(output)
    if Path(args.prefix).name!=args.prefix or '/' in args.prefix or '\\' in args.prefix:
        raise ValueError('prefix must be a single directory name')
    files=select_files(); total=sum(f.stat().st_size for f in files)
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free < total + 1024**3:
        raise RuntimeError('Insufficient free disk space for the full archive plus 1 GiB reserve')
    prefix=args.prefix+'/'; rows=[];written=0;last_progress=0
    print(f'Exporting {len(files)} files, {total/1024**3:.2f} GiB of source content',flush=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for index,source in enumerate(files,1):
            before=source.stat();relative=source.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo.from_file(source,arcname=prefix+relative)
            info.compress_type=zipfile.ZIP_STORED if source.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            h=hashlib.sha256();size=0
            with source.open('rb') as stream, archive.open(info,'w',force_zip64=True) as target:
                for block in iter(lambda:stream.read(4*1024*1024),b''):
                    target.write(block);h.update(block);size+=len(block);written+=len(block)
                    if written-last_progress>=512*1024*1024:
                        print(f'Packed {written/1024**3:.2f}/{total/1024**3:.2f} GiB; {index}/{len(files)} files',flush=True)
                        last_progress=written
            after=source.stat()
            if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or size!=before.st_size:
                raise RuntimeError(f'Source changed during snapshot: {relative}')
            rows.append({'path':relative,'bytes':size,'sha256':h.hexdigest()})
        manifest={'created_at':datetime.now(timezone.utc).isoformat(),'package_kind':'complete_project_with_inputs_weights_outputs',
            'prefix':args.prefix,'source_bytes':total,'file_count':len(rows),'files':rows,
            'excluded':['.venv','.idea','.pytest_cache','.deps_ball_mask','.deps_mask','.deps_jersey','tmp','deliverables','__pycache__','*.pyc','*.pyo'],
            'sam2_source_and_helpers_included':True,'human_verified':False,'accuracy_validated':False,
            'install_environment_separately':True,'fresh_environment_install_tested':False}
        archive.writestr(prefix+'EXPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Archive created; verifying every entry CRC and SHA-256',flush=True)
    checked=0
    with zipfile.ZipFile(output) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or len(names)!=len(rows)+1:
            raise ValueError('Archive entry count/uniqueness failed')
        for index,row in enumerate(rows,1):
            h=hashlib.sha256();size=0
            with archive.open(prefix+row['path']) as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):
                    h.update(block);size+=len(block)
            if h.hexdigest()!=row['sha256'] or size!=row['bytes']:
                raise ValueError(f'Archive integrity failed: {row["path"]}')
            checked+=1
            if index%2000==0:
                print(f'Verified {index}/{len(rows)} entries',flush=True)
        packaged=json.loads(archive.read(prefix+'EXPORT_MANIFEST.json'))
        if packaged!=manifest:
            raise ValueError('Manifest verification failed')
    result={'zip':str(output),'zip_bytes':output.stat().st_size,'zip_sha256':sha(output),
        'source_bytes':total,'files_verified':checked,'all_entry_crc_and_sha256_verified':True,
        'models_included':len([r for r in rows if r['path'].startswith('models/') and r['path'].endswith('.pt')]),
        'canonical_raw_clip_videos_included':len([r for r in rows if r['path'].startswith('data/raw_clips/') and r['path'].endswith('.mp4')]),
        'readme_included':any(r['path']=='README.md' for r in rows),
        'pycharm_launcher_included':any(r['path']=='run_project.py' for r in rows),
        'fresh_environment_install_tested':False,'accuracy_validated':False}
    sidecar=output.with_name(output.stem+'_validation.json')
    sidecar.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
