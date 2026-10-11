"""Prepare, inspect and run isolated lab jobs over the existing SSH alias."""
from pathlib import Path
import argparse
import json
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = '/home/boris/workspace/ind5003-vision-ii-experiments/2026-10-12'
BASE_PYTHON = '/home/boris/anaconda3/envs/agenticsim/bin/python'
ULTRALYTICS_VERSION = '8.4.175'
MODELS = {
    'yolo11m-pose.pt': (42459307, '29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2'),
    'yolo11n.pt': (5613764, '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'),
    'yolo11m.pt': (40684120, 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95'),
    'sam2_t.pt': (78064050, '94375f988270836169320bd901960c67b5770e8bef3867d70102f01a8b5ca501'),
}


REMOTE_COMMON = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib, json, os, platform, shutil, subprocess, sys
base = Path(sys.argv[1]).resolve()
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()
def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
def gpu_state():
    command = ['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version',
               '--format=csv,noheader,nounits']
    text = subprocess.check_output(command, text=True).strip()
    values = [v.strip() for v in text.splitlines()[0].split(',')]
    return dict(name=values[0],memory_total_mib=int(values[1]),memory_used_mib=int(values[2]),
                utilization_percent=int(values[3]),driver_version=values[4])
'''


REMOTE_STATUS = REMOTE_COMMON + r'''
os.environ['YOLO_CONFIG_DIR'] = str(base/'ultralytics_config')
record = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
              ssh_alias='lab4090',project_base=str(base),
              python=platform.python_version(),platform=platform.system(),
              environment_isolated=True,shared_conda_modified=False,
              shared_site_packages_readonly=True,gpu=gpu_state(),
              disk_free_bytes=shutil.disk_usage(base).free,
              new_source_video_inference=False)
packages = {}
for name in ('numpy','cv2','torch','torchvision','ultralytics','PIL','yaml','scipy','pandas'):
    try:
        module = importlib.import_module(name)
        packages[name] = dict(version=str(getattr(module,'__version__','imported')),
                              location=str(Path(module.__file__).resolve()))
    except Exception as exc:
        packages[name] = dict(error_type=type(exc).__name__,error=str(exc))
record['packages'] = packages
torch = importlib.import_module('torch')
record['cuda_runtime'] = torch.version.cuda
record['cuda_available'] = torch.cuda.is_available()
record['cuda_gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
record['model_files'] = []
from ultralytics import YOLO, SAM
for path in sorted((base/'models').glob('*.pt')):
    model = SAM(str(path)) if path.name.startswith('sam2_') else YOLO(str(path))
    record['model_files'].append(dict(path=str(path.relative_to(base)),bytes=path.stat().st_size,
                                     sha256=digest(path),load_verified=True,task=model.task))
    del model
record['input_files'] = []
for path in sorted((base/'project/data/raw_clips').glob('*.mp4')):
    record['input_files'].append(dict(path=str(path.relative_to(base)),bytes=path.stat().st_size,sha256=digest(path)))
record['source_files'] = []
for folder in ('src','scripts','configs'):
    for path in sorted((base/'project'/folder).rglob('*')):
        if path.is_file() and path.suffix in ('.py','.yaml','.json'):
            record['source_files'].append(dict(path=str(path.relative_to(base)),sha256=digest(path)))
record['job_records'] = [json.loads(path.read_text()) for path in sorted((base/'logs').glob('*/job.json'))]
if record['job_records']:
    record['new_source_video_inference'] = None
    record['inference_evidence'] = 'Inspect individual job outputs and their runtime protocols'
record['ffmpeg'] = subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(base/'pip-freeze.txt').write_text(freeze,encoding='utf-8')
(base/'pip-freeze-local.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--local'],text=True),encoding='utf-8')
record['pip_freeze_sha256'] = digest(base/'pip-freeze.txt')
record['pip_freeze_local_sha256'] = digest(base/'pip-freeze-local.txt')
write_json(base/'runtime.json',record)
print('__LAB_RUNTIME_JSON__'+json.dumps(record,indent=2))
'''


REMOTE_PREPARE = REMOTE_COMMON + r'''
import urllib.request
models = json.loads(sys.argv[2])
for folder in ('models','logs','ultralytics_config','project'):
    (base/folder).mkdir(parents=True,exist_ok=True)
venv = base/'.venv'
if not (venv/'bin/python').is_file():
    subprocess.run([sys.argv[3],'-m','venv','--system-site-packages',str(venv)],check=True)
python = str(venv/'bin/python')
subprocess.run([python,'-m','pip','install','--disable-pip-version-check','--no-input',
                'ultralytics=='+sys.argv[4]],check=True)
receipts = []
for name,(size,expected) in models.items():
    destination = base/'models'/name
    url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'+name
    if destination.exists():
        if destination.stat().st_size!=size or digest(destination)!=expected:
            raise ValueError('Existing model differs from pinned official asset: '+name)
    else:
        temporary = destination.with_suffix('.download')
        try:
            with urllib.request.urlopen(url,timeout=30) as response, temporary.open('wb') as output:
                count = 0
                for chunk in iter(lambda: response.read(1024**2),b''):
                    count += len(chunk)
                    if count>size:
                        raise ValueError('Model transfer exceeds pinned byte length')
                    output.write(chunk)
            if temporary.stat().st_size!=size or digest(temporary)!=expected:
                raise ValueError('Official model does not match pinned manifest bytes/SHA256')
            os.replace(temporary,destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    receipts.append(dict(path=str(destination.relative_to(base)),source_url=url,bytes=size,sha256=expected,
                         original_project_manifest_match=name in ('yolo11n.pt', 'yolo11m-pose.pt'),
                         new_source_video_inference=False))
write_json(base/'model_download_receipts.json',receipts)
print(json.dumps(dict(prepared=True,base=str(base),models=receipts),indent=2))
'''


REMOTE_START = REMOTE_COMMON + r'''
import contextlib, fcntl, runpy, traceback
module,arguments,job,fraction = sys.argv[2],json.loads(sys.argv[3]),sys.argv[4],float(sys.argv[5])
project = base/'project'
if not (project/'src/runtime.py').is_file():
    raise RuntimeError('Upload the project source before starting an experiment')
state = gpu_state()
if state['utilization_percent']>20 or state['memory_total_mib']-state['memory_used_mib']<8192:
    raise RuntimeError('GPU is busy or has less than 8 GiB free; defer this job')
lock = (base/'active_gpu_job.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
logdir = base/'logs'/job
logdir.mkdir(parents=True,exist_ok=False)
record = dict(job_id=job,module=module,arguments=arguments,pid=os.getpid(),status='running',
              started_at_utc=datetime.now(timezone.utc).isoformat(),gpu_before=state,
              torch_memory_fraction_limit=fraction)
write_json(logdir/'job.json',record)
if (base/'runtime.json').is_file():
    record['preparation_runtime_sha256'] = digest(base/'runtime.json')
    shutil.copy2(base/'runtime.json', logdir/'runtime_snapshot.json')
for name in ('pip-freeze.txt', 'pip-freeze-local.txt', 'model_download_receipts.json'):
    if (base/name).is_file():
        shutil.copy2(base/name, logdir/name)
source_files = []
for folder in ('src','scripts','configs'):
    for path in sorted((project/folder).rglob('*')):
        if path.is_file() and path.suffix in ('.py','.yaml','.json') and '__pycache__' not in path.parts:
            relative = path.relative_to(project)
            target = logdir/'source_snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            source_files.append(dict(path=str(relative),sha256=digest(target)))
record['source_snapshot'] = source_files
write_json(logdir/'job.json',record)
os.chdir(project);sys.path.insert(0,str(project))
os.environ['YOLO_CONFIG_DIR'] = str(base/'ultralytics_config')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import torch
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(fraction,0)
sys.argv = [module,*arguments]
with (logdir/'run.log').open('w',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    try:
        runpy.run_module(module,run_name='__main__')
    except SystemExit as exc:
        record['exit_code'] = exc.code or 0
        record['status'] = 'complete' if not record['exit_code'] else 'failed'
        raise
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise
    else:
        record.update(status='complete',exit_code=0)
    finally:
        record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(logdir/'job.json',record)
'''


def remote_python(alias, python, script, *arguments, capture=False):
    command = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',alias,
               shlex.join([python,'-',*map(str,arguments)])]
    return subprocess.run(command,input=script,text=True,capture_output=capture,check=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','status','start','fetch'])
    parser.add_argument('--ssh-alias',default='lab4090')
    parser.add_argument('--remote-base',default=REMOTE_BASE)
    parser.add_argument('--runtime-output',type=Path,default=ROOT/'docs/LAB_RUNTIME_2026-10-12.json')
    parser.add_argument('--module')
    parser.add_argument('--job-id')
    parser.add_argument('--memory-fraction',type=float,default=.35)
    parser.add_argument('--remote-artifact',help='Path relative to the isolated experiment base')
    parser.add_argument('--local-artifact',type=Path)
    args,arguments=parser.parse_known_args(argv)
    if args.mode!='start' and arguments:
        parser.error('Unexpected arguments: '+' '.join(arguments))
    base=Path(args.remote_base)
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',args.ssh_alias):
        parser.error('SSH alias must be an existing simple alias')
    if not base.is_absolute() or '..' in base.parts:
        parser.error('Remote base must be an absolute directory without traversal')
    python=str(base/'.venv/bin/python')
    if args.mode=='prepare':
        remote_python(args.ssh_alias,BASE_PYTHON,REMOTE_PREPARE,base,json.dumps(MODELS),BASE_PYTHON,ULTRALYTICS_VERSION)
        args.mode='status'
    if args.mode=='status':
        result=remote_python(args.ssh_alias,python,REMOTE_STATUS,base,capture=True)
        record=json.loads(result.stdout.split('__LAB_RUNTIME_JSON__',1)[1])
        args.runtime_output.parent.mkdir(parents=True,exist_ok=True)
        args.runtime_output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(record,indent=2))
    elif args.mode=='start':
        if not args.module or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*',args.module):
            parser.error('--module is required and must be a Python module name')
        if not args.job_id or args.job_id in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+',args.job_id):
            parser.error('--job-id must name a new isolated job')
        if not 0<args.memory_fraction<=.5:
            parser.error('--memory-fraction must be greater than 0 and at most 0.5')
        arguments=arguments[1:] if arguments[:1]==['--'] else arguments
        remote_python(args.ssh_alias,python,REMOTE_START,base,args.module,json.dumps(arguments),
                      args.job_id,args.memory_fraction)
    elif args.mode=='fetch':
        if not args.remote_artifact or args.local_artifact is None:
            parser.error('fetch requires --remote-artifact and --local-artifact')
        relative=Path(args.remote_artifact)
        if relative.is_absolute() or '..' in relative.parts:
            parser.error('Artifact must be relative to the isolated experiment base')
        args.local_artifact.parent.mkdir(parents=True,exist_ok=True)
        remote=str(base/relative)
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=10',
                        args.ssh_alias+':'+shlex.quote(remote),str(args.local_artifact)],check=True)


if __name__=='__main__':
    main()
