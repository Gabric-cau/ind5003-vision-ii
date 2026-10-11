"""Portable device selection, FFmpeg discovery and inference provenance."""
from pathlib import Path
import hashlib
import importlib
import os
import platform
import re
import shutil
import sys


def resolve_device(requested='auto', torch_module=None):
    requested = str(requested).strip().lower()
    if requested == 'cuda':
        requested = 'cuda:0'
    if requested not in ('auto', 'cpu', 'mps') and not re.fullmatch(r'cuda:\d+', requested):
        raise ValueError('Device must be auto, cpu, mps or cuda:<index>')
    if requested == 'cpu':
        return 'cpu'
    try:
        torch = torch_module or importlib.import_module('torch')
    except ImportError:
        if requested == 'auto':
            return 'cpu'
        raise RuntimeError(f'{requested} requires a working PyTorch installation') from None
    cuda = torch.cuda.is_available()
    mps = getattr(getattr(torch, 'backends', None), 'mps', None)
    mps_available = bool(mps and mps.is_available())
    if requested == 'auto':
        return 'cuda:0' if cuda else ('mps' if mps_available else 'cpu')
    if requested.startswith('cuda:'):
        index = int(requested.split(':')[1])
        if not cuda or index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested {requested} is unavailable; select --device cpu or fix CUDA')
    elif requested == 'mps' and not mps_available:
        raise RuntimeError('Requested mps is unavailable; select --device cpu or auto')
    return requested


def resolve_ffmpeg(configured=None, root=None):
    configured = configured or os.environ.get('VISIONII_FFMPEG')
    if configured:
        candidate = shutil.which(str(configured))
        if candidate is None:
            path = Path(configured).expanduser()
            candidate = str(path.resolve()) if path.is_file() else None
        if candidate and _usable_ffmpeg(Path(candidate)):
            return candidate
        raise FileNotFoundError(f'FFmpeg is not executable on {sys.platform}: {configured}')
    system = shutil.which('ffmpeg')
    if system and _usable_ffmpeg(Path(system)):
        return system
    if root:
        folder = Path(root) / 'outputs/review_tools/imageio_ffmpeg/binaries'
        for path in sorted(folder.glob('ffmpeg*')):
            if _usable_ffmpeg(path):
                return str(path.resolve())
    raise FileNotFoundError('FFmpeg not found. Install system FFmpeg or pass --ffmpeg /path/to/ffmpeg')


def _usable_ffmpeg(path):
    return path.is_file() and (sys.platform == 'win32' or
                              (path.suffix.lower() != '.exe' and os.access(path, os.X_OK)))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def inference_provenance(requested, device, model_paths=(), source_paths=(), torch_module=None):
    record = {'requested_device': str(requested), 'device': device,
              'python': platform.python_version(), 'platform': platform.system(),
              'model_sha256': {str(p): file_sha256(p) for p in model_paths},
              'source_sha256': {str(p): file_sha256(p) for p in source_paths}}
    try:
        torch = torch_module or importlib.import_module('torch')
        record.update(torch_version=str(torch.__version__), cuda_runtime=getattr(torch.version, 'cuda', None))
        if device.startswith('cuda:'):
            record['gpu_name'] = torch.cuda.get_device_name(int(device.split(':')[1]))
    except ImportError:
        record['torch_version'] = None
    return record


def add_runtime_arguments(parser, ffmpeg=True):
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or cuda:<index>; explicit unavailable devices fail')
    if ffmpeg:
        parser.add_argument('--ffmpeg', help='FFmpeg executable; default: VISIONII_FFMPEG, PATH, compatible bundled binary')
