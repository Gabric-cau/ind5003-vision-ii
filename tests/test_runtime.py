import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.runtime import resolve_device, resolve_ffmpeg, inference_provenance


def fake_torch(cuda=False, mps=False, count=1):
    return SimpleNamespace(
        __version__='test', version=SimpleNamespace(cuda='test-cuda'),
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count,
                             get_device_name=lambda index: f'Test GPU {index}'),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


class DeviceSelectionTest(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(resolve_device('auto', fake_torch(True, True)), 'cuda:0')
        self.assertEqual(resolve_device('auto', fake_torch(False, True)), 'mps')
        self.assertEqual(resolve_device('auto', fake_torch()), 'cpu')

    def test_explicit_device_never_silently_falls_back(self):
        with self.assertRaises(RuntimeError):
            resolve_device('cuda:0', fake_torch())
        with self.assertRaises(RuntimeError):
            resolve_device('cuda:1', fake_torch(True, count=1))
        with self.assertRaises(RuntimeError):
            resolve_device('mps', fake_torch())
        self.assertEqual(resolve_device('cuda:1', fake_torch(True, count=2)), 'cuda:1')

    def test_auto_and_cpu_work_without_torch(self):
        with patch('src.runtime.importlib.import_module', side_effect=ImportError):
            self.assertEqual(resolve_device('auto'), 'cpu')
            self.assertEqual(resolve_device('cpu'), 'cpu')
            with self.assertRaises(RuntimeError):
                resolve_device('cuda:0')

    def test_invalid_device_is_rejected(self):
        for device in ('gpu', 'cuda:-1', 'cuda:foo', 'cuda:0,1'):
            with self.assertRaises(ValueError):
                resolve_device(device, fake_torch(True))


class FFmpegDiscoveryTest(unittest.TestCase):
    def test_system_ffmpeg_precedes_windows_bundle_on_linux(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root/'ffmpeg'; binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o755)
            windows = root/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win.exe'
            windows.parent.mkdir(parents=True); windows.write_bytes(b'MZ')
            with patch('src.runtime.sys.platform','linux'), patch.dict('os.environ',{},clear=True), \
                 patch('src.runtime.shutil.which',return_value=str(binary)):
                self.assertEqual(resolve_ffmpeg(root=root),str(binary))

    def test_windows_binary_is_rejected_on_linux(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder)/'ffmpeg.exe'; binary.write_bytes(b'MZ'); binary.chmod(0o755)
            with patch('src.runtime.sys.platform','linux'), patch('src.runtime.shutil.which',return_value=None):
                with self.assertRaises(FileNotFoundError):
                    resolve_ffmpeg(binary)

    def test_environment_can_select_executable_with_spaces(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder)/'media tools/ffmpeg'
            binary.parent.mkdir(); binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o755)
            with patch.dict('os.environ',{'VISIONII_FFMPEG':str(binary)}), \
                 patch('src.runtime.shutil.which',return_value=None):
                self.assertEqual(resolve_ffmpeg(),str(binary.resolve()))


class RuntimeProvenanceTest(unittest.TestCase):
    def test_records_real_source_and_model_hashes_and_selected_device(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'model.pt'; model.write_bytes(b'model')
            video = Path(folder)/'clip.mp4'; video.write_bytes(b'source')
            result = inference_provenance('auto','cuda:0',[model],[video],fake_torch(True))
            self.assertEqual(result['requested_device'],'auto')
            self.assertEqual(result['device'],'cuda:0')
            self.assertEqual(result['gpu_name'],'Test GPU 0')
            self.assertEqual(result['model_sha256'][str(model)],hashlib.sha256(b'model').hexdigest())
            self.assertEqual(result['source_sha256'][str(video)],hashlib.sha256(b'source').hexdigest())


if __name__ == '__main__':
    unittest.main()
