import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from restore_project_files import restore, safe_path


class RestoreProjectFilesTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / 'tmp/restore_tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(temporary_root.resolve()))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'docs').mkdir()
        (self.root / 'project_files').mkdir()
        self.payload = b'fixture model bytes\x00\x01'
        self.package = self.root / 'project_files/fixture.zip'
        with zipfile.ZipFile(self.package, 'w') as package:
            package.writestr('models/weights/fixture.pt', self.payload)
        self.manifest = {
            'schema_version': 1, 'file_count': 1,
            'archives': [{'path': 'project_files/fixture.zip',
                          'bytes': self.package.stat().st_size,
                          'sha256': hashlib.sha256(self.package.read_bytes()).hexdigest()}],
            'files': [{'path': 'models/weights/fixture.pt', 'bytes': len(self.payload),
                       'sha256': hashlib.sha256(self.payload).hexdigest(),
                       'archive': 'project_files/fixture.zip'}],
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'docs/git_supplement_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def run_quiet(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return restore(self.root, **kwargs)

    def test_restore_is_repeatable_and_check_detects_modified_file(self):
        self.assertFalse(self.run_quiet())
        target = self.root / 'models/weights/fixture.pt'
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse(self.run_quiet())
        target.write_bytes(b'modified model')
        self.assertTrue(self.run_quiet(check_only=True))
        with self.assertRaises(FileExistsError):
            self.run_quiet()
        self.assertFalse(self.run_quiet(overwrite=True))
        self.assertEqual(target.read_bytes(), self.payload)

    def test_lfs_pointer_has_actionable_error(self):
        self.package.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 456\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            self.run_quiet()

    def test_changed_archive_rejected_before_restoration(self):
        with self.package.open('ab') as writer:
            writer.write(b'damaged')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_quiet()
        self.assertFalse((self.root / 'models').exists())

    def test_manifest_cannot_write_outside_payload_directories(self):
        for relative in ['../escape.txt', 'outputs/../../escape.txt', 'C:/escape.txt',
                         '.git/config', 'outputs\\escape.txt']:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(self.root, relative, payload=True)

    def test_unlisted_archive_entries_are_rejected(self):
        with zipfile.ZipFile(self.package, 'a') as package:
            package.writestr('outputs/unlisted.json', '{}')
        archive = self.manifest['archives'][0]
        archive.update(bytes=self.package.stat().st_size,
                       sha256=hashlib.sha256(self.package.read_bytes()).hexdigest())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'entries'):
            self.run_quiet()


if __name__ == '__main__':
    unittest.main()
