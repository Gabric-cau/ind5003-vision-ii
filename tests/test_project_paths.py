import tempfile
import unittest
from pathlib import Path
from src.project_paths import resolve_project_video


def local_temp():
    target=Path(__file__).resolve().parents[1]/'tmp/test_project_paths'
    target.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=target)


class ProjectVideoPathsTest(unittest.TestCase):
    def test_relocated_clip_wins_over_existing_old_checkout(self):
        with local_temp() as folder:
            parent = Path(folder)
            old = parent / 'old.mp4'; old.write_bytes(b'old')
            root = parent / 'new project'; local = root / 'data/raw_clips/example.mp4'
            local.parent.mkdir(parents=True); local.write_bytes(b'new')
            self.assertEqual(resolve_project_video(root, old, 'example'), local.resolve())

    def test_missing_packaged_clip_does_not_fall_back_to_old_checkout(self):
        with local_temp() as folder:
            parent = Path(folder); old = parent / 'old.mp4'; old.write_bytes(b'old')
            with self.assertRaises(FileNotFoundError):
                resolve_project_video(parent / 'new', old, 'example')

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_project_video('.', 'any.mp4', '../outside')
