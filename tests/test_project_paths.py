import tempfile
import unittest
from pathlib import Path
from src.project_paths import resolve_project_video, resolve_project_file


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

    def test_explicit_source_directory_is_supported(self):
        with local_temp() as folder:
            parent=Path(folder);source=parent/'shared clips';source.mkdir()
            clip=source/'example.mp4';clip.write_bytes(b'video')
            self.assertEqual(resolve_project_video(parent/'project','C:\\old\\example.mp4','example',source),clip)

    def test_explicit_missing_source_does_not_use_other_local_video(self):
        with local_temp() as folder:
            root=Path(folder);old=root/'data/raw_clips/example.mp4'
            old.parent.mkdir(parents=True);old.write_bytes(b'old')
            with self.assertRaises(FileNotFoundError):
                resolve_project_video(root,old,'example',root/'new source')

    def test_windows_and_posix_artifact_paths_relocate_to_this_checkout(self):
        with local_temp() as folder:
            root=Path(folder);artifact=root/'outputs/poses/example.json'
            artifact.parent.mkdir(parents=True);artifact.write_text('{}')
            for recorded in ('C:\\old\\project\\outputs\\poses\\example.json',
                             '/old/project/outputs/poses/example.json','outputs\\poses\\example.json'):
                self.assertEqual(resolve_project_file(root,recorded),artifact)

    def test_missing_relocated_artifact_does_not_read_old_checkout(self):
        with local_temp() as folder:
            parent=Path(folder);old=parent/'old/outputs/pose.json'
            old.parent.mkdir(parents=True);old.write_text('{}')
            with self.assertRaises(FileNotFoundError):
                resolve_project_file(parent/'new',old)

    def test_artifact_traversal_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            resolve_project_file('.', 'outputs/../../outside.json')
