"""Resolve packaged input videos without relying on the exporting machine."""
from pathlib import Path


def resolve_project_video(root, recorded_path, clip_id):
    root = Path(root).resolve()
    if Path(clip_id).name != clip_id or '/' in clip_id or '\\' in clip_id:
        raise ValueError('clip_id must be a filename stem')
    local = root / 'data' / 'raw_clips' / f'{clip_id}.mp4'
    if local.is_file():
        return local
    recorded = Path(recorded_path)
    if not recorded.is_absolute():
        recorded = root / recorded
    recorded = recorded.resolve()
    # Never silently read an old checkout if this package lacks its own input.
    if recorded.is_relative_to(root) and recorded.is_file():
        return recorded
    raise FileNotFoundError(f'Packaged video is missing: {local}. Extract the complete project first.')
