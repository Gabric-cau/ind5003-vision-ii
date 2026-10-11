"""Resolve packaged input videos without relying on the exporting machine."""
from pathlib import Path, PureWindowsPath


def resolve_project_video(root, recorded_path, clip_id, source_dir=None):
    root = Path(root).resolve()
    if Path(clip_id).name != clip_id or '/' in clip_id or '\\' in clip_id:
        raise ValueError('clip_id must be a filename stem')
    local = (Path(source_dir).expanduser().resolve() if source_dir else root / 'data' / 'raw_clips') / f'{clip_id}.mp4'
    if local.is_file():
        return local
    if source_dir is not None:
        raise FileNotFoundError(f'Explicit source clip is missing: {local}')
    recorded = Path(recorded_path)
    if not recorded.is_absolute():
        recorded = root / recorded
    recorded = recorded.resolve()
    # Never silently read an old checkout if this package lacks its own input.
    if recorded.is_relative_to(root) and recorded.is_file():
        return recorded
    raise FileNotFoundError(f'Packaged video is missing: {local}. Extract the complete project first.')


def resolve_project_file(root, recorded_path):
    """Relocate a recorded project artifact; never read an old external checkout."""
    root = Path(root).resolve()
    recorded = str(recorded_path)
    path = Path(recorded)
    if path.is_absolute() and path.resolve().is_relative_to(root) and path.is_file():
        return path.resolve()
    if not path.is_absolute() and not PureWindowsPath(recorded).is_absolute():
        relative = Path(*PureWindowsPath(recorded).parts) if '\\' in recorded else path
        candidate = (root / relative).resolve()
    else:
        parts = PureWindowsPath(recorded).parts if '\\' in recorded else path.parts
        indexes = [i for i, part in enumerate(parts) if part in ('outputs', 'models', 'data')]
        if not indexes:
            raise FileNotFoundError(f'Cannot relocate project artifact: {recorded}')
        candidate = (root / Path(*parts[indexes[-1]:])).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate
    raise FileNotFoundError(f'Project artifact is missing after relocation: {candidate}')
