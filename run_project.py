"""PyCharm entry point: inspect, replay, or rerun individual analysis modules."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from src.runtime import add_runtime_arguments, resolve_device, resolve_ffmpeg

ROOT = Path(__file__).resolve().parent
BALL_REVIEW = ROOT / 'outputs/ball_goal_all_events_v2_2026-10-11'
FRESH_CACHE = ROOT / 'outputs/ball_goal_all_fresh_contact_2026-10-11'
TEST_MODULES = [
    'test_ball_goal_decision', 'test_ball_keeper_rebound', 'test_ball_direction_corridor',
    'test_ball_mask_continuity', 'test_ball_mask_seed', 'test_ball_continuity',
    'test_ball_shot_focus', 'test_project_paths',
]


def bootstrap():
    # Existing local environments used Python 3.14; never load their compiled
    # packages into another Python minor version. Exported ZIP has no .venv.
    if sys.platform == 'win32' and sys.version_info[:2] == (3, 14):
        local = ROOT / '.venv/Lib/site-packages'
        if local.is_dir() and str(local) not in sys.path:
            sys.path.append(str(local))
            os.environ['PYTHONPATH'] = os.pathsep.join(filter(None,
                [os.environ.get('PYTHONPATH'), str(local)]))


def canonical_ids(input_dir=None):
    folder = input_dir or ROOT / 'outputs/ball_mask_review_ready_2026-10-11/inputs'
    protocol = folder.parent / 'protocol.json'
    if protocol.is_file():
        ids = json.loads(protocol.read_text(encoding='utf-8'))['canonical_clips']
    else:
        import yaml
        manifest=yaml.safe_load((ROOT/'configs/penalty_slicing_manifest.yaml').read_text(encoding='utf-8'))
        ids=[f"{match}_p{row['number']:02d}_{row['kicker']}" for match,info in manifest.items() for row in info['penalties']]
    if len(ids) != 39 or len(set(ids)) != 39:
        raise ValueError('Canonical manifest must contain exactly 39 distinct clips')
    return ids


def new_output(value, mode):
    target = Path(value) if value else ROOT / 'outputs' / (mode.replace('-','_') + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not target.is_absolute():
        target = ROOT / target
    target = target.resolve()
    if not target.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Choose a new folder under this project\'s outputs directory')
    if target.exists():
        raise FileExistsError(f'Output already exists; choose a new name: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def invoke(module, *args):
    subprocess.run([sys.executable, '-X', 'utf8', '-m', module, *map(str,args)], cwd=ROOT, check=True)


def test_ball(log=None):
    command = [sys.executable, '-X', 'utf8', '-m', 'unittest', '-v', *['tests.'+n for n in TEST_MODULES]]
    run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    print(run.stdout, end='')
    if log:
        Path(log).write_text(run.stdout, encoding='utf-8')
    if run.returncode:
        raise RuntimeError('Ball and portability tests failed; see the log above')


def check(args):
    print(f'Project: {ROOT}\nPython: {sys.executable}\nVersion: {sys.version.split()[0]}')
    required = {'numpy':'NumPy', 'cv2':'OpenCV', 'PIL':'Pillow', 'yaml':'PyYAML', 'torch':'PyTorch', 'ultralytics':'Ultralytics'}
    failures = []
    for module, label in required.items():
        try:
            package = importlib.import_module(module)
            print(f'OK {label}: {getattr(package,"__version__","imported")}')
        except Exception as exc:
            failures.append(label); print(f'MISSING {label}: {exc}')
    for module in ['scipy','pandas','matplotlib','jupyterlab','pytest']:
        try:
            package = importlib.import_module(module)
            print(f'Optional/full-project {module}: {getattr(package,"__version__","imported")}')
        except Exception:
            print(f'Optional/full-project {module}: not installed; use requirements-pycharm.txt if needed')
    try:
        ids = canonical_ids(args.ball_input_dir)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        ids=[];failures.append(f'Canonical manifest unavailable: {exc}')
    from src.project_paths import resolve_project_video
    for cid in ids:
        try:
            resolve_project_video(ROOT, '', cid, args.source_dir)
        except FileNotFoundError as exc:
            failures.append(str(exc))
        for folder in [args.ball_cache_dir, args.track_dir, args.ball_input_dir, args.fresh_dir]:
            if not (folder / f'{cid}.json').is_file():
                failures.append(f'Missing cache {folder}/{cid}.json')
    for name in ['yolo11n_object_detection.pt','yolo11n_pose.pt','yolo11m_pose.pt','sam2.1_hiera_tiny.pt']:
        if not (ROOT / 'models/weights' / name).is_file():
            failures.append(f'Missing model {name}')
    try:
        print(f'Device: {resolve_device(args.device)}')
        print(f'FFmpeg: {resolve_ffmpeg(args.ffmpeg, ROOT)}')
    except (RuntimeError, FileNotFoundError) as exc:
        failures.append(str(exc))
    if failures:
        raise RuntimeError('\n'.join(failures) + '\nInstall dependencies with: python -m pip install -r requirements-pycharm.txt')
    print('READY: 39 clips, 39 sets of ball caches, four model weights, FFmpeg and required ball imports.')


def replay_ball(args):
    target = new_output(args.output_dir, args.mode)
    ids = canonical_ids(args.ball_input_dir)
    fresh = args.fresh_dir
    ffmpeg = resolve_ffmpeg(args.ffmpeg, ROOT)
    device = resolve_device(args.device) if args.mode == 'refresh-ball' else None
    if args.mode == 'refresh-ball':
        fresh = target.with_name(target.name + '_fresh')
        if fresh.exists():
            raise FileExistsError(fresh)
        invoke('scripts.cache_ball_keeper_contact', '--output-dir', fresh, '--device', device,
               '--source-dir', args.source_dir, '--ball-cache-dir', args.ball_cache_dir,
               '--track-dir', args.track_dir, '--clips', *ids)
    test_temp=ROOT/'tmp';test_temp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='visionii_tests_',dir=test_temp) as temp:
        log = Path(temp) / 'ball_tests.log'
        test_ball(log)
        invoke('scripts.detect_all_ball_goals', '--fresh-dir', fresh, '--output-dir', target,
               '--ffmpeg', ffmpeg, '--source-dir', args.source_dir,
               '--ball-cache-dir', args.ball_cache_dir, '--track-dir', args.track_dir,
               '--input-dir', args.ball_input_dir, '--audit-media-dir', args.audit_media_dir,
               '--new-inference-clips', '39' if args.mode == 'refresh-ball' else '0')
        # Reuse the already-reviewed source-image judgments, never as algorithm
        # input. Fail if the source clips differ from that original assessment.
        import shutil
        review = json.loads((args.review_dir/'review_summary.json').read_text(encoding='utf-8'))
        for row in review['rows']:
            result = json.loads((target/'results'/f'{row["clip_id"]}.json').read_text(encoding='utf-8'))
            if result['source_sha256'] != row['source_sha256']:
                raise ValueError('Saved AI assessment belongs to different source video bytes')
        shutil.copy2(args.review_dir/'visual_assessment.json', target/'visual_assessment.json')
        invoke('scripts.finalize_all_ball_goal_review', '--output-dir', target,
               '--dense-source-dir', args.review_dir/'media', '--test-log', log)
    run_info = {
        'mode':args.mode,'project_root':str(ROOT),'python':sys.executable,
        'source_dir':str(args.source_dir),'requested_device':args.device,'device':device,'ffmpeg':ffmpeg,
        'ball_cache_dir':str(args.ball_cache_dir),'track_dir':str(args.track_dir),
        'ball_input_dir':str(args.ball_input_dir),
        'inference_scope':'short_goal_area_contact_window' if device else 'cached_event_replay',
        'complete_raw_to_all_reconstruction':False,
        'new_neural_inference_clips':39 if args.mode=='refresh-ball' else 0,
        'saved_ai_source_assessments_reused_for_display_only':True,
        'human_verified':False,'accuracy_validated':False,
    }
    (target/'pycharm_run.json').write_text(json.dumps(run_info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Finished: {target / "report.html"}')
    if args.open:
        webbrowser.open((target/'report.html').as_uri())


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', nargs='?', default='check', choices=['check','report','test-ball','replay-ball','refresh-ball','steps','camera','goalkeeper','notebook','final-results','eda'])
    p.add_argument('--output-dir', help='New directory under outputs; default is a timestamped directory')
    p.add_argument('--open', action='store_true', help='Open the resulting HTML in your default browser')
    add_runtime_arguments(p)
    p.add_argument('--source-dir', type=Path, default=ROOT/'data/raw_clips')
    p.add_argument('--ball-cache-dir', type=Path, default=ROOT/'outputs/ball_detection_all_2026-10-11/detections')
    p.add_argument('--track-dir', type=Path, default=ROOT/'outputs/ball_narrow_channel_2026-10-11/results')
    p.add_argument('--ball-input-dir', type=Path, default=ROOT/'outputs/ball_mask_review_ready_2026-10-11/inputs')
    p.add_argument('--fresh-dir', type=Path, default=FRESH_CACHE)
    p.add_argument('--audit-media-dir', type=Path, default=ROOT/'outputs/ball_goal_event_audit_2026-10-11/media')
    p.add_argument('--review-dir', type=Path, default=BALL_REVIEW)
    p.add_argument('--features-csv', type=Path, default=ROOT/'outputs/final_analysis/features_39.csv')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.chdir(ROOT)
    bootstrap()
    if args.mode == 'check':
        check(args)
    elif args.mode == 'report':
        print(args.review_dir/'report.html')
        if args.open:
            webbrowser.open((args.review_dir/'report.html').as_uri())
    elif args.mode == 'test-ball':
        test_ball()
    elif args.mode in ['replay-ball','refresh-ball']:
        replay_ball(args)
    elif args.mode == 'notebook':
        subprocess.run([sys.executable,'-m','jupyterlab',str(ROOT/'notebooks')],cwd=ROOT,check=True)
    elif args.mode == 'steps':
        out = new_output(args.output_dir,args.mode)
        invoke('scripts.run_step_counting','--output-csv',out/'penalty_step_counts.csv','--diagnostics-json',out/'step_diagnostics.json',
               '--device',args.device,'--clips-dir',args.source_dir)
    elif args.mode == 'camera':
        invoke('scripts.run_camera_timing_analysis','--stage','all','--output-dir',new_output(args.output_dir,args.mode),
               '--device',args.device,'--source-dir',args.source_dir)
    elif args.mode == 'goalkeeper':
        invoke('scripts.run_goalkeeper_height_phase','--output-dir',new_output(args.output_dir,args.mode),
               '--device',args.device,'--ffmpeg',resolve_ffmpeg(args.ffmpeg,ROOT),
               '--source-dir',args.source_dir,'--clips',*canonical_ids(args.ball_input_dir))
    elif args.mode == 'final-results':
        invoke('scripts.build_final_results','--root',ROOT,'--output-dir',new_output(args.output_dir,args.mode))
    elif args.mode == 'eda':
        invoke('src.eda_analysis','--features-csv',args.features_csv,'--output-dir',new_output(args.output_dir,args.mode))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
