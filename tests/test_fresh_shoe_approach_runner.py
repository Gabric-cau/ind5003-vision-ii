from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import run_fresh_shoe_approach_experiment as runner


def fixture_inputs(tmp_path, monkeypatch):
    args = SimpleNamespace(clips_dir=tmp_path / 'clips', pose_dir=tmp_path / 'pose',
                           phase_dir=tmp_path / 'phase', ball_dir=tmp_path / 'ball')
    for directory in (args.clips_dir, args.pose_dir, args.phase_dir, args.ball_dir):
        directory.mkdir()
    video = args.clips_dir / 'event.mp4'
    video.write_bytes(b'hashed original course pixels')
    source = dict(clip_id='event', clip_filename='event.mp4', clip_sha256=runner.sha(video),
                  fps=10., frame_count=20)
    observations = [dict(frame=frame, bbox=[10., 10., 30., 70.]) for frame in (2, 4, 6, 8, 9, 10, 11, 12)]
    pose = dict(fps=10., frame_width=100, frame_height=100, source_frame_count=20,
                shot_start_frame=0, shot_end_frame=20, selected_track_id=3,
                reference_labels_used_for_prediction=False, tracks=[dict(track_id=3, observations=observations)])
    pose_path = args.pose_dir / 'event.json'
    runner.write(pose_path, pose)
    pose_hash = runner.sha(pose_path)
    identity = dict(source_manifest_sha256='manifest', source_clip_sha256=source['clip_sha256'],
                    pose_sha256=pose_hash, shot_interval=[0, 20])
    contact = dict(status='tentative', contact_bracket_frames=[9, 10], contact_associated_track_id=3)
    camera = dict(inference_identity=identity, shot_interval=[0, 20],
                  transforms={str(frame): np.eye(3).tolist() for frame in range(20)})
    camera_path = args.phase_dir / 'camera_motion' / 'event.json'
    runner.write(camera_path, camera)
    phase = dict(clip_id='event', old_reference_labels_used=False, user_step_labels_used=False,
                 inference_identity=identity, pose_input={'sha256': pose_hash},
                 contact=contact, analysis_actor_track_id=3, camera_motion_sha256=runner.sha(camera_path),
                 phase_result={'approach_start_frame': 2})
    runner.write(args.phase_dir / 'event.json', phase)
    monkeypatch.setattr(runner, '_pose_input', lambda path, row: (pose, {'sha256': pose_hash, 'status': 'supported'}))
    monkeypatch.setattr(runner, '_bound_contact', lambda *unused: (contact, {'status': 'supported'}))
    return args, source, pose, phase, camera_path


def test_plan_keeps_missing_same_frame_actor_without_interpolation(tmp_path, monkeypatch):
    args, source, _, _, _ = fixture_inputs(tmp_path, monkeypatch)
    bound, _ = runner.bind_event(args, source, 'manifest')
    assert bound['interval_frames'] == [2, 9]
    assert bound['inference_interval_frames'] == [2, 12]
    assert bound['plan'][1]['frame'] == 3
    assert bound['plan'][1]['anchor_bbox'] is None
    assert bound['plan'][1]['anchor_source'] == 'missing_no_interpolation'
    assert {tuple(row['crop_xyxy']) for row in bound['plan']} == {tuple(bound['fixed_roi'])}


@pytest.mark.parametrize('field,value', [('analysis_actor_track_id', 4), ('old_reference_labels_used', True)])
def test_changed_actor_or_label_using_phase_is_rejected(tmp_path, monkeypatch, field, value):
    args, source, _, phase, _ = fixture_inputs(tmp_path, monkeypatch)
    phase[field] = value
    runner.write(args.phase_dir / 'event.json', phase)
    with pytest.raises(ValueError):
        runner.bind_event(args, source, 'manifest')


def test_changed_motion_bytes_are_rejected(tmp_path, monkeypatch):
    args, source, _, _, camera_path = fixture_inputs(tmp_path, monkeypatch)
    runner.write(camera_path, {'transforms': {}})
    with pytest.raises(ValueError, match='camera-motion artifact'):
        runner.bind_event(args, source, 'manifest')


def test_unobserved_onset_keeps_event_unavailable(tmp_path, monkeypatch):
    args, source, _, phase, _ = fixture_inputs(tmp_path, monkeypatch)
    phase['phase_result']['approach_start_frame'] = None
    runner.write(args.phase_dir / 'event.json', phase)
    bound, provenance = runner.bind_event(args, source, 'manifest')
    assert bound is None
    assert provenance['unavailable_reason'] == 'visible_approach_onset_unavailable'


def test_source_hash_mismatch_is_rejected_before_pose_lookup(tmp_path, monkeypatch):
    args, source, _, _, _ = fixture_inputs(tmp_path, monkeypatch)
    (args.clips_dir / source['clip_filename']).write_bytes(b'new or external footage')
    with pytest.raises(ValueError, match='Course clip hash mismatch'):
        runner.bind_event(args, source, 'manifest')


class FakeCapture:
    def __init__(self, bad_clock=False):
        self.next_frame = 0
        self.released = False
        self.bad_clock = bad_clock

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 10., cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_FRAME_WIDTH: 4, cv2.CAP_PROP_FRAME_HEIGHT: 3,
                cv2.CAP_PROP_POS_FRAMES: self.next_frame,
                cv2.CAP_PROP_POS_MSEC: max(0, self.next_frame - 1) * 100. + (20. if self.bad_clock else 0.)}[key]

    def read(self):
        image = np.full((3, 4, 3), self.next_frame, np.uint8)
        self.next_frame += 1
        return True, image

    def release(self):
        self.released = True


def decoding_bound():
    return dict(video=Path('unused.mp4'), source_mapping={'frame_count': 5},
                fps=10., width=4, height=3, inference_interval_frames=[2, 4])


def test_sequential_decode_reads_discarded_prefix_without_seeking(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(runner.cv2, 'VideoCapture', lambda path: capture)
    rows = list(runner.decode_interval(decoding_bound()))
    assert [row[0] for row in rows] == [2, 3, 4]
    assert [int(row[1][0, 0, 0]) for row in rows] == [2, 3, 4]
    assert [row[2] for row in rows] == [200., 300., 400.]
    assert capture.next_frame == 5 and capture.released


def test_bad_decoder_timestamp_fails_even_before_visible_interval(monkeypatch):
    capture = FakeCapture(bad_clock=True)
    monkeypatch.setattr(runner.cv2, 'VideoCapture', lambda path: capture)
    with pytest.raises(ValueError, match='frame clock'):
        list(runner.decode_interval(decoding_bound()))
    assert capture.released


def test_reflected_views_have_separate_trackers_and_repeated_pixels_are_not_new_support(monkeypatch):
    trackers = []

    class BaseOptions:
        Delegate = SimpleNamespace(CPU='CPU')

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Tracker:
        def __init__(self):
            self.timestamps = []
            trackers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            return False

        def detect_for_video(self, image, timestamp):
            self.timestamps.append(timestamp)
            return SimpleNamespace(pose_landmarks=[])

    vision = SimpleNamespace(PoseLandmarkerOptions=lambda **kwargs: kwargs,
                             RunningMode=SimpleNamespace(VIDEO='VIDEO'),
                             PoseLandmarker=SimpleNamespace(create_from_options=lambda options: Tracker()))
    python = SimpleNamespace(BaseOptions=BaseOptions, vision=vision)
    monkeypatch.setitem(sys.modules, 'mediapipe', SimpleNamespace(Image=lambda **kwargs: kwargs,
                                                               ImageFormat=SimpleNamespace(SRGB='SRGB')))
    monkeypatch.setitem(sys.modules, 'mediapipe.tasks', SimpleNamespace(python=python))
    monkeypatch.setitem(sys.modules, 'mediapipe.tasks.python', python)
    image = np.zeros((10, 10, 3), np.uint8)
    bound = dict(fps=10., fixed_roi=[0, 0, 10, 10], plan=[dict(frame=frame, anchor_bbox=[0, 0, 10, 10],
                 rival_source_boxes=[]) for frame in (2, 3, 4)])
    monkeypatch.setattr(runner, 'decode_interval', lambda original: iter([(frame, image, frame * 100.) for frame in (2, 3, 4)]))
    records, hashes, quality = runner.infer_event(bound, Path('official-model.task'))
    assert len(trackers) == 2
    assert trackers[0].timestamps == trackers[1].timestamps == [200, 300, 400]
    assert len(set(hashes.values())) == 1
    assert [row['unique_decoded_support'] for row in records] == [True, True, False, False, False, False]
    assert [row['duplicate_of_frame'] for row in records] == [None, None, 2, 2, 2, 2]
    assert quality['actual_inference_calls'] == 6
    assert quality['distinct_decoded_pixel_frames'] == 1
    assert quality['repeated_decoded_pixel_frames'] == 2


def test_added_source_or_changed_original_interval_cannot_enter_cohort(tmp_path):
    sources = {filename: {'sha256': filename, 'frame_count': 100} for filename in runner.COURSE_SOURCES}
    rows = [dict(clip_id=str(index), source_filename='col_eng_pen.mp4',
                 source_video_sha256='col_eng_pen.mp4', source_start_frame=0,
                 source_end_frame_exclusive=10, frame_count=10, clip_filename=f'{index}.mp4') for index in range(39)]
    path = tmp_path / 'manifest.json'
    runner.write(path, {'sources': sources, 'clips': rows})
    assert len(runner.course_manifest(path)[1]) == 39
    sources['extra.mp4'] = {'sha256': 'extra', 'frame_count': 100}
    runner.write(path, {'sources': sources, 'clips': rows})
    with pytest.raises(ValueError, match='original four-video'):
        runner.course_manifest(path)
    del sources['extra.mp4']
    rows[0]['frame_count'] = 9
    runner.write(path, {'sources': sources, 'clips': rows})
    with pytest.raises(ValueError, match='original source frame interval'):
        runner.course_manifest(path)
