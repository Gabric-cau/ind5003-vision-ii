from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from scripts import run_fresh_ball_experiment as experiment
from src.fresh_event_analysis import partition_predictions


def result(classes, boxes=None):
    if boxes is None:
        boxes = [[10, 10, 16, 16] for _ in classes]
    return SimpleNamespace(boxes=SimpleNamespace(
        xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
        conf=np.full(len(classes), .9), cls=np.asarray(classes)))


def test_default_partition_retains_coco_classes_and_raw_schema():
    frame = np.full((200, 240, 3), 255, np.uint8)
    prediction = result([32, 0], [[10, 10, 16, 16], [40, 40, 90, 140]])
    balls, people, raw = partition_predictions([7], [prediction], [frame])
    assert len(balls['7']) == len(people['7']) == 1
    assert [row['class_id'] for row in raw['7']] == [32, 0]
    assert all(set(row) == {'bbox', 'confidence', 'class_id', 'mode', 'crop_translation_xy'}
               for row in raw['7'])


def test_custom_ball_zero_never_enters_people_and_records_native_class():
    frame = np.full((200, 240, 3), 255, np.uint8)
    balls, people, raw = partition_predictions([7], [result([0])], [frame],
                                              ball_class_id=0, person_class_id=None)
    assert len(balls['7']) == 1
    assert people['7'] == []
    assert raw['7'][0]['class_id'] == 32
    assert raw['7'][0]['native_class_id'] == 0
    with pytest.raises(ValueError, match='distinct'):
        partition_predictions([7], [result([0])], [frame], ball_class_id=0)


def test_specialist_requires_single_ball_class_not_coco_or_pose():
    experiment.validate_ball_specialist(SimpleNamespace(task='detect', names={0: 'ball'}))
    experiment.validate_ball_specialist(SimpleNamespace(task='detect', names=['sports ball']))
    for model in [SimpleNamespace(task='detect', names={0: 'person', 32: 'sports ball'}),
                  SimpleNamespace(task='pose', names={0: 'ball'}),
                  SimpleNamespace(task='detect', names={1: 'ball'})]:
        with pytest.raises(ValueError, match='single-class'):
            experiment.validate_ball_specialist(model)


@pytest.mark.parametrize('mode', ['full', 'tiles'])
def test_optional_specialist_augments_balls_preserves_coco_people_and_uses_source_frames(mode):
    frame = np.full((200, 240, 3), 255, np.uint8)
    pose = {'clip_id': 'clip', 'fps': 25., 'selected_track_id': 1, 'tracks': [], 'kick_frame': 7}
    record = {'clip_sha256': 'source', 'source_start_frame': 100, 'fps': 25.}
    settings = {'imgsz': 1280, 'batch_size': 4, 'confidence': .01, 'device': 'cpu'}
    coco = Mock()
    coco.predict.return_value = [result([32, 0], [[10, 10, 16, 16], [40, 40, 90, 140]])]
    specialist = SimpleNamespace(task='detect', names={0: 'ball'})
    with patch.object(experiment, 'contact_window', return_value=((7, 8), 'qualified')), \
         patch.object(experiment, '_read_window', return_value=([7], [frame], 25., 240, 200)), \
         patch.object(experiment, 'detect_goal_frame', return_value={'selected': None}), \
         patch.object(experiment, 'predict_images', return_value=[[{'bbox': [150, 100, 156, 106],
                                                                   'confidence': .8}]]) as predict:
        baseline, _, _ = experiment.detect_event(Path('unused.mp4'), pose, record, coco, settings)
        predict.assert_not_called()
        assert 'ball_specialist' not in baseline
        config = {'mode': mode, 'imgsz': 960}
        augmented, _, _ = experiment.detect_event(Path('unused.mp4'), pose, record, coco,
            {**settings, 'ball_specialist': config}, ball_specialist=specialist)
        assert predict.call_args.args[0] is specialist
        assert predict.call_args.args[1][0] is frame
        assert predict.call_args.args[2:] == (mode, 960, 'cpu', 4)
    assert augmented['people'] == baseline['people']
    assert len(augmented['detections']['7']) == len(baseline['detections']['7']) + 1
    rows = augmented['raw_neural_detections']['7']
    assert rows[-1]['class_id'] == 32 and rows[-1]['native_class_id'] == 0
    assert rows[-1]['mode'] == 'ball_specialist_'+mode
    assert all(call.kwargs['classes'] == [0, 32] for call in coco.predict.call_args_list)
    assert augmented['contact'] == baseline['contact']


def test_specialist_prediction_count_mismatch_fails_instead_of_dropping_source_frames():
    frames = [np.zeros((20, 20, 3), np.uint8)]
    settings = {'device': 'cpu', 'batch_size': 4, 'ball_specialist': {'mode': 'full', 'imgsz': 1280}}
    with patch.object(experiment, 'predict_images', return_value=[]), \
         pytest.raises(ValueError, match='batch output length'):
        experiment.partition_specialist([1], frames, object(), settings)
