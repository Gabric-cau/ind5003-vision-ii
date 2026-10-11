import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.test_runtime import fake_torch


class InferenceRuntimeTest(unittest.TestCase):
    def test_camera_predict_receives_selected_device(self):
        from scripts.camera_timing_features import infer
        model=SimpleNamespace(_visionii_device='cuda:0',
                              predict=Mock(return_value=[SimpleNamespace(keypoints=None)]))
        self.assertEqual(infer(model,np.zeros((8,8,3),np.uint8)),[])
        self.assertEqual(model.predict.call_args.kwargs['device'],'cuda:0')

    def test_contact_refresh_uses_explicit_source_cache_track_and_device(self):
        ultralytics=ModuleType('ultralytics')
        model=SimpleNamespace(predict=Mock(side_effect=lambda batch,**kwargs:
                                          [SimpleNamespace(boxes=[]) for _ in batch]))
        ultralytics.YOLO=Mock(return_value=model)
        with patch.dict(sys.modules,{'ultralytics':ultralytics}):
            module=importlib.import_module('scripts.cache_ball_keeper_contact')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';cache=root/'cache';tracks=root/'tracks';out=root/'out'
            weights=root/'models/weights/yolo11n_object_detection.pt'
            weights.parent.mkdir(parents=True);weights.write_bytes(b'model')
            for path in (source,cache,tracks):path.mkdir()
            clip=source/'example.mp4';clip.write_bytes(b'actual selected source')
            digest=hashlib.sha256(clip.read_bytes()).hexdigest()
            (cache/'example.json').write_text(json.dumps({
                'video_path':'C:\\old\\example.mp4','source_sha256':digest,'start_frame':0,
                'end_frame_exclusive':4,'kick_frame':1,'fps':10,'width':8,'height':8,
                'goal_proposals':{},'goal_roi':[0,0,4,4],'camera':{'transforms':{}}}))
            (tracks/'example.json').write_text(json.dumps({'source_sha256':digest,'phase':{'main_end_frame':2}}))
            capture=SimpleNamespace(set=Mock(),read=lambda:(True,np.zeros((8,8,3),np.uint8)),release=Mock())
            args=['refresh','--output-dir',str(out),'--source-dir',str(source),
                  '--ball-cache-dir',str(cache),'--track-dir',str(tracks),'--device','cuda:0','--clips','example']
            with patch.object(module,'ROOT',root),patch.object(module,'YOLO',ultralytics.YOLO), \
                 patch.object(module,'torch',fake_torch(True)),patch.object(module.cv2,'VideoCapture',return_value=capture), \
                 patch.object(sys,'argv',args):
                module.main()
            result=json.loads((out/'example.json').read_text())
            self.assertEqual(model.predict.call_args.kwargs['device'],'cuda:0')
            self.assertEqual(result['source_sha256'],digest)
            self.assertEqual(result['runtime']['device'],'cuda:0')
            self.assertEqual(result['runtime']['model_sha256'][str(weights)],hashlib.sha256(b'model').hexdigest())
            self.assertEqual(result['inference_scope'],'short_goal_area_contact_window')
            self.assertFalse(result['complete_raw_to_all_reconstruction'])


if __name__ == '__main__':
    unittest.main()
