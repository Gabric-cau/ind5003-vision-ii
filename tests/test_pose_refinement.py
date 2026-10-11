"""Synthetic coordinate/identity tests; not claims about model accuracy."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.pose_refinement import (anchor_at_frame,crop_bounds,pose_sequence,
                                 refine_contact_track,restore_detection,select_anchored_detection)


def detection(box=(10,10,30,60)):
    return {"bbox":list(box),"box_confidence":.8,"keypoints":np.tile([20.,30.],(17,1)).tolist(),
            "keypoint_confidences":[.9]*17}


class RefinementTests(unittest.TestCase):
    def test_crop_clamped(self):
        self.assertEqual(crop_bounds([0,0,30,50],100,80),(0,0,48,68))
        self.assertEqual(crop_bounds([75,55,99,79],100,80),(66,46,100,80))

    def test_invalid_box(self):
        for box in ([20,20,10,40],[0,0,float("nan"),30],[200,200,210,230]):
            with self.assertRaises(ValueError):
                crop_bounds(box,100,80)

    def test_restore_original_coordinates_and_frame(self):
        record=detection()
        result=restore_detection(record,[100,200],77)
        self.assertEqual(result["bbox"],[110,210,130,260])
        self.assertEqual(result["keypoints"][15],[120,230])
        self.assertEqual(result["frame"],77)
        self.assertEqual(record["bbox"],[10,10,30,60])

    def test_invalid_keypoints_not_moved_to_crop_origin(self):
        record=detection()
        record["keypoints"][15]=[0,0]
        record["keypoints"][16]=[float("nan"),10]
        result=restore_detection(record,[100,200],8)
        self.assertEqual(result["keypoints"][15],[0,0])
        self.assertEqual(result["keypoint_confidences"][16],0)
        self.assertFalse(pose_sequence([result])["frames"])

    def test_rival_confidence_does_not_win(self):
        wanted=detection([100,100,130,180])
        rival=detection([150,100,180,180])
        rival["box_confidence"]=.99
        selected,reason=select_anchored_detection([rival,wanted],[100,100,130,180],[[150,100,180,180]])
        self.assertIs(selected,wanted)
        self.assertEqual(reason,"accepted")

    def test_ambiguous_overlap_abstains(self):
        selected,reason=select_anchored_detection([detection(),detection()],[10,10,30,60])
        self.assertIsNone(selected)
        self.assertEqual(reason,"ambiguous_crop_identity")

    def test_short_interpolation_no_extrapolation(self):
        source={10:{"bbox":[0,0,10,20]},14:{"bbox":[4,0,14,20]}}
        self.assertEqual(anchor_at_frame(source,12),([2.,0.,12.,20.],"short_gap_interpolation"))
        self.assertEqual(anchor_at_frame(source,9),(None,"no_extrapolation"))
        self.assertEqual(anchor_at_frame({10:source[10],20:source[14]},15),(None,"source_track_gap"))

    def test_abstention_does_not_open_video_or_call_model(self):
        with patch("src.pose_refinement.cv2.VideoCapture") as capture:
            result=refine_contact_track("unused",{}, {"status":"abstain"},None)
            self.assertEqual(result["status"],"abstain")
            capture.assert_not_called()

    def test_clip_mismatch_abstains(self):
        result=refine_contact_track("unused",{"clip_id":"a"},{"clip_id":"b","status":"supported","contact_frame":4,"contact_associated_track_id":1},None)
        self.assertEqual(result["reason"],"clip_id_mismatch")

    def test_full_api_uses_associated_not_previous_selected(self):
        class Capture:
            def isOpened(self): return True
            def set(self,*args): pass
            def read(self): return True,np.zeros((100,100,3),np.uint8)
            def release(self): pass
        class Model:
            def predict(self,crops,**kwargs):
                for _ in crops:
                    points=np.tile([30.,35.],(1,17,1))
                    yield SimpleNamespace(keypoints=SimpleNamespace(xy=points,conf=np.ones((1,17))*.9),
                                          boxes=SimpleNamespace(xyxy=np.array([[20.,15.,40.,65.]]),conf=np.array([.8])))
        original={"clip_id":"a","selected_track_id":9,"shot_start_frame":10,"shot_end_frame":15,"fps":30,
                  "tracks":[{"track_id":2,"observations":[dict(detection([30,20,50,70]),frame=f) for f in range(10,15)]}]}
        previous=copy.deepcopy(original)
        contact={"clip_id":"a","status":"tentative","contact_associated_track_id":2,"contact_frame":13}
        with patch("src.pose_refinement.cv2.VideoCapture",return_value=Capture()):
            result=refine_contact_track("unused",original,contact,Model())
        self.assertEqual(original,previous)
        self.assertEqual(result["selected_track_id"],2)
        self.assertEqual(result["status"],"refined")
        self.assertEqual(result["selected_pose_sequence"]["frames"],list(range(10,15)))
        self.assertTrue(result["diagnostics"]["input_contact_remains_tentative"])
        self.assertFalse(result["human_verified"])


if __name__=="__main__":
    unittest.main()
