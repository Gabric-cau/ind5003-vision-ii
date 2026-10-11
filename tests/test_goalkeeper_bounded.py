import unittest
from unittest.mock import patch
from src.goalkeeper_pipeline import select_goalkeeper_track, _has_distinct_competitor


class GoalkeeperBoundedTests(unittest.TestCase):
    def test_self_is_not_competitor(self):
        selected = {"track_id":"stitched:1+4", "source_track_ids":[1,4], "score":10}
        candidates = [{"track_id":i,"score":11,"qualifies":True} for i in (1,4)]
        self.assertFalse(_has_distinct_competitor(selected, candidates))
        self.assertTrue(_has_distinct_competitor(selected, candidates + [
            {"track_id":7,"score":9.5,"qualifies":True}]))

    def test_disqualified_track_not_a_competitor(self):
        self.assertFalse(_has_distinct_competitor({"track_id":1,"score":10}, [
            {"track_id":2,"score":12,"qualifies":False}]))

    def test_gap_cannot_bridge_to_late_person(self):
        def obs(f):
            return {"frame":f,"bbox":[90,80,110,120],"box_confidence":.9}
        tracks = [{"track_id":1,"observations":[obs(f) for f in range(0,34)]},
                  {"track_id":2,"observations":[obs(f) for f in range(48,61)]}]
        result={"fps":30,"frame_width":640,"frame_height":360,"tracks":tracks,
                "selected_track_id":None,"video_path":"no-file"}
        with patch("cv2.VideoCapture") as capture:
            capture.return_value.read.return_value=(False,None)
            old,_=select_goalkeeper_track(result,30,seed_track_id=1)
            new,_=select_goalkeeper_track(result,30,seed_track_id=1,tracking_policy="bounded")
        self.assertIn(2,old["source_track_ids"])
        self.assertNotIn(2,new["source_track_ids"])
        self.assertLess(new["post_count"],old["post_count"])

    def test_stale_seed_not_teleported_to_kick(self):
        def obs(f,x):
            return {"frame":f,"bbox":[x-10,80,x+10,120],"box_confidence":.9}
        result={"fps":30,"frame_width":640,"frame_height":360,"selected_track_id":None,
                "video_path":"no-file","tracks":[
                    {"track_id":1,"observations":[obs(f,100) for f in range(0,13)]},
                    {"track_id":2,"observations":[obs(f,100) for f in range(25,61)]}]}
        with patch("cv2.VideoCapture") as capture:
            capture.return_value.read.return_value=(False,None)
            new,_=select_goalkeeper_track(result,30,seed_track_id=1,tracking_policy="bounded")
        self.assertIsNone(new)


if __name__ == "__main__":
    unittest.main()
