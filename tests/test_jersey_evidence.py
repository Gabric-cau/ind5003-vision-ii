import unittest,json
import numpy as np
from src.jersey_evidence import build_tracklets,consolidate,query_evidence,parse_number,torso_crop,select_keyframes


def observation(fid,text,confidence=.95):
    return {'frame':fid,'time_sec':fid/30,'tokens':[{'text':text,'confidence':confidence}]}


def person(box):return {'bbox':box,'area_fraction':.2}


class JerseyEvidenceTests(unittest.TestCase):
    def test_single_number_not_enough(self):self.assertEqual(consolidate([observation(1,'9')])['status'],'unknown')
    def test_repeated_number_supported(self):self.assertEqual(consolidate([observation(0,'9'),observation(9,'9')])['number'],9)
    def test_same_frame_not_two_votes(self):self.assertEqual(consolidate([observation(9,'9')]*3)['status'],'unknown')
    def test_conflicting_numbers_abstain(self):self.assertEqual(consolidate([observation(0,'9'),observation(9,'8')])['status'],'unknown')
    def test_multi_number_same_frame_not_cherry_picked(self):
        a=observation(0,'9');a['tokens']+=observation(0,'19')['tokens']
        self.assertEqual(consolidate([a,observation(9,'9')])['status'],'unknown')
    def test_low_confidence_not_used(self):self.assertEqual(consolidate([observation(0,'9',.5),observation(9,'9',.5)])['number'],None)
    def test_no_letter_to_digit_hallucination(self):
        self.assertIsNone(parse_number('I'));self.assertIsNone(parse_number('9O'));self.assertIsNone(parse_number('09'));self.assertIsNone(parse_number('2018'))
    def test_query_conflict_cannot_force_target(self):
        evidence=consolidate([observation(0,'14'),observation(9,'14')]);result=query_evidence(evidence,10,'Modric')
        self.assertEqual(result['status'],'number_conflict');self.assertEqual(evidence['number'],14)
    def test_name_without_number_is_separate_candidate(self):
        e=consolidate([observation(0,'FALCAO'),observation(9,'FALCAO')])
        self.assertEqual(query_evidence(e,9,'falcao')['status'],'name_match_candidate')
    def test_nearby_frames_do_not_fake_temporal_support(self):self.assertEqual(consolidate([observation(0,'9'),observation(1,'9')])['status'],'unknown')
    def test_tracks_do_not_bridge_gap(self):
        rows=[{'frame':0,'people':[person([0,0,100,200])]},{'frame':30,'people':[person([0,0,100,200])]}]
        self.assertEqual(len(build_tracklets(rows,30)),2)
    def test_unambiguous_track_kept(self):
        rows=[{'frame':f,'people':[person([f,0,f+100,200])]} for f in (0,3,6)]
        self.assertEqual(len(build_tracklets(rows,30)),1)
    def test_identical_competing_boxes_not_joined(self):
        rows=[{'frame':0,'people':[person([0,0,100,200])]*2},{'frame':3,'people':[person([0,0,100,200])]}]
        self.assertEqual(len(build_tracklets(rows,30)),3)
    def test_crop_bounds(self):
        frame=np.zeros((200,100,3),np.uint8);crop,meta=torso_crop(frame,person([-20,-20,200,300]))
        self.assertTrue(crop.size>0);self.assertTrue(all(x>=0 for x in meta['roi']));self.assertLessEqual(meta['roi'][2],100)
    def test_selection_independent_of_ocr(self):
        rows=[{'frame':i*9,'time_sec':i*.3,'crop_info':{'quality':10+i},'tokens':[str(i)]} for i in range(9)]
        selected=select_keyframes(rows,max_frames=4)
        self.assertEqual(len(selected),4);self.assertEqual([r['frame'] for r in selected],sorted(r['frame'] for r in selected))
    def test_crop_metadata_is_json_serializable(self):
        _,meta=torso_crop(np.zeros((300,180,3),np.uint8),person([10,5,170,290]))
        json.dumps(meta,allow_nan=False)


if __name__=='__main__':unittest.main()
