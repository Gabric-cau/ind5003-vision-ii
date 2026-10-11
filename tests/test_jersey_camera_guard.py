import unittest
from src.jersey_camera_guard import text_camera_decision


class JerseyCameraGuardTests(unittest.TestCase):
    def person(self):return {'role':'kicker_candidate','height_fraction':.8,'area_fraction':.2}
    def test_number_conflict_does_not_delete_existing_candidate(self):
        r=text_camera_decision('kicker_closeup','kicker_closeup',{'status':'number_conflict'},[self.person()],{})
        self.assertFalse(r['changed']);self.assertEqual(r['vote'],'kicker_closeup');self.assertIsNotNone(r['warning'])
    def test_repeated_text_can_relax_grass_only(self):
        r=text_camera_decision('person_closeup_identity_unresolved','kicker_closeup',{'status':'number_match_candidate'},[self.person()],{'green_outside_people_fraction':0})
        self.assertTrue(r['changed']);self.assertFalse(r['identity_verified'])
    def test_text_cannot_override_same_kit_crowd(self):
        r=text_camera_decision('person_closeup_identity_unresolved','kicker_closeup',{'status':'number_and_name_candidate'},[self.person(),self.person()],{})
        self.assertFalse(r['changed'])
    def test_no_original_kicker_cue_no_promotion(self):
        r=text_camera_decision('unknown','unknown',{'status':'number_match_candidate'},[self.person()],{})
        self.assertFalse(r['changed'])


if __name__=='__main__':unittest.main()
