import unittest
from scripts.render_goalkeeper_qa import nearest_sample


class GoalkeeperQAGapTests(unittest.TestCase):
    def test_no_stale_position_painted_over_later_video(self):
        c={"pre_trajectory":[(0.,1,2,100)],"post_trajectory":[(.1,3,4,100)]}
        self.assertIsNone(nearest_sample(c,30,30))
        self.assertEqual(nearest_sample(c,3,30),(.1,3,4,100))
