import unittest
from src.ball_hypothesis_selection import select_early_flight
from tests.test_ball_continuity import detection


class HypothesisSelectionTests(unittest.TestCase):
    def test_long_late_clothing_tail_cannot_beat_early_ball_evidence(self):
        seed=[detection(i,1000,400) for i in range(8,11)]
        actual=[detection(14,850,350),detection(16,760,320)]
        late=[detection(i,850-(i-30),350) for i in range(30,100)]
        proposals={'early':{'seed':seed,'trajectory':actual},'late':{'seed':seed,'trajectory':late}}
        name,diag=select_early_flight(proposals,10,[550,250],'late')
        self.assertEqual(name,'early')
        self.assertFalse(diag['later_tail_used_for_selection'])
        proposals['late']['trajectory'] += [detection(i,820,340) for i in range(100,200)]
        self.assertEqual(select_early_flight(proposals,10,[550,250],'late'),(name,diag))

    def test_seed_only_path_does_not_defeat_detected_flight(self):
        seed=[detection(i,1000,400) for i in range(8,11)]
        proposals={'short':{'seed':seed,'trajectory':seed},
          'flight':{'seed':seed,'trajectory':[seed[-1],detection(14,850,350)]}}
        name,_=select_early_flight(proposals,10,[550,250],'short')
        self.assertEqual(name,'flight')


if __name__=='__main__':unittest.main()
