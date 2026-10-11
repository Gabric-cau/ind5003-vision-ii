from copy import deepcopy
import unittest

from src.fresh_contact_selection import select_contact_producer


class FreshContactSelectionTest(unittest.TestCase):
    def candidate(self, score=.8, actor=1, bracket=(10, 13), x=50):
        return {'score': score, 'contact_associated_track_id': actor,
                'contact_bracket_frames': list(bracket),
                'stationary_ball_observations': [{'frame': f, 'xy': [x, 50], 'diameter': 8,
                                                'bbox': [x-4, 46, x+4, 54]} for f in (7, 8, 9, 10)],
                'departure_ball_observations': [{'confidence': .3}, {'confidence': .4}]}

    def test_single_supported_producer_is_selected_and_empty_both_abstain(self):
        self.assertEqual(select_contact_producer({'n': None, 'm': self.candidate()})['selected_label'], 'm')
        self.assertIsNone(select_contact_producer({'n': None, 'm': None})['selected_label'])

    def test_compatible_contacts_choose_score_without_using_outcome(self):
        n, m = self.candidate(.9), self.candidate(.8, bracket=(11, 13), x=51)
        n['rule_outcome'] = 'no_goal_likely'
        m['rule_outcome'] = 'goal_likely'
        selected = select_contact_producer({'n': n, 'm': m})
        self.assertEqual(selected['selected_label'], 'n')
        self.assertTrue(selected['compatible'])
        n['rule_outcome'], m['rule_outcome'] = m['rule_outcome'], n['rule_outcome']
        self.assertEqual(select_contact_producer({'n': n, 'm': m}), selected)

    def test_conflicting_actor_time_spot_or_insufficient_shared_observation_abstains(self):
        n = self.candidate()
        m_values = [self.candidate(.99, actor=2), self.candidate(.99, bracket=(14, 17)),
                    self.candidate(.99, x=100)]
        missing = self.candidate(.99)
        missing['stationary_ball_observations'] = missing['stationary_ball_observations'][:2]
        m_values.append(missing)
        for m in m_values:
            with self.subTest(m=m):
                self.assertIsNone(select_contact_producer({'n': n, 'm': m})['selected_label'])

    def test_ties_use_narrower_bracket_then_departure_confidence_then_fixed_m(self):
        n, m = self.candidate(), self.candidate(bracket=(11, 13))
        self.assertEqual(select_contact_producer({'n': n, 'm': m})['selected_label'], 'm')
        n = deepcopy(m)
        n['departure_ball_observations'] = [{'confidence': .5}, {'confidence': .6}]
        self.assertEqual(select_contact_producer({'n': n, 'm': m})['selected_label'], 'n')
        self.assertEqual(select_contact_producer({'n': m, 'm': m})['selected_label'], 'm')


if __name__ == '__main__':
    unittest.main()
