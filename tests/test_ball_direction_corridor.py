import unittest
from dataclasses import replace
from src.ball_direction_corridor import (
    DirectionCorridorConfig, assess_candidate, corridor_geometry, corridor_forecasts)
from src.ball_continuity_v4 import track_ball_continuity
from tests.test_ball_continuity import detection, flight


class DirectionCorridorTests(unittest.TestCase):
    def run_track(self, ds, **kw):
        corridor_cfg = kw.pop('corridor_config', DirectionCorridorConfig())
        return track_ball_continuity(ds, 10, [[1000, 400]], [550, 250],
                                     corridor_config=corridor_cfg, **kw)

    def test_lane_is_finite_forward_and_gap_growth_is_bounded(self):
        one = corridor_geometry([0, 0], [30, 0], 1, 7, 45)
        six = corridor_geometry([0, 0], [30, 0], 6, 7, 45)
        self.assertLess(six['half_width_px'] - one['half_width_px'], 6)
        self.assertTrue(assess_candidate([40, 0], one)['inside_forward_corridor'])
        self.assertFalse(assess_candidate([-10, 0], one)['inside_forward_corridor'])
        self.assertFalse(assess_candidate([80, 0], one)['inside_forward_corridor'])

    def test_high_confidence_side_branch_cannot_take_over_after_a_gap(self):
        ds = flight()
        # A nearby parallel false stream lies within the old angular cone
        # after a short gap, but remains outside the finite-width corridor.
        for f in range(16, 26):
            ball = ds[str(f)][0]
            ds[str(f)].append(detection(f, ball['x'] - 7.28, ball['y'] + 21.83,
                                       confidence=.99, width=150, height=150))
        ds['16'] = ds['16'][1:]
        path, diag = self.run_track(ds)
        self.assertEqual(path[-1]['frame'], 25)
        self.assertTrue(all(q['width'] == 14 for q in path))
        self.assertGreater(diag['rejected_outside_forward_corridor'], 0)

    def test_continuing_real_ball_and_missing_frames_are_preserved(self):
        ds = flight()
        for f in (16, 17, 18): ds[str(f)] = []
        path, diag = self.run_track(ds)
        self.assertEqual(path[-1]['frame'], 25)
        self.assertEqual(diag['max_gap_frames'], 4)
        self.assertFalse(any(q['frame'] in (16, 17, 18) for q in path))

    def test_large_turn_needs_prior_contact_proximity_and_multiple_points(self):
        ds = flight()
        for f in range(21, 31):
            ds[str(f)] = [detection(f, 700 + 15 * (f - 20), 300 - 15 * (f - 20))]
        path, diag = self.run_track(ds)
        self.assertEqual(path[-1]['frame'], 20)
        regions = {str(f): [{'kind': 'post', 'segment': [[700, 250], [700, 350]]}]
                   for f in range(20, 24)}
        path, diag = self.run_track(ds, collision_regions=regions)
        self.assertEqual(path[-1]['frame'], 30)
        self.assertTrue(diag['turn_frames'])
        self.assertFalse(diag['whole_ball_crossing_established'])
        short = {f: qs for f, qs in ds.items() if int(f) <= 21}
        path, _ = self.run_track(short, collision_regions=regions)
        self.assertEqual(path[-1]['frame'], 20)

    def test_destination_cannot_enlarge_its_own_lane(self):
        g = corridor_geometry([0, 0], [30, 0], 1, 7, 45)
        result = assess_candidate([30, 30], g)
        self.assertFalse(result['inside_forward_corridor'])
        self.assertFalse(result['destination_size_used_for_width'])

    def test_motion_ghost_currently_on_grass_loses_to_real_ball(self):
        ds = flight()
        for f in range(16, 26):
            ball = ds[str(f)][0]
            ds[str(f)].append(detection(f, ball['x'] - 2, ball['y'] + 2,
                sources=['motion_blob'], confidence=0, current_pixel_mask_support=False))
        path, _ = self.run_track(ds)
        self.assertTrue(all('motion_blob' not in q['sources'] for q in path))
        self.assertEqual(len(ds['20']), 2)

    def test_motion_ghost_cannot_establish_initial_direction(self):
        ds = flight()
        for f in range(11, 16):
            ds[str(f)][0].update(sources=['motion_blob'], confidence=0, current_pixel_mask_support=False)
        path, diag = self.run_track(ds, corridor_config=DirectionCorridorConfig(require_current_pixels_for_motion_bootstrap=True))
        self.assertEqual(path[1]['frame'], 16)
        self.assertEqual(path[-1]['frame'], 25)
        self.assertGreater(diag['departure_hypotheses_rejected_without_current_pixels'], 0)
        self.assertEqual(len(ds['12']), 1)

    def test_shadowed_neural_departure_is_not_masked_out(self):
        ds = flight()
        for qs in ds.values(): qs[0]['current_pixel_mask_support'] = False
        path, _ = self.run_track(ds)
        self.assertEqual(path[-1]['frame'], 25)

    def test_smooth_image_curve_can_update_the_lane(self):
        ds = flight()
        x, y = ds['15'][0]['stabilized_xy']
        import math
        for f in range(16, 26):
            heading = math.atan2(-10, -30) - math.radians(2 * (f - 15))
            x += 32 * math.cos(heading); y += 32 * math.sin(heading)
            ds[str(f)] = [detection(f, x, y)]
        path, _ = self.run_track(ds)
        self.assertEqual(path[-1]['frame'], 25)

    def test_overlay_uses_prior_observations_and_expires(self):
        ds = flight(); path, diag = self.run_track(ds)
        a = corridor_forecasts(path, diag['motion_states'], 10, 33)
        corrupted = [{**q, 'width': 900, 'height': 900} if q['frame'] >= 20 else q for q in path]
        b = corridor_forecasts(corrupted, diag['motion_states'], 10, 33)
        self.assertEqual([q for q in a if q['frame'] <= 20], [q for q in b if q['frame'] <= 20])
        self.assertTrue(all(q['last_observation_frame'] < q['frame'] for q in a))
        self.assertEqual(a[-1]['frame'], 31)


if __name__ == '__main__': unittest.main()
