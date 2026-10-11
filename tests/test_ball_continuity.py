import unittest

from src.ball_continuity import (ContinuityConfig, collision_proximity, motion_forecasts,
                                 search_predictions, track_ball_continuity)


def detection(f, x, y, **extra):
    return {"frame": f, "x": x, "y": y, "stabilized_xy": [x, y],
            "confidence": .8, "width": 14, "height": 14, "sources": ["full"],
            "motion_mean_abs_gray": 50., "detection_id": f"{f}:{x}:{y}", **extra}


def flight():
    ds = {str(i): [detection(i, 1000, 400)] for i in range(11)}
    for f in range(11, 26):
        ds[str(f)] = [detection(f, 1000 - 30 * (f - 10), 400 - 10 * (f - 10))]
    return ds


def run(ds, **kwargs):
    return track_ball_continuity(ds, 10, [[1000, 400]], [550, 250], **kwargs)


class ContinuityTests(unittest.TestCase):
    def test_far_high_confidence_object_cannot_replace_ball(self):
        ds = flight()
        for f in range(15, 26):
            ds[str(f)].append(detection(f, 80, 80, confidence=.99))
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 25)
        self.assertGreater(diag["rejected_unreachable_associations"], 0)
        self.assertTrue(all(p["x"] > 500 for p in path))

    def test_reacquire_after_three_missing_source_frames(self):
        ds = flight()
        for f in (16, 17, 18):
            ds[str(f)] = []
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 25)
        self.assertEqual(diag["max_gap_frames"], 4)
        self.assertFalse(any(p["frame"] in (16, 17, 18) for p in path))
        predictions = search_predictions(path, 10, 26)
        self.assertTrue(all(not p["is_observation"] for p in predictions))
        self.assertEqual([p["frame"] for p in predictions], [16, 17, 18])

    def test_predictions_expire_instead_of_bridging_long_loss(self):
        ds = flight()
        for f in range(16, 25):
            ds[str(f)] = []
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 15)
        predictions = search_predictions(path, 10, 26)
        self.assertEqual(predictions[-1]["frame"], 21)

    def test_rebound_is_continuous_and_persists_beyond_eight_frames(self):
        ds = flight()
        for f in range(21, 36):
            ds[str(f)] = [detection(f, 700 + 15 * (f - 20), 300 - 15 * (f - 20))]
        regions = {str(f): [{"kind": "post", "segment": [[700, 250], [700, 350]]}]
                   for f in range(20, 24)}
        path, diag = run(ds, collision_regions=regions)
        self.assertEqual(path[-1]["frame"], 35)
        self.assertTrue(diag["turn_frames"])
        self.assertEqual(diag["outcome"], "unknown")

    def test_person_overlap_does_not_hide_ball_at_impact(self):
        ds = flight()
        for points in ds.values():
            points[0].update(person_overlap=True, kit_neighborhood_fraction=1.)
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 25)
        self.assertFalse(diag["semantic_masks_used"])

    def test_camera_coordinate_break_stops_search(self):
        ds = flight()
        path, diag = run(ds, frame_metadata={str(f): {"coordinate_supported": f < 19}
                                             for f in range(26)})
        self.assertEqual(path[-1]["frame"], 18)
        self.assertEqual(diag["coordinate_break_frame"], 19)

    def test_missing_seed_and_bad_configuration(self):
        path, diag = run({str(i): [] for i in range(10)})
        self.assertFalse(path)
        self.assertEqual(diag["tracking_status"], "abstain_no_seed")
        with self.assertRaises(ValueError):
            run(flight(), config=ContinuityConfig(max_gap_frames=0))

    def test_late_branch_cannot_replace_fixed_incoming_observations(self):
        ds = flight()
        prefix = [ds[str(f)][0] for f in range(11, 16)]
        for f in range(11, 26):
            ds[str(f)].append(detection(f, 1000 - 10 * (f - 10), 400))
        path, diag = run(ds, initial_path=prefix)
        self.assertEqual([p["detection_id"] for p in path[:len(prefix)]],
                         [p["detection_id"] for p in prefix])
        self.assertEqual(diag["fixed_prefix_observations"], len(prefix))

    def test_original_frame_intervals_include_encoded_repetitions(self):
        ds = flight()
        for f, points in ds.items():
            points[0]["motion_time"] = int(f) / 3
        path, _ = run(ds)
        self.assertEqual(path[-1]["frame"], 25)
        self.assertEqual(ds["25"][0]["motion_time"], 25 / 3)
        self.assertEqual(path[-1]["motion_time"], 25)

    def test_turn_without_collision_cue_is_rejected(self):
        ds = flight()
        for f in range(21, 36):
            ds[str(f)] = [detection(f, 700 + 15 * (f - 20), 300 - 15 * (f - 20))]
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 20)
        self.assertFalse(diag["turn_frames"])
        self.assertGreater(diag["rejected_turns_without_collision_cue"], 0)

    def test_distant_post_does_not_unlock_turn(self):
        ds = flight()
        for f in range(21, 29):
            ds[str(f)] = [detection(f, 700 + 15 * (f - 20), 300 - 15 * (f - 20))]
        regions = {str(f): [{"kind": "post", "segment": [[50, 250], [50, 350]]}]
                   for f in range(20, 29)}
        path, diag = run(ds, collision_regions=regions)
        self.assertEqual(path[-1]["frame"], 20)

    def test_goal_projection_is_not_contact_evidence(self):
        a, b = detection(1, 700, 300), detection(2, 715, 285)
        regions = {"2": [{"kind": "goal_projection", "strength": "weak",
                           "polygon": [[600, 200], [800, 200], [800, 400], [600, 400]]}]}
        self.assertIsNone(collision_proximity(a, b, regions, 22))

    def test_keeper_polygon_proximity_does_not_assert_contact(self):
        a, b = detection(1, 700, 300), detection(2, 715, 285)
        regions = {"2": [{"kind": "keeper", "polygon": [[690, 290], [710, 290],
                                                               [710, 310], [690, 310]]}]}
        cue = collision_proximity(a, b, regions, 22)
        self.assertEqual(cue["kind"], "keeper")
        self.assertFalse(cue["physical_contact_established"])

    def test_candidate_on_keeper_cannot_create_its_own_collision_permission(self):
        a, b = detection(1, 700, 300), detection(2, 715, 350)
        regions = {"2": [{"kind": "keeper", "polygon": [[700, 340], [730, 340],
                                                          [730, 365], [700, 365]]}]}
        self.assertIsNone(collision_proximity(a, b, regions, 22))

    def test_supported_flight_beats_a_longer_false_keeper_branch(self):
        ds = flight()
        for f in range(21, 38):
            ds.setdefault(str(f), []).append(detection(f, 700 + 15 * (f - 20),
                                                        300 - 15 * (f - 20), confidence=.99))
        regions = {str(f): [{"kind": "keeper", "polygon": [[680, 280], [720, 280],
                                                               [720, 320], [680, 320]]}]
                   for f in range(20, 24)}
        path, _ = run(ds, collision_regions=regions)
        actual = {p["frame"]: p for p in path}
        for f in range(21, 26):
            self.assertEqual(actual[f]["detection_id"], ds[str(f)][0]["detection_id"])

    def test_gradual_image_curve_does_not_require_a_collision(self):
        ds = flight()
        for f in range(11, 26):
            t = f - 10
            ds[str(f)] = [detection(f, 1000 - 30 * t, 400 - 10 * t + .4 * t * t)]
        path, diag = run(ds, initial_path=[ds[str(f)][0] for f in range(11, 14)])
        self.assertEqual(path[-1]["frame"], 25)
        self.assertFalse(diag["turn_frames"])
        self.assertEqual(len([p for p in path if p["frame"] >= 11]), 15)

    def test_existing_whole_flight_is_not_reduced_to_a_shorter_prefix(self):
        ds = flight()
        prefix = [ds[str(f)][0] for f in range(11, 26)]
        for f in range(18, 40):
            ds.setdefault(str(f), []).append(detection(f, 850, 350, confidence=.99))
        path, diag = run(ds, initial_path=prefix)
        self.assertEqual(path[:len(prefix)], [{**d, "motion_time": float(d["frame"])} for d in prefix])
        self.assertEqual(diag["fixed_prefix_observations"], len(prefix))

    def test_forecast_on_observed_frame_is_causal_and_separate(self):
        path = [detection(1, 100, 100), detection(2, 110, 100), detection(3, 120, 100)]
        first = motion_forecasts(path, 1, 4)[0]
        self.assertEqual(first["frame"], 3)
        self.assertEqual(first["predicted_stabilized_xy"], [120., 100.])
        self.assertTrue(first["observation_available"])
        self.assertFalse(first["is_observation"])
        changed_future = path[:2] + [detection(3, 1000, 1000)]
        self.assertEqual(motion_forecasts(changed_future, 1, 4)[0], first)

    def test_no_collision_keeps_model_velocity_direction(self):
        ds = flight()
        path, diag = run(ds)
        velocities = [s["velocity_px_per_source_frame"] for s in diag["motion_states"]]
        for vx, vy in velocities:
            self.assertAlmostEqual(vy / vx, 1 / 3)
        forecasts = motion_forecasts(path, 10, 28, model_states=diag["motion_states"])
        self.assertTrue(forecasts)
        self.assertTrue(all(p["frame"] > diag["motion_states"][0]["frame"] for p in forecasts))

    def test_apparent_speed_change_is_not_a_directional_impact(self):
        ds = flight()
        for f in range(21, 36):
            ds[str(f)] = [detection(f, 700 - 9 * (f - 20), 300 - 3 * (f - 20))]
        path, diag = run(ds)
        self.assertEqual(path[-1]["frame"], 35)
        self.assertFalse(diag["turn_frames"])
        for state in diag["motion_states"]:
            vx, vy = state["velocity_px_per_source_frame"]
            self.assertAlmostEqual(vy / vx, 1 / 3)


if __name__ == "__main__":
    unittest.main()
