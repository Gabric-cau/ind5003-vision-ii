import unittest
from src.goalkeeper_dive_phase import analyse_phases, mask_phase_candidates


def sequence(peak=.60):
    rows = [dict(frame=f, hip=[0, 50], hip_clearance_body=.5, torso_lean_degrees=0,
                 bottom_clearance_body=0, wrist_clearance_body=None) for f in range(7, 10)]
    for f in range(10, 23):
        landing = f >= 18
        rows.append(dict(frame=f, hip=[30, 60], hip_clearance_body=.15 if landing else peak,
                         torso_lean_degrees=70, bottom_clearance_body=0 if landing else .10,
                         wrist_clearance_body=None))
    return rows


class PhaseTests(unittest.TestCase):
    def test_high_then_land_not_low(self):
        result = analyse_phases(sequence(.9), 10, 30, 100)
        self.assertEqual(result['height_candidate'], 'high')
        self.assertTrue(result['complete_first_dive_observed'])
        self.assertLess(max(result['height_supporting_frames']), result['landing_frame'])

    def test_mid(self):
        self.assertEqual(analyse_phases(sequence(), 10, 30, 100)['height_candidate'], 'mid')

    def test_low(self):
        self.assertEqual(analyse_phases(sequence(.25), 10, 30, 100)['height_candidate'], 'low')

    def test_source_cut_before_landing(self):
        result = analyse_phases(sequence(.9)[:9], 10, 30, 100, source_censored=True)
        self.assertFalse(result['height_supported'])
        self.assertEqual(result['height_candidate'], 'unknown')

    def test_pose_gap_cannot_bridge_to_landing(self):
        rows = [s for s in sequence(.9) if not 13 <= s['frame'] <= 19]
        self.assertFalse(analyse_phases(rows, 10, 30, 100)['height_supported'])

    def test_camera_failure(self):
        self.assertFalse(analyse_phases(sequence(), 10, 30, 100, camera_supported=False)['height_supported'])

    def test_boundary_abstain(self):
        self.assertFalse(analyse_phases(sequence(.35), 10, 30, 100)['height_supported'])

    def test_upright_protest_not_dive(self):
        rows = sequence()
        for s in rows:
            s.update(hip=[0, 50], torso_lean_degrees=0)
        self.assertEqual(analyse_phases(rows, 10, 30, 100)['height_candidate'], 'unknown')

    def test_no_reference(self):
        self.assertFalse(analyse_phases(sequence()[3:], 10, 30, 100)['height_supported'])

    def test_depth_motion_not_low_height(self):
        rows = sequence(.25)
        for s in rows[3:8]: s['bottom_clearance_body'] = -.3
        self.assertFalse(analyse_phases(rows,10,30,100)['height_supported'])


def silhouettes(air_center_y=85):
    rows=[dict(frame=f,center=[0,50],lower_edge=[0,100],axis_lean=0,
               vertical_extent_body=1.,elongation=3.) for f in range(0,10)]
    rows += [dict(frame=f,center=[30,110 if f>=18 else air_center_y],
                  lower_edge=[30,120 if f>=18 else 95],axis_lean=70,
                  vertical_extent_body=.4,elongation=3.) for f in range(10,25)]
    return rows


class MaskPhaseTests(unittest.TestCase):
    def test_low_proxy(self):
        result=mask_phase_candidates(silhouettes(),10,30,100)
        self.assertEqual(result['height_proxy_candidate'],'low')
        self.assertEqual(result['landing_frame'],18)
        self.assertLess(max(result['supporting_frames']),18)
        self.assertFalse(result['physical_ground_plane_calibrated'])

    def test_high_flight_low_landing(self):
        result=mask_phase_candidates(silhouettes(20),10,30,100)
        self.assertEqual(result['height_proxy_candidate'],'high')

    def test_missing_landing_abstains(self):
        self.assertFalse(mask_phase_candidates(silhouettes()[:17],10,30,100)['supported'])

    def test_gap_cannot_fake_five_frame_settling(self):
        rows=[s for s in silhouettes() if s['frame'] not in [20,23]]
        self.assertFalse(mask_phase_candidates(rows,10,30,100)['complete_first_dive_candidate'])

    def test_upright_motion_not_landing(self):
        rows=silhouettes()
        for s in rows: s['axis_lean']=0; s['vertical_extent_body']=1.
        self.assertFalse(mask_phase_candidates(rows,10,30,100)['supported'])


if __name__ == '__main__':
    unittest.main()
