import unittest
import numpy as np
from src.goal_plane_rectification import fit_goal_plane,project_image_point,ball_projection_evidence


class GoalPlaneRectificationTests(unittest.TestCase):
    def setUp(self):self.corners=np.array([[40,80],[220,60],[230,190],[35,210]],float)
    def test_corners_map_to_rectangle(self):
        h=fit_goal_plane(self.corners)
        for src,dst in zip(self.corners,[[60,85],[660,85],[660,285],[60,285]]):
            np.testing.assert_allclose(project_image_point(h,src),dst,atol=1e-5)
    def test_degenerate_rejected(self):
        for points in ([[0,0],[1,0],[2,0],[3,0]],[[0,0],[9,0],[9,9],[0,0]],[[0,0],[9,9],[9,0],[0,9]]):
            with self.assertRaises(ValueError):fit_goal_plane(points)
    def test_invalid_coordinates_rejected(self):
        c=self.corners.copy();c[0,0]=float("nan")
        with self.assertRaises(ValueError):fit_goal_plane(c)
    def test_inside_projection_does_not_mean_goal(self):
        h=fit_goal_plane([[0,0],[10,0],[10,10],[0,10]])
        e=ball_projection_evidence(h,{"frame":12,"x":5,"y":5})
        self.assertTrue(e["center_projects_inside_mouth"])
        self.assertEqual(e["outcome"],"unknown")
        self.assertFalse(e["whole_ball_crossing_established"])
    def test_outside_projection_does_not_mean_no_goal(self):
        h=fit_goal_plane([[0,0],[10,0],[10,10],[0,10]])
        e=ball_projection_evidence(h,{"frame":2,"x":25,"y":5})
        self.assertFalse(e["center_projects_inside_mouth"])
        self.assertEqual(e["outcome"],"unknown")
    def test_single_view_depth_ambiguity(self):
        # Camera (0,0,-10), goal plane Z=0: two points on opposite sides
        # of the plane lie on the same viewing ray and have identical pixels.
        before=np.array([.8,.4,-2.]);after=np.array([1.2,.6,2.])
        np.testing.assert_allclose(before[:2]/(before[2]+10),after[:2]/(after[2]+10))
        self.assertLess(before[2],0);self.assertGreater(after[2],0)
    def test_horizon_or_nonfinite_projection_unavailable(self):
        h=np.array([[1,0,0],[0,1,0],[1,0,0]],float)
        self.assertIsNone(project_image_point(h,[0,1]))
        self.assertIsNone(project_image_point(np.eye(3),[float("nan"),1]))


if __name__=="__main__":unittest.main()
