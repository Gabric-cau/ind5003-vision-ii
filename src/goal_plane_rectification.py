"""Goal-mouth planar rectification, NOT novel-view or 3-D reconstruction.

Maps image rays onto a display of the goal plane. Off-plane players/balls can
be distorted and have unknown depth. An in-mouth projection is NOT a goal.
"""
import cv2
import numpy as np


def fit_goal_plane(corners, *, target_rect=(60.0, 85.0, 660.0, 285.0)):
    """Corners are image TL,TR,BR,BL; target units are display pixels only."""
    src=np.asarray(corners,dtype=np.float64)
    if src.shape!=(4,2) or not np.all(np.isfinite(src)):
        raise ValueError("Four finite TL/TR/BR/BL image points required")
    if len(np.unique(src,axis=0))!=4 or not cv2.isContourConvex(src.astype(np.float32)):
        raise ValueError("Corners must form a convex non-crossing quadrilateral")
    if abs(cv2.contourArea(src.astype(np.float32)))<25:
        raise ValueError("Degenerate or tiny goal quadrilateral")
    x0,y0,x1,y1=map(float,target_rect)
    if not np.all(np.isfinite(target_rect)) or x1<=x0 or y1<=y0:
        raise ValueError("Invalid target rectangle")
    dst=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float32)
    matrix=cv2.getPerspectiveTransform(src.astype(np.float32),dst)
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix)<3:
        raise ValueError("Singular homography")
    return matrix


def project_image_point(matrix, point):
    xy=np.asarray(point,float)
    if xy.shape!=(2,) or not np.all(np.isfinite(xy)):
        return None
    q=np.asarray(matrix,float)@np.array([xy[0],xy[1],1.])
    if not np.all(np.isfinite(q)) or abs(q[2])<1e-9:return None
    return [float(q[0]/q[2]),float(q[1]/q[2])]


def ball_projection_evidence(matrix, detection, *, target_rect=(60.,85.,660.,285.)):
    """Report a 2-D projection only; never infer whole-ball goal-line crossing."""
    xy=project_image_point(matrix,[detection["x"],detection["y"]])
    x0,y0,x1,y1=target_rect
    inside=bool(xy is not None and x0<=xy[0]<=x1 and y0<=xy[1]<=y1)
    return {"frame":int(detection["frame"]),"source_xy":[detection["x"],detection["y"]],
            "rectified_xy":xy,"center_projects_inside_mouth":inside,
            "goal_plane_depth":None,"whole_ball_crossing_established":False,
            "goal_plane_contact_established":False,"outcome":"unknown",
            "reason":"Planar image projection does not establish ball depth or whole-ball crossing"}
