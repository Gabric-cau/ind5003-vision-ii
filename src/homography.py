import cv2
import numpy as np
from typing import List, Tuple, Optional


class PitchHomographyTransformer:
    def __init__(self, src_pts: Optional[np.ndarray] = None, dst_pts: Optional[np.ndarray] = None,
                 *, destination_unit: str = "arbitrary"):
        if destination_unit not in {"arbitrary", "m"}:
            raise ValueError("destination_unit must be 'arbitrary' or 'm'")
        self.H = None
        self.destination_unit = destination_unit
        self.diagnostics = {}
        if src_pts is not None and dst_pts is not None:
            self.compute_homography(src_pts, dst_pts)

    def compute_homography(self, src_points: np.ndarray, dst_points: np.ndarray,
                           *, reprojection_threshold_px: float = 3.0):
        """Fit ground landmarks. A four-point fit is not independent validation."""
        self.H = None
        self.diagnostics = {}
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
            raise ValueError("Need at least four matched Nx2 point pairs")
        if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
            raise ValueError("Landmarks must be finite")
        if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0:
            raise ValueError("Reprojection threshold must be finite and positive")
        for points in (src, dst):
            if len(np.unique(points, axis=0)) != len(points) or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
                raise ValueError("Duplicate or collinear landmarks cannot calibrate the pitch")
        # Inverse fit keeps RANSAC tolerance in image pixels for metric destinations.
        inverse, inliers = cv2.findHomography(dst, src, cv2.RANSAC, reprojection_threshold_px)
        if inverse is None or not np.all(np.isfinite(inverse)) or np.linalg.matrix_rank(inverse) < 3:
            raise ValueError("Homography estimation failed or is singular")
        selected = inliers.reshape(-1).astype(bool)
        if selected.sum() < 4:
            raise ValueError("Fewer than four inlier landmarks")
        projected = cv2.perspectiveTransform(dst.reshape(1, -1, 2), inverse)[0]
        errors = np.linalg.norm(projected - src, axis=1)
        self.H = np.linalg.inv(inverse)
        self.diagnostics = {"point_count": len(src), "inlier_count": int(selected.sum()),
                            "inlier_rmse_px": float(np.sqrt(np.mean(errors[selected] ** 2))),
                            "all_point_rmse_px": float(np.sqrt(np.mean(errors ** 2))),
                            "destination_unit": self.destination_unit,
                            "independently_validated": False}
        return dict(self.diagnostics)

    @property
    def metric_calibrated(self):
        return self.H is not None and self.destination_unit == "m"

    def transform_trajectory(self, trajectory: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not trajectory or self.H is None:
            return trajectory

        clean_pts = []
        for pt in trajectory:
            px, py = pt
            clean_pts.append((float(px), float(py)))

        pts = np.array([clean_pts], dtype=np.float32)
        homogeneous = np.column_stack((np.asarray(clean_pts), np.ones(len(clean_pts)))) @ self.H.T
        if not np.all(np.isfinite(homogeneous)) or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            raise ValueError("Trajectory projects onto the homography horizon")
        transformed = cv2.perspectiveTransform(pts, self.H)

        results = []
        for item in transformed[0]:
            tx, ty = item
            results.append((float(tx), float(ty)))

        return results
