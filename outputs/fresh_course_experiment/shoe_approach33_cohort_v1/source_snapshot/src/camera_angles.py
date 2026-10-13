import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.video_utils import VideoHandler

class ShotDetector:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def detect_cuts(self, video_path: str) -> List[Tuple[int, float]]:
        cuts = [(0, 0.0)]
        prev_hist = None
        with VideoHandler(video_path) as handler:
            for f_idx, t_sec, frame in handler.frame_generator():
                small = cv2.resize(frame, (160, 90))
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], (0, 1), None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                    if diff > self.threshold:
                        cuts.append((f_idx, t_sec))
                prev_hist = hist
            cuts.append((handler.frame_count - 1, handler.duration))
        return cuts

class CameraAngleClassifier:
    def extract_shot_features(self, frame: np.ndarray) -> Dict[str, float]:
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_top = np.sum(mask[:h // 2, :] > 0) / (h * w / 2)
        total_green = np.sum(mask > 0) / (h * w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
        dominant_angle = 0.0
        if lines is not None and len(lines) > 0:
            angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
            dominant_angle = float(np.median(angles))

        return {"green_top": green_top, "total_green": total_green, "angle": dominant_angle}

    def classify_shot(self, sample_frames: List[np.ndarray]) -> str:
        if not sample_frames:
            return "other"
        features = [self.extract_shot_features(f) for f in sample_frames]
        avg_green = np.mean([f["total_green"] for f in features])
        avg_angle = np.mean([f["angle"] for f in features])
        green_top = np.mean([f["green_top"] for f in features])

        if avg_green < 0.25:
            return "slowmo_kicker_closeup"
        elif 25.0 <= abs(avg_angle) <= 65.0:
            return "angle_45_right"
        elif green_top > 0.4:
            return "back_of_kicker"
        else:
            return "slowmo_gk_back"

def extract_camera_timings(video_path: str) -> pd.DataFrame:
    detector = ShotDetector()
    clf = CameraAngleClassifier()
    cuts = detector.detect_cuts(video_path)
    records = []
    with VideoHandler(video_path) as handler:
        for i in range(len(cuts) - 1):
            start_f, start_t = cuts[i]
            end_f, end_t = cuts[i + 1]
            if end_f - start_f < int(handler.fps * 0.5):
                continue
            mid_f = (start_f + end_f) // 2
            samples = []
            for idx in [start_f, mid_f, end_f - 1]:
                handler.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = handler.cap.read()
                if ret:
                    samples.append(frame)
            label = clf.classify_shot(samples)
            records.append({
                "video_name": os.path.basename(video_path),
                "camera_angle": label,
                "start_time_sec": round(start_t, 2),
                "end_time_sec": round(end_t, 2),
                "duration_sec": round(end_t - start_t, 2)
            })
    return pd.DataFrame(records)