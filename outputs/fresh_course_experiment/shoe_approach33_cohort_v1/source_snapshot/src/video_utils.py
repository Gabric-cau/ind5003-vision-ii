import os
import cv2
from typing import Dict, Generator, Tuple, Any, Optional

class VideoHandler:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at: {video_path}")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "file_path": self.video_path,
            "filename": os.path.basename(self.video_path),
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "resolution": (self.width, self.height),
            "duration_sec": round(self.duration, 2)
        }

    def frame_generator(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[Tuple[int, float, Any], None, None]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        target_end = end_frame if end_frame is not None else self.frame_count
        while curr_frame < target_end:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield curr_frame, curr_frame / self.fps, frame
            curr_frame += 1

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

def slice_video(input_path: str, output_path: str, start_sec: float, end_sec: float) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with VideoHandler(input_path) as handler:
        start_frame = int(start_sec * handler.fps)
        end_frame = int(end_sec * handler.fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, handler.fps, (handler.width, handler.height))
        for _, _, frame in handler.frame_generator(start_frame, end_frame):
            out.write(frame)
        out.release()
    return output_path