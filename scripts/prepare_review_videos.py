"""Make browser-compatible, full-duration review copies of four failed clips."""
import json
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(r"C:\Users\30605\Downloads\VisionII_failed_four_review")
CLIPS = [
    ("cro_den_p06_modric_cro10", "01_Modric_Croatia_Denmark.mp4"),
    ("esp_rus_p06_golovin_rus17", "02_Golovin_Spain_Russia.mp4"),
    ("rus_cro_p07_ignashevich_rus4", "03_Ignashevich_Russia_Croatia.mp4"),
    ("rus_cro_p08_vida_cro21", "04_Vida_Russia_Croatia.mp4"),
]


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
    ok, _ = cap.read()
    cap.release()
    return {"frame_count": count, "fps": fps, "duration": count / fps,
            "codec": "".join(chr((codec >> (8 * i)) & 255) for i in range(4)), "last_frame_decoded": ok}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for clip_id, name in CLIPS:
        source = ROOT / "data/raw_clips" / (clip_id + ".mp4")
        target = DEST / name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing review copy: {target}")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-movflags", "+faststart", str(target)], check=True)
        original, converted = metadata(source), metadata(target)
        if original["frame_count"] != converted["frame_count"] or not converted["last_frame_decoded"]:
            raise ValueError(f"Incomplete conversion: {clip_id}")
        records.append({"clip_id": clip_id, "source": str(source), "playback_copy": str(target),
                        "original": original, "converted": converted})
        print(json.dumps(records[-1]), flush=True)
    (DEST / "validation.json").write_text(json.dumps(records, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
