"""Assemble a silent review video from hashed clips and actual saved figures."""

import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs/fresh_course_experiment"
OUTPUT = EXP / "submission_review_2026-10-21"
SLIDES = OUTPUT / "video_scenes"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR = FONT_DIR / "Arial.ttf"
BOLD = FONT_DIR / "Arial Bold.ttf"
SIZE = (1280, 720)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canvas(title, lines, image_path=None):
    page = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((0, 0, 1280, 78), fill="#202626")
    draw.text((48, 22), title, font=ImageFont.truetype(str(BOLD), 31), fill="white")
    if image_path:
        picture = Image.open(image_path).convert("RGB")
        picture = ImageOps.contain(picture, (1152, 516))
        page.paste(picture, (64 + (1152 - picture.width) // 2, 96 + (516 - picture.height) // 2))
    for i, text in enumerate(lines):
        font = ImageFont.truetype(str(REGULAR), 22)
        if draw.textlength(text, font=font) > 1184:
            raise ValueError(f"Caption does not fit: {text}")
        draw.text((48, 628 + i * 28), text, font=font, fill="#202626")
    draw.line((48, 692, 1232, 692), fill="#c8c8c8")
    draw.text((48, 698), "IND5003 | SILENT REVIEW | Human narration pending | Saved results; no new neural inference",
              font=ImageFont.truetype(str(REGULAR), 14), fill="#626262")
    return page


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    SLIDES.mkdir(parents=True, exist_ok=True)
    scenes = [
        (40, "What can be measured from 39 penalty attempts?", EXP / "final_eda_v3_shoe_fusion/01_fresh_coverage.png",
         ["Four course shootouts. Automatic candidates vary greatly in availability.",
          "Coverage is not accuracy: independent human review remains pending."], False),
        (32, "Rashford: original course clip at normal speed", ROOT / "data/raw_clips/col_eng_p04_rashford_eng19.mp4",
         ["Source clip: col_eng_p04_rashford_eng19 | 960 frames at 30 FPS | 1x playback",
          "Clip bytes match the frozen experiment source hash. No audio is included."], True),
        (20, "Actor, ball and contact evidence", EXP / "review_v3_shoe_fusion_ui3/evidence/col_eng_p04_rashford_eng19_evidence.jpg",
         ["Saved detector and pose evidence for the same event; each feature retains its source.",
          "A visible landmark or high model score does not establish physical contact."], False),
        (40, "Landing support repair: same saved inputs", OUTPUT / "landing_replay_comparison.png",
         ["12 candidates in 7 events became 22 candidates in 12 events after support collection repair.",
          "Rashford: 15 ankle extrema, 3 old shoe proposals, 6 repaired proposals. True steps unverified."], False),
        (34, "Muriel: a no-proposal window at normal speed", ROOT / "data/raw_clips/col_eng_p05_muriel_col14.mp4",
         ["Source clip: col_eng_p05_muriel_col14 | 1,020 frames at 30 FPS | 1x playback",
          "No accepted landing proposal does not mean that the player took zero steps."], True),
        (20, "Inspect an unresolved measurement", EXP / "shoe_landing_source_review_2026-10-18_v1/col_eng_p05_muriel_col14.jpg",
         ["Source-frame review sheet from the saved replay investigation.",
          "Keep the visible interval, model output and a future human reference separate."], False),
        (35, "Source outcome evidence and conflict abstention", EXP / "final_eda_v3_shoe_fusion/05_fresh_scoreboard_outcomes.png",
         ["Scoreboard and combined rules each cover 35/39, but their event sets differ.",
          "Koke conflicts and abstains; Croatia-Denmark Rakitic is supplied by the ball rule."], False),
        (25, "Sparse ground points are not a complete path", EXP / "review_v3_shoe_fusion_ui3/evidence/col_eng_p04_rashford_eng19_ground.png",
         ["32/39 events have pre-contact ground candidates. Continuous path accuracy remains unverified.",
          "Strict complete dive height and verified event-final ball position remain 0/39."], False),
        (24, "Before the final presentation and submission", None,
         ["Use natural human narration and keep the final video within 10 minutes.",
          "This silent review is an editing aid, not the final group presentation."], False),
    ]
    records = []
    cursor = 0
    encoding = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", "30"]
    for index, (duration, title, source, captions, is_video) in enumerate(scenes, 1):
        image_path = SLIDES / f"scene_{index:02d}.png"
        page = canvas(title, captions, None if is_video else source)
        if source is None:
            draw = ImageDraw.Draw(page)
            items = ["1. Review the report and verify every claim.", "2. Complete genuine human annotations.",
                     "3. Resolve incomplete measurements with the teaching team.",
                     "4. Record human narration and a real code demonstration.",
                     "5. Include authorized data, code and reproducible instructions."]
            for n, item in enumerate(items):
                draw.text((78, 142 + n * 82), item, font=ImageFont.truetype(str(REGULAR), 29), fill="#202626")
        page.save(image_path)
        segment = SLIDES / f"scene_{index:02d}.mp4"
        if is_video:
            frozen = json.loads((EXP / "shoe_landing_replay_2026-10-18_v1/events" / source.stem / "landing_result.json").read_text())
            if digest(source) != frozen["source_sha256"]:
                raise ValueError(f"Source clip differs from frozen experiment: {source}")
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-loop", "1", "-i", str(image_path),
                   "-filter_complex", "[0:v]scale=1152:516:force_original_aspect_ratio=decrease,pad=1152:516:(ow-iw)/2:(oh-ih)/2:color=white[v];[1:v][v]overlay=64:96:shortest=1[out]",
                   "-map", "[out]", "-an", "-t", str(duration), *encoding, str(segment)]
        else:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", str(image_path),
                   "-t", str(duration), "-an", *encoding, str(segment)]
        subprocess.run(cmd, check=True)
        records.append({"scene": index, "start_sec": cursor, "duration_sec": duration, "title": title,
                        "source": str(source.relative_to(ROOT)) if source else None,
                        "source_sha256": digest(source) if source else None,
                        "source_playback_speed": 1.0 if is_video else None,
                        "scene_frame_sha256": digest(image_path)})
        cursor += duration
    concat = SLIDES / "concat.txt"
    concat.write_text("".join(f"file 'scene_{i:02d}.mp4'\n" for i in range(1, len(scenes) + 1)), encoding="utf-8")
    final = OUTPUT / "IND5003_Vision_II_Silent_Review_2026-10-21.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    cap = cv2.VideoCapture(str(final))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count != cursor * 30 or fps != 30:
        raise ValueError("Rendered video timing differs from its planned normal-speed segments")
    previews = []
    for record in records:
        cap.set(cv2.CAP_PROP_POS_MSEC, (record["start_sec"] + min(2, record["duration_sec"] / 2)) * 1000)
        ok, frame = cap.read()
        if not ok or frame.std() < 5:
            raise ValueError("Blank or missing scene in rendered video")
        previews.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((426, 240)))
    cap.release()
    contact = Image.new("RGB", (1278, 720), "white")
    for i, preview in enumerate(previews):
        contact.paste(preview, ((i % 3) * 426, (i // 3) * 240))
    contact.save(OUTPUT / "silent_review_contact_sheet.jpg", quality=92)
    manifest = {"status": "silent_review_not_final_submission", "duration_sec": cursor,
                "fps": fps, "frames": frame_count, "audio": False, "human_narration": False,
                "source_video_speed_changed": False, "new_neural_inference": False,
                "video_sha256": digest(final), "scenes": records}
    (OUTPUT / "silent_video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"video": str(final), "duration_sec": cursor, "scenes": len(records), "frames": frame_count}, indent=2))


if __name__ == "__main__":
    main()
