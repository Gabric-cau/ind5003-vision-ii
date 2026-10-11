"""Build source-frame inventory and a review gallery without changing predictions."""
from pathlib import Path
import argparse
import json
import html
import shutil
import zipfile
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/goal_right_camera_review_2026-10-11"
VIDEO = ROOT / "deliverables/VisionII_Current_Deliverable_2026-10-10/02_video_review/camera/videos"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", nargs="*")
    args = parser.parse_args()
    rows = json.loads((ROOT / "outputs/optimization_evening_2026-10-09/integration/canonical_results_39.json").read_text(encoding="utf-8"))["rows"]
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    panels = []
    inventory = []
    for row in rows:
        cid = row["clip_id"]
        frame = int(row["contact"]["contact_frame"] or 0) + 3
        cap = cv2.VideoCapture(str(VIDEO / (cid + ".mp4")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Frame decode failed: {cid} / {frame}")
        dest = OUT / "frames" / (cid + ".jpg")
        cv2.imwrite(str(dest), img)
        panel = np.zeros((280, 480, 3), dtype=np.uint8)
        panel[40:280] = cv2.resize(img, (480, 240))
        cv2.putText(panel, cid, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
        cv2.putText(panel, f"frame {frame}: old={row['foot']['baseline_candidate']} new={row['foot']['nominal_v4_candidate']}", (5, 33), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
        panels.append(panel)
        inventory.append({"clip_id": cid, "sample_frame": frame, "old_foot": row["foot"]["baseline_candidate"], "new_foot": row["foot"]["nominal_v4_candidate"], "new_foot_stable": row["foot"]["parameter_stable"], "frame_file": str(dest.relative_to(OUT)), "source_video": str(VIDEO / (cid + ".mp4"))})
    for start in range(0, len(panels), 12):
        batch = panels[start:start+12]
        sheet = np.zeros((280 * 4, 480 * 3, 3), dtype=np.uint8)
        for j, panel in enumerate(batch):
            y, x = divmod(j, 3)
            sheet[y*280:(y+1)*280, x*480:(x+1)*480] = panel
        cv2.imwrite(str(OUT / f"inventory_{start//12+1}.jpg"), sheet)
    (OUT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.selected is None:
        print(json.dumps({"sampled_source_clips": len(inventory), "output": str(OUT)}))
        return
    selected = set(args.selected)
    assert selected <= {r["clip_id"] for r in inventory}
    (OUT / "videos").mkdir(exist_ok=True)
    cards = []
    checks = []
    for row in inventory:
        if row["clip_id"] not in selected:
            continue
        cid = row["clip_id"]
        target = OUT / "videos" / (cid + ".mp4")
        shutil.copy2(row["source_video"], target)
        cap = cv2.VideoCapture(str(target))
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == expected and count > 0
        checks.append({"clip_id": cid, "decoded_frames": count, "expected_frames": expected})
        speed = ''.join(f'<button onclick="this.closest(\'article\').querySelector(\'video\').playbackRate={v}">{v}×</button> ' for v in (.25, .5, 1))
        cards.append(f'<article><h2>{html.escape(cid)}</h2><p>旧版判脚：{html.escape(row["old_foot"])}；新版候选：{html.escape(row["new_foot"])}；新版参数稳定：{row["new_foot_stable"]}。均非逐片人工确认。</p><video controls preload="none" poster="{row["frame_file"]}" src="videos/{cid}.mp4"></video><p>{speed}<a href="videos/{cid}.mp4" download>下载视频</a> · <a href="{row["frame_file"]}">查看原画面采样帧</a></p></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球门在画面右侧：判脚复核</title><style>body{font-family:system-ui;background:#15202b;color:#eee;max-width:1200px;margin:auto;padding:20px}article{background:#243443;padding:16px;margin:20px 0;border-radius:10px}video{width:100%;max-height:560px}a{color:#88ceff}h2{font-size:19px}p{line-height:1.6}</style><h1>球门在画面右侧：原视频与新旧判脚对照</h1><p>按用户示例的画面布局筛选：球门右侧、罚球者从左侧接近。这是助手对源帧的视觉分组，不是水平镜像检测，也不是解剖左右脚真值。没有互换或修改任何预测。视频为同原片段的兼容播放副本，保留后续机位变化。</p>'''
    page += f"<p>已检查39条触球附近源画面，本页纳入{len(cards)}条。窗口锚点可能有误，不能保证找全所有回放中的同类机位。</p>" + "".join(cards) + "</html>"
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "selection.json").write_text(json.dumps({"grouping": "AI visual inspection of all 39 sampled source frames; goal on image right", "user_example": "cro_den_p07_schone_den19", "human_foot_labels_used": False, "predictions_changed": False, "selected": [r for r in inventory if r["clip_id"] in selected]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "validation.json").write_text(json.dumps({"source_frame_samples": len(inventory), "selected_clips": len(cards), "decode_checks": checks, "accuracy_validated": False}, indent=2), encoding="utf-8")
    bundle = ROOT / "deliverables/VisionII_Goal_Right_Camera_Review_10_Videos_2026-10-11.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("index.html", "selection.json", "validation.json"):
            z.write(OUT / name, name)
        for cid in sorted(selected):
            for folder, suffix in (("videos", ".mp4"), ("frames", ".jpg")):
                p = OUT / folder / (cid + suffix)
                z.write(p, str(p.relative_to(OUT)))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
    print(json.dumps({"selected_clips": len(cards), "all_videos_decoded": True, "index": str(OUT / "index.html")}))


if __name__ == "__main__":
    main()
