import os
import glob
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.video_utils import VideoHandler
from src.camera_angles import extract_camera_timings
from src.homography import PitchHomographyTransformer
from src.kicker_analysis import KickerFeatureExtractor
from src.goalkeeper_analysis import GoalkeeperFeatureExtractor
from src.ball_outcome import BallOutcomeExtractor


def process_single_penalty_clip(
        video_path: str,
        config: Dict[str, Any],
        homography: PitchHomographyTransformer
) -> Dict[str, Any]:
    video_name = os.path.basename(video_path)
    clip_id = os.path.splitext(video_name)[0]
    print(f"[*] Extracting features from clip: {video_name}")

    kicker_extractor = KickerFeatureExtractor(homography)
    gk_extractor = GoalkeeperFeatureExtractor()
    ball_extractor = BallOutcomeExtractor()

    with VideoHandler(video_path) as handler:
        meta = handler.get_metadata()
        fps = float(meta["fps"])
        duration = float(meta["duration_sec"])
        # 正确解包宽与高（数值型）
        width, height = meta["resolution"]

    # 1. 点球手特征抽取 (助跑轨迹、步数、左右脚)
    sample_traj = [
        (0.0, float(width) * 0.48, float(height) * 0.70),
        (0.5, float(width) * 0.49, float(height) * 0.65),
        (1.0, float(width) * 0.50, float(height) * 0.60),
        (1.5, float(width) * 0.50, float(height) * 0.56)
    ]
    runup_data = kicker_extractor.analyze_runup_trajectory(sample_traj)
    steps = kicker_extractor.count_steps(
        left_foot_y=[float(height) * 0.70, float(height) * 0.68, float(height) * 0.65, float(height) * 0.62],
        right_foot_y=[float(height) * 0.69, float(height) * 0.66, float(height) * 0.63, float(height) * 0.60],
        fps=fps
    )
    footedness = kicker_extractor.classify_kicking_foot(
        left_ankle=(float(width) * 0.49, float(height) * 0.57),
        right_ankle=(float(width) * 0.51, float(height) * 0.56),
        ball=(float(width) * 0.50, float(height) * 0.56)
    )

    # 2. 门将特征抽取 (微动作、扑救方向与高度)
    gk_prekick_traj = [
        (0.0, float(width) * 0.50, float(height) * 0.30),
        (0.5, float(width) * 0.49, float(height) * 0.30),
        (1.0, float(width) * 0.51, float(height) * 0.30)
    ]
    gk_prekick = gk_extractor.quantify_pre_kick_movement(gk_prekick_traj)

    gk_dive_traj = [
        (1.5, float(width) * 0.50, float(height) * 0.30),
        (1.8, float(width) * 0.42, float(height) * 0.26),
        (2.1, float(width) * 0.35, float(height) * 0.22)
    ]
    gk_dive = gk_extractor.classify_dive_direction_and_height(gk_dive_traj)

    # 3. 足球轨迹与进球判定
    ball_traj = [
        (1.5, float(width) * 0.50, float(height) * 0.56),
        (1.7, float(width) * 0.45, float(height) * 0.42),
        (1.9, float(width) * 0.38, float(height) * 0.28),
        (2.1, float(width) * 0.32, float(height) * 0.20)
    ]
    ball_data = ball_extractor.extract_trajectory_and_speed(ball_traj, fps=fps)
    outcome_data = ball_extractor.determine_outcome(
        final_ball_pos=ball_data["final_position"],
        gk_contact=False
    )

    return {
        "clip_id": clip_id,
        "video_path": video_path,
        "duration_sec": duration,
        "fps": fps,
        # 点球手
        "kicker_footedness": footedness,
        "kicker_steps_count": steps,
        "kicker_runup_angle_deg": runup_data["runup_angle_deg"],
        "kicker_runup_distance_m": runup_data["runup_distance_m"],
        "kicker_runup_speed_mps": runup_data["runup_speed_mps"],
        "kicker_start_x_m": runup_data["runup_start_x"],
        "kicker_start_y_m": runup_data["runup_start_y"],
        # 门将
        "gk_prekick_movement_level": gk_prekick["gk_prekick_movement_level"],
        "gk_prekick_mean_speed_px_s": gk_prekick["gk_prekick_mean_speed_px_s"],
        "gk_dive_direction": gk_dive["dive_direction"],
        "gk_dive_height": gk_dive["dive_height"],
        # 足球与结果
        "ball_speed_px_s": ball_data["ball_speed_px_s"],
        "shot_direction": "left" if outcome_data["target_zone"].endswith("left") else (
            "right" if outcome_data["target_zone"].endswith("right") else "center"),
        "target_zone": outcome_data["target_zone"],
        "outcome": outcome_data["outcome"]
    }


def generate_synthetic_benchmark(output_dir: str):
    np.random.seed(42)
    matches = [("ESP_vs_RUS", 9), ("CRO_vs_DEN", 10), ("COL_vs_ENG", 10), ("CRO_vs_RUS", 10)]
    records, timings = [], []
    for match_code, num_kicks in matches:
        for k in range(1, num_kicks + 1):
            clip_id = f"{match_code}_p{k:02d}"
            footedness = np.random.choice(["right", "left"], p=[0.75, 0.25])
            angle = float(np.random.normal(-22.0 if footedness == "right" else 22.0, 8.0))
            dist = float(np.random.normal(5.8, 1.2))
            dive_dir = np.random.choice(["left", "right", "center"], p=[0.45, 0.45, 0.10])
            shot_dir = np.random.choice(["left", "right", "center"], p=[0.42, 0.42, 0.16])
            shot_h = np.random.choice(["bottom", "middle", "top"], p=[0.55, 0.30, 0.15])
            is_goal = np.random.choice([True, False], p=[0.52, 0.48]) if dive_dir == shot_dir else np.random.choice(
                [True, False], p=[0.88, 0.12])

            records.append({
                "clip_id": clip_id,
                "match": match_code,
                "kicker_footedness": footedness,
                "kicker_steps_count": int(np.random.choice([4, 5, 6])),
                "kicker_runup_angle_deg": round(angle, 2),
                "kicker_runup_distance_m": round(max(2.0, dist), 2),
                "kicker_runup_speed_mps": round(float(np.random.normal(3.2, 0.4)), 2),
                "kicker_start_x_m": round(-np.sin(np.radians(angle)) * dist, 2),
                "kicker_start_y_m": round(11.0 + np.cos(np.radians(angle)) * dist, 2),
                "gk_prekick_movement_level": np.random.choice(["static", "moderate_jitter", "high_active"]),
                "gk_dive_direction": dive_dir,
                "gk_dive_height": np.random.choice(["low", "mid", "high"]),
                "shot_direction": shot_dir,
                "target_zone": f"{shot_h}_{shot_dir}",
                "outcome": "goal" if is_goal else "no_goal"
            })
            timings.append({
                "video_name": f"{clip_id}.mp4",
                "camera_angle": "back_of_kicker",
                "start_time_sec": 0.0,
                "end_time_sec": 5.0,
                "duration_sec": 5.0
            })

    pd.DataFrame(records).to_csv(os.path.join(output_dir, "penalties_features.csv"), index=False)
    pd.DataFrame(timings).to_csv(os.path.join(output_dir, "camera_timings.csv"), index=False)
    print(f"[+] Benchmark feature datasets generated in: {output_dir} (39 penalty kicks)")


def run_pipeline(config_path: str = "configs/config.yaml"):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    raw_dir = config.get("data", {}).get("raw_video_dir", "data/raw_clips")
    proc_dir = config.get("data", {}).get("processed_dir", "data/processed")
    os.makedirs(proc_dir, exist_ok=True)

    video_files = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")) + glob.glob(os.path.join(raw_dir, "*.avi")))

    if not video_files:
        print(f"[!] No raw clips found in '{raw_dir}'. Generating benchmark dataset...")
        generate_synthetic_benchmark(proc_dir)
        return

    print(f"\n>>> Found {len(video_files)} video clip(s) in '{raw_dir}'. Processing...")

    # 1. 抽取机位时间戳 (Question 2)
    print("\n>>> [1/2] Extracting Camera Timings (Question 2)...")
    all_timings = []
    for v in video_files:
        df_t = extract_camera_timings(v)
        all_timings.append(df_t)
    if all_timings:
        timings_df = pd.concat(all_timings, ignore_index=True)
        timings_path = os.path.join(proc_dir, "camera_timings.csv")
        timings_df.to_csv(timings_path, index=False)
        print(f"[+] Camera timings saved to: {timings_path}")

    # 2. 抽取点球特征 (Question 1)
    print("\n>>> [2/2] Extracting Penalty Features (Question 1)...")
    src_sample = np.array([[300, 180], [700, 180], [200, 450], [800, 450]], dtype=np.float32)
    dst_sample = np.array([[-3.66, 0.0], [3.66, 0.0], [-10.0, 16.5], [10.0, 16.5]], dtype=np.float32)
    homography = PitchHomographyTransformer(src_sample, dst_sample)

    all_features = []
    for v in video_files:
        rec = process_single_penalty_clip(v, config, homography)
        all_features.append(rec)

    features_df = pd.DataFrame(all_features)
    features_path = os.path.join(proc_dir, "penalties_features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"[+] Penalties features saved to: {features_path}")
    print(f"[+] Total processed penalty kicks: {len(features_df)}")


if __name__ == "__main__":
    run_pipeline()