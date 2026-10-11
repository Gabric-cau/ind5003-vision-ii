import cv2
import numpy as np
import pytest

from src.broadcast_scoreboard import (extract_penalty_scoreboard, infer_first_shooting_side,
                                      outcome_from_ordinal, stable_scoreboard_runs)


def panel(left=("green", "green", "green", "pending", "pending"),
          right=("green", "green", "red", "pending", "pending")):
    image = np.full((720, 1280, 3), (30, 65, 35), np.uint8)
    cv2.rectangle(image, (64, 590), (1215, 620), (0, 0, 155), -1)
    colors = {"green": (10, 180, 10), "red": (0, 0, 180), "pending": (30, 50, 30)}
    for states, start in ((left, 422), (right, 764)):
        for index, state in enumerate(states):
            center = (start + index * 24, 641)
            cv2.circle(image, center, 9, colors[state], -1)
            cv2.circle(image, center, 9, (235, 245, 235), 2)
            cv2.putText(image, str(index + 1), (center[0] - 3, center[1] + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, .27, (250, 250, 250), 1)
    return image


def observations(image, frames=(0, 6, 12)):
    return [extract_penalty_scoreboard(image, frame_id=frame) for frame in frames]


def test_complete_panel_and_even_attempt_use_colors_not_numeric_score():
    rows = observations(panel())
    assert all(row["status"] == "candidate_scoreboard" for row in rows)
    assert rows[0]["states"]["right"] == ["green", "green", "red", "pending", "pending"]
    runs = stable_scoreboard_runs(rows, fps=30)
    result = outcome_from_ordinal(runs, attempt_ordinal=6, after_frame=0, first_shooting_side="left")
    assert result["rule_outcome"] == "no_goal_likely"
    assert result["witnesses"][0]["slot_index"] == 2
    assert result["numeric_score_used"] is False
    assert result["human_verified"] is False
    assert result["ball_goal_plane_crossing_verified"] is False


def test_odd_cumulative_count_infers_first_side_without_country_labels():
    image = panel(left=("red", "green", "pending", "pending", "pending"),
                  right=("red", "green", "green", "pending", "pending"))
    runs = stable_scoreboard_runs(observations(image), fps=30)
    assert infer_first_shooting_side(runs) == "right"
    result = outcome_from_ordinal(runs, attempt_ordinal=5, after_frame=0)
    assert result["rule_outcome"] == "goal_likely"
    assert result["witnesses"][0]["side"] == "right"


def test_even_attempt_without_first_side_and_wrong_ordinal_abstain():
    runs = stable_scoreboard_runs(observations(panel()), fps=30)
    assert outcome_from_ordinal(runs, attempt_ordinal=6, after_frame=0)["rule_outcome"] == "unknown"
    assert outcome_from_ordinal(runs, attempt_ordinal=5, after_frame=0)["rule_outcome"] == "unknown"
    assert outcome_from_ordinal(runs, attempt_ordinal=6, after_frame=1, first_shooting_side="left")["rule_outcome"] == "unknown"


def test_missing_bar_and_incomplete_animation_do_not_detect_icons():
    image = panel()
    image[590:621] = (30, 65, 35)
    assert extract_penalty_scoreboard(image, frame_id=0)["status"] == "abstain"
    image = panel()
    image[630:652, 835:878] = (30, 65, 35)
    assert extract_penalty_scoreboard(image, frame_id=0)["status"] == "abstain"


def test_out_of_order_resolved_icon_is_rejected():
    image = panel(left=("green", "pending", "green", "pending", "pending"))
    assert extract_penalty_scoreboard(image, frame_id=0)["status"] == "abstain"


def test_short_runs_gaps_and_absent_frames_cannot_establish_stability():
    assert stable_scoreboard_runs(observations(panel(), (0, 6)), fps=30) == []
    assert stable_scoreboard_runs(observations(panel(), (0, 90, 180)), fps=30) == []
    rows = observations(panel(), (0, 6, 12, 18, 24))
    rows[2] = {"frame_id": 12, "status": "abstain"}
    assert stable_scoreboard_runs(rows, fps=30) == []


def test_contradictory_stable_icons_and_shooting_orders_abstain():
    first = stable_scoreboard_runs(observations(panel()), fps=30)[0]
    second = {**first, "start_frame": 100, "end_frame": 112, "frame_ids": [100, 106, 112],
              "states": {**first["states"], "right": ["green", "green", "green", "pending", "pending"]}}
    result = outcome_from_ordinal([first, second], attempt_ordinal=6, after_frame=0, first_shooting_side="left")
    assert result["reason"] == "contradictory_ordinal_icon_witnesses"
    assert result["rule_outcome"] == "unknown"
    odd_left = {"resolved_total": 1, "resolved_counts": {"left": 1, "right": 0}}
    odd_right = {"resolved_total": 3, "resolved_counts": {"left": 1, "right": 2}}
    assert infer_first_shooting_side([odd_left, odd_right]) is None


@pytest.mark.parametrize("ordinal", [True, 0, 11, 1.5])
def test_noncanonical_ordinals_rejected(ordinal):
    with pytest.raises(ValueError):
        outcome_from_ordinal([], attempt_ordinal=ordinal, after_frame=0)
