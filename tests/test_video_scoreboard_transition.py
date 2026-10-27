from copy import deepcopy

from src.video_scoreboard_transition import outcome_from_transition


def run(start, left, right):
    return {"start_frame": start, "end_frame": start + 12, "frame_ids": [start, start + 6, start + 12],
            "states": {"left": left, "right": right}}


def test_one_changed_icon_does_not_require_human_shootout_ordinal():
    pending = ["pending"] * 5
    before = run(0, ["green", *pending[:4]], ["red", *pending[:4]])
    after = run(90, ["green", "red", *pending[:3]], ["red", *pending[:4]])
    result = outcome_from_transition([before, after], 60, 30)
    assert result["rule_outcome"] == "no_goal_likely"
    assert result["attempt_ordinal_supplied"] is False


def test_missing_pre_contact_panel_cannot_turn_latest_icon_into_current_outcome():
    after = run(90, ["green"] * 5, ["red"] * 5)
    assert outcome_from_transition([after], 60, 30)["rule_outcome"] == "unknown"
    assert outcome_from_transition([after], None, 30)["rule_outcome"] == "unknown"


def test_multiple_changes_and_contradictory_colors_are_rejected():
    before = run(0, ["pending"] * 5, ["pending"] * 5)
    after = run(90, ["green", *["pending"] * 4], ["pending"] * 5)
    second = deepcopy(after)
    second["start_frame"] = 150
    second["states"]["right"][0] = "green"
    assert outcome_from_transition([before, after, second], 60, 30)["rule_outcome"] == "unknown"
    second = deepcopy(after)
    second["states"]["left"][0] = "red"
    assert outcome_from_transition([before, after, second], 60, 30)["rule_outcome"] == "unknown"


def test_panel_remaining_visible_through_contact_has_real_pre_contact_support():
    before = run(0, ["pending"] * 5, ["pending"] * 5)
    before.update(end_frame=72, frame_ids=list(range(0, 73, 6)))
    after = run(90, ["green", *["pending"] * 4], ["pending"] * 5)
    assert outcome_from_transition([before, after], 60, 30)["rule_outcome"] == "goal_likely"
