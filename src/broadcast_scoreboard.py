"""Conservative parser for the observed red broadcast penalty-scoreboard style."""
import cv2
import numpy as np


def _groups(values):
    groups = []
    for value in values:
        if not groups or value != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(int(value))
    return groups


def _prefix_length(states):
    resolved = [state in ("green", "red") for state in states]
    count = sum(resolved)
    return count if resolved == [True] * count + [False] * (5 - count) else None


def _icon_state(hsv, cx, cy, radius):
    yy, xx = np.ogrid[:hsv.shape[0], :hsv.shape[1]]
    pixels = hsv[(xx - cx) ** 2 + (yy - cy) ** 2 < (.65 * radius) ** 2]
    green = np.mean((pixels[:, 0] >= 30) & (pixels[:, 0] <= 90)
                    & (pixels[:, 1] > 100) & (pixels[:, 2] > 115))
    red = np.mean(((pixels[:, 0] < 12) | (pixels[:, 0] > 168))
                  & (pixels[:, 1] > 100) & (pixels[:, 2] > 100))
    state = ("green" if green >= .35 and red < .10 else
             "red" if red >= .35 and green < .10 else
             "pending" if green < .12 and red < .12 and np.median(pixels[:, 2]) < 135 else "unknown")
    return state, {"green_fraction": float(green), "red_fraction": float(red),
                   "median_value": float(np.median(pixels[:, 2]))}


def extract_penalty_scoreboard(image, *, frame_id):
    """Require a long red bar and two complete, regular five-circle rows."""
    result = {"frame_id": frame_id, "status": "abstain", "reason": "no_supported_scoreboard_geometry",
              "human_verified": False, "accuracy_validated": False,
              "numeric_score_used": False, "reference_labels_used": False}
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a decoded uint8 BGR source frame")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168))
           & (hsv[:, :, 1] > 125) & (hsv[:, :, 2] > 60))
    rows = np.flatnonzero((red.sum(axis=1) > .60 * width)
                          & (np.arange(height) >= .62 * height)
                          & (np.arange(height) < .94 * height))
    hypotheses = []
    for row_group in _groups(rows):
        top, bottom = row_group[0], row_group[-1] + 1
        bar_height = bottom - top
        if not .012 * height <= bar_height <= .07 * height:
            continue
        xs = np.flatnonzero(np.mean(red[top:bottom], axis=0) >= .45)
        if not len(xs) or xs[-1] - xs[0] < .75 * width:
            continue
        left, right = int(xs[0]), int(xs[-1] + 1)
        bar_width = right - left
        low = bottom + max(2, int(round(.004 * height)))
        high = min(height, bottom + int(round(2.0 * bar_height)))
        white = ((hsv[:, :, 1] < 145) & (hsv[:, :, 2] > 145)).astype(np.uint8) * 255
        white[:low] = 0
        white[high:] = 0
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / perimeter ** 2 if perimeter else 0.
            cx, cy = x + (cw - 1) / 2., y + (ch - 1) / 2.
            fraction = (cx - left) / bar_width
            if (.016 * height <= cw <= .045 * height and .016 * height <= ch <= .045 * height
                    and .80 <= cw / ch <= 1.25 and circularity >= .67
                    and (.20 < fraction < .47 or .53 < fraction < .80)):
                circles.append({"xy": [cx, cy], "radius": (cw + ch) / 4.,
                                "circularity": float(circularity), "bbox": [x, y, x + cw, y + ch]})
        circle_rows = []
        for circle in sorted(circles, key=lambda row: row["xy"][1]):
            if not circle_rows or circle["xy"][1] - np.mean([r["xy"][1] for r in circle_rows[-1]]) > .006 * height:
                circle_rows.append([])
            circle_rows[-1].append(circle)
        for circle_row in circle_rows:
            teams = [[r for r in circle_row if (r["xy"][0] < (left + right) / 2.) == is_left]
                     for is_left in (True, False)]
            if any(len(team) != 5 for team in teams):
                continue
            teams = [sorted(team, key=lambda row: row["xy"][0]) for team in teams]
            if any(np.std(np.diff([r["xy"][0] for r in team])) / np.mean(np.diff([r["xy"][0] for r in team])) > .12
                   or not 1.05 <= np.mean(np.diff([r["xy"][0] for r in team])) / (2 * np.mean([r["radius"] for r in team])) <= 1.60
                   or np.std([r["radius"] for r in team]) / np.mean([r["radius"] for r in team]) > .12
                   for team in teams):
                continue
            states = []
            for team in teams:
                group = []
                for circle in team:
                    state, color = _icon_state(hsv, *circle["xy"], circle["radius"])
                    circle.update(state=state, color_evidence=color)
                    group.append(state)
                states.append(group)
            if any("unknown" in team or _prefix_length(team) is None for team in states):
                continue
            counts = [_prefix_length(team) for team in states]
            if abs(counts[0] - counts[1]) > 1:
                continue
            hypotheses.append({"bar_bbox": [left, top, right, bottom], "icons": {"left": teams[0], "right": teams[1]},
                               "states": {"left": states[0], "right": states[1]},
                               "resolved_counts": {"left": counts[0], "right": counts[1]},
                               "resolved_total": sum(counts)})
    if len(hypotheses) != 1:
        if hypotheses:
            result["reason"] = "ambiguous_scoreboard_geometry"
        return result
    return {**result, **hypotheses[0], "status": "candidate_scoreboard", "reason": "complete_regular_icon_geometry",
            "icon_semantics": "green=goal candidate, red=no-goal candidate, dark numbered ring=pending; unvalidated broadcast convention"}


def stable_scoreboard_runs(observations, *, fps, maximum_gap_seconds=1.1, minimum_frames=3):
    """Retain only consecutive matching full icon layouts, with no hidden gaps."""
    runs = []
    current = []
    previous_frame = None
    signature = None
    for row in sorted(observations, key=lambda value: value["frame_id"]):
        fid = row["frame_id"]
        new_signature = (tuple(row["states"]["left"]), tuple(row["states"]["right"])) if row["status"] == "candidate_scoreboard" else None
        if (new_signature is None or new_signature != signature or previous_frame is None
                or fid - previous_frame > fps * maximum_gap_seconds):
            if len(current) >= minimum_frames and (current[-1]["frame_id"] - current[0]["frame_id"]) / fps >= .4:
                runs.append(current)
            current = []
        if new_signature is not None:
            current.append(row)
        previous_frame, signature = fid, new_signature
    if len(current) >= minimum_frames and (current[-1]["frame_id"] - current[0]["frame_id"]) / fps >= .4:
        runs.append(current)
    return [{"start_frame": run[0]["frame_id"], "end_frame": run[-1]["frame_id"],
             "frame_ids": [row["frame_id"] for row in run], "states": run[0]["states"],
             "resolved_counts": run[0]["resolved_counts"], "resolved_total": run[0]["resolved_total"]} for run in runs]


def infer_first_shooting_side(runs):
    sides = set()
    for run in runs:
        if run["resolved_total"] % 2:
            left, right = run["resolved_counts"]["left"], run["resolved_counts"]["right"]
            if abs(left - right) == 1:
                sides.add("left" if left > right else "right")
    return next(iter(sides)) if len(sides) == 1 else None


def outcome_from_ordinal(runs, *, attempt_ordinal, after_frame, first_shooting_side=None):
    """Read an ordinal's colored icon, requiring a stable complete cumulative panel."""
    unknown = {"status": "unknown", "rule_outcome": "unknown", "reason": "no_stable_post_attempt_ordinal_panel",
               "attempt_ordinal": attempt_ordinal, "human_verified": False, "accuracy_validated": False,
               "numeric_score_used": False, "ball_goal_plane_crossing_verified": False}
    if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int) or not 1 <= attempt_ordinal <= 10:
        raise ValueError("Penalty ordinal must be 1..10")
    candidates = [run for run in runs if run["start_frame"] >= after_frame and run["resolved_total"] == attempt_ordinal]
    witness = []
    for run in candidates:
        if attempt_ordinal % 2:
            side = "left" if run["resolved_counts"]["left"] > run["resolved_counts"]["right"] else "right"
        elif first_shooting_side in ("left", "right"):
            side = "right" if first_shooting_side == "left" else "left"
        else:
            continue
        slot = (attempt_ordinal - 1) // 2
        state = run["states"][side][slot]
        if state not in ("green", "red"):
            continue
        witness.append({"side": side, "slot_index": slot, "state": state,
                        "frame_ids": run["frame_ids"], "states": run["states"],
                        "resolved_total": run["resolved_total"]})
    if not witness:
        if candidates:
            unknown["reason"] = "shooting_order_side_not_established"
        return unknown
    if len({(row["side"], row["slot_index"], row["state"]) for row in witness}) != 1:
        return {**unknown, "reason": "contradictory_ordinal_icon_witnesses"}
    state = witness[0]["state"]
    return {**unknown, "status": "candidate_scoreboard_outcome", "rule_outcome": "goal_likely" if state == "green" else "no_goal_likely",
            "reason": "stable_complete_panel_matches_attempt_ordinal", "witnesses": witness,
            "first_shooting_side": first_shooting_side,
            "scope": "Broadcast icon outcome candidate; not independently verified event endpoint or ball crossing",
            "reference_labels_used": False}


def render_scoreboard_evidence(image, detection):
    overlay = image.copy()
    if detection["status"] != "candidate_scoreboard":
        return overlay
    left, top, right, bottom = detection["bar_bbox"]
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 0, 255), 2)
    colors = {"green": (0, 255, 0), "red": (0, 0, 255), "pending": (220, 220, 220)}
    for side, icons in detection["icons"].items():
        for icon in icons:
            center = tuple(int(round(value)) for value in icon["xy"])
            cv2.circle(overlay, center, int(round(icon["radius"])) + 3, colors[icon["state"]], 1)
        states = {"green": "G", "red": "R", "pending": "-"}
        label = side[0].upper() + ": " + " ".join(states[icon["state"]] for icon in icons)
        first = icons[0]["xy"]
        cv2.putText(overlay, label, (int(first[0]) - 9, int(first[1]) + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (245, 245, 245), 1)
    cv2.putText(overlay, f"f{detection['frame_id']} | broadcast icon candidates | G=green R=red -=pending | unverified",
                (64, 25), cv2.FONT_HERSHEY_SIMPLEX, .44, (245, 245, 245), 1)
    return overlay
