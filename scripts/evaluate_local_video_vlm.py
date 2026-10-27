"""Run a frozen local image-sequence baseline, then score saved predictions."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import file_sha256


CLASSES = {"kicking_foot": ["left", "right", "unknown"],
           "goal_outcome": ["goal", "no_goal", "unknown"],
           "keeper_ordinal_height": ["low", "middle", "high", "unknown"],
           "keeper_dive_direction": ["image_left", "center", "image_right", "unknown"]}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {**{key: {"type": "string", "enum": values} for key, values in CLASSES.items()},
                         "visual_evidence": {"type": "string", "maxLength": 320}},
          "required": [*CLASSES, "visual_evidence"]}
PROMPT = """Inspect these images as one chronological sequence from one continuous camera shot of a football penalty. Use only visible evidence in these supplied images. Do not infer an answer from player identity, team identity, remembered matches, or expected outcomes. An automatic contact estimate anchors sampling but may be wrong.
Return one JSON object matching the supplied schema. kicking_foot means the kicker's anatomical left or right foot that strikes the ball, not the side of the screen. goal_outcome is goal only when the visible ball outcome supports entry into the goal; no_goal when visible evidence supports a save, miss or goal-frame rebound without entry. Use unknown if the ball result is unresolved. keeper_ordinal_height describes the observed save or dive height in the image: low near the ground, middle roughly torso level, high elevated or overhead. Do not infer dive height from the ball's height or from the goalkeeper merely standing. keeper_dive_direction uses screen coordinates, image_left or image_right, with center for an observed central response. Return unknown for an unobserved or ambiguous action. Briefly state the visual evidence and any uncertainty. These are qualitative categories, not physical measurements.
"""
OPTIONS = {"temperature": .6, "top_p": .95, "top_k": 20, "repeat_penalty": 1.,
           "seed": 27, "num_ctx": 32768, "num_predict": 8192}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def api(base, endpoint, payload=None, timeout=900):
    if urlparse(base).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Only a local Ollama endpoint is permitted")
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base.rstrip("/") + endpoint, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_identity(base, name):
    rows = [row for row in api(base, "/api/tags")["models"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError("The requested model must already be installed locally")
    return rows[0]


def sampling_plan(contact, frame_count, fps, pre=.8, post=1.5, regular_samples=13):
    prediction, window = contact["prediction"], contact["automatic_window"]
    frame = prediction.get("frame")
    if window.get("automatic") is not True or not isinstance(frame, int):
        raise ValueError("A source-bound automatic contact and camera window are required")
    start, stop = int(window["start_frame"]), int(window["end_frame_exclusive"])
    if not 0 <= start <= frame < stop <= frame_count:
        raise ValueError("Contact is outside the automatic continuous camera shot")
    first, last = max(start, frame-round(pre*fps)), min(stop-1, frame+round(post*fps))
    samples = sorted({round(first + i*(last-first)/(regular_samples-1)) for i in range(regular_samples)} | {frame})
    return {"contact_frame": frame, "automatic_shot_start_frame": start,
            "automatic_shot_end_frame_exclusive": stop, "first_frame": first, "last_frame": last,
            "sampled_frames": samples, "pre_context_clipped": first > frame-round(pre*fps),
            "post_context_clipped": last < frame+round(post*fps)}


def extract_frames(video, folder, plan, fps):
    import cv2
    cap = cv2.VideoCapture(str(video))
    wanted, records = set(plan["sampled_frames"]), []
    cap.set(cv2.CAP_PROP_POS_FRAMES, plan["first_frame"])
    try:
        for index in range(plan["first_frame"], plan["last_frame"]+1):
            ok, frame = cap.read()
            decoded = round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            if not ok or decoded != index:
                raise ValueError(f"Video decoding/frame mapping failed: {index} -> {decoded}")
            if index not in wanted:
                continue
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if abs(timestamp-index/fps) > 1/fps:
                raise ValueError("Decoded timestamp differs from the contact frame clock; VFR needs explicit alignment")
            height, width = frame.shape[:2]
            scale = min(1., 960/max(width, height))
            image = cv2.resize(frame, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
            name = f"frame_{index:06d}.jpg"
            if not cv2.imwrite(str(folder/name), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise ValueError("Could not save sampled frame")
            records.append({"frame": index, "decoded_timestamp_seconds": timestamp,
                "contact_relative_seconds": (index-plan["contact_frame"])/fps,
                "source_image_size": [width, height], "input_image_size": [image.shape[1], image.shape[0]],
                "source_pixels_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "image": name, "image_sha256": file_sha256(folder/name)})
    finally:
        cap.release()
    if len(records) != len(wanted):
        raise ValueError("Not all requested frames were decoded")
    return records


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Use a new experiment directory")
    source = read(args.inputs)
    if source.get("reference_labels_included") is not False or len(source["samples"]) != 8:
        raise ValueError("Expected the existing eight source-only video records")
    identity = model_identity(args.endpoint, args.model)
    info = api(args.endpoint, "/api/show", {"model": args.model})
    if "vision" not in info.get("capabilities", []):
        raise ValueError("Installed model does not declare vision support")
    protocol = {"schema_version": "local-video-vlm-development-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_digest": identity["digest"], "model_info": identity,
        "endpoint": args.endpoint, "capabilities": info["capabilities"], "prompt": PROMPT,
        "schema": SCHEMA, "options": OPTIONS, "think": True, "keep_alive": "20m",
        "generation_configuration_sources": ["https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction",
                                             "https://github.com/ollama/ollama/issues/13353"],
        "generation_configuration_reason": "Installed model has fixed thinking renderer; prior short greedy pilots exhausted their output budget without a final answer",
        "sampling": {"pre_seconds": .8, "post_seconds": 1.5, "regular_samples": 13,
                     "include_automatic_contact_frame": True, "max_image_edge": 960, "jpeg_quality": 92,
                     "camera_cut_policy": "truncate_to_the_existing_automatic_contact_shot"},
        "input_manifest_sha256": file_sha256(args.inputs), "script_sha256": file_sha256(__file__),
        "prompt_frozen_before_predictions": True, "reference_labels_at_inference": False,
        "evaluation_kind": "development_retest_on_previously_inspected_match",
        "source_only_input_fields": ["video_path", "video_sha256", "fps", "frame_count", "automatic_contact"],
        "default_pipeline_changed": False}
    save(args.output/"protocol.json", protocol)
    save(args.output/"ollama_show.json", info)
    samples = []
    for sample in source["samples"]:
        started = time.perf_counter()
        cid, video = sample["core_id"], Path(sample["video_path"])
        if Path(cid).name != cid or file_sha256(video) != sample["video_sha256"]:
            raise ValueError("Invalid event ID or source video binding")
        contact_path = args.contacts/cid/"prediction.json"
        contact = read(contact_path)
        if contact["video_sha256"] != sample["video_sha256"]:
            raise ValueError("Contact is bound to a different video")
        folder = args.output/"events"/cid
        folder.mkdir(parents=True)
        plan = sampling_plan(contact, sample["frame_count"], sample["fps"])
        frames = extract_frames(video, folder, plan, sample["fps"])
        record = {"core_id": cid, "video_path": str(video), "video_sha256": sample["video_sha256"],
            "fps": sample["fps"], "contact_prediction_sha256": file_sha256(contact_path),
            "automatic_window": plan, "frames": frames, "reference_labels_used": False,
            "frame_preparation_seconds": time.perf_counter()-started}
        save(folder/"inputs.json", record)
        samples.append({"core_id": cid, "video_sha256": sample["video_sha256"],
                        "input_record_sha256": file_sha256(folder/"inputs.json")})
    save(args.output/"manifest.json", {"protocol_sha256": file_sha256(args.output/"protocol.json"),
                                     "samples": samples, "reference_labels_included": False})
    print(json.dumps({"prepared": len(samples), "output": str(args.output)}, ensure_ascii=False), flush=True)


def validate_prediction(prediction):
    import jsonschema
    jsonschema.validate(prediction, SCHEMA)
    return prediction


def run(args):
    manifest, protocol = read(args.output/"manifest.json"), read(args.output/"protocol.json")
    if file_sha256(args.output/"protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    if file_sha256(__file__) != protocol["script_sha256"]:
        raise ValueError("Inference script differs from the frozen protocol")
    if model_identity(protocol["endpoint"], protocol["model"])["digest"] != protocol["model_digest"]:
        raise ValueError("Installed model differs from frozen protocol")
    completed = 0
    for sample in manifest["samples"]:
        folder = args.output/"events"/sample["core_id"]
        if (folder/"prediction.json").exists():
            continue
        if args.limit is not None and completed >= args.limit:
            break
        started = time.perf_counter()
        record = read(folder/"inputs.json")
        if file_sha256(folder/"inputs.json") != sample["input_record_sha256"]:
            raise ValueError("Frozen input record changed")
        images = []
        for frame in record["frames"]:
            path = folder/frame["image"]
            if file_sha256(path) != frame["image_sha256"]:
                raise ValueError("Sampled image changed")
            images.append(base64.b64encode(path.read_bytes()).decode())
        times = [round(frame["contact_relative_seconds"], 3) for frame in record["frames"]]
        prompt = protocol["prompt"] + "\nImage times in seconds relative to the automatic contact estimate, in supplied order: " + json.dumps(times)
        payload = {"model": protocol["model"], "messages": [{"role": "user", "content": prompt, "images": images}],
            "stream": False, "format": protocol["schema"], "options": protocol["options"],
            "think": protocol["think"], "keep_alive": protocol["keep_alive"]}
        save(folder/"request.json", {"prompt": prompt, "payload_sha256": object_sha(payload),
            "image_count": len(images), "image_hashes": [f["image_sha256"] for f in record["frames"]],
            "schema": protocol["schema"], "options": protocol["options"],
            "event_identifier_sent_to_model": False, "reference_labels_sent_to_model": False})
        result = {"core_id": sample["core_id"], "video_sha256": sample["video_sha256"],
            "input_record_sha256": sample["input_record_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "prediction": {**{key: "unknown" for key in CLASSES}, "visual_evidence": "Inference did not complete"}}
        try:
            raw = api(protocol["endpoint"], "/api/chat", payload)
            save(folder/"raw_response.json", raw)
            if raw.get("done") is not True:
                raise ValueError("Ollama response is incomplete")
            result.update(prediction=validate_prediction(json.loads(raw["message"]["content"])), status="completed",
                          raw_response_sha256=file_sha256(folder/"raw_response.json"),
                          ollama_timing={key: raw.get(key) for key in (
                              "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")})
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save(folder/"failure.json", {"error": result["error"]})
        result["inference_seconds"] = time.perf_counter()-started
        result["end_to_end_stage_seconds"] = result["inference_seconds"]+record["frame_preparation_seconds"]
        save(folder/"prediction.json", result)
        completed += 1
        print(json.dumps({"event": sample["core_id"], "status": result["status"],
                          "seconds": result["inference_seconds"], "prediction": result["prediction"]}), flush=True)
    paths = [args.output/"events"/sample["core_id"]/"prediction.json" for sample in manifest["samples"]]
    if all(path.exists() for path in paths):
        save(args.output/"predictions_frozen.json", {"protocol_sha256": manifest["protocol_sha256"],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "predictions": [{"path": str(path.relative_to(args.output)), "sha256": file_sha256(path)} for path in paths]})


def score(args):
    from scripts.evaluate_worldcup_full_pipeline import read_references
    from scripts.evaluate_offline_video_batch import categorical_metrics
    frozen, manifest = read(args.output/"predictions_frozen.json"), read(args.output/"manifest.json")
    if len(frozen["predictions"]) != 8 or file_sha256(args.output/"protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("All eight predictions and their frozen protocol are required")
    predictions = {}
    for entry in frozen["predictions"]:
        path = args.output/entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("A frozen prediction changed")
        result = read(path)
        predictions[result["core_id"]] = result
    samples = {sample["core_id"]: sample for sample in manifest["samples"]}
    if set(predictions) != set(samples) or any(predictions[cid]["video_sha256"] != samples[cid]["video_sha256"] for cid in samples):
        raise ValueError("Prediction population/source binding mismatch")
    if any(predictions[cid]["protocol_sha256"] != frozen["protocol_sha256"] or
           predictions[cid]["input_record_sha256"] != samples[cid]["input_record_sha256"] for cid in samples):
        raise ValueError("Prediction protocol/input binding mismatch")
    # References are first opened after all frozen predictions pass integrity checks.
    references, binding = read_references(args.references, samples)
    metrics = {}
    for target, refkey in (("kicking_foot", "foot"), ("goal_outcome", "goal_reference"),
                           ("keeper_ordinal_height", "keeper_ordinal_height")):
        rows = [{"core_id": cid, "reference": references[cid][refkey],
                 "prediction": predictions[cid]["prediction"][target]} for cid in sorted(samples)]
        metrics[target] = categorical_metrics(rows, CLASSES[target][:-1])
    runtimes = [p["inference_seconds"] for p in predictions.values()]
    report = {"schema_version": "local-video-vlm-evaluation-v1", "evaluation_kind": "development_retest",
        "model": read(args.output/"protocol.json")["model"], "events": 8, "matches": 1,
        "prediction_freeze_sha256": file_sha256(args.output/"predictions_frozen.json"), "reference_binding": binding,
        "references_opened_after_all_predictions": True, "metrics": metrics,
        "failures": sum(p["status"] != "completed" for p in predictions.values()),
        "runtime": {"inference_median_seconds": statistics.median(runtimes), "inference_total_seconds": sum(runtimes),
                    "per_event_seconds": {cid: p["inference_seconds"] for cid, p in predictions.items()},
                    "automatic_contact_extraction_included": False},
        "unscored": {"keeper_dive_direction": "Reference coordinate conventions are unresolved"},
        "default_pipeline_changed": False,
        "limitations": ["Eight events from one previously inspected match are correlated development evidence.",
            "Images are sparsely sampled from an automatically bounded single shot, not native continuous-video tokens.",
            "Qualitative labels do not establish exact trajectory, timing, coordinates or physical height.",
            "Visual identity and pretrained world knowledge cannot be completely excluded despite source-only prompting.",
            "Unknown and failed predictions count as misses in strict accuracy."]}
    if (args.output/"evaluation.json").exists():
        raise FileExistsError("Evaluation already exists")
    save(args.output/"evaluation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--contacts", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
