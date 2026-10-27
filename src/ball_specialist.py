"""Label-free full-frame and tiled inference for a ball detector."""
from __future__ import annotations


def tile_windows(width, height, edge=960, overlap=.25):
    if edge <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile geometry")

    def starts(length):
        if length <= edge:
            return [0]
        step = max(1, round(edge * (1 - overlap)))
        return sorted(set([*range(0, length - edge + 1, step), length - edge]))

    return [(x, y, min(x + edge, width), min(y + edge, height))
            for y in starts(height) for x in starts(width)]


def ball_class(model):
    names = model.names.items() if isinstance(model.names, dict) else enumerate(model.names)
    matches = [int(key) for key, name in names if name in {"ball", "sports ball"}]
    if len(matches) != 1:
        raise ValueError("Ambiguous or unsupported ball class mapping")
    return matches[0]


def predict_images(model, images, mode, imgsz, device, batch=4):
    import torch
    from torchvision.ops import nms

    if mode not in {"full", "tiles"} or min(imgsz, batch) <= 0:
        raise ValueError("Invalid ball inference settings")
    class_id = ball_class(model)
    patches, origins = [], []
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        windows = [(0, 0, width, height)]
        if mode == "tiles":
            windows += [w for w in tile_windows(width, height) if w != windows[0]]
        for x1, y1, x2, y2 in windows:
            patches.append(image[y1:y2, x1:x2])
            origins.append((index, x1, y1))
    candidates = [[] for _ in images]
    for start in range(0, len(patches), batch):
        group = patches[start:start + batch]
        results = model.predict(group, imgsz=imgsz, conf=.05, iou=.6,
                                max_det=20, classes=[class_id], device=device, verbose=False)
        if len(results) != len(group):
            raise ValueError("Ball detector batch output length mismatch")
        for result, (index, x, y) in zip(results, origins[start:start + batch]):
            boxes, scores = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
            if len(boxes) != len(scores):
                raise ValueError("Unaligned ball detection tensors")
            for box, confidence in zip(boxes, scores):
                candidates[index].append({"bbox": [box[0] + x, box[1] + y, box[2] + x, box[3] + y],
                                          "confidence": confidence})
    for index, rows in enumerate(candidates):
        if rows:
            keep = nms(torch.tensor([r["bbox"] for r in rows], dtype=torch.float32),
                       torch.tensor([r["confidence"] for r in rows], dtype=torch.float32), .6)
            candidates[index] = [rows[i] for i in keep.tolist()[:20]]
    return candidates
