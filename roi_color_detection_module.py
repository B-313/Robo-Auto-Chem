import json

from color_detection_module import detect_colour_in_frame


def load_roi_from_json(path="roi_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["x1"], data["y1"], data["x2"], data["y2"]


def clamp_roi_to_frame(roi, frame_shape):
    x1, y1, x2, y2 = roi
    frame_h, frame_w = frame_shape[:2]

    x1 = max(0, min(int(x1), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    x2 = max(0, min(int(x2), frame_w))
    y2 = max(0, min(int(y2), frame_h))

    return x1, y1, x2, y2


def detect_colour_in_roi(frame, roi, min_pixels=800):
    x1, y1, x2, y2 = clamp_roi_to_frame(roi, frame.shape)

    if x2 > x1 and y2 > y1:
        roi_frame = frame[y1:y2, x1:x2]
    else:
        roi_frame = frame

    counts = detect_colour_in_frame(roi_frame)
    dominant_colour = max(counts, key=counts.get)
    dominant_pixels = counts[dominant_colour]
    state = dominant_colour if dominant_pixels >= min_pixels else "none"

    return {
        "state": state,
        "counts": counts,
        "roi": (x1, y1, x2, y2),
    }
