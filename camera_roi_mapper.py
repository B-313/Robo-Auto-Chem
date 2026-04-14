import json
from datetime import datetime

import cv2 as cv

# Match these with your main routine camera settings
camera_index = 0
frame_width = 1280
frame_height = 720

# Backend candidates to probe in order. Keep only the ones relevant to your OS.
backend_candidates = [
    ("CAP_V4L2", cv.CAP_V4L2),
    ("CAP_GSTREAMER", cv.CAP_GSTREAMER),
    ("CAP_ANY", cv.CAP_ANY),
]

# Output file for reuse in your routine
roi_output_file = "roi_config.json"


def open_camera_with_candidates(index, width, height, candidates):
    for backend_name, backend_id in candidates:
        cap = cv.VideoCapture(index, backend_id)
        if not cap.isOpened():
            cap.release()
            print(f"Backend {backend_name}: open failed")
            continue

        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            print(f"Backend {backend_name}: opened but frame read failed")
            continue

        print(f"Backend {backend_name}: OK ({frame.shape[1]}x{frame.shape[0]})")
        return cap, backend_name

    return None, None


def main():
    cap, selected_backend = open_camera_with_candidates(
        camera_index,
        frame_width,
        frame_height,
        backend_candidates,
    )
    if cap is None:
        raise RuntimeError("Cannot open camera with any configured backend candidate.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture a frame for ROI selection.")

    print("Instructions:")
    print("1) Drag a box around the vial and press Enter or Space.")
    print("2) Press c to cancel selection and retry.")

    x, y, w, h = cv.selectROI("Select vial ROI", frame, showCrosshair=True, fromCenter=False)
    cv.destroyAllWindows()
    cap.release()

    if w == 0 or h == 0:
        print("No ROI selected. Exiting.")
        return

    roi = {
        "x1": int(x),
        "y1": int(y),
        "x2": int(x + w),
        "y2": int(y + h),
        "width": int(w),
        "height": int(h),
        "camera_index": camera_index,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "backend": selected_backend,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(roi_output_file, "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)

    print("\nROI selected:")
    print(f"x1={roi['x1']}, y1={roi['y1']}, x2={roi['x2']}, y2={roi['y2']}")
    print(f"Saved to {roi_output_file}")
    print("\nPaste into your routine as:")
    print(f"ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = {roi['x1']}, {roi['y1']}, {roi['x2']}, {roi['y2']}")


if __name__ == "__main__":
    main()
