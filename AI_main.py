import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO


VIDEO_FOLDER = Path("robochem_videos_alexius")
OUTPUT_FOLDER = Path("batch_results")
MODEL_PATH = Path("best.pt")   # change this if your model file has a different name
FRAME_STEP = 15                # analyse every 15th frame for speed
CONF_THRESHOLD = 0.25


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))


def detect_liquid_bbox_with_model(frame, model, conf_threshold=0.25):
    """
    Use the trained YOLO model to detect the liquid in a frame.
    Returns (x1, y1, x2, y2) for the best detection.
    """
    results = model.predict(frame, conf=conf_threshold, verbose=False)

    if not results or len(results[0].boxes) == 0:
        raise ValueError("No liquid detected by model in first frame.")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best_idx]

    return int(x1), int(y1), int(x2), int(y2)


def refine_liquid_mask_in_bbox(frame, bbox):
    """
    Inside the model-predicted box, refine to a liquid mask using HSV.
    This improves the region used for colour analysis.
    """
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Broad threshold for likely liquid colours
    lower = np.array([0, 30, 20])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask)

    if contours:
        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) > 30:
            cv2.drawContours(cleaned_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = cleaned_mask

    return full_mask


def save_previews(first_frame, liquid_bbox, liquid_mask, output_folder, stem):
    x1, y1, x2, y2 = liquid_bbox

    # Model detection box
    bbox_preview = first_frame.copy()
    cv2.rectangle(bbox_preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(output_folder / f"{stem}_detected_liquid_box.jpg"), bbox_preview)

    # Overlay mask
    overlay = first_frame.copy()
    green = np.zeros_like(first_frame)
    green[:, :] = (0, 255, 0)

    mask_3ch = cv2.merge([liquid_mask, liquid_mask, liquid_mask])
    overlay = np.where(mask_3ch > 0, (0.6 * overlay + 0.4 * green).astype(np.uint8), overlay)
    cv2.imwrite(str(output_folder / f"{stem}_detected_liquid_overlay.jpg"), overlay)

    # Binary mask
    cv2.imwrite(str(output_folder / f"{stem}_detected_liquid_mask.jpg"), liquid_mask)

    # Tight box around mask
    tight_preview = first_frame.copy()
    ys, xs = np.where(liquid_mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        lx1, lx2 = xs.min(), xs.max()
        ly1, ly2 = ys.min(), ys.max()
        cv2.rectangle(tight_preview, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)

    cv2.imwrite(str(output_folder / f"{stem}_detected_liquid_tight_box.jpg"), tight_preview)


def compute_masked_stats(frame, mask, frame_idx, fps, video_name):
    pixels = mask > 0
    if pixels.sum() == 0:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    rgb_vals = rgb[pixels]
    hsv_vals = hsv[pixels]

    return {
        "video_name": video_name,
        "frame": frame_idx,
        "time_s": frame_idx / fps,
        "area_px": int(pixels.sum()),
        "r_mean": float(rgb_vals[:, 0].mean()),
        "g_mean": float(rgb_vals[:, 1].mean()),
        "b_mean": float(rgb_vals[:, 2].mean()),
        "h_mean": float(hsv_vals[:, 0].mean()),
        "s_mean": float(hsv_vals[:, 1].mean()),
        "v_mean": float(hsv_vals[:, 2].mean()),
    }


def analyse_video(video_path, output_folder, model, frame_step=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the first frame.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Detect liquid ONCE on the first frame
    liquid_bbox = detect_liquid_bbox_with_model(first_frame, model, CONF_THRESHOLD)

    # Refine the liquid mask on the first frame and save previews
    first_mask = refine_liquid_mask_in_bbox(first_frame, liquid_bbox)
    save_previews(first_frame, liquid_bbox, first_mask, output_folder, video_path.stem)

    # Restart video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            # Reuse the same bounding box for the whole video
            liquid_mask = refine_liquid_mask_in_bbox(frame, liquid_bbox)
            row = compute_masked_stats(frame, liquid_mask, frame_idx, fps, video_path.name)

            if row is not None:
                rows.append(row)

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows), liquid_bbox


def save_plot(df, output_folder, stem):
    plot_path = output_folder / f"{stem}_results_plot.png"

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["r_mean"], label="Red mean")
    plt.plot(df["time_s"], df["g_mean"], label="Green mean")
    plt.plot(df["time_s"], df["b_mean"], label="Blue mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean channel value")
    plt.title(f"Liquid colour change over time: {stem}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def process_video(video_path, output_folder, model, frame_step=5):
    print(f"\nProcessing: {video_path.name}")

    df, liquid_bbox = analyse_video(video_path, output_folder, model, frame_step)

    print(f"Detected liquid bbox in first frame: {liquid_bbox}")

    if df.empty:
        print(f"No liquid region detected for {video_path.name}")
        return None

    csv_path = output_folder / f"{video_path.stem}_results.csv"
    df.to_csv(csv_path, index=False)
    save_plot(df, output_folder, video_path.stem)

    print(f"Saved results to {csv_path}")
    print(f"Saved plot to {output_folder / f'{video_path.stem}_results_plot.png'}")

    return df


def main():
    if not VIDEO_FOLDER.exists():
        print(f"Video folder not found: {VIDEO_FOLDER}")
        return

    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}")
        return

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    video_files = sorted(VIDEO_FOLDER.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {VIDEO_FOLDER}")
        return

    print(f"Found {len(video_files)} mp4 video(s).")
    print(f"Using model: {MODEL_PATH}")

    model = load_model(MODEL_PATH)

    all_data = []

    for video_path in video_files:
        try:
            df = process_video(video_path, OUTPUT_FOLDER, model, FRAME_STEP)
            if df is not None:
                all_data.append(df)
        except Exception as e:
            print(f"Skipped {video_path.name} because of error: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv_path = OUTPUT_FOLDER / "all_videos_combined_results.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined results to {combined_csv_path}")

    print("\nBatch analysis complete.")


if __name__ == "__main__":
    main()