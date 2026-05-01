import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


VIDEO_FOLDER = Path("group_B_videos")
OUTPUT_FOLDER = Path("batch_results")
FRAME_STEP = 5


def select_roi_from_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(f"Could not read the first frame of: {video_path}")

    roi = cv2.selectROI(
        f"Select Vial ROI - {video_path.name}",
        frame,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise ValueError(f"No ROI selected for {video_path.name}")

    return int(x), int(y), int(x + w), int(y + h)


def save_roi_preview(video_path, roi, output_folder):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(f"Could not read the first frame of: {video_path}")

    x1, y1, x2, y2 = roi

    preview = frame.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cropped = frame[y1:y2, x1:x2]

    stem = video_path.stem
    preview_path = output_folder / f"{stem}_roi_preview.jpg"
    crop_path = output_folder / f"{stem}_roi_crop.jpg"

    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(crop_path), cropped)

    print(f"Saved ROI preview to {preview_path}")
    print(f"Saved cropped ROI to {crop_path}")


def analyse_video(video_path, frame_step=5, roi=None):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    rows = []

    if roi is None:
        raise ValueError("ROI must be provided.")

    x1, y1, x2, y2 = roi

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            rgb_roi = rgb[y1:y2, x1:x2]
            hsv_roi = hsv[y1:y2, x1:x2]

            if rgb_roi.size == 0 or hsv_roi.size == 0:
                frame_idx += 1
                continue

            r_mean = rgb_roi[:, :, 0].mean()
            g_mean = rgb_roi[:, :, 1].mean()
            b_mean = rgb_roi[:, :, 2].mean()

            h_mean = hsv_roi[:, :, 0].mean()
            s_mean = hsv_roi[:, :, 1].mean()
            v_mean = hsv_roi[:, :, 2].mean()

            rows.append({
                "video_name": video_path.name,
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "r_mean": r_mean,
                "g_mean": g_mean,
                "b_mean": b_mean,
                "h_mean": h_mean,
                "s_mean": s_mean,
                "v_mean": v_mean,
            })

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def save_plot(df, video_path, output_folder):
    stem = video_path.stem
    plot_path = output_folder / f"{stem}_results_plot.png"

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_s"], df["r_mean"], label="Red mean")
    plt.plot(df["time_s"], df["g_mean"], label="Green mean")
    plt.plot(df["time_s"], df["b_mean"], label="Blue mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean channel value")
    plt.title(f"Colour change over time: {video_path.name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to {plot_path}")


def process_video(video_path, output_folder, frame_step=5):
    print(f"\nProcessing: {video_path.name}")

    roi = select_roi_from_first_frame(video_path)
    print(f"Using ROI coordinates: {roi}")

    save_roi_preview(video_path, roi, output_folder)

    df = analyse_video(video_path, frame_step=frame_step, roi=roi)

    if df.empty:
        print(f"No data extracted for {video_path.name}")
        return None

    csv_path = output_folder / f"{video_path.stem}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    save_plot(df, video_path, output_folder)

    return df


def main():
    if not VIDEO_FOLDER.exists():
        print(f"Video folder not found: {VIDEO_FOLDER}")
        return

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    video_files = sorted(VIDEO_FOLDER.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 files found in {VIDEO_FOLDER}")
        return

    print(f"Found {len(video_files)} mp4 video(s).")

    all_data = []

    for video_path in video_files:
        try:
            df = process_video(video_path, OUTPUT_FOLDER, FRAME_STEP)
            if df is not None:
                all_data.append(df)
        except Exception as e:
            print(f"Skipped {video_path.name} بسبب error: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_csv_path = OUTPUT_FOLDER / "all_videos_combined_results.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\nSaved combined results to {combined_csv_path}")

    print("\nBatch analysis complete.")


if __name__ == "__main__":
    main()