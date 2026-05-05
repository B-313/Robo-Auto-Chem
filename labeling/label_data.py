import cv2
import csv
import shutil
from pathlib import Path


# ====== USER SETTINGS ======
INPUT_FOLDER = Path("training_data")
OUTPUT_FOLDER = Path("labelling_output")
LABELLED_FOLDER = OUTPUT_FOLDER / "labelled_images"
UNLABELLED_FOLDER = OUTPUT_FOLDER / "unlabelled_images"
LABELS_CSV = OUTPUT_FOLDER / "liquid_labels.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ===========================


def ensure_folders():
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    LABELLED_FOLDER.mkdir(exist_ok=True)
    UNLABELLED_FOLDER.mkdir(exist_ok=True)


def get_image_files(folder):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    )


def load_existing_labels(csv_path):
    labels = {}
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["filename"]] = row
    return labels


def save_labels(csv_path, labels):
    fieldnames = ["filename", "x1", "y1", "x2", "y2"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filename in sorted(labels.keys()):
            writer.writerow(labels[filename])


def copy_labelled_images(image_files, labels):
    # Clear old contents first
    for f in LABELLED_FOLDER.iterdir():
        if f.is_file():
            f.unlink()

    for image_path in image_files:
        if image_path.name in labels:
            shutil.copy2(image_path, LABELLED_FOLDER / image_path.name)


def copy_unlabelled_images(image_files, labels):
    # Clear old contents first
    for f in UNLABELLED_FOLDER.iterdir():
        if f.is_file():
            f.unlink()

    for image_path in image_files:
        if image_path.name not in labels:
            shutil.copy2(image_path, UNLABELLED_FOLDER / image_path.name)


def save_progress(image_files, labels):
    save_labels(LABELS_CSV, labels)
    copy_labelled_images(image_files, labels)
    copy_unlabelled_images(image_files, labels)


def print_instructions():
    print("\n=== Liquid Labelling Tool ===")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("\nInstructions:")
    print("1. An image window will appear.")
    print("2. Draw a box around the LIQUID ONLY.")
    print("3. Press ENTER or SPACE to confirm.")
    print("4. Press C to cancel and redraw.")
    print("5. If you want to stop, type FINISH in the terminal when prompted.")
    print("6. If you close the ROI window or make no selection, that image will be skipped for now.\n")


def annotate_images():
    if not INPUT_FOLDER.exists():
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    ensure_folders()
    image_files = get_image_files(INPUT_FOLDER)

    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        return

    labels = load_existing_labels(LABELS_CSV)

    print_instructions()
    print(f"Found {len(image_files)} image(s).")
    print(f"Already labelled: {len(labels)}")

    for idx, image_path in enumerate(image_files, start=1):
        if image_path.name in labels:
            print(f"Skipping already labelled image: {image_path.name}")
            continue

        print(f"\nImage {idx}/{len(image_files)}: {image_path.name}")
        user_choice = input("Press Enter to label this image, or type FINISH to stop: ").strip()

        if user_choice.upper() == "FINISH":
            print("\nStopping early at user request.")
            break

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not open image: {image_path.name}")
            continue

        # Resize for easier display on screen
        max_width = 1000
        max_height = 800

        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)

        display_image = cv2.resize(
            image,
            (int(w * scale), int(h * scale))
        )

        print("Highlight the liquid area and press Enter or Space.")

        roi = cv2.selectROI(
            f"Label liquid - {image_path.name}",
            display_image,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyAllWindows()

        x, y, rw, rh = roi

        if rw == 0 or rh == 0:
            print(f"No valid selection made for {image_path.name}. Leaving it unlabelled.")
            save_progress(image_files, labels)
            continue

        # Convert back to original image coordinates
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + rw) / scale)
        y2 = int((y + rh) / scale)

        labels[image_path.name] = {
            "filename": image_path.name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

        print(f"Saved label for {image_path.name}: ({x1}, {y1}, {x2}, {y2})")
        save_progress(image_files, labels)

    # Final save
    save_progress(image_files, labels)

    labelled_count = len(labels)
    unlabelled_count = len([p for p in image_files if p.name not in labels])

    print("\n=== Finished ===")
    print(f"Labelled images saved in: {LABELLED_FOLDER}")
    print(f"Unlabelled images saved in: {UNLABELLED_FOLDER}")
    print(f"Labels CSV saved to: {LABELS_CSV}")
    print(f"Total labelled: {labelled_count}")
    print(f"Total unlabelled: {unlabelled_count}")


if __name__ == "__main__":
    annotate_images()