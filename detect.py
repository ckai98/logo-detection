from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/logo_detection/weights/best.pt"
SOURCE = "dataset/test/images"
CONF = 0.25
OUTPUT_DIR = "runs/detect_results"

def main():
    model = YOLO(MODEL_PATH)
    names = model.names
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(Path(SOURCE).glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {SOURCE}")
        return

    for img_path in image_paths:
        results = model(img_path, conf=CONF)[0]
        img = cv2.imread(str(img_path))
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{names[cls_id]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
