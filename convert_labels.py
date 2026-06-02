"""Convert YOLO segmentation polygon labels to bounding box format."""
from pathlib import Path

DATASET_DIR = Path("dataset")

def poly_to_bbox(values):
    xs = values[0::2]
    ys = values[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]

def convert_file(label_path):
    with open(label_path) as f:
        lines = f.readlines()
    
    converted = []
    changed = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        cls_id = int(parts[0])
        coords = parts[1:]
        if len(coords) == 4:
            converted.append(line)
        else:
            xc, yc, w, h = poly_to_bbox(coords)
            converted.append(f"{cls_id} {xc:.10f} {yc:.10f} {w:.10f} {h:.10f}")
            changed = True
    
    if changed:
        with open(label_path, "w") as f:
            f.write("\n".join(converted) + "\n")
        print(f"  Converted: {label_path.name}")
        return True
    return False

def main():
    for split in ["train", "valid", "test"]:
        label_dir = DATASET_DIR / split / "labels"
        if not label_dir.exists():
            continue
        print(f"\nProcessing {split}/labels ...")
        count = 0
        for label_path in sorted(label_dir.iterdir()):
            if label_path.suffix == ".txt":
                if convert_file(label_path):
                    count += 1
        print(f"  {count} files converted in {split}")

if __name__ == "__main__":
    main()
