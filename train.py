from ultralytics import YOLO

DATA_YAML = "dataset/data.yaml"
EPOCHS = 100
BATCH = 16
IMGSZ = 640
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

def main():
    model = YOLO("yolo26n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        workers=4,
        project="runs",
        name="logo_detection",
        exist_ok=True,
        patience=20,
        lr0=0.01,
        augment=True,
    )
    print(f"Training complete. Best model saved to {results.save_dir / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()
