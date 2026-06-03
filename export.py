from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/logo_detection/weights/best.pt"

def main():
    model = YOLO(MODEL_PATH)
    # 导出为 ONNX 格式
    model.export(format="onnx", imgsz=640)
    print("ONNX 模型导出完成！")

if __name__ == "__main__":
    main()