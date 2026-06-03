from ultralytics import YOLO
model = YOLO("runs/detect/runs/logo_detection/weights/best.pt")
print(model.names)  # {0: 'bureau_veritas', 1: 'eurofins', ...}