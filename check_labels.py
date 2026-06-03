from ultralytics import YOLO
model = YOLO("best.pt")
print(model.names)  # {0: 'bureau_veritas', 1: 'eurofins', ...}