from ultralytics import YOLO
import torch

device = 0 if torch.cuda.is_available() else 'cpu'

model = YOLO("yolov8n.pt")

model.train(
    data="path/to/your/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=device,
    name="ball_detector",
    save=True
)

print("\nTraining complete. Model saved to 'runs/detect/ball_detector/weights/best.pt'")
