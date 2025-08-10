from ultralytics import YOLO
import torch
import os

# ✅ 1. Check if GPU is available
if torch.cuda.is_available():
    device = 0
    print(f"✅ Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("⚠️ CUDA GPU not found, training on CPU.")

# ✅ 2. Paths
# Change these paths to match your setup
data_yaml_path = r"C:\Users\SPS\Desktop\1\yolo\data.yaml"  # Path to your data.yaml
pretrained_model = "yolov8n.pt"  # You can change to yolov8s.pt for better accuracy

# ✅ 3. Load YOLOv8 model
model = YOLO(pretrained_model)

# ✅ 4. Train
model.train(
    data=data_yaml_path,
    epochs=50,       # Increase for better accuracy
    imgsz=640,
    batch=16,        # Reduce if you get out-of-memory errors
    device=device,   # 0 for GPU, 'cpu' for CPU
    name="ball_detector",
    save=True
)

# ✅ 5. Save final model path
print("\n Training complete. Model saved to 'runs/detect/ball_detector/weights/best.pt'")
