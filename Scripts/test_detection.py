from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/ball_detector/weights/best.pt")

# Video path
video_path = r"C:\Users\SPS\Desktop\1\IMG_7370.MOV"

# Open video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ball
    results = model(frame, conf=0.5)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Ball Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
