
from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained on COCO (80 classes)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run detection
    annotated_frame = results[0].plot()  # Draw boxes

    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
