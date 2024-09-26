# Real-Time Object Detection using YOLOv9 and OpenCV

This project demonstrates real-time object detection using YOLOv9, OpenCV, and PyTorch. It utilizes the YOLOv9 model to capture video from a webcam, detects objects in each frame, and displays bounding boxes around detected objects along with their class labels in real-time.

## Aim

The aim of this project is to implement an efficient real-time object detection system using a pre-trained YOLOv9 model. The system captures video from the webcam, processes each frame, and identifies objects by drawing bounding boxes around them with corresponding labels, ensuring high performance with the aid of GPU acceleration.

## Features

- Real-time object detection using YOLOv9.
- Utilizes a webcam for live video capture.
- Adjustable confidence and IoU thresholds for fine-tuning detection sensitivity.
- Customizable bounding box visualization with class labels.
- Supports GPU acceleration using CUDA (if available).
## Program
```
import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import os

@smart_inference_mode()
def detect_real_time(weights='yolov9/yolov9-c.pt', imgsz=640, conf_thres=0.60, iou_thres=0.45):
    # Initialize
    device = select_device('0')
    # Check if model file exists
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Model file {weights} not found.")
    
    try:
        model = DetectMultiBackend(weights=weights, device=device, fp16=False, data='data/coco.yaml')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    stride, names, pt = model.stride, model.names, model.pt

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess image
        img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        try:
            pred = model(img, augment=False, visualize=False)
            print(f"Inference prediction: {pred}")
        except Exception as e:
            print(f"Error during inference: {e}")
            continue

        # Apply NMS
        try:
            pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)
            print(f"NMS predictions: {pred}")
        except Exception as e:
            print(f"Error during NMS: {e}")
            continue

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                print(f"Rescaled detections: {det}")

                # Draw bounding boxes
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=[0, 255, 0], line_thickness=3)

        # Display the resulting frame
        cv2.imshow('Real-Time Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    if color is None:
        color = [255, 0, 0]
    if line_thickness is None:
        line_thickness = 3
    xyxy = list(map(int, xyxy))
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=line_thickness)
    if label:
        tf = max(line_thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
        c1 = (xyxy[0], xyxy[1] - t_size[1] - 3)
        c2 = (xyxy[0] + t_size[0], xyxy[1])
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 2), 0, 0.5, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
```
# Run the detection
detect_real_time()

## Output
![imagerec](https://github.com/user-attachments/assets/db2bcd69-1cfc-4fdd-afdb-ed357e966457)

## Result

- **Real-Time Object Detection**: The program successfully detects multiple objects in real-time, drawing bounding boxes and labels with high accuracy.
- **Performance**: On a CUDA-enabled GPU, the frame rate and detection speed are optimal for real-time applications.
