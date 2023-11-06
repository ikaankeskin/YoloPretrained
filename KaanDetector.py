import cv2
import torch
import os

def convert_to_yolo_format(xyxy, img_width, img_height):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2

    # Normalize
    w /= img_width
    h /= img_height
    x_center /= img_width
    y_center /= img_height

    return x_center, y_center, w, h

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

video_path = 'Results\Recordings\kaan.avi'
video_out = 'Results\Recordings\kaan_out.avi'

output_images_dir = 'Results/KImages'
output_labels_dir = 'Results/KLabels'

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(video_out, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
confidence_threshold = 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_objects = results.pred[0]

    # Create a list to collect YOLO format annotations for this frame
    yolo_annotations = []
    
    for *xyxy, conf, cls in detected_objects:
        if conf > confidence_threshold and results.names[int(cls)] == 'person':
            label = 'kaan'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Convert bbox to YOLO format and append to annotations list
            yolo_bbox = convert_to_yolo_format(xyxy, frame.shape[1], frame.shape[0])
            yolo_annotations.append((80, *yolo_bbox))  # Assuming "kaan" is class index 80

    # Save frame as image and annotations as .txt
    if yolo_annotations:
        img_name = f"frame_{frame_count}.jpg"
        label_name = f"frame_{frame_count}.txt"
        
        cv2.imwrite(os.path.join(output_images_dir, img_name), frame)
        
        with open(os.path.join(output_labels_dir, label_name), 'w') as label_file:
            for annotation in yolo_annotations:
                label_file.write(f"{annotation[0]} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f}\n")

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
