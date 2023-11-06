### I used this with the Yolo to see if it worked. it didn't.



import torch
import cv2
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import time_sync

# Path to your custom trained model's weights
custom_weights_path = 'best.pt'

# Load your trained model directly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(custom_weights_path).to(device)  # Load the model

# Ensure model is in evaluation mode
model.eval()

print(torch.cuda.is_available())

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam is initialized correctly
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Set desired output fps.

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'Results/Recordings/webcam_output.avi'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor and normalize to [0, 1]
    frame_tensor = torch.tensor(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # Perform inference
    results = model(frame_tensor)

    # Apply NMS (non-maximum suppression) and process raw detections
    detections = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)  # You can adjust conf_thres and iou_thres as needed

    # Draw the detections on the frame using OpenCV
    for det in detections:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Log detection to terminal
                print(f"Detected: {label}")

    # Write the frame with detections to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Processed Webcam Feed', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
