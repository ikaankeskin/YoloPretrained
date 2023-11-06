import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

print(torch.cuda.is_available())  

input_dir = 'Inputs/Videos'
output_dir = 'Results/Videos'
os.makedirs(output_dir, exist_ok=True)

# List all files in the directory
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output video writer with a modified name
    output_video_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + '.avi')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Use tqdm to wrap the while loop and display a progress bar
    for _ in tqdm(range(total_frames), desc=f"Processing {video_file}", ncols=100):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)
        
        # Render the results on the frame
        rendered_frame = results.render()[0]

        display_frame = cv2.resize(rendered_frame, (int(width // 2), int(height // 2)))  # Reducing the size by half

        # Write the frame to the output video
        out.write(rendered_frame)

         # If you want to display the video frame by frame (optional)
        cv2.imshow('Processed Video', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
