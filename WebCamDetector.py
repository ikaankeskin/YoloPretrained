import torch
import cv2

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

print(torch.cuda.is_available()) 

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam is initialized correctly
if not cap.isOpened():
    print("Error: Couldn't open the webcam.")
    exit()

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
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

    # Perform inference
    results = model(frame)

    # Render the results on the frame
    rendered_frame = results.render()[0]

    # Write the frame with detections to the output video
    out.write(rendered_frame)


    # Display the frame
    cv2.imshow('Processed Webcam Feed', rendered_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()