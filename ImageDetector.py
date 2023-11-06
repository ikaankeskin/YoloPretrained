
##loops over the Inputs and Outputs the Results with label boxes

import torch
import cv2
import os
from pathlib import Path

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Loop through all images in the folder
input_dir = 'Inputs/Images'
output_dir = Path('Results/Images')
output_dir.mkdir(exist_ok=True)

for image_file in os.listdir(input_dir):
    if not (image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png') or image_file.lower().endswith('.jpeg')):
        continue

    img_path = os.path.join(input_dir, image_file)
    img = cv2.imread(img_path)

    # Perform inference
    results = model(img)
    
    # Render the results on the image
    rendered_img = results.render()[0]
    
    # Save the results using OpenCV
    save_path = output_dir / image_file
    cv2.imwrite(str(save_path), rendered_img)

    # If you want to display the image
    # cv2.imshow('Result', rendered_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
