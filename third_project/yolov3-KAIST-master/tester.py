import os

from ultralytics import YOLO

import cv2

 

IMAGES_DIR = '/hpc/home/federico.rovighi/3Dperception/yolov3-KAIST-master/test/lwir/'
              #'/hpc/home/federico.rovighi/3Dperception/yolov3-KAIST-master/test/rgb/'

RESULTS_DIR = '/hpc/home/federico.rovighi/3Dperception/yolov3-KAIST-master/test/fused/'

 

# If the output folder doesn't exists create it

os.makedirs(RESULTS_DIR, exist_ok=True)

 

# Images on wich you want to make inference

image_folder=IMAGES_DIR

image_names=[f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

 

model_path = '/hpc/home/federico.rovighi/3Dperception/yolov3-KAIST-master/weights/multiple_step/20epochs_multiple/last.pt'

model = YOLO(model_path)

 

threshold = 0.3

 

for image_name in image_names:

    image_path = os.path.join(IMAGES_DIR, image_name)

 

    # Read the image

    frame = cv2.imread(image_path)

    H, W, _ = frame.shape

 

    # Do the actual inference

    results = model(frame)[0]

 

    for result in results.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = result

 

        if score > threshold:

            # Draw rectangles and writer class name

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 255, 4)

            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),

                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2, cv2.LINE_AA)
            
            cv2.putText(frame, results.names[int(score)].upper(), (int(x1), int(y1 + 10)),

                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2, cv2.LINE_AA)

 

    # Save the image

    output_path = os.path.join(RESULTS_DIR, f'{image_name}_out.jpg')

    cv2.imwrite(output_path, frame)

 

print("Elaboration cpompleted. Images saved in:", RESULTS_DIR)