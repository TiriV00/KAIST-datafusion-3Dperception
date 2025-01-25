import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = '/hpc/home/federico.rovighi/3Dperception/fine_presentazione/immagini'
RESULTS_DIR = '/hpc/home/federico.rovighi/3Dperception/fine_presentazione/risultati'

# Create the results directory if it doesn't exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modify the path of the image folder
image_folder = IMAGES_DIR

#  List of all the element in the image dir
image_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

model_path = '/hpc/home/federico.rovighi/3Dperception/yolov8/runs/detect/trainrgb/train_10+5+10epoche_18k_pretrained_yolo/weights/last.pt'
model = YOLO(model_path)

threshold = 0.45

for image_name in image_names:
    image_path = os.path.join(IMAGES_DIR, image_name)

    # Load the image
    frame = cv2.imread(image_path)
    H, W, _ = frame.shape

    # Model inference
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw the rectangle and write class name
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Save the image in the result dir
    output_path = os.path.join(RESULTS_DIR, f'{image_name}_out.jpg')
    cv2.imwrite(output_path, frame)

print("Processing completed. Images saved in:", RESULTS_DIR)
