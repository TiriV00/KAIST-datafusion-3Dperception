from ultralytics import YOLO  
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import copy

def bbox_extractor(input_path, model):
    predictions = model.predict(input_path) 
    # Make prediction on the image link:https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode

    # Get the bbox coord and sizes alongsides the confidences and the classes
    bbox = []
    confidence = []
    classes  = []

    for box in predictions:
        bbox.append(box.boxes.xywhn)
        confidence.append(box.boxes.conf)
        classes.extend([int(c) for c in box.boxes.cls])

    bbox = bbox[0]
    confidence = confidence[0]
    confidence = confidence.view(-1, 1) # Convert confidence into a column element

    # initialize the list of the output and join altogether the elements of bbox and confidence
    info_bbox = torch.cat((bbox, confidence), dim=1)
    infobbox = []
    precision = 4 # Floating point element precision (number of decimal digit)

    # Iterates all the bbox
    for i, bbox in enumerate(info_bbox):
        center_x = round(float((bbox[0] + bbox[2] / 2).item()), precision)
        center_y = round(float((bbox[1] + bbox[3] / 2).item()), precision)
        width = round(float(bbox[2].item()), precision)
        height = round(float(bbox[3].item()), precision)
        confidence_val = round(float(confidence[i].item()), precision)
        # What we called centerx and centery are actually the bottom right vertex coord

        # Create a list with all the values
        infobbox.append([ center_x, 
            center_y,
            width,
            height,
            confidence_val,
            classes[i]])
    return infobbox

def calculate_iou(box1, box2):

    # Compute the Intersection over Union (IoU) between two boxes
    x1, y1, w1, h1, conf1, classes1 = box1
    x2, y2, w2, h2, conf2, classes2 = box2
    
    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection_area = x_intersection * y_intersection
    union_area = w1 * h1 + w2 * h2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def data_fusion(boxes1, boxes2, confidence_threshold=0.5, iou_threshold=0.5):
    # Cost matrix creation, based on IoU
    cost_matrix = np.zeros((len(boxes1), len(boxes2)))
    print(cost_matrix.shape)

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            cost_matrix[i, j] = 1 - calculate_iou(box1, box2)

    # Resolution of the assignment problem through the hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    fused_boxes = []

     # Data fusion of the corresponding boxes
    for i, j in zip(row_ind, col_ind):
        iou = calculate_iou(boxes1[i], boxes2[j])
        confidence = (boxes1[i][4] + boxes2[j][4]) / 2

        if iou > iou_threshold and confidence > confidence_threshold:
            fused_box = copy.deepcopy(boxes1[i])  # Create a copy of the object
            fused_box[4] = confidence
            fused_boxes.append(fused_box)

    return fused_boxes

def draw_and_save_bbox(boxes, image, image_path):
    # Scroll through all the subject detected
    for elements in boxes:
        centerx, centery, width, height, confidence, classes = elements
        topleftv = (int((centerx - width) * width_image), int((centery - height) * height_image))
        bottomrightv = (int((centerx) * width_image), int((centery) * height_image))
        # Draw the bbox
        cv2.rectangle(image, topleftv, bottomrightv, 255, 2)
        print('class before modication is :', classes)

        if classes == 0:
            classes_str = "person"
        elif classes == 1:
            classes_str = "people"
        elif classes == 2:
            classes_str = "cyclist"

        print('class after modification is:', classes_str)
        # Write the class name and the confidence 
        cv2.putText(image, classes_str, (int((centerx - width) * width_image), int((centery - height) * height_image - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.putText(image, "{:.3f}".format(confidence), (int((centerx - width) * width_image), int((centery + height) * height_image - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    cv2.imwrite(image_path, image)

# Part to modify
model_path_lwir = '/hpc/home/federico.rovighi/3Dperception/yolov8/runs/detect/trainlwir/train_pretrained_yolo/train_15+10epoche_18k/weights/last.pt'
model_path_rgb = '/hpc/home/federico.rovighi/3Dperception/yolov8/runs/detect/trainrgb/train_10+5+10epoche_18k_pretrained_yolo/weights/last.pt'
model_lwir = YOLO(model_path_lwir)
model_rgb = YOLO(model_path_rgb)
image_lwir_path = "/hpc/home/federico.rovighi/3Dperception/yolov8/datafusion_nolearn/imagetest/lwir/I00009.jpg"
image_rgb_path = "/hpc/home/federico.rovighi/3Dperception/yolov8/datafusion_nolearn/imagetest/rgb/I00009.jpg"
lwir_bbox =bbox_extractor(image_lwir_path, model_lwir) 
rgb_bbox = bbox_extractor(image_rgb_path, model_rgb)
fused_boxes = data_fusion(lwir_bbox, rgb_bbox)

print('the rgb boxes=', rgb_bbox, '\nthe lwir boxes=', lwir_bbox, '\nthe fused boxes=', fused_boxes)

image = cv2.imread('imagetest/rgb/I00009.jpg')
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_lwir = cv2.imread('imagetest/lwir/I00009.jpg', cv2.IMREAD_GRAYSCALE)
image_rgbcopy = np.copy(image_rgb)
image_lwircopy = np.copy(image_lwir)
height_image, width_image = image_lwir.shape
print('image lwir shape =', image_lwir.shape, f'width value is {width_image} and height value is {height_image}')
draw_and_save_bbox(lwir_bbox, image_lwir, "imagetest/fused/lwir.jpg")
draw_and_save_bbox(rgb_bbox, image_rgb, "imagetest/fused/rgb.jpg")
draw_and_save_bbox(fused_boxes, image_lwircopy, "imagetest/fused/fusedlwir.jpg")
draw_and_save_bbox(fused_boxes, image_rgbcopy, "imagetest/fused/fusedrgb.jpg")
print('all images saved')