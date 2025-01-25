from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Load a pretrained model of ultralytics
# model = YOLO("yolov8n.pt")

# Load a pretrained model
model_path = '/hpc/home/federico.rovighi/3Dperception/yolov8/runs/detect/trainrgb/train_10+5epoche_18k_rgb_pretrained_yolo/weights/last.pt'
model = YOLO(model_path)

# Use the model
model.train(data="config_rgb.yaml", epochs=10)  # train the model