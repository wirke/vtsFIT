from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=1)  # train the model

