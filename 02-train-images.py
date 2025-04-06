#!/usr/bin/python3

from ultralytics import YOLO

#model_name = 'yolov8s.pt'
#target_name = 'yolov8m_birds'
model_name = 'yolo11s.pt'
target_name = 'yolo11s_birds'

# Load the model:
# - use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' to start with pretrained default
# - use a former custom model to build up on last training
model = YOLO(model_name)
 
# Training.
results = model.train(
   data='02-train-dataset.yml',
   imgsz=320,
   epochs=10, #50
   batch=4, #8
   name=target_name,
   )
