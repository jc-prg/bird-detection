#!/usr/bin/python3

from ultralytics import YOLO
 
# Load the model:
# - use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' to start with pretrained default
# - use a former custom model to build up on last training
model = YOLO('yolov8m.pt')
 
# Training.
results = model.train(
   data='02-train-dataset.yml',
   imgsz=640,
   epochs=50,
   batch=8,
   name='yolov8m_custom'
   )
