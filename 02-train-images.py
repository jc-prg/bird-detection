#!/usr/bin/python3

import os
from ultralytics import YOLO
from dotenv import load_dotenv


def get_env(var_name):
   """
   get value from .env-file if exists

   Args:
       var_name (str): key in .env file
   Returns:
       Any: value from .env file
   """
   try:
      value = os.environ.get(var_name)
   except Exception as e:
      value = None
   return value


if __name__ == "__main__":
   # Load configuration
   path = os.path.join(os.path.dirname(__file__), ".env")
   load_dotenv(path)

   # Load model
   model_name = get_env("TRAIN_SOURCE_MODEL")
   model = YOLO(model_name)

   # Training
   results = model.train(
      data=get_env("TRAIN_DATA_SET"),
      imgsz=int(get_env("TRAIN_IMAGE_SIZE")),
      epochs=int(get_env("TRAIN_EPOCHS")),
      batch=int(get_env("TRAIN_BATCH_SIZE")),
      name=get_env("TRAIN_TARGET_MODEL"),
      device=get_env("TRAIN_DEVICE"),
      )
