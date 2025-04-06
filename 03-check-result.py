#!/bin/python3


import os
import random
import json
import codecs
import logging
import shutil

from detection_v8 import ImageHandling, DetectionModel
from dotenv import load_dotenv


custom_model_path = custom_model_file = yolo_run_path = ""


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


def set_vars():
    global custom_model_path, custom_model_file, yolo_run_path

    custom_model_path = str(get_env("TRAIN_CUSTOM_MODEL_PATH"))
    custom_model_file = str(get_env("TRAIN_TARGET_MODEL"))
    yolo_run_path = "runs/detect/"


def copy_file(source, target):

    if str(os.path.dirname(__file__)) not in source:
        source = os.path.join(os.path.dirname(__file__), source)

    if str(os.path.dirname(__file__)) not in target:
        target = os.path.join(os.path.dirname(__file__), target)

    try:
        shutil.copy(source, target)
        main_logging.info("Copy file: " + source + " to " + target)
    except Exception as e:
        main_logging.info("Error while copying file: " + str(e))


def get_latest_run_dir():

    global yolo_run_path, custom_model_path, custom_model_file

    counter = 1
    path_1 = os.path.join(os.path.dirname(__file__), yolo_run_path, custom_model_file)
    path_2 = os.path.join(os.path.dirname(__file__), yolo_run_path, custom_model_file+"2")
    path_last = path_1

    if os.path.exists(path_1) and not os.path.exists(path_2):
        return path_last

    while True:
        counter += 1
        path_next = os.path.join(yolo_run_path, custom_model_file+str(counter))
        if os.path.exists(path_next):
            path_last = path_next
        else:
            break

    return [path_last, (counter-1)]


log_level = logging.INFO
number_test_images = 20
dir_test_images = "train/validate/images"
dir_check_images = "train/check"
all_files = {}


if __name__ == "__main__":
    # initialize logging
    logging.basicConfig(level=log_level, format='%(levelname)-8s %(name)-10s | %(message)s')
    main_logging = logging.getLogger("main")

    # Load configuration
    path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(path)
    set_vars()

    [latest_run_dir, counter] = get_latest_run_dir()
    latest_run_dir = os.path.join(latest_run_dir, "weights/best.pt")
    main_logging.info("Check latest run dir: " + latest_run_dir)

    target_dir = os.path.join(custom_model_path, custom_model_file+str(counter)+".pt")
    copy_file(latest_run_dir, target_dir)

    # check image analysis with default and custom model
    main_logging.info("Check image analysis with custom model ...")

    default_model = DetectionModel()
    custom_model = DetectionModel(target_dir)
    image = ImageHandling(target_dir)

    main_logging.info(str(default_model.get_labels()))
    main_logging.info(str(custom_model.get_labels()))

    count = 0
    sel_images = []
    all_images = image.get_list(dir_test_images)
    random.shuffle(all_images)

    main_logging.info("Try with DEFAULT model ...")

    for file in all_images:
        if count < number_test_images:
            count += 1
            sel_images.append(file)
            # image = load_image(file)
            img, infos1 = default_model.analyze(file, False)
            img, infos2 = custom_model.analyze(file, False)
            img = image.load(file)
            img = image.render_detection(img, infos1, 3, default_model.threshold)
            img = image.render_detection(img, infos2, 1, custom_model.threshold)

            filename = file.split("/")
            filename = filename[len(filename)-1]
            infos1["detections"].extend(infos2["detections"])
            all_files[infos1["source_file"]] = infos1
            image.save(filename, img)

    try:
        json_file = os.path.join(dir_check_images, "00_index.json")
        with open(json_file, 'wb') as outfile:
            json.dump(all_files, codecs.getwriter('utf-8')(outfile), ensure_ascii=False, sort_keys=True, indent=4)
    except Exception as e:
        main_logging.error("Error while saving JSON file: " + str(e))

    print("")
    main_logging.info("Check image analysis with custom model done.")
    main_logging.info("Check out files in the path: " + dir_check_images)
