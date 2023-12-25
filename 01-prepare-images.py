#!/bin/python3

import os
import glob
import shutil
import random
import logging

directories ={
    "train": "train",
    "source": "train-preparation",
    "labels": "labels",
    "images": "images",
    "validate": "validate",
    "yolo": "yolov5"
}


def prepare_labels():
    source = os.path.join(directories["source"], directories["labels"])
    goal = os.path.join(directories["train"], directories["labels"])

    files = glob.glob(os.path.join(source, '*.txt'))

    if not os.path.exists(os.path.join(directories["train"])):
        os.makedirs(os.path.join(directories["train"]))
    if not os.path.exists(goal):
        os.makedirs(goal)

    for file in files:
        shutil.copy(file, os.path.join(goal, os.path.basename(file)))

    print("copied "+str(len(files))+" label files.")


def prepare_images(test_ratio=0.2):
    source = os.path.join(directories["source"], directories["images"])
    goal = os.path.join(directories["train"], directories["images"])
    validate = os.path.join(directories["train"], directories["validate"])
    labels = os.path.join(directories["train"], directories["labels"])

    if not os.path.exists(os.path.join(directories["train"])):
        os.makedirs(os.path.join(directories["train"]))
    if not os.path.exists(goal):
        os.makedirs(goal)
    if not os.path.exists(validate):
        os.makedirs(validate)

    file_count = 0
    img_dirs = glob.glob(os.path.join(source, '*'))
    for img_dir in img_dirs:
        files = glob.glob(os.path.join(img_dir, '*.jpeg'))
        random.shuffle(files)
        num_test_samples = int(len(files) * test_ratio)
        print(img_dir)

        for i, file in enumerate(files):
            if i < num_test_samples:
                output_folder = validate
                label_file = file.split("/")
                label_file = label_file[len(label_file)-1]
                label_file = label_file.replace(".jpeg", ".txt")
                # print(label_file)
                if os.path.exists(os.path.join(labels, label_file)):
                    shutil.copy(os.path.join(labels, label_file), os.path.join(validate, label_file))
            else:
                output_folder = goal

            shutil.copy(file, os.path.join(output_folder, os.path.basename(file)))
            file_count += 1
            print(file)
            print(output_folder)

    print("copied "+str(file_count)+" image files.")


def train_with_images():
    pass


def validate_with_images(count):
    pass


prepare_labels()
prepare_images()
