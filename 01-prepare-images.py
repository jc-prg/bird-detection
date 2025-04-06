#!/bin/python3

import os
import glob
import shutil
import random
import logging


directories = {
    "train-check": "train-check",
    "train": "train",
    "source": "train-preparation",
    "labels": "labels",
    "images": "images",
    "images_v8": "train",
    "validate": "validate",
    "yolo": "yolov5"
}
dataset_yml = "02-train-dataset.yml"
version = "v8"  # v8 or v5 or 11


def read(filename):
    """
    read json file including check if locked
    """
    try:
        with open(filename) as text_file:
            data = text_file.read()
        text_file.close()
        return str(data)

    except Exception as e:
        print("Could not read TEXT file: " + filename + " - " + str(e))
        return ""


def write(filename, data):
    """
    write json file including locking mechanism
    """
    try:
        with open(filename, 'w') as text_file:
            text_file.write(data)
            text_file.close()
        logging.debug("Write TEXT file: " + filename)

    except Exception as e:
        print("Could not write TEXT file: " + filename + " - " + str(e))


def read_file_data():
    """
    read all file data into a dict and return
    :return dict:
    """
    file_data = {}
    dir_source = os.path.join(directories["source"], directories["images"])
    dir_labels = os.path.join(directories["source"], directories["labels"])
    path_classes = os.path.join(dir_labels, "classes.txt")
    if not os.path.exists(path_classes):
        print("ERROR: could not find " + str(path_classes))
        return
    classes = read(path_classes).split("\n")

    path_images = glob.glob(os.path.join(dir_source, '*'))
    for img_dir in path_images:
        files = []
        files_jpeg = glob.glob(os.path.join(img_dir, '*.jpeg'))
        files_jpg = glob.glob(os.path.join(img_dir, '*.jpg'))
        files_png = glob.glob(os.path.join(img_dir, '*.png'))
        files = files + files_jpeg + files_jpg + files_png
        for i, file in enumerate(files):
            file_img = file
            if "jpeg" in file:
                file_label = str(file).replace(".jpeg", ".txt").split("/")[-1]
            elif "jpg" in file:
                file_label = str(file).replace(".jpg", ".txt").split("/")[-1]
            elif "png" in file:
                file_label = str(file).replace(".png", ".txt").split("/")[-1]
            file_label_path = os.path.join(str(dir_labels), str(file_label))
            # print(file_img + " / " + file_label)

            key = file_label.replace(".txt","")
            file_data[key] = {
                "file-image": file_img,
                "file-label": file_label_path,
                "detections": [],
                "class-names": [],
                "classes": []
            }
            yolo_data = read(file_label_path).split("\n")
            for detection in yolo_data:
                if len(detection) > 4:
                    data_set = detection.split(" ")
                    class_name = classes[int(data_set[0])]
                    file_data[key]["detections"].append(data_set)
                    file_data[key]["class-names"].append(class_name)
                    if class_name not in file_data[key]["classes"]:
                        file_data[key]["classes"].append(class_name)

            file_data[key]["classes"] = len(file_data[key]["classes"])

            #print(str(file_data[key]).replace("txt',", "txt',\n") + "\n")

    file_count = len(file_data.keys())
    print("------------")
    bird_data = {"-multiple-": []}
    multiple_birds = []
    count = 0
    for i, class_name in enumerate(classes):
        if class_name == "":
            continue
        bird_data[class_name] = []
        for file in file_data:
            entry = file_data[file]
            if int(entry["classes"]) == 1 and entry["class-names"][0] == class_name:
                bird_data[class_name].append(file)
                count += 1
            elif (len(entry["class-names"]) > 0 and entry["class-names"][0] == class_name
                  and file not in bird_data["-multiple-"]):
                bird_data["-multiple-"].append(file)

        print(class_name + ": " + str(len(bird_data[class_name])))

    print("multiple: " + str(len(bird_data["-multiple-"])))
    print("total: " + str(file_count))
    print("not counted: " + str(file_count - count - len(multiple_birds)))
    print("------------")
    #print(str(bird_data["bird"][0]))

    data = {
        "object_list": bird_data,
        "file_data": file_data
        }

    return data


def prepare_images_check(data, test_ratio=0.2):
    """
    compy files to destination folder based on object names
    :param data:- dictionary from 'read_file_data()'
    :param test_ratio:
    :return:
    """
    dirs = {
        "destination": os.path.join(directories["train-check"], directories["images"]),
        "validate": os.path.join(directories["train-check"], directories["validate"]),
        "labels": os.path.join(directories["train-check"], directories["labels"])
        #"labels": directories["train"]
    }
    if os.path.exists(directories["train-check"]):
        shutil.rmtree(directories["train-check"])
    os.makedirs(directories["train-check"])

    for key in dirs:
        if not os.path.exists(dirs[key]):
            os.makedirs(dirs[key])

    path_classes_source = os.path.join(directories["source"], directories["labels"], "classes.txt")
    path_classes_destination = os.path.join(directories["train-check"], directories["labels"], "classes.txt")
    data_classes = read(path_classes_source)
    shutil.copy(path_classes_source, path_classes_destination)

    for object_key in data["object_list"]:
        print("Copy files for " + object_key + " ...")
        dir_destination = os.path.join(dirs["destination"], object_key)
        os.makedirs(dir_destination)
        dir_validate = os.path.join(dirs["validate"], object_key)
        os.makedirs(dir_validate)

        image_files = data["object_list"][object_key]
        random.shuffle(image_files)

        for i, filename in enumerate(image_files):
            # print(filename)
            entry = data["file_data"][filename]
            source_file = entry["file-image"]
            destination_file = os.path.join(dir_destination, entry["file-image"].split("/")[-1])
            validate_file = os.path.join(dir_validate, entry["file-image"].split("/")[-1])

            source_file_label = entry["file-label"]
            #destination_file_label = os.path.join(directories["train"], entry["file-label"].split("/")[-1])
            destination_file_label = os.path.join(dirs["labels"], entry["file-label"].split("/")[-1])
            shutil.copy(source_file_label, destination_file_label)

            # print(" - " + source_file)
            if i < test_ratio*len(image_files):
                shutil.copy(source_file, validate_file)
            else:
                shutil.copy(source_file, destination_file)


def prepare_images_v2(data, test_ratio=0.2):
    """
    compy files to destination folder based on object names
    :param data:- dictionary from 'read_file_data()'
    :param test_ratio:
    :return:
    """
    global version
    dirs = {
        "destination": os.path.join(directories["train"], directories["images"]),
        "destination_v8_img": os.path.join(directories["train"], directories["images"], "images"),
        "destination_v8_lab": os.path.join(directories["train"], directories["images"], "labels"),
        "validate": os.path.join(directories["train"], directories["validate"]),
        "validate_v8_img": os.path.join(directories["train"], directories["validate"], "images"),
        "validate_v8_lab": os.path.join(directories["train"], directories["validate"], "labels"),
        "labels": os.path.join(directories["train"], directories["labels"])
        #"labels": directories["train"]
    }
    if os.path.exists(directories["train"]):
        shutil.rmtree(directories["train"])
    os.makedirs(directories["train"])

    for key in dirs:
        if not os.path.exists(dirs[key]):
            if version == "v5" or (version == "v5" and key != "labels"):
                os.makedirs(dirs[key])

    path_classes_source = os.path.join(directories["source"], directories["labels"], "classes.txt")
    path_classes_destination = os.path.join(directories["train-check"], directories["labels"], "classes.txt")
    data_classes = read(path_classes_source)
    shutil.copy(path_classes_source, path_classes_destination)

    for object_key in data["object_list"]:
        print("Copy files for " + object_key + " ...")
        dir_destination = dirs["destination"]

        if version == "v5":
            # dir_destination = os.path.join(dirs["destination"], object_key)
            if not os.path.exists(dir_destination):
                os.makedirs(dir_destination)

            # dir_validate = os.path.join(dirs["validate"], object_key)
            dir_validate = dirs["validate"]
            if not os.path.exists(dir_validate):
                os.makedirs(dir_validate)

        elif version == "v8":
            dir_destination_v8_img = dirs["destination_v8_img"]
            dir_destination_v8_lab = dirs["destination_v8_lab"]
            if not os.path.exists(dir_destination_v8_img):
                os.makedirs(dir_destination_v8_img)
            if not os.path.exists(dir_destination_v8_lab):
                os.makedirs(dir_destination_v8_lab)

            dir_validate_v8_img = dirs["validate_v8_img"]
            dir_validate_v8_lab = dirs["validate_v8_lab"]
            if not os.path.exists(dir_validate_v8_img):
                os.makedirs(dir_validate_v8_img)
            if not os.path.exists(dir_validate_v8_lab):
                os.makedirs(dir_validate_v8_lab)

        image_files = data["object_list"][object_key]
        random.shuffle(image_files)

        for i, filename in enumerate(image_files):
            # print(filename)
            entry = data["file_data"][filename]
            source_file = entry["file-image"]
            if version == "v5":
                destination_file = os.path.join(dir_destination, entry["file-image"].split("/")[-1])
                validate_file = os.path.join(dir_validate, entry["file-image"].split("/")[-1])

                source_file_label = entry["file-label"]
                #destination_file_label = os.path.join(directories["train"], entry["file-label"].split("/")[-1])
                destination_file_label = os.path.join(dirs["labels"], entry["file-label"].split("/")[-1])
                shutil.copy(source_file_label, destination_file_label)

            elif version == "v8":
                destination_file = os.path.join(dir_destination_v8_img, entry["file-image"].split("/")[-1])
                validate_file = os.path.join(dir_validate_v8_img, entry["file-image"].split("/")[-1])

                source_file_label = entry["file-label"]
                destination_file_label1 = os.path.join(dir_destination_v8_lab, entry["file-label"].split("/")[-1])
                destination_file_label2 = os.path.join(dir_validate_v8_lab, entry["file-label"].split("/")[-1])
                shutil.copy(source_file_label, destination_file_label1)
                shutil.copy(source_file_label, destination_file_label2)

            if i < test_ratio * len(image_files):
                shutil.copy(source_file, validate_file)
            else:
                shutil.copy(source_file, destination_file)

    current_working_directory = os.getcwd()

    yml_data = "path: " + current_working_directory + "/train\n"
    yml_data += "train: " + directories["images"] + "\n"
    yml_data += "val: " + directories["validate"] + "\n"
    yml_data += "test:\n\n"

    count = 0
    for i, class_name in enumerate(data_classes.split("\n")):
        if class_name != "":
            count += 1

    yml_data += "nc: " + str(count) + "\n"
    yml_data += "names: \n"

    for i, class_name in enumerate(data_classes.split("\n")):
        if class_name != "":
            if version == "v5":
                yml_data += "  " + str(i) + ": " + class_name + "\n"
            elif version == "v8":
                yml_data += "  - " + class_name + "\n"
    yml_data += "\n"

    write(dataset_yml, yml_data)


def prepare_labels():
    """
    copy label files to destination directory defined in global var directories["labels"]
    :return:
    """
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
    """
    split images into source and validation data and copy to destinations defined in global vars directories

    :param test_ratio: amount of test files, default = 20%
    :return:
    """
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


analyze_data = read_file_data()
prepare_images_check(analyze_data)
prepare_images_v2(analyze_data)

#prepare_labels()
#prepare_images()
