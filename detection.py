import os.path

import numpy
import torch
import numpy as np
import cv2
import glob
import logging
import os

# !!! to be removed, set via initial command ...
dir_check_images = "train/check"
all_files = {}


class ImageHandling:
    """
    Class to handle and modify images, supporting class for the class DetectionModel
    """

    def __init__(self):
        self.supported_image_types = ["jpg", "jpeg", "png", "bmp", "gif"]
        self.colors = np.random.uniform(0, 155, size=(100, 3))

        self.logging = logging.getLogger("image")
        self.logging.setLevel = logging.INFO

    def load(self, file_path):
        """load file from given path"""
        self.logging.debug("Load image file: " + file_path)
        return cv2.imread(file_path)

    def show(self, file_path, img):
        """show image file if GUI loaded"""
        self.logging.debug("Show image in a window: " + file_path)
        cv2.imshow(file_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, file_path, img):
        """
        save image date to file
        """
        self.logging.debug("Save image to file: " + file_path)
        if not os.path.exists(dir_check_images):
            os.makedirs(dir_check_images)
        # print(dir_check_images+"/"+filename)
        cv2.imwrite(os.path.join(dir_check_images, file_path), img)

    def get_list(self, file_path):
        """list all supported image files in a path"""
        self.logging.debug("Get a list of all images in path: " + file_path)
        image_list = []
        for image_type in self.supported_image_types:
            image_list.extend(glob.glob(os.path.join(file_path, '*.'+image_type.lower())))
            image_list.extend(glob.glob(os.path.join(file_path, '*.'+image_type.upper())))
        self.logging.info("Found " + str(len(image_list)) + " files in directory " + file_path + "...")
        return image_list

    def render_detection(self, img, detection_info, label_position=1, threshold=-1):
        """create boxes with title for each detected object"""
        self.logging.debug("Render detection into images ...")

        if 1 < threshold < 100:
            threshold = threshold / 100
        font_scale = 0.5
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        # height, width = map(float, detection_info["image_size"])
        if img is not None:
           height, width, channels = img.shape
        else:
           return None

        for detect in detection_info["detections"]:
            if threshold == -1 or detect["confidence"] >= threshold:
                color = self.colors[detect["class"]]

                box_x = int(detect["coordinates"][0] * width)
                box_y = int(detect["coordinates"][1] * height)
                box_width = int((detect["coordinates"][2]) * width)
                box_height = int((detect["coordinates"][3]) * height)
                label = detect["label"] + " " + str(round(detect["confidence"] * 100, 1))
                label_width, label_height = cv2.getTextSize(label, font_type, font_scale, font_thickness)[0]

                cv2.rectangle(img, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=1)

                text_x = box_x
                text_y = box_y
                if label_position == 3:
                    text_x = box_width - label_width - 10
                    text_y = box_height - 20

                cv2.rectangle(img, (int(text_x), int(text_y)), (int(text_x) + label_width + 10, int(text_y + 20)), color, -1)

                cv2.putText(img, label, (int(text_x + 5), int(text_y + 15)),
                            font_type, font_scale, (255, 255, 255), font_thickness)

                threshold_info = "Threshold: " + str(threshold)
                cv2.putText(img, threshold_info, (int(20), int(height -  40)), font_type, font_scale,
                            (255, 255, 255), font_thickness)
        return img


class DetectionModel:
    """
    Class to load YOLOv5 detection model and analyze images
    """

    def __init__(self, model_name="", threshold=-1):
        """
        Constructor for this class

        Parameters:
            model_name (str): model to be loaded (full path to *.pt file if custom model or yolov5 model name)
            threshold (float): detection threshold to be used, if not set or -1, the default value will be used (0.4)
        """
        self.model = None
        self.loaded = False
        self.name = model_name
        self.labels = None

        self.repro_default = 'ultralytics/yolov5'
        self.repro_default_model = 'yolov5m'
        self.default_dir = "train"
        self.default_dir_test = "train/validate"
        self.default_dir_check = "train/check"
        self.default_threshold = 0.4

        if threshold != -1:
            if 1 < threshold < 100:
                threshold = threshold / 100
            self.threshold = threshold
        else:
            self.threshold = self.default_threshold

        self.logging = logging.getLogger("detect")
        self.logging.setLevel = logging.INFO

        self.load(model_name)

    def load(self, model_name=""):
        """
        Load custom detection model, default model defined above or other yolov5\* model

        Parameters:
            model_name (str): full path to \*.pt file if custom model or yolov5\* model name
        """
        if model_name is None:
            self.logging.error("No model given to be loaded!")
            self.loaded = False

        elif ".pt" in model_name:
            if not os.path.isfile(model_name):
                self.logging.error("Custom model '" + model_name + "' not found in path.")
                self.loaded = False
            else:
                try:
                    self.logging.info("Load custom model '" + model_name + "' ...")
                    self.model = torch.hub.load(self.repro_default, 'custom', path=model_name, force_reload=True)
                    self.labels = self.get_labels()
                    self.loaded = True
                    self.logging.info("OK.")
                except Exception as e:
                    self.logging.error("Could not load default detection model '" + model_name + "': " + str(e))
                    self.loaded = False

        elif model_name == "" or "yolov5" in model_name:
            try:
                selected_model = self.repro_default_model
                if model_name != "":
                    selected_model = model_name
                self.logging.info("Load default model ...")
                self.model = torch.hub.load(self.repro_default, selected_model)
                self.labels = self.get_labels()
                self.loaded = True
                self.logging.info("OK.")
            except Exception as e:
                self.logging.error("Could not load default detection model '" + self.repro_default + "': " + str(e))
                self.loaded = False

        else:
            self.logging.error("Model name doesn't match expected format: " + str(model_name))
            self.loaded = False

    def get_labels(self):
        """
        Get a list of labels defined in the model

        Returns:
            list: list of labels in the model
        """
        if self.loaded:
            # Check if the model has an attribute containing labels
            if hasattr(self.model, 'labels'):
                labels = self.model.labels
                self.logging.debug("Labels:", labels)
                return labels
            elif hasattr(self.model, 'names'):
                labels = self.model.names
                self.logging.debug("Labels:", labels)
                return labels
            else:
                try:
                    with open(self.model.model[-1].yaml['names']) as f:
                        labels = f.read().strip().split('\n')
                    self.logging.debug("Labels:", labels)
                    return labels
                except Exception as e:
                    self.logging.warning("Labels information not found. Error:", str(e))
        else:
            self.logging.warning("No model loaded yet.")

    def analyze(self, file_path, threshold=-1, return_image=True, render_detection=False):
        """
        analyze image and return image including annotations as well as analyzed values as dict

        Parameters:
            file_path (str): path of the file to be analyzed
            threshold (float): threshold in %, if threshold=-1 use default threshold
            return_image (bool): return images
            render_detection (bool): visualize detections in the returned image
        Returns:
            numpy.ndarray, list of dict: image incl. detections if render_detection, array of detections
        """
        empty_image = None
        if not self.loaded:
            return empty_image, [{"error": "Detection model not loaded"}]

        if not os.path.exists(file_path):
            return empty_image, [{"error": "File doesn't exist: " + file_path}]

        if threshold == -1:
            threshold = self.threshold
        elif 1 < threshold < 100:
            threshold = threshold / 100

        results = self.model(file_path)
        info = str(results).split("\n")
        detect_info = {
            "source_file": file_path,
            "image_size": info[0].split(": ")[1].split(" ")[0].split("x"),
            "summary": info[0].split(": ")[1].replace(info[0].split(": ")[1].split(" ")[0] + " ", ""),
            "detections": []
        }
        if "cuda" in str(results.xyxyn[0]):
            labels, cord_thres = results.xyxyn[0][:, -1].detach().cpu().numpy(), results.xyxyn[0][:, :-1].detach().cpu().numpy()
        else:
            labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

        label_names = self.get_labels()

        i = 0
        for label in labels:
            cord_thres[i] = list(cord_thres[i])
            confidence = round(float(cord_thres[i][-1:][0]), 6)
            coordinates = list(map(float, cord_thres[i][0:-1]))
            coordinates = [round(num, 6) for num in coordinates]

            if confidence >= threshold:
                label_name = label_names[label]
                detect_info["detections"].append({
                    "class": int(label),
                    "label": label_name,
                    "coordinates": coordinates,
                    "confidence": confidence,
                    "threshold": threshold
                })
                i += 1

        self.logging.debug(detect_info)
        # print(results)
        # print(results.pandas().xyxy[6].value_counts('name'))
        # print(results.pandas().xyxy[0]["name"])
        img = None
        if return_image:
            if render_detection:
                img = np.squeeze(results.render(threshold))
                self.logging.info(str(img.shape))
                height, width, color = img.shape
                if color == 3:
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(file_path)

        return img, detect_info

    def analyze_file(self):
        self.logging.info("'analyze_file' not implemented yet")
        pass

    def analyze_image(self):
        self.logging.info("'analyze_image' not implemented yet")
        pass

