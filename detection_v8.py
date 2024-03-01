import os.path

import numpy
import numpy as np
import cv2
import glob
import logging
import os
import ultralytics

# ------------------------------------------
# https://medium.com/mlearning-ai/exploring-using-yolov8n-with-pytorch-mobile-android-479b0b866a3d
# https://docs.ultralytics.com/de/usage/python/#benchmark
# ------------------------------------------

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

    def render_detection(self, img, detection_info, label_position=1, threshold=-1, test=False):
        """
        create boxes with title for each detected object

        Args:
             img (numpy.ndarray): image data
             detection_info (dict): detection infos
             label_position (int): position of label (values: 1..4)
             threshold (float): threshold, don't visualize detections below
        Returns:
             numpy.ndarray: image with rendered detections
        """
        self.logging.debug("Render detection into image ...")

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

                position_info = str(box_x) + ", " + str(box_y) + ", " + str(box_width) + ", " + str(box_height)
                position_info += " (" + str(width) + "x" + str(height) + ")"

                cv2.rectangle(img, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=1)

                text_x = box_x
                text_y = box_y
                position_info_y = 40
                if label_position == 3:
                    position_info_y = 60
                    text_x = box_width - label_width - 10
                    text_y = box_height - 20

                cv2.rectangle(img, (int(text_x), int(text_y)), (int(text_x) + label_width + 10, int(text_y + 20)), color, -1)

                cv2.putText(img, label, (int(text_x + 5), int(text_y + 15)),
                            font_type, font_scale, (255, 255, 255), font_thickness)

                threshold_info = "Threshold: " + str(threshold)
                cv2.putText(img, threshold_info, (int(20), int(height - 40)), font_type, font_scale,
                            (255, 255, 255), font_thickness)

                if test:
                    cv2.putText(img, position_info, (int(20), int(position_info_y)), font_type, font_scale,
                                (255, 255, 255), font_thickness)
        return img


class DetectionModel:
    """
    Class to load YOLOv5 detection model and analyze images
    """

    def __init__(self, model_name="", threshold=-1):
        """
        Constructor for this class

        Args:
            model_name (str): model to be loaded (full path to *.pt file if custom model or yolov5 model name)
            threshold (float): detection threshold to be used, if not set or -1, the default value will be used (0.4)
        """
        self.model = None
        self.loaded = False
        self.name = model_name
        self.labels = None
        self.image = ImageHandling()

        self.default_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        self.default_model = 'yolov8m'
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

    def test_yolo(self, model_name="", image=""):
        print("Load model " + model_name)
        model = ultralytics.YOLO(model_name)
        print("Loaded: " + str(model.names))
        print("Read image " + image)
        image = cv2.imread(image)
        print("Predict ...")
        results = model.predict(source=image)
        print(str(results))
        print("-> " + str(len(results)) + " results")
        print("-> " + str(len(results[0].boxes)) + " boxes in results[0]")
        print("---\n" + str(results[0].boxes[0]))
        print("---\n")
        print("-> names:             " + str(results[0].names))
        print("-> type:              " + str(int(results[0].boxes[0].cls[0].item())))
        print("-> class name:        " + str(results[0].names[results[0].boxes[0].cls[0].item()]))
        print("-> coordinates xyxy:  " + str(results[0].boxes[0].xyxy[0].tolist()))
        print("-> coordinates xywhn: " + str(results[0].boxes[0].xywhn[0].tolist()))
        print("-> confidence:        " + str(results[0].boxes[0].conf[0].item()))

    def load(self, model_name=""):
        """
        Load custom detection model, default model defined above or other yolov5\* model

        Args:
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
                    self.model = ultralytics.YOLO(model_name, task="detect")
                    self.loaded = True
                    self.labels = self.get_labels()
                    self.logging.info("OK.")
                except Exception as e:
                    self.logging.error("Could not load default detection model '" + model_name + "': " + str(e))
                    self.loaded = False

        elif model_name == "" or "yolov8" in model_name:
            if model_name == "yolov8" or model_name == "":
                selected_model = self.default_model + ".pt"
            else:
                selected_model = model_name + ".pt"
            try:
                self.logging.info("Load default model '"+selected_model+"' ...")
                self.model = ultralytics.YOLO(selected_model, task="detect")
                self.loaded = True
                self.labels = self.get_labels()
                self.logging.info("OK.")
            except Exception as e:
                self.logging.error("Could not load default detection model '" + selected_model + "': " + str(e))
                self.loaded = False

        else:
            self.logging.error("Model name doesn't match expected format: '" + str(model_name) + "'")
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
                self.logging.warning("Labels information not found.")
        else:
            self.logging.warning("No model loaded yet.")

    def analyze(self, file_path, threshold=-1, return_image=True, render_detection=False):
        """
        analyze image and return image including annotations as well as analyzed values as dict

        Args:
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

        image = cv2.imread(file_path)
        #results = self.model.predict(source=image)
        results = self.model.predict(source=file_path)

        detect_summary = {}
        detect_info = {
            "source_file": file_path,
            "image_size": [image.shape[0], image.shape[1]],
            "summary": "",
            "detections": []
        }
        for box in results[0].boxes:
            detection = {
                "class": int(box.cls[0].item()),
                "label": self.labels[box.cls[0].item()],
                "coordinates": box.xyxyn[0].tolist(),
                "confidence": box.conf[0].item(),
                "threshold": threshold
            }

            if detection["confidence"] >= threshold:
                detect_info["detections"].append(detection)
                label = self.labels[box.cls[0].item()]
                if label in detect_summary:
                    detect_summary[label] += 1
                else:
                    detect_summary[label] = 1

        if len(detect_summary) == 0:
            detect_info["summary"] = "empty"
        for label in detect_summary:
            detect_info["summary"] += str(detect_summary[label]) + " " + label + "  "

        img = None
        if return_image:
            if render_detection:
                img = self.image.render_detection(image, detect_info, label_position=1, threshold=threshold)
            else:
                img = image
        return img, detect_info

    def analyze_v5(self, file_path, threshold=-1, return_image=True, render_detection=False):
        """
        analyze image and return image including annotations as well as analyzed values as dict

        Args:
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

