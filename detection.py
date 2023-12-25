import os.path
import torch
import numpy as np
import cv2
import glob
import logging

# !!! to be removed, set via initial command ...
dir_check_images = "train/check"
all_files = {}


class ImageHandling:

    def __init__(self):
        self.supported_image_types = ["jpg", "jpeg", "png", "bmp", "gif"]
        self.colors = np.random.uniform(0, 155, size=(100, 3))

        self.logging = logging.getLogger("image")
        self.logging.setLevel = logging.INFO

    def load(self, file_path):
        """load file from given path"""
        self.logging.debug("Load image file: " +file_path)
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

    def render_detection(self, img, detection_info, label_position=1, threshold=0):
        """create boxes with title for each detected object"""
        font_scale = 0.5
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1

        self.logging.debug("Render detection into images ...")
        height, width = map(float, detection_info["image_size"])
        for detect in detection_info["detections"]:
            if threshold == 0 or detect["confidence"] > threshold:
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
        return img


class DetectionModel:

    def __init__(self, model_name=""):
        self.model = None

        self.repro_default = 'ultralytics/yolov5'
        self.repro_default_model = 'yolov5m'
        self.default_dir = "train"
        self.default_dir_test = "train/validate"
        self.default_dir_check = "train/check"
        self.default_threshold = 0.4

        self.threshold = self.default_threshold

        self.logging = logging.getLogger("detect")
        self.logging.setLevel = logging.INFO

        self.load(model_name)

    def load(self, model_name=""):
        """Load custom detection model or default model defined above"""
        if model_name == "":
            self.logging.info("Load default model: ")
            self.model = torch.hub.load(self.repro_default, self.repro_default_model)
        else:
            self.logging.info("Load custom model '" + model_name + "':")
            self.model = torch.hub.load(self.repro_default, 'custom', path=model_name, force_reload=True)

    def analyze(self, file_path, threshold=-1, return_image=True):
        """
        analyze image and return image including annotations as well as analyzed values as dict
        """
        if threshold == -1:
            threshold = self.default_threshold

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
        i = 0
        for label in labels:
            cord_thres[i] = list(cord_thres[i])
            confidence = 0
            if len(cord_thres[i]) > 0:
                confidence = cord_thres[i][4]
            detect_info["detections"].append({
                "class": int(label),
                "label": str(results.pandas().xyxy[0]["name"]).split("\n")[i].split("    ")[1],
                "coordinates": list(map(float, cord_thres[i][0:-1])),
                "confidence": float(cord_thres[i][-1:][0])
            })
            i += 1

        self.logging.debug(detect_info)
        # print(results)
        # print(results.pandas().xyxy[6].value_counts('name'))
        # print(results.pandas().xyxy[0]["name"])
        img = None
        if return_image:
            img = np.squeeze(results.render(threshold))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

        return img, detect_info

