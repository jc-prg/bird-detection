#!/bin/python3


import os.path
import random
import json
import codecs
import logging


from detection import ImageHandling, DetectionModel

log_level = logging.INFO
number_test_images = 20
custom_model_path = "custom_models/birdhouse_birds_v03.pt"
dir_test_images = "train/validate"
dir_check_images = "train/check"
all_files = {}


if __name__ == "__main__":

    # initialize logging
    logging.basicConfig(level=log_level, format='%(levelname)-8s %(name)-10s | %(message)s')
    main_logging = logging.getLogger("main")

    # check image analysis with default and custom model
    main_logging.info("Check image analysis with custom model ...")

    default_model = DetectionModel()
    custom_model = DetectionModel(custom_model_path)
    image = ImageHandling()

    main_logging.info(str(custom_model.get_labels()))

    count = 0
    sel_images = []
    all_images = image.get_list(default_model.default_dir_test)
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
        logging.error("Error while saving JSON file: " + str(e))
