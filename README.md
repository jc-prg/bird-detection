# Create YOLO detection model (v5, v8, v11)

This is an easy way to create a self-trained detection model.

## Install preconditions

```commandline
sudo apt-get install python3-opencv
pip install ultralytics

git clone https://github.com/tzutalin/labelImg
```

If you want to use the former version with YOLOv5 install it:

```commandline
pip install torch torchvision torchaudio

git clone https://github.com/ultralytics/yolov5
cd yolov5 & pip install -r requirements.txt
```

## Check if GPU tools installed

To use your GPU to train the model, CUPA has to be installed:

```commandline
python3 00-gpu-test.py
```

Install CUPA if not done yet:

```commandline
sudo apt install cuda
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
nvcc -V
nvidia-smi
```

## Create configuration

* _Create YOLO configuration_
  - copy [sample.env](sample.env) to .env
  - edit the .env-file and change the default configuration to your needs


## Prepare images

1. _Collect images in sub-folder "train-preparation"_
   - to collect you might use the image search of Google, DuckDuckGo or similar
   - download all images using the browser plug-in "Download All Images"
   - remove those images that doesn't fit your requirements and are too small
   

2. Create or adapt classes

   * for labelImg in `./labelImg/train-preparation/labels/classes.txt`

      ```commandline
      list
      of
      your
      names
      ```

3. _Label the objects using labelImg_
   - change values for default class definition if required: [labelImg/data/predefined_classes.txt](labelImg/data/predefined_classes.txt)
   - select sub-folder [train-preparation/images](train-preparation/images) via "Open Dir"
   - select sub-folder [train-preparation/labels](train-preparation/labels) via "Change Save Dir"

   ```commandline
   cd labelImg
   python3 ./labelImg.py
   ```

4. Create trainings data

   * Copy files to `train` folder and split into training and validation data.
      ```commandline
      python3 01-prepare-images.py
      ```
     
   * This command will (re)create data in two folders:
     * [train-check](train-check) contains data sorted by label. If something isn't as expected go back to 3. and 
       adapt your source data
     * [train](train) contains the data for the training.

## Train model

1. Check the settings in the script [02-train-images.py](./02-train-images.py), if required modify, and start:

    ```commandline
    ./02-train-images.py
    ```

2. The training will create a directory 'runs/detect' with numbered subdirectories per run. In the
   lastest directory you'll find the trained model in the subdirectory 'weights'. Copy the file 'best.pt' to
   the directory [custom_models](custom_models)

## Validate results

1. Adapt the settings in the file [03-check-results.py](./03-check-results.py) to the name of your freshly trained model.

2. Try detection with created model based on some test files.

    ```commandline
    ./03-check-results.py
    ```

3. The results will be stored in the directory [train/check](train/check).
