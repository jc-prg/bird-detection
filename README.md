# Create YOLOv5/PyTorch detection model

This is an easy way to create a self-trained detection model.

## Source and HowTo

Thanks for this great tutorial:

* https://github.com/nicknochnack/YOLO-Drowsiness-Detection/blob/main/Drowsiness%20Detection%20Tutorial.ipynb
* https://www.youtube.com/watch?v=tFNJGim3FXw

## Install preconditions

```commandline
pip install torch torchvision torchaudio

git clone https://github.com/ultralytics/yolov5
cd yolov5 & pip install -r requirements.txt

cd ..
git clone https://github.com/tzutalin/labelImg
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

## Prepare images

1. _Collect images in sub-folder "train-preparation"_
   - to collect you might use the image search of Google, DuckDuckGo or similar
   - download all images using the browser plug-in "Download All Images"
   - remove those images that doesn't fit your requirements and are too small
   

2. _Label the objects using labelImg_
   - change values for default class definition if required: [labelImg/data/predefined_classes.txt](labelImg/data/predefined_classes.txt)
   - select sub-folder [train-preparation/images](train-preparation/images) via "Open Dir"
   - select sub-folder [train-preparation/labels](train-preparation/labels) via "Change Save Dir"
```commandline
cd labelImg
python3 ./labelImg.py
```


3. Create or adapt yolov5/dataset.yml

```yaml
# train data sets
path: ../train
train: images
val: validate

# classes
nc: 10 # number of names
names: ["list", "of", "your", "names"] # have to be the same such as in train-preparation/labels/classes.txt
```


4. Create trainings data

Copy files to train folder and split into training and validation data.
```commandline
python3 01-prepare-images.py
```

## Train model

Check, if required modify, and start the following script:

```commandline
./02-train-images
```

## Validate results

Try detection with created model based on some test files.

```commandline
python3 03-check-results
```

The results will be stored in the directory [train/check](train/check).
