#     Multi_Resolution_Rescored_ByteTrack

The code in this repository is based on the code from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [YOLOV](https://github.com/YuHengsss/YOLOV/tree/master)
In order to reproduce our results with MR2ByteTrack on the YOLOXS network you will need to:

- Install the dependencies you can:
    - create a new environment using conda with the provided .yaml file '''conda -conda env create -f Multiresolution_ByteTrack.yml'''
    - install the dependencies via pip with ''' pip install -r requirements.txt'''
- Download the [weights](https://drive.google.com/file/d/1n8wkByqpHdrGy6z9fsoZpBtTa0I3JOcG/view?usp=sharing) of the YOLOXs network
- Download the ILSVRC2015 VID dataset from [IMAGENET](https://image-net.org/challenges/LSVRC/2015/2015-downloads)
    - unzip the datase and keep the structure of the folder unchanged
- Modify the file in the Experiments folder so that it cointains the directories for the 
