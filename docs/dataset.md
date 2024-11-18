## Dataset Preparation
Please prepare the dataset in the COCO format
```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```
All config files of  datasets are put at /configs/obb/_base_/dataset. Before training and testing, you need to add the dataset path to config files.
```
cd \configs\_base_\datasets\isaid_new.py
```
