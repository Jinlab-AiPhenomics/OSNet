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
All config files of  datasets are put at `/configs/_base_/dataset`. Before training and testing, you need to change the `data_root` and other information in the config file.
```
cd \configs\_base_\datasets\isaid_new.py
```
