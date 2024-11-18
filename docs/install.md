
# Installation
Create a conda virtual environment and activate it.
```
conda create -n os python==3.7
conda activate os
```
install PyTorch and torchvision following the [official instructions](https://pytorch.org/),we used pytorch1.7.1 cuda10.1
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
install the BboxToolkit
```
cd BboxToolkit
pip install -v -e .  # or "python setup.py develop"
```
install mmcv-full and mmdet
```
sudo pip install --upgrade urllib3==1.26.15
pip install -U openmim
mim install mmcv-full
mim install mmdet
```
install OSNet
```
cd OSNet
pip install -r requirements/build.txt
pip install mmpycocotools
pip install -v -e .  # or "python setup.py develop"
```
If the 'COCO object has no attribute get_cat_ids error' occurs:Please download[cocoAPI](https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools)
```
cd cocoapi-master/pycocotools
pip uninstall pycocotools
pip install -e .
```
