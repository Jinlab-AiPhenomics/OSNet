
# Installation
1. Create a conda virtual environment and activate it.

   ```PYTHON
   conda create -n os python==3.7
   conda activate os
   ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), we used pytorch1.7.1 and cuda10.1.

   ```PYTHON
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
   ```

3. Install the BboxToolkit.

   ```PYTHON
   cd BboxToolkit
   pip install -v -e .  # or "python setup.py develop"
   ```

4. Install mmcv-full and mmdet.

   ```PYTHON
   sudo pip install --upgrade urllib3==1.26.15
   pip install -U openmim
   mim install mmcv-full
   mim install mmdet
   ```

5. Install OSNet.

   ```PYTHON
   cd OSNet
   pip install -r requirements/build.txt
   pip install mmpycocotools
   pip install -v -e .  # or "python setup.py develop"
   ```

6. If the 'COCO object has no attribute get_cat_ids error' occurs:Please download [cocoAPI](https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools).

   ```PYTHON
   cd cocoapi-master/pycocotools
   pip uninstall pycocotools
   pip install -e .
   ```

   
