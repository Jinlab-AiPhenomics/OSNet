import glob
import os
from tqdm import tqdm

img_list = glob.glob("./work_dirs/*.png")
for i in tqdm(img_list):
    os.remove(i)