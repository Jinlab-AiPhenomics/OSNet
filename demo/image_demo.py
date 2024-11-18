import copy
from argparse import ArgumentParser

import numpy as np
import mmcv
from PIL import Image

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--out-file', default='/root/autodl-tmp/OBBDetection-master/demo/oriented/')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    #test Multi-image
    #path = '/root/autodl-tmp/OBBDetection-master/demo/images_angle/'
    path = '/root/autodl-tmp/OBBDetection-master_2/data/2/images/'
    #path = '/root/autodl-tmp/OBBDetection-master_2/demo/20210310_36_15.png'
    for file in os.listdir(path):
        args.img=path+file
        result = inference_detector(model, args.img)
    # test a single image

        # show the results
        show_result_pyplot(model, args.img, result, score_thr=args.score_thr,out_file='/root/autodl-tmp/OBBDetection-master_2/data/2/2/' + file)
        print(file + " :Finished!")


if __name__ == '__main__':
    main()
