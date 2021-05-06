# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


from os import listdir
from os.path import isfile, join, exists

def preprocessing(model_dir, device_id, num_classes, src_dir, dst_dir, threshold):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    print(src_dir)

    onlyfiles = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]
    for f in onlyfiles:
        file_path = join(src_dir, f)
        image = cv2.imread(file_path)
        image_bbox = model_test.get_bbox(image)
        
        image_cropped = []
        prediction = np.zeros((1, num_classes))
        count_model = 0
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            

            image_cropped.append({
                "scale": str(scale),
                "image": img
            })

            if threshold > 0:
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                count_model = count_model + 1

        directory = dst_dir
        if threshold > 0:
            label = np.argmax(prediction)
            value = prediction[0][label]/count_model
            directory = join(dst_dir, str(label))
        
        for cropped in image_cropped:
            dst_path_image = join(directory, cropped["scale"])
            if not exists(dst_path_image):
                os.makedirs(dst_path_image)

            cv2.imwrite(join(dst_path_image, f), cropped["image"])


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="number of classes")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="image_F1.jpg",
        help="source directory")
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="image_F1.jpg",
        help="source directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Fake and Real threshold")

    args = parser.parse_args()
    preprocessing(args.model_dir, args.device_id, args.num_classes, args.src_dir, args.dst_dir, args.threshold)
