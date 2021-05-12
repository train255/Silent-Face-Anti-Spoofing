# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
from os.path import isfile, join, exists

import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(model_dir, device_id, num_classes, src_dir, dst_dir, draw_bbox):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    onlyfiles = [os.path.join(path, name) for path, subdirs, files in os.walk(src_dir) for name in files]

    for file_path in onlyfiles:
        image_name = os.path.basename(file_path)
        image = cv2.imread(file_path)
        image_bbox = model_test.get_bbox(image)
        # if you have n clasees => prediction = np.zeros((1, n))
        prediction = np.zeros((1, num_classes))
        
        test_speed = 0
        # sum the prediction from single model's result
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
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            count_model = count_model + 1
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/count_model
        if label == 1:
            label_text = "Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value)
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            label_text = "Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value)
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        
        if debug == True:
            print(label_text)
            print("Prediction cost {:.2f} s".format(test_speed))

        if draw_bbox == True:
            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

        dst_path_image = join(dst_dir, str(label))
        if not exists(dst_path_image):
            os.makedirs(dst_path_image)

        format_ = os.path.splitext(image_name)[-1]
        result_image_name = image_name.replace(format_, "_result" + format_)
        cv2.imwrite(os.path.join(dst_path_image, result_image_name), image)


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
        "--num_classes",
        type=int,
        default=3,
        help="number of classes")
    parser.add_argument(
        "--draw_bbox",
        action='store_true')
    parser.add_argument(
        "--debug",
        action='store_true')
    args = parser.parse_args()
    test(args.model_dir, args.device_id, args.num_classes, args.src_dir, args.dst_dir, args.draw_bbox)
