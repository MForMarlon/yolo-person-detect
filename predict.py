#!/usr/bin/env python3

import glob
import argparse

import numpy as np
import cv2

from os import path
from random import choice

from utils.network import yolo
from utils.detector import detect
from utils.drawer import draw_boxes

min_threshold = 0.5

model_file = 'temp/pretrained.h5'
image_dir = 'tests/images'

video_exts = ['.avi', '.mp4']

model = yolo()
model.load_weights(model_file)


def _predict(image):
    image = cv2.resize(image, (416, 416))
    boxes, labels = detect(image, model)
    image = draw_boxes(image, boxes, labels)
    image = cv2.resize(image, (800, 600))
    return image


def predict_image(image_path):
    image = cv2.imread(image_path)
    image = _predict(image)

    while True:
        k = cv2.waitKey(30)
        if k == 27:
            break
        cv2.imshow('Image prediction', image)
    cv2.destroyAllWindows()


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Fail to load video "{}" file'.format(video_path))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        k = cv2.waitKey(30)
        if k == 27:
            break
        frame = _predict(frame)
        cv2.imshow('Video prediction', frame)
    cap.release()
    cv2.destroyAllWindows()


def check(f=None):
    if f:
        if not path.exists(f):
            print('File not found: "{}"'.format(f))
        else:
            ext = path.splitext(f)[1]
            if ext in video_exts:
                predict_video(f)
            else:
                predict_image(f)
    else:
        images = glob.glob(image_dir + '/*.jpg')
        f = choice(images)
        predict_image(f)


# start
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    help='Image file to predict'
)
args = parser.parse_args()
if not args.file:
    check()
else:
    check(path.normpath(args.file))
