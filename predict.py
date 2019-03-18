#!/usr/bin/env python3

import glob
import argparse

import numpy as np
import cv2

from os import path, mkdir
from shutil import rmtree
from random import choice

from tqdm import tqdm

from utils.network import yolo
from utils.detector import detect
from utils.drawer import draw_boxes

min_threshold = 0.5

model_file = 'temp/checkpoints/latest.h5'
image_dir = 'tests/images'

video_exts = ['.avi', '.mp4', '.mkv', '.h264']

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
        if k == 27: # Escape key
            break
        cv2.imshow('Image prediction', image)
    cv2.destroyAllWindows()


def predict_multi(images, output):
    print('Founded {} images. Start handling...'.format(len(images)))
    for img_path in tqdm(images):
        image = cv2.imread(img_path)
        image = _predict(image)
        fname = path.basename(img_path)
        f = output + '/' + fname
        print('Finish handling "{}"'.format(fname))
        cv2.imwrite(f, image)


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Fail to load video "{}" file'.format(video_path))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        k = cv2.waitKey(30)
        if k == 27: # Escape key
            break
        frame = _predict(frame)
        cv2.imshow('Video prediction', frame)
    cap.release()
    cv2.destroyAllWindows()


def check(f=None, o=None):
    if isinstance(f, int):
        return predict_video(f)

    if not f:
        images = glob.glob(image_dir + '/*.jpg')
        f = choice(images)

    if not path.exists(f):
        return print('File/folder not found: "{}"'.format(f))

    if path.isfile(f):
        ext = path.splitext(f)[1]
        if ext in video_exts:
            return predict_video(f)
        return predict_image(f)
    if path.isdir(f):
        if path.exists(o):
            rmtree(o)
        mkdir(o)
        images = glob.glob(f + '/*.jpg')
        return predict_multi(images, o)


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--file',
        help='Image file to predict'
    )
    parser.add_argument(
        '-c',
        '--cam',
        help='Camera source to predict'
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='Image dir to predict'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Image dir to export the output'
    )

    args = parser.parse_args()

    if args.cam:
        check(int(args.cam))
    elif args.dir:
        check(path.normpath(args.dir), path.normpath(args.output))
    elif args.file:
        check(path.normpath(args.file))
    else:
        check()


if __name__ == '__main__':
    start()
