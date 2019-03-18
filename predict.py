#!/usr/bin/env python3

import glob
import argparse
import logging
import time

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

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='temp/predict.log',
                    filemode='w')
logger = logging.getLogger()
model_file = 'temp/checkpoints/latest.h5'
image_dir = 'tests/images'

video_exts = ['.avi', '.mp4', '.mkv', '.h264']

model = yolo()
model.load_weights(model_file)


def _predict(image):
    image = cv2.resize(image, (416, 416))
    start_time = time.time()
    boxes, labels = detect(image, model)
    end_time = time.time()
    image = draw_boxes(image, boxes, labels)
    image = cv2.resize(image, (800, 600))
    return image, boxes, end_time - start_time


def predict_image(image_path):
    image = cv2.imread(image_path)
    image, boxes, time = _predict(image)
    _log_results(boxes, time)

    while True:
        k = cv2.waitKey(30)
        if k == 27: # Escape key
            break
        cv2.imshow('Image prediction', image)
    cv2.destroyAllWindows()


def predict_multi(images, output):
    print('Founded {} images. Start handling...'.format(len(images)))
    for img_path in tqdm(images):
        logger.info('Image: %s', img_path)
        image = cv2.imread(img_path)
        image, boxes, time = _predict(image)
        _log_results(boxes, time)
        fname = path.basename(img_path)
        f = output + '/' + fname
        print('Finish handling "{}"'.format(fname))
        cv2.imwrite(f, image)


def predict_video(video_path, output):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Fail to load video "{}" file'.format(video_path))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        #k = cv2.waitKey(30)
        #if k == 27: # Escape key
        #    break
        frame, boxes, time = _predict(frame)
        logger.info('Frame id: %i', frame_id)
        _log_results(boxes, time)
        #cv2.imshow('Video prediction', frame)
        f = output + '/' + str(frame_id) + '.jpg'
        cv2.imwrite(f, frame)
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()


def _log_results(boxes, time):
    rects = []
    for box in boxes:
        rects.append({
            'coords': box.get_coords(),
            'score': box.get_score()
        })
    logger.info('Boxes: %s', rects)
    logger.info('Time: %f', time)

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
        if path.exists(o):
            rmtree(o)
        mkdir(o)
        if ext in video_exts:
            logger.debug('Predicting video: %s', f)
            return predict_video(f, o)
        logger.debug('Predicting image: %s', f)
        return predict_image(f, o)
    if path.isdir(f):
        if path.exists(o):
            rmtree(o)
        mkdir(o)
        logger.debug('Predicting multiple images from folder: %s', f)
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
