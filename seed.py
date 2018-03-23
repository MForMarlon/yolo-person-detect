#!/usr/bin/env python3

import glob
import argparse

from random import randint
from os import path, mkdir
from shutil import rmtree, copy

from tqdm import tqdm


ratio = 0.1

data_dir = './temp/dataset'
train_dir = data_dir + '/train'
eval_dir = data_dir + '/eval'

train_image_dir = train_dir + '/images'
train_label_dir = train_dir + '/labels'

eval_image_dir = eval_dir + '/images'
eval_label_dir = eval_dir + '/labels'


def make_all_dirs():
    dirs = [
        data_dir,
        train_dir,
        eval_dir,
        train_image_dir,
        train_label_dir,
        eval_image_dir,
        eval_label_dir,
    ]
    for d in dirs:
        mkdir(d)


def move(label, image):
    r = randint(0, 100)
    if r / 100 < ratio:
        copy(label, eval_label_dir)
        copy(image, eval_image_dir)
    else:
        copy(label, train_label_dir)
        copy(image, train_image_dir)


def process(images, labels):
    if len(images) > 0 and len(labels) > 0:
        if path.exists(data_dir):
            rmtree(data_dir)
        make_all_dirs()

        for label_file in tqdm(labels):
            image_file = label_file.replace(
                '/labels/',
                '/images/'
            ).replace(
                '.xml',
                '.jpg'
            )
            if path.exists(image_file):
                move(label_file, image_file)


def load(d):
    images = []
    labels = []
    if path.isdir(d):
        images = glob.glob(d + '/images/*.jpg')
        labels = glob.glob(d + '/labels/*.xml')
    print('Get {} images & {} labels'.format(len(images), len(labels)))
    return process(images, labels)


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        help='Relative path to dataset directory'
    )
    args = parser.parse_args()
    if not args.dir:
        print('Please specify path to extracted folder')
    else:
        entries = load(path.normpath(args.dir))


if __name__ == '__main__':
    start()
