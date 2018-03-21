#!/usr/bin/env python3

from os import path, mkdir

dirs = [
    'temp',
    'temp/dataset',
    'temp/dataset/train',
    'temp/dataset/train/images',
    'temp/dataset/train/labels',
    'temp/dataset/eval',
    'temp/dataset/eval/images',
    'temp/dataset/eval/labels',
    'temp/weights',
    'temp/logs'
]


def make(ds):
    for d in ds:
        if not path.exists(d):
            mkdir(d)


if __name__ == '__main__':
    make(dirs)
