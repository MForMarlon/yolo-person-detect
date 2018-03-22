#!/usr/bin/env python3

import time
from os import listdir, path, rename

from keras import models
from keras import layers
from keras import callbacks
from keras import optimizers

import keras.backend as K
import tensorflow as tf
import numpy as np

from utils.preprocessing import parse_annotation, BatchGenerator, WeightReader
from utils.drawer import draw_boxes
from utils.helpers import decode_netout
from utils.custom_loss import custom_loss


from utils.network import yolo, true_boxes, LABELS, \
    IMAGE_H, IMAGE_W, GRID_H, GRID_W, \
    BOX, CLASS, CLASS_WEIGHTS, OBJ_THRESHOLD, NMS_THRESHOLD, \
    ANCHORS, NO_OBJECT_SCALE, OBJECT_SCALE, COORD_SCALE, CLASS_SCALE, \
    BATCH_SIZE, WARM_UP_BATCHES, TRUE_BOX_BUFFER, ALPHA


dataset_folder = 'temp/dataset/'
train_image_folder = dataset_folder + 'train/images/'
train_annot_folder = dataset_folder + 'train/labels/'
val_image_folder = dataset_folder + 'eval/images/'
val_annot_folder = dataset_folder + 'eval/labels/'

checkpoint_folder = 'temp/checkpoints/'
pre_trained_weights = 'temp/weights/yolo.weights'
pre_trained_model = checkpoint_folder + 'latest.h5'
temp_training_model = checkpoint_folder + 'tmp.h5'

tensorboard_log = 'temp/logs/'


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def get_pretrained_model():
    if path.exists(pre_trained_model):
        return pre_trained_model
    f = checkpoint_folder + 'initial-model.h5'
    if path.exists(f):
        return f
    return False


def save_checkpoint(logs, tmp_model, prev_model):
    if path.exists(tmp_model):
        if prev_model.endswith('latest.h5'):
            fname = time.strftime('%d%m%y%H%M%S') + '.h5'
            rename(prev_model, checkpoint_folder + fname)
        latest_model = checkpoint_folder + 'latest.h5'
        rename(tmp_model, latest_model)
        print('Trained model has been saved at "{}"'.format(latest_model))


def start():
    model = yolo()
    model.summary()

    num_of_epochs = 100
    nb_conv = 23
    weight_reader = WeightReader(pre_trained_weights)
    weight_reader.reset()

    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'BOX': BOX,
        'LABELS': LABELS,
        'CLASS': CLASS,
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
    }

    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(
                np.prod(conv_layer.get_weights()[1].shape)
            )
            kernel = weight_reader.read_bytes(
                np.prod(conv_layer.get_weights()[0].shape)
            )
            kernel = kernel.reshape(
                list(reversed(conv_layer.get_weights()[0].shape))
            )
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])

        else:
            kernel = weight_reader.read_bytes(
                np.prod(conv_layer.get_weights()[0].shape)
            )
            kernel = kernel.reshape(
                list(reversed(conv_layer.get_weights()[0].shape))
            )
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    # Get last convolutional layer
    layer = model.layers[-4]
    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H*GRID_W)
    new_bias = np.random.normal(size=weights[1].shape) / (GRID_H*GRID_W)

    layer.set_weights([new_kernel, new_bias])

    train_imgs, seen_train_labels = parse_annotation(
        train_annot_folder,
        train_image_folder,
        labels=LABELS
    )

    train_batch = BatchGenerator(
        train_imgs,
        generator_config
    )

    val_imgs, seen_val_labels = parse_annotation(
        val_annot_folder,
        val_image_folder,
        labels=LABELS
    )

    prev_model = get_pretrained_model()
    model.load_weights(prev_model)

    valid_batch = BatchGenerator(
        val_imgs,
        generator_config,
        jitter=False
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        mode='min',
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        temp_training_model,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    onfinish = callbacks.LambdaCallback(
        on_train_end=lambda logs: save_checkpoint(
            logs,
            temp_training_model,
            prev_model
        )
    )

    dirs = listdir(path.expanduser(tensorboard_log))
    arr_log = [log for log in dirs if 'wr_' in log]
    tb_counter = len(arr_log) + 1
    tensorboard = callbacks.TensorBoard(
        log_dir=path.expanduser(
            tensorboard_log
        ) + 'wr_' + '_' + str(tb_counter),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

    optimizer = optimizers.Adam(
        lr=0.5e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0,
    )

    model.compile(loss=custom_loss, optimizer=optimizer)

    model.fit_generator(
        generator=train_batch.get_generator(),
        steps_per_epoch=train_batch.get_dateset_size(),
        epochs=num_of_epochs,
        verbose=1,
        validation_data=valid_batch.get_generator(),
        validation_steps=valid_batch.get_dateset_size(),
        callbacks=[early_stop, checkpoint, tensorboard, onfinish],
        max_queue_size=3,
    )


if __name__ == '__main__':
    start()
