#!/usr/bin/env python3


import os
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


from utils.network import yolo, true_boxes, LABELS, \
    IMAGE_H, IMAGE_W, GRID_H, GRID_W, \
    BOX, CLASS, CLASS_WEIGHTS, OBJ_THRESHOLD, NMS_THRESHOLD, \
    ANCHORS, NO_OBJECT_SCALE, OBJECT_SCALE, COORD_SCALE, CLASS_SCALE, \
    BATCH_SIZE, WARM_UP_BATCHES, TRUE_BOX_BUFFER, ALPHA


train_image_folder = 'temp/dataset/train/images/'
train_annot_folder = 'temp/dataset/train/labels/'
val_image_folder = 'temp/dataset/eval/images/'
val_annot_folder = 'temp/dataset/eval/labels/'

pre_trained_weights = 'temp/weights/yolo.weights'
pre_trained_model = 'temp/pretrained.h5'

tensorboard_log = 'temp/logs/'


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


model = yolo()
model.summary()

weight_reader = WeightReader(pre_trained_weights)
weight_reader.reset()
nb_conv = 23

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


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]

    cell_x = tf.to_float(
        tf.reshape(
            tf.tile(
                tf.range(GRID_W),
                [GRID_H]
            ),
            (1, GRID_H, GRID_W, 1, 1)
        )
    )
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(
        tf.concat([cell_x, cell_y], -1),
        [BATCH_SIZE, 1, 1, 5, 1]
    )

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)

    total_AP = tf.Variable(0.)

    """
    Adjust prediction
    """
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    pred_box_wh = tf.exp(
        y_pred[..., 2:4]
    ) * np.reshape(
        ANCHORS,
        [1, 1, 1, BOX, 2]
    )
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    pred_box_class = y_pred[..., 5:]

    true_box_xy = y_true[..., 0:2]
    true_box_wh = y_true[..., 2:4]

    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]

    true_box_class = tf.to_int32(y_true[..., 5])

    """
    Determine the masks
    """
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(
        best_ious < 0.6
    ) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

    class_mask = y_true[..., 4] * tf.gather(
        CLASS_WEIGHTS, true_box_class
    ) * CLASS_SCALE

    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(
        tf.less(seen, WARM_UP_BATCHES),
        lambda: [
            true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                ANCHORS,
                [1, 1, 1, BOX, 2]
            ) * no_boxes_mask,
            tf.ones_like(coord_mask)
        ],
        lambda: [true_box_xy, true_box_wh, coord_mask]
    )

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(
        tf.square(
            true_box_xy-pred_box_xy
        ) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(
        tf.square(
            true_box_wh-pred_box_wh
        ) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(
        tf.square(
            true_box_conf-pred_box_conf
        ) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_box_class,
        logits=pred_box_class
    )
    loss_class = tf.reduce_sum(
        loss_class * class_mask
    ) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(
        tf.to_float(
            true_box_conf > 0.5
        ) * tf.to_float(
            pred_box_conf > OBJ_THRESHOLD
        )
    )

    total_AP = tf.assign_add(total_AP, nb_pred_box/nb_true_box)

    loss = tf.Print(
        loss,
        [
            loss_xy,
            loss_wh,
            loss_conf,
            loss_class,
            loss,
            total_AP/seen
        ],
        message='DEBUG',
        summarize=1000
    )

    return loss


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
    pre_trained_model,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1
)

model.load_weights(pre_trained_model)

dirs = os.listdir(os.path.expanduser(tensorboard_log))
arr_log = [log for log in dirs if 'wr_' in log]
tb_counter = len(arr_log) + 1
tensorboard = callbacks.TensorBoard(
    log_dir=os.path.expanduser(
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
    epochs=100,
    verbose=1,
    validation_data=valid_batch.get_generator(),
    validation_steps=valid_batch.get_dateset_size(),
    callbacks=[early_stop, checkpoint, tensorboard],
    max_queue_size=3,
)
