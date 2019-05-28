# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:26:58 2018
@author: manjotms10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
import tensorflow as tf
from keras.optimizers import Adam
from scipy.misc import imresize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img


#
# def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
#     while True:
#         ix = np.random.choice(np.arange(len(images)), batch_size)
#         imgs = []
#         labels = []
#         for i in ix:
#             # images
#             original_img = load_img(data_dir + images[i])
#             resized_img = imresize(original_img, dims + [3])
#             array_img = img_to_array(resized_img) / 255
#             imgs.append(array_img)
#
#             # masks
#             original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
#             resized_mask = imresize(original_mask, dims + [3])
#             array_mask = img_to_array(resized_mask) / 255
#             labels.append(array_mask[:, :, 0])
#         imgs = np.array(imgs)
#         labels = np.array(labels)
#         yield imgs, labels.reshape(-1, dims[0], dims[1], 1)
#
#
# train_gen = data_gen_small(data_dir, mask_dir, train_images, 5, [128, 128])
# img, msk = next(train_gen)


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2


def get_unet_model(filters=64, input_size=(256, 256, 1)):
    # Make a custom U-nets implementation.
    filters = 64
    input_layer = Input(input_size)
    layers = [input_layer]
    residuals = []

    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)
    filters *= 2

    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)
    filters *= 2

    # Down 3, 32
    d3, res3 = down(d2, filters)
    residuals.append(res3)
    filters *= 2

    # Down 4, 16
    d4, res4 = down(d3, filters)
    residuals.append(res4)
    filters *= 2

    # Down 5, 8
    d5 = down(d4, filters, pool=False)

    # Up 1, 16
    up1 = up(d5, residual=residuals[-1], filters=filters / 2)
    filters /= 2

    # Up 2,  32
    up2 = up(up1, residual=residuals[-2], filters=filters / 2)
    filters /= 2

    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters / 2)
    filters /= 2

    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters / 2)
    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, out)

    def dice_coef(y_true, y_pred):
        smooth = 1e-5

        y_true = tf.round(tf.reshape(y_true, [-1]))
        y_pred = tf.round(tf.reshape(y_pred, [-1]))

        isct = tf.reduce_sum(y_true * y_pred)

        return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    # model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# model.fit_generator(train_gen, steps_per_epoch=10, epochs=10)