#!/usr/bin/env python3
import numpy as np
import os
from sys import argv
import tensorflow as tf

from keras import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import concatenate, add, multiply, Reshape
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
# import tensorflow.keras.backend as K

from sklearn.metrics import matthews_corrcoef

def conv_bn(x, fliters, kernel_size, padding='same', activation='relu'):

    x = Conv1D(fliters, kernel_size, padding=padding, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    return x

def se_layer(x, ratio):

    out_dim = int(x.shape[-1])

    squeeze = GlobalAveragePooling1D()(x)

    excitation = Dense(units=out_dim // ratio, activation='relu')(squeeze)
    excitation = Dense(units=out_dim, activation='sigmoid')(excitation)
    excitation = Reshape((1, out_dim))(excitation)

    return  multiply([x, excitation])

def inception_se_res_bolck(x, filter_1, filter_5_r, filter_5, filter_9_r, filter_9, filter_p, se_ratio=None, res=False):

    block_1 = conv_bn(x, filter_1, 1)

    block_5 = conv_bn(x, filter_5_r, 1)
    block_5 = conv_bn(block_5, filter_5, 5)

    block_9 = conv_bn(x, filter_9_r, 1)
    block_9 = conv_bn(block_9, filter_9, 9)

    block_p = MaxPooling1D(strides=1, padding='same')(x)
    block_p = conv_bn(x, filter_p, 1)

    block_inception = concatenate([block_1, block_5, block_9, block_p])

    if se_ratio:
        block_inception = se_layer(block_inception, se_ratio)

    if not res:
        return block_inception

    model = conv_bn(block_inception, int(x.shape[-1]), 1)
    model = add([x, model])

    return model

def build_model(input_shape, kernel_size, layer_count, filter_base, filter_growth):

    inputs = Input(shape=input_shape)

    model = conv_bn(inputs, filter_base, kernel_size)
    model = se_layer(model, 8)

    for i in range(layer_count):
        model = conv_bn(model, filter_base + filter_growth, kernel_size)
    model = AveragePooling1D(2)(model)

    for i in range(layer_count):
        model = conv_bn(model, filter_base + filter_growth * 2, kernel_size)
    model = AveragePooling1D(2)(model)

    for i in range(layer_count):
        model = conv_bn(model, filter_base + filter_growth *3, kernel_size)
    model = AveragePooling1D(2)(model)

    model = GlobalAveragePooling1D()(model)
    model = Dense(2, activation='softmax')(model)

    return Model(inputs, model)

def train(model_name, train_x, train_y, valid_x, valid_y, kernel_size, layer_count, filter_base, filter_growth):

    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('runs'): os.makedirs('runs')

    model_ckpt = ModelCheckpoint(f'models/{model_name}.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir=f'runs/{model_name}' , histogram_freq=0, write_graph=True, write_images=False)
    early_stp = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

    model = build_model((64, 20), kernel_size, layer_count, filter_base, filter_growth)
    model.summary()

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x, train_y,
        validation_data = (valid_x, valid_y),\
        # validation_split = 0.2,\
        shuffle=True,\
        batch_size=32,\
        epochs=100,\
        callbacks=[model_ckpt, tensorboard, early_stp])

    pred_y = model.predict(valid_x)

    return matthews_corrcoef(valid_y, np.argmax(pred_y, axis=1)), model.count_params()

if '__main__' == __name__:
    model_name = argv[1]

    train_x, train_y = np.load('features/train_x_0.npy'), np.load('features/train_y_0.npy')
    valid_x, valid_y = np.load('features/valid_x_0.npy'), np.load('features/valid_y_0.npy')

    kernel_size=3
    layer_count=1
    filter_base=24
    filter_growth=8

    mcc, parameter_count = train(model_name, train_x, train_y,\
                                 valid_x, valid_y,\
                                 kernel_size, layer_count, filter_base, filter_growth)

    print(mcc, parameter_count)
