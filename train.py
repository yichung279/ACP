#!/usr/bin/env python3
import numpy as np
import os
from sys import argv
import tensorflow as tf

from keras.layers import Input, Dense, Flatten, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, concatenate, add
from keras import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
# import tensorflow.keras.backend as K

from sklearn.metrics import matthews_corrcoef

def inception_res_bolck(inputs, filter_1, filter_5_r, filter_5, filter_9_r, filter_9, filter_p, res=False):

    block_1 = Conv1D(filter_1, 1, padding='same', activation='relu')(inputs)

    block_5 = Conv1D(filter_5_r, 1, padding='same', activation='relu')(inputs)
    block_5 = Conv1D(filter_5, 5, padding='same', activation='relu')(block_5)

    block_9 = Conv1D(filter_9_r, 1, padding='same', activation='relu')(inputs)
    block_9 = Conv1D(filter_9, 9, padding='same', activation='relu')(block_9)

    block_p = MaxPooling1D(strides=1, padding='same')(inputs)
    block_p = Conv1D(filter_p, 1, padding='same', activation='relu')(inputs)

    block_inception = concatenate([block_1, block_5, block_9, block_p])

    if not res:
        return block_inception

    model = Conv1D(int(inputs.shape[-1]), 1, padding='same', activation='relu')(block_inception)
    model = add([inputs, model])

    return model

def build_model(input_shape):

    inputs = Input(shape=input_shape)

    model = inception_res_bolck(inputs, 64, 64, 80, 16, 48, 64)
    model = inception_res_bolck(model, 128, 128, 192, 32, 96, 64)
    model = MaxPooling1D(2)(model)

    model = inception_res_bolck(inputs, 192, 96, 208, 16, 48, 64)
    model = inception_res_bolck(model, 128, 112, 224, 24, 64, 64)
    model = MaxPooling1D(2)(model)

    model = inception_res_bolck(model, 128, 112, 224, 24, 64, 64)
    model = inception_res_bolck(model, 128, 80, 160, 64, 128, 64)
    model = GlobalAveragePooling1D()(model)

    model = Dropout(0.4)(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(2, activation='softmax')(model)

    model = Model(inputs, model)

    model.summary()

    return model

def train(model_name, train_x, train_y, valid_x, valid_y, **kwargs):

    if not os.path.exists('models'): os.makedirs('models')
    if not os.path.exists('runs'): os.makedirs('runs')

    model_ckpt = ModelCheckpoint(f'models/{model_name}.h5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir=f'runs/{model_name}' , histogram_freq=0, write_graph=True, write_images=False)
    early_stp = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

    model = build_model((60, 20))

    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x, train_y,
        validation_data = (valid_x, valid_y),\
        # validation_split = 0.2,\
        shuffle=True,\
        batch_size=32,\
        epochs=100,\
        callbacks=[model_ckpt, tensorboard, early_stp])

    pred_y = model.predict(valid_x)

    return matthews_corrcoef(valid_y, np.argmax(pred_y, axis=1))

if '__main__' == __name__:
    model_name = argv[1]

    train_x, train_y = np.load('features/train_x_0.npy'), np.load('features/train_y_0.npy')
    valid_x, valid_y = np.load('features/valid_x_0.npy'), np.load('features/valid_y_0.npy')

    mcc = train(model_name, train_x, train_y, valid_x, valid_y)

    print(mcc)
