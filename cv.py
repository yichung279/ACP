#!/usr/bin/env python3
from keras.models import load_model
from sys import argv
import numpy as np
from sklearn.metrics import matthews_corrcoef

from train import train

def cross_validate(model_name_base, n_folds, kernel_size, layer_count, filter_base, filter_growth):

    test_x, test_y = np.load(f'features/test_x.npy'), np.load(f'features/test_y.npy')

    mccs = []
    for i in range(n_folds):

        model_name = f'{model_name_base}_{i}'

        train_x, train_y = np.load(f'features/train_x_{i}.npy'), np.load(f'features/train_y_{i}.npy')
        valid_x, valid_y = np.load(f'features/valid_x_{i}.npy'), np.load(f'features/valid_y_{i}.npy')

        _, parameter_count = train(model_name, train_x, train_y,\
                                     valid_x, valid_y,\
                                     kernel_size, layer_count, filter_base, filter_growth)

        model = load_model(f'models/{model_name_base}_{i}.h5')
        pred_y = model.predict(test_x)

        mcc = matthews_corrcoef(test_y, np.argmax(pred_y, axis=1))

        mccs.append(mcc)

    mccs = np.array(mccs)

    return np.average(mccs), parameter_count

if '__main__' == __name__:
    model_name_base = argv[1]
    n_folds = int(argv[2])

    kernel_size_space = [3, 5, 7, 9, 11, 13]
    layer_count_space = [1, 2, 3]
    filter_base_space = [16, 24, 32, 40]
    filter_growth_space = [4, 8, 16, 24]

    for kernel_size in kernel_size_space:
        for layer_count in layer_count_space:
            for filter_base in filter_base_space:
                for filter_growth in filter_growth_space:

                    mcc, parameter_count = cross_validate(\
                        model_name_base, n_folds,\
                        kernel_size, layer_count,\
                        filter_base, filter_growth)

                    print(f'k, l, f_b, f_g: {kernel_size}, {layer_count}, {filter_base}, {filter_growth}')
                    print(f'mcc, p_count: {mcc}, {parameter_count}')
