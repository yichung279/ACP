#!/usr/bin/env python3
from sys import argv
import numpy as np

from train import train

if '__main__' == __name__:
    model_name_base = argv[1]

    mccs = []

    for i in range(10):

        model_name = f'{model_name_base}_{i}'

        train_x, train_y = np.load(f'features/train_x_{i}.npy'), np.load(f'features/train_y_{i}.npy')
        valid_x, valid_y = np.load(f'features/valid_x_{i}.npy'), np.load(f'features/valid_y_{i}.npy')

        mcc = train(model_name, train_x, train_y, valid_x, valid_y)

        mccs.append(mcc)

    print(mccs)
