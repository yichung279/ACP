#!/usr/bin/env python3
from keras.models import load_model
from sys import argv
import numpy as np
from sklearn.metrics import matthews_corrcoef

from train import train

if '__main__' == __name__:
    model_name_base = argv[1]
    n_folds = int(argv[2])

    mccs = []

    for i in range(n_folds):

        model_name = f'{model_name_base}_{i}'

        train_x, train_y = np.load(f'features/train_x_{i}.npy'), np.load(f'features/train_y_{i}.npy')
        valid_x, valid_y = np.load(f'features/valid_x_{i}.npy'), np.load(f'features/valid_y_{i}.npy')

        mcc = train(model_name, train_x, train_y, valid_x, valid_y)

        mccs.append(mcc)

    print(mccs)

    mccs = np.array(mccs)
    best_model_index = np.argmax(mccs)

    test_x, test_y = np.load(f'features/teste_x.npy'), np.load(f'features/test_y.npy')
    model = load_model(f'models/{model_name_base}_{best_model_index}.h5')
    pred_y = model.predict(test_x)

    print(matthews_corrcoef(test_y, np.argmax(pred_y, axis=1)))
