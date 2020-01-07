#!/usr/bin/env python3

import os, sys
from random import shuffle
from Bio import SeqIO
import numpy as np

amino_acid_list = 'VLIMFWYGAPSTCHRKQEND'
sequence_length = 60

def build_protein(protein):
    features = []
    for c in protein[:sequence_length]:
        onehot = np.zeros(20)
        if c in amino_acid_list:
            onehot[amino_acid_list.index(c)] = 1.
            features.append(onehot)
        else:
            features.append(onehot)

    onehot = np.zeros(20)
    while len(features) < sequence_length:
        features.append(onehot)

    return np.array(features)

def build_label(label):
    return float(label)
    # if '0' == label:
    #     return np.array([1, 0])
    # elif '1' == label:
    #     return np.array([0, 1])
    # else:
    #     print('Error labeling')

def parse_fasta(filename):
    records = list(SeqIO.parse(filename, "fasta"))
    shuffle(records)
    proteins = [record.seq for record in records]
    labels = [record.id.split('|')[1] for record in records]

    lens = [len(protein) for protein in proteins]
    lens = np.array(lens)
    print(f'Proteins Analysis\n\
          Max Length: {lens.max()},\n\
          Min Length: {lens.min()},\n\
          Avg Length: {np.mean(lens)},\n\
          Length Std: {np.std(lens)}')

    return proteins, labels

def build_data(input_file, verbose=False):
    proteins, labels = parse_fasta(input_file)

    proteins = [build_protein(protein) for protein in proteins]
    proteins = np.array(proteins)

    labels = [build_label(label) for label in labels]
    labels = np.array(labels)

    print(f'features_shape: {proteins.shape}, {labels.shape}')

    np.save('features/test_x.npy', proteins)
    np.save('features/test_y.npy', labels)

def build_cv_data(input_file, folds, verbose=False):
    proteins, labels = parse_fasta(input_file)

    proteins = [build_protein(protein) for protein in proteins]
    proteins = np.array(proteins)

    labels = [build_label(label) for label in labels]
    labels = np.array(labels)

    chunk_size = len(proteins) // folds

    for i in range(folds-1):
        train_proteins = np.concatenate((proteins[: chunk_size*i], proteins[chunk_size*(i+1):]), axis=0)
        train_labels = np.concatenate((labels[:chunk_size*i], labels[chunk_size*(i+1):]), axis=0)

        valid_proteins = proteins[chunk_size*i: chunk_size*(i+1)]
        valid_labels = labels[chunk_size*i: chunk_size*(i+1)]

        np.save(f'features/train_x_{i}.npy', train_proteins)
        np.save(f'features/train_y_{i}.npy', train_labels)

        np.save(f'features/valid_x_{i}.npy', valid_proteins)
        np.save(f'features/valid_y_{i}.npy', valid_labels)

        if verbose:
            print(f'train_{i}_shape: {train_proteins.shape}, {train_labels.shape}')
            print(f'valid_{i}_shape: {valid_proteins.shape}, {valid_labels.shape}')

    train_proteins = proteins[: chunk_size*(folds-1)]
    train_labels = labels[:chunk_size*(folds-1)]

    valid_proteins = proteins[chunk_size*(folds-1): chunk_size*(folds+1)]
    valid_labels = labels[chunk_size*(folds-1): chunk_size*folds+1]

    np.save(f'features/train_x_{folds-1}.npy', train_proteins)
    np.save(f'features/train_y_{folds-1}.npy', train_labels)

    np.save(f'features/valid_x_{folds-1}.npy', valid_proteins)
    np.save(f'features/valid_y_{folds-1}.npy', valid_labels)

    if verbose:
        print(f'train_{folds-1}_shape: {train_proteins.shape}, {train_labels.shape}')
        print(f'valid_{folds-1}_shape: {valid_proteins.shape}, {valid_labels.shape}')

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print('Please specific file name and features type')
        print('$./build_features.py [file_name] [train/test] [n_folds]')
        exit(2)

    if sys.argv[2] == 'test':
        build_data(sys.argv[1])

    elif sys.argv[2] == 'train':
        build_cv_data(sys.argv[1], int(sys.argv[3]))

    else:
        print('features type: train / test')
