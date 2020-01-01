#!/usr/bin/env python3

import os, sys
import random
from Bio import SeqIO
import numpy as np

amino_acid_list = 'VLIMFWYGAPSTCHRKQEND'
sequence_length = 60

def build_protein(protein):
    feature = []
    for c in protein[:sequence_length]:
        onehot = np.zeros(20)
        if c in amino_acid_list:
            onehot[amino_acid_list.index(c)] = 1.
            feature.append(onehot)
        else:
            feature.append(onehot)

    onehot = np.zeros(20)
    while len(feature) < sequence_length:
        feature.append(onehot)

    return np.array(feature)

def build_label(label):
    if '0' == label:
        return np.array([1, 0])
    elif '1' == label:
        return np.array([0, 1])
    else:
        print('Error labeling')

def parse_fasta(filename):
    records = list(SeqIO.parse(filename, "fasta"))
    proteins = [record.seq for record in records]
    labels = [record.id.split('|')[1] for record in records]

    return proteins, labels

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print('Please specific file name')
        exit(2)

    proteins, labels = parse_fasta(sys.argv[1])

    lens=[len(protein) for protein in proteins]
    lens = np.array(lens)
    print(f'Proteins Analysis\n\
          Max Length: {lens.max()},\n\
          Min Length: {lens.min()},\n\
          Avg Length: {np.mean(lens)},\n\
          Length Std: {np.std(lens)}')

    proteins = [build_protein(protein) for protein in proteins]
    proteins = np.array(proteins)

    labels = [build_label(label) for label in labels]
    labels = np.array(labels)

    np.save(sys.argv[2], proteins)
    np.save(sys.argv[3], labels)
