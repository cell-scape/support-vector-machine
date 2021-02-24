#! /usr/bin/env python3

import sys
import os.path
import argparse

import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mnist import MNIST


def svm_predict(train, test, kernel="linear"):
    linear_svm = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel))
    linear_svm.fit(train['X'], train['y'])
    predictions = [linear_svm.predict(x) for x in test['X']]
    return [(x == y, int(x), y) for x, y in zip(predictions, test['y'])]


def getdata(dataset='digits'):
    mndata = MNIST('./emnist_data')
    mndata.gz = True
    mndata.select_emnist(dataset)
    return mndata.load_training()


def experiment1(train, test, kernel):
    results = svm_predict(train, test, kernel)
    accuracy = sum(results[i][0] for i in range(len(results))) / len(results)
    print(f"Training set: {len(train)} images, Test set: {len(test)} images")
    print(f"Accuracy: {accuracy}")
    for i, result in enumerate(results):
        if result[0]:
            continue
        print(f"Sample {i}: Predicted {result[1]}, actually {result[2]}")



if __name__ == '__main__':
    images, labels = getdata()
    test_X = [np.array(image).reshape(1, -1) for image in images[30000:31000]]
    train = {'X': images[:10000], 'y': labels[:10000]}
    test = {'X': test_X, 'y': labels[30000:31000]}
    experiment1(train, test, "linear")
