#! /usr/bin/env python3

import sys
from os.path import abspath
from argparse import ArgumentParser

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from mnist import MNIST


def get_classifier(kernel='linear', C=1.0, coef0=0.0, gamma="scale", degree=3):
    return make_pipeline(StandardScaler(), SVC(kernel=kernel,
                                               C=C,
                                               degree=degree,
                                               gamma=gamma,
                                               coef0=coef0))


def get_data(dataset='digits'):
    mndata = MNIST(abspath('./emnist_data'))
    mndata.gz = True
    mndata.select_emnist(dataset)
    return mndata.load_training()


def predict(clf, train, test):
    clf.fit(train['X'], train['y'])
    predictions = [clf.predict(x) for x in test['X']]
    results = [(x == y, int(x), y) for x, y in zip(predictions, test['y'])]
    accuracy = sum(results[i][0] for i in range(len(results))) / len(results)
    print(f"\nTraining set: {len(train['X'])} images, Test set: {len(test['X'])} images\n")
    correct = 0
    incorrect = 0
    for i, result in enumerate(results):
        if result[0]:
            correct += 1
            continue
        print(f"Sample {i}: Predicted {result[1]}, actually {result[2]}")
        incorrect += 1
    print(f"\nAccuracy: {accuracy}, Correct: {correct}, Incorrect: {incorrect}\n")
    return accuracy, list(int(x) for x in predictions), list(int(y) for y in test['y'])


def get_binary_images(images, threshold):
    binary_images = []
    for image in images:
        binary_image = []
        for pixel in image:
            if pixel >= threshold:
                binary_image.append(255)
            else:
                binary_image.append(0)
        binary_images.append(binary_image)
    return binary_images


def experiment(clf, images, labels, exp=0, threshold=100):
    if exp == 1:
        print()
        testX = [np.array(image).reshape(1, -1) for image in images[2000:2100]]
        train = {'X': images[:1000], 'y': labels[:1000]}
        test = {'X': testX, 'y': labels[2000:2100]}
        accuracy, predictions, actual = predict(clf, train, test)
        print(confusion_matrix(actual, predictions))
        return accuracy
    elif exp == 2:
        print()
        testX = [np.array(image).reshape(1, -1) for image in images[20000:20100]]
        train = {'X': images[:10000], 'y': labels[:10000]}
        test = {'X': testX, 'y': labels[20000:20100]}
        accuracy, predictions, actual = predict(clf, train, test)
        print(confusion_matrix(actual, predictions))
        return accuracy
    elif exp == 3:
        print()
        testX = [np.array(image).reshape(1, -1) for image in images[20000:21000]]
        train = {'X': images[:10000], 'y': labels[:10000]}
        test = {'X': testX, 'y': labels[20000:21000]}
        accuracy, predictions, actual = predict(clf, train, test)
        print(confusion_matrix(actual, predictions))
        return accuracy
    elif exp == 4:
        a1 = experiment(clf, images, labels, exp=1)
        a2 = experiment(clf, images, labels, exp=2)
        a3 = experiment(clf, images, labels, exp=3)
        return
    elif exp == 5:
        print("\nRBF Kernel\n")
        rbf = get_classifier(kernel="rbf")
        ar = experiment(rbf, images, labels, exp=3)
        print("\nPolynomial Kernel\n")
        poly = get_classifier(kernel="poly")
        ap = experiment(poly, images, labels, exp=3)
        print("\nSigmoid Kernel\n")
        sigmoid = get_classifier(kernel="sigmoid")
        aS = experiment(sigmoid, images, labels, exp=3)
        return ar, ap, aS
    elif exp == 6:
        print("\nBinary Images\n")
        binary_images = get_binary_images(images[:10000], threshold)
        testX = [np.array(image).reshape(1, -1) for image in images[20000:21000]]
        train = {'X': binary_images, 'y': labels[:10000]}
        test = {'X': testX, 'y': labels[20000:21000]}
        accuracy, predictions, actual = predict(clf, train, test)
        print(confusion_matrix(actual, predictions))
        return accuracy
    else:
        a1 = experiment(clf, images, labels, exp=1)
        a2 = experiment(clf, images, labels, exp=2)
        a3 = experiment(clf, images, labels, exp=3)
        a5r, a5p, a5s = experiment(clf, images, labels, exp=5)
        a6 = experiment(clf, images, labels, exp=6, threshold=threshold)
        print("\nPrediction Accuracy:\n")
        print(f"Part 1: \t\t\t{a1}")
        print(f"Part 2: \t\t\t{a2}")
        print(f"Part 3: \t\t\t{a3}")
        print(f"Part 5 - RBF kernel: \t\t{a5r}")
        print(f"Part 5 - Polynomial kernel: \t{a5p}")
        print(f"Part 5 - Sigmoid kernel: \t{a5s}")
        print(f"Part 6: \t\t\t{a6}")
    return [a1, a2, a3, a5r, a5p, a5s, a6]


def setup_parser():
    parser = ArgumentParser(description="Predict EMNIST data with SVM")
    parser.add_argument("-e", "--experiment",
                        help="Choose assignment 3 experiment",
                        metavar="N",
                        dest="ex",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("-k", "--kernel",
                        help="Choose kernel function",
                        metavar="string",
                        type=str,
                        default="linear",
                        choices=["linear", "rbf", "poly", "sigmoid"],
                        dest="kernel",
                        required=False)
    parser.add_argument("-C", "--C",
                        help="Regularization Parameter C, strictly positive",
                        metavar="float",
                        type=float,
                        default=1.0,
                        dest="C",
                        required=False)
    parser.add_argument("-g", "--gamma",
                        help="Kernel coefficient for rbf, poly, sigmoid",
                        choices=["scale", "auto"],
                        type=str,
                        metavar="string",
                        dest="gamma",
                        default="scale",
                        required=False)
    parser.add_argument("-c", "--coef0",
                        help="independent term in kernel function",
                        metavar="float",
                        type=float,
                        default=0.0,
                        dest="coef0",
                        required=False)
    parser.add_argument("-d", "--degree",
                        help="polynomial kernel degree",
                        metavar="int",
                        type=int,
                        dest="degree",
                        default=3,
                        required=False)
    parser.add_argument("-m", "--mnist-data",
                        help="choose emnist dataset",
                        metavar="string",
                        type=str,
                        dest="mnist",
                        choices=["digits", "balanced", "byclass", "bymerge", "letters", "mnist"],
                        default="digits",
                        required=False)
    parser.add_argument("-b", "--binary-image-threshold",
                        help="create binary images by threshold",
                        metavar="int",
                        type=int,
                        dest="binary_threshold",
                        default=100,
                        required=False)
    return parser


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    images, labels = get_data(args.mnist)
    clf = get_classifier(args.kernel, args.C, args.coef0, args.gamma, args.degree)
    _ = experiment(clf, images, labels, args.ex)
    sys.exit(0)
