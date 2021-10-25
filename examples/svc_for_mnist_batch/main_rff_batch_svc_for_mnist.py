#!/usr/bin/env python3
#
# This python script provides an example of RFFSVC with batch learning
# on the MNIST dataset.
#################################### SOURCE START ###################################

import sys
import os

import numpy as np

### Load train/test image data
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0

### Load train/test label data
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)

### PCA analysis for dimention reduction
def mat_transform_pca(Xs, dim = 100):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])

### Main procedure
def main():

    ### Fix seed for random fourier feature calclation
    rfflearn.seed(111)

    ### Create classifier instance
    svc = rfflearn.RFFBatchSVC(dim_kernel = 1024, std_kernel = 0.10, num_epochs = 10, num_batches = 10, alpha = 0.05)

    ### Load training data
    with utils.Timer("Loading training data: "):
        Xs_train = vectorise_MNIST_images("../../dataset/mnist/MNIST_train_images.npy")
        ys_train = vectorise_MNIST_labels("../../dataset/mnist/MNIST_train_labels.npy")

    ### Load test data
    with utils.Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images("../../dataset/mnist/MNIST_test_images.npy")
        ys_test = vectorise_MNIST_labels("../../dataset/mnist/MNIST_test_labels.npy")

    ### Create matrix for principal component analysis
    with utils.Timer("Calculate PCA matrix: "):
        T = mat_transform_pca(Xs_train, dim = 256)

    ### Train SVM with batch random fourier features
    with utils.Timer("Batch RFF SVM learning time: "):
        svc.fit(Xs_train.dot(T), ys_train, test = (Xs_test.dot(T), ys_test), max_iter = 1E6, tol = 1.0E-2)

    ### Calculate score for test data
    with utils.Timer("RFF SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):
        score = 100 * svc.score(Xs_test.dot(T), ys_test)
    print("Score = %.2f [%%]" % score)

if __name__ == "__main__":

    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu   as rfflearn
    import rfflearn.utils as utils

    ### Run main procedure.
    main()

#################################### SOURCE FINISH ##################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
