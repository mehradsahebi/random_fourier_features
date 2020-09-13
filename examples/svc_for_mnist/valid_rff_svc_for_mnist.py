#!/usr/bin/env python3
#
# This Python script provides an example usage of RFFSVC class which is a class for
# SVM classifier using RFF. Interface of RFFSVC is quite close to sklearn.svm.SVC.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : February 08, 2020
##################################################### SOURCE START #####################################################

"""
Overview:
  Validate Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same condition with RFF SVM.

Usage:
    main_rff_svc_for_mnist.py cpu [--input <str>] [--model <str>]
    main_rff_svc_for_mnist.py tensorflow [--input <str>] [--model <str>] [--batch_size <int>]
    main_rff_svc_for_mnist.py -h|--help

Options:
    cpu                 Run inference on CPU.
    tensorflow          Run inference on GPU using Tensorflow 2.
    --input <str>       Directory path to the MNIST dataset.   [default: ../../dataset/mnist]
    --model <str>       File path to the output pickle file.   [default: result.pickle]
    --batch_size <int>  Batch size for GPU inference.          [default: 2000]
    -h, --help          Show this message.
"""

import sys
import os

### Add path to PyRFF.py.
### The followings are not necessary if you copied PyRFF.py to the current directory
### or other directory which is included in the Python path.
current_dir = os.path.dirname(__file__)
module_path = os.path.join(current_dir, "../../source")
sys.path.append(module_path)

import time
import pickle
import docopt
import numpy     as np
import sklearn   as skl
import PyRFF     as pyrff
import utils

### Import utility functions from training script.
from train_rff_svc_for_mnist import vectorise_MNIST_images, vectorise_MNIST_labels, mat_transform_pca


### Load train/test image data.
def vectorise_MNIST_images(filepath):
    Xs = np.load(filepath)
    return np.array([Xs[n, :, :].reshape((28 * 28, )) for n in range(Xs.shape[0])]) / 255.0


### Load train/test label data.
def vectorise_MNIST_labels(filepath):
    return np.load(filepath)


### PCA analysis for dimention reduction.
def mat_transform_pca(Xs, dim):
    _, V = np.linalg.eig(Xs.T.dot(Xs))
    return np.real(V[:, :dim])


### Inference using CPU.
def main(args):

    ### Load test data.
    with utils.Timer("Loading test data: "):
        Xs_test = vectorise_MNIST_images(os.path.join(args["--input"], "MNIST_test_images.npy")).astype(np.float32)
        ys_test = vectorise_MNIST_labels(os.path.join(args["--input"], "MNIST_test_labels.npy")).astype(np.float32)

    ### Load pickled result.
    with utils.Timer("Loading model: "):
        with open(args["--model"], "rb") as ifp:
            result = pickle.load(ifp)
        svc = result["svc"]
        T   = result["pca"]

    ### If GPU inference, convert PyRFF.RFFSVC -> PyRFF_GPU.RFFSVC_GPU and overwrite 'svc' instance.
    if args["tensorflow"]:
        import PyRFF_GPU  as pyrff_gpu
        svc = pyrff_gpu.RFFSVC(batch_size = args["--batch_size"]).from_RFFSVC_cpu(svc, T)

    ### Calculate score for test data.
    with utils.Timer("SVM prediction time for 1 image: ", unit = "us", devide_by = ys_test.shape[0]):

        ### In case of GPU inference, Calculation of PCA ".dot(T)" is not necessary because PCA matrix T is
        ### embedded to GPU computation graph as a preprocessing matrix for faster calculation.
        if   args["cpu"]       : score = 100 * svc.score(Xs_test.dot(T), ys_test)
        elif args["tensorflow"]: score = 100 * svc.score(Xs_test,        ys_test)

    print("Score = %.2f [%%]" % score)


if __name__ == "__main__":

    ### Parse input arguments.
    args = docopt.docopt(__doc__)

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try   : args[k] = eval(str(v))
        except: args[k] = str(v)

    ### Run main procedure.
    main(args)


##################################################### SOURCE FINISH ####################################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
