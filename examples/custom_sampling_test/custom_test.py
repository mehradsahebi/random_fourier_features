#!/usr/bin/env python3
#
# This Python script provides an example usage of CUSSVC class which is a class for
# SVM classifier using RFF. Interface of CUSSVC is quite close to sklearn.svm.SVC.
##################################################### SOURCE START #####################################################

"""
Overview:
  Train Random Fourier Feature SVM. Before running this script, make sure to create MNIST dataset.
  As a comparison with Kernel SVM, this script has a capability to run a Kernel SVM as the same
  condition with RFF SVM.

Usage:
    custom_test.py kernel [--input <str>] [--output <str>] [--pcadim <int>] [--kernel <str>]
                                     [--gamma <float>] [--C <float>] [--seed <int>] [--use_fft]
    custom_test.py cpu [--input <str>] [--output <str>] [--pcadim <int>] [--rtype <str>]
                                  [--kdim <int>] [--stdev <float>] [--seed <int>] [--cpus <int>] [--use_fft]
    custom_test.py gpu [--input <str>] [--output <str>] [--pcadim <int>] [--rtype <str>]
                                  [--kdim <int>] [--stdev <float>] [--seed <int>] [--cpus <int>] [--use_fft]
    custom_test.py (-h | --help)

Options:
    --max_freq       Maximum frequency of the data set.                  [default: 5]
    --dim            Dimension of the data set.                          [default: 2]
    --n_samples      Number of samples.                                  [default: 1000]
    kernel           Run kernel SVM classifier.
    cpu              Run RFF SVM on CPU.
    gpu              Run RFF SVM on GPU.
    --input <str>    Directory path to the MNIST dataset.                [default: ../../dataset/mnist]
    --output <str>   File path to the output pickle file.                [default: result.pickle]
    --pcadim <int>   Output dimension of Principal Component Analysis.   [default: 128]
    --kernel <str>   Hyper parameter of kernel SVM (type of kernel).     [default: rbf]
    --gamma <float>  Hyper parameter of kernel SVM (softness of kernel). [default: auto]
    --C <float>      Hyper parameter of kernel SVM (margin allowance).   [default: 1.0]
    --rtype <str>    Type of random matrix (rff/orf/qrf).                [default: cus]
    --kdim <int>     Hyper parameter of RFF/ORF SVM (dimention of RFF).  [default: 1024]
    --stdev <float>  Hyper parameter of RFF/ORF SVM (stdev of RFF).      [default: 0.05]
    --seed <int>     Random seed.                                        [default: 111]
    --cpus <int>     Number of available CPUs.                           [default: -1]
    --use_fft        Apply FFT to the MNIST images.                      [default: False]
    --dist           Distribution of the random matrix.                  [default: uniform]
    -h, --help       Show this message.
"""

import os
import sys
import sys
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parents[2]))
import pickle
import sys
from random_fourier_features.rfflearn.cpu.rfflearn_cpu_svc import CustomOneClassSVM
import docopt
import numpy as np
import sklearn as skl
import sklearn.svm
from itertools import product


### Load train/test image data.
def make_dataset(dim, n_samples, max_freq):
    temp = np.arange(max_freq+1)
    W0 = np.array(list(product(temp, repeat=dim))).reshape(dim, -1)
    b0 = np.random.uniform(0, 2 * np.pi, (W0.shape[1], 1))
    C0 = np.random.randn(W0.shape[1], 1)
    X = 2 * np.pi * np.random.randn(n_samples, dim)
    dic_bound = lambda x:  np.cos(x@W0 + b0.T)@C0
    plt.plot(np.array(list((np.linspace(-np.pi, np.pi, 100)))), dic_bound(np.array(list(product(np.linspace(-np.pi, np.pi, 100))))), label = 'decision bound')

    y = np.sign(dic_bound(X))

    return X, y


### Main procedure.
def main(args):
    dim = args["--dim"]
    max_freq = args["--max_freq"]
    ### Print all arguments for debuging purposes.
    print("Program starts: args =", args)

    ### Fix seed for random fourier feature calculation.
    if args["cpu"] or args["gpu"]:
        rfflearn.seed(args["--seed"])

    if args["--dist"] == 'uniform':
        p = np.ones((*[2*max_freq+1 for _ in range(dim)], ))
        p = p / np.sum(p)
        dist = p
    elif args["--dist"] == 'delta':
        dist = np.zeros((*[2*max_freq+1 for _ in range(dim)], ))
        dist[tuple([max_freq for _ in range(dim)])] = 1
    else:
        dist = args["dist"]

    ### Create classifier instance.
    if args["kernel"]:
        svc = CustomOneClassSVM(kernel=args["--kernel"], gamma=args["--gamma"])
    elif args["--rtype"] == "rff":
        svc = rfflearn.RFFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], tol=1.0E-3, n_jobs=args["--cpus"],
                              oc=args["--oc"])
    elif args["--rtype"] == "orf":
        svc = rfflearn.ORFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], tol=1.0E-3, n_jobs=args["--cpus"],
                              oc=args["--oc"])
    elif args["--rtype"] == "qrf":
        svc = rfflearn.QRFSVC(dim_kernel=args["--kdim"], std_kernel=args["--stdev"], tol=1.0E-3, n_jobs=args["--cpus"],
                              oc=args["--oc"])
    elif args["--rtype"] == "cus":
        svc = rfflearn.CUSSVC(dim_kernel=args["--kdim"], tol=1.0E-3, n_jobs=args["--cpus"],
                              dist=dist, oc=args["--oc"])
    else:
        exit("Error: First argument must be 'kernel', 'rff' or 'orf'.")

    ### Load training data.
    with utils.Timer("Loading training and test data: "):
        X, y = make_dataset(dim, args["--n_samples"], max_freq)

        Xs_train = X[:int(0.9*args['--n_samples']),:]
        ys_train = y[:int(0.9*args['--n_samples'])]
        Xs_test = X[int(0.9*args['--n_samples']):, :]
        ys_test = y[int(0.9*args['--n_samples']):]

    ### Train SVM.
    with utils.Timer("SVM learning: "):
        svc.fit(Xs_train, ys_train)

    ### Calculate score for test data.
    with utils.Timer("SVM prediction time for 1 image: ", unit="us", devide_by=ys_test.shape[0]):
        score = 100 * svc.score(Xs_test, ys_test)
    print("Score = %.2f [%%]" % score)
    print(svc.W)
    db=  svc.decision_function
    plt.plot(np.array(list((np.linspace(-np.pi, np.pi, 100)))), db(np.array(list(product(np.linspace(-np.pi, np.pi, 100))))), label = 'training bound')
    plt.legend()
    plt.show()
    ### Save training results.
    with utils.Timer("Saving model: "):
        with open(args["--output"], "wb") as ofp:
            pickle.dump({"svc": svc, "args": args}, ofp)


if __name__ == "__main__":

    ### Parse input arguments.
    # args = docopt.docopt(__doc__)
    args = dict({'--n_samples': 5000, '--dim': 1, '--max_freq': 10,"cpu": True, "--seed": 111, "kernel": False, '--rtype': 'cus', '--kdim': 35 , '--stdev': 0.05, '--cpus': -1,
                 '--use_fft': False, '--input': '../../dataset/mnist', '--output': 'result.pickle', '--pcadim': 256,
                 '--kernel': 'rbf', '--gamma': 'auto', '--C': 1.0, '--oc': False, '--dist': 'uniform'})
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    ### Add path to 'rfflearn/' directory.
    ### The followings are not necessary if you copied 'rfflearn/' to the current
    ### directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    if args["cpu"]:
        import rfflearn.cpu as rfflearn
    elif args["gpu"]:
        import rfflearn.gpu as rfflearn
    import rfflearn.utils as utils

    ### Convert all arguments to an appropriate type.
    for k, v in args.items():
        try:
            args[k] = eval(str(v))
        except:
            args[k] = str(v)

    ### Run main procedure.
    main(args)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
