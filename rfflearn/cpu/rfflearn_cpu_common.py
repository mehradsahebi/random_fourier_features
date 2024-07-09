"""
Common functions/classes for the other classes.
All classes except 'seed' function is not visible from users.
"""

import functools

import numpy as np
import scipy.stats
from itertools import product
from random import choices


def seed(seed):
    """
    Fix random seed used in this script.

    Args:
        seed (int): Random seed.
    """
    # Now it is enough to fix the random seed of Numpy.
    np.random.seed(seed)


def get_custom_matrix(dim_in, dim_out, dist):
    n = dist.shape

    if len(dist.shape) != dim_in:
        print(dist.shape, dim_in)
        raise ValueError("Dimension of the distribution should be the same as the input dimension")
    print('custom matrix is being used')
    d = dim_in
    temp = [np.array([i - nt // 2 for i in range(nt)]) for nt in n]
    a = np.array(list(product(*temp)))
    population = np.arange(len(a))
    idx = choices(population, weights = dist.flatten(), k=dim_out)
    return a[idx].T

def get_kron_matrix(dim_in, dim_out, dist):
    """
    Generates random matrix of seperable distributions.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        dist       (list)  : Number of frequencies to sample.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    n = [len(dist[i]) for i in range(len(dist))]
    if len(n) != dim_in:
        print(len(n), dim_in)
        raise ValueError("Dimension of the distribution should be the same as the input dimension")
    freqs = [np.array([i - nt // 2 for i in range(nt)]) for nt in n]
    print('kron matrix is being used')
    W = np.array([choices(freqs[j], weights = dist[j], k= dim_out) for j in range(dim_in)])
    return W

def get_rff_matrix(dim_in, dim_out, std, **args):
    """
    Generates random matrix of random Fourier features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    return std * np.random.randn(dim_in, dim_out)


def get_orf_matrix(dim_in, dim_out, std, **args):
    """
    Generates random matrix of orthogonal random features.

    Args:
        dim_in  (int)  : Input dimension of the random matrix.
        dim_out (int)  : Output dimension of the random matrix.
        std     (float): Standard deviation of the random matrix.

    Returns:
        (np.ndarray): Random matrix with shape (dim_out, dim_in).
    """
    # Initialize matrix W.
    W = None

    for _ in range(dim_out // dim_in + 1):
        s = scipy.stats.chi.rvs(df=dim_in, size=(dim_in,))
        Q = np.linalg.qr(np.random.randn(dim_in, dim_in))[0]
        V = std * np.dot(np.diag(s), Q)
        W = V if W is None else np.concatenate([W, V], axis=1)

    # Trim unnecessary part.
    return W[:dim_in, :dim_out]


def get_qrf_matrix(dim_in, dim_out, std, **args):
    """
    Generates random matrix for quasi-random Fourier features.

    Args:
        dim_in  (int)  : Input dimension of the quasi-random matrix.
        dim_out (int)  : Output dimension of the quasi-random matrix.
        std     (float): Standard deviation of the quasi-random matrix.

    Returns:
        (np.ndarray): Quasi-random matrix with shape (dim_out, dim_in).
    """
    # Parameters for quasi random numbers generation.
    QUASI_MC_SKIP = 1000
    QUASI_MC_LEAP = 100

    # Implementation of Box-Muller method for converting
    # uniform random sequence to normal random sequence.
    def box_muller_method(xs, ys):
        zs1 = np.sqrt(-2 * np.log(xs)) * np.cos(2 * np.pi * ys)
        zs2 = np.sqrt(-2 * np.log(xs)) * np.sin(2 * np.pi * ys)
        return np.array([zs1, zs2])

    # PyTorch is necessary for quasi-random numbers.
    import torch

    # Generate sobol sequence engine and throw away the first several values.
    sobol = torch.quasirandom.SobolEngine(dim_in, scramble=True)
    sobol.fast_forward(QUASI_MC_SKIP)

    # Generate uniform random matrix.
    W = np.zeros((dim_in, dim_out))
    for index in range(dim_out):
        W[:, index] = sobol.draw(1).numpy()
        sobol.fast_forward(QUASI_MC_LEAP)

    # Convert the uniform random matrix to normal random matrix.
    for index in range(0, dim_out, 2):
        W[:, index:index + 2] = box_muller_method(W[:, index], W[:, index + 1]).T

    return std * W


def get_matrix_generator(rand_type, std, dim_kernel, dist=None):
    """
    This function returns a function which generate RFF/ORF matrix.
    The usage of the returned value of this function are:
        f(dim_input:int) -> np.array with shape (dim_input, dim_kernel)
    """
    if rand_type == "rff":
        return functools.partial(get_rff_matrix, std=std, dim_out=dim_kernel)
    elif rand_type == "orf":
        return functools.partial(get_orf_matrix, std=std, dim_out=dim_kernel)
    elif rand_type == "qrf":
        return functools.partial(get_qrf_matrix, std=std, dim_out=dim_kernel)
    elif rand_type == "cus":
        return functools.partial(get_custom_matrix, dim_out=dim_kernel)
    elif rand_type == "kron":
        return functools.partial(get_kron_matrix, dim_out=dim_kernel)
    else:
        raise RuntimeError("matrix_generator: 'rand_type' must be 'rff', 'orf', or 'qrf' or 'cus', 'kron'.")


class Base:
    """
    Base class of the following RFF/ORF related classes.
    """

    def __init__(self, rand_type, dim_kernel, std_kernel, W, b, dist=None):
        """
        Constractor of the Base class.
        Create random matrix generator and random matrix instance.

        Args:
            rand_type  (str)       : Type of random matrix ("rff", "orf", "qrf", etc).
            dim_kernel (int)       : Dimension of the random matrix.
            std_kernel (float)     : Standard deviation of the random matrix.
            W          (np.ndarray): Random matrix for the input `X`. If None then generated automatically.
            b          (np.ndarray): Random bias for the input `X`. If None then generated automatically.

        Notes:
            If `W` is None then the appropriate matrix will be set just before the training.
        """
        self.dim = dim_kernel

        self.dist = dist
        self.s_k = std_kernel
        self.rand_type = rand_type
        self.mat = get_matrix_generator(rand_type, std_kernel, dim_kernel, dist)
        self.W = W


    def conv(self, X, index=None):
        """
        Applies random matrix to the given input vectors `X` and create feature vectors.

        Args:
            X     (np.ndarray): Input matrix with shape (n_samples, n_features).
            index (int)       : Index of the random matrix. This value should be specified only
                                when multiple random matrices are used.

        Notes:
            This function can manipulate multiple random matrix. If argument 'index' is given,
            then use self.W[index] as a random matrix, otherwise use self.W itself.
            Also, computation of `ts` is equivarent with ts = X @ W, however, for reducing memory 
            consumption, split X to smaller matrices and concatenate after multiplication wit W.
        """
        W = self.W if index is None else self.W[index]
        return np.concatenate((np.cos(X @ W ), np.sin(X @ W)), axis=1)

    def set_weight(self, dim_in):
        """
        Set the appropriate random matrix to 'self.W' if 'self.W' is None (i.e. empty).

        Args:
            dim_in (int): Input dimension of the random matrix.

        Notes:
            This function can manipulate multiple random matrix. If argument 'dim_in' is
            a list/tuple of integers, then generate multiple random matrixes.
        """
        # Generate matrix W.
        if self.W is not None:
            pass
        elif hasattr(dim_in, "__iter__"):
            self.W = tuple([self.mat(dim_in[i], self.dist[i]) for i in range(len(dim_in))])
        else:
            self.W = self.mat(dim_in, dist=self.dist)

        # Generate vector b.
        #print(self.W.shape)
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
