#!/usr/bin/env python3
#
# Python module of Gaussian process with random matrix for CPU.
##################################################### SOURCE START #####################################################

import numpy as np
import sklearn.metrics

from .rfflearn_cpu_common import Base

### Gaussian Process Regression with random matrix (RFF/ORF).
class GPR(Base):

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, std_error = 0.1, W = None, b = None, a = None, S = None):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, W, b)
        self.s_e = std_error
        self.a   = a
        self.S   = S

    ### Run training. The interface of this function imitate the interface of
    ### the 'sklearn.gaussian_process.GaussianProcessRegressor.fit'.
    def fit(self, X, y, **args):
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        P = F @ F.T
        I = np.eye(self.dim)
        s = self.s_e**2
        M = I - np.linalg.solve((P + s * I), P)
        self.a = (y.T @ F.T) @ M / s
        self.S = I - P @ M / s
        return self

    ### Run prediction. The interface of this function imitate the interface of
    ### the 'sklearn.gaussian_process.GaussianProcessRegressor.predict'.
    ### If shape of the vector p is (*, 1), then reshape to (*, ).
    def predict(self, X, return_std = False, return_cov = False):
        self.set_weight(X.shape[1])
        F = self.conv(X).T
        p = np.array(self.a.dot(F)).T
        p = np.squeeze(p, axis = 1) if len(p.shape) > 1 and p.shape[1] == 1 else p
        if return_std and return_cov: return [p, self.std(F), self.cov(F)]
        elif return_std             : return [p, self.std(F)]
        elif return_cov             : return [p, self.cov(F)]
        else                        : return  p

    ### Return predicted standard deviation.
    def std(self, F):
        clip_flt = lambda x: max(0.0, float(x))
        pred_var = [clip_flt(F[:, n].T @ self.S @ F[:, n]) for n in range(F.shape[1])]
        return np.sqrt(np.array(pred_var))

    ### Return predicted covariance.
    def cov(self, F):
        return F.T @ self.S @ F

    ### Return score.
    def score(self, X, y, **args):
        self.set_weight(X.shape[1])
        return sklearn.metrics.r2_score(y, self.predict(X))

### Gaussian Process Classification with random matrix (RFF/ORF).
class GPC(GPR):

    ### RFFGPC is essentially the same as RFFGPR, but some pre-processing and post-processing are necessary.
    ### The required processings are:
    ###   - Assumed input label is a vector of class indexes, but the input of
    ###     the RFFGPR should be a one hot vector of the class indexes.
    ###   - Output of the RFFGPR is log-prob, not predicted class indexes.
    ### The purpouse of this RFFGPC class is only to do these pre/post-processings.

    ### Constractor. Save hyperparameters as member variables.
    def __init__(self, rand_mat_type, dim_kernel = 128, std_kernel = 0.1, std_error = 0.1, W = None, b = None, a = None, S = None):
        super().__init__(rand_mat_type, dim_kernel, std_kernel, std_error, W, b, a, S)

    def fit(self, Xs, ys):
        ys_onehot = np.eye(int(np.max(ys) + 1))[ys]
        return super().fit(Xs, ys_onehot)

    ### Inference of Gaussian process.
    ###   - Xs         (np.array, shape = [N, K]): inference data
    ###   - return_std (boolean, scalar)         : return standard deviation vector if true
    ###   - return_cov (boolean, scalar)         : return covariance matrix if true
    ### where N is the number of training data and K is dimension of the input data.
    def predict(self, Xs, return_std = False, return_cov = False):

        ### Run GPC prediction. Note that the returned value is one-hot vector.
        res = super().predict(Xs, return_std, return_cov)

        ### Convert one-hot vector to class index.
        if return_std or return_cov: res[0] = np.argmax(res[0], axis = 1)
        else                       : res    = np.argmax(res,    axis = 1)

        return res

    ### Returns classification accuracy.
    def score(self, Xs, ys):
        return np.mean(self.predict(Xs) == ys)

### The above functions/classes are not visible from users of this library,
### becasue of the complicated usage. The following classes are simplified
### version of the classes. These classes are visible from users.

### Gaussian process regression with RFF.
class RFFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Gaussian process regression with ORF.
class ORFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

### Gaussian process regression with QRF.
class QRFGPR(GPR):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

### Gaussian process classifier with RFF.
class RFFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("rff", *pargs, **kwargs)

### Gaussian process classifier with ORF.
class ORFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("orf", *pargs, **kwargs)

### Gaussian process classifier with QRF.
class QRFGPC(GPC):
    def __init__(self, *pargs, **kwargs):
        super().__init__("qrf", *pargs, **kwargs)

##################################################### SOURCE FINISH ####################################################
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
