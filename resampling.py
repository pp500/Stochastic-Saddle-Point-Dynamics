

from __future__ import division, print_function

import numpy as np
from numpy import random

def exp_and_normalise(lw):
    """Exponentiate, then normalise (so that sum equals one).

    Arguments
    ---------
    lw: ndarray
        log weights.

    Returns
    -------
    W: ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))

    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max
    before exponentiating)

    See also
    --------
    log_sum_exp
    log_mean_exp

    """
    w = np.exp(lw - lw.max())
    return w / w.sum()


def essl(lw):
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw: (N,) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = np.exp(lw - lw.max())
    return (w.sum())**2 / np.sum(w**2)

class Weights(object):
    """ A class to store N log-weights, and automatically compute normalised
    weights and their ESS.

    Parameters
    ----------
    lw: (N,) array or None
        log-weights (if None, object represents a set of equal weights)

    Attributes
    ----------
    lw: (N), array
        log-weights (un-normalised)
    W: (N,) array
        normalised weights
    ESS: scalar
        the ESS (effective sample size) of the weights

    Warning
    -------
    Objects of this class should be considered as immutable; in particular,
    method add returns a *new* object. Trying to modifying directly the
    log-weights may introduce bugs.

    """

    def __init__(self, lw=None):
        self.lw = lw
        if lw is not None:
            self.lw[np.isnan(self.lw)] = - np.inf
            self.W = exp_and_normalise(lw)
            self.ESS  = 1. / np.sum(self.W ** 2)

    def add(self, delta):
        """Increment weights: lw <-lw + delta.

        Parameters
        ----------
        delta: (N,) array
            incremental log-weights

        """
        if self.lw is None:
            return Weights(lw=delta)
        else:
            return Weights(lw=self.lw + delta)

def log_sum_exp(v):
    """Log of the sum of the exp of the arguments.

    Parameters
    ----------
    v: ndarray

    Returns
    -------
    l: float
        l = log(sum(exp(v)))

    Note
    ----
    use the log_sum_exp trick to avoid overflow: i.e. we remove the max of v
    before exponentiating, then we add it back

    See also
    --------
    log_mean_exp

    """
    m = v.max()
    return m + np.log(np.sum(np.exp(v - m)))






#@jit(nopython=True)
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.

        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    N=  W.shape[0]
#    print(N)
    A = np.empty(M, np.int32)
    fail=False
    for n in range(M):
        while su[n] > s:
            j += 1
#            j=np.minimum(j,N-1) #hack
            if j>N-1:
                fail=True
#                print("Warning: j>N",j,N)
                j=N-1
#            print(j)
            s += W[j]
        A[n] = j
    return A,fail


def stratified_sample(W, M):
    su = (random.rand(M) + np.arange(M)) / M
    A,fail=inverse_cdf(su, W)
    return A,fail
#inverse_cdf(su, W)


#rs_index,fail=rs.resampling("stratified",wgts.W,N_p)
#      if fail is True:
#          f=f+1
#          print(f," Try again")
#          wgts = rs.Weights(lw=wgts_lw[:,-1])
#          rs_index,fail=rs.resampling("stratified",wgts.W,N_p)
#          if fail is True:
#              f=f+1
#              print(f," Failed again")
#              rs_index[:]=prtcl_index[:]

