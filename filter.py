
import numpy as np
from scipy import signal


def bandpass_filter(rate=None, low=None, high=None, order=None):
    assert low < high
    assert order >= 1
    return signal.butter(order,
                         (low / (rate / 2.), high / (rate / 2.)),
                         'pass')

def apply_filter(x, filter=None, axis=0):
    """Apply a filter to an array."""
    if isinstance(x, list):
        x = np.asarray(x)
    if x.shape[axis] == 0:
        return x
    b, a = filter
    return signal.filtfilt(b, a, x[:], axis=axis)

class Filter(object):

    def __init__(self, rate=None, low=None, high=None, order=None):
        self._filter = bandpass_filter(rate=rate,
                                       low=low,
                                       high=high,
                                       order=order,
                                       )

    def __call__(self, data):
        return apply_filter(data, filter=self._filter)

class Whitening(object):
    """Compute a whitening matrix and apply it to data.

    Contributed by Pierre Yger.

    """
    def fit(self, x, fudge=1e-18):
        """Compute the whitening matrix.

        Parameters
        ----------

        x : array
            An `(n_samples, n_channels)` array.

        """
        assert x.ndim == 2
        ns, nc = x.shape
        x_cov = np.cov(x, rowvar=0)
        assert x_cov.shape == (nc, nc)
        d, v = np.linalg.eigh(x_cov)
        d = np.diag(1. / np.sqrt(d + fudge))
        # This is equivalent, but seems much slower...
        # w = np.einsum('il,lk,jk->ij', v, d, v)
        w = np.dot(np.dot(v, d), v.T)
        self._matrix = w
        return w

    def transform(self, x):
        """Whiten some data.

        Parameters
        ----------

        x : array
            An `(n_samples, n_channels)` array.

        """
        return np.dot(x, self._matrix)
