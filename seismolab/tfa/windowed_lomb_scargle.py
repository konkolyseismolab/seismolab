import numpy as np
from astropy.timeseries import LombScargle
from astropy.modeling import models
import warnings

__all__ = ['windowed_lomb_scargle']

def windowed_lomb_scargle(time,brightness,
                          minimum_frequency=None,
                          maximum_frequency=None,
                          nyquist_factor=1,
                          samples_per_peak=10,
                          Ntimes=100,
                          sigma=0.5
                          ):
    """
    Calculates the windowed (short-term) Lomb-Scargle transform.

    Parameters
    ----------
    time : array-like
        Time values of the light curve.
    brightness : array-like
        Brightness values of the light curve.
    minimum_frequency : float, optional
        If specified, then use this minimum frequency rather than one chosen based on the size
        of the baseline.
    maximum_frequency : float, optional
        If specified, then use this maximum frequency rather than one chosen based on the average
        nyquist frequency.
    nyquist_factor : float, default: 1
        The multiple of the average nyquist frequency used to choose the maximum frequency
        if ``maximum_frequency`` is not provided.
    samples_per_peak:  float, default: 10
        The approximate number of desired samples across the typical frequency peak.
    Ntimes: int, default: 100
        The number of times points to generate a uniformly sampled time grid.
    sigma: float, default: 0.5
        The width of the Gaussian analyzing window.

    Returns
    -------
    t_grid : array-like
        Time grid.
    nu_grid : array-like
        Frequency grid.
    powers : array-like
        Windowed Lomb-Scargle transform
        at the time-frequency grid points.
    """

    magcorr = brightness-brightness.mean()

    powers = []
    t_grid = np.linspace(time.min(),time.max(), Ntimes )

    for midtime in t_grid:
        g = models.Gaussian1D(amplitude=1, mean=midtime, stddev=sigma)
        ls = LombScargle(time,magcorr*g(time))
        freq, power = ls.autopower(maximum_frequency=maximum_frequency,
                                    minimum_frequency=minimum_frequency,
                                    nyquist_factor=nyquist_factor,
                                    normalization='psd',
                                    samples_per_peak=samples_per_peak)

        norm = np.sqrt(np.sum( magcorr*g(time) > 1e-04 ))
        with warnings.catch_warnings(record=True):
            power = 2*np.sqrt(power)/norm
        powers.append(power)

    powers = np.array(powers)

    t_grid,nu_grid = np.meshgrid(t_grid,freq)

    return t_grid,nu_grid,powers
