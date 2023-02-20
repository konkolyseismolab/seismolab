import numpy as np

from tqdm.auto import tqdm
from joblib import delayed

from .tools import ProgressParallel
from multiprocessing import cpu_count

__all__ = ['gabor']

def gabor(time, brightness,
            minimum_frequency=None,
            maximum_frequency=None,
            nyquist_factor=1.,
            samples_per_peak=10,
            Ntimes=100,
            sigma=0.5,
            ncores=-1
        ):
    """
    Calculates the Gabor transform.

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
    ncores: int, default: -1
        Number of CPU cores to be used for parallel error estimation.
        If `-1`, then all available cores will be used.

    Returns
    -------
    t_grid : array-like
        Time grid.
    nu_grid : array-like
        Frequency grid.
    stFT : array-like
        Gabor transform at the time-frequency grid points.
    """

    ncores = int(ncores)
    if ncores < 1:
        ncores = cpu_count()

    if ncores == 1:
        t_grid,nu_grid,stFT = gabor_single(time,brightness,
                                        nyquist_factor=nyquist_factor,
                                        samples_per_peak=samples_per_peak,
                                        minimum_frequency=minimum_frequency,
                                        maximum_frequency=maximum_frequency,
                                        Ntimes=Ntimes,
                                        sigma=sigma)
    else:
        t_grid,nu_grid,stFT = gabor_parallel(time,brightness,
                                            nyquist_factor=nyquist_factor,
                                            samples_per_peak=samples_per_peak,
                                            minimum_frequency=minimum_frequency,
                                            maximum_frequency=maximum_frequency,
                                            Ntimes=Ntimes,
                                            sigma=sigma,
                                            ncores=ncores)

    return t_grid,nu_grid,stFT


def _h(t,sigma):
    return np.exp(-t**2 /(2*sigma**2) )

def _gabor_kernel(t,magcorr,time,nu_grid,sigma):
    Ftnu = magcorr * np.conj(_h(time-t,sigma)) * np.exp(-1j * 2*np.pi * time * nu_grid)
    Ftnu = np.nansum(Ftnu,axis=1)

    return np.abs(Ftnu)

def gabor_parallel(time,mag,
            nyquist_factor=1.,
            samples_per_peak=10,
            minimum_frequency=None,
            maximum_frequency=None,
            Ntimes=100,
            sigma=0.5,
            ncores=1
            ):

    sampling_time = np.median(np.diff(time))
    maxfreq = 0.5/sampling_time * nyquist_factor

    minfreq = 1/time.ptp()

    if minimum_frequency is not None:
        minfreq = float(minimum_frequency)
    if maximum_frequency is not None:
        maxfreq = float(maximum_frequency)

    Nfreqs = int(maxfreq/(1/time.ptp())*samples_per_peak)
    nu_grid = np.linspace(minfreq,maxfreq,Nfreqs)
    nu_grid = nu_grid[:,np.newaxis]

    t_grid = np.linspace(time.min(),time.max(),Ntimes)

    magcorr = mag-np.nanmean(mag)

    stFT = ProgressParallel(n_jobs=ncores,total=len(t_grid))(delayed(_gabor_kernel)(t,magcorr,time,nu_grid,sigma) for t in t_grid)

    stFT = np.asarray(stFT)

    stFT = 2*stFT/len(time)

    return t_grid,nu_grid,stFT


def gabor_single(time,mag,
            nyquist_factor=1.,
            samples_per_peak=10,
            minimum_frequency=None,
            maximum_frequency=None,
            Ntimes=100,
            sigma=0.5,
            ):

    sampling_time = np.median(np.diff(time))
    maxfreq = 0.5/sampling_time * nyquist_factor

    minfreq = 1/time.ptp()

    if minimum_frequency is not None:
        minfreq = float(minimum_frequency)
    if maximum_frequency is not None:
        maxfreq = float(maximum_frequency)

    Nfreqs = int(maxfreq/(1/time.ptp())*samples_per_peak)
    nu_grid = np.linspace(minfreq,maxfreq,Nfreqs)
    nu_grid = nu_grid[:,np.newaxis]

    t_grid = np.linspace(time.min(),time.max(),Ntimes)

    magcorr = mag-np.nanmean(mag)

    h = lambda t : np.exp(-t**2 /(2*sigma**2) )

    stFT = np.empty((Ntimes,Nfreqs))

    for ii,t in tqdm(enumerate(t_grid),total=len(t_grid)):

        Ftnu = magcorr * np.conj(h(time-t)) * np.exp(-1j * 2*np.pi * time * nu_grid)
        Ftnu = np.nansum(Ftnu,axis=1)

        stFT[ii,:] = np.abs(Ftnu)

    stFT = 2*stFT/len(time)

    return t_grid,nu_grid,stFT
