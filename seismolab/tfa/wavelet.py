import numpy as np

from tqdm.auto import tqdm
from joblib import delayed
from .tools import ProgressParallel
from multiprocessing import cpu_count

__all__ = ['wavelet']

def wavelet(time, brightness,
            minimum_frequency=None,
            maximum_frequency=None,
            nyquist_factor=1.,
            samples_per_peak=10,

            Ntimes=100,
            c=2*np.pi,
            ncores=-1
        ):
    """
    Calculates the wavelet transform wit Morlet kernel.

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
    c: float, default: 2*pi
        The scale parameter.
        The ratio of the time and frequency resolution.
    ncores: int, default: -1
        Number of CPU cores to be used for parallel error estimation.
        If `-1`, then all available cores will be used.

    Returns
    -------
    t_grid : array-like
        Time grid.
    nu_grid : array-like
        Frequency grid.
    morlet : array-like
        Morlet wavelet transform at the time-frequency grid points.
    """

    ncores = int(ncores)
    if ncores < 1:
        ncores = cpu_count()

    if ncores == 1:
        t_grid,nu_grid,morlet = wavelet_single(time,brightness,
                                        nyquist_factor=nyquist_factor,
                                        samples_per_peak=samples_per_peak,
                                        minimum_frequency=minimum_frequency,
                                        maximum_frequency=maximum_frequency,
                                        Ntimes=Ntimes,
                                        c=c)
    else:
        t_grid,nu_grid,morlet = wavelet_parallel(time,brightness,
                                            nyquist_factor=nyquist_factor,
                                            samples_per_peak=samples_per_peak,
                                            minimum_frequency=minimum_frequency,
                                            maximum_frequency=maximum_frequency,
                                            Ntimes=Ntimes,
                                            c=c,
                                            ncores=ncores)

    return t_grid,nu_grid,morlet

def _g(x,c):
    return np.exp( -x**2/2 + 1j*c*x )

def _wavelet_kernel(t,time,magcorr,c,nu_grid):
    a = c/(2*np.pi*nu_grid)

    Ttnu = (a**(-0.5)).reshape(-1) * np.nansum( magcorr * np.conj(_g((time-t)/a,c)) ,axis=1)

    Ttnu = (a**(-0.5)).reshape(-1) * Ttnu

    return np.abs(Ttnu).reshape(-1)

def wavelet_parallel(time,mag,
                    minimum_frequency=None,
                    maximum_frequency=None,
                    nyquist_factor = 1,
                    samples_per_peak = 10,
                    Ntimes = 100,
                    c = 2*np.pi,
                    ncores = -1
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

    morlet = ProgressParallel(n_jobs=ncores,total=len(t_grid))(delayed(_wavelet_kernel)(t,time,magcorr,c,nu_grid) for t in t_grid)

    morlet = np.asarray(morlet)

    morlet = 2*morlet/len(time)

    return t_grid,nu_grid,morlet

def wavelet_single(time,mag,
                    minimum_frequency=None,
                    maximum_frequency=None,
                    nyquist_factor = 1,
                    samples_per_peak = 10,
                    Ntimes = 100,
                    c = 2*np.pi
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

    g = lambda x : np.exp( -x**2/2 + 1j*c*x )

    morlet = np.empty((Ntimes,Nfreqs))

    for ii,t in tqdm(enumerate(t_grid),total=len(t_grid)):

        a = c/(2*np.pi*nu_grid)

        Stanu = (a**(-0.5)).reshape(-1) * np.nansum( magcorr * np.conj(g((time-t)/a)) ,axis=1)

        Ttnu = (a**(-0.5)).reshape(-1) * Stanu
        morlet[ii,:] = np.abs(Ttnu).reshape(-1)

    morlet = 2*morlet/len(time)

    return t_grid,nu_grid,morlet
