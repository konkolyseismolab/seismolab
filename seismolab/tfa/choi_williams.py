import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tqdm.auto import tqdm
from joblib import delayed, parallel_backend

from .tools import proper_round
from .tools import ProgressParallel
from multiprocessing import cpu_count

__all__ = ['choi_williams']

def choi_williams(time, brightness,
                    minimum_frequency=None,
                    maximum_frequency=None,
                    nyquist_factor=1.,
                    samples_per_peak=1,
                    Ntimes=100,
                    sigma = 1.,
                    M = 128,
                    max_gap_size = 0.5,
                    ncores=-1
                ):
    """
    Calculates the Choi-Williams transform.

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
        The width of the kernel of the analyzing window.
        More or less this controls the resolution in time.
    M: int, default: 128
        This controls the length of the temporal window, which
        is given by M times the sampling time.
        More or less this controls the resolution in frequency.
    max_gap_size : float, default: 0.5
        Maximal size of gaps which is used to split the time series into chunks.
    ncores: int, default: -1
        Number of CPU cores to be used for parallel error estimation.
        If `-1`, then all available cores will be used.

    Returns
    -------
    t_grid : array-like
        Time grid.
    nu_grid : array-like
        Frequency grid.
    Ctnu : array-like
        Choi-Williams transform at the time-frequency grid points.
    """

    if M//2 > len(time):
        raise ValueError("Temporal window length is too large!\n" + \
            "Lower the value of M!" )

    ncores = int(ncores)
    if ncores < 1:
        ncores = cpu_count()

    if ncores == 1:
        t_grid,nu_grid,Ctnu = choi_williams_single(time,brightness,
                                        sigma=sigma,
                                        M=M,
                                        nyquist_factor=nyquist_factor,
                                        samples_per_peak=samples_per_peak,
                                        minimum_frequency=minimum_frequency,
                                        maximum_frequency=maximum_frequency,
                                        Ntimes=Ntimes,
                                        max_gap_size=max_gap_size)

    else:
        t_grid,nu_grid,Ctnu = choi_williams_parallel(time,brightness,
                                            sigma=sigma,
                                            M=M,
                                            nyquist_factor=nyquist_factor,
                                            samples_per_peak=samples_per_peak,
                                            minimum_frequency=minimum_frequency,
                                            maximum_frequency=maximum_frequency,
                                            Ntimes=Ntimes,
                                            max_gap_size=max_gap_size,
                                            ncores=ncores)

    return t_grid,nu_grid,Ctnu

def _choi_kernel(t,nu,time,magconj,taustep,sigma):
    dtausum =  sliding_window_view(magconj,len(time))[:len(taustep)] * sliding_window_view(magconj[::-1],len(time))[:len(taustep),::-1]
    dtausum *= np.exp(-sigma*( (time-t)+1e-8 )**2/taustep**2)
    dtausum *= (taustep**2*sigma)**(-0.5)

    dtausum =  dtausum * np.exp(-1j * 2*np.pi * taustep * nu)

    dtausum = np.nansum(dtausum)

    dtausum = np.abs( 1/(2 * np.pi**0.5) * dtausum  )
    dtausum = 2*dtausum / len(time)

    return dtausum

def choi_williams_parallel(time,mag,
                        sigma = 1.,
                        M = 128,
                        minimum_frequency=None,
                        maximum_frequency=None,
                        nyquist_factor = 1,
                        samples_per_peak = 1,
                        Ntimes  = 100,
                        max_gap_size = 0.5,
                        ncores = -1
                        ):

    # --- Choi-Williams stars here ---
    M = int(M) # because tau/2 is needed in ff^* correlation

    sampling_time = np.median(np.diff(time))
    temporal_window_length = M * sampling_time

    minfreq = 1/time.ptp()
    maxfreq = 0.5/sampling_time * nyquist_factor

    if minimum_frequency is not None:
        minfreq = float(minimum_frequency)
    if maximum_frequency is not None:
        maxfreq = float(maximum_frequency)

    maxfreq = maxfreq/2 # Due to sampling effect

    Nfreqs = int(maxfreq/(1/time.ptp())*samples_per_peak)
    nu_grid = np.linspace(minfreq,maxfreq,Nfreqs)

    t_grid = None

    gapat = np.where(np.diff(time) > max_gap_size)[0]
    gapat = np.repeat(gapat,2)
    gapat[1::2] += 1
    gapat = np.r_[0,gapat,len(time)-1]

    fullT = 0
    for tstart,tend in zip(time[gapat][::2],time[gapat][1::2]):
        dt = tend-tstart
        fullT += dt

    for tstart,tend in zip(time[gapat][::2],time[gapat][1::2]):
        dt = tend-tstart
        Nt = proper_round(dt/fullT * Ntimes)

        t_grid = np.r_[t_grid, np.linspace(tstart,tend,Nt)]

    t_grid = t_grid[1:].astype(float)
    Ntimes = len(t_grid)

    magconj = mag - np.nanmean(mag)
    magconj = np.r_[magconj,magconj]

    # u == time points
    # tau = kernel FWHM

    taustep = np.linspace(1e-10,temporal_window_length,M//2)
    taustep = taustep[:,np.newaxis]

    ncores = int(ncores)
    if ncores < 1:
        ncores = cpu_count()//2
        threads = 2
    else:
        maxcores = cpu_count()
        threads = maxcores//ncores

    with parallel_backend("loky", inner_max_num_threads=threads):
        Ctnu = ProgressParallel(n_jobs=ncores,total=len(t_grid)*len(nu_grid))(delayed(_choi_kernel)(t,nu,time,magconj,taustep,sigma) for t in t_grid for nu in nu_grid)

    Ctnu = np.asarray(Ctnu).reshape(len(t_grid),len(nu_grid))

    return t_grid,nu_grid,Ctnu


def choi_williams_single(time,mag,
                        sigma = 1.,
                        M = 128,
                        minimum_frequency=None,
                        maximum_frequency=None,
                        nyquist_factor = 1,
                        samples_per_peak = 1,
                        Ntimes  = 100,
                        max_gap_size = 0.5
                        ):

    # --- Choi-Williams stars here ---
    M = int(M) # because tau/2 is needed in ff^* correlation

    sampling_time = np.median(np.diff(time))
    temporal_window_length = M * sampling_time

    minfreq = 1/time.ptp()
    maxfreq = 0.5/sampling_time * nyquist_factor

    if minimum_frequency is not None:
        minfreq = float(minimum_frequency)
    if maximum_frequency is not None:
        maxfreq = float(maximum_frequency)

    maxfreq = maxfreq/2 # Due to sampling effect

    Nfreqs = int(maxfreq/(1/time.ptp())*samples_per_peak)
    nu_grid = np.linspace(minfreq,maxfreq,Nfreqs)

    t_grid = None

    gapat = np.where(np.diff(time) > max_gap_size)[0]
    gapat = np.repeat(gapat,2)
    gapat[1::2] += 1
    gapat = np.r_[0,gapat,len(time)-1]

    fullT = 0
    for tstart,tend in zip(time[gapat][::2],time[gapat][1::2]):
        dt = tend-tstart
        fullT += dt

    for tstart,tend in zip(time[gapat][::2],time[gapat][1::2]):
        dt = tend-tstart
        Nt = proper_round(dt/fullT * Ntimes)

        t_grid = np.r_[t_grid, np.linspace(tstart,tend,Nt)]

    t_grid = t_grid[1:].astype(float)
    Ntimes = len(t_grid)

    magconj = mag - np.nanmean(mag)
    magconj = np.r_[magconj,magconj]

    # u == time points
    # tau = kernel FWHM

    taustep = np.linspace(1e-10,temporal_window_length,M//2)
    taustep = taustep[:,np.newaxis]

    Ctnu = np.empty((Ntimes,Nfreqs))

    for ii,t in tqdm(enumerate(t_grid),total=len(t_grid)):
        for jj,nu in enumerate(nu_grid):

            dtausum = sliding_window_view(magconj,len(time))[:len(taustep)] * sliding_window_view(magconj[::-1],len(time))[:len(taustep),::-1]
            dtausum *=  np.exp(-sigma*( (time-t)+1e-8 )**2/(4*taustep**2) )
            dtausum *=  (taustep**2*sigma)**(-0.5)

            dtausum =  dtausum * np.exp(-1j * 2*np.pi * taustep * nu)

            dtausum = np.nansum(dtausum)

            Ctnu[ii,jj] = np.abs( 1/(2 * np.pi**0.5) * dtausum  )
            Ctnu[ii,jj] = 2*Ctnu[ii,jj] / len(time)

    return t_grid,nu_grid,Ctnu
