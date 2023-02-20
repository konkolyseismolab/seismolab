import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from warnings import warn
from uncertainties import ufloat
import corner
import multiprocessing
from joblib import delayed
import joblib
from tqdm.auto import tqdm
from scipy import stats

__all__ = ['Fourier','MultiHarmonicFitter','MultiFrequencyFitter']

class ProgressParallel(joblib.Parallel):
    def __init__(self, total=None, **kwds):
        self.total = total
        super().__init__(**kwds)
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self._pbar.total = self.n_dispatched_tasks
        else:
            self._pbar.total = self.total
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

class BaseFitter():
    '''

    Attributes
    ----------
    t : array-like
        Time values of the light curve.
    y : array-like
        Flux/mag values of the light curve.
    error : array-like, optional
        Flux/mag errors values of the light curve. If not given, Fourier parameter
        errors will be less reliable. In this case use `error_estimation`.
    '''
    def __init__(self, t, y, error=None):
        t = np.asarray(t,dtype=float)
        y = np.asarray(y,dtype=float)
        if error is not None:
            error = np.asarray(error,dtype=float)

        goodpts = np.isfinite(t)
        goodpts &= np.isfinite(y)
        if error is not None:
            goodpts &= np.isfinite(error)

        self.t = t[goodpts]
        self.y = y[goodpts]
        if error is not None:
            self.error = error[goodpts]
        else:
            self.error = error

    def _func(self, time, amp, best_freq, phase, kind='sin'):
        # sin or cos funtion to be fitted
        if kind=='sin':
            y = amp*np.sin(2*np.pi*best_freq*time + phase )
        elif kind=='cos':
            y = amp*np.cos(2*np.pi*best_freq*time + phase )
        else:
            raise TypeError('%s format does not exist. Select \'sin\' or \'cos\'.' % str(kind))
        return y

    def _analytic_uncertainties(self,time,residual,amp):
        N = len(time)
        T = np.ptp(time)
        sigma_m = np.std(residual)

        sigma_f = np.sqrt(6/N) * 1/(np.pi*T) * sigma_m/amp
        sigma_a = np.sqrt(2/N) * sigma_m
        sigma_phi = 1/(2*np.pi) * np.sqrt(2/N) * sigma_m/amp

        return sigma_f,sigma_a,sigma_phi

class Fourier(BaseFitter):
    '''

    Attributes
    ----------
    t : array-like
        Time values of the light curve.
    y : array-like
        Flux/mag values of the light curve.
    error : array-like, optional
        Flux/mag errors values of the light curve. If not given, Fourier parameter
        errors will be less reliable. In this case use `error_estimation`.
    '''

    def spectral_window(self,
                        minimum_frequency=None,
                        maximum_frequency=None,
                        nyquist_factor=1,
                        samples_per_peak=10,
                        plotting=False):
        """
        Calculates the spectral window.

        Parameters
        ----------
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
        plotting: bool, default: False
            If `True`, spectral window will be displayed.

        Returns
        -------
        freq : array-like
            Frequency grid.
        sw : array-like
            Spectral window at given frequencies.
        """

        ls = LombScargle(self.t, self.y)
        lsf = ls.autofrequency( samples_per_peak=samples_per_peak,
                                nyquist_factor=nyquist_factor,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency)

        costerm = np.cos(2*np.pi*lsf[:,np.newaxis]*self.t).sum(axis=1)
        sinterm = np.sin(2*np.pi*lsf[:,np.newaxis]*self.t).sum(axis=1)

        sw = np.sqrt(costerm**2 + sinterm**2)/len(self.t)

        if plotting:
            fig = plt.figure(figsize=(15,3))
            plt.plot(lsf, sw)
            plt.xlabel('Frequency (c/d)')
            plt.ylabel('Spectral window')
            plt.show()
            plt.close(fig)

        return lsf,sw

    def spectrum(self,
                minimum_frequency=None,
                maximum_frequency=None,
                nyquist_factor=1,
                samples_per_peak=10,
                plotting=False):
        """
        Calculates the classic Fourier spectrum.

        Parameters
        ----------
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
        plotting: bool, default: False
            If `True`, spectrum will be displayed.

        Returns
        -------
        freq : array-like
            Frequency grid.
        spec : array-like
            The Fourier spectrum at given frequencies.
        """

        sampling_time = np.median(np.diff(self.t))
        maxfreq = 0.5/sampling_time * nyquist_factor

        minfreq = 1/self.t.ptp()

        if minimum_frequency is not None:
            minfreq = float(minimum_frequency)
        if maximum_frequency is not None:
            maxfreq = float(maximum_frequency)

        Nfreqs = int(maxfreq/(1/self.t.ptp())*samples_per_peak)
        nu_grid = np.linspace(minfreq,maxfreq,Nfreqs)

        nu_grid = nu_grid[:,np.newaxis]

        magcorr = self.y - np.mean(self.y)

        Ftnu = magcorr  * np.exp(-1j * 2*np.pi * self.t * nu_grid)
        Ftnu = np.nansum(Ftnu,axis=1) / len(self.y)

        FFT = 2*np.abs(Ftnu)

        if plotting:
            fig = plt.figure(figsize=(15,3))
            plt.plot(nu_grid.reshape(-1), FFT)
            plt.xlabel('Frequency (c/d)')
            plt.ylabel('Amplitude')
            plt.show()
            plt.close(fig)

        return nu_grid.reshape(-1), FFT

class MultiHarmonicFitter(BaseFitter):
    '''

    Attributes
    ----------
    t : array-like
        Time values of the light curve.
    y : array-like
        Flux/mag values of the light curve.
    error : array-like, optional
        Flux/mag errors values of the light curve. If not given, Fourier parameter
        errors will be less reliable. In this case use `error_estimation`.
    '''

    def lc_model(self, *arg):
        """
        Get model light curve with all harmonic components at the same time.

        Parameters
        ----------
        time : array
            Desired time points where multi-harmonic component fit is desired.
        arg : arguments
            List of arguments containing the frequency, amplitudes, phases, zero point.

        Returns
        ----------
        y : array
            Multi-harmonic model light curve.
        """
        if not hasattr(self,"kind"):
            warn("Please run \'fit_harmonics\' first!")
            return None

        time = arg[0]
        best_freq = arg[1]
        nparams = (len(arg)-3)//2
        amps = arg[2:2+nparams]
        phases = arg[2+nparams:-1]
        const = arg[-1]

        y = 0
        for i in range(len(amps)):
            y += self._func(time, amps[i], (i+1)*best_freq, phases[i], kind=self.kind)
        y += const
        return y

    def _estimate_errors(self,seed):
        np.random.seed(seed)

        if self.error_estimation == 'bootstrap':
            tmp_lc = self.lc[np.random.choice( self.lc.shape[0], int(len(self.lc)*self.sample_size), replace=False), :]
        elif self.error_estimation == 'montecarlo':
            tmp_lc = self.lc.copy()

            if self.error is None:
                tmp_lc[:,1] += np.random.normal(0,self.yerror,tmp_lc.shape[0])
            else:
                tmp_lc[:,1] += np.random.normal(0,self.error,tmp_lc.shape[0])

        # Subtract mean from time points to decouple frequencies from phases
        tmp_lc[:,0] -= tmp_lc[:,0].mean()

        lbound = [0]*(1+len(self.amps)) + [-np.inf]*len(self.phases) + [-np.inf]
        ubound = [2*self.freqs[0]] + [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
        bounds = (lbound,ubound)

        try:
            if self.error is None:
                tmp_pfit, _ = curve_fit(lambda *args: self.lc_model(*args), tmp_lc[:,0], tmp_lc[:,1],
                                        p0=(self.freqs[0], *self.amps, *self.phases, np.mean(tmp_lc[:,1])),
                                        bounds=bounds, maxfev=5000)
            else:
                tmp_pfit, _ = curve_fit(lambda *args: self.lc_model(*args), tmp_lc[:,0], tmp_lc[:,1],
                                        p0=(self.freqs[0], *self.amps, *self.phases, np.mean(tmp_lc[:,1])) ,
                                        sigma=tmp_lc[:,2], absolute_sigma=self.absolute_sigma, bounds=bounds, maxfev=5000)

            tmp_pfit[1+len(self.amps):-1] = tmp_pfit[1+len(self.amps):-1]%(2*np.pi)
        except RuntimeError:
            tmp_pfit = [np.nan] * (2 + len(self.amps) +len(self.phases) )

        return tmp_pfit

    def fit_harmonics(self,maxharmonics = 3,
                  absolute_sigma=True,
                  plotting = False, scale='flux',
                  minimum_frequency=None, maximum_frequency=None,
                  nyquist_factor=1,samples_per_peak=100,
                  kind='sin',
                  error_estimation='analytic',ntry=1000,
                  sample_size=0.7,
                  parallel=True, ncores=-1,
                  refit=False,
                  best_freq=None):
        """
        ``fit_harmonics`` performs Fourier pre-whitening with harmonic fitting.

        Parameters
        ----------
        maxharmonics : int, default: 3
            The maximum number of harmonics to be fitted. Pass a very large number
            to fit all harmonics, limited by the signal-to-noise ratio.
        absolute_sigma : bool, default: True
            If `True`, error is used in an absolute sense and the estimated parameter covariance
            reflects these absolute values.
        plotting: bool, default: False
            If `True`, fitting steps will be displayed.
        scale: 'mag' or 'flux', default: 'flux'
            Lightcurve plot scale.
        minimum_frequency : float, optional
            If specified, then use this minimum frequency rather than one chosen based on the size
            of the baseline.
        maximum_frequency : float, optional
            If specified, then use this maximum frequency rather than one chosen based on the average
            nyquist frequency.
        nyquist_factor : float, default: 1
            The multiple of the average nyquist frequency used to choose the maximum frequency
            if ``maximum_frequency`` is not provided.
        samples_per_peak:  float, default: 100
            The approximate number of desired samples across the typical frequency peak.
        kind: str, 'sin' or 'cos'
            Harmonic _function to be fitted.
        error_estimation: `analytic`, `bootstrap` or `montecarlo`, default: `analytic`
            If `bootstrap` or `montecarlo` is choosen, boostrap or monte carlo method will be used to estimate parameter uncertainties.
            Otherwise given uncertainties are calculated analytically.
        ntry: int, default: 1000
            Number of resamplings for error estimation.
        sample_size: float, default: 0.7
            The ratio of data points to be used for bootstrap error estimation in each step.
            Applies only if `error_estimation` is set to `bootstrap`.
        parallel: bool, default : True
            If `True`, sampling for error estimation is performed parallel to speed up the process.
        ncores: int, default: -1
            Number of CPU cores to be used for parallel error estimation. If `-1`, then all available
            cores will be used.
        best_freq : float, default: None
            If given, then this frequency will be used as the basis of the harmonics,
            instead of calculating a Lomb-Scargle spectrum to get a frequency.

        Returns
        -------
        pfit : array-like
            Array of fitted parameters. The main frequency, amplitudes and phases of the harmonics,
            and the zero point.
        perr : array-like
            Estimated error of the parameters.
        """
        self.sample_size = sample_size
        self.kind = kind
        self.absolute_sigma = absolute_sigma
        self.error_estimation = error_estimation

        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.nyquist_factor    = nyquist_factor
        self.samples_per_peak  = samples_per_peak

        if minimum_frequency is not None and maximum_frequency is not None:
            if minimum_frequency > maximum_frequency:
                raise ValueError('Minimum frequency is larger than maximum frequency.')

        if maxharmonics<1:
            raise ValueError('Number of frequencies must be >=1.')

        if maximum_frequency is None and nyquist_factor * (0.5/np.median( np.diff(self.t) )) < 2.:
            # Nyquist is low
            warn('Nyquist frequency is low!\nYou might want to set maximum frequency instead.')

        if error_estimation not in ['analytic','bootstrap','montecarlo']:
            raise TypeError('%s method is not supported! Please choose \'analytic\', \'bootstrap\' or \'montecarlo\'.' % str(error_estimation))

        # fit periodic funtions and do prewhitening
        yres = self.y.copy()

        self.freqs = []
        self.amps = []
        self.phases = []
        self.zeropoints = []
        self.freqserr = []
        self.ampserr = []
        self.phaseserr = []
        self.zeropointerr = []

        for i in range(maxharmonics):
            if i == 0:
                if best_freq is None:
                    ls = LombScargle(self.t, yres, nterms=1)
                    freq, power = ls.autopower(normalization='psd',
                                               minimum_frequency=minimum_frequency,
                                               maximum_frequency=maximum_frequency,
                                               samples_per_peak=10,
                                               nyquist_factor=nyquist_factor)

                    freq, power = ls.autopower(normalization='psd',
                                               samples_per_peak=2000,
                                               minimum_frequency=max(1e-10,freq[power.argmax()]-5/self.t.ptp()),
                                               maximum_frequency=freq[power.argmax()]+20/self.t.ptp())

                    # LS may return inf values
                    goodpts = np.isfinite(power)
                    freq  = freq[goodpts]
                    power = power[goodpts]

                # get first spectrum and fit first periodic component
                try:
                    if best_freq is None: best_freq = freq[power.argmax()]

                    pfit1, pcov1 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, best_freq, phase ,kind=kind) + const,
                                            self.t, yres,
                                            p0=(np.ptp(yres)/4,2.,np.mean(yres)), bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                            sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)
                    pfit2, pcov2 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, best_freq, phase ,kind=kind) + const,
                                            self.t, yres,
                                            p0=(np.ptp(yres)/4,5.,np.mean(yres)), bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                            sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

                    chi2_1 = np.sum( (yres-pfit1[-1] - self._func(self.t, pfit1[0], best_freq, pfit1[1] ,kind=kind))**2 )
                    chi2_2 = np.sum( (yres-pfit2[-1] - self._func(self.t, pfit2[0], best_freq, pfit2[1] ,kind=kind))**2 )

                    if chi2_1 < chi2_2:
                        pfit = pfit1
                    else:
                        pfit = pfit2

                    pfit, pcov = curve_fit(lambda time, amp, freq, phase, const: self._func(time, amp, freq, phase, kind=kind) + const,
                                            self.t, yres,
                                            p0=(pfit[0],best_freq,pfit[1],np.mean(yres)),
                                            bounds=([0,0,0,-np.inf], [np.ptp(yres), 2*best_freq, 2*np.pi, np.inf]) ,
                                            sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

                    best_freq = pfit[1]

                except (RuntimeError,ValueError) as err:
                    warn(err)

                    self.pfit = self.perr = [np.nan]*(3+1)
                    return np.asarray(self.pfit), np.asarray(self.perr)

            else:
                # --- Check if frequency is greater than maximum_frequency ---
                if maximum_frequency is not None and (i+1)*best_freq > maximum_frequency:
                    # freq > maximum_frequency
                    warn('Frequency is larger than maximum frequency!\nIncrease maximum frequency to include more peaks!\nSkipping...')
                    break
                elif maximum_frequency is None and (i+1)*best_freq > nyquist_factor * (0.5/np.median( np.diff(self.t) )):
                    # freq > nyquist_factor * Nyquist
                    warn('Frequency is larger than nyquist_factor (%d) x Nyquist frequency!\nSet maximum frequency to avoid problems!\nSkipping...' % int(nyquist_factor))
                    break

                # --- Check if max power is still above the noise level ---
                ls = LombScargle(self.t, yres, nterms=1)
                minf = (i+1)*best_freq - max(0.5,10*1/np.ptp(self.t))
                if minf<0 : minf = 0

                with np.errstate(divide='ignore',invalid='ignore'):
                    freq, power = ls.autopower(normalization='psd',
                                               minimum_frequency=minf,
                                               maximum_frequency=(i+1)*best_freq + max(0.5,10*1/np.ptp(self.t)),
                                               samples_per_peak=samples_per_peak)

                goodpts = np.isfinite(power)
                freq = freq[goodpts]
                power = power[goodpts]
                power[power<0] = 0

                df = 1/np.ptp(self.t)
                umpeak = (freq > (i+1)*best_freq - df) & (freq < (i+1)*best_freq + df)
                if np.nanmax(np.sqrt(power[umpeak])) <= np.nanmean(np.sqrt(power)) + 3*np.nanstd(np.sqrt(power)):
                    # peak height < mean + 3 std
                    break

                # --- Check if best period is longer than 2x data duration ---
                if np.allclose(freq[np.argmax(power)] ,0) or 1./freq[np.argmax(power)] > 2*np.ptp(self.t):
                    warn('Period is longer than 2x data duration!\nSet minimum frequency to avoid problems!\nSkipping...')
                    break

                # --- get ith spectrum and fit ith periodic component ---
                pfit1, pcov1 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, (i+1)*best_freq, phase, kind=kind) + const,
                                            self.t, yres,
                                            p0=(np.ptp(yres)/4,2.,np.mean(yres)),
                                            bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                            sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)
                pfit2, pcov2 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, (i+1)*best_freq, phase, kind=kind) + const,
                                            self.t, yres,
                                            p0=(np.ptp(yres)/4,5.,np.mean(yres)),
                                            bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                            sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

                chi2_1 = np.sum( (yres-pfit1[-1] - self._func(self.t, pfit1[0], (i+1)*best_freq, pfit1[1], kind=kind ))**2 )
                chi2_2 = np.sum( (yres-pfit2[-1] - self._func(self.t, pfit2[0], (i+1)*best_freq, pfit2[1], kind=kind ))**2 )

                if chi2_1 < chi2_2:
                    pfit = pfit1
                    pcov = pcov1
                else:
                    pfit = pfit2
                    pcov = pcov2

            if plotting:
                # plot phased light curve and fit
                if i==0:
                    per = 1/pfit[1]
                else:
                    per = 1/( (i+1)*best_freq )

                # Calculate spectrum for plotting
                ls = LombScargle(self.t, yres, nterms=1)
                freq, power = ls.autopower(normalization='psd',
                                           minimum_frequency=minimum_frequency,
                                           maximum_frequency=maximum_frequency,
                                           samples_per_peak=samples_per_peak,
                                           nyquist_factor=nyquist_factor)

                plt.figure(figsize=(15,3))
                plt.subplot(121)
                plt.plot(freq, 2*np.sqrt(power/len(self.t))  )
                plt.xlabel('Frequency (c/d)')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.subplot(122)
                plt.plot(self.t%per/per,yres-pfit[-1],'k.')
                plt.plot(self.t%per/per+1,yres-pfit[-1],'k.')
                if i == 0:
                    plt.plot(self.t%per/per,self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind),'C1.',ms=1)
                    plt.plot(self.t%per/per+1,self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind),'C1.',ms=1)
                else:
                    plt.plot(self.t%per/per,self._func(self.t,pfit[0],(i+1)*best_freq,pfit[1], kind=kind),'C1.',ms=1)
                    plt.plot(self.t%per/per+1,self._func(self.t,pfit[0],(i+1)*best_freq,pfit[1], kind=kind),'C1.',ms=1)
                if scale == 'mag': plt.gca().invert_yaxis()
                plt.xlabel('Phase (f=%.6f c/d; P=%.5f d)' % (1/per,per))
                plt.ylabel('Brightness')
                plt.show()

            if i==0:
                # start to collect results
                self.freqs.append( pfit[1] )
                self.amps.append( pfit[0] )
                self.phases.append( pfit[2] )
                self.zeropoints.append( pfit[3] )

                pcov = np.sqrt(np.diag(pcov))
                self.freqserr.append( pcov[1] )
                self.ampserr.append( pcov[0] )
                self.phaseserr.append( pcov[2] )
                self.zeropointerr.append( pcov[3] )

                yres -= self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind) + pfit[3]
            else:
                # collect ith results
                self.freqs.append( (i+1)*best_freq )
                self.amps.append( pfit[0] )
                self.phases.append( pfit[1] )
                self.zeropoints.append( pfit[2] )

                pcov = np.sqrt(np.diag(pcov))
                self.freqserr.append( self.freqserr[0] )
                self.ampserr.append( pcov[0] )
                self.phaseserr.append( pcov[1] )
                self.zeropointerr.append( pcov[2] )

                yres -= self._func(self.t,pfit[0],(i+1)*best_freq,pfit[1], kind=kind) + pfit[2]

        try:
            # fit all periodic components at the same time
            lbound = [0]*(1+len(self.amps)) + [-np.inf]*len(self.phases) + [-np.inf]
            ubound = [2*best_freq] + [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
            bounds = (lbound,ubound)
            pfit, pcov = curve_fit(lambda *args: self.lc_model(*args), self.t, self.y,
                                    p0=(self.freqs[0], *self.amps, *self.phases, np.mean(self.y)),
                                    bounds=bounds, sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

            # convert all phases into the 0-2pi range
            pfit[1+len(self.amps):-1] = pfit[1+len(self.amps):-1]%(2*np.pi)

            '''
            # if any of the errors is inf, shift light curve by period/2 and get new errors
            if not error_estimation and np.any(np.isinf(pcov)):
                warn('One of the errors is inf! Shifting light curve by half period to calculate errors...')
                lbound = [0]*(1+len(self.amps)) + [-np.inf]*len(self.phases) + [-np.inf]
                ubound = [2*best_freq] + [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
                bounds = (lbound,ubound)
                _, pcov = curve_fit(lambda *args: self.lc_model(*args), self.t + 0.5/pfit[0], self.y, p0=(self.freqs[0], *self.amps, *[(pha+np.pi)%(2*np.pi) for pha in self.phases], np.mean(self.y)) ,
                                    bounds=bounds, sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)
            '''

            if error_estimation == 'analytic':
                self.pfit = pfit
                self.perr = self.get_analytic_uncertainties() + [ np.sqrt(np.diag(pcov))[-1] ]

                return np.asarray(self.pfit), np.asarray(self.perr)

            elif error_estimation != 'analytic' and refit is False:
                # use bootstrap or MC to get realistic errors (bootstrap = get subsample and redo fit n times)
                if self.error is None: self.lc = np.c_[self.t, self.y]
                else:                  self.lc = np.c_[self.t, self.y, self.error]

                self.pfit = pfit
                if self.error is None:
                    #self.yerror = 0.5*np.std(self.get_residual()[1])
                    self.yerror = stats.median_abs_deviation(self.get_residual()[1])

                if   error_estimation == 'bootstrap':  print('Bootstrapping...',flush=True)
                elif error_estimation == 'montecarlo': print('Performing monte carlo...',flush=True)

                error_estimation_fit = np.empty( (ntry,len(pfit)) )
                seeds = np.random.randint(1e09,size=ntry)
                if parallel:
                    # do error estimation fit parallal
                    available_ncores = multiprocessing.cpu_count()
                    if ncores <= -1:
                        ncores = available_ncores
                    elif available_ncores<ncores:
                        ncores = available_ncores

                    error_estimation_fit = ProgressParallel(n_jobs=ncores,total=ntry)(delayed(self._estimate_errors)(par) for par in seeds)
                    error_estimation_fit = np.asarray(error_estimation_fit)

                else:
                    for i,seed in tqdm(enumerate(seeds),total=ntry):
                        error_estimation_fit[i,:] = self._estimate_errors(seed)

                # Get rid of nan values
                goodpts = np.all(np.isfinite(error_estimation_fit),axis=1)
                error_estimation_fit = error_estimation_fit[ goodpts ]

                if len(error_estimation_fit) < 4:
                    print("Raise \'ntry\' or set a higher \'sample_size\'!")
                    raise RuntimeError('Not enough points to estimate error!')

                #meanval = np.nanmean(error_estimation_fit,axis=0)
                #perr = np.nanstd(error_estimation_fit,axis=0)
                bspercentiles = np.percentile(error_estimation_fit,[16, 50, 84],axis=0)
                perr = np.min(np.c_[bspercentiles[2]-bspercentiles[1],bspercentiles[1]-bspercentiles[0]],axis=1)

                if plotting:
                    labels = [r'Freq'] + [r'A$_'+str(i+1)+'$' for i in range(len(self.amps))] + \
                             [r'$\Phi_'+str(i+1)+'$' for i in range(len(self.phases))] + [r'Zero point']

                    if np.sum( is_outlier(error_estimation_fit) ) < 0.05*error_estimation_fit.shape[0]:
                        # Drop outliers if their number is low to not to change the distributions
                        _ = corner.corner(error_estimation_fit[~is_outlier(error_estimation_fit)],
                                           labels=labels,
                                           quantiles=[0.16, 0.5, 0.84],
                                           show_titles=True, title_kwargs={"fontsize": 12})
                    else:
                        _ = corner.corner(error_estimation_fit,
                                           labels=labels,
                                           quantiles=[0.16, 0.5, 0.84],
                                           show_titles=True, title_kwargs={"fontsize": 12})

                if np.any(perr == 0.0):
                    # if any of the errors is 0, shift light curve by period/2 and get new errors
                    warn('One of the errors is too small! Shifting light curve by half period to calculate errors...')

                    torig = self.t.copy()
                    yorig = self.y.copy()

                    # shifting light curve by period/2
                    self.t = self.t + 0.5/pfit[0]

                    _,newperr = self.fit_harmonics(maxharmonics = maxharmonics,
                                                    plotting = False,
                                                    kind=kind,
                                                    minimum_frequency=minimum_frequency,
                                                    maximum_frequency=maximum_frequency,
                                                    nyquist_factor=nyquist_factor,
                                                    samples_per_peak=samples_per_peak,
                                                    error_estimation=error_estimation,
                                                    ntry=ntry,
                                                    parallel=parallel, ncores=ncores,
                                                    sample_size=self.sample_size,
                                                    refit=True)

                    try:
                        um = np.where( perr == 0.0 )[0]
                        perr[um] = np.copy(newperr[um])
                    except IndexError:
                        pass
                    self.t = torig
                    self.y = yorig
                    self.pfit = pfit
                    self.perr = perr
                    return np.asarray(self.pfit), np.asarray(self.perr)

                self.pfit = pfit
                self.perr = perr
                return np.asarray(self.pfit), np.asarray(self.perr)

            elif error_estimation != 'analytic' and refit:
                # get new errors for shifted light curve
                if self.error is None: self.lc = np.c_[self.t, self.y]
                else:                  self.lc = np.c_[self.t, self.y, self.error]

                self.pfit = pfit
                if self.error is None:
                    #self.yerror = 0.5*np.std(self.get_residual()[1])
                    self.yerror = stats.median_abs_deviation(self.get_residual()[1])

                error_estimation_fit = np.empty( (ntry,len(pfit)) )
                seeds = np.random.randint(1e09,size=ntry)
                if parallel:
                    # do error_estimation fit parallal
                    available_ncores = multiprocessing.cpu_count()
                    if ncores <= -1:
                        ncores = available_ncores
                    elif available_ncores<ncores:
                        ncores = available_ncores

                    error_estimation_fit = ProgressParallel(n_jobs=ncores,total=ntry)(delayed(self._estimate_errors)(par) for par in seeds)
                    error_estimation_fit = np.asarray(error_estimation_fit)

                else:
                    for i,seed in tqdm(enumerate(seeds),total=ntry):
                        error_estimation_fit[i,:] = self._estimate_errors(seed)

                # Get rid of nan values
                goodpts = np.all(np.isfinite(error_estimation_fit),axis=1)
                error_estimation_fit = error_estimation_fit[ goodpts ]

                if len(error_estimation_fit) < 4:
                    print("Raise \'ntry\' or set a higher \'sample_size\'!")
                    raise RuntimeError('Not enough points to estimate error!')

                #meanval = np.nanmean(error_estimation_fit,axis=0)
                #newperr = np.nanstd(error_estimation_fit,axis=0)
                bspercentiles = np.percentile(error_estimation_fit,[16, 50, 84],axis=0)
                newperr = np.min(np.c_[bspercentiles[2]-bspercentiles[1],bspercentiles[1]-bspercentiles[0]],axis=1)

                self.pfit = pfit
                self.perr = newperr
                return np.asarray(self.pfit), np.asarray(self.perr)

        except RuntimeError:
            # if fit all components at once fails return previous results, and errors from covariance matrix
            if error_estimation:
                warn('%s method failed! Returning errors from pre-whitening steps.' % str(error_estimation))

            warn('Something went wrong! Returning errors from pre-whitening steps.')

            self.pfit = np.array([self.freqs[0], *self.amps, *self.phases, self.zeropoints[0] ])
            self.perr = np.array([self.freqserr[0], *self.ampserr, *self.phaseserr, self.zeropointerr[0] ])
            return self.pfit, self.perr

    def get_fourier_parameters(self):
        """
        Calculates Fourier parameters from given amplitudes and phases.

        Returns
        -------
        freq : number with uncertainty
            Main frequency and its estimated error.
        period : number with uncertainty
            Main period and its estimated error.
        Rn1 : number with uncertainty
            Rn1 value(s) and its estimated error(s),
            where n is the harmonics order.
        Pn1 : number with uncertainty
            Phin1 value(s) and its estimated error(s),
            where n is the harmonics order.
        """
        if not hasattr(self,"pfit"):
            warn("Please run \'fit_harmonics\' first!")
            return None,None,None,None

        if np.all(np.isnan(self.pfit)) or len(self.pfit)<6:
            warn('Not enough components to get Fourier parameters!')
            #raise ValueError('Not enough components to get Fourier parameters!')

            freq = ufloat(np.nan,np.nan)
            period = ufloat(np.nan,np.nan)
            Rn1 = [ufloat(np.nan,np.nan)]
            Pn1 = [ufloat(np.nan,np.nan)]

        else:
            nfreq = int(  (len(self.pfit)-1)/2  )

            freq = ufloat(self.pfit[0],self.perr[0])
            period = 1/freq

            Rn1 = []
            Pn1 = []
            for i in range(2,nfreq+1):
                Rn1.append( ufloat(self.pfit[i],self.perr[i]) / ufloat(self.pfit[1],self.perr[1]) )
                Pn1.append( ( ufloat(self.pfit[nfreq+i],self.perr[nfreq+i] )- i* ufloat(self.pfit[nfreq+1],self.perr[nfreq+1]) )%(2*np.pi) )

        return freq,period,Rn1,Pn1

    def get_residual(self):
        """
        Calculates the residual light curve after multi-harmonic Fourier fitting.

        Returns
        -------
        t : array
            Time stamps.
        y : array
            Residual flux/amp.
        yerr : array, optional
            If input errors were given, then error of residual flux/amp.
        """
        if not hasattr(self,"pfit"):
            warn("Please run \'fit_harmonics\' first!")
            if self.error is None:
                return None,None
            else:
                return None,None,None

        if self.error is None:
            return self.t, self.y - self.lc_model(self.t, *self.pfit), np.ones_like(self.t)*np.nan
        else:
            return self.t, self.y - self.lc_model(self.t, *self.pfit), self.error

    def get_analytic_uncertainties(self):
        """
        Calculates analytically derived uncertainties for multi-harmonic fit.
        Method is based on Breger et al. 1999.

        Returns
        -------
        perr : array-like
            Estimated error of the frequency and amplitudes, phases.
        """
        if not hasattr(self,"pfit"):
            warn("Please run \'fit_harmonics\' first!")
            return None

        resy = self.get_residual()[1]
        sigf = self._analytic_uncertainties(self.t,resy,self.pfit[1])[0]

        sigA = []
        sigPhi = []
        ncomponents = int((len(self.pfit)-1)/2)
        for i in range(1, ncomponents + 1 ):
            sigmas = self._analytic_uncertainties(self.t,resy,self.pfit[i])
            sigA.append( sigmas[1] )
            sigPhi.append( sigmas[2] )

        return [sigf] + sigA + sigPhi


class MultiFrequencyFitter(BaseFitter):

    def lc_model(self, *arg):
        """
        Get model light curve with all frequency components at the same time.

        Parameters
        ----------
        time : array
            Desired time points where multi-frequency fit is desired.
        arg : arguments
            List of arguments containing the frequencies, amplitudes, phases
            of each periodic component and the zero point.

        Returns
        ----------
        y : array
            Multi-frequency model light curve.
        """
        if not hasattr(self,"kind"):
            warn("Please run \'fit_freqs\' first!")
            return None

        time = arg[0]
        nparams = (len(arg)-2)//3
        freqs = arg[1:nparams+1]
        amps = arg[1+nparams:1+2*nparams]
        phases = arg[1+2*nparams:-1]
        const = arg[-1]

        y = 0
        for i in range(len(amps)):
            y += self._func(time, amps[i], freqs[i], phases[i], kind=self.kind)
        y += const
        return y

    def _estimate_errors(self,seed):
        np.random.seed(seed)

        if self.error_estimation == 'bootstrap':
            tmp_lc = self.lc[np.random.choice( self.lc.shape[0], int(len(self.lc)*self.sample_size), replace=False), :]
        elif self.error_estimation == 'montecarlo':
            tmp_lc = self.lc.copy()

            if self.error is None:
                tmp_lc[:,1] += np.random.normal(0,self.yerror,tmp_lc.shape[0])
            else:
                tmp_lc[:,1] += np.random.normal(0,self.error,tmp_lc.shape[0])

        # Subtract mean from time points to decouple frequencies from phases
        tmp_lc[:,0] -= tmp_lc[:,0].mean()

        lbound = list( np.array(self.freqs) - 0.1 )
        lbound += [0]*len(self.amps) + [-np.inf]*len(self.phases) + [-np.inf]
        ubound = list( np.array(self.freqs) + 0.1 )
        ubound += [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
        bounds = (lbound,ubound)

        try:
            if self.error is None:
                tmp_pfit, _ = curve_fit(lambda *args: self.lc_model(*args), tmp_lc[:,0], tmp_lc[:,1],
                                        p0=(*self.freqs, *self.amps, *self.phases, np.mean(tmp_lc[:,1])),
                                        bounds=bounds, maxfev=5000)
            else:
                tmp_pfit, _ = curve_fit(lambda *args: self.lc_model(*args), tmp_lc[:,0], tmp_lc[:,1],
                                        p0=(*self.freqs, *self.amps, *self.phases, np.mean(tmp_lc[:,1])) ,
                                        sigma=tmp_lc[:,2], absolute_sigma=self.absolute_sigma, bounds=bounds, maxfev=5000)

            tmp_pfit[2*len(self.amps):-1] = tmp_pfit[2*len(self.amps):-1]%(2*np.pi)
        except RuntimeError:
            tmp_pfit = [np.nan] * ( len(self.freqs) + len(self.amps) + len(self.phases) + 1 )

        return tmp_pfit

    def fit_freqs(self, maxfreqs = 3, sigma = 4,
                  absolute_sigma=True,
                  plotting = False, scale='flux',
                  minimum_frequency=None, maximum_frequency=None,
                  nyquist_factor=1,samples_per_peak=100,
                  kind='sin',
                  error_estimation='analytic',ntry=1000,
                  sample_size=0.7,
                  parallel=True, ncores=-1,
                  refit=False):
        """
        ``fit_freqs`` performs consecutive Fourier pre-whitening with given number of frequencies.

        Parameters
        ----------
        maxfreqs : int, default: 3
            The number of frequencies to be fitted. Pass a very large number to fit all frequencies,
            limited by the signal-to-noise ratio (`sigma`).
        sigma : float, default: 4
            Signal-to-noise ratio above which a frequency is considered significant and kept.
        absolute_sigma : bool, default: True
            If `True`, error is used in an absolute sense and the estimated parameter covariance
            reflects these absolute values.
        plotting: bool, default: False
            If `True`, fitting steps will be displayed.
        scale: 'mag' or 'flux', default: 'flux'
            Lightcurve plot scale.
        minimum_frequency : float, optional
            If specified, then use this minimum frequency rather than one chosen based on the size
            of the baseline.
        maximum_frequency : float, optional
            If specified, then use this maximum frequency rather than one chosen based on the average
            nyquist frequency.
        nyquist_factor : float, default: 1
            The multiple of the average nyquist frequency used to choose the maximum frequency
            if ``maximum_frequency`` is not provided.
        samples_per_peak:  float, default: 100
            The approximate number of desired samples across the typical frequency peak.
        kind: str, 'sin' or 'cos'
            Harmonic _function to be fitted.
        error_estimation: `analytic`, `bootstrap` or `montecarlo`, default: `analytic`
            If `bootstrap` or `montecarlo` is choosen, boostrap or monte carlo method will be used to estimate parameter uncertainties.
            Otherwise given uncertainties are calculated analytically.
        ntry: int, default: 1000
            Number of resamplings for error estimation.
        sample_size: float, default: 0.7
            The ratio of data points to be used for bootstrap error estimation in each step.
            Applies only if `error_estimation` is set to `bootstrap`.
        parallel: bool, default : True
            If `True`, sampling for error estimation is performed parallel to speed up the process.
        ncores: int, default: -1
            Number of CPU cores to be used for parallel error estimation. If `-1`, then all available
            cores will be used.

        Returns
        -------
        pfit : array-like
            Array of fitted parameters.
            The frequencies, amplitudes and phases, and the zero point.
        perr : array-like
            Estimated error of the parameters.
        """
        self.sample_size = sample_size
        self.kind = kind
        self.absolute_sigma = absolute_sigma
        self.error_estimation = error_estimation

        if minimum_frequency is not None and maximum_frequency is not None:
            if minimum_frequency > maximum_frequency:
                raise ValueError('Minimum frequency is larger than maximum frequency.')

        if maxfreqs<1:
            raise ValueError('Number of frequencies must be >=1.')

        if maximum_frequency is None and nyquist_factor * (0.5/np.median( np.diff(self.t) )) < 2.:
            # Nyquist is low
            warn('Nyquist frequency is low!\nYou might want to set maximum frequency instead.')

        if error_estimation not in ['analytic','bootstrap','montecarlo']:
            raise TypeError('%s method is not supported! Please choose \'analytic\', \'bootstrap\' or \'montecarlo\'.' % str(error_estimation))

        # fit periodic funtions and do prewhitening
        yres = self.y.copy()

        self.freqs = []
        self.amps = []
        self.phases = []
        self.zeropoints = []
        self.freqserr = []
        self.ampserr = []
        self.phaseserr = []
        self.zeropointerr = []

        for i in range(maxfreqs):
            ls = LombScargle(self.t, yres, nterms=1)

            with np.errstate(divide='ignore',invalid='ignore'):
                freq, power = ls.autopower(normalization='psd',
                                           minimum_frequency=minimum_frequency,
                                           maximum_frequency=maximum_frequency,
                                           samples_per_peak=10,
                                           nyquist_factor=nyquist_factor)

            # Convert LS power to amplitude
            power = np.sqrt(4*power/len(self.t))

            # LS may return inf values
            goodpts = np.isfinite(power)
            freq  = freq[goodpts]
            power = power[goodpts]
            power[power<0] = 0

            #if np.nanmax(power) <= np.nanmean(power) + sigma*np.nanstd(power):
            if np.nanmax(power) / np.nanmean(power) < sigma:
                # s/n < sigma
                break

            # --- Check if best period is longer than 2x data duration ---
            if np.allclose(freq[np.argmax(power)] ,0) or 1./freq[np.argmax(power)] > 2*np.ptp(self.t):
                warn('Period is longer than 2x data duration!\nSet minimum frequency to avoid problems!\nSkipping...')
                break

            # Resample spectrum around highest peak
            with np.errstate(divide='ignore',invalid='ignore'):
                freq, power = ls.autopower(normalization='psd',
                                           samples_per_peak=2000,
                                           minimum_frequency=max(1e-10,freq[power.argmax()]-5/self.t.ptp()),
                                           maximum_frequency=freq[power.argmax()]+20/self.t.ptp())

            # Convert LS power to amplitude
            power = np.sqrt(4*power/len(self.t))

            # LS may return inf values
            goodpts = np.isfinite(power)
            freq  = freq[goodpts]
            power = power[goodpts]
            power[power<0] = 0

            # fit periodic component
            try:
                best_freq = freq[power.argmax()]

                pfit1, pcov1 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, best_freq, phase ,kind=kind) + const, self.t, yres,
                                      p0=(np.ptp(yres)/4,2.,np.mean(yres)), bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                      sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)
                pfit2, pcov2 = curve_fit(lambda time, amp, phase, const: self._func(time, amp, best_freq, phase ,kind=kind) + const, self.t, yres,
                                      p0=(np.ptp(yres)/4,5.,np.mean(yres)), bounds=([0,0,-np.inf], [np.ptp(yres), 2*np.pi, np.inf]) ,
                                      sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

                chi2_1 = np.sum( (yres-pfit1[-1] - self._func(self.t, pfit1[0], best_freq, pfit1[1] ,kind=kind))**2 )
                chi2_2 = np.sum( (yres-pfit2[-1] - self._func(self.t, pfit2[0], best_freq, pfit2[1] ,kind=kind))**2 )

                if chi2_1 < chi2_2:
                    pfit = pfit1
                else:
                    pfit = pfit2

                pfit, pcov = curve_fit(lambda time, amp, freq, phase, const: self._func(time, amp, freq, phase, kind=kind) + const,
                                      self.t, yres,
                                      p0=(pfit[0],best_freq,pfit[1],np.mean(yres)),
                                      bounds=([0,0,0,-np.inf], [np.ptp(yres), 2*best_freq, 2*np.pi, np.inf]) ,
                                      sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

                best_freq = pfit[1]

            except (RuntimeError,ValueError) as err:
                if i == 0:
                    warn(err)

                    self.pfit = self.perr = [np.nan]*(3+1)
                    return self.pfit, self.perr
                else:
                    break

            if plotting:
                # plot phased light curve and fit
                per = 1/pfit[1]

                # Calculate spectrum for plotting
                ls = LombScargle(self.t, yres, nterms=1)
                freq, power = ls.autopower(normalization='psd',
                                           minimum_frequency=minimum_frequency,
                                           maximum_frequency=maximum_frequency,
                                           samples_per_peak=samples_per_peak,
                                           nyquist_factor=nyquist_factor)

                # Convert LS power to amplitude
                power = np.sqrt(4*power/len(self.t))

                plt.figure(figsize=(15,3))
                plt.subplot(121)
                plt.plot(freq, power)
                plt.xlabel('Frequency (c/d)')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.subplot(122)
                plt.plot(self.t%per/per,yres-pfit[-1],'k.')
                plt.plot(self.t%per/per+1,yres-pfit[-1],'k.')
                plt.plot(self.t%per/per,self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind),'C1.',ms=1)
                plt.plot(self.t%per/per+1,self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind),'C1.',ms=1)
                if scale == 'mag': plt.gca().invert_yaxis()
                plt.xlabel('Phase (f=%.6f c/d; P=%.5f d)' % (1/per,per))
                plt.ylabel('Brightness')
                plt.show()

            # Collect results
            self.freqs.append( pfit[1] )
            self.amps.append( pfit[0] )
            self.phases.append( pfit[2] )
            self.zeropoints.append( pfit[3] )

            pcov = np.sqrt(np.diag(pcov))
            self.freqserr.append( pcov[1] )
            self.ampserr.append( pcov[0] )
            self.phaseserr.append( pcov[2] )
            self.zeropointerr.append( pcov[3] )

            yres -= self._func(self.t,pfit[0],pfit[1],pfit[2], kind=kind) + pfit[3]

        # --- Error estimation after all frequencies are given ---
        try:
            # fit all amplitudes+phases at the same time
            lbound = [0]*len(self.amps) + [-np.inf]*len(self.phases) + [-np.inf]
            ubound = [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
            bounds = (lbound,ubound)
            pfit, pcov = curve_fit(lambda *args: self.lc_model(args[0], *self.freqs, *args[1:]), self.t, self.y,
                                    p0=(*self.amps, *self.phases, np.mean(self.y)),
                                    bounds=bounds, sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

            self.amps   = pfit[:len(self.amps)]
            self.phases = pfit[len(self.amps):-1]
            zpfit = pfit[-1]

            # fit all periodic components at the same time
            lbound = list( np.array(self.freqs) - 0.1 )
            lbound += [0]*len(self.amps) + [-np.inf]*len(self.phases) + [-np.inf]
            ubound = list( np.array(self.freqs) + 0.1 )
            ubound += [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
            bounds = (lbound,ubound)
            pfit, pcov = curve_fit(lambda *args: self.lc_model(*args), self.t, self.y,
                                    p0=(*self.freqs, *self.amps, *self.phases, zpfit),
                                    bounds=bounds, sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)

            # convert all phases into the 0-2pi range
            pfit[2*len(self.amps):-1] = pfit[2*len(self.amps):-1]%(2*np.pi)

            '''
            # if any of the errors is inf, shift light curve by period/2 and get new errors
            if not error_estimation and np.any(np.isinf(pcov)):
                warn('One of the errors is inf! Shifting light curve by half period to calculate errors...')
                lbound = [0]*(1+len(self.amps)) + [-np.inf]*len(self.phases) + [-np.inf]
                ubound = [2*best_freq] + [np.ptp(self.y)]*len(self.amps) + [np.inf]*len(self.phases) + [np.inf]
                bounds = (lbound,ubound)
                _, pcov = curve_fit(lambda *args: self.lc_model(*args), self.t + 0.5/pfit[0], self.y, p0=(self.freqs[0], *self.amps, *[(pha+np.pi)%(2*np.pi) for pha in self.phases], np.mean(self.y)) ,
                                    bounds=bounds, sigma=self.error, absolute_sigma=absolute_sigma, maxfev=5000)
            '''

            if error_estimation == 'analytic':
                self.pfit = pfit
                self.perr = self.get_analytic_uncertainties() + [ np.sqrt(np.diag(pcov))[-1] ]

                return self.pfit, self.perr

            elif error_estimation != 'analytic' and refit is False:
                # use bootstrap or MC to get realistic errors (bootstrap = get subsample and redo fit n times)
                if self.error is None: self.lc = np.c_[self.t, self.y]
                else:                  self.lc = np.c_[self.t, self.y, self.error]

                self.pfit = pfit
                if self.error is None:
                    #self.yerror = 0.5*np.std(self.get_residual()[1])
                    self.yerror = stats.median_abs_deviation(self.get_residual()[1])

                if   error_estimation == 'bootstrap':  print('Bootstrapping...',flush=True)
                elif error_estimation == 'montecarlo': print('Performing monte carlo...',flush=True)

                error_estimation_fit = np.empty( (ntry,len(pfit)) )
                seeds = np.random.randint(1e09,size=ntry)
                if parallel:
                    # do error estimation fit parallal
                    available_ncores = multiprocessing.cpu_count()
                    if ncores <= -1:
                        ncores = available_ncores
                    elif available_ncores<ncores:
                        ncores = available_ncores

                    error_estimation_fit = ProgressParallel(n_jobs=ncores,total=ntry)(delayed(self._estimate_errors)(par) for par in seeds)
                    error_estimation_fit = np.asarray(error_estimation_fit)

                else:
                    for i,seed in tqdm(enumerate(seeds),total=ntry):
                        error_estimation_fit[i,:] = self._estimate_errors(seed)

                # Get rid of nan values
                goodpts = np.all(np.isfinite(error_estimation_fit),axis=1)
                error_estimation_fit = error_estimation_fit[ goodpts ]

                if len(error_estimation_fit) < 4:
                    print("Raise \'ntry\' or set a higher \'sample_size\'!")
                    raise RuntimeError('Not enough points to estimate error!')

                #meanval = np.nanmean(error_estimation_fit,axis=0)
                #perr = np.nanstd(error_estimation_fit,axis=0)
                bspercentiles = np.percentile(error_estimation_fit,[16, 50, 84],axis=0)
                perr = np.min(np.c_[bspercentiles[2]-bspercentiles[1],bspercentiles[1]-bspercentiles[0]],axis=1)

                if plotting:
                    labels = [r'f$_'+str(i+1)+'$' for i in range(len(self.freqs))] + \
                             [r'A$_'+str(i+1)+'$' for i in range(len(self.amps))] + \
                             [r'$\Phi_'+str(i+1)+'$' for i in range(len(self.phases))] + [r'Zero point']

                    if np.sum( is_outlier(error_estimation_fit) ) < 0.05*error_estimation_fit.shape[0]:
                        # Drop outliers if their number is low to not to change the distributions
                        _ = corner.corner(error_estimation_fit[~is_outlier(error_estimation_fit)],
                                           labels=labels,
                                           quantiles=[0.16, 0.5, 0.84],
                                           show_titles=True, title_kwargs={"fontsize": 12})
                    else:
                        _ = corner.corner(error_estimation_fit,
                                           labels=labels,
                                           quantiles=[0.16, 0.5, 0.84],
                                           show_titles=True, title_kwargs={"fontsize": 12})

                if np.any(perr == 0.0):
                    # if any of the errors is 0, shift light curve by period/2 and get new errors
                    warn('One of the errors is too small! Shifting light curve by half period to calculate errors...')

                    torig = self.t.copy()
                    yorig = self.y.copy()

                    # shifting light curve by period/2
                    self.t = self.t + 0.5/pfit[0]

                    _,newperr = self.fit_freqs(maxfreqs = maxfreqs, sigma=sigma,
                                                plotting = False,
                                                kind=kind,
                                                minimum_frequency=minimum_frequency,
                                                maximum_frequency=maximum_frequency,
                                                nyquist_factor=nyquist_factor,
                                                samples_per_peak=samples_per_peak,
                                                error_estimation=error_estimation,
                                                ntry=ntry,
                                                parallel=parallel, ncores=ncores,
                                                sample_size=self.sample_size,
                                                refit=True)

                    try:
                        um = np.where( perr == 0.0 )[0]
                        perr[um] = np.copy(newperr[um])
                    except IndexError:
                        pass
                    self.t = torig
                    self.y = yorig
                    self.pfit = pfit
                    self.perr = perr
                    return self.pfit, self.perr

                self.pfit = pfit
                self.perr = perr
                return self.pfit, self.perr

            elif error_estimation != 'analytic' and refit:
                # get new errors for shifted light curve
                if self.error is None: self.lc = np.c_[self.t, self.y]
                else:                  self.lc = np.c_[self.t, self.y, self.error]

                self.pfit = pfit
                if self.error is None:
                    #self.yerror = 0.5*np.std(self.get_residual()[1])
                    self.yerror = stats.median_abs_deviation(self.get_residual()[1])

                error_estimation_fit = np.empty( (ntry,len(pfit)) )
                seeds = np.random.randint(1e09,size=ntry)
                if parallel:
                    # do error_estimation fit parallal
                    available_ncores = multiprocessing.cpu_count()
                    if ncores <= -1:
                        ncores = available_ncores
                    elif available_ncores<ncores:
                        ncores = available_ncores

                    error_estimation_fit = ProgressParallel(n_jobs=ncores,total=ntry)(delayed(self._estimate_errors)(par) for par in seeds)
                    error_estimation_fit = np.asarray(error_estimation_fit)

                else:
                    for i,seed in tqdm(enumerate(seeds),total=ntry):
                        error_estimation_fit[i,:] = self._estimate_errors(seed)

                # Get rid of nan values
                goodpts = np.all(np.isfinite(error_estimation_fit),axis=1)
                error_estimation_fit = error_estimation_fit[ goodpts ]

                if len(error_estimation_fit) < 4:
                    print("Raise \'ntry\' or set a higher \'sample_size\'!")
                    raise RuntimeError('Not enough points to estimate error!')

                #meanval = np.nanmean(error_estimation_fit,axis=0)
                #newperr = np.nanstd(error_estimation_fit,axis=0)
                bspercentiles = np.percentile(error_estimation_fit,[16, 50, 84],axis=0)
                newperr = np.min(np.c_[bspercentiles[2]-bspercentiles[1],bspercentiles[1]-bspercentiles[0]],axis=1)

                self.pfit = pfit
                self.perr = newperr
                return self.pfit, self.perr

        except RuntimeError:
            # if fit all components at once fails return previous results, and errors from covariance matrix
            if error_estimation:
                warn('%s method failed! Returning errors from pre-whitening steps.' % str(error_estimation))

            warn('Something went wrong! Returning errors from pre-whitening steps.')

            self.pfit = np.array([self.freqs, *self.amps, *self.phases, self.zeropoints[0] ])
            self.perr = np.array([self.freqserr, *self.ampserr, *self.phaseserr, self.zeropointerr[0] ])
            return self.pfit, self.perr

    def get_residual(self):
        """
        Calculates the residual light curve after multi-frequency fitting.

        Returns
        -------
        t : array
            Time stamps.
        y : array
            Residual flux/amp.
        yerr : array, optional
            If input errors were given, then error of residual flux/amp.
        """
        if not hasattr(self,"pfit"):
            warn("Please run \'fit_freqs\' first!")
            if self.error is None:
                return None,None
            else:
                return None,None,None

        if self.error is None:
            return self.t, self.y - self.lc_model(self.t, *self.pfit), np.ones_like(self.t)*np.nan
        else:
            return self.t, self.y - self.lc_model(self.t, *self.pfit), self.error

    def get_analytic_uncertainties(self):
        """
        Calculates analytically derived uncertainties for multi-frequency fit.
        Method is based on Breger et al. 1999.

        Returns
        -------
        perr : array-like
            Estimated error of the frequencies, amplitudes, phases.
        """
        resy = self.get_residual()[1]

        sigf = []
        sigA = []
        sigPhi = []
        ncomponents = int((len(self.pfit)-1)/3)
        for i in range(ncomponents):
            sigmas = self._analytic_uncertainties(self.t,resy,self.pfit[ncomponents+i])
            sigf.append( sigmas[0] )
            sigA.append( sigmas[1] )
            sigPhi.append( sigmas[2] )

        return sigf + sigA + sigPhi
