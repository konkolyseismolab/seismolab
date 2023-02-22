import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from astropy.convolution import Gaussian1DKernel, convolve
from psutil import cpu_count
from joblib import parallel_backend

import warnings
warnings.filterwarnings("ignore")

from seismolab.fourier import MultiHarmonicFitter

from matplotlib.collections import LineCollection

import joblib
from joblib import delayed
from tqdm.auto import tqdm

__all__ = ['TemplateFitter']

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

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    if ax is None: ax = plt.gca()
    ax.add_collection(lc)
    return lc


def splitthem(inputBJD, inputflux,fluxerror,span,step,n):
    """
    Split light curve into chunks
    """
    um = (inputBJD>=inputBJD[0]+step*n) & (inputBJD<inputBJD[0]+step*n+span)
    dfbitlistBJD   = inputBJD[um]
    dfbitlistflux  = inputflux[um]
    if fluxerror is None:
        dfbitlistfluxerror = None
    else:
        dfbitlistfluxerror = fluxerror[um]

    midBJD = inputBJD[0]+step*n + span/2

    return midBJD, dfbitlistBJD, dfbitlistflux, dfbitlistfluxerror

def modulated_lc_model(time, a0, a, dPhi, pfit, kind):
    """
    Sum of modulated sin/cos curves.

    Parameters
    ----------
    time : array
        Light curve time points.
    a0 : array
        Relative zero point variation.
    a : array
        Relative amplitude variation.
    dPhi : array
        Relative phase variation.
    pfit : array-like
        Array of fitted parameters. The main frequency, amplitudes and phases of the harmonics,
        and the zero point.
    kind : 'sin' or 'cos'
        Function type to construct template.
    """
    best_freq = pfit[0]
    nparams = (len(pfit)-2)//2
    amps = pfit[1:1+nparams]
    phases = pfit[1+nparams:-1]
    const = pfit[-1]

    y = 0
    if kind == 'sin':
        for i,(Amp,Phi) in enumerate(zip(amps,phases)):
            y += a*Amp*np.sin(2*np.pi*(i+1)*best_freq*time + Phi+(i+1)*dPhi)
    elif kind == 'cos':
        for i,(Amp,Phi) in enumerate(zip(amps,phases)):
            y += a*Amp*np.cos(2*np.pi*(i+1)*best_freq*time + Phi+(i+1)*dPhi)
    y += a0*const

    return y

def get_stat(data):
    """
    Statistical values of MCMC posterior
    """
    value = np.median(data)
    ep = np.percentile(data,84.1)-value
    em = value-np.percentile(data,15.9)
    return value,ep,em

def smooth_data(a0values,avalues,psivalues,gapat,smoothness_factor,step):
    gapat = np.concatenate((np.array([0]),gapat,np.array([ len(a0values) ])))

    a0values_out  = np.empty_like(a0values)
    avalues_out   = np.empty_like(avalues)
    psivalues_out = np.empty_like(psivalues)

    for i in range(gapat.shape[0]-1):
        a0values_cut  = a0values[gapat[i]:gapat[i+1]]
        avalues_cut   = avalues[gapat[i]:gapat[i+1]]
        psivalues_cut = psivalues[gapat[i]:gapat[i+1]]

        a0values_smooth  = np.concatenate((a0values_cut[::-1],a0values_cut,a0values_cut[::-1]))
        avalues_smooth   = np.concatenate((avalues_cut[::-1],avalues_cut,avalues_cut[::-1]))
        psivalues_smooth = np.concatenate((psivalues_cut[::-1],psivalues_cut,psivalues_cut[::-1]))

        gauss = Gaussian1DKernel(stddev = smoothness_factor * 1/step)
        a0values_smooth  = convolve(a0values_smooth, gauss)
        avalues_smooth   = convolve(avalues_smooth, gauss)
        psivalues_smooth = convolve(psivalues_smooth, gauss)

        a0values_smooth  = a0values_smooth[a0values_cut.shape[0]:2*a0values_cut.shape[0]]
        avalues_smooth   = avalues_smooth[avalues_cut.shape[0]:2*avalues_cut.shape[0]]
        psivalues_smooth = psivalues_smooth[psivalues_cut.shape[0]:2*psivalues_cut.shape[0]]

        a0values_out[gapat[i]:gapat[i+1]]  = a0values_smooth
        avalues_out[gapat[i]:gapat[i+1]]   = avalues_smooth
        psivalues_out[gapat[i]:gapat[i+1]] = psivalues_smooth

    return a0values_out, avalues_out, psivalues_out

def fit_lightcurve_chunk(midBJD,bitBJD,bitflux,bitfluxerror,
                        LSPfreq,span,pfit,
                        duty_cycle,error_estimation,kind,
                        debug=False):

    # ---- Skip chunk if number of pts is low ----
    if debug: print('N points:',len(bitBJD))
    if len(bitBJD)<4:
        return [np.nan]*7

    # ---- Skip chunk if duty cycle is low ----
    if debug: print('Duty cycle:', np.ptp(bitBJD) / (span*1/LSPfreq) )
    if np.ptp(bitBJD) < duty_cycle * span*1/LSPfreq:
        return [np.nan]*7

    # ---- Skip chunk if there is a large gap ----
    if ~np.all(np.diff(bitBJD) < duty_cycle * span*1/LSPfreq):
        if debug: print('Skipping due to large gap...')
        return [np.nan]*7

    # ---- Fit zp, amp, phase ----
    if error_estimation == 'montecarlo':
        if debug: print('Running MCMC...')

        import pymc3 as pm
        import seaborn as sns

        with pm.Model() as model:
            ## define Normal priors to give Ridge regression
            a0 = pm.Uniform("a0", 0., 2., testval=1.)
            a = pm.Uniform("a", 0., 2., testval=1.)
            psi = pm.Uniform("psi", -1., 1., testval=0.)

            ## define model
            yest = modulated_lc_model(bitBJD, a0, a, psi, pfit, kind)

            ## define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
            likelihood = pm.Normal("likelihood", mu=yest,
                                    sigma=10*bitfluxerror if bitfluxerror is not None else np.sqrt(bitflux),
                                    observed=bitflux)

            #Populate MCMC sampler
            traces = pm.sample(1000,chains=1,cores=1)

        a0, a0ep, a0em = get_stat(traces['a0'])
        a, aep, aem = get_stat(traces['a'])
        psi, psiep, psiem = get_stat(traces['psi'])

        if debug:
            df_trace = pm.trace_to_dataframe(traces)
            _ = sns.pairplot(df_trace)

        return np.min(bitBJD) + np.ptp(bitBJD)/2, a0, max(a0ep,a0em), a, max(aep,aem), psi, max(psiep,psiem)

    else:
        try:
            _pfit, _pcov = optimize.curve_fit(lambda x, a0, a, psi: modulated_lc_model(x, a0, a, psi, pfit, kind),
                                          bitBJD,bitflux,
                                          p0=[0.9,0.9,0.1],
                                          sigma=bitfluxerror,absolute_sigma=True,method='trf')
        except RuntimeError:
            return [np.nan]*7

        perr_curvefit = np.sqrt(np.diag(_pcov))

        return midBJD, _pfit[0], perr_curvefit[0], _pfit[1], perr_curvefit[1], _pfit[2], perr_curvefit[2]

class TemplateFitter:
    def __init__(self, time,flux,fluxerror=None):
        """
        time : array
            Light curve time points.
        flux : array
            Corresponding flux/mag values.
        fluxerror : array, optional
            Corresponding flux/mag error values.
        """
        time = np.asarray(time,dtype=float)
        flux = np.asarray(flux,dtype=float)
        if fluxerror is not None:
            fluxerror = np.asarray(fluxerror,dtype=float)

        goodpts = np.isfinite(time)
        goodpts &= np.isfinite(flux)
        if fluxerror is not None:
            goodpts &= np.isfinite(fluxerror)

        self.time = time[goodpts]
        self.flux = flux[goodpts]
        if fluxerror is not None:
            self.fluxerror = fluxerror[goodpts]
        else:
            self.fluxerror = fluxerror

    def fit(self,
        span = 3,
        step = 1,
        error_estimation='analytic',

        maxharmonics = 10,
        minimum_frequency=None,
        maximum_frequency=None,
        nyquist_factor=1,
        samples_per_peak=100,
        kind='sin',

        plotting=False,
        scale='flux',
        saveplot=False,
        saveresult=False,
        filename='result',
        showerrorbar=True,

        smoothness_factor=0.5,
        duty_cycle = 0.6,

        debug=False,
        best_freq=None
        ):
        """
        Compute amplitude/phase/zero point variation based on template fitting.

        Parameters
        ----------
        span : float, default: 5
            Number of puls cycles to be fitted.
        step : float, default: 3
            Steps in number of puls cycle.
        error_estimation : 'analytic' or 'montecarlo', default 'analytic'
            Type of error estimation for results.

        maxharmonics : int, default: 5
            Max number of harmonics to be used in template.
        minimum_frequency : float, optional
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float, optional
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.
        nyquist_factor : float, optional, default: 1
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        samples_per_peak : float, optional, default: 100
            The approximate number of desired samples across the typical peak.
        kind : 'sin' or 'cos', default: 'sin'
            Function type to construct template.

        plotting : bool, deaful: False
            Show result.
        scale: 'mag' or 'flux', default: 'flux'
            Lightcurve plot scale.
        saveplot : bool, default: False
            Save result as txt file.
        saveresult : bool, default: False
            Save results as txt
        filename : str, default: 'result'
            Beginning of txt filename.
        showerrorbar : bool, default: True
            Plot errorbars as well.

        smoothness_factor : float, optional, default: 0.5
            Level of Gaussian smoothing of amp/phase/zp values.
            0: no smoothing, 0.5-1: slight smoothing, >=1: significant smoothing
        duty_cycle : float, optional, default: 0.6
            Minimum duty cycle that is needed in case of each light curve chunk.
            Should be between 0-1.

        best_freq : float, default: None
            If given, then this frequency will be used as the basis of the harmonics,
            instead of calculating a Lomb-Scargle spectrum to get a frequency.

        debug : bool, default False
            Verbose output.


        Returns:
        -------
        times : array
            Time points.
        amp : array
            Amplitude variation.
        amperr : array
            Amplitude variation error.
        phase : array
            Phase variation.
        phaseerr : array
            Phase variation error.
        zp : array
            Zero point variation.
        zperr : array
            Zero point variation error.
        """

        if error_estimation not in ['analytic','montecarlo']:
            raise TypeError('%s method is not supported! Please set \'error_estimation\' to \'analytic\' or \'montecarlo\'.' % str(error_estimation))

        # Initialize Fourier fitter by passing your light curve
        if debug: print('Calculating Lomb-Scargle...')
        fitter = MultiHarmonicFitter(self.time,self.flux)

        pfit,perr = fitter.fit_harmonics(maxharmonics = maxharmonics,
                                         plotting = debug,
                                         minimum_frequency=minimum_frequency,
                                         maximum_frequency=maximum_frequency,
                                         nyquist_factor=nyquist_factor,
                                         samples_per_peak=samples_per_peak,
                                         kind=kind,
                                         best_freq=best_freq)

        # Make Fourier results attribute
        self.pfit = pfit
        self.perr = perr
        self.kind = kind

        LSPfreq = pfit[0]

        if debug:
            plt.title('Template')
            plt.scatter(self.time, self.flux)
            plt.plot(self.time,fitter.lc_model(self.time,*pfit),c='C1')
            plt.xlim(self.time.max()-3/pfit[0],self.time.max())
            plt.show()

        BJDmidP=[]
        a0values=[]
        a0errorvalues=[]
        avalues=[]
        aerrorvalues=[]
        psivalues=[]
        psierrorvalues=[]

        if debug: print('Splitting them...')

        params = []
        for counter in range( int(np.ceil( self.time.ptp() / (step*1/LSPfreq))) ):

            # ---- Get chunk ----
            midBJD,bitBJD,bitflux,bitfluxerror=splitthem(
                                                    self.time,self.flux,self.fluxerror,
                                                    span=span*1/LSPfreq,
                                                    step=step*1/LSPfreq,
                                                    n=counter)

            if debug:
                result = fit_lightcurve_chunk(midBJD,bitBJD,bitflux,bitfluxerror,
                                            LSPfreq,span,pfit,
                                            duty_cycle,error_estimation,kind,
                                            debug)

                BJDmidP.append( result[0] )

                a0values.append(result[1])
                a0errorvalues.append(result[2])

                avalues.append(result[3])
                aerrorvalues.append(result[4])

                psivalues.append(result[5])
                psierrorvalues.append(result[6])

                if ~np.all(np.isnan(bitBJD)):
                    plt.figure()
                    plt.title('Fit to subsample %d' % (counter+1))
                    plt.scatter(bitBJD,bitflux)

                    xxxx = np.linspace(min(bitBJD),max(bitBJD),1000)

                    plt.plot(xxxx,modulated_lc_model(xxxx, a0values[-1], avalues[-1], psivalues[-1], pfit, kind))
                    plt.plot(xxxx,modulated_lc_model(xxxx, a0values[-1], avalues[-1], 0, pfit, kind))

                    plt.show()
            else:
                params.append( [midBJD,bitBJD,bitflux,bitfluxerror,
                                LSPfreq,span,pfit,
                                duty_cycle,error_estimation,kind,
                                debug] )

        if not debug:
            ncores = cpu_count(logical=False)

            with parallel_backend('multiprocessing'):
                result = ProgressParallel(n_jobs=ncores,total=len(params))(delayed(fit_lightcurve_chunk)(*par) for par in params)
            result = np.asarray(result)

            BJDmidP        = result[:,0]
            a0values       = result[:,1]
            a0errorvalues  = result[:,2]
            avalues        = result[:,3]
            aerrorvalues   = result[:,4]
            psivalues      = result[:,5]
            psierrorvalues = result[:,6]

        BJDmidP        = np.asarray(BJDmidP)
        a0values       = np.asarray(a0values)
        a0errorvalues  = np.asarray(a0errorvalues)
        avalues        = np.asarray(avalues)
        aerrorvalues   = np.asarray(aerrorvalues)
        psivalues      = np.asarray(psivalues)
        psierrorvalues = np.asarray(psierrorvalues)

        goodpts = np.isfinite(BJDmidP)
        BJDmidP        = BJDmidP[goodpts]
        a0values       = a0values[goodpts]
        a0errorvalues  = a0errorvalues[goodpts]
        avalues        = avalues[goodpts]
        aerrorvalues   = aerrorvalues[goodpts]
        psivalues      = psivalues[goodpts]
        psierrorvalues = psierrorvalues[goodpts]

        # ----- Add Fourier errorbars to OC errors -----
        nharmonics = (len(pfit)-2)//2
        a0values  = pfit[-1] * a0values
        avalues   = pfit[1] * avalues
        psivalues = pfit[1+nharmonics] + psivalues
        a0errorvalues  = perr[-1] + a0errorvalues
        aerrorvalues   = perr[1] + aerrorvalues
        psierrorvalues = perr[1+nharmonics] + psierrorvalues

        # ----- Find large gaps in dataset -----
        gapatorig = np.where( np.diff(self.time) > max(4.,200*np.nanmedian(np.diff(self.time))) )[0]
        gapatorig += 1
        if len(gapatorig) == 0:
             gapsize = np.inf
        else:
             gapsize = np.min( self.time[gapatorig] - self.time[gapatorig-1] )

        gapsize = max( gapsize, 2*np.median(np.diff(BJDmidP)) )
        gapat = np.where( np.diff(BJDmidP) > gapsize )[0]
        gapat += 1

        # ----- Smoothing OC curve -----
        if len(a0values)>0:
            a0values_smooth,avalues_smooth,psivalues_smooth = smooth_data(a0values,avalues,psivalues,gapat,smoothness_factor,step)
        else:
            return np.array(BJDmidP), avalues,aerrorvalues, psivalues,psierrorvalues, a0values,a0errorvalues

        if ~np.all(np.isnan(avalues_smooth)):
            a0values = a0values_smooth
            avalues = avalues_smooth
            psivalues = psivalues_smooth

        # ----- Plot results -----
        if plotting or saveplot:
            period=1/LSPfreq
            BJDmodP_extended = self.time%period/period
            BJDmodP_extended = np.concatenate((BJDmodP_extended,1+BJDmodP_extended))
            flux_extended = np.tile(self.flux,2)

            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4), gridspec_kw = {'width_ratios':[1, 2, 2, 2]})
            #plt.subplots_adjust(wspace=0.28,left=0.05,right=0.98)

            #ax[0].set_aspect(0.7)
            ax[0].scatter(BJDmodP_extended,flux_extended,s=1)
            ax[0].set_xlim(0,1.5)
            ax[0].set_xlabel("Phase")
            ax[0].set_ylabel("Brightness")
            if scale == 'mag': ax[0].invert_yaxis()

            #ax[1].set_title(os.path.basename(line))
            ax[1].plot(np.insert(self.time,gapatorig,np.nan),np.insert(self.flux,gapatorig,np.nan))
            ax[1].set_xlabel("Time")
            #ax[1].set_ylabel("flux")
            ax[1].errorbar(np.insert(BJDmidP,gapat,np.nan),np.insert(a0values,gapat,np.nan),
                            np.insert(a0errorvalues,gapat,np.nan) if showerrorbar else None,c='C1',alpha=0.5,label="ZP",ecolor='lightgray')
            if scale == 'mag': ax[1].invert_yaxis()

            ax[2].errorbar(np.insert(BJDmidP,gapat,np.nan),np.insert(avalues,gapat,np.nan),
                           np.insert(aerrorvalues,gapat,np.nan) if showerrorbar else None,c='r',label="Amp",ecolor='lightgray')
            ax[2].set_xlabel("Time")
            #ylmin,ylmax = ax[2].get_ylim()
            #ylmin = min(ylmin,0.85)
            #ylmax = max(ylmax,1.15)
            #ax[2].set_ylim(ylmin,ylmax)
            ax[2].set_title('P='+str(round(period,8))+' days')
            ylmin,ylmax = ax[2].get_ylim()
            ylmin = min(ylmin,avalues.mean()-10*aerrorvalues.mean())
            ylmax = max(ylmax,avalues.mean()+10*aerrorvalues.mean())
            ax[2].set_ylim(ylmin,ylmax)

            ax2b=ax[2].twinx()
            ax2b.errorbar(np.insert(BJDmidP,gapat,np.nan),np.insert(psivalues,gapat,np.nan),
                          np.insert(psierrorvalues,gapat,np.nan) if showerrorbar else None,label=r"$\Phi$",ecolor='lightgray',c='C0')
            ylmin,ylmax = ax2b.get_ylim()
            ylmin = min(ylmin,psivalues.mean()-0.10)
            ylmax = max(ylmax,psivalues.mean()+0.10)
            ax2b.set_ylim(ylmin,ylmax)

            ax[2].legend(loc='upper left')
            ax2b.legend(loc='upper right')

            colorline( np.insert(avalues,gapat,np.nan),np.insert(psivalues,gapat,np.nan),ax=ax[3])
            ax[3].errorbar(avalues,psivalues,
                           xerr=aerrorvalues if showerrorbar else None,yerr=psierrorvalues if showerrorbar else None,ecolor='lightgray',fmt='.',ms=0,zorder=0)
            ax[3].set_xlim(np.min(avalues) - 0.05*np.ptp(avalues), np.max(avalues) + 0.05*np.ptp(avalues))
            ax[3].set_ylim(np.min(psivalues) - 0.05*np.ptp(psivalues), np.max(psivalues) + 0.05*np.ptp(psivalues))
            ax[3].set_xlabel("Amp")
            ax[3].set_ylabel("Phase")

            plt.tight_layout()
            if saveplot:
                plt.savefig(filename+'_template_OC.jpg')

            if plotting:
                plt.show()

            plt.close(fig)

        # ----- Save OC -----
        if saveresult:
            period=1/LSPfreq

            np.savetxt(filename+'_template_OC.txt',
                        np.c_[BJDmidP,avalues,aerrorvalues,psivalues,psierrorvalues,a0values,a0errorvalues],
                        fmt='%.8f',
                        header='Calculated with period of %.6f\nTIME AVALS AVALS_ERR PSIVALS PSIVALS_ERR ZP ZP_ERR' % period)

        # Store OC curve results
        self.oc_time        = np.array(BJDmidP)
        self.avalues        = avalues
        self.aerrorvalues   = aerrorvalues
        self.psivalues      = psivalues
        self.psierrorvalues = psierrorvalues
        self.a0values       = a0values
        self.a0errorvalues  = a0errorvalues

        return np.array(BJDmidP), avalues,aerrorvalues, psivalues,psierrorvalues, a0values,a0errorvalues

    def get_lc_model(self, time=None, amp=None, phase=None, zp=None):
        """
        Get modulated model light curve.

        Parameters
        ----------
        time : array
            Time points where modulated model light curve is desired.
        amp : array
            Amplitude variation by time.
        phase : array
            Phase variation by time.
        zp : array
            Zero point variation by time.

        Returns:
        -------
        ymodel : array
            Modulated model light curve.
        """
        if not hasattr(self,"pfit"):
            warnings.warn("Please run \'fit\' first!")
            return None

        nharmonics = (len(self.pfit)-2)//2

        if time is None:
            time  = self.oc_time
            amp   = self.avalues
            phase = self.psivalues
            zp    = self.a0values

        ymodel = modulated_lc_model( time,
                                    zp/self.pfit[-1], amp/self.pfit[1], phase-self.pfit[nharmonics+1],
                                    self.pfit, self.kind)

        return ymodel

    def get_lc_model_interp(self,kind='slinear'):
        """
        Get modulated model light curve interpolated at the original time points.

        Parameters
        ----------
        kind : str or int, optional
            Specifies the kind of interpolation. Default is ‘slinear’.
            See `scipy.interpolate.interp1d` for the kinds.

        Returns:
        -------
        ymodel : array
            Modulated model light curve interpolated at the original time points.
        """

        from scipy.interpolate import interp1d

        goodpts = np.isfinite(self.avalues)
        ampinterp = interp1d(self.oc_time[goodpts],self.avalues[goodpts],
                            kind=kind,fill_value="extrapolate")

        goodpts = np.isfinite(self.psivalues)
        phiinterp = interp1d(self.oc_time[goodpts],self.psivalues[goodpts],
                            kind=kind,fill_value="extrapolate")

        goodpts = np.isfinite(self.a0values)
        zpinterp = interp1d(self.oc_time[goodpts],self.a0values[goodpts],
                            kind=kind,fill_value="extrapolate")

        ymodel = self.get_lc_model(self.time, ampinterp(self.time), phiinterp(self.time), zpinterp(self.time))

        return ymodel
