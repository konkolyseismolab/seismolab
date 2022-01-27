import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from astropy.stats import LombScargle

import warnings
warnings.filterwarnings("ignore")

from fourierfitter import FourierFitter

def splitthem(inputBJD, inputflux,fluxerror,span,step,n):
    if fluxerror is None:
        df = pd.DataFrame({'BJD':inputBJD,'flux':inputflux})
    else:
        df = pd.DataFrame({'BJD':inputBJD,'flux':inputflux,'fluxerror':fluxerror})

    dfbit = df[ (df['BJD']>=df['BJD'].iloc[0]+step*n) & (df['BJD']<df['BJD'].iloc[0]+step*n+span) ]
    dfbitlistBJD       = dfbit['BJD'].to_numpy()
    dfbitlistflux      = dfbit['flux'].to_numpy()
    if fluxerror is None:
        dfbitlistfluxerror = None
    else:
        dfbitlistfluxerror = dfbit['fluxerror'].to_numpy()

    return dfbitlistBJD, dfbitlistflux, dfbitlistfluxerror

def FFF(time, a0, a, dPhi, pfit, kind):
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
    value = np.median(data)
    ep = np.percentile(data,84.1)-value
    em = value-np.percentile(data,15.9)
    return value,ep,em

class TemplateFitter:
    def __init__(self, time,flux,fluxerror=None):
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
        span = 5,
        step = 3,
        error_estimation='analytic',

        nfreq = 5,
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

        debug=False
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

        nfreq : int, default: 5
            Max number of harmonics to be used in template.
        minimum_frequency : float, optional
            If specified, then use this minimum frequency rather than one
            chosen based on the size of the baseline.
        maximum_frequency : float, optional
            If specified, then use this maximum frequency rather than one
            chosen based on the average nyquist frequency.
        nyquist_factor : float, optional, default 1
            The multiple of the average nyquist frequency used to choose the
            maximum frequency if maximum_frequency is not provided.
        samples_per_peak : float, optional, default 100
            The approximate number of desired samples across the typical peak.
        kind : 'sin' or 'cos', default 'sin'
            Function type to construct template.

        plotting : bool, deaful: False
            Show result.
        scale: 'mag' or 'flux', default 'flux'
            Lightcurve plot scale.
        saveplot : bool, default: False
            Save result as txt file.
        saveresult : bool, default: False
            Save results as txt
        filename : str, default 'result'
            Beginning of txt filename.

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

        # Initialize fitter by passing your light curve
        if debug: print('Calculating Lomb-Scargle...')
        fitter = FourierFitter(self.time,self.flux)

        pfit,perr = fitter.fit_freqs(nfreq = nfreq,
                                 plotting = debug,
                                 minimum_frequency=minimum_frequency,
                                 maximum_frequency=maximum_frequency,
                                 nyquist_factor=nyquist_factor,
                                 samples_per_peak=samples_per_peak,
                                 kind=kind)

        LSPfreq = pfit[0]

        if debug:
            plt.title('Template')
            plt.scatter(self.time, self.flux)
            plt.plot(self.time,fitter.fit_all(self.time,*pfit,fitter.kind),c='C1')
            plt.xlim(self.time.max()-3/pfit[0],self.time.max())
            plt.show()

        BJDmidP=[]
        a0values=[]
        a0errorvalues=[]
        avalues=[]
        aerrorvalues=[]
        psivalues=[]
        psierrorvalues=[]

        counter=0
        while True:
            if debug: print('Splitting them...')

            if self.time[-1] < self.time[0]+step*1/LSPfreq*counter:
                break

            bitBJD,bitflux,bitfluxerror=splitthem(self.time,self.flux,self.fluxerror,
                                                    span=span*1/LSPfreq,
                                                    step=step*1/LSPfreq,
                                                    n=counter)
            counter+=1

            if debug: print('N points:',len(bitBJD))
            if len(bitBJD)<10: continue

            if error_estimation == 'montecarlo':
                print('Running MCMC...')
                import pymc3 as pm
                import seaborn as sns

                with pm.Model() as model:
                    ## define Normal priors to give Ridge regression
                    a0 = pm.Uniform("a0", 0., 2., testval=1.)
                    a = pm.Uniform("a", 0., 2., testval=1.)
                    psi = pm.Uniform("psi", -1., 1., testval=0.)

                    ## define model
                    yest = FFF(bitBJD, a0, a, psi, pfit, kind)

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


                BJDmidP.append( np.min(bitBJD) + np.ptp(bitBJD)/2 )

                a0values.append(a0)
                a0errorvalues.append(max(a0ep,a0em))

                avalues.append(a)
                aerrorvalues.append(max(aep,aem))

                psivalues.append(psi)
                psierrorvalues.append(max(psiep,psiem))

            else:
                _pfit, _pcov = optimize.curve_fit(lambda x, a0, a, psi: FFF(x, a0, a, psi, pfit, kind),
                                                  bitBJD,bitflux,
                                                  p0=[0.9,0.9,0.1],
                                                  sigma=bitfluxerror,absolute_sigma=True,method='trf')

                perr_curvefit = np.sqrt(np.diag(_pcov))

                BJDmidP.append( np.min(bitBJD) + np.ptp(bitBJD)/2 )

                a0values.append(_pfit[0])
                a0errorvalues.append(perr_curvefit[0])

                avalues.append(_pfit[1])
                aerrorvalues.append(perr_curvefit[1])

                psivalues.append(_pfit[2])
                psierrorvalues.append(perr_curvefit[2])

            if debug:

                plt.figure()
                plt.title('Fit to subsample %d' % (counter+1))
                plt.scatter(bitBJD,bitflux)

                xxxx = np.linspace(min(bitBJD),max(bitBJD),1000)

                plt.plot(xxxx,FFF(xxxx, a0values[-1], avalues[-1], psivalues[-1], pfit, kind))
                plt.plot(xxxx,FFF(xxxx, a0values[-1], avalues[-1], 0, pfit, kind))

                plt.show()

        nfreq = (len(pfit)-2)//2
        a0values  = pfit[-1] * np.asarray(a0values)
        avalues   = pfit[1] * np.asarray(avalues)
        psivalues = pfit[1+nfreq] + np.asarray(psivalues)
        a0errorvalues  = perr[-1] + np.asarray(a0errorvalues)
        aerrorvalues   = perr[1] + np.asarray(aerrorvalues)
        psierrorvalues = perr[1+nfreq] + np.asarray(psierrorvalues)

        period=1/LSPfreq
        BJDmodP_extended = self.time%period/period
        BJDmodP_extended = np.concatenate((BJDmodP_extended,1+BJDmodP_extended))
        flux_extended = np.tile(self.flux,2)


        if plotting or saveplot:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4), gridspec_kw = {'width_ratios':[1, 2, 2]})
            #plt.subplots_adjust(wspace=0.28,left=0.05,right=0.98)

            #ax[0].set_aspect(0.7)
            ax[0].scatter(BJDmodP_extended,flux_extended,s=1)
            ax[0].set_xlim(0,1.5)
            ax[0].set_xlabel("Phase")
            ax[0].set_ylabel("Brightness")
            if scale == 'mag': ax[0].invert_yaxis()

            #ax[1].set_title(os.path.basename(line))
            ax[1].plot(self.time,self.flux)
            ax[1].set_xlabel("Time")
            #ax[1].set_ylabel("flux")
            ax[1].errorbar(BJDmidP,a0values,a0errorvalues,c='C1',alpha=0.5,label="ZP")
            if scale == 'mag': ax[1].invert_yaxis()

            ax[2].errorbar(BJDmidP,avalues,aerrorvalues,c='r',label="Amp")
            ax[2].set_xlabel("Time")
            #ylmin,ylmax = ax[2].get_ylim()
            #ylmin = min(ylmin,0.85)
            #ylmax = max(ylmax,1.15)
            #ax[2].set_ylim(ylmin,ylmax)
            ax[2].set_title('P='+str(round(period,8)))

            ax2b=ax[2].twinx()
            ax2b.errorbar(BJDmidP,psivalues,psierrorvalues,label="$\Phi$")
            ylmin,ylmax = ax2b.get_ylim()
            ylmin = min(ylmin,psivalues.mean()-0.15)
            ylmax = max(ylmax,psivalues.mean()+0.15)
            ax2b.set_ylim(ylmin,ylmax)

            ax[2].legend(loc='upper left')
            ax2b.legend(loc='upper right')

            plt.tight_layout()
            if saveplot:
                plt.savefig(filename+'_template_OC.jpg')

            if plotting:
                plt.show()

            plt.close(fig)


        if saveresult:
            np.savetxt(filename+'_template_OC.txt',
                        np.c_[BJDmidP,avalues,aerrorvalues,psivalues,psierrorvalues,a0values,a0errorvalues],
                        fmt='%.8f',
                        header='Calculated with period of %.6f\nTIME AVALS AVALS_ERR PSIVALS PSIVALS_ERR ZP ZP_ERR' % period)

        return BJDmidP, avalues,aerrorvalues, psivalues,psierrorvalues, a0values,a0errorvalues