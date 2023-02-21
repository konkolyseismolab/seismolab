import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed
import warnings
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from statsmodels.nonparametric.kernel_regression import KernelReg

from scipy.stats import binned_statistic
from scipy.optimize import minimize

from .shift_curves import shift_phase_curves_vertically

__all__ = ['OCFitter']

def chi2model(params,x,y,sig,pol):
    xoffset = params[0]
    yoffset = params[1]
    model = pol(x + xoffset) + yoffset
    chi2 = (np.power(y - model,2) / (sig**2)).sum()

    return chi2

def mintime_parallel(params):
    """
    Refit minima with generating new observations from noise
    """
    fittype = params[-1]
    if fittype=='model':
        x,y,err,pol,zero_time,x0,y0 = params[:7]
        phaseoffset,i,period = params[7:-1]
    else:
        x,y,order,zero_time = params[:4]
        bound1, bound2 = params[4:-1]

    if fittype=='nonparametric':
        ksrmv = KernelReg(endog=y, exog=x, var_type='c',
                          reg_type='ll', bw=np.array([np.median(np.diff(x))]) )
        p_fit = lambda x : ksrmv.fit(np.atleast_1d(x))[0][0] if isinstance(x,float) else ksrmv.fit(np.atleast_1d(x))[0]

        result = minimize_scalar(p_fit, bounds=(bound1, bound2), method='bounded')
        t = result.x + zero_time
    elif fittype=='model':
        res = minimize(chi2model, (x0,y0), args=(x,y,err,pol) ,
                       method='Powell',
                       bounds=((-period*0.1,period*0.1),(-np.inf,np.inf)))

        #yoffset = res.x[1]
        xoffset = res.x[0]

        t = zero_time +phaseoffset +(i-1)*period -xoffset
    else:
        with warnings.catch_warnings(record=True):
            y_model = np.polyfit(x,y,order)
            p_fit = np.poly1d(y_model)

        result = minimize_scalar(p_fit, bounds=(bound1, bound2), method='bounded')
        t = result.x + zero_time

    return t

class OCFitter:
    def __init__(self,time,flux,fluxerror,period):
        '''
        time : array
            Light curve time points.
        flux : array
            Corresponding flux/mag values.
        fluxerror : array, optional
            Corresponding flux/mag error values.
        period : float
            Period of given variable star.
        '''

        self.period = float(period)

        time = np.asarray(time,dtype=float)
        flux = np.asarray(flux,dtype=float)
        fluxerror = np.asarray(fluxerror,dtype=float)

        goodpts = np.isfinite(time)
        goodpts &= np.isfinite(flux)
        goodpts &= np.isfinite(fluxerror)

        self.x = time[goodpts]
        self.y = flux[goodpts]
        self.err = fluxerror[goodpts]

    def get_model(self,phase=0,show_plot=False,smoothness=1):
        times = self.x.copy()
        zero_time = np.floor(times[0])
        times -= zero_time

        flux = self.y.copy()
        fluxerr = self.err.copy()

        period = self.period

        # Loop over each cycle and shift them vertically to match each other
        corrflux = shift_phase_curves_vertically(times, flux, fluxerr, period)

        # Shift minimum to middle of the phase curve
        times -= phase
        times += period/2

        # Bin phase shifted phase curve
        ybinned,xbinned,_ = binned_statistic(times%period,corrflux,statistic='median', bins=100, range=(0,period))
        xbinned = (xbinned[1:] + xbinned[:-1])/2

        xbinned += phase
        xbinned -= period/2

        goodpts = np.where( np.isfinite(ybinned) )[0]
        xbinned = xbinned[goodpts]
        ybinned = ybinned[goodpts]

        # Get model fit
        ksrmv = KernelReg(endog=ybinned, exog=xbinned, var_type='c',
                          reg_type='ll', bw=smoothness*np.array([np.median(np.diff(xbinned))]) )
        pol = lambda x : ksrmv.fit(np.atleast_1d(x))[0][0] if isinstance(x,float) else ksrmv.fit(np.atleast_1d(x))[0]

        if show_plot:
            phasetoplot = times%period +phase -period/2
            x2plot = np.linspace(phasetoplot.min(),phasetoplot.max(),1000)

            plt.figure(figsize=(10,6))
            ax = plt.subplot(111)
            plt.title('Light curve model to be shifted to each minimum')
            plt.plot( phasetoplot, flux , '.', c='lightgray',label='Original data')
            plt.plot( phasetoplot, corrflux , 'k.',label='Veritically shifted')
            plt.plot(xbinned,ybinned,'.',label='Binned')
            plt.plot( x2plot ,pol(x2plot) ,label='Model')
            plt.xlabel('Cycle (= one period)')
            plt.ylabel('Brightness')
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        return pol , phase

    def fit_minima(self,
                    fittype='model',
                    phase_interval=0.1,
                    order=3,
                    smoothness=1,

                    epoch='auto',

                    npools=-1,
                    samplings=100,

                    showplot=False,
                    saveplot=False,
                    showfirst=False,
                    filename='',

                    debug=False):
        """
        Fit all minima(!) one by one.

        Parameters
        ----------
        fittype : 'poly', 'nonparametric' or 'model'.
            The type of the fitted function.
            - `poly` fits given order polynomial to each minimum individually.
              Requires the `order` to be set.
            - `nonparametric` fits a smooth function to each minimum individually.
              Very sensitive to outliers! Requires `smoothness` to be set.
            - `model` fits a smooth function to the median of phase folded light curve.
              The resulted function is shifted to each minimum.
              Error estimation is very slow, set `samplings` to ~100.
        phase_interval : float
            The phase interval around an expected minimum, which is
            used to fit the selected function.
        order : int
            Order of the polynomial to be fitted to each minimum.
            Applies only if `fittype` is `poly`.
        smoothness : float
            The smoothness of fitted nonparametric function.
            Use ~1, to follow small scale (noise-like) variations.
            Use >1 to fit a really smooth function.
            Applies only if `fittype` is `nonparametric` or `model`.
        epoch : float or 'auto'
            The time stamp of the first minimium.
            If `auto`, then it is inferred automatically by fitting a model.
        npools : int, default: -1
            Number of cores during error estimation.
            If `-1`, then all cores are used.
        samplings : int, default: 100000
            Number of resamplings for error estimation.
        showplot : bool, default: False
            Show each fitted minima and other useful plots.
        saveplot : bool, default: True
            Save all plots.
        filename : str, default: ''
            Filename to be used to save plots.
        showfirst : bool, default: True
            Show epoch estimation and first cycle fit
            to check parameters of the fitted function.

        Returns:
        -------
        times_of_minimum : array
            The calculated minimum times.
        error_of_minimum : array
            The error of the minimum times.
        """

        if fittype not in ['poly','nonparametric','model']:
            raise NameError('Fittype is not known! Use \'poly\', \'nonparametric\' or \'model\'.')

        if npools == -1:
            npools = cpu_count()

        print('Calculating minima times...')
        if saveplot: pp = PdfPages(filename+'_minima_fit.pdf')

        x = self.x
        y = self.y
        err = self.err
        period = self.period

        cadence = np.median(np.diff(x))

        zero_time = np.floor(x[0])

        # List to store times of minima
        mintimes = []

        ####################################################
        # Fit phase folded and binned lc to estimate epoch #
        ####################################################
        if epoch == 'auto':
            self.epoch = None

            # Loop over each cycle and shift them vertically to match each other
            corrflux = shift_phase_curves_vertically(x, y, err, period)

            # Estimate epoch from minimum of binned lc
            ybinned,xbinned,_ = binned_statistic((x-x[0])%period, corrflux,
                                                statistic='median',bins=100,range=(0,period))
            xbinned = (xbinned[1:] + xbinned[:-1])/2

            mean_t_at = np.nanargmin(ybinned)
            mean_t = xbinned[mean_t_at]+x[0]

            if showfirst or showplot or debug:
                plt.figure(figsize=(8,6))
                plt.suptitle("Initial epoch based on shifted and binned light curve")
                plt.plot((x-x[0])%period +x[0],corrflux,'.',label='Original')
                plt.plot(xbinned+x[0],ybinned,'.',label='Shifted and Binned')
                plt.axvline( mean_t ,c='r')
                plt.xlabel('Time')
                plt.ylabel('Brightness')
                plt.legend()
                plt.show()

            # Fit the data within this phase interval
            pm = period*max(0.1,phase_interval) #days

            um=np.where((mean_t-pm<x) & (x<=mean_t+pm))[0]

            pol,phaseoffset = self.get_model(phase=mean_t-zero_time,show_plot=debug,smoothness=smoothness)

            with warnings.catch_warnings(record=True):
                y0 = np.mean(y[um]) - np.mean(pol(x[um]-zero_time))
                x0 = 0
                res = minimize(chi2model, (x0,y0),
                               args=(x[um]-zero_time, y[um],err[um],pol),
                               method='Powell',
                               bounds=((-period*0.1,period*0.1),(-np.inf,np.inf)))

            yoffset = res.x[1]
            xoffset = res.x[0]

            mean_t = mean_t - xoffset

            '''

            if fittype=='nonparametric':
                ksrmv = KernelReg(endog=y[um], exog=x[um]-zero_time, var_type='c',
                                  reg_type='ll', bw=np.array([np.median(np.diff(x[um]))]) )
                p = lambda x : ksrmv.fit(np.atleast_1d(x))[0][0] if isinstance(x,float) else ksrmv.fit(np.atleast_1d(x))[0]
            else:
                with warnings.catch_warnings(record=True):
                    z = np.polyfit(x[um]-zero_time, y[um], 5)
                p = np.poly1d(z)

            result = minimize_scalar(p, bounds=(mean_t-zero_time-pm, mean_t-zero_time+pm), method='bounded')
            mean_t = result.x + zero_time

            '''
        else:
            try:
                mean_t = float(epoch)
                self.epoch = epoch
            except ValueError:
                raise ValueError("Epoch must be `auto` or a number!")

        if showplot or showfirst:
            umcycle = (x[0]<=x) & (x<=x[0]+period)

            plt.figure(figsize=(8,6))
            plt.scatter(x[umcycle],y[umcycle],c='k',label='First cycle')
            if epoch == 'auto':
                x2plot = np.linspace(x[um].min(),x[um].max(),100)
                plt.plot(x2plot,pol(x2plot-zero_time+xoffset)+yoffset,label='Model')
            plt.axvline( mean_t, c='r',label='Epoch',zorder=0 )
            plt.xlabel('Time')
            plt.ylabel('Brightness')
            if epoch == 'auto':
                plt.suptitle('Estimating epoch by fitting a model')
            else:
                plt.suptitle('Using the given epoch')
            plt.legend()
            plt.show()

        #######################################
        # Fit each cycle to get minimum times #
        #######################################

        # Initialize progress bar
        _total = np.sum(np.diff(x[y<np.mean(y)]%period/period)<0)
        pbar = tqdm(total=_total,desc='Fitting cycles')

        # Range to be fitted around the expected minimum:
        pm = period*phase_interval #days

        if fittype=='model':
            pol,phaseoffset = self.get_model(phase=mean_t-zero_time,smoothness=smoothness,
                                             show_plot=showfirst or showplot)

        i=1 #First minimum
        firstmin = True
        while True:
            # If duty cycle is lower than 20% do not fit
            dutycycle = 0.2
            um = np.where((mean_t-pm<x) & (x<=mean_t+pm)  )[0]
            if len(x[um])<(pm/cadence*dutycycle):
                mean_t = mean_t + period
                i=i+1
                if mean_t> np.max(x):
                    break
                else:
                    continue

            ###################################################
            # First fit the data around expected minimum time #
            ###################################################
            if fittype=='nonparametric':
                ksrmv = KernelReg(endog=y[um], exog=x[um]-zero_time, var_type='c',
                                  reg_type='ll', bw=smoothness*np.array([np.median(np.diff(x[um]))]) )
                p = lambda x : ksrmv.fit(np.atleast_1d(x))[0][0] if isinstance(x,float) else ksrmv.fit(np.atleast_1d(x))[0]

                try:
                    result = minimize_scalar(p, bounds=(mean_t-zero_time-pm, mean_t-zero_time+pm), method='bounded')
                except np.linalg.LinAlgError as w:
                    warnings.warn( \
                        str(w) + "\nSkipping this minimum! Try to use a larger phase interval to avoid problems!")

                    mean_t = mean_t + period
                    i=i+1
                    continue

                t_initial = result.x + zero_time
            elif fittype=='model':
                with warnings.catch_warnings(record=True):
                    y0 = np.mean(y[um]) - np.mean(pol(x[um]-zero_time -(i-1)*period))
                    x0 = 0
                    res = minimize(chi2model, (x0,y0),
                                   args=(x[um]-zero_time -(i-1)*period , y[um],err[um],pol),
                                   method='Powell',
                                   bounds=((-period/2,period/2),(-np.inf,np.inf)))

                yoffset = res.x[1]
                xoffset = res.x[0]

                t_initial = zero_time +phaseoffset +(i-1)*period -xoffset

                if debug:
                    plt.title('First fit')
                    plt.plot(x[um],y[um],'.')
                    xtobeplotted = np.linspace( x[um].min(),x[um].max(), 1000 )
                    plt.plot(xtobeplotted,pol(xtobeplotted-zero_time-(i-1)*period+x0 ) + y0 ,'r',label='Initial')
                    plt.plot(xtobeplotted,pol(xtobeplotted-zero_time-(i-1)*period + xoffset) + yoffset,'k',label='Final fit')
                    plt.axvline(t_initial,c='r')
                    plt.xlim(x[um][0],x[um][-1])
                    #plt.ylim(y[um].min() - 0.1*y[um].ptp(), y[um].max() + 0.1*y[um].ptp() )
                    plt.legend()
                    plt.show()
                    plt.close('all')
            else:
                with warnings.catch_warnings(record=True):
                    z = np.polyfit(x[um]-zero_time, y[um], order)
                p = np.poly1d(z)

                result = minimize_scalar(p, bounds=(mean_t-zero_time-pm, mean_t-zero_time+pm), method='bounded')
                t_initial = result.x + zero_time

                if debug:
                    plt.title('First fit')
                    plt.plot(x[um],y[um],'.')
                    xtobeplotted = np.linspace( x[um].min(),x[um].max(), 1000 )
                    plt.plot(xtobeplotted,p(xtobeplotted-zero_time ) ,'r',label='Polyfit')
                    plt.axvline(t_initial,c='r')
                    plt.xlim(x[um][0],x[um][-1])
                    #plt.ylim(y[um].min() - 0.1*y[um].ptp(), y[um].max() + 0.1*y[um].ptp() )
                    plt.legend()
                    plt.show()
                    plt.close('all')


            # Continue if duty cycle is lower than 20%
            um = np.where((t_initial-pm<x) & (x<=t_initial+pm)  )
            um_before = np.where((t_initial-pm<x) & (x<=t_initial)  )
            um_after  = np.where((t_initial<x) & (x<=t_initial+pm)  )
            if len(x[um_before])<(pm/cadence*dutycycle) or len(x[um_after])<(pm/cadence*dutycycle):
                mean_t = mean_t + period
                i=i+1

                continue

            ########################################################
            # Second fit the data again around fitted minimum time #
            ########################################################
            if fittype=='nonparametric':
                ksrmv = KernelReg(endog=y[um], exog=x[um]-zero_time, var_type='c',
                                  reg_type='ll', bw=smoothness*np.array([np.median(np.diff(x[um]))]) )
                p = lambda x : ksrmv.fit(np.atleast_1d(x))[0][0] if isinstance(x,float) else ksrmv.fit(np.atleast_1d(x))[0]

                result = minimize_scalar(p, bounds=(t_initial-zero_time-pm, t_initial-zero_time+pm), method='bounded')
                t = result.x + zero_time
            elif fittype=='model':
                with warnings.catch_warnings(record=True):
                    y0 = yoffset
                    x0 = xoffset
                    res = minimize(chi2model, (x0,y0),
                                   args=(x[um]-zero_time-(i-1)*period , y[um],err[um],pol),
                                   method='Powell',
                                   bounds=((-period*0.1,period*0.1),(-np.inf,np.inf)))

                yoffset = res.x[1]
                xoffset = res.x[0]

                t = zero_time +phaseoffset +(i-1)*period -xoffset

                if debug:
                    plt.title('Second fit')
                    plt.plot(x[um],y[um],'.')
                    xtobeplotted = np.linspace( x[um].min(),x[um].max(), 1000 )
                    plt.plot(xtobeplotted,pol(xtobeplotted-zero_time-(i-1)*period +x0)+y0 ,label='Initial')
                    plt.plot(xtobeplotted,pol(xtobeplotted-zero_time-(i-1)*period + xoffset) + yoffset ,label='Final fit')
                    plt.axvline(t,c='r')
                    plt.legend()
                    plt.show()
                    plt.close('all')
            else:
                with warnings.catch_warnings(record=True):
                    z = np.polyfit(x[um]-zero_time, y[um], order)
                p = np.poly1d(z)

                result = minimize_scalar(p, bounds=(t_initial-zero_time-pm, t_initial-zero_time+pm), method='bounded')
                t = result.x + zero_time


                if debug:
                    plt.title('Second fit')
                    plt.plot(x[um],y[um],'.')
                    xtobeplotted = np.linspace( x[um].min(),x[um].max(), 1000 )
                    plt.plot(xtobeplotted,p(xtobeplotted-zero_time ) ,'r',label='Polyfit')
                    plt.axvline(t_initial,c='r')
                    plt.xlim(x[um][0],x[um][-1])
                    #plt.ylim(y[um].min() - 0.1*y[um].ptp(), y[um].max() + 0.1*y[um].ptp() )
                    plt.legend()
                    plt.show()
                    plt.close('all')

            #######################
            # Stopping conditions #
            #######################
            # Break if the data is over
            if mean_t> np.max(x):
                break

            # Continue if the number of points is low (duty cycle is lower than 20%)
            um_before = np.where((t-pm<x) & (x<=t)  )
            um_after = np.where((t<x) & (x<=t+pm)  )
            if len(x[um_before])<(pm/cadence*0.2) or len(x[um_after])<(pm/cadence*0.2) or len(x[um])<(pm/cadence*0.2):
                mean_t = mean_t + period
                i=i+1

                continue

            # Continue if there are too few points
            if len(y[um]) < 3:
                mean_t = mean_t + period
                i=i+1

                continue

            # Continue if fit is not a minimum
            first_point = y[um][0]
            last_point = y[um][-1]
            middle_point = np.min(y[um][1:-1])
            if not (middle_point<=first_point and middle_point<=last_point):
                mean_t = mean_t + period
                i=i+1

                continue

            ###########################################################
            # Calculate error by sampling from y errors and refitting #
            ###########################################################
            z_fit_parallel = []
            if fittype=='model':
                for _ in range(samplings):
                    y_resampled = y[um] + np.random.normal(loc=0,scale=err[um],size=err[um].shape[0])
                    z_fit_parallel.append([x[um]-zero_time-(i-1)*period, y_resampled, err[um], pol, zero_time,
                                           xoffset, yoffset, phaseoffset, i, period, fittype ])
            else:
                for _ in range(samplings):
                    y_resampled = y[um] + np.random.normal(loc=0,scale=err[um],size=err[um].shape[0])
                    z_fit_parallel.append([x[um]-zero_time, y_resampled, order, zero_time, t-zero_time-pm, t-zero_time+pm , fittype ])

            t_trace = Parallel(n_jobs=npools)(delayed(mintime_parallel)(par) for par in z_fit_parallel)
            t_trace = np.array(t_trace)

            try:
                del z_fit_parallel
                del y_resampled

                OC_err = np.median(t_trace)-np.percentile(t_trace,15.9)
            except UnboundLocalError:
                OC_err = 0

            #Append minimum time
            mintimes.append([t,OC_err])

            ################
            # Plot the fit #
            ################
            #plt.plot(x-zero_time,y,'o',c='gray')
            plt.errorbar(x[um]-zero_time,y[um],yerr=err[um],color='k',fmt='.',zorder=0,ecolor='lightgray')
            #plt.plot(x[um_before]-zero_time,y[um_before],'m.',zorder=5)
            if fittype=='model':
                xtobeplotted = np.linspace( x[um].min(),x[um].max(), 1000 )
                plt.plot(xtobeplotted-zero_time,pol(xtobeplotted-zero_time +xoffset -(i-1)*period)+yoffset ,c='r',zorder=10,label='Model')
            else:
                xtobeplotted = np.linspace( (x[um]-zero_time).min(),(x[um]-zero_time).max(), 1000 )
                plt.plot(xtobeplotted,p(xtobeplotted),c='r',zorder=10,label=fittype)
            plt.axvline(t-zero_time,zorder=0,label='Observed min')
            if firstmin:
                plt.suptitle('First cycle to check phase interval and model')
                epoch = t-zero_time-(i-1)*period
            else:
                plt.suptitle('%d. cycle' % (i))
                plt.axvline(epoch+(i-1)*period,c='lightgray',zorder=0,label='Calculated')
            #plt.axvline(t_initial_final-zero_time)
            plt.xlabel('Time')
            plt.ylabel('Brightness')
            plt.legend()
            if saveplot: plt.savefig(pp,format='pdf',dpi=300)
            if showplot or (firstmin and showfirst): plt.show()
            plt.close()

            firstmin = False

            ############################
            # Step to the next minimum #
            ############################
            pbar.update()

            mean_t = mean_t + period
            i=i+1
            #If the data is over, break
            if mean_t > np.max(x):
                break

        pbar.close()

        if saveplot: pp.close()

        mintimes = np.array(mintimes)
        time_of_minimum = mintimes[:,0]
        err_of_minimum = mintimes[:,1]

        print("Done!")

        self.min_times = time_of_minimum
        self.min_times_err = err_of_minimum

        return time_of_minimum,err_of_minimum

    def calculate_OC(self,
                    min_times=None,
                    period=None,
                    epoch=None,
                    min_times_err=None,

                    showplot=False,
                    saveplot=False,
                    saveOC=False,
                    filename=''):
        """
        Calculate O-C curve from given period and minimum times.

        Parameters
        ----------
        min_times : array, optional
            Observed (O) times of minima.
        period : float, optional
            Period to be used to construct calculated (C) values.
        epoch : float, default: first `min_times` value, optional
            Epoch to be used to construct calculated (C) values.
            If note given, then the first minimum time is
            used as epoch.
        min_times_err : array, optional
            Error of observed (O) times of minima.

        showplot : bool, default: False
            Show results.
        saveplot : bool, deaful: False
            Save results.
        saveOC : bool, default: True
            Save constructed OC as txt file.
        filename : str, default: ''
            Beginning of txt filename.

        Returns:
        -------
        mid_times : array
            The given minimum times.
        OC : array
            The calculated O-C values.
        OCerr : array
            If `min_times_err` was given, the error of the O-C values.
        """
        print('Calculating the O-C...')

        if min_times is None:
            min_times = self.min_times
            min_times_err = self.min_times_err

        if epoch is None:
            epoch = self.epoch

        if period is None:
            period = self.period
        period = float(period)

        min_times = min_times[ min_times.argsort() ]
        if min_times_err is not None:
            min_times_err = min_times_err[ min_times.argsort() ]

        OC_all = [] #List to store OC values

        if epoch is None:
            t0 = min_times.min()
        else:
            t0 = epoch

        data_length = min_times.max() - min_times.min()

        i=0
        for t in min_times:
            #Calculate O-C value
            OC = (t-t0)-i*period
            while True:
                if np.abs(OC)>0.9*period:
                    i=i+1
                    OC = (t-t0)-i*period
                    if np.abs(OC) > data_length:
                        # if OC value not converged
                        OC = np.nan
                        break
                    continue
                else:
                    break
            i=i+1
            OC_all.append( np.array([t,OC]) )

        OC_all = np.array(OC_all)
        if min_times_err is not None and saveOC:
            np.savetxt(filename+'_OC.txt',np.c_[ OC_all,min_times_err] )
        elif saveOC:
            np.savetxt(filename+'_OC.txt',OC_all )

        # Plot to get ylim without errorbars
        plt.plot(OC_all[:,0],OC_all[:,1],'.',ms=0)
        yliml1,ylimu1 = plt.gca().get_ylim()
        plt.clf()

        plt.errorbar(OC_all[:,0],OC_all[:,1],yerr=min_times_err,fmt='.',ecolor='lightgray')
        yliml2 = np.nanmin(OC_all[:,1]) \
                - (2*np.nanmedian(min_times_err) if min_times_err is not None else np.nanstd(OC_all[:,1]))
        ylimu2 = np.nanmax(OC_all[:,1]) \
                + (2*np.nanmedian(min_times_err) if min_times_err is not None else np.nanstd(OC_all[:,1]))

        ylimu = ylimu1 if ylimu1 > ylimu2 else ylimu2
        yliml = yliml1 if yliml1 < yliml2 else yliml2
        plt.ylim(yliml,ylimu)

        plt.axhline(0,color='lightgray',zorder=0,ls='--')
        plt.xlabel('Time')
        plt.ylabel('O-C (days)')
        plt.tight_layout()
        if saveplot:
            plt.savefig(filename+'_OC.pdf',format='pdf',dpi=100)
        if showplot:
            plt.show()
        plt.close('all')

        if min_times_err is not None:
            return OC_all[:,0],OC_all[:,1],min_times_err
        else:
            return OC_all[:,0],OC_all[:,1],np.ones_like(OC_all)*np.nan
