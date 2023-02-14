from scipy.interpolate import interp1d
import numpy as np

def regression(X, flux, flux_err, prior_sigma = None, prior_mu = None):
    # Model normalization
    is_const = np.nanstd(X, axis=0) == 0
    X[:,~is_const] = X[:,~is_const] - np.median(X[:,~is_const],axis=0)
    X[:,~is_const] /= np.std(X[:,~is_const],axis=0)
    
    # Compute `X^T cov^-1 X + 1/prior_sigma^2`
    sigma_w_inv = X.T.dot(X / flux_err[:, None] ** 2)
    # Compute `X^T cov^-1 y + prior_mu/prior_sigma^2`
    B = np.dot(X.T, flux / flux_err ** 2)

    if prior_sigma is not None:
        sigma_w_inv = sigma_w_inv + np.diag(1.0 / prior_sigma ** 2)
    if prior_sigma is not None:
        B = B + (prior_mu / prior_sigma ** 2)

    w = np.linalg.solve(sigma_w_inv, B).T

    model_flux = X.dot(w)
    
    return w, model_flux

def shift_phase_curves_vertically(time,flux,fluxerr,period):
    # Create output array
    shifted_flux = flux.copy()
    
    # Convert time to phase
    phase = time%period/period
    
    # Get where phase jumps
    jumpat = np.where(np.diff(phase)<0)[0] + 1
    jumpat = np.r_[0,jumpat,len(phase)+1]
    
    # Get most covered phase interval
    largest_phase = np.argmax(np.diff(jumpat))
    largest_phase_start = jumpat[largest_phase]
    largest_phase_end = jumpat[largest_phase+1]
    
    # Drop baseline curve, i.e. do not fit baseline to baseline
    jumpat = np.delete( jumpat , jumpat == largest_phase_start )
    jumpat = np.delete( jumpat , jumpat == largest_phase_end )
    
    # Use most covered phase interval as baseline
    model_time = phase[largest_phase_start:largest_phase_end]
    model_flux = flux[largest_phase_start:largest_phase_end]
    
    # Get interpolated baseline flux
    model_interp = interp1d(model_time,model_flux,kind='cubic',fill_value='extrapolate')
    
    # Loop over each phase and shift them to the baseline phase
    for jumpstart,jumpend in zip(jumpat[:-1],jumpat[1:]):
        
        time2shift    = phase[jumpstart:jumpend]
        flux2shift    = flux[jumpstart:jumpend]
        fluxerr2shift = fluxerr[jumpstart:jumpend]

        # Get baseline flux at current phase points
        model = model_interp(time2shift)

        # Build model for regression
        X = np.c_[ np.ones_like(model), model ]

        # Fit current phase curve to interpolated baseline curve
        w, fitted_model = regression(X, flux2shift, fluxerr2shift)

        # Get current shifted phase curve
        shifted_flux[jumpstart:jumpend] = flux2shift + np.median(model) - w[0]
        
    return shifted_flux