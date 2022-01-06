# Codes to process RRL/Cepheid data from Kepler-TESS-Gaia surveys

# Useful informations

If any...

# Code descriptions

Below I provide descriptions for each of the codes.

# 1. Calculating Gaia absolute magnitudes

This code intended to get all possible information from Gaia Input Catalog, 2MASS, VSX and Simbad.

### The code works as follows:
- query Gaia archive for RA, DEC, parallax, magnitudes, *(only for DR2)* RRL/Cep periods
- query VSX star-by-star if Gaia period not known *(only for DR2)*
- query SIMBAD catalogue for V mag and 2MASS JHKs mags
- probabilistically estimate distances *(only for DR2)*
- download BJ dinstances *(only for EDR3)*
- get extinctions from MWDUST maps
- calculate absolute magnitudes in G,BP,RP,V,J,H,Ks bands

## Usage:
```
python query_gaia.py <inputfile> (<options>)
```
Input file __must be__ in one of the following formats:
```
GaiaID  RA  DEC  Name
GaiaID Name
GaiaID
```

## Available options
 - `--EDR3`    Query EDR3 catalog w/ new BJ distances
 - `--photo`   Use photogeometric BJ distance instead of geometric ones
 - `--Stassun` use Gaia parallax offset -80   μas for DR2 (Stassun et al. 2018)
 - `--Riess`   use Gaia parallax offset -46   μas for DR2 (Riess et al. 2018)
 - `--BJ`      use Gaia parallax offset -29   μas for DR2 (BJ et al. 2018)
 - `--Zinn`    use Gaia parallax offset -52.8 μas for DR2 (Zinn et al. 2019)

### Notes

 - We checked all available MWDUST implemeted dust maps. SFD is __not__ sensitive to distance!
 - Best dust map is Combined19, which gives you the E(B-V).
 - Absorption values are calculated using extinction vectors from Green et al. 2019 and [IsoClassify](https://github.com/danxhuber/isoclassify)

### TODO
 - ~~query all input targets at once whereever it is possible~~
 - handle if Gaia EDR3 X DR2 returns w/ more than one targets

# 2. Get Fourier parameters

The purpose of this code is to safely determine the Fourier coefficients of any given dataset.

### The steps are as follows:
- finding the main frequency using Lomb-Scargle
- fitting sine or cosine curve to get Fourier parameters
- pre-whitenning with the fitted curve
- iteratively fitting a sine or cosine curve with frequency = *n* * *main frequency* to get Fourier parameters (n=[2,`nfreq`])
- estimating errors...
  - from analytically using the formulae of Breger, 1999, A&A, 349, 225
  - using bootstrap (generating subsamples and refitting those ones; **optional**)
  - using monte carlo (generating new samples and refitting those ones; **optional**)

## Example usage:
Load (Cepheid) dataset from OGLE data and save columns as new variables (*time*, *flux/mag* and *error* if available).
```
lc = np.loadtxt("https://ogledb.astrouw.edu.pl/~ogle/OCVS/data/I/01/OGLE-LMC-CEP-0001.dat").T

# Store results in separate arrays for clarity
t = lc[0]
y = lc[1]
yerr = lc[2]
```

Do the Fourier calculation and fitting w/ maximum 10 iterative steps (i.e. determine max 10 harmonic components). The result will be two lists containing the Fourier parameters and their errors, respectively.
```
# Initialize fitter by passing your light curve
fitter = FourierFitter(t,y)

# The same can be done with measurement errors
#fitter = FourierFitter(t,y,yerr)

# Do the Fourier calculation
nfreq = 10  # Set to e.g. 9999 to fit all harmonics

pfit,perr = fitter.fit_freqs( nfreq = nfreq,
                             plotting = False,
                             minimum_frequency=None,
                             maximum_frequency=20,        # Overwrites nyquist_factor!
                             nyquist_factor=1,
                             samples_per_peak=100,        # Oversampling factor in Lomb-Scargle spectrum calculation
                             error_estimation='analytic', # Method of the error estimation
                             ntry=1000,                   # Number of samplings if method is NOT analytic
                             sample_size=0.999,           # Subsample size if method is bootstrap
                             parallel=True,ncores=-1,
                             kind='sin' )
```

Calculate the Fourier parameters
```
freq,period,P21,P31,R21,R31 = fitter.get_fourier_parameters()

print('freq = ',  freq.n,   freq.s)
print('period = ',period.n, period.s)
print('R21 = ',   R21.n,    R21.s)
print('R31 = ',   R31.n,    R31.s)
print('P21 = ',   P21.n,    P21.s)
print('P31 = ',   P31.n,    P31.s)
```

## Available options
 - `nfreq` number of frequencies to be determined; the main frequency and its harmonics will be calculated
 - `absolute_sigma` If `True`, error is used in an absolute sense and the estimated parameter covariance reflects these absolute values.
 - `nfreq` The number of harmonics to be fitted. Pass a very large number to fit all harmonics.
 - `plotting` If `True`, fitting steps will be displayed.
 - `minimum_frequency` If specified, then use this minimum frequency rather than one chosen based on the size
     of the baseline.
 - `maximum_frequency` If specified, then use this maximum frequency rather than one chosen based on the average
     nyquist frequency.
 - `nyquist_factor` The multiple of the average nyquist frequency used to choose the maximum frequency
     if ``maximum_frequency`` is not provided.
 - `samples_per_peak` The approximate number of desired samples across the typical frequency peak.
 - `kind` Harmonic function to be fitted.
 - `error_estimation` If `bootstrap` or `montecarlo` is choosen, boostrap or monte carlo method will be used to estimate parameter uncertainties.
     Otherwise given uncertainties are calculated analytically.
 - `ntry` Number of resamplings for error estimation.
 - `sample_size` The ratio of data points to be used for bootstrap error estimation in each step.
     Applies only if `error_estimation` is set to `bootstrap`.
 - `parallel` If `True`, sampling for error estimation is performed parallel to speed up the process.
 - `ncores` Number of CPU cores to be used for parallel error estimation. If `-1`, then all available cores will be used.

# 3. OC fitter

To calculate the O-C diagram of a variable star, each minima can be fitted with a given function and associated OC errors are estimated in different ways.
- `MCMC version:` written in python2. Only given order polynomials are fitted, and errors are from MCMC realizations. _This is an old and slow method._
- `Bruteforce:` Three kind of functions are available to fit each minimum. Polynomial, non-parametric and model (obtained from fitting the median of phase curve). Errors are from resampling the light curves using their brightness measurement errors. __This the newer and suggested solution!__
