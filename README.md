# Codes to process RRL/Cepheid data from Kepler-TESS-Gaia surveys

# Useful informations

If any...

# Code descriptions

Below I provide descriptions for each of the codes.

# 1. Calculating Gaia absolute magnitudes

This code intended to get all possible information from Gaia Input Catalog using others such as VSX.

### The code works as follows:
- query Gaia, 2MASS, VSX, SIMBAD catalogues for available measurements
- probabilistically estimates distance, extinctions, absolute magnitudes

## Usage:
```
python query_gaia.py <inputfile> (<options>)
```
Input file __must be__ in the following format:
```
GaiaID  RA  DEC  Name
```

## Available options
 - `--Stassun` use Gaia parallax offset -80   mas (Stassun et al. 2018)
 - `--Riess`   use Gaia parallax offset -46   mas (Riess et al. 2018)
 - `--BJ`      use Gaia parallax offset -29   mas (BJ et al. 2018)
 - `--Zinn`    use Gaia parallax offset -52.8 mas (Zinn et al. 2019)
 
### Notes
 
 - We checked all available MWDUST implemeted dust maps. SFD __not__ sensitive to distance!
 - Best dust map is Combined19, which gives you the E(B-V).
 - Absorption values are calculated using constants from Green et al. 2019.
 
### TODO
 - query all input targets at once whereever it is possible
 
 # 2. Get Fourier parameters

The purpose of this code is to safely determine the Fourier coefficients of any given dataset.

### The steps are as follows:
- finding the main frequency using Lomb-Scargle
- fitting sine or cosine curve to get Fourier parameters
- pre-whitenning with the fitted curve
- iteratively fitting a sine or cosine curve with frequency = *n* * *main frequency* to get Fourier parameters (n=[2,`nfreq`])
- estimating errors...
  - from covariance matrix of non-linear Levenberg-Marquardt fit
  - or using bootstrap (generating subsamples and refitting those ones; **optional**)

## Example usage:
Load (RR Lyr) dataset from Kepler data and save columns as new variables (*time*, *flux/mag* and *error* if available).
```
import lightkurve as lk

lc = lk.search_lightcurvefile('RR Lyr',quarter=1).download_all()
lc = lc.PDCSAP_FLUX.stitch().remove_outliers().remove_nans()

t = lc.time
y = lc.flux
yerr = lc.flux_err
```

Initialize fitter
```
fitter = FourierFitter()
```

Do the Fourier calculation and fitting w/ 3 iterative steps (i.e. determine 3 harmonic components). The result will be two lists containing the Fourier parameters and their errors, respectively.
```
nfreq = 3
pfit,err = fitter.fit_freqs( t,y, yerr,
                             nfreq = nfreq,
                             plotting = False,
                             minimum_frequency=None,
                             maximum_frequency=None,
                             nyquist_factor=1,
                             samples_per_peak=10,
                             bootstrap=False,ntry=100,sample_size=0.9, parallel=False,ncores=-1,
                             kind='sin' )
```

Calculate the Fourier parameters
```
freq,period,P21,P31,R21,R31 = fitter.get_fourier_parameters()
```

## Available options
 - `nfreq` number of frequencies to be determined; the main frequency and its harmonics will be calculated
 - `plotting` *True* or *False* – plot the Lomb-Scargle periodograms and the fits?
 - `minimum_frequency` *None* or *value* - if *None* the samellest value will be used based on the sampling rate
 - `maximum_frequency` *None* or *value* - if *None* `nyquist_factor`**Nyquist frequency* will be used
 - `nyquist_factor` if `maximum_frequency` is *None*, `nyquist_factor`**Nyquist frequency* will be used as `maximum_frequency`
 - `samples_per_peak` oversampling factor
 - `bootstrap` *True* or *False* – use boostrap method to estimate errorbars (better if S/N is small, but ~10 times slower)
 - `ntry` number of random samplings in boostrap
 - `sample_size` ratio to generate subsamples
 - `parallel`*True* or *False* - do bootstrap parallel
 - `ncores` number of cores to be used, if *-1* all available cores will used
 - `kind` *sin* or *cos* – core function to be fitted

