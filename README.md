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
python rrl_acep_Mv.py <inputfile> (<options>)
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
 
 ## TODO
 - query all input targets at once whereever it is possible
