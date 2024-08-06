---
title: 'seismolab: A Python package for analyzing space-based observations of variable stars'
tags:
  - Python
  - astronomy
  - variable stars
authors:
  - name: Attila Bodi
    orcid: 0000-0002-8585-4544
    affiliation: "1, 2"
affiliations:
 - name: HUN-REN CSFK Konkoly Observatory, Konkoly Thege M. \`ut 15-17, Budapest, 1121, Hungary
   index: 1
 - name: Department of Astrophysical Sciences, Princeton University, 4 Ivy Lane, Princeton, NJ 08544, USA
   index: 2
date: 5 August 2024
bibliography: paper.bib
---

# Summary

Classical pulsating stars are characterized by periodic or multi-periodic brightness variations from several hundredths to a few tenths of relative magnitudes. The observation of these variable stars is essential for testing pulsation and stellar evolution models. The different forms of frequency spectra are powerful tool for comparing observations and theoretical models. Long-term period changes can reveal information about the motion of the variable star in a binary system [@Plachy21]. Proper classification of variable stars, which usually requires rigorous analysis, is crucial to studying the properties of clear samples [@Tarczay23].

Upon entering the era of photometric space missions, the launch of the NASA *Kepler* and *TESS* missions have brought a new opportunity to expand the science of variable star, enabling the characterization of short-term variability and the detection of millimagnitude-level variations. The latest generation of large photometric and astrometric surveys has greatly expanded the number of known, observed and classified variable stars. Mining these large data sets has led to new discoveries and more detailed analysis in this field [@Plachy21, @Molnar22, @Benko23].

Several techniques have been developed to search for periodicities of the light curves and also for any deviation from the strictly periodic behavior. We have developed a Python package, ``seismolab``, which implements various methods for downloading, analyzing, and visualizing data of variable stars from space-based surveys. The framework is primary intended to be used with data obtained by the *Kepler*, *TESS* and *Gaia* surveys, but can also be used by other similar existing and future surveys. Some modules are also useful for analyzing ground-based observations.

The purpose of ``seismolab`` is sixfold. It is able to combine *Gaia* data with the Bailer-Jones distance catalog [@BJ21], galactic extinction maps [@Bovy16] and magnitudes from the Simbad catalog [@Wenger00]. Different modules are implemented to extract the Fourier coefficients and Fourier parameters of light curves, and to derive the temporal variation of the amplitude, phase and zero point of the dominant variation [@Benko23]. One can extract the minimum or maximum times and derive an O-C diagram [@Sterken05]. The package can be use to fill gaps in time series data using the method of inpainting [@Pires09, @Pires15]. There is also a module that provides different transformation methods for time-frequency analysis [@Kollath97].

The documentation of `seismolab` consists of pages describing the various
available functions, as well as tutorial notebooks.

# References