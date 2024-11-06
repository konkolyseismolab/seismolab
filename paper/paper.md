---
title: 'seismolab: A Python package for analyzing space-based observations of variable stars'
tags:
  - Python
  - astronomy
  - variable stars
authors:
  - name: Attila Bódi
    orcid: 0000-0002-8585-4544
    affiliation: "1, 2"
affiliations:
 - name: HUN-REN CSFK Konkoly Observatory, Konkoly Thege M. út 15-17, Budapest, 1121, Hungary
   index: 1
 - name: Department of Astrophysical Sciences, Princeton University, 4 Ivy Lane, Princeton, NJ 08544, USA
   index: 2
date: 5 August 2024
bibliography: paper.bib
---

# Summary

Classical pulsating stars are characterized by periodic or multi-periodic brightness variations from several hundredths to a few tenths of relative magnitudes. The observation of these variable stars is essential for testing pulsation and stellar evolution models. The different forms of frequency spectra are powerful tools for comparing observations and theoretical models. Long-term period changes can reveal information about the motion of the variable star in a binary system [@Plachy21]. Proper classification of variable stars, which usually requires rigorous analysis, is crucial to studying the properties of clear samples [@Tarczay23].

Upon entering the era of photometric space missions, the launch of the NASA *Kepler* and *TESS* missions have brought a new opportunity to expand the science of variable stars, enabling the characterization of short-term variability and the detection of millimagnitude-level variations. The latest generation of large photometric and astrometric surveys has greatly expanded the number of known, observed and classified variable stars. Mining these large data sets has led to new discoveries and more detailed analysis in this field [@Plachy21; @Molnar22; @Benko23].

Several techniques have been developed to search for periodicity in light curves and also for any deviation from strictly periodic behavior. We have developed a Python package, ``seismolab``, which implements various methods for downloading, analyzing, and visualizing data of variable stars from space-based surveys. The framework is primary intended to be used with data obtained by the *Kepler*, *TESS* and *Gaia* surveys, but can also be used by other similar existing and future surveys. Some modules are also useful for analyzing ground-based observations.

# Statement of need

``seismolab`` is a fully Python-based, open-source package, built on top of popular Python packages such as NumPy [@numpy], SciPy [@scipy], PyMC [@pymc] and Astropy [@astropy1; @astropy2; @astropy3]. The framework contains the main analysis tools for the astronomical community working on the light curves of variable stars, primarily, but not exclusively, from space-based surveys. ``seismolab`` is a modular library that allows users to select the best methods for their particular science problem. It has six main modules which implement common operations often used by the variable star community, but available in limited or no form in other popular codes.

The *Gaia* module facilitates the derivation of basic stellar parameters (such as distance and corrected-brightnesses) by combining Bailer-Jones distance catalog [@BJ21], galactic extinction maps [@Bovy16] and magnitudes from the Simbad catalog [@Wenger00] using different methods. The *fourier* module extends the standard Fourier analysis with easily available visualization tools and instant estimation of Fourier parameters. The *template* and *OC* modules provide a flexible and automated version of commonly used methods for extracting the temporal variation of the amplitude, phase and zero-point of the dominant variation [@Benko23; @Sterken05]. The *inpainting* module helps to eliminate the artifacts seen in the Fourier- and time-frequency analysis caused by gaps and uneven sampling using the method of inpainting [@Pires09; @Pires15]. The *tfa* module implements various techniques not available in other popular python packages to characterize light curves in the two-dimensional time-frequency plane [@Kollath97].

The documentation of `seismolab` consists of pages describing the various
available functions, as well as tutorial notebooks.

# Acknowledgements
This project has been supported by the KKP-137523 'SeismoLab' Élvonal grant of the Hungarian Research, Development and Innovation Office (NKFIH).

# References
