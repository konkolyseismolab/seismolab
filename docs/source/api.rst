.. _api:

API
===

Query Gaia catalog
------------------

.. autofunction:: seismolab.gaia.query_gaia

Fourier
-------

Fourier spectrum
~~~~~~~~~~~~~~~~

.. autoclass:: seismolab.fourier.Fourier
    :members:

Main frequency and its harmonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: seismolab.fourier.MultiHarmonicFitter
    :members:

All frequencies
~~~~~~~~~~~~~~~

.. autoclass:: seismolab.fourier.MultiFrequencyFitter
    :members:

Template fitting
----------------

.. autoclass:: seismolab.template.TemplateFitter
    :members:

Light curve minima fitting
--------------------------

.. autoclass:: seismolab.OC.OCFitter
    :members:

Inpainting
----------

K-inpainting
~~~~~~~~~~~~~

.. autofunction:: seismolab.inpainting.kinpainting

Gap insertion
~~~~~~~~~~~~~

.. autofunction:: seismolab.inpainting.insert_gaps

Time-frequency analysis
-----------------------

Windowed Lomb-Scargle transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: seismolab.tfa.windowed_lomb_scargle

Gábor transform
~~~~~~~~~~~~~~~

.. autofunction:: seismolab.tfa.gabor

Morlet Wavelet transform
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: seismolab.tfa.wavelet

Choi and Williams transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: seismolab.tfa.choi_williams