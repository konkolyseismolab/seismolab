.. seismolab documentation master file, created by
   sphinx-quickstart on Tue Feb 7 15:43:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

seismolab: tools for analyzing space-based observations
=======================================================

**seismolab** is an open-source python framework for downloading, analyzing, and visualizing data of variable stars from Kepler-TESS-Gaia surveys.

It has four main modules:

.. hlist::
   :columns: 1

   * The *Gaia* module combines Gaia data with the Bailer-Jones distance catalog, galactic extinction maps and magnitudes from the Simbad catalog.
   * The *fourier* module calculates the Fourier coefficients and Fourier parameters of light curves.
   * The *OC* module fits a model to the extrema of a light curve to extract the minimum or maximum times and derives an O-C diagram.
   * The *template* module fits a set of Fourier harmonics to a light curve and derives the temporal variation of the amplitude/phase/zero point of the dominant variation.

To starts using seismolab, check out the :ref:`installation <installation>` first.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
========

.. toctree::

  about
  installation
  query_gaia
  query_gaia_cmd
  fourier
  usage2
  get_fourier_params

.. toctree::

  api

Bug reports
-----------

**seismolab** is an open source project under the MIT license. The source code is available on `GitHub <https://github.com/konkolyseismolab/seismolab>`_. In case of any questions or problems, please contact us via the `Git Issues <https://github.com/konkolyseismolab/seismolab/issues>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

