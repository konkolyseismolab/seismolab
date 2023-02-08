Fourier module
==============

.. _installation:

Installation
------------

To use seismolab, first install it using pip:

.. code-block:: console

   $ pip install seismolab


Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``seismolab.fourier.MultiHarmonicFitter()`` function:

.. autoclass:: seismolab.fourier.MultiHarmonicFitter

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`seismolab.fourier.MultiHarmonicFitter`
will raise an exception.