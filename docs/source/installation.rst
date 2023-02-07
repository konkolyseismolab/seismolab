Installation
===============

To install seismolab with pip:

.. code-block:: console

   $ pip install seismolab


Alternatively you can install the current development version of seismolab:

.. code-block:: console

  $ git clone https://github.com/konkolyseismolab/seismolab.git
  $ cd seismolab
  $ python setup.py install



Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``seismolab.fourier.MultiHarmonicFitter()`` function:

.. autoclass:: seismolab.fourier.MultiHarmonicFitter

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`seismolab.fourier.MultiHarmonicFitter`
will raise an exception.

.. note::

   This project is under active development.