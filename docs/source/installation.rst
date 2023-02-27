Installation
===============

To install seismolab with pip:

.. code-block:: console

   $ pip install seismolab


Building from source
--------------------

You can also install the current development version of seismolab. Just clone the GitHub repository and install the code with pip:

.. code-block:: console

  $ git clone https://github.com/konkolyseismolab/seismolab.git
  $ cd seismolab
  $ pip install .

For unit tests to pass you will also need ``mwdust``, which can be installed from source (see: https://github.com/jobovy/mwdust). Because we only need the `Marshall06` map for now, we can download it beforehand. First, we have to define an environment variable ``DUST_DIR``, e.g. in bash:

.. code-block:: console

  $ echo export DUST_DIR=~/DUST_DIR >> ~/.bashrc
  $ source ~/.bashrc

Then, we can create the appropriate directory and download the dust map:

.. code-block:: console

  $ mkdir ~/DUST_DIR
  $ cd ~/DUST_DIR
  $ mkdir marshall06
  $ cd marshall06
  $ curl ftp://cdsarc.u-strasbg.fr/pub/cats/J/A%2BA/453/635/table1.dat.gz --output table1.dat.gz
  $ curl ftp://cdsarc.u-strasbg.fr/pub/cats/J/A%2BA/453/635/ReadMe --output ReadMe
  $ gzip -d table1.dat.gz

Afterwards, ``mwdust`` can be installed without downloading a set of dust maps:

.. code-block:: console

  $ git clone https://github.com/jobovy/mwdust.git
  $ cd mwdust
  $ python setup.py install --user --no-downloads

To run the unit tests you may need to install ``pytest``:

.. code-block:: console

  $ pip install pytest

Check if the unit tests pass:

.. code-block:: console

  $ cd seismolab/tests
  $ python3 -m pytest
