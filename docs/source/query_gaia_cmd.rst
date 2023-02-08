Query Gaia catalog from command line
====================================

The Gaia module can be used from the command line without firing up a Python console or notebook. The description and available options are given in :ref:`Query Gaia catalog`. The usage can be checked by typing

.. code-block:: console

   $ query_gaia --help


.. code-block:: console

    usage: query_gaia [-h] [--gaiaDR {2,3}] [--photodist] [--dustmodel DUSTMODEL]
                      [--Stassun] [--Riess] [--BJ] [--Zinn]
                      [--plxoffset PLXOFFSET]
                      <inputfile>

    Query Gaia catalog w/ extinction correction

    positional arguments:
      <inputfile>           path to the inputfile
                            The first column is the Gaia ID
                            in the given catalog which is being used.

    optional arguments:
      -h, --help            show this help message and exit
      --gaiaDR {2,3}, -G {2,3}
                            Gaia DataRelease number.
      --photodist           Use photogeometric distance instead of geometric ones.
      --dustmodel DUSTMODEL
                            The mwdust model to be used for reddening corrections.
      --Stassun             use plx zeropoint -80   uas for DR2 (Stassun et al. 2018)
      --Riess               use plx zeropoint -46   uas for DR2 (Riess et al. 2018)
      --BJ                  use plx zeropoint -29   uas for DR2 (BJ et al. 2018)
      --Zinn                use plx zeropoint -52.8 uas for DR2 (Zinn et al. 2019)
      --plxoffset PLXOFFSET
                            The parallax offset (in mas) to be added to the parallaxes.

Example usage
-------------

This module requires a file, where the first column contains the Gaia DR3 (or DR2) IDs. If the file is named as *my_sample.txt*, then a query can be done e.g. with photodistances in Gaia DR3 using the dust map of Green19.

.. code-block:: console

   $ query_gaia my_sample.txt --photo --dustmodel "Green19"

The results will be stored in *my_sample_Mgaia_DR3_photo.txt*.

