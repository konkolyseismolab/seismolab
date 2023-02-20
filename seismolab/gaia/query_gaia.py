# -*- coding: utf-8 -*-

try:
    import mwdust
except ModuleNotFoundError:
    msg = 'No module named \'mwdust\'\nFollow the installation details here: https://github.com/jobovy/mwdust'
    raise ModuleNotFoundError(msg)

import numpy as np
from astropy.io import ascii
from astropy.table import join,Table,unique
import requests
from time import sleep
import regex

from .querytools import perform_query,get_dist_absmag,get_dist_absmag_edr3

import joblib
from joblib import delayed
from tqdm.auto import tqdm
from multiprocessing import cpu_count

import argparse
from argparse import RawTextHelpFormatter

__all__ = ['query_gaia','query_from_commandline']

class ProgressParallel(joblib.Parallel):
    def __init__(self, total=None, **kwds):
        self.total = total
        super().__init__(**kwds)
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self._pbar.total = self.n_dispatched_tasks
        else:
            self._pbar.total = self.total
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Update returned Simbad fields
from astroquery.simbad import Simbad
Simbad.add_votable_fields('ids',
                          'flux(B)','flux_error(B)',
                          'flux(V)','flux_error(V)',
                          'flux(J)','flux_error(J)',
                          'flux(H)','flux_error(H)',
                          'flux(K)','flux_error(K)')

def _query_simbad(targs):
    simbadcols = ['IDS','FLUX_B','FLUX_ERROR_B','FLUX_V','FLUX_ERROR_V',\
                  'FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K']

    with warnings.catch_warnings(record=True):
        simbadqueryresult = Simbad.query_objects(targs)

    if simbadqueryresult is None:
        warnings.warn(
            "Simbad query resulted in 0 objects!")

        simbadqueryresult = Table(names=simbadcols,dtype=['O']*len(simbadcols),masked=True)
        simbadqueryresult.add_row( {'IDS':'Gaia DR3 0000000000000000000'} )
    else:
        simbadqueryresult = simbadqueryresult[simbadcols]

    return simbadqueryresult

def query_gaia(targets,gaiaDR=3,use_photodist=False,dustmodel='Combined19',plx_offset=None):
    '''
    ``query_gaia`` performs Gaia database query and
    calculates distance, reddening corrected apparent
    and absolute magnitudes

    Parameters
    ----------
    targets : int, list, Table, DataFrame
        Input Gaia IDs in the given catalog which is currently being used.
    gaiaDR : int, default: 3
        Gaia DataRelease number (2 or 3).
    use_photodist : bool, default: False
        If `True` photogeometric distances are used.
        Otherwise geometric distances are used.
    dustmodel : str, default: 'Combined19'
        The `mwdust` model to be used for redding correction.
        See `mwdust` documentation for available maps:
        https://github.com/jobovy/mwdust
    plx_offset : str or float, default: 0.0
        If float, the parallax offset (in mas) to be added to the parallaxes.
        It also can can be one of following names:
            "Stassun", which is +0.08   mas (Stassun et al. 2018)
            "Riess",   which is +0.046  mas (Riess et al. 2018)
            "BJ",      which is +0.029  mas (Bailer-Jones et al. 2018)
            "Zinn",    which is +0.0528 mas (Zinn et al 2019)

    Returns
    -------
    time : Astropy Table
        Calculated distance, brightness and absorption values.
    '''

    # --- Convert target list to Astropy Table ---
    targets = np.atleast_1d(targets).ravel().astype(int)
    targets = Table( {'Source':targets} )

    # --- Filter input data ---
    targets = unique(targets, keys='Source')

    # ------ Select Gaia catalog -----------
    if int(gaiaDR) == 3:
        useEDR3 = True
    elif int(gaiaDR) == 2:
        useEDR3 = False
    else:
        raise ValueError("Please set gaiaDR to 2 or 3!")

    # ------ For plx correction in DR2 -----------
    if isinstance(plx_offset,str) and not useEDR3:
        if plx_offset == "Stassun": plx_offset = +0.08   #mas #Stassun et al. 2018
        elif plx_offset == "Riess": plx_offset = +0.046  #mas #Riess et al. 2018
        elif plx_offset == "BJ":    plx_offset = +0.029  #mas #BJ et al. 2018
        elif plx_offset == "Zinn":  plx_offset = +0.0528 #mas #Zinn et al 2019
        else:
            raise ValueError("plx_offset %s not recognized\n" % plx_offset + \
            "Please use Stassun, Riess, BJ or Zinn, or do not set plx_offset!")
    elif not useEDR3 and plx_offset is not None:
        plx_offset = float(plx_offset) # from cmdline
    else:
        plx_offset = 0.0

    # ------ Set MW dust model -----------
    print("Using %s map from mwdust" % str(dustmodel))
    dustmodel = getattr(mwdust, dustmodel)()

    # --- Query Gaia -----------
    try:
        gaiaquery, BJdist = perform_query(targets,useEDR3)
    except requests.exceptions.ConnectionError:
        sleep(1)
        gaiaquery, BJdist = perform_query(targets,useEDR3)

    # --- Calculate magnitude errors from fluxes ---
    gaiaquery['phot_g_mean_mag_error'] = gaiaquery['phot_g_mean_flux_error']*2.5*1/(gaiaquery['phot_g_mean_flux']*np.log(10))
    gaiaquery['phot_bp_mean_mag_error'] = gaiaquery['phot_bp_mean_flux_error']*2.5*1/(gaiaquery['phot_bp_mean_flux']*np.log(10))
    gaiaquery['phot_rp_mean_mag_error'] = gaiaquery['phot_rp_mean_flux_error']*2.5*1/(gaiaquery['phot_rp_mean_flux']*np.log(10))
    if not useEDR3:
        gaiaquery = gaiaquery['source_id','ra','dec',
                             'parallax','parallax_error',
                             'phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag',
                             'phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error']
    else:
        # Merge catalogues
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gaiaquery = join(gaiaquery,BJdist,join_type='left',keys='source_id')
            gaiaquery = gaiaquery['source_id','ra','dec',
                                 'parallax','parallax_error',
                                 'phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag',
                                 'phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error',
                                 'r_med_geo','r_lo_geo','r_hi_geo',
                                 'r_med_photogeo','r_lo_photogeo','r_hi_photogeo'
                                 ]

    gaiaquery.rename_columns(['source_id'],['Source'])

    # ------ SIMBAD catalog for V,J,H,K mags -----------
    print('Downloading Simbad data for all stars...')

    # Prepare target names
    sources = []
    if useEDR3:
        for source in targets['Source']:
            sources.append('Gaia DR3 '+source.astype('str')  )
    else:
        for source in targets['Source']:
            sources.append('Gaia DR2 '+source.astype('str')  )

    # Query all objects from Simbad at once
    try:
        simbadquery = _query_simbad(sources)
    except requests.exceptions.ConnectionError:
        sleep(1)
        simbadquery = _query_simbad(sources)

    # --- Get proper Gaia catalog IDs ---
    gaiaids = simbadquery['IDS'].data.data

    if useEDR3:
        for i in range(len(gaiaids)):
            currentid = regex.findall("Gaia DR3 [0-9]+",gaiaids[i])[0]
            simbadquery[i]['IDS'] = currentid.split(' ')[-1]
    else:
        for i in range(len(gaiaids)):
            currentid = regex.findall("Gaia DR2 [0-9]+",gaiaids[i])[0]
            simbadquery[i]['IDS'] = currentid.split(' ')[-1]

    simbadquery.rename_columns(['IDS'],['Source'])

    # ------ Merge catalogs -----------
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        targets = join(targets,gaiaquery,join_type='left',keys='Source')
        targets['Source']     = targets['Source'].astype('str')
        simbadquery['Source'] = simbadquery['Source'].astype('str')
        targets = join(targets,simbadquery,join_type='left',keys='Source')

    targets.rename_columns(['parallax','parallax_error',\
                        'phot_g_mean_mag','phot_g_mean_mag_error',\
                        'phot_bp_mean_mag','phot_bp_mean_mag_error',\
                        'phot_rp_mean_mag','phot_rp_mean_mag_error'],
                        ['plx','sig_plx','gamag','sig_gamag',\
                        'bpmag','sig_bpmag','rpmag','sig_rpmag'])

    targets.rename_columns(['FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K'],
                        ['jmag','sig_jmag','hmag','sig_hmag','kmag','sig_kmag'])

    targets.rename_columns(['FLUX_B','FLUX_ERROR_B','FLUX_V','FLUX_ERROR_V'],
                        ['bmag','sig_bmag','vmag','sig_vmag'])
    targets = targets.filled(-99)

    # ------ Start calculations -----------
    print('Calculating distances, absolute magnitudes...')
    outdata = []

    max_ = len(targets)

    if not useEDR3:
        ncores = cpu_count()
        if max_ >= 10*ncores:
            outdata = ProgressParallel(n_jobs=ncores,total=max_)(delayed(get_dist_absmag)(i,targets,dustmodel,plx_offset) for i in np.arange(max_))
        else:
            for i in tqdm(np.arange(max_)):
                result = get_dist_absmag(i,targets,dustmodel,plx_offset)
                outdata.append(result)
    else:
        for i in tqdm(np.arange(max_)):
            result = get_dist_absmag_edr3(i,targets,dustmodel,plx_offset,use_photodist)
            outdata.append(result)

    outdataTable = Table(names=['Source',
                          'dist','distep','distem',
                          'lon_deg', 'lat_deg',
                          'aG','aGep','aGem','MG','MGep','MGem','mG','mGep','mGem',
                          'aBP','aBPep','aBPem','MBP','MBPep','MBPem','mBP','mBPep','mBPem',
                          'aRP','aRPep','aRPem','MRP','MRPep','MRPem','mRP','mRPep','mRPem',
                          'aB','aBep','aBem','MB','MBep','MBem','mB','mBep','mBem',
                          'aV','aVep','aVem','MV','MVep','MVem','mV','mVep','mVem',
                          'aJ','aJep','aJem','MJ','MJep','MJem','mJ','mJep','mJem',
                          'aH','aHep','aHem','MH','MHep','MHem','mH','mHep','mHem',
                          'aK','aKep','aKem','MK','MKep','MKem','mK','mKep','mKem'],
                          dtype=('i8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    for row in outdata:
        outdataTable.add_row(row)

    outdataTable['Source'] = outdataTable['Source'].astype('str')
    if useEDR3:
        targets.remove_columns(['r_med_geo', 'r_lo_geo', 'r_hi_geo'])
        targets.remove_columns(['r_med_photogeo', 'r_lo_photogeo', 'r_hi_photogeo'])
    targets = join(targets,outdataTable,join_type='left',keys='Source')

    targets.rename_columns(['plx','sig_plx','gamag','sig_gamag',\
                            'bpmag','sig_bpmag','rpmag','sig_rpmag',\
                            'jmag','sig_jmag','hmag','sig_hmag','kmag','sig_kmag',\
                            'bmag','sig_bmag','vmag','sig_vmag'],
                            ['parallax','sig_parallax','Gmag','sig_Gmag',\
                            'BPmag','sig_BPmag','RPmag','sig_RPmag',\
                            'Jmag','sig_Jmag','Hmag','sig_Hmag','Kmag','sig_Kmag',\
                            'Bmag','sig_Bmag','Vmag','sig_Vmag'])

    return targets

def query_from_commandline():
    # --- Create the parser ---
    my_parser = argparse.ArgumentParser(description='Query Gaia catalog w/ extinction correction',
                                        formatter_class=RawTextHelpFormatter)

    # --- Add the arguments ---
    my_parser.add_argument('Path',
                           metavar='<inputfile>',
                           type=str,
                           help="path to the inputfile\n"
                                "The first column is the Gaia ID\n"
                                "in the given catalog which is being used.")

    my_parser.add_argument('--gaiaDR','-G',
                           type=int,
                           default=3,
                           choices=[2,3],
                           help='Gaia DataRelease number.')

    my_parser.add_argument('--photodist',
                           action='store_true',
                           help='Use photogeometric distance instead of geometric ones.')

    my_parser.add_argument('--dustmodel',
                           type=str,
                           default='Combined19',
                           help='The mwdust model to be used for reddening corrections.')

    my_parser.add_argument('--Stassun', action='store_true', help='use plx zeropoint -80   uas for DR2 (Stassun et al. 2018)')
    my_parser.add_argument('--Riess', action='store_true', help='use plx zeropoint -46   uas for DR2 (Riess et al. 2018)')
    my_parser.add_argument('--BJ', action='store_true', help='use plx zeropoint -29   uas for DR2 (BJ et al. 2018)')
    my_parser.add_argument('--Zinn', action='store_true', help='use plx zeropoint -52.8 uas for DR2 (Zinn et al. 2019)')

    my_parser.add_argument('--plxoffset',
                           type=float,
                           help='The parallax offset (in mas) to be added to the parallaxes.')

    # --- Execute parse_args() ---
    args = my_parser.parse_args()
    gaiaDR = args.gaiaDR
    use_photodist = args.photodist
    dustmodel = args.dustmodel

    # ------ Define filenames -----------
    infilename = args.Path
    lenOfExtension = len(infilename.split('.')[-1])
    infilenameShort = infilename[:-(lenOfExtension+1)] # remove extension
    if gaiaDR == 2:
        if args.Stassun: outfilename = infilenameShort+'_Mgaia_DR2_Stassun.txt'
        elif args.Riess: outfilename = infilenameShort+'_Mgaia_DR2_Riess.txt'
        elif args.BJ:    outfilename = infilenameShort+'_Mgaia_DR2_BJ.txt'
        elif args.Zinn:  outfilename = infilenameShort+'_Mgaia_DR2_Zinn.txt'
        else:            outfilename = infilenameShort+'_Mgaia_DR2.txt'
    elif gaiaDR == 3 and not use_photodist:
        outfilename = infilenameShort+'_Mgaia_DR3.txt'
    elif use_photodist:
        outfilename = infilenameShort+'_Mgaia_DR3_photo.txt'

    # ------ Load input file -----------
    targets = np.genfromtxt(infilename,usecols=0,dtype=int)
    targets = targets[ targets > -1 ]

    # ------ For plx correction -----------
    if args.Stassun: plx_offset = "Stassun"
    elif args.Riess: plx_offset = "Riess"
    elif args.BJ:    plx_offset = "BJ"
    elif args.Zinn:  plx_offset = "Zinn"
    elif isinstance(args.plxoffset,float):
        plx_offset = args.plxoffset
    else:            plx_offset = None

    # --- Run Gaia query ---
    result = query_gaia(targets,gaiaDR=gaiaDR,use_photodist=use_photodist,dustmodel=dustmodel,plx_offset=plx_offset)

    # --- Save results ---
    ascii.write(result, outfilename, format='basic', fast_writer=False, overwrite=True)
