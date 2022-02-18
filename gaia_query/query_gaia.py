# -*- coding: utf-8 -*-

#import ebf
import numpy as np
#import scipy.interpolate
#import pdb
#import asfgrid
from astropy.io import ascii
from astropy.table import join,Table,unique
#from classify_direct import *
from classify_direct_2_Mv import stparas,stparas_edr3,obsdata
from astroquery.vizier import Vizier
import warnings
import requests
from time import sleep
from astropy.coordinates import Angle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from astropy.io.ascii import InconsistentTableError
import argparse
from argparse import RawTextHelpFormatter
import sys
import mwdust

warnings.filterwarnings("ignore", category=RuntimeWarning)

from astroquery.simbad import Simbad
# Simbad.add_votable_fields('flux(J)','flux_bibcode(J)')
Simbad.add_votable_fields('id(Gaia)',
                          'flux(V)','flux_error(V)',
                          'flux(J)','flux_error(J)',
                          'flux(H)','flux_error(H)',
                          'flux(K)','flux_error(K)')

def get_dist_absmag(i):
    # Check if it makes sense to run the calculations
    if ((data['plx'][i] + plx_offset) <= 0.) or ((data['sig_plx'][i]/ (data['plx'][i] + plx_offset)) > 1. ):
        dis,disep,disem = np.nan,np.nan,np.nan
        lon_deg, lat_deg = np.nan,np.nan
        agaia,agaiaep,agaiaem = np.nan,np.nan,np.nan
        absgaia,absgaiaep,absgaiaem = np.nan,np.nan,np.nan
        aBP,aBPep,aBPem = np.nan,np.nan,np.nan
        absBP,absBPep,absBPem = np.nan,np.nan,np.nan
        aRP,aRPep,aRPem = np.nan,np.nan,np.nan
        absRP,absRPep,absRPem = np.nan,np.nan,np.nan
        aV,aVep,aVem = np.nan,np.nan,np.nan
        absV,absVep,absVem = np.nan,np.nan,np.nan
        aJ,aJep,aJem = np.nan,np.nan,np.nan
        absJ,absJep,absJem = np.nan,np.nan,np.nan
        aH,aHep,aHem = np.nan,np.nan,np.nan
        absH,absHep,absHem = np.nan,np.nan,np.nan
        aK,aKep,aKem = np.nan,np.nan,np.nan
        absK,absKep,absKem = np.nan,np.nan,np.nan

        outdata = [data['Source'][i],
                        dis,disep,disem,
                        lon_deg, lat_deg,
                        agaia,agaiaep,agaiaem,
                        absgaia,absgaiaep,absgaiaem,
                        aBP,aBPep,aBPem,
                        absBP,absBPep,absBPem,
                        aRP,aRPep,aRPem,
                        absRP,absRPep,absRPem,
                        aV,aVep,aVem,
                        absV,absVep,absVem,
                        aJ,aJep,aJem,
                        absJ,absJep,absJem,
                        aH,aHep,aHem,
                        absH,absHep,absHem,
                        aK,aKep,aKem,
                        absK,absKep,absKem]
        return outdata

    # ------ Gaia mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addgaia(data['gamag'][i],data['sig_gamag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    dis,disep,disem = paras.dis, paras.disep, paras.disem
    agaia,agaiaep,agaiaem = paras.avs, paras.avsep, paras.avsem
    absgaia,absgaiaep,absgaiaem = paras.absmag, paras.absmagep, paras.absmagem
    lon_deg, lat_deg = paras.lon_deg, paras.lat_deg

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addBP(data['bpmag'][i],data['sig_bpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aBP,aBPep,aBPem = paras.avs, paras.avsep, paras.avsem
    absBP,absBPep,absBPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addRP(data['rpmag'][i],data['sig_rpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aRP,aRPep,aRPem = paras.avs, paras.avsep, paras.avsem
    absRP,absRPep,absRPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ V mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([-99.,data['vmag'][i],-99.,-99.],[-99.,data['sig_vmag'][i],-99.,-99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aV,aVep,aVem = paras.avs, paras.avsep, paras.avsem
    absV,absVep,absVem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ J mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([data['jmag'][i], -99., -99. ],\
             [data['sig_jmag'][i], -99., -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aJ,aJep,aJem = paras.avs, paras.avsep, paras.avsem
    absJ,absJep,absJem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ H mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([ -99., data['hmag'][i], -99. ],\
             [ -99., data['sig_hmag'][i], -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aH,aHep,aHem = paras.avs, paras.avsep, paras.avsem
    absH,absHep,absHem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ K mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([-99., -99., data['kmag'][i] ],\
             [-99., -99., data['sig_kmag'][i]])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aK,aKep,aKem = paras.avs, paras.avsep, paras.avsem
    absK,absKep,absKem = paras.absmag, paras.absmagep, paras.absmagem

    outdata = [data['Source'][i],
                    dis,disep,disem,
                    lon_deg, lat_deg,
                    agaia,agaiaep,agaiaem,
                    absgaia,absgaiaep,absgaiaem,
                    aBP,aBPep,aBPem,
                    absBP,absBPep,absBPem,
                    aRP,aRPep,aRPem,
                    absRP,absRPep,absRPem,
                    aV,aVep,aVem,
                    absV,absVep,absVem,
                    aJ,aJep,aJem,
                    absJ,absJep,absJem,
                    aH,aHep,aHem,
                    absH,absHep,absHem,
                    aK,aKep,aKem,
                    absK,absKep,absKem]
    return outdata

def get_dist_absmag(i):
    # Check if it makes sense to run the calculations
    if ((data['plx'][i] + plx_offset) <= 0.) or ((data['sig_plx'][i]/ (data['plx'][i] + plx_offset)) > 1. ):
        dis,disep,disem = np.nan,np.nan,np.nan
        lon_deg, lat_deg = np.nan,np.nan
        agaia,agaiaep,agaiaem = np.nan,np.nan,np.nan
        absgaia,absgaiaep,absgaiaem = np.nan,np.nan,np.nan
        aBP,aBPep,aBPem = np.nan,np.nan,np.nan
        absBP,absBPep,absBPem = np.nan,np.nan,np.nan
        aRP,aRPep,aRPem = np.nan,np.nan,np.nan
        absRP,absRPep,absRPem = np.nan,np.nan,np.nan
        aV,aVep,aVem = np.nan,np.nan,np.nan
        absV,absVep,absVem = np.nan,np.nan,np.nan
        aJ,aJep,aJem = np.nan,np.nan,np.nan
        absJ,absJep,absJem = np.nan,np.nan,np.nan
        aH,aHep,aHem = np.nan,np.nan,np.nan
        absH,absHep,absHem = np.nan,np.nan,np.nan
        aK,aKep,aKem = np.nan,np.nan,np.nan
        absK,absKep,absKem = np.nan,np.nan,np.nan

        outdata = [data['Source'][i],
                        dis,disep,disem,
                        lon_deg, lat_deg,
                        agaia,agaiaep,agaiaem,
                        absgaia,absgaiaep,absgaiaem,
                        aBP,aBPep,aBPem,
                        absBP,absBPep,absBPem,
                        aRP,aRPep,aRPem,
                        absRP,absRPep,absRPem,
                        aV,aVep,aVem,
                        absV,absVep,absVem,
                        aJ,aJep,aJem,
                        absJ,absJep,absJem,
                        aH,aHep,aHem,
                        absH,absHep,absHem,
                        aK,aKep,aKem,
                        absK,absKep,absKem]
        return outdata

    # ------ Gaia mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addgaia(data['gamag'][i],data['sig_gamag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    dis,disep,disem = paras.dis, paras.disep, paras.disem
    agaia,agaiaep,agaiaem = paras.avs, paras.avsep, paras.avsem
    absgaia,absgaiaep,absgaiaem = paras.absmag, paras.absmagep, paras.absmagem
    lon_deg, lat_deg = paras.lon_deg, paras.lat_deg

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addBP(data['bpmag'][i],data['sig_bpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aBP,aBPep,aBPem = paras.avs, paras.avsep, paras.avsem
    absBP,absBPep,absBPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addRP(data['rpmag'][i],data['sig_rpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aRP,aRPep,aRPem = paras.avs, paras.avsep, paras.avsem
    absRP,absRPep,absRPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ V mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([-99.,data['vmag'][i],-99.,-99.],[-99.,data['sig_vmag'][i],-99.,-99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aV,aVep,aVem = paras.avs, paras.avsep, paras.avsem
    absV,absVep,absVem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ J mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([data['jmag'][i], -99., -99. ],\
             [data['sig_jmag'][i], -99., -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aJ,aJep,aJem = paras.avs, paras.avsep, paras.avsem
    absJ,absJep,absJem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ H mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([ -99., data['hmag'][i], -99. ],\
             [ -99., data['sig_hmag'][i], -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aH,aHep,aHem = paras.avs, paras.avsep, paras.avsem
    absH,absHep,absHem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ K mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([-99., -99., data['kmag'][i] ],\
             [-99., -99., data['sig_kmag'][i]])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel,useav=0.,plot=0)

    aK,aKep,aKem = paras.avs, paras.avsep, paras.avsem
    absK,absKep,absKem = paras.absmag, paras.absmagep, paras.absmagem

    outdata = [data['Source'][i],
                    dis,disep,disem,
                    lon_deg, lat_deg,
                    agaia,agaiaep,agaiaem,
                    absgaia,absgaiaep,absgaiaem,
                    aBP,aBPep,aBPem,
                    absBP,absBPep,absBPem,
                    aRP,aRPep,aRPem,
                    absRP,absRPep,absRPem,
                    aV,aVep,aVem,
                    absV,absVep,absVem,
                    aJ,aJep,aJem,
                    absJ,absJep,absJem,
                    aH,aHep,aHem,
                    absH,absHep,absHem,
                    aK,aKep,aKem,
                    absK,absKep,absKem]
    return outdata

def get_dist_absmag_edr3(i):
    global use_photodist

    # ------ Gaia mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addgaia(data['gamag'][i],data['sig_gamag'][i])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    dis,disep,disem = paras.dis, paras.disep, paras.disem
    agaia,agaiaep,agaiaem = paras.avs, paras.avsep, paras.avsem
    absgaia,absgaiaep,absgaiaem = paras.absmag, paras.absmagep, paras.absmagem
    lon_deg, lat_deg = paras.lon_deg, paras.lat_deg

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addBP(data['bpmag'][i],data['sig_bpmag'][i])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aBP,aBPep,aBPem = paras.avs, paras.avsep, paras.avsem
    absBP,absBPep,absBPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addRP(data['rpmag'][i],data['sig_rpmag'][i])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aRP,aRPep,aRPem = paras.avs, paras.avsep, paras.avsem
    absRP,absRPep,absRPem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ V mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([-99.,data['vmag'][i],-99.,-99.],[-99.,data['sig_vmag'][i],-99.,-99.])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aV,aVep,aVem = paras.avs, paras.avsep, paras.avsem
    absV,absVep,absVem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ J mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([data['jmag'][i], -99., -99. ],\
             [data['sig_jmag'][i], -99., -99.])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aJ,aJep,aJem = paras.avs, paras.avsep, paras.avsem
    absJ,absJep,absJem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ H mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([ -99., data['hmag'][i], -99. ],\
             [ -99., data['sig_hmag'][i], -99.])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aH,aHep,aHem = paras.avs, paras.avsep, paras.avsem
    absH,absHep,absHem = paras.absmag, paras.absmagep, paras.absmagem

    # ------ K mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([-99., -99., data['kmag'][i] ],\
             [-99., -99., data['sig_kmag'][i]])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aK,aKep,aKem = paras.avs, paras.avsep, paras.avsem
    absK,absKep,absKem = paras.absmag, paras.absmagep, paras.absmagem

    outdata = [data['Source'][i],
                    dis,disep,disem,
                    lon_deg, lat_deg,
                    agaia,agaiaep,agaiaem,
                    absgaia,absgaiaep,absgaiaem,
                    aBP,aBPep,aBPem,
                    absBP,absBPep,absBPem,
                    aRP,aRPep,aRPem,
                    absRP,absRPep,absRPem,
                    aV,aVep,aVem,
                    absV,absVep,absVem,
                    aJ,aJep,aJem,
                    absJ,absJep,absJem,
                    aH,aHep,aHem,
                    absH,absHep,absHem,
                    aK,aKep,aKem,
                    absK,absKep,absKem]
    return outdata

if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Gaia Query w/ extinction correction',
                                        formatter_class=RawTextHelpFormatter)

    # Add the arguments
    my_parser.add_argument('Path',
                           metavar='<inputfile>',
                           type=str,
                           help="the path to inputfile\n"
                                "Inputfile must be in one of the following column formats:\n"
                                " GaiaID RA DEC Name\n"
                                " GaiaID Name\n"
                                " GaiaID\n")

    my_parser.add_argument('--EDR3',
                           action='store_true',
                           help='Query EDR3 catalog w/ new BJ distances')

    my_parser.add_argument('--photo',
                           action='store_true',
                           help='Use photogeometric distance instead of geometric ones')

    my_parser.add_argument('--Stassun', action='store_true', help='use plx zeropoint -80   uas for DR2 (Stassun et al. 2018)')
    my_parser.add_argument('--Riess', action='store_true', help='use plx zeropoint -46   uas for DR2 (Riess et al. 2018)')
    my_parser.add_argument('--BJ', action='store_true', help='use plx zeropoint -29   uas for DR2 (BJ et al. 2018)')
    my_parser.add_argument('--Zinn', action='store_true', help='use plx zeropoint -52.8 uas for DR2 (Zinn et al. 2019)')

    # Execute parse_args()
    args = my_parser.parse_args()
    useEDR3 = args.EDR3
    use_photodist = args.photo

    # ------ For plx correction -----------
    if not useEDR3:
        if args.Stassun: plx_offset = +0.08   #mas #Stassun et al. 2018
        elif args.Riess: plx_offset = +0.046  #mas #Riess et al. 2018
        elif args.BJ:    plx_offset = +0.029  #mas #BJ et al. 2018
        elif args.Zinn:  plx_offset = +0.0528 #mas #Zinn et al 2019
        else:            plx_offset = 0.0
    else:
        plx_offset = 0.0

    # ------ Define filenames -----------
    infilename = args.Path
    lenOfExtension = len(infilename.split('.')[-1])
    infilenameShort = infilename[:-(lenOfExtension+1)] # remove extension
    if not useEDR3:
        if args.Stassun: outfilename = infilenameShort+'_MgaiaJHK_Stassun.txt'
        elif args.Riess: outfilename = infilenameShort+'_MgaiaJHK_Riess.txt'
        elif args.BJ:    outfilename = infilenameShort+'_MgaiaJHK_BJ.txt'
        elif args.Zinn:  outfilename = infilenameShort+'_MgaiaJHK_Zinn.txt'
        else:            outfilename = infilenameShort+'_MgaiaJHK.txt'
    elif useEDR3 and not use_photodist:
        outfilename = infilenameShort+'_MgaiaJHK_EDR3.txt'
    elif use_photodist:
        outfilename = infilenameShort+'_MgaiaJHK_EDR3_photo.txt'

    # ------ Guess input file format and load it -----------
    try:
        data=ascii.read(infilename,names=['Source','ra','dec','Name'])
        data = data['Source','Name']
    except InconsistentTableError:
        try:
            data=ascii.read(infilename,names=['Source','Name'])
        except InconsistentTableError:
            try:
                data=ascii.read(infilename,names=['Source'])
            except InconsistentTableError:
                warnings.warn('\nInconsistent Input Data!\n \
                Inputfile must be in one of the following column formats:\n \
                GaiaID RA DEC Name\n \
                GaiaID Name\n \
                GaiaID')
                exit(-1)
    #data['Source'].format = '%d'
    data = unique(data, keys='Source')

    #dnumodel = asfgrid.Seism()
    #bcmodel = h5py.File('bcgrid.h5', 'r')
    #dustmodel = mwdust.Green19()
    dustmodel = mwdust.Combined19()

    # --- Query Gaia -----------
    from astroquery.gaia import Gaia
    print('Querying Gaia/BJ catalog...')
    try:
        if useEDR3:
            import pyvo as vo
            # Initialize BJ catalog
            service = vo.dal.TAPService("https://dc.zah.uni-heidelberg.de/__system__/tap/run/tap")
            # Query BJ catalog
            if len(data['Source']) >1:
                BJdist = service.search("SELECT TOP 1000000 BJ.source_id, \
                                    BJ.r_med_geo,BJ.r_lo_geo,BJ.r_hi_geo, \
                                    BJ.r_med_photogeo,BJ.r_lo_photogeo,BJ.r_hi_photogeo \
                                    FROM gedr3dist.litewithdist as BJ \
                                    WHERE BJ.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                BJdist = service.search("SELECT TOP 1000000 BJ.source_id, \
                                    BJ.r_med_geo,BJ.r_lo_geo,BJ.r_hi_geo, \
                                    BJ.r_med_photogeo,BJ.r_lo_photogeo,BJ.r_hi_photogeo \
                                    FROM gedr3dist.litewithdist as BJ \
                                    WHERE BJ.source_id = "+str(tuple(data['Source'])[0]))
            BJdist = BJdist.to_table()

            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT TOP 1000000 EDR3.source_id,EDR3.ra,EDR3.dec,"
                          "EDR3.parallax,EDR3.parallax_error,"
                          "EDR3.phot_g_mean_flux,EDR3.phot_g_mean_flux_error,EDR3.phot_g_mean_mag,"
                          "EDR3.phot_bp_mean_flux,EDR3.phot_bp_mean_flux_error,EDR3.phot_bp_mean_mag,"
                          "EDR3.phot_rp_mean_flux,EDR3.phot_rp_mean_flux_error,EDR3.phot_rp_mean_mag "
                          "FROM gaiaedr3.gaia_source AS EDR3 "
                          "WHERE EDR3.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT TOP 1000000 EDR3.source_id,EDR3.ra,EDR3.dec,"
                          "EDR3.parallax,EDR3.parallax_error,"
                          "EDR3.phot_g_mean_flux,EDR3.phot_g_mean_flux_error,EDR3.phot_g_mean_mag,"
                          "EDR3.phot_bp_mean_flux,EDR3.phot_bp_mean_flux_error,EDR3.phot_bp_mean_mag,"
                          "EDR3.phot_rp_mean_flux,EDR3.phot_rp_mean_flux_error,EDR3.phot_rp_mean_mag "
                          "FROM gaiaedr3.gaia_source AS EDR3 "
                          "WHERE EDR3.source_id = "+str(tuple(data['Source'])[0]))
            gaiaquery = job.get_results()

            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT * "
                          "FROM gaiaedr3.dr2_neighbourhood AS EDR3 "
                          "WHERE EDR3.dr3_source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT * "
                          "FROM gaiaedr3.dr2_neighbourhood AS EDR3 "
                          "WHERE EDR3.dr3_source_id = "+str(tuple(data['Source'])[0]))
            dr2_neighbourhood = job.get_results()
            # TODO: select best DR2 ID if there are >1
            dr2_neighbourhood = dr2_neighbourhood['dr2_source_id','dr3_source_id']
        else:
            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT TOP 1000000 DR2.source_id,DR2.ra,DR2.dec,"
                          "DR2.parallax,DR2.parallax_error,"
                          "DR2.phot_g_mean_flux,DR2.phot_g_mean_flux_error,DR2.phot_g_mean_mag,"
                          "DR2.phot_bp_mean_flux,DR2.phot_bp_mean_flux_error,DR2.phot_bp_mean_mag,"
                          "DR2.phot_rp_mean_flux,DR2.phot_rp_mean_flux_error,DR2.phot_rp_mean_mag,"
                          "RRL.pf,RRL.p1_o,"
                          "CEP.pf,CEP.p1_o,CEP.p2_o,CEP.p3_o "
                          "FROM gaiadr2.gaia_source AS DR2 "
                          "LEFT OUTER JOIN gaiadr2.vari_rrlyrae AS RRL "
                          "ON DR2.source_id=RRL.source_id "
                          "LEFT OUTER JOIN gaiadr2.vari_cepheid AS CEP "
                          "ON DR2.source_id=CEP.source_id "
                          "WHERE DR2.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT TOP 1000000 DR2.source_id,DR2.ra,DR2.dec,"
                          "DR2.parallax,DR2.parallax_error,"
                          "DR2.phot_g_mean_flux,DR2.phot_g_mean_flux_error,DR2.phot_g_mean_mag,"
                          "DR2.phot_bp_mean_flux,DR2.phot_bp_mean_flux_error,DR2.phot_bp_mean_mag,"
                          "DR2.phot_rp_mean_flux,DR2.phot_rp_mean_flux_error,DR2.phot_rp_mean_mag,"
                          "RRL.pf,RRL.p1_o,"
                          "CEP.pf,CEP.p1_o,CEP.p2_o,CEP.p3_o "
                          "FROM gaiadr2.gaia_source AS DR2 "
                          "LEFT OUTER JOIN gaiadr2.vari_rrlyrae AS RRL "
                          "ON DR2.source_id=RRL.source_id "
                          "LEFT OUTER JOIN gaiadr2.vari_cepheid AS CEP "
                          "ON DR2.source_id=CEP.source_id "
                          "WHERE DR2.source_id = "+str(tuple(data['Source'])[0]))
            gaiaquery = job.get_results()
    except requests.exceptions.ConnectionError:
        sleep(1)
        if useEDR3:
            import pyvo as vo
            # Initialize BJ catalog
            service = vo.dal.TAPService("https://dc.zah.uni-heidelberg.de/__system__/tap/run/tap")
            # Query BJ catalog
            if len(data['Source']) >1:
                BJdist = service.search("SELECT TOP 1000000 BJ.source_id, \
                                    BJ.r_med_geo,BJ.r_lo_geo,BJ.r_hi_geo, \
                                    BJ.r_med_photogeo,BJ.r_lo_photogeo,BJ.r_hi_photogeo \
                                    FROM gedr3dist.litewithdist as BJ \
                                    WHERE BJ.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                BJdist = service.search("SELECT TOP 1000000 BJ.source_id, \
                                    BJ.r_med_geo,BJ.r_lo_geo,BJ.r_hi_geo, \
                                    BJ.r_med_photogeo,BJ.r_lo_photogeo,BJ.r_hi_photogeo \
                                    FROM gedr3dist.litewithdist as BJ \
                                    WHERE BJ.source_id = "+str(tuple(data['Source'])[0]))
            BJdist = BJdist.to_table()

            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT TOP 1000000 EDR3.source_id,EDR3.ra,EDR3.dec,"
                          "EDR3.parallax,EDR3.parallax_error,"
                          "EDR3.phot_g_mean_flux,EDR3.phot_g_mean_flux_error,EDR3.phot_g_mean_mag,"
                          "EDR3.phot_bp_mean_flux,EDR3.phot_bp_mean_flux_error,EDR3.phot_bp_mean_mag,"
                          "EDR3.phot_rp_mean_flux,EDR3.phot_rp_mean_flux_error,EDR3.phot_rp_mean_mag "
                          "FROM gaiaedr3.gaia_source AS EDR3 "
                          "WHERE EDR3.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT TOP 1000000 EDR3.source_id,EDR3.ra,EDR3.dec,"
                          "EDR3.parallax,EDR3.parallax_error,"
                          "EDR3.phot_g_mean_flux,EDR3.phot_g_mean_flux_error,EDR3.phot_g_mean_mag,"
                          "EDR3.phot_bp_mean_flux,EDR3.phot_bp_mean_flux_error,EDR3.phot_bp_mean_mag,"
                          "EDR3.phot_rp_mean_flux,EDR3.phot_rp_mean_flux_error,EDR3.phot_rp_mean_mag "
                          "FROM gaiaedr3.gaia_source AS EDR3 "
                          "WHERE EDR3.source_id = "+str(tuple(data['Source'])[0]))
            gaiaquery = job.get_results()

            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT * "
                          "FROM gaiaedr3.dr2_neighbourhood AS EDR3 "
                          "WHERE EDR3.dr3_source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT * "
                          "FROM gaiaedr3.dr2_neighbourhood AS EDR3 "
                          "WHERE EDR3.dr3_source_id = "+str(tuple(data['Source'])[0]))
            dr2_neighbourhood = job.get_results()
            # TODO: select best DR2 ID if there are >1
            dr2_neighbourhood = dr2_neighbourhood['dr2_source_id','dr3_source_id']
        else:
            if len(data['Source']) >1:
                job = Gaia.launch_job("SELECT TOP 1000000 DR2.source_id,DR2.ra,DR2.dec,"
                          "DR2.parallax,DR2.parallax_error,"
                          "DR2.phot_g_mean_flux,DR2.phot_g_mean_flux_error,DR2.phot_g_mean_mag,"
                          "DR2.phot_bp_mean_flux,DR2.phot_bp_mean_flux_error,DR2.phot_bp_mean_mag,"
                          "DR2.phot_rp_mean_flux,DR2.phot_rp_mean_flux_error,DR2.phot_rp_mean_mag,"
                          "RRL.pf,RRL.p1_o,"
                          "CEP.pf,CEP.p1_o,CEP.p2_o,CEP.p3_o "
                          "FROM gaiadr2.gaia_source AS DR2 "
                          "LEFT OUTER JOIN gaiadr2.vari_rrlyrae AS RRL "
                          "ON DR2.source_id=RRL.source_id "
                          "LEFT OUTER JOIN gaiadr2.vari_cepheid AS CEP "
                          "ON DR2.source_id=CEP.source_id "
                          "WHERE DR2.source_id IN "+str(tuple(data['Source'])))
            elif len(data['Source']) ==1:
                job = Gaia.launch_job("SELECT TOP 1000000 DR2.source_id,DR2.ra,DR2.dec,"
                          "DR2.parallax,DR2.parallax_error,"
                          "DR2.phot_g_mean_flux,DR2.phot_g_mean_flux_error,DR2.phot_g_mean_mag,"
                          "DR2.phot_bp_mean_flux,DR2.phot_bp_mean_flux_error,DR2.phot_bp_mean_mag,"
                          "DR2.phot_rp_mean_flux,DR2.phot_rp_mean_flux_error,DR2.phot_rp_mean_mag,"
                          "RRL.pf,RRL.p1_o,"
                          "CEP.pf,CEP.p1_o,CEP.p2_o,CEP.p3_o "
                          "FROM gaiadr2.gaia_source AS DR2 "
                          "LEFT OUTER JOIN gaiadr2.vari_rrlyrae AS RRL "
                          "ON DR2.source_id=RRL.source_id "
                          "LEFT OUTER JOIN gaiadr2.vari_cepheid AS CEP "
                          "ON DR2.source_id=CEP.source_id "
                          "WHERE DR2.source_id = "+str(tuple(data['Source'])[0]))
            gaiaquery = job.get_results()

    # Calculate magnitude errors from fluxes
    gaiaquery['phot_g_mean_mag_error'] = gaiaquery['phot_g_mean_flux_error']*2.5*1/(gaiaquery['phot_g_mean_flux']*np.log(10))
    gaiaquery['phot_bp_mean_mag_error'] = gaiaquery['phot_bp_mean_flux_error']*2.5*1/(gaiaquery['phot_bp_mean_flux']*np.log(10))
    gaiaquery['phot_rp_mean_mag_error'] = gaiaquery['phot_rp_mean_flux_error']*2.5*1/(gaiaquery['phot_rp_mean_flux']*np.log(10))
    if not useEDR3:
        # Select periods from Gaia RRL/Cep catalogues
        select_period = np.array([gaiaquery['pf'],gaiaquery['p1_o'],gaiaquery['pf_2'],gaiaquery['p1_o_2'],gaiaquery['p2_o'],gaiaquery['p3_o']]).T
        select_period[ np.isnan(select_period) ] = -99
        select_period = np.max(  select_period ,axis=1)
        select_period = np.where( select_period==-99., np.nan, select_period)
        # Update Gaia table
        gaiaquery['Period'] = select_period
        gaiaquery = gaiaquery['source_id','ra','dec',
                             'parallax','parallax_error',
                             'phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag',
                             'phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error',
                             'Period']
    else:
        # Merge catalogues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gaiaquery = join(gaiaquery,BJdist,join_type='left',keys='source_id')
            gaiaquery = gaiaquery['source_id','ra','dec',
                                 'parallax','parallax_error',
                                 'phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag',
                                 'phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error',
                                 'r_med_geo','r_lo_geo','r_hi_geo',
                                 'r_med_photogeo','r_lo_photogeo','r_hi_photogeo'
                                 ]
            dr2_neighbourhood.rename_columns(['dr3_source_id'],['source_id'])
            gaiaquery = join(gaiaquery,dr2_neighbourhood,join_type='left',keys='source_id')

    # --- Query VSX catalogue for missing Gaia periods  --------
    if not useEDR3:
        max_ = len(gaiaquery[ np.isnan(gaiaquery['Period']) ] )
        print('Query VSX catalog...')
        for source in tqdm( gaiaquery[ np.isnan(gaiaquery['Period']) ] , total=max_):
            try:
                # Query VSX catalog using Gaia DR2 ID
                vsxqueryper = Vizier(timeout=300).query_object('Gaia DR2 '+source['source_id'].astype('str'),catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-source['ra'])**2 + \
                                        (vsxqueryper['DEJ2000']-source['dec'])**2
                vsxqueryper.sort('dist')
                vsxqueryper = vsxqueryper['Period'][0]
                gaiaquery['Period'][ gaiaquery['source_id'] == source['source_id'] ] = vsxqueryper
            except requests.exceptions.ConnectionError:
                sleep(1)
                vsxqueryper = Vizier(timeout=300).query_object('Gaia DR2 '+source['source_id'].astype('str'),catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-source['ra'])**2 + \
                                        (vsxqueryper['DEJ2000']-source['dec'])**2
                vsxqueryper.sort('dist')
                vsxqueryper = vsxqueryper['Period'][0]
                gaiaquery['Period'][ gaiaquery['source_id'] == source['source_id'] ] = vsxqueryper
            except (IndexError,KeyError):
                try:
                    # Query VSX catalog using Name
                    name = data['Name'][ data['Source'] == source['source_id'] ][0]
                    name = str(name).replace('_',' ')
                    vsxqueryper = Vizier(timeout=300).query_object(name,catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                    vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-source['ra'])**2 + \
                                            (vsxqueryper['DEJ2000']-source['dec'])**2
                    vsxqueryper.sort('dist')
                    vsxqueryper = vsxqueryper['Period'][0]
                    gaiaquery['Period'][ gaiaquery['source_id'] == source['source_id'] ] = vsxqueryper
                except requests.exceptions.ConnectionError:
                    sleep(1)
                    name = data['Name'][ data['Source'] == source['source_id'] ][0]
                    name = str(name).replace('_',' ')
                    vsxqueryper = Vizier(timeout=300).query_object(name,catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                    vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-source['ra'])**2 + \
                                            (vsxqueryper['DEJ2000']-source['dec'])**2
                    vsxqueryper.sort('dist')
                    vsxqueryper = vsxqueryper['Period'][0]
                    gaiaquery['Period'][ gaiaquery['source_id'] == source['source_id'] ] = vsxqueryper
                except (IndexError,KeyError):
                    pass

    gaiaquery.rename_columns(['source_id'],['Source'])

    # Handle missing DR2 IDs
    if useEDR3:
        gaiaquery['dr2_source_id'][ gaiaquery['dr2_source_id'].mask ] = 0000000000

    # ------ SIMBAD catalog for V,J,H,K mags -----------
    # Query all objects from Simbad at once
    print('Downloading Simbad data for all stars...')
    sources = []
    if not useEDR3:
        for source in data['Source']:
            sources.append('Gaia DR2 '+source.astype('str')  )
    else:
        for source in gaiaquery['dr2_source_id']:
            try:
                sources.append('Gaia DR2 '+source.astype('str')  )
            except:
                sources.append('Gaia DR2 '+'0000000000'  )

    simbadcols = ['ID_Gaia','FLUX_V','FLUX_ERROR_V','FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K']
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            simbadquery = Simbad.query_objects(sources)[simbadcols]
        except requests.exceptions.ConnectionError:
            sleep(1)
            simbadquery = Simbad.query_object(sources)[simbadcols]

    # Convert 'Gaia DR2 ID' format to 'ID' format in Simbad Query
    simbadquery.rename_columns(['ID_Gaia'],['Source'])
    for i in range(len(simbadquery)):
        simbadquery['Source'][i] = int(simbadquery['Source'].astype('str')[i][9:])
    simbadquery['Source'] = simbadquery['Source'].astype('str')

    # ------ Merge catalogs -----------
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = join(data,gaiaquery,join_type='left',keys='Source')
        data['Source'] = data['Source'].astype('str') # Convert Source to str to do join
        if not useEDR3:
            data = join(data,simbadquery,join_type='left',keys='Source')
        else:
            simbadquery.rename_columns(['Source'],['dr2_source_id'])
            data['dr2_source_id'] = data['dr2_source_id'].astype('str')
            # Handle missing DR2 IDs
            data['dr2_source_id'][ data['dr2_source_id'].mask ] = 0000000000
            data = join(data,simbadquery,join_type='left',keys='dr2_source_id')

    data.rename_columns(['parallax','parallax_error','phot_g_mean_mag','phot_g_mean_mag_error','phot_bp_mean_mag','phot_bp_mean_mag_error','phot_rp_mean_mag','phot_rp_mean_mag_error'],
                        ['plx','sig_plx','gamag','sig_gamag','bpmag','sig_bpmag','rpmag','sig_rpmag'])

    data.rename_columns(['FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K'],
                        ['jmag','sig_jmag','hmag','sig_hmag','kmag','sig_kmag'])

    data.rename_columns(['FLUX_V','FLUX_ERROR_V'],
                        ['vmag','sig_vmag'])
    data = data.filled(-99)

    # ----- If Gaia ID is not listed in Simbad use Name instead -----------
    # ----- NOT USED BECAUSE Gaia IDs AND NAMES DO NOT MATCH !!! -----------
    #where_simbad_failed = np.logical_and( data['vmag'] == -99, np.logical_and(data['jmag'] == -99, np.logical_and(data['hmag'] == -99, data['kmag'] == -99)))
    #with warnings.catch_warnings(record=True) as w:
    #    warnings.simplefilter("always")
    #    try:
    #        simbadquery = Simbad.query_object(data[ where_simbad_failed ]['Name'])[simbadcols]
    #    except requests.exceptions.ConnectionError:
    #        sleep(1)
    #        simbadquery = Simbad.query_object(data[ where_simbad_failed ]['Name'])[simbadcols]

    # Convert 'Gaia DR2 ID' format to 'ID' format in Simbad Query
    #simbadquery.rename_columns(['ID_Gaia'],['Source'])
    #for i in range(len(simbadquery)):
    #    simbadquery['Source'][i] = int(simbadquery['Source'].astype('str')[i][9:])
    #simbadquery['Source'] = simbadquery['Source'].astype('str')

    # Replace Simbad result by Gaia ID with Simbad results by Name
    #data = join(data,simbadquery,join_type='left',keys='Source')
    #data['vmag','sig_vmag','jmag','sig_jmag','hmag','sig_hmag','kmag','sig_kmag'][where_simbad_failed] = data['FLUX_V','FLUX_ERROR_V','FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K'][where_simbad_failed]
    #data.remove_columns(['FLUX_V','FLUX_ERROR_V','FLUX_J','FLUX_ERROR_J','FLUX_H','FLUX_ERROR_H','FLUX_K','FLUX_ERROR_K'])
    #data = data.filled(-99)

    # ------ Start calculations -----------
    print('Calculating distances, absolute magnitudes...')
    outdata = []

    if sys.version_info < (3, 8):
        ncores = cpu_count()
        with Pool(processes=ncores) as p:
            max_ = len(data)
            with tqdm(total=max_) as pbar:
                if not useEDR3:
                    for i,result in enumerate(p.imap_unordered(get_dist_absmag, np.arange(max_).tolist() )):
                        outdata.append(result)
                        pbar.update()
                else:
                    for i,result in enumerate(p.imap_unordered(get_dist_absmag_edr3, np.arange(max_).tolist() )):
                        outdata.append(result)
                        pbar.update()
    else:
        warnings.warn(
            'Python>=3.8 detected, running on single thread')

        max_ = len(data)
        if not useEDR3:
            for i in tqdm(np.arange(max_).tolist() ):
                result = get_dist_absmag(i)
                outdata.append(result)
        else:
            for i in tqdm(np.arange(max_).tolist() ):
                result = get_dist_absmag_edr3(i)
                outdata.append(result)

    outdataTable = Table(names=['Source',
                          'dis','disep','disem',
                          'lon_deg', 'lat_deg',
                          'ag','agep','agem','Mg','Mgep','Mgem',
                          'aBP','aBPep','aBPem','MBP','MBPep','MBPem',
                          'aRP','aRPep','aRPem','MRP','MRPep','MRPem',
                          'aV','aVep','aVem','MV','MVep','MVem',
                          'aJ','aJep','aJem','MJ','MJep','MJem',
                          'aH','aHep','aHem','MH','MHep','MHem',
                          'aK','aKep','aKem','MK','MKep','MKem'],
                          dtype=('i8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))

    #outdataTable['Source'].format = '%d'
    for row in outdata:
        outdataTable.add_row(row)

    outdataTable['Source'] = outdataTable['Source'].astype('str')
    if useEDR3:
        data.remove_columns(['r_med_geo', 'r_lo_geo', 'r_hi_geo'])
        data.remove_columns(['r_med_photogeo', 'r_lo_photogeo', 'r_hi_photogeo'])
    data = join(data,outdataTable,join_type='left',keys='Source')
    ascii.write(data, outfilename, format='basic', fast_writer=False, overwrite=True)
