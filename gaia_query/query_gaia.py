import sys
#import ebf
import numpy as np
#import scipy.interpolate
#import pdb
#import asfgrid
from astropy.io import ascii
from astropy.table import join,vstack,Table,Column,unique
#from classify_direct import *
from classify_direct_2_Mv import *
from astroquery.vizier import Vizier
import warnings
import requests
from time import sleep
from astropy.coordinates import Angle
import sys
from multiprocessing import Pool, cpu_count
from tqdm import *

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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("# Usage:")
        print("\t python query_gaia.py <inputfile> (<options>)")
        print('\t Inputfile must be in format:')
        print('\t GaiaID RA DEC Name')
        print('\t Options:')
        print('\t --Stassun : use plx offset -80   mas (Stassun et al. 2018)')
        print('\t --Riess   : use plx offset -46   mas (Riess et al. 2018)')
        print('\t --BJ      : use plx offset -29   mas (BJ et al. 2018)')
        print('\t --Zinn    : use plx offset -52.8 mas (Zinn et al. 2019)')
        exit(0)

    # ------ For plx correction -----------
    if len(sys.argv)==3:
        if sys.argv[2]   == '--Stassun': plx_offset = -0.08   #mas #Stassun et al. 2018
        elif sys.argv[2] == '--Riess':   plx_offset = -0.046  #mas #Riess et al. 2018
        elif sys.argv[2] == '--BJ':      plx_offset = -0.029  #mas #BJ et al. 2018
        elif sys.argv[2] == '--Zinn':    plx_offset = -0.0528 #mas #Zinn et al 2018
        else: plx_offset = 0.0
    else:
         plx_offset = 0.0

    # ------ Parse stdin -----------
    infilename = sys.argv[1]
    lenOfExtension = len(infilename.split('.')[-1])
    infilenameShort = infilename[:-(lenOfExtension+1)] # remove extension
    if len(sys.argv)==3:
        if sys.argv[2] == '--Stassun': outfilename = infilenameShort+'_MgaiaJHK_Stassun.txt'
        elif sys.argv[2] == '--Riess': outfilename = infilenameShort+'_MgaiaJHK_Riess.txt'
        elif sys.argv[2] == '--BJ': outfilename = infilenameShort+'_MgaiaJHK_BJ.txt'
        elif sys.argv[2] == '--Zinn': outfilename = infilenameShort+'_MgaiaJHK_Zinn.txt'
        else: outfilename = infilenameShort+'_MgaiaJHK.txt'
    else: outfilename = infilenameShort+'_MgaiaJHK.txt'
    data=ascii.read(infilename,names=['Source','ra','dec','Name'])
    #data['Source'].format = '%d'
    data = unique(data, keys='Source')


    #dnumodel = asfgrid.Seism()
    #bcmodel = h5py.File('bcgrid.h5', 'r')
    #dustmodel = mwdust.Green19()
    dustmodel = mwdust.Combined19()

    # --- Query Gaia, 2MASS & VSX catalogues --------
    gaia = Table(names=['Source','Plx','e_Plx','Gmag','e_Gmag','BPmag','e_BPmag','RPmag','e_RPmag'],
                 dtype=['int64','float64','float64','float64','float64','float64','float64','float64','float64'])
    #twomass = Table(names=['Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'],
    #                dtype=['float32','float32','float32','float32','float32','float32'])
    period = Table(names=['Period'], dtype=['float64'])

    max_ = len(data['Source'])
    print('Dowloading Gaia & VSX data...')
    for i,source in tqdm(enumerate(data['Source']), total=max_):
        #print('Dowloading Gaia & VSX data for %s (%d/%d)' % (data['Name'][i],i+1,len(data)) )

        # ------ Gaia catalog -----------
        try:
            gaiaquery = Vizier(catalog=['I/345/gaia2'],
                           column_filters={"Source":source.astype('str')},
                           columns=['Source','Plx','e_Plx','Gmag','e_Gmag','BPmag','e_BPmag','RPmag','e_RPmag']).get_catalogs('I/345/gaia2')[0]
        except requests.exceptions.ConnectionError:
            sleep(1)
            gaiaquery = Vizier(catalog=['I/345/gaia2'],
                           column_filters={"Source":source.astype('str')},
                           columns=['Source','Plx','e_Plx','Gmag','e_Gmag','BPmag','e_BPmag','RPmag','e_RPmag']).get_catalogs('I/345/gaia2')[0]

        '''
        # ------ 2MASS catalog -----------
        try:
            twomassquery = Vizier.query_object('Gaia DR2 ' + \
                source.astype('str'),catalog='II/246',radius=Angle(10, "arcsec"))[0]
            twomassquery['dist'] = (twomassquery['RAJ2000']-data['ra'][i])**2 + \
                                    (twomassquery['DEJ2000']-data['dec'][i])**2
            twomassquery.sort('dist')
            twomassquery = twomassquery['Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'][0]
            twomassquery = Table(twomassquery)
            twomassquery.add_column( Column(source, name='Source') )
        except requests.exceptions.ConnectionError:
            sleep(1)
            twomassquery = Vizier.query_object('Gaia DR2 ' + \
                source.astype('str'),catalog='II/246',radius=Angle(10, "arcsec"))[0]
            twomassquery['dist'] = (twomassquery['RAJ2000']-data['ra'][i])**2 + \
                                    (twomassquery['DEJ2000']-data['dec'][i])**2
            twomassquery.sort('dist')
            twomassquery = twomassquery['Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'][0]
            twomassquery = Table(twomassquery)
            twomassquery.add_column( Column(source, name='Source') )
        except IndexError:
            twomassquery = Column([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
            twomassquery = Table(twomassquery)
            twomassquery.rename_columns(['col0','col1','col2','col3','col4','col5'],['Jmag','e_Jmag','Hmag','e_Hmag','Kmag','e_Kmag'])
            twomassquery.add_column( Column(source, name='Source') )
        '''

        # ------ Gaia period or VSX period -----------
        try:
            try:
                # Query Gaia RRL catalog
                vsxqueryper = Vizier(catalog=['I/345/gaia2'],column_filters={"Source":source.astype('str')},
                   columns=['Source','Pf','P1O']).get_catalogs('I/345/rrlyrae')[0]
                if vsxqueryper['Pf'] > 0:
                    vsxqueryper = vsxqueryper['Pf']
                elif vsxqueryper['P1O'] > 0:
                    vsxqueryper = vsxqueryper['P1O']
                vsxquery = Table(names=['Period'])
                vsxquery.add_row( Column([vsxqueryper], name='Period') )
            except requests.exceptions.ConnectionError:
                sleep(1)
                vsxqueryper = Vizier(catalog=['I/345/gaia2'],column_filters={"Source":source.astype('str')},
                   columns=['Source','Pf','P1O']).get_catalogs('I/345/rrlyrae')[0]
                if vsxqueryper['Pf'] > 0:
                    vsxqueryper = vsxqueryper['Pf']
                elif vsxqueryper['P1O'] > 0:
                    vsxqueryper = vsxqueryper['P1O']
                vsxquery = Table(names=['Period'])
                vsxquery.add_row( Column([vsxqueryper], name='Period') )
        except (IndexError,KeyError):
            try:
                try:
                    # Query Gaia Cepheid catalog
                    vsxqueryper = Vizier(catalog=['I/345/gaia2'],column_filters={"Source":source.astype('str')},
                       columns=['Source','Pf','P1O','P2O','P3O']).get_catalogs('I/345/cepheid')[0]
                    if vsxqueryper['Pf'] > 0:
                        vsxqueryper = vsxqueryper['Pf']
                    elif vsxqueryper['P1O'] > 0:
                        vsxqueryper = vsxqueryper['P1O']
                    elif vsxqueryper['P2O'] > 0:
                        vsxqueryper = vsxqueryper['P2O']
                    elif vsxqueryper['P3O'] > 0:
                        vsxqueryper = vsxqueryper['P3O']
                    vsxquery = Table(names=['Period'])
                    vsxquery.add_row( Column([vsxqueryper], name='Period') )
                except requests.exceptions.ConnectionError:
                    sleep(1)
                    vsxqueryper = Vizier(catalog=['I/345/gaia2'],column_filters={"Source":source.astype('str')},
                       columns=['Source','Pf','P1O','P2O','P3O']).get_catalogs('I/345/cepheid')[0]
                    if vsxqueryper['Pf'] > 0:
                        vsxqueryper = vsxqueryper['Pf']
                    elif vsxqueryper['P1O'] > 0:
                        vsxqueryper = vsxqueryper['P1O']
                    elif vsxqueryper['P2O'] > 0:
                        vsxqueryper = vsxqueryper['P2O']
                    elif vsxqueryper['P3O'] > 0:
                        vsxqueryper = vsxqueryper['P3O']
                    vsxquery = Table(names=['Period'])
                    vsxquery.add_row( Column([vsxqueryper], name='Period') )
            except (IndexError,KeyError):
                try:
                    # Query VSX catalog using Gaia DR2 ID
                    vsxqueryper = Vizier(timeout=300).query_object('Gaia DR2 '+source.astype('str'),catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                    vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-data['ra'][i])**2 + \
                                            (vsxqueryper['DEJ2000']-data['dec'][i])**2
                    vsxqueryper.sort('dist')
                    vsxqueryper = vsxqueryper['Period'][0]
                    vsxquery = Table(names=['Period'])
                    vsxquery.add_row( Column([vsxqueryper], name='Period') )
                except requests.exceptions.ConnectionError:
                    sleep(1)
                    vsxqueryper = Vizier(timeout=300).query_object('Gaia DR2 '+source.astype('str'),catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                    vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-data['ra'][i])**2 + \
                                            (vsxqueryper['DEJ2000']-data['dec'][i])**2
                    vsxqueryper.sort('dist')
                    vsxqueryper = vsxqueryper['Period'][0]
                    vsxquery = Table(names=['Period'])
                    vsxquery.add_row( Column([vsxqueryper], name='Period') )
                except (IndexError,KeyError):
                    try:
                        # Query VSX catalog using Name
                        vsxqueryper = Vizier(timeout=300).query_object(data['Name'][i],catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                        vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-data['ra'][i])**2 + \
                                                (vsxqueryper['DEJ2000']-data['dec'][i])**2
                        vsxqueryper.sort('dist')
                        vsxqueryper = vsxqueryper['Period'][0]
                        vsxquery = Table(names=['Period'])
                        vsxquery.add_row( Column([vsxqueryper], name='Period') )
                    except requests.exceptions.ConnectionError:
                        sleep(1)
                        vsxqueryper = Vizier(timeout=300).query_object(data['Name'][i],catalog='vsx',radius=Angle(10, unit='arcsec'))[0]
                        vsxqueryper['dist'] = (vsxqueryper['RAJ2000']-data['ra'][i])**2 + \
                                                (vsxqueryper['DEJ2000']-data['dec'][i])**2
                        vsxqueryper.sort('dist')
                        vsxqueryper = vsxqueryper['Period'][0]
                        vsxquery = Table(names=['Period'])
                        vsxquery.add_row( Column([vsxqueryper], name='Period') )
                    except (IndexError,KeyError):
                        vsxquery = Column([np.nan])
                        vsxquery = Table(vsxquery)
                        vsxquery.rename_columns(['col0'],['Period'])
        vsxquery.add_column( Column(source, name='Source') )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gaia = vstack([gaia,gaiaquery])
            #twomass = vstack([twomass,twomassquery])
            period = vstack([period,vsxquery])
            #vmag = vstack([vmag,simbadquery])

    # ------ SIMBAD catalog for V,J,H,K mags -----------
    # Query all objects from Simbad at once
    print('Dowloading Simbad data for all stars...')
    sources = []
    for source in data['Source']:
        sources.append('Gaia DR2 '+source.astype('str')  )

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
        data = join(data,gaia,join_type='left',keys='Source')
        #data = join(data,twomass,join_type='left',keys='Source')
        data = join(data,period,join_type='left',keys='Source')
        data['Source'] = data['Source'].astype('str') # Convert Source to str to do join
        data = join(data,simbadquery,join_type='left',keys='Source')

    data.rename_columns(['Plx','e_Plx','Gmag','e_Gmag','BPmag','e_BPmag','RPmag','e_RPmag'],
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

    ncores = cpu_count()
    with Pool(processes=ncores) as p:
        max_ = len(data)
        with tqdm(total=max_) as pbar:
            for i,result in enumerate(p.imap_unordered(get_dist_absmag, np.arange(max_).tolist() )):
                outdata.append(result)
                pbar.update()

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
    data = join(data,outdataTable,join_type='left',keys='Source')
    ascii.write(data, outfilename, format='basic', fast_writer=False, overwrite=True)
