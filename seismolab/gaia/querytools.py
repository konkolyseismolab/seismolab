import numpy as np
import ephem
import warnings
from astroquery.gaia import Gaia

def get_dist_absmag(i,data,dustmodel,plx_offset):
    # Check if it makes sense to run the calculations
    if ((data['plx'][i] + plx_offset) <= 0.) or ((data['sig_plx'][i]/ (data['plx'][i] + plx_offset)) > 1. ):
        dis,disep,disem = np.nan,np.nan,np.nan
        lon_deg, lat_deg = np.nan,np.nan
        agaia,agaiaep,agaiaem = np.nan,np.nan,np.nan
        absgaia,absgaiaep,absgaiaem = np.nan,np.nan,np.nan
        appgaia,appgaiaep,appgaiaem = np.nan,np.nan,np.nan
        aBP,aBPep,aBPem = np.nan,np.nan,np.nan
        absBP,absBPep,absBPem = np.nan,np.nan,np.nan
        appBP,appBPep,appBPem = np.nan,np.nan,np.nan
        aRP,aRPep,aRPem = np.nan,np.nan,np.nan
        absRP,absRPep,absRPem = np.nan,np.nan,np.nan
        appRP,appRPep,appRPem = np.nan,np.nan,np.nan
        aB,aBep,aBem = np.nan,np.nan,np.nan
        absB,absBep,absBem = np.nan,np.nan,np.nan
        appB,appBep,appBem = np.nan,np.nan,np.nan
        aV,aVep,aVem = np.nan,np.nan,np.nan
        absV,absVep,absVem = np.nan,np.nan,np.nan
        appV,appVep,appVem = np.nan,np.nan,np.nan
        aJ,aJep,aJem = np.nan,np.nan,np.nan
        absJ,absJep,absJem = np.nan,np.nan,np.nan
        appJ,appJep,appJem = np.nan,np.nan,np.nan
        aH,aHep,aHem = np.nan,np.nan,np.nan
        absH,absHep,absHem = np.nan,np.nan,np.nan
        appH,appHep,appHem = np.nan,np.nan,np.nan
        aK,aKep,aKem = np.nan,np.nan,np.nan
        absK,absKep,absKem = np.nan,np.nan,np.nan
        appK,appKep,appKem = np.nan,np.nan,np.nan

        outdata = [data['Source'][i],
                        dis,disep,disem,
                        lon_deg, lat_deg,
                        agaia,agaiaep,agaiaem,
                        absgaia,absgaiaep,absgaiaem,
                        appgaia,appgaiaep,appgaiaem,
                        aBP,aBPep,aBPem,
                        absBP,absBPep,absBPem,
                        appBP,appBPep,appBPem,
                        aRP,aRPep,aRPem,
                        absRP,absRPep,absRPem,
                        appRP,appRPep,appRPem,
                        aB,aBep,aBem,
                        absB,absBep,absBem,
                        appB,appBep,appBem,
                        aV,aVep,aVem,
                        absV,absVep,absVem,
                        appV,appVep,appVem,
                        aJ,aJep,aJem,
                        absJ,absJep,absJem,
                        appJ,appJep,appJem,
                        aH,aHep,aHem,
                        absH,absHep,absHem,
                        appH,appHep,appHem,
                        aK,aKep,aKem,
                        absK,absKep,absKem,
                        appK,appKep,appKem]
        return outdata

    # ------ Gaia mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addgaia(data['gamag'][i],data['sig_gamag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    dis,disep,disem = paras.dis, paras.disep, paras.disem
    agaia,agaiaep,agaiaem = paras.avs, paras.avsep, paras.avsem
    absgaia,absgaiaep,absgaiaem = paras.absmag, paras.absmagep, paras.absmagem
    appgaia,appgaiaep,appgaiaem = paras.appmag,paras.appmagep,paras.appmagem
    lon_deg, lat_deg = paras.lon_deg, paras.lat_deg

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addBP(data['bpmag'][i],data['sig_bpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aBP,aBPep,aBPem = paras.avs, paras.avsep, paras.avsem
    absBP,absBPep,absBPem = paras.absmag, paras.absmagep, paras.absmagem
    appBP,appBPep,appBPem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ Gaia BP mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addRP(data['rpmag'][i],data['sig_rpmag'][i])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aRP,aRPep,aRPem = paras.avs, paras.avsep, paras.avsem
    absRP,absRPep,absRPem = paras.absmag, paras.absmagep, paras.absmagem
    appRP,appRPep,appRPem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ B mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([data['bmag'][i],-99.,-99.,-99.],[data['sig_bmag'][i],-99.,-99.,-99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aB,aBep,aBem = paras.avs, paras.avsep, paras.avsem
    absB,absBep,absBem = paras.absmag, paras.absmagep, paras.absmagem
    appB,appBep,appBem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ V mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([-99.,data['vmag'][i],-99.,-99.],[-99.,data['sig_vmag'][i],-99.,-99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aV,aVep,aVem = paras.avs, paras.avsep, paras.avsem
    absV,absVep,absVem = paras.absmag, paras.absmagep, paras.absmagem
    appV,appVep,appVem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ J mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([data['jmag'][i], -99., -99. ],\
             [data['sig_jmag'][i], -99., -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aJ,aJep,aJem = paras.avs, paras.avsep, paras.avsem
    absJ,absJep,absJem = paras.absmag, paras.absmagep, paras.absmagem
    appJ,appJep,appJem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ H mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([ -99., data['hmag'][i], -99. ],\
             [ -99., data['sig_hmag'][i], -99.])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aH,aHep,aHem = paras.avs, paras.avsep, paras.avsem
    absH,absHep,absHem = paras.absmag, paras.absmagep, paras.absmagem
    appH,appHep,appHem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ K mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addjhk([-99., -99., data['kmag'][i] ],\
             [-99., -99., data['sig_kmag'][i]])
    x.addplx((data['plx'][i])/1e3 + plx_offset/1e3 , data['sig_plx'][i]/1e3)

    paras=stparas(input=x,dustmodel=dustmodel)

    aK,aKep,aKem = paras.avs, paras.avsep, paras.avsem
    absK,absKep,absKem = paras.absmag, paras.absmagep, paras.absmagem
    appK,appKep,appKem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ Output -----------
    outdata = [data['Source'][i],
                    dis,disep,disem,
                    lon_deg, lat_deg,
                    agaia,agaiaep,agaiaem,
                    absgaia,absgaiaep,absgaiaem,
                    appgaia,appgaiaep,appgaiaem,
                    aBP,aBPep,aBPem,
                    absBP,absBPep,absBPem,
                    appBP,appBPep,appBPem,
                    aRP,aRPep,aRPem,
                    absRP,absRPep,absRPem,
                    appRP,appRPep,appRPem,
                    aB,aBep,aBem,
                    absB,absBep,absBem,
                    appB,appBep,appBem,
                    aV,aVep,aVem,
                    absV,absVep,absVem,
                    appV,appVep,appVem,
                    aJ,aJep,aJem,
                    absJ,absJep,absJem,
                    appJ,appJep,appJem,
                    aH,aHep,aHem,
                    absH,absHep,absHem,
                    appH,appHep,appHem,
                    aK,aKep,aKem,
                    absK,absKep,absKem,
                    appK,appKep,appKem]
    return outdata

def get_dist_absmag_edr3(i,data,dustmodel,plx_offset,use_photodist):

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
    appgaia,appgaiaep,appgaiaem = paras.appmag,paras.appmagep,paras.appmagem
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
    appBP,appBPep,appBPem = paras.appmag,paras.appmagep,paras.appmagem

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
    appRP,appRPep,appRPem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ B mag -----------
    x=obsdata()
    x.addcoords(data['ra'][i],data['dec'][i])
    x.addbvri([data['bmag'][i],-99.,-99.,-99.],[data['sig_bmag'][i],-99.,-99.,-99.])
    if use_photodist:
        x.addBJdis(data['r_med_photogeo'][i],data['r_hi_photogeo'][i],data['r_lo_photogeo'][i])
    else:
        x.addBJdis(data['r_med_geo'][i],data['r_hi_geo'][i],data['r_lo_geo'][i])

    paras=stparas_edr3(input=x,dustmodel=dustmodel)

    aB,aBep,aBem = paras.avs, paras.avsep, paras.avsem
    absB,absBep,absBem = paras.absmag, paras.absmagep, paras.absmagem
    appB,appBep,appBem = paras.appmag,paras.appmagep,paras.appmagem

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
    appV,appVep,appVem = paras.appmag,paras.appmagep,paras.appmagem

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
    appJ,appJep,appJem = paras.appmag,paras.appmagep,paras.appmagem

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
    appH,appHep,appHem = paras.appmag,paras.appmagep,paras.appmagem

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
    appK,appKep,appKem = paras.appmag,paras.appmagep,paras.appmagem

    # ------ Output -----------
    outdata = [data['Source'][i],
                    dis,disep,disem,
                    lon_deg, lat_deg,
                    agaia,agaiaep,agaiaem,
                    absgaia,absgaiaep,absgaiaem,
                    appgaia,appgaiaep,appgaiaem,
                    aBP,aBPep,aBPem,
                    absBP,absBPep,absBPem,
                    appBP,appBPep,appBPem,
                    aRP,aRPep,aRPem,
                    absRP,absRPep,absRPem,
                    appRP,appRPep,appRPem,
                    aB,aBep,aBem,
                    absB,absBep,absBem,
                    appB,appBep,appBem,
                    aV,aVep,aVem,
                    absV,absVep,absVem,
                    appV,appVep,appVem,
                    aJ,aJep,aJem,
                    absJ,absJep,absJem,
                    appJ,appJep,appJem,
                    aH,aHep,aHem,
                    absH,absHep,absHem,
                    appH,appHep,appHem,
                    aK,aKep,aKem,
                    absK,absKep,absKem,
                    appK,appKep,appKem]
    return outdata

def perform_query(data,useEDR3):
    # Prepare target list by number of targets
    if len(data['Source']) >1:
        targetlist = str(tuple(data['Source']))
        qrelation = "IN"
    elif len(data['Source']) ==1:
        targetlist = str(tuple(data['Source'])[0])
        qrelation = "="

    if useEDR3:
        print('Querying Gaia DR3 and BJ catalogs...')
        import pyvo as vo
        # Initialize BJ catalog
        service = vo.dal.TAPService("https://dc.zah.uni-heidelberg.de/__system__/tap/run/tap")

        # Query BJ catalog
        BJdist = service.search("SELECT TOP 1000000 BJ.source_id, \
                            BJ.r_med_geo,BJ.r_lo_geo,BJ.r_hi_geo, \
                            BJ.r_med_photogeo,BJ.r_lo_photogeo,BJ.r_hi_photogeo \
                            FROM gedr3dist.litewithdist as BJ \
                            WHERE BJ.source_id "+qrelation+" "+targetlist)
        BJdist = BJdist.to_table()

        # Query Gaia catalog
        job = Gaia.launch_job("SELECT TOP 1000000 EDR3.source_id,EDR3.ra,EDR3.dec,"
                  "EDR3.parallax,EDR3.parallax_error,"
                  "EDR3.phot_g_mean_flux,EDR3.phot_g_mean_flux_error,EDR3.phot_g_mean_mag,"
                  "EDR3.phot_bp_mean_flux,EDR3.phot_bp_mean_flux_error,EDR3.phot_bp_mean_mag,"
                  "EDR3.phot_rp_mean_flux,EDR3.phot_rp_mean_flux_error,EDR3.phot_rp_mean_mag "
                  "FROM gaiaedr3.gaia_source AS EDR3 "
                  "WHERE EDR3.source_id "+qrelation+" "+targetlist)
        gaiaquery = job.get_results()

        return gaiaquery, BJdist

    else:
        print('Querying Gaia DR2 catalog...')

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
                  "WHERE DR2.source_id "+qrelation+" "+targetlist)
        gaiaquery = job.get_results()

        return gaiaquery, None

def stparas(input,dustmodel):

    # object containing output values
    out = resdata()

    ## extinction coefficients
    extfactors=extinction()

    if input.plxe>0.:

        # pick an apparent magnitude from input
        map=-99.
        if (input.vmag > -99.):
            map = input.vmag
            if input.vmage>-99: mape = input.vmage
            else: mape = 0.02
            avtoext=extfactors.av

        elif (input.bmag > -99.):
            map = input.bmag
            if input.bmage>-99: mape = input.bmage
            else: mape = 0.02
            avtoext=extfactors.ab

        elif (input.imag > -99.):
            map = input.imag
            if input.image>-99: mape = input.image
            else: mape = 0.02
            avtoext=0.

        elif (input.vtmag > -99.):
            map = input.vtmag
            if input.vtmage>-99: mape = input.vtmage
            else: mape = 0.02
            avtoext=extfactors.avt

        elif (input.jmag > -99.):
            map = input.jmag
            if input.jmage>-99: mape = input.jmage
            else: mape = 0.02
            avtoext=extfactors.aj

        elif (input.hmag > -99.):
            map = input.hmag
            if input.hmage>-99: mape = input.hmage
            else: mape = 0.02
            avtoext=extfactors.ah

        elif (input.kmag > -99.):
            map = input.kmag
            if input.kmage>-99: mape = input.kmage
            else: mape = 0.02
            avtoext=extfactors.ak

        elif (input.gamag > -99.):
            map = input.gamag
            if input.gamage>-99: mape = input.gamage
            else: mape = 0.02
            avtoext=extfactors.aga

        elif (input.bpmag > -99.):
            map = input.bpmag
            if input.bpmage>-99: mape = input.bpmage
            else: mape = 0.02
            avtoext=extfactors.abp

        elif (input.rpmag > -99.):
            map = input.rpmag
            if input.rpmage>-99: mape = input.rpmage
            else: mape = 0.02
            avtoext=extfactors.arp

        else:
            return out

        ### Monte Carlo starts here

        # number of samples
        nsample=int(1e5)

        # length scale for exp decreasing vol density prior in pc
        L=1350.

        # maximum distance to sample (in pc)
        maxdis = 1e5

        # get a rough maximum distance
        tempdis=1./input.plx
        tempdise=input.plxe/input.plx**2
        maxds = tempdis + 5.0*tempdise
        minds = tempdis - 5.0*tempdise

        ds=np.arange(1.,maxdis,1.)
        lh = ((1.0/(np.sqrt(2.0*np.pi)*input.plxe)) * \
             np.exp( (-1./(2.*input.plxe**2))*(input.plx-1./ds)**2))
        prior=(ds**2/(2.*L**3.))*np.exp(-ds/L)
        dis = lh*prior
        dis2=dis/np.sum(dis)
        norm=dis2/np.max(dis2)

        # Deal with negative and positive parallaxes differently:
        if tempdis > 0:
            # Determine maxds based on posterior:
            um = np.where((ds > tempdis) & (norm < 0.001))[0]

            # Determine minds just like maxds:
            umin = np.where((ds < tempdis) & (norm < 0.001))[0]
        else:
            # Determine maxds based on posterior, taking argmax
            # instead of tempdis which is wrong:
            um = np.where((ds > np.argmax(norm)) & (norm < 0.001))[0]

            # Determine minds just like maxds:
            umin = np.where((ds < np.argmax(norm)) & (norm < 0.001))[0]

        if (len(um) > 0):
            maxds = np.min(ds[um])
        else:
            maxds = 1e5

        if (len(umin) > 0):
            minds = np.max(ds[umin])
        else:
            minds = 1.0

        #print( 'using max distance:',maxds,tempdis )
        ds=np.linspace(minds,maxds,nsample)
        lh = (1./(np.sqrt(2.*np.pi)*input.plxe))*\
             np.exp( (-1./(2.*input.plxe**2))*(input.plx-1./ds)**2)
        prior=(ds**2/(2.*L**3.))*np.exp(-ds/L)
        #prior=np.zeros(len(lh))+1.
        dis = lh*prior
        dis2=dis/np.sum(dis)

        # sample distances following the discrete distance posterior
        np.random.seed(seed=10)
        try:
            dsamp=np.random.choice(ds,p=dis2,size=nsample)
        except ValueError as error:
            warnings.warn(error)
            return out

        # get l,b from RA,DEC
        equ = ephem.Equatorial(input.ra*np.pi/180., input.dec*np.pi/180., epoch=ephem.J2000)
        gal = ephem.Galactic(equ)
        lon_deg=gal.lon*180./np.pi
        lat_deg=gal.lat*180./np.pi

        # Get a well sampled extinction map at a given l,b
        distanceSamples = np.geomspace(0.063095734448,59.5662143529) # in kpc

        reddenContainer = dustmodel(lon_deg,lat_deg,distanceSamples)

        # Sample extinction
        xp = np.concatenate(( [0.0], distanceSamples ))
        fp = np.concatenate(( [0.0], reddenContainer ))
        ebvs=np.interp(x=dsamp/1000., xp=xp, fp=fp)
        avs = avtoext*ebvs

        del reddenContainer # To clear reddenMap from memory
        del distanceSamples

        np.random.seed(seed=12)
        map_samp=map+np.random.randn(nsample)*mape
        absmag = -5.*np.log10(dsamp)-avs+map_samp+5.

        appmag = map_samp-avs

        out.avs=np.median(avs)
        out.avsep=np.percentile(avs,84.1)-out.avs
        out.avsem=out.avs-np.percentile(avs,15.9)

        out.absmag,out.absmagep,out.absmagem=getstat(absmag)
        out.dis,out.disep,out.disem=getstat(dsamp)

        out.appmag,out.appmagep,out.appmagem=getstat(appmag)

        out.lon_deg = lon_deg
        out.lat_deg = lat_deg

    return out

def stparas_edr3(input,dustmodel):

    # object containing output values
    out = resdata()

    # extinction coefficients
    extfactors=extinction()

    # pick an apparent magnitude from input
    map=-99.
    if (input.vmag > -99.):
        map = input.vmag
        if input.vmage>-99: mape = input.vmage
        else: mape = 0.02
        avtoext=extfactors.av

    elif (input.bmag > -99.):
        map = input.bmag
        if input.bmage>-99: mape = input.bmage
        else: mape = 0.02
        avtoext=extfactors.ab

    elif (input.imag > -99.):
        map = input.imag
        if input.image>-99: mape = input.image
        else: mape = 0.02
        avtoext=0.

    elif (input.vtmag > -99.):
        map = input.vtmag
        if input.vtmage>-99: mape = input.vtmage
        else: mape = 0.02
        avtoext=extfactors.avt

    elif (input.jmag > -99.):
        map = input.jmag
        if input.jmage>-99: mape = input.jmage
        else: mape = 0.02
        avtoext=extfactors.aj

    elif (input.hmag > -99.):
        map = input.hmag
        if input.hmage>-99: mape = input.hmage
        else: mape = 0.02
        avtoext=extfactors.ah

    elif (input.kmag > -99.):
        map = input.kmag
        if input.kmage>-99: mape = input.kmage
        else: mape = 0.02
        avtoext=extfactors.ak

    elif (input.gamag > -99.):
        map = input.gamag
        if input.gamage>-99: mape = input.gamage
        else: mape = 0.02
        avtoext=extfactors.aga

    elif (input.bpmag > -99.):
        map = input.bpmag
        if input.bpmage>-99: mape = input.bpmage
        else: mape = 0.02
        avtoext=extfactors.abp

    elif (input.rpmag > -99.):
        map = input.rpmag
        if input.rpmage>-99: mape = input.rpmage
        else: mape = 0.02
        avtoext=extfactors.arp
    else:
        return out

    # get l,b from RA,DEC
    equ = ephem.Equatorial(input.ra*np.pi/180., input.dec*np.pi/180., epoch=ephem.J2000)
    gal = ephem.Galactic(equ)
    lon_deg=gal.lon*180./np.pi
    lat_deg=gal.lat*180./np.pi

    # Collect given BJ distances
    distanceSamples = np.array([input.BJdisem,input.BJdis,input.BJdisep]) # in pc

    # Get extinction at a given l,b and BJ distance
    ebvs = dustmodel(lon_deg,lat_deg,distanceSamples/1000) # dist in kpc
    avs = avtoext*ebvs

    absmag   = -5.*np.log10(distanceSamples[1]) -avs[1] +map +5.
    absmagep = -5.*np.log10(distanceSamples[0]) -avs[0] +(map+mape) +5.
    absmagem = -5.*np.log10(distanceSamples[2]) -avs[2] +(map-mape) +5.

    appmag   = map-avs[1]
    appmagep = (map+mape)-avs[0]
    appmagem = (map-mape)-avs[2]

    # Update results
    out.avs   = avs[1]
    out.avsep = avs[2]-out.avs
    out.avsem = out.avs-avs[0]

    out.absmag   = absmag
    out.absmagep = absmagep-out.absmag
    out.absmagem = out.absmag-absmagem

    out.appmag   = appmag
    out.appmagep = appmagep-out.appmag
    out.appmagem = out.appmag-appmagem

    out.dis   = distanceSamples[1]
    out.disep = distanceSamples[2]-out.dis
    out.disem = out.dis-distanceSamples[0]

    out.lon_deg = lon_deg
    out.lat_deg = lat_deg

    del ebvs # To clear dustmodel from memory

    return out

def getstat(indat):
    p16, med, p84 = np.percentile(indat,[16,50,84])
    emed1 = med - p16
    emed2 = p84 - med
    return med, emed2, emed1

class obsdata():
    def __init__(self):

        self.ra = -99.
        self.dec = -99.

        self.plx = -99.
        self.plxe = -99.

        self.bmag = -99.
        self.bmage = -99.
        self.vmag = -99.
        self.vmage = -99.
        self.rmag = -99.
        self.rmage = -99.
        self.imag = -99.
        self.image = -99.

        self.btmag = -99.
        self.btmage = -99.
        self.vtmag = -99.
        self.vtmage = -99.

        self.gsmag = -99.
        self.gsmage = -99.
        self.rsmag = -99.
        self.rsmage = -99.
        self.ismag = -99.
        self.ismage = -99.
        self.zsmag = -99.
        self.zsmage = -99.
        self.jmag = -99.
        self.jmage = -99.
        self.hmag = -99.
        self.hmage = -99.
        self.kmag = -99.
        self.kmage = -99.

        self.gamag = -99.
        self.gamage = -99.
        self.bpmag = -99.
        self.bpmage = -99.
        self.rpmag = -99.
        self.rpmage = -99.

        self.BJdis   = -99.
        self.BJdisep = -99.
        self.BJdisem = -99.

    def addbvri(self,value,sigma):
        self.bmag = value[0]
        self.bmage = sigma[0]
        self.vmag = value[1]
        self.vmage = sigma[1]
        self.rmag = value[2]
        self.rmage = sigma[2]
        self.imag = value[3]
        self.image = sigma[3]

    def addbvt(self,value,sigma):
        self.btmag = value[0]
        self.btmage = sigma[0]
        self.vtmag = value[1]
        self.vtmage = sigma[1]

    def addgaia(self,value1,value2):
        self.gamag = value1
        self.gamage = value2

    def addBP(self,value1,value2):
        self.bpmag = value1
        self.bpmage = value2

    def addRP(self,value1,value2):
        self.rpmag = value1
        self.rpmage = value2

    def addgriz(self,value,sigma):
        self.gsmag = value[0]
        self.gsmage = sigma[0]
        self.rsmag = value[1]
        self.rsmage = sigma[1]
        self.ismag = value[2]
        self.ismage = sigma[2]
        self.zsmag = value[3]
        self.zsmage = sigma[3]

    def addjhk(self,value,sigma):
        self.jmag = value[0]
        self.jmage = sigma[0]
        self.hmag = value[1]
        self.hmage = sigma[1]
        self.kmag = value[2]
        self.kmage = sigma[2]

    def addcoords(self,value1,value2):
        self.ra = value1
        self.dec = value2

    def addplx(self,value,sigma):
        self.plx = value
        self.plxe = sigma

    def addBJdis(self,dis,disep,disem):
        self.BJdis   = dis
        self.BJdisep = disep
        self.BJdisem = disem

class resdata():
    def __init__(self):
        self.avs = -99.
        self.avsep = -99.
        self.avsem = -99.
        self.dis = -99.
        self.dise = -99.
        self.disep = -99.
        self.disem = -99.
        self.plx = -99.
        self.plxe = -99.
        self.plxep = -99.
        self.plxem = -99.
        self.absmag = -99.
        self.absmagep = -99.
        self.absmagem = -99.
        self.appmag = -99.
        self.appmagep = -99.
        self.appmagem = -99.

        self.mabs = -99.
        self.lon_deg = -99.
        self.lat_deg = -99.

# R_lambda values to convert E(B-V) given by dustmaps to extinction in
# a given passband.  The two main caveats with this are: - strictly
# speaking only cardelli is consistent with the BC tables used in the
# MIST grid, but using wrong R_lambda's for the newer Green et
# al. dustmaps is (probably) worse.  - some values were interpolated
# to passbands that aren't included in the Schlafly/Green tables.
# ---- extinction-vector-green19-iso.txt
class extinction():
    def __init__(self):
        self.ab=3.868052213359704
        self.av=3.057552170073521

        self.abt=3.995764979598262
        self.avt=3.186976550023362

        self.aus=4.58899934302815
        self.ags=3.6133715121629546
        self.ars=2.6198257176283373
        self.ais=1.9871958324892112
        self.azs=1.4617280489796902

        self.aj=0.7927
        self.ah=0.46900000000000003
        self.ak=0.3026

        self.aga=2.8134632694403003
        self.abp=3.360851699241455
        self.arp=1.942713701900902
