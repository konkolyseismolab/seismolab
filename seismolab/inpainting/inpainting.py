import numpy as np
import subprocess
import os
from astropy.io import fits
import platform
from scipy.ndimage import median_filter
import tempfile

__all__ = ['kinpainting']

# NAME:
#       REGULAR_GRID
#
# PURPOSE:
#       Puts the points of data into a regular grid with
#       dt=median(real dt)
#       Points where there are no points are set to zero
#
# CALLING:
#       regular_grid,data,out,dt=dt,out_ng=out_ng
#
# INPUTS:
#       data --- irregularly sampled 1d signal
#
# OUTPUT:
#       out_reg ---  regularly sampled signal with gaps
#
# KEYWORD:
#       dt --- regular timing value
#       maxi --- to just process the beginning of the time series
#       aver --- resize with larger bien
#       out_irreg --- out no gap: existing points are in the original
#                  timing, gaps are filled with regular grid
#                       IT DOES NOT WORK WHEN AVERAGING!!
#
# HISTORY:
#	Written: R. Garcia 2013
#-
#---------------------------------------------------------
def proper_round_float(val):
    if (float(val) % 1) >= 0.5:
        x = int(np.ceil(val))
    else:
        x = round(val)
    return x

def proper_round(val):
    if isinstance(val, np.ndarray):
        return np.array([proper_round_float(v) for v in val],dtype=int)
    else:
        return proper_round_float(val)

def regular_grid(data,dt=None,maxi=None,aver=None):

    t = data[0,:]-data[0,0]
    gd = np.where(np.isfinite(data[1,:]))[0]

    if dt is None:
        dt = np.median(np.diff(t))
    if maxi is None:
        fin = np.max(t)
    else:
        fin = maxi

    np_int = proper_round((fin+dt)/dt)
    out_reg = np.zeros((2,np_int))

    out_reg[0,:] = np.arange(np_int)*dt
    if aver is None:
        ndx = proper_round(t[gd]/dt)
        out_reg[1,ndx] = data[1,gd]
        out_irreg = out_reg.copy()
        out_irreg[0,:] = out_irreg[0,:]+data[0,0]
        out_irreg[0,ndx] = data[0,gd]
    else:
        gd = np.where(np.bitwise_and(np.isfinite(data[1,:]) , data[1,:] != 0.0))[0]
        ndx = proper_round(t[gd]/dt)
        for i in range(np_int):
            a = np.where(i == ndx)[0]
            na = len(a)
            if na > 0:
                out_reg[1,i] = np.sum(data[1,gd[a]])/na

    out_reg[0,:] = out_reg[0,:] + data[0,0]

    return out_reg, out_irreg

# NAME:
#       SIZE_GAP
#
# PURPOSE:
#       Search the size of the larger gap in a 1d signal
#
# CALLING:
#       size_gap, in_data, max
#
# INPUTS:
#       in_data --- incomplete 1d data
#
# OUTPUT:
#       size max of the gaps
#
# HISTORY:
#	Written: S. Pires Nov. 2012
#-
#---------------------------------------------------------
def size_gap(in_data):
    index = np.where(in_data != 0)[0]
    nd = len(index)

    gap_size = index[1:nd-1] - index[0:nd-2]

    return np.max(gap_size)

#+
# NAME:
#       NEW_WINDOW
#
# PURPOSE:
#       Inpaint only the gaps with a length smaller than max_sz_gap
#
# CALLING:
#       new_window,data,out,out_ng,max_sz_gap
#
# INPUTS:
#       data --- irregularly sampled time series
#
# OUTPUT:
#       inp_reg --- inpainted regular sampled data
#
# KEYWORD:
#       inp_irreg --- contains the original data with its original timing
#                  + gap filled with regular timing
#       max_sz_gap --- maximal size of gaps to be filled in out_ng
#
# HISTORY:
#	Written: R. Garcia 2013
#-
#---------------------------------------------------------
def new_window(data,inp_reg,max_sz_gap,inp_irreg=None,wdw_inp=None):

    time_org = data[0,:]
    dif = np.diff(time_org)
    indx = np.where(dif > max_sz_gap)[0]       # Indexes of the initial times of the big gaps
    nn = len(indx)
    if nn > 0:
        if inp_irreg is not None:
            for i in range(nn):
                # We select the big gaps
                a = np.where(np.bitwise_and(inp_irreg[0,:] > time_org[indx[i]] , inp_irreg[0,:] < time_org[indx[i]+1]))[0]
                inp_irreg[1,a] = np.nan    # We put NaNs in the big gaps
            a = np.where(np.isfinite(inp_irreg[1,:]))[0]
            inp_irreg = inp_irreg[:,a]  # We remove the NaNs

        for i in range(nn):
            # We select the big gaps
            a = np.where(np.bitwise_and(inp_reg[0,:] > time_org[indx[i]] , inp_reg[0,:] < time_org[indx[i]+1]))[0]
            inp_reg[1,a] = 0.0     # We put the big gaps to zero
            wdw_inp[a] = 0

    return inp_reg, wdw_inp

# NAME:
#       RUN_MCA1D
#
# PURPOSE:
#       process the inpainting of an incomplete 1d signal
#       using Multiscale Discrete Cosine Transform
#
# CALLING:
#       run_mca1d, in_data, out_data, /expo
#
# INPUTS:
#       in_data --- incomplete 1d signal
#
# OUTPUT:
#       out_data --- inpainted data
#
# KEYWORD:
#      iter --- number of iterations
#      plot --- plot the result of inpainting
#      verbose --- verbose mode
#      write --- write the original & inpainted signal
#      sigmabounded --- sigma bounded per scale
#      nscale --- To fix the number of scales in the MDCT
#      dct --- if the gaps are in the same dynamic use the DCT
#
# HISTORY:
#	Written: S. Pires Nov. 2012
#-
#---------------------------------------------------------
def run_mca1d(in_data,
            niters = 100,
            verbose = False,
            sigmabounded = 0,
            nscale = None,
            dct = None,
            setenv='./',
            tempdir='./'):
            #, noise=noise

    #Signal should be around zero to make inapinting work
    ind = np.where(in_data != 0.0)[0]
    nind = np.where(in_data == 0.0)[0]
    m = np.mean(in_data[ind])

    if verbose: print('mean = ', m)

    in_data2 = in_data - m
    in_data2[nind] = 0

    mgap = size_gap(in_data2)

    if nscale is None:
        nscale = np.floor(np.log(mgap)/np.log(2))+1.
    y = np.floor(np.log(mgap*8.)/np.log(2))+1.
    DCTBlockSize = int(np.floor(2.**y))

    name1 = ''
    if sigmabounded is None:
        name1 = ' -d'
    name2 = ''


    if dct is not None:
        y = np.floor(np.log(len(in_data))/np.log(2))
        DCTBlockSize = int(np.floor(2.**y))
        if verbose:
            print('**** DCTBlockSize = ', DCTBlockSize)
            print('**** ITER = ', niters)
            out_data2 = cb_mca1d(in_data2, opt='-H -s0 -t3 -O -B '+str(DCTBlockSize)+' -L -v -i '+str(niters)+' -D2 '+name1+name2,
                                 setenv=setenv, tempdir=tempdir)
        else:
            out_data2 = cb_mca1d(in_data2, opt='-H -s0 -t3 -O -B '+str(DCTBlockSize)+' -L -i '+str(niters)+' -D2 '+name1+name2,
                                 setenv=setenv, tempdir=tempdir)
    else:
        if verbose:
            print('**** NSCALE = ', nscale)
            print('**** DCTBlockSize = ', DCTBlockSize)
            print('**** ITER = ', niters)
            #com = '-H -s0 -n'+str(nscale)+' -t4 -O -B'+str(DCTBlockSize)+' -L -v -i'+str(niters)+' -D2'+name1+name2
            #print(com)
            out_data2 = cb_mca1d(in_data2, opt='-H -s0 -n '+str(nscale)+' -t4 -O -B '+str(DCTBlockSize)+' -L -v -i '+str(niters)+' -D2 '+name1+name2,
                                 setenv=setenv, tempdir=tempdir)
        else:
            #com = '-H -s0 -n'+str(nscale)+' -t4 -O -B'+str(DCTBlockSize)+' -L -i'+str(niters)+' -D2 '+name1+name2
            #print(com)
            out_data2 = cb_mca1d(in_data2, opt='-H -s0 -n '+str(nscale)+' -t4 -O -B '+str(DCTBlockSize)+' -L -i '+str(niters)+' -D2 '+name1+name2,
                                 setenv=setenv, tempdir=tempdir)

    out_data = out_data2 + m

    return out_data

# NAME:
#       CB_MCA1D
#
# PURPOSE:
#	      Inpainting by decomposition of an image on multiple bases
#
# CALLING:
#
#      CB_MCA1D, Imag, Struct, OPT=Opt
#
#
# INPUTS:
#     Imag -- 2D IDL array: image we want to decompose
#
# OUTPUTS:
#     Result -- Image inpainted
#     Struct -- Decomosition of Imag in each of the base
#
# KEYWORDS:
#
#      Opt -- string: string which contains the differents options. Options are:
#   where options =
#         [-t TransformSelection]
#              1: A trous algorithm
#              2: bi-orthogonal WT with 7/9 filters
#              3: Ridgelet transform
#              4: Curvelet transform 02
#              5: Local Discrete Cosinus Transform
#              6: Wavelet Packet basis
#              7: Curvelet transform 05 (Pyramidal Construction)
#              8: Curvelet transform 04 (Fast Transform)
#
#         [-d ]
#             SigmaBounded Sigma inside the gaps is equal to sigma
#             outside the gaps
#             default is no.
#
#         [-n number_of_scales]
#             Number of scales used in the WT, the a trous, the PMT & the curvelet transform.
#             default is 5.
#
#         [-b BlockSize]
#             Block Size in the ridgelet transform.
#             Default is image size.
#
#         [-i NbrIter]
#             Number of iteration. Default is 30.
#
#         [-B DCT_BlockSize]
#             Local DCT block size.
#             By default, a global DCT is used.
#
#         [-S FirstThresholdLevel]
#            First thresholding value.
#            Default is derived from the data.
#
#         [-s LastThresholdLevel]
#            Last thresholding value..
#            default is 3.000000.
#
#         [-N]
#             Minimize the L1 norm.
#             Default is L0 norm.
#
#         [-L]
#             Replacing the linear descent by a non linear descent.
#             (Should be used when one components is much larger than another one).
#
#         [-l]
#             Remove last scale. Default is no.
#
#         [-g sigma]
#             sigma = noise standard deviation.
#             Default is 0.5 (quantification noise).
#             if sigma is set to 0, noise standard deviation
#             is automatically estimated.
#
#         [-O]
#             Supress the block overlapping. Default is no.
#
#         [-P]
#             Supress the positivity constraint. Default is no.
#
#         [-G RegulVal[,NbrScale]]
#             Total Variation regularization term. Default is 0.
#             NbrScale = number of scales used in Haar TV regularizarion.
#                        default is 2.
#
#         [-H]
#             Data contained masked area (must have a zero value). Default is no.
#
#         [-I]
#             Interpolate the data (super-resolution). Default is no.
#
#         [-z]
#             Use virtual memory.
#                default limit size: 4
#                default directory: .
#
#         [-Z VMSize:VMDIR]
#             Use virtual memory.
#                VMSize = limit size (megabytes)
#                VMDIR = directory name
#
#         [-v]
#             Verbose. Default is no.
#
#
# EXTERNAL CALLS:
#       cb_mca (C++ program)
#
# EXAMPLE:
#
#       Decomposition of an image I into 2 bases (Curvelet & DCT with block size of 16).
# 			The result is stored in the strcture Struct
#               CB_MCA, I, Struct, OPT='-t4 -t5 -B16'
#
# HISTORY:
#	Written: Sandrine Pires 2006.
def cb_mca1d(Imag, opt=' ',setenv='./',tempdir='./'):

    if Imag.ndim != 1:
        IndexError('Flux must be 1 dimensional!')

    noise = get_noise(Imag)

    filename = 'tmpResult'
    filename = os.path.join(tempdir,filename)

    NameImag = 'tmp' + str(np.random.randint(1e5)) + '.fits'
    NameImag = os.path.join(tempdir,NameImag)

    hdu = fits.PrimaryHDU(Imag)
    hdul = fits.HDUList([hdu])
    hdul.writeto(NameImag,overwrite=True)

    com = [setenv] + opt.split() + ["-g"+str(noise) , NameImag , filename]

    #print(com)
    subprocess.run(com)

    h = fits.open(filename+'.fits')
    Result = h[0].data

    os.remove(NameImag)
    os.remove(filename+'_mcos.fits')
    os.remove(filename+'_resi.fits')

    return Result

# NAME:
#       INIT_VAR
#
# PURPOSE:
#       Initialize the C++ environment variable.
#
# CALLING:
#      init_var
#
# HISTORY:
#	Written: Sandrine Pires Sept. 2013
#-
#---------------------------------------------------------
def init_var(bool=False):
    PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

    if platform.system() == 'Darwin':
        arch = platform.architecture()[0]
        if arch == '64bit':
            setenv = os.path.join(PACKAGEDIR,'Exec_C++/cb_mca1d_MacOSX_Intel_64')
        else:
            setenv = os.path.join(PACKAGEDIR,'Exec_C++/cb_mca1d_MacOSX_Intel_32')

    elif platform.system() == 'Linux':
        arch = platform.architecture()[0]
        if arch == '64bit':
            setenv = os.path.join(PACKAGEDIR,'Exec_C++/cb_mca1d_Linux_64')
        else:
            setenv = os.path.join(PACKAGEDIR,'Exec_C++/cb_mca1d_Linux_32')

    else:
        NotImplementedError('This platform is not supported!')

    return setenv


# NAME:
#     GET_NOISE
#
# PURPOSE:
#    Find the standard deviation of a white gaussian noise in the data.
#
# CALLING SEQUENCE:
#   output=GET_NOISE(Data)
#
# INPUTS:
#   Data -- IDL array
#
# OPTIONAL INPUT PARAMETERS:
#   none
#
# KEYED INPUTS:
#   Niter --scalar: number of iterations for k-sigma clipping
#                   default is 3.
#
# OUTPUTS:
#    output
#
# MODIFICATION HISTORY:
#    17-Jan-1996 JL Starck written with template_gen
#------------------------------------------------------------
def get_noise(Data, Niter=3):

    sigma = sigma_clip(Data - median_filter(Data,3), Niter=Niter) / 0.893421

    return sigma


# NAME:
#       sigma_clip
#
# PURPOSE:
#       return the sigma obtained by k-sigma. Default sigma_clip value is 3.
#       if mean is set, the mean (taking into account outsiders) is returned.
#
# CALLING SEQUENCE:
#   output=sigma_clip(Data, sigma_clip=sigma_clip, mean=mean)
#
# INPUTS:
#   Data -- IDL array: data
#
# OPTIONAL INPUT PARAMETERS:
#   none
#
# KEYED INPUTS:
#   sigma_clip -- float : sigma_clip value
#
# KEYED OUTPUTS:
#   mean -- float : mean value
#
# OUTPUTS:
#    output
#
# EXAMPLE:
#    output_sigma = sigma_clip(Image, sigma_clip=2.5)
#
# MODIFICATION HISTORY:
#    25-Jan-1995 JL Starck written with template_gen
#-
def sigma_clip(Data, sigma=3, Niter=3):

    m = np.sum(Data)/ len(Data)
    Sig = np.std(Data)

    index = np.where((np.abs(Data-m) < sigma*Sig))[0]

    for _ in range(Niter-1):
        m = np.sum(Data[index]) / len(Data[index])
        Sig = np.std(Data[index])
        index = np.where((np.abs(Data-m) < sigma*Sig))[0]


    return Sig

# NAME:
#       INPAINT_KEPLER
#
# PURPOSE:
#       gaps filled with inpainting
#
# CALLING:
#       inpaint_kepler,in,out,out_ng=out_ng,max_sz_gap=max_sz_gap,dt=dt
#
# INPUTS:
#       data --- irregularly sampled time series
#
# OUTPUT:
#       inp_reg --- inpainted regular sampled data
#
# KEYWORD:
#       dt --- regular timing
#       out_irreg --- contains the original data with its original timing
#                  + gap filled with regular timing => *** Introduce WINDOW EFFECTS ****
#       max_sz_gap --- maximal size of gaps to be filled in inp_irreg
#       wdw_inp --- window with the interpolated points (1-original # 2-interpolated # 0 - unchanged)
#
# HISTORY:
#	Written: Rafael A. Garcia Aug. 2013
#       Modified by Sandrine Pires Oct. 2013
#-
#---------------------------------------------------------
def inpaint_kepler(data,
        max_sz_gap=None,
        dt=None,
        verbose=False,
        sigmabounded=0,
        niters=100,
        setenv='./'):

        # out_reg_gap=out_reg_gap

    tempdir = tempfile.gettempdir()

    data_reg,out_irreg = regular_grid(data,dt=dt)

    t = data_reg[0,:]
    flux = data_reg[1,:]
    bad = np.where(flux == 0.0)[0]

    #run_mca1d, flux,iflux,verbose=verbose,sigmabounded=sigmabounded
    iflux = run_mca1d(flux,verbose=verbose,sigmabounded=sigmabounded,setenv=setenv,tempdir=tempdir,niters=niters)

    npoints = len(iflux)
    inp_reg = np.zeros((2,npoints))
    inp_reg[0,:] = t
    inp_reg[1,:] = flux
    inp_reg[1,bad] = iflux[bad]
    wdw_inp = np.ones_like(flux)
    wdw_inp[bad] = 2
    if max_sz_gap is None:
        out_irreg[1,bad] = iflux[bad]
    else:
        inp_reg, wdw_inp = new_window(data,inp_reg,max_sz_gap,wdw_inp=wdw_inp) #out_irreg  # Don't use in KEPLER data

    os.remove(os.path.join(tempdir,'tmpResult.fits'))

    return inp_reg, out_irreg


#===========================================
#       Inpainting of kepler data
#===========================================
def kinpainting(time,brightness,max_sz_gap=None,dt=None,niters=100,verbose=False):
    """
    Inpainting method to fill gaps in time series data.
    Mathematical details are in Pires+, 2009, MNRAS, 395, 1265
    and Pires+, 2015, A&A, 574, 18.

    Parameters
    ----------
    time : array-like
        Time values of the light curve.
    brightness : array-like
        Brightness values of the light curve.
    max_sz_gap : float, default: None
        Maximal size of gaps to be filled.
    dt : float, default: None
        Regular timing used to create a uniform time grid.
        By default, it is the median sampling time.
    niters : int, default: 100
        Number of iterations.
    verbose : boolean, default: False
        Verbose output.

    Returns
    -------
    inpainted : 2D array
        The filled light curve sampled in a regular grid.
    inpainted_irreg : 2D array
        The filled light curve sampled in an irregular grid.
        The sampling is similar to the input times series'.
    """

    data=np.zeros((2,len(time)))
    data[0,:]=time
    data[1,:]=brightness

    #===========================================
    #         Initialize C++ Variable
    #===========================================

    setenv = init_var()
    #If the initialization is not working alone, use the following command
    #setenv = init_var(bool=True)

    inpainted, inpainted_irreg = inpaint_kepler(data,
                                                dt=dt,
                                                setenv=setenv,
                                                niters=niters,
                                                max_sz_gap=max_sz_gap,
                                                verbose=verbose)

    #np.savetxt(path+outfile+'_inpainted_regular.dat',out.T,fmt='%.10f')
    #np.savetxt(path+outfile+'_inpainted_irregular.dat',out_irreg.T,fmt='%.10f')

    return inpainted.T, inpainted_irreg.T
