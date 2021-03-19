import multiprocessing

import numpy as np
from numpy import array,arange,linspace
from scipy import constants,special

from . import tools
from . import convert

def calculate_spectrum(
        x,
        line_function,
        *line_args,
        ncpus=1,
        multiprocess_divide='lines', # 'lines' or 'x'
        **line_kwargs,
):
    """Compute a spectrum of lines using line_function which is applied to
    positional arguments line_args and with common kwargs.
    line_function must add to y in place with a yin argument."""
    ncpus = min(ncpus,multiprocessing.cpu_count())
    ## run as a single process
    if ncpus == 1:
        y = np.zeros(x.shape,dtype=float)
        for args in zip(*line_args):
            line_function(x,*args,**line_kwargs,yin=y)
        return y
    ## multiprocessing version  -- divide lines between processes
    elif multiprocess_divide == 'lines':
        y = np.zeros(x.shape,dtype=float)
        ## get indices to divide lines
        nlines = len(line_args[0])
        js = [int(t) for t in linspace(0,nlines,ncpus)]
        with multiprocessing.Pool(ncpus) as p:
            y = []
            for jbeg,jend in zip(js[:-1],js[1:]):
                ## start async processes
                y.append(
                    p.apply_async(
                        calculate_spectrum,
                        args=(x,line_function,*[t[jbeg:jend] for t in line_args]),
                        kwds=line_kwargs,))
            ## run proceses, tidy keyboard interrupt, sum to give full spectrum
            try:
                p.close()
                p.join()
            except KeyboardInterrupt as err:
                p.terminate()
                p.join()
                raise err
            y = np.sum([t.get() for t in y],axis=0)
        return y
    ## multiprocessing version  -- divide x-range between processes
    elif multiprocess_divide == 'x':
        ## get indices to divide lines
        js = [int(t) for t in linspace(0,len(x),ncpus)]
        with multiprocessing.Pool(ncpus) as p:
            y = []
            for jbeg,jend in zip(js[:-1],js[1:]):
                ## start async processes
                y.append(
                    p.apply_async(
                        calculate_spectrum,
                        args=(x[jbeg:jend],line_function,*line_args),
                        kwds=line_kwargs,))
            ## run proceses, tidy keyboard interrupt, sum to give full spectrum
            try:
                p.close()
                p.join()
            except KeyboardInterrupt as err:
                p.terminate()
                p.join()
                raise err
            y = np.concatenate([t.get() for t in y])
        return y
    else:
        raise Exception('Unknown {multiprocess_divide=}')

    # def sinc(x,x0=0,S=1,Γ=1):
    # """Lorentzian profile.""" 
    # return(S*np.sinc((x-x0)/Γ*1.2)*1.2/Γ)

def lorentzian(x,x0=0,S=1,Γ=1,nfwhm=None,yin=None): 
    """Lorentzian profile."""
    if nfwhm is None:
        ## whole domain
        if yin is None:
            return S*Γ/2./constants.pi/((x-x0)**2+Γ**2/4.)
        else:
            yin += S*Γ/2./constants.pi/((x-x0)**2+Γ**2/4.)
            return yin
    else:
        ## partial domain inside nfwhm cutoff
        ibeg,iend = np.searchsorted(x,[x0-nfwhm*Γ,x0+nfwhm*Γ])
        if yin is None:
            y = np.zeros(x.shape)
        else:
            y = yin    
        lorentzian(x[ibeg:iend],x0,S,Γ,nfwhm=None,yin=y[ibeg:iend])
        return y

# def stepwise_lorentzian(x,x0=0,S=1,Γ=1,nfwhm=np.inf):
    # """Lorentzian profile, preserves integral even if under resolved.."""
    # if np.isinf(nfwhm):
        # retval = np.zeros(x.shape,dtype=float)
        # _fortran_tools.stepwise_lorentzian(x0,S,Γ,x,retval)
        # return(retval)
    # else:
        # retval = np.zeros(x.shape)
        # t = nfwhm*Γ
        # ibeg,iend = np.searchsorted(x,[x0-t,x0+t])
        # ibeg,iend = max(ibeg-1,0),min(iend+1,len(x))
        # if iend-ibeg>1:
            # retval[ibeg:iend] = stepwise_lorentzian(x[ibeg:iend],x0,S,Γ)
        # return(retval)

def gaussian(x,x0=0.,S=1,Γ=1.,nfwhm=None,yin=None):
    """A gaussian with area normalised to one. Γ is FWHM. If y is
    provided add to this in place."""
    if nfwhm is None:
        if yin is None:
            return S/Γ*np.sqrt(4*np.log(2)/constants.pi)*np.exp(-(x-x0)**2*4*np.log(2)/Γ**2)
        else:
            yin +=  S/Γ*np.sqrt(4*np.log(2)/constants.pi)*np.exp(-(x-x0)**2*4*np.log(2)/Γ**2)
            return yin
    else:
        ## partial domain inside nfwhm cutoff
        ibeg,iend = np.searchsorted(x,[x0-nfwhm*Γ,x0+nfwhm*Γ])
        if yin is None:
            y = np.zeros(x.shape)
        else:
            y = yin    
        gaussian(x[ibeg:iend],x0,S,Γ,nfwhm=None,yin=y[ibeg:iend])
        return y

def voigt(x,
          x0=0,                 # centre
          S=1,                  # integrated value
          ΓL=1,ΓG=1,            # fwhm
          nfwhmL=None,          # maximum Lorentzian widths to include
          nfwhmG=None,          # maximum Gaussian widths to include -- pure Lorentzian outside this
          yin = None            # set to add this line in place to an existing specturm of size matching x
):
    "Voigt profile -- x must be sorted for correct and efficient noninfinity nfwhmG and nfwhmL."
    from scipy import special
    b = 0.8325546111576 # np.sqrt(np.log(2))
    norm = 1.0644670194312262*ΓG # sqrt(2*pi/8/log(2))
    if ΓG == 0:
        ## pure Lorentzian
        return lorentzian(x,x0,S,ΓL,nfwhm=nfwhmL,yin=yin)
    elif ΓL == 0:
        ## pure Gaussian
        return gaussian(x,x0,S,ΓG,yin=yin)
    elif nfwhmG is None:
        ## full calculation
        if yin is None:
            return S*special.wofz((2.*(x-x0)+1.j*ΓL)*b/ΓG).real/norm
        else:
            yin += S*special.wofz((2.*(x-x0)+1.j*ΓL)*b/ΓG).real/norm
            return yin
    elif nfwhmL is None and nfwhmG is not None:
        ## Lorentzian wings
        y = lorentzian(x,x0,S,ΓL,yin=yin)
        i0,i1 = np.searchsorted(x,[x0-nfwhmG*ΓG,x0+nfwhmG*ΓG])
        y[i0:i1] = voigt(x[i0:i1],x0,S,ΓL,ΓG)
    elif nfwhmL is not None and nfwhmG is not None:
        ## Lorentzian wings and cutoff
        i0,i1 = np.searchsorted(x,[x0-(nfwhmG*ΓG+nfwhmL*ΓL),x0+(nfwhmG*ΓG+nfwhmL*ΓL)])
        j0,j1 = np.searchsorted(x,[x0-(nfwhmG*ΓG),x0+(nfwhmG*ΓG)])
        if yin is None:
            y = np.zeros(x.shape,dtype=float)
            y[i0:j0] = lorentzian(x[i0:j0],x0,S,ΓL)
            y[j0:j1] = voigt(x[j0:j1],x0,S,ΓL,ΓG)
            y[j1:i1] = lorentzian(x[j1:i1],x0,S,ΓL)
        else:
            y = yin
            y[i0:j0] += lorentzian(x[i0:j0],x0,S,ΓL)
            y[j0:j1] += voigt(x[j0:j1],x0,S,ΓL,ΓG)
            y[j1:i1] += lorentzian(x[j1:i1],x0,S,ΓL)
    else:
        raise Exception(f'Not implemented: nfwhmL={repr(nfwhmL)} and nfwhmG={repr(nfwhmG)}')
    return y

def rautian(
        x,
        x0=0,                 # centre
        S=1,                  # integrated value
        ΓL=0,
        ΓG=0,
        νvc=0,
        # nfwhmL=None,          # maximum Lorentzian widths to include
        # nfwhmG=None,          # maximum Gaussian widths to include -- pure Lorentzian outside this
        # yin = None            # set to add this line in place to an existing specturm of size matching x
):
    """Rautian profile. E.g,. schreier2020, 10.1016/j.jqsrt.2020.107385"""
    # if ζ == 0:
        # return voigt(x,x0,S,ΓL,ΓG)
    from scipy import special
    b = 0.8325546111576 # np.sqrt(np.log(2))
    γG,γL = ΓG/2,ΓL/2   # FWHM to HWHM
    if γG == 0:
        ## hack to prevent divide by zero
        γG = 1e-10
    x = b*(x-x0)/γG
    y = b*γL/γG
    z = x + 1j*y
    ζ = b*νvc/γG
    retval = (special.wofz(z)/(1-np.sqrt(constants.pi)*ζ*special.wofz(z))).real
    ## normalise and scale
    retval = retval/(1.0644670194312262*ΓG) * S
    return retval

def hartmann_tran(
        x,        # frequency scale (cm-1)
        x0,       # centre unshifted frequency (cm-1)
        S,        # line strength
        m,        # mass of the abosrbing molecule (amu)
        T,        # temperature (K)
        νVC,      # frequncy of velocity-changing collisions
        η,        # correlation parameter
        Γ0,       # speed-averaged halfwidth
        Γ2,       # quadratic halfwidth
        Δ0,       # speed-averaged shift
        Δ2,       # quadratic shift
        ## Y,        # First-order (Rosenkranz) line coupling coefficient
        yin=None, # add line in place to this array
        nfwhmL=None,            # how many widths of the approximate Lorentzian component to include before cutting off line
        nfwhmG=None,            # how many widths of approximate Gaussian component to include before switching to a pure Lorentzian
        # method='python',        # uses scipy functiosn
        method='tran2014',        # uses tran2014/trans2015 fortran subroutine
):
    """The Hartmann-Tran line profile, based on ngo2013 doi:
    10.1016/j.jqsrt.2013.05.034."""
    if method == 'python':
        π = constants.pi
        w = special.wofz
        kB = constants.Boltzmann
        sqrt = np.sqrt
        c = constants.c
        ## full calculation
        if nfwhmG is None:
            ## ## using equation in tran2013
            ## vtilde = sqrt(2*kB*T/convert.units(m,'amu','kg')) # test tran2013
            ## C0 = Γ0-1j*Δ0                # Eq. (3) tran2013
            ## C2 = Γ2-1j*Δ2                # Eq. (3) tran2013
            ## C0t = νVC+(1-η)*(C0-3*C2/2)  # text and Eq. (3) tran2013
            ## C2t = (1-η)*C2               # text and Eq. (3) tran2013
            ## X = (1j*(x-x0)+C0t)/C2t      # Eq. (8) tran2013
            ## Y = (x0*vtilde/(2*c*C2t))**2 # Eq. (8) tran2013
            ## Z1 = sqrt(X+Y) - sqrt(Y)     # Eq. (7) tran2013
            ## Z2 = sqrt(X+Y) + sqrt(Y)     # Eq. (7) tran2013
            ## A = sqrt(π)*c/(x0*vtilde)*(w(1j*Z1)-w(1j*Z2)) # Eq. (5) tran2013
            ## B = vtilde**2/C2t*(-1+sqrt(π)/(2*sqrt(Y))*(1-Z1**2)*w(1j*Z1)-sqrt(π)/(2*sqrt(Y))*(1-Z2**2)*w(1j*Z2)) # Eq. (5) tran2013
            ## y = S*1/π*np.real(A/(1-(νVC-η*(C0-3*C2/2))*A+(η*C2/vtilde**2)*B)) # Eq. (4) tran2013
            ##
            ## using equation in tran2013 with correction of tran2014
            vtilde = sqrt(2*kB*T/convert.units(m,'amu','kg')) # test tran2013
            C0 = Γ0+1j*Δ0                # Eq. (3) tran2013
            C2 = Γ2+1j*Δ2                # Eq. (3) tran2013
            C0t = νVC+(1-η)*(C0-3*C2/2)  # text and Eq. (3) tran2013
            C2t = (1-η)*C2               # text and Eq. (3) tran2013
            X = (-1j*(x-x0)+C0t)/C2t      # Eq. (8) tran2013
            Y = (x0*vtilde/(2*c*C2t))**2 # Eq. (8) tran2013
            Z1 = sqrt(X+Y) - sqrt(Y)     # Eq. (7) tran2013
            Z2 = sqrt(X+Y) + sqrt(Y)     # Eq. (7) tran2013
            A = sqrt(π)*c/(x0*vtilde)*(w(1j*Z1)-w(1j*Z2)) # Eq. (5) tran2013
            B = vtilde**2/C2t*(-1+sqrt(π)/(2*sqrt(Y))*(1-Z1**2)*w(1j*Z1)-sqrt(π)/(2*sqrt(Y))*(1-Z2**2)*w(1j*Z2)) # Eq. (5) tran2013
            y = S*1/π*np.real(A/(1-(νVC-η*(C0-3*C2/2))*A+(η*C2/vtilde**2)*B)) # Eq. (4) tran2013
            ##
            if yin is None:
                return y
            else:
                yin += y
                return yin
        ## Hartmann-Tran within nfwhmG and Lorentzian wings outside, if
        ## nfwhmL is not None then cut off completely outide this
        ## Γ0*nfwhmL
        else:
            ΓG = 2.*6.331e-8*np.sqrt(T*32./m)*x0
            i0,i1 = np.searchsorted(x,[x0-nfwhmG*ΓG,x0+nfwhmG*ΓG])
            if yin is None:
                y = np.zeros(x.shape,dtype=float)
            else:
                y = yin    
            lorentzian(x[:i0],x0,S,Γ0*2,nfwhm=nfwhmL,yin=y[:i0])
            hartmann_tran(x[i0:i1],x0,S,m,T,νVC,η,Γ0,Γ2,Δ0,Δ2,nfwhmL=None,nfwhmG=None,yin=y[i0:i1])
            lorentzian(x[i1:],x0,S,Γ0*2,nfwhm=nfwhmL,yin=y[i1:])
            return y
    elif method == 'tran2014':
        ## use f77 implementation in line_profiles_tran2014.f
        from .line_profiles_tran2014 import pcqsdhc
        ΓD = 2.*6.331e-8*np.sqrt(T*32./m)*x0/2 # Doppler width HWHM
        if yin is None:
            yreal,yimag = S*np.array([pcqsdhc(x0,ΓD,Γ0,Γ2,Δ0,Δ2,νVC,η,xi) for xi in x]).transpose()
            y = yreal
            return y
        else:
            yreal,yimag = S*np.array([pcqsdhc(x0,ΓD,Γ0,Γ2,Δ0,Δ2,νVC,η,xi) for xi in x]).transpose()
            yin += yreal
            

            
        
    # ## Lorentzian wings and cutoff
    # elif nfwhmL is not None and nfwhmG is not None:
        # i0,i1 = np.searchsorted(x,[x0-(nfwhmG*ΓG+nfwhmL*ΓL),x0+(nfwhmG*ΓG+nfwhmL*ΓL)])
        # j0,j1 = np.searchsorted(x,[x0-(nfwhmG*ΓG),x0+(nfwhmG*ΓG)])
        # if yin is None:
            # y = np.zeros(x.shape,dtype=float)
            # y[i0:j0] = lorentzian(x[i0:j0],x0,S,ΓL)
            # y[j0:j1] = voigt(x[j0:j1],x0,S,ΓL,ΓG)
            # y[j1:i1] = lorentzian(x[j1:i1],x0,S,ΓL)
        # else:
            # y = yin
            # y[i0:j0] += lorentzian(x[i0:j0],x0,S,ΓL)
            # y[j0:j1] += voigt(x[j0:j1],x0,S,ΓL,ΓG)
            # y[j1:i1] += lorentzian(x[j1:i1],x0,S,ΓL)
    # else:

    else:
        raise Exception(f"Invalid Hartmann-Tran method: {repr(method)}")

    

# def fano(x,x0=0,S=1,Γ=1,q=0,ρ=1):
    # """Calculate a Fano profile."""
    # tx = 2*(x-x0)/Γ
    # return(S*((1.-ρ**2+ρ**2*(q+tx)**2./(1.+tx**2.))/(ρ**2.*(1.+q**2.)*constants.pi*Γ/2.)))

# def rautian_spectrum(
        # x,                      # frequency scale, assumed monotonically increasing (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # ΓL,                     # Lorentzian linewidth (cm-1 FWHM)
        # ΓG,                     # Gaussian linewidth (cm-1 FWHM)
        # νvc,
        # Smin=None,              # do not include lines weaker than this
# ):
    # """Convert some lines into a spectrum."""
    # ## no x, nothing to do
    # if len(x)==0 or len(x0)==0:
        # return np.zeros(x.shape,dtype=float) 
    # ## strip lines that are too weak
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,ΓL,ΓG = x0[i],S[i],ΓL[i],ΓG[i]
    # ## remove lines outside of fwhm calc range
    # # if not nfwhmG is None and not nfwhmL is None:
        # # i = ((x0+nfwhmL*ΓL+nfwhmG*ΓG)<x[0]) | ((x0-nfwhmL*ΓL-nfwhmG*ΓG)>x[-1])
        # # x0,S,ΓL,ΓG = x0[~i],S[~i],ΓL[~i],ΓG[~i]
    # ## test if nothing left
    # if len(x0) == 0:
        # return np.full(x.shape,0.) 
    # ## calc single process
    # y = np.zeros(x.shape,dtype=float)
    # for args in zip(x0,S,ΓL,ΓG,νvc):
        # y += rautian(x,*args)
    # return y

# def voigt_spectrum(
        # x,                      # frequency scale, assumed monotonically increasing (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # ΓL,                     # Lorentzian linewidth (cm-1 FWHM)
        # ΓG,                     # Gaussian linewidth (cm-1 FWHM)
        # nfwhmL=None,              # Number of Lorentzian full-width half-maxima to compute (zero for infinite)
        # nfwhmG=None,             # Number of Gaussian full-width half-maxima to compute
        # Smin=None,              # do not include lines weaker than this
        # use_multiprocessing= True,
        # multiprocessing_max_cpus=999,
# ):
    # """Convert some lines into a spectrum."""
    # ## no x, nothing to do
    # if len(x)==0 or len(x0)==0:
        # return(np.zeros(x.shape,dtype=float))
    # ## strip lines that are too weak
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,ΓL,ΓG = x0[i],S[i],ΓL[i],ΓG[i]
    # ## remove lines outside of fwhm calc range
    # if not nfwhmG is None and not nfwhmL is None:
        # i = ((x0+nfwhmL*ΓL+nfwhmG*ΓG)<x[0]) | ((x0-nfwhmL*ΓL-nfwhmG*ΓG)>x[-1])
        # x0,S,ΓL,ΓG = x0[~i],S[~i],ΓL[~i],ΓG[~i]
    # ## test if nothing left
    # if len(x0)==0:
        # return(np.full(x.shape,0.))
    # ## calc single process
    # if (not use_multiprocessing) or len(x)<10000:
        # y = np.zeros(x.shape,dtype=float)
        # for (x0i,Si,ΓLi,ΓGi) in zip(x0,S,ΓL,ΓG):
            # voigt(x,x0i,Si,ΓLi,ΓGi,nfwhmL,nfwhmG,yin=y)
            # # y += voigt(x,x0i,Si,ΓLi,ΓGi,nfwhmL,nfwhmG)
    # else:
        # if tools.isnumeric(use_multiprocessing):
            # nparallel = int(use_multiprocessing)
        # else:
            # nparallel = min(multiprocessing.cpu_count()-1,multiprocessing_max_cpus)
        # step = max(1,int(len(x0)/(nparallel-1)))
        # nmax = len(x0)
        # # y = [None for t in range(nparallel)]
        # y = []
        # p = multiprocessing.Pool()
        # for iprocess,ibeg in enumerate(range(0,nmax,step)):
            # iend = min(ibeg+step,nmax)
            # def callback(ypartial,iprocess=iprocess):
                # # y[iprocess] = ypartial
                # y.append(ypartial)
            # # p.apply_async(voigt_spectrum,
            # p.apply_async(
                # voigt_spectrum,
                # args=(x,x0[ibeg:iend],S[ibeg:iend],ΓL[ibeg:iend],ΓG[ibeg:iend], nfwhmL,nfwhmG,Smin,False),
                # callback=callback)
        # try:
            # p.close();p.join()
        # except KeyboardInterrupt as err:
            # p.terminate()
            # p.join()
            # raise err
        # y = np.sum(y,axis=0)
    # return y 

# def centroided_spectrum(
        # x,                      # frequency scale (cm-1) -- ORDERED AND REGULAR!
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # Smin=None,              # do not include lines weaker than this
# ):
    # """Convert some lines into a stick spectrum with each linestrength
    # divided between the two nearest neighbour x-points.."""
    # print('UNRELIABLE NEEDS WORK -- only implemented for regular grid')
    # dx = (x[-1]-x[0])/len(x)
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S = x0[i],S[i]
    # x = np.array(x,dtype=float)
    # y = np.zeros(x.shape,dtype=float)
    # ## get indices of output points above and below data
    # i = np.argsort(x0)
    # x0,S = x0[i],S[i]
    # ib = np.searchsorted(x,x0)
    # ia = ib-1
    # ## add left and rigth divided strength to spectrum
    # for iai,ibi,x0i,Si in zip(ia,ib,x0,S):
        # if iai<=0 or ibi>=len(x):
            # ## points outside x domain
            # continue
        # ## weights to above and below points -- COULD BE MODIFIED FOR IRREGULAR GRID
        # ca = (x0i-x[iai])/(x[ibi]-x[iai])
        # y[iai] += ca*Si/dx
        # y[ibi] += (1.-ca)*Si/dx
    # return(y)

# def voigt_spectrum_with_gaussian_doppler_width(
        # x,                      # frequency scale (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # ΓL,                     # Natural (Lorentzian) linewidth (cm-1 FWHM)
        # mass,                   # Mass for Doppler width (scalar) (amu)
        # temperature=None,       # Temperature for Doppler width (scalar) (K)
        # nfwhmL=np.inf,              # Number of Lorentzian full-width half-maxima to compute (zero for infinite)
        # nfwhmG=10.,             # Number of Gaussian full-width half-maxima to compute, if set np.inf then actually maxes out at 100
        # Smin=None,              # do not include lines weaker than this
# ):
    # """Convert some lines into a spectrum."""
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,ΓL = x0[i],S[i],ΓL[i]
    # ## compute Lorentzian spectrum
    # if np.isinf(nfwhmL): nfwhmL = 0
    # if np.isinf(nfwhmG): nfwhmG = 100
    # if len(x0)==0: return(np.zeros(x.shape,dtype=float))
    # ## 'fortran' computes Lorentzians and then convolves with
    # ## a common Gaussian for speed. 'fortran_stepwise' does
    # ## this also but uses a precomputed definite integral for
    # ## the Loretnzian line centre if it is unresolved, to
    # ## preserve the integrated value of the line. The Gaussian
    # ## convolution also explicitly conserves the
    # ## integral. Both these methods require a
    # ## monotonically-increasing regular x grid.
    # yL = np.zeros(x.shape,dtype=float)
    # _fortran_tools.calculate_stepwise_lorentzian_spectrum(x0,S,ΓL,x.astype(float),yL,float(nfwhmL))
    # assert np.isscalar(mass),"Mass must be scalar."
    # assert np.isscalar(temperature),"Temperature must be scalar."
    # yV = np.zeros(x.shape)      # Voigt spectrum
    # _fortran_tools.convolve_with_doppler(mass,temperature,nfwhmG,x,yL,yV)
    # return(yV)

# def gaussian_spectrum(
        # x,                      # frequency scale (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # Γ,                     # Gaussian linewidth (cm-1 FWHM)
        # nfwhm=10.,             # Number of Gaussian full-width half-maxima to compute
        # Smin=None,
        # method='fortran'
# ):
    # """Convert some lines into a spectrum."""
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,Γ = x0[i],S[i],Γ[i]
    # if method=='python':
        # raise Exception(f'Not implemented.')
        # _fortran_tools.calculate_gaussian_spectrum(x0,S,Γ,x.astype(float),y,nfwhm)
    # elif method=='fortran':
        # y = np.zeros(x.shape,dtype=float)
        # _fortran_tools.calculate_gaussian_spectrum(x0,S,Γ,x.astype(float),y,nfwhm)
    # elif method=='fortran stepwise':
        # y = np.zeros(x.shape,dtype=float)
        # _fortran_tools.calculate_stepwise_gaussian_spectrum(x0,S,Γ,x.astype(float),y,nfwhm)
    # else:
        # raise Exception(f'Unknown gaussian_method: {repr(method)}')
    # return(y)

# def lorentzian_spectrum(
        # x,                      # frequency scale (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # Γ,                     # Linewidth (cm-1 FWHM)
        # nfwhm=10.,             # Number of full-width half-maxima to compute
        # Smin=None,
# ):
    # """Convert some lines into a spectrum."""
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,Γ = x0[i],S[i],Γ[i]
    # # y = np.zeros(x.shape,dtype=float) 
    # # _fortran_tools.calculate_lorentzian_spectrum(x0,S,Γ,x.astype(float),y,float(nfwhm))
    # # _fortran_tools.calculate_lorentzian_spectrum_bin_average(x0,S,Γ,x.astype(float),y,float(nfwhm))
    # # _fortran_tools.calculate_stepwise_lorentzian_spectrum(x0,S,Γ,x.astype(float),y,float(nfwhm))
    # y = sum(lorentzian(x,*t,nfwhm=nfwhm) for t in zip(x0,S,Γ))
    # return(y)

# def fit_fwhm(x,y,subtract_background=True,make_plot=False):
    # """Roughly calculate full-width half-maximum of data x,y. Linearly
    # interpolates nearest points to half-maximum to get
    # full-width. Removes a linear background based on the extreme
    # points."""
    # if subtract_background:
        # y = y - y[0] + (x-x[0])*(y[-1]-y[0])/(x[-1]-x[0])
    # hm = (y.max()-y.min())/2.
    # i = np.argwhere(((y[1:]-hm)*(y[:-1]-hm))<0)
    # assert len(i)==2,"Poorly defined peak, cannot find fwhm."
    # x0 = (hm-y[i[0]])*(x[i[0]]-x[i[0]+1])/(y[i[0]]-y[i[0]+1]) + x[i[0]]
    # x1 = (hm-y[i[1]])*(x[i[1]]-x[i[1]+1])/(y[i[1]]-y[i[1]+1]) + x[i[1]]
    # if make_plot:
        # ax = plt.gca()
        # ax.plot(x,y)
        # ax.plot([x0,x1],[hm,hm],'-o')
    # return x1-x0

# def fit_gaussian(x,y,xmean=None,yint=None,width=None,printOutput=False,plotOutput=False):
    # """Fit a Gaussian line shape. Attempts to guess good initial values.
    # \nInputs:\n
    # k -- array energy scale
    # sigma -- cross-section to fit
    # printOutput -- set to True for a littel table
    # plotOutput -- set to True to issue matplotlib plotting commands
    # \nOutput a dictionary with the following information:\n
    # k0 -- center energy
    # strength  -- integrated cross-section
    # gaussian_width -- width of Gaussian component
    # fcn -- fitted function
    # """
    # ## guess initial parameter values
    # i=np.argmax(y)
    # if xmean is None: xmean=x[i]
    # if yint is None: yint=y[i];
    # if width is None:
        # try:
            # width=fwhm(x,y)
        # except:
            # width=1.
    # ## initial fit parameters
    # p = [xmean,width,yint]
    # ## function and residual
    # res=lambda p: y-p[2]*gaussian(x,fwhm=p[1],xmean=p[0],norm='area')
    # def func_res(p):
        # return y-p[2]*gaussian(x,fwhm=np.abs(p[1]),xmean=p[0],norm='area')
    # [p,pcov,info,mesg,success]=optimize.leastsq(func_res,p,full_output=1)
    # if not success: raise Exception('exit code '+str(success)+' '+mesg)
    # ## fitted parameters, standard errors, and residuals
    # xmean,width,yint = p
    # ## ensure positive
    # width=abs(width)
    # ## calculate error of parameters
    # chisq=sum(info["fvec"]*info["fvec"])
    # dof=len(info["fvec"])-len(p)
    # psigma =np.array([np.sqrt(pcov[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])
    # [dxmean,dyint,dwidth]=psigma
    # fcn = lambda x: p[2]*gaussian(x,fwhm=p[1],xmean=p[0],norm='area')
    # yf = fcn(x)
    # residual = info['fvec']
    # ## print parameters
    # if printOutput:
        # print(("\n".join([
                # "xmean = "+format(xmean,'12.2f')+" +- "+format(dxmean,'8.3g'),
                # "yint  = "+format(yint,'12.4g')+" +- "+format(dyint,'8.3g'),
                # "width = "+format(width,'12.4g')+" +- "+format(dwidth,'8.3g'),
                # ])))
        # # print(parameterString)
    # ## plot commands
    # if plotOutput:
        # # myAnnotate(parameterString)
        # ax = plt.gca()
        # ax.plot(x,y,'ro',label=r'data',markeredgecolor='red')
        # ax.plot(x,yf,'b-',label=r'fit')
        # ax.plot(x,residual,'g-',label=r'residual')
        # my.legend()
    # ## return data
    # return dict(xmean=xmean,yint=yint,width=width,fcn=fcn,yf=yf,residual=residual)

# def fit_lorentzian(x,y,x0=None,S=None,Γ=None,nfwhm=np.inf):
    # ## guess initial parameter values
    # i=np.argmax(y)
    # if x0 is None:
        # x0 = x[i]
    # if Γ is None:
        # Γ = my.fwhm(x,y,return_None_on_error=True)
        # if Γ is None:
            # Γ = 1
    # if S is None:
        # S = y[i]*Γ/2
    # o = Optimiser()
    # o.monitor_frequency = 'never'
    # p = o.add_parameter_set(x0=(x0,True,1e-3), Γ=(Γ,True,Γ*1e-2), S=(S,True,S*1e-2),)
    # def f():
        # return(lorentzian(x,p['x0'],p['S'],p['Γ'],nfwhm=nfwhm))
    # o.construct_functions.append(lambda:y-f())
    # o.optimise(monitor_frequency='never',verbose=False)
    # return(p,f())

# def fit_sinc(x,y,xmean=None,yint=None,width=None,print_output=False,plot_output=False):
    # """Fit a sinc line shape. Attempts to guess good initial values.
    # \nInputs:\n
    # x,y -- input data
    # xmean -- center of sinc
    # yint -- integrated area of sinc
    # width -- full-width half maximum
    # print_output -- set to True for a littel table
    # plot_output -- set to True to issue matplotlib plotting commands
    # \nOutput a dictionary with the following information:\n
    # x,y,yint,width and f (fitted function)
    # """
    # ## guess initial parameter values
    # i=np.argmax(y)
    # if xmean is None: xmean=float(x[i])
    # if width is None:
        # try:
            # width=float(fwhm(x,y))
        # except:
            # width=float((x.max()-x.min())/2.)
    # if yint is None:
        # yint=float(y[i]*width)
    # dxmean,dyint,dwidth = np.abs(x[1]-x[0])/2.,yint/10.,width/10.
    # ##  residual function and optimise
    # def func_res(p): return y-my.sinc(x,fwhm=p[2],strength=p[1],mean=p[0],norm='area')
    # (xmean,yint,width),(dxmean,dyint,dwidth) = my.leastsq(func_res,[xmean,yint,width],[dxmean,dyint,dwidth])
    # ## ensure positive
    # width=abs(width)
    # f = lambda x: y-my.sinc(x,fwhm=width,strength=yint,mean=xmean,norm='area')
    # yf = f(x)
    # residual = y-yf
    # if print_output:
        # print(("\n".join([
            # "xmean = "+format(xmean,'12f')+" +- "+format(dxmean,'8g'),
            # "yint  = "+format(yint, '12g') +" +- "+format(dyint,'8g'),
            # "width = "+format(width,'12g')+" +- "+format(dwidth,'8g'),
        # ])))
    # if plot_output:
         # ax = plt.gca()
         # ax.plot(x,y,'ro',label=r'data',markeredgecolor='red')
         # ax.plot(x,yf,'b-',label=r'fit')
         # ax.plot(x,residual,'g-',label=r'residual')
         # my.legend()
    # return dict(x=x,y=y,xmean=xmean,yint=yint,width=width,f=f,yf=yf,residual=residual)

# def fit_voigt(k,sigma,printOutput=False,plotOutput=False):
    # """Fit a voigt line shape. Attempts to guess good initial values.
    # \nInputs:\n
    # k -- array energy scale
    # sigma -- cross-section to fit
    # printOutput -- set to True for a littel table
    # plotOutput -- set to True to issue matplotlib plotting commands
    # \nOutput a dictionary with the following information:\n
    # k0 -- center energy
    # strength  -- integrated cross-section
    # gaussian_width -- width of Gaussian component
    # lorentzian_width -- width of Lorentzian component 
    # Dictionary also contains errors and arrays of data residuals etc.
    # """
    # ## guess initial parameter values
    # i=np.argmax(sigma); k0=k[i]; strength=sigma[i];
    # try: gaussian_width=fwhm(k,sigma)/2.
    # except: gaussian_width=1.
    # lorentzian_width = gaussian_width
    # ## fit parameters
    # p = [k0,strength,gaussian_width,lorentzian_width]
    # f=lambda p: sigma-voigtProfile(k,k0=p[0],strength=p[1],gaussian_width=p[2],
                                    # lorentzian_width=p[3])
    # [p,pcov,info,mesg,success]=optimize.leastsq(f,p,full_output=1)
    # if not success: raise Exception('exit code '+str(success)+' '+mesg)
    # ## fitted parameters, standard errors, and residuals
    # [k0,strength,gaussian_width,lorentzian_width]=p
    # ## ensure positive
    # strength=abs(strength);gaussian_width=abs(gaussian_width)
    # lorentzian_width=abs(lorentzian_width)
    # ## calculate error of parameters
    # chisq=sum(info["fvec"]*info["fvec"])
    # dof=len(info["fvec"])-len(p)
    # psigma =np.array([np.sqrt(pcov[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])
    # [dk0,dstrength,dgaussian_width,dlorentzian_width]=psigma
    # residual = info['fvec']
    # sigmaFit = sigma-residual
    # ## print parameters
    # parameterString = "\n".join([
            # "k0              = "+format(k0,'12.2f')+" +- "+format(dk0,'8.3g'),
            # "strength        = "+format(strength,'12.4g')+" +- "+format(dstrength,'8.3g'),
            # "gaussian_width   = "+format(gaussian_width,'12.4g')+" +- "+format(dgaussian_width,'8.3g'),
            # "lorentzian_width = "+format(lorentzian_width,'12.4g')+" +- "+format(dlorentzian_width,'8.3g'),
            # ])
    # if printOutput: print(parameterString)
    # ## plot commands
    # if plotOutput:
        # ax = plt.gca()
        # ax.plot(k,sigma,'ro',label=r'data',markeredgecolor='red')
        # ax.plot(k,sigmaFit,'b-',label=r'fit')
        # ax.plot(k,residual,'g-',label=r'residual')
    # ## return data
    # return {'k0':k0,'strength':strength, 'dk0':dk0,'dstrength':dstrength,
            # 'gaussian_width':gaussian_width, 'lorentzian_width':lorentzian_width,
            # 'dgaussian_width':dgaussian_width, 'dlorentzian_width':dlorentzian_width,
            # 'k':k,'sigma':sigma,'sigmaFit':sigmaFit,'residual':residual}

# def fit_fano(
        # x,y,
        # x0=None,x0vary=True,
        # Γ=None,
        # Γvary=True,
        # S=None,
        # Svary=True,
        # ρ=None,
        # ρvary=True,
        # q=None,qvary=True,
        # print_result=True,plot_result=True,
        # # monitor_frequency='every iteration', # 'rms decrease'
        # monitor_frequency='never', # 'rms decrease','every iteration'
        # extended_output=True,
        # limit_fitted_fwhms=10,    # set to a number of fwhms to limit the data before fitting by the initial fwhm estimate, None for full range
        # return_optimiser=False,   # pretty much internal use only
# ):
    # """Fit a fano profile to x,y data. The definition of the parameters is
    # probably in heays2011_thesis. Outputs dictionary with fitted
    # parameters, their uncertainties, and various other products. Tries
    # intelligently to guess the parameters. This will likely fail if
    # you provide multipeaked data. """
    # ## unless initial values provided, make intelligent initial guesses
    # imax = np.argmax(y)
    # if x0 is None:
        # x0 = x[imax]        # at maximum
    # if Γ is None:
        # try:
            # Γ = float(fit_fwhm(x,y)) 
        # except:
            # Γ = (x.max()-x.min())/4.
    # if limit_fitted_fwhms:
        # ibeg,iend = np.searchsorted(x,[x0-Γ*limit_fitted_fwhms,x0+Γ*limit_fitted_fwhms])
        # x,y = x[ibeg:iend],y[ibeg:iend]
        # imax = np.argmax(y)
    # if S is None:
        # S = integrate.trapz(y,x)
    # if ρ is None:
        # ρ = 1                     # fits reliably
    # ## if q is not provided run twice with a postivie and negative
    # ## value and adopt the best fit. I tired to predict in advance but
    # ## found difficult where resonances are poorly sampled
    # if q is None:
        # p1 = fit_fano(x,y,x0=x0,x0vary=x0vary,Γ=Γ,Γvary=Γvary,S=S,Svary=Svary,ρ=ρ,ρvary=ρ,
                      # q=1,qvary=qvary,
                      # print_result=False,plot_result=False, monitor_frequency='never',
                      # extended_output=extended_output, limit_fitted_fwhms=None,return_optimiser=True)
        # o1 = p1.pop('optimiser')
        # p2 = fit_fano(x,y,x0=x0,x0vary=x0vary,Γ=Γ,Γvary=Γvary,S=S,Svary=Svary,ρ=ρ,ρvary=ρ,
                      # q=-1,qvary=qvary,
                      # print_result=False,plot_result=False, monitor_frequency='never',
                      # extended_output=extended_output, limit_fitted_fwhms=None,return_optimiser=True)
        # o2 = p2.pop('optimiser')
        # if my.rms(o1.residual)<my.rms(o2.residual):
            # return(p1)
        # else:
            # return(p2)
    # ## optimiser
    # o = Optimiser()
    # p = o.add_parameter_set(
        # x0=(x0 ,x0vary,Γ*1e-3)        ,
        # Γ=(Γ*2.,Γvary ,Γ*1e-3)        ,
        # S=(S   ,Svary ,S*1e-3)        ,
        # q=(q   ,ρvary ,np.abs(q*1e-4)),
        # ρ=(ρ   ,qvary ,1e-4)          ,)
    # ## fitting function
    # def f():
        # return(fano(x,p['x0'],p['S'],p['Γ'],p['q'],p['ρ'],))
    # def r(): return(y-f())
    # o.add_construct(r)
    # o.optimise(monitor_frequency=monitor_frequency,verbose=False)
    # if print_result: print(o)
    # # ## the actual fit
    # if plot_result or extended_output:
        # yf,r = f(),r()
    # if plot_result:
        # ax = plt.gca()
        # ax.plot(x,y,'r-',label=r'data',markeredgecolor='red')
        # ax.plot(x,yf,'b-',label=r'fit')
        # ax.plot(x,r,'g-',label=r'residual')
        # my.legend_colored_text()
    # retval = dict(x=x,y=y,yf=yf,r=r,p=p,f=f,
                   # x0=p['x0'],S=p['S'],Γ=p['Γ'],ρ=p['ρ'],q=p['q'],
                   # dx0=p.dp['x0'],dS=p.dp['S'],dΓ=p.dp['Γ'],dρ=p.dp['ρ'],dq=p.dp['q'],)
    # if return_optimiser:
        # retval['optimiser'] = o
    # return(retval)

# def resolve_resonances(
        # fy, # function returning a function as a function of x
        # xbeg,xend,xstep, # range to search for resonances
        # peak_fraction=0.05, # fit to this precision of peak amplitue
        # wings_range=2,      # define wings in FWHM
        # wings_step=0.2, # define wing in FWHM
        # # fit_profile = None,
        # ignore_resonance_below=-np.inf,
        # verbose=True,
# ):
    # """Calculate continuum cross section and make sure resonances
    # are well resolved given a reasonable peaked shape."""
    # ## scan full range
    # xscan = np.arange(xbeg,xend,xstep)
    # yscan = fy(xscan)
    # xs,ys = [xscan],[yscan]
    # ## find and resolve maxima
    # imaxs = my.find(((yscan[1:-1]>yscan[:-2])&(yscan[1:-1]>yscan[2:])))+1
    # imaxs = imaxs[yscan[imaxs]>ignore_resonance_below]
    # xmaxima,ymaxima = [],[]                # position of maxima
    # for iimax,imax in enumerate(imaxs):
        # if verbose:
            # print(f'fitting resonance {iimax+1} of {len(imaxs)}')
        # ## ensure neighbours around peak are within a peak_fraction peak
        # ## value -- indicating peak found
        # xcenter = xscan[imax-1:imax+1+1]
        # ycenter = yscan[imax-1:imax+1+1]
        # i = 1
        # safety_stop=0
        # while min(ycenter[i-1],ycenter[i+1])/ycenter[i]<(1-peak_fraction):
            # safety_stop += 1
            # if safety_stop>100:
                # warnings.warn('safety stop')
                # break
            # x0 = 0.5*(xcenter[i]+xcenter[i+1])
            # x1 = 0.5*(xcenter[i]+xcenter[i-1])
            # xcenter = np.concatenate((xcenter,(x0,x1)))
            # ycenter = np.concatenate((ycenter,fy([x0,x1])))
            # i = np.argsort(xcenter)
            # xcenter,ycenter = xcenter[i],ycenter[i]
            # i = np.argmax(ycenter)
        # xmaxima.append(float(xcenter[i]))
        # ymaxima.append(float(ycenter[i]))
        # ## estimate FWHM from new data -- very simple triangular assumption
        # if ycenter.min()/ycenter.max()<(1-peak_fraction):
            # i = np.argmax(ycenter)                    # peak index
            # j = np.argmin(np.abs(ycenter/ycenter[i]-0.5)) # closest point to half maximum
            # fwhm = np.abs(xcenter[j]-xcenter[i])*2/(1-ycenter[j]/ycenter[i])
            # ## make sure line is well defined within requested fwhms
            # xwings = np.arange(max(xcenter[i]-fwhm*wings_range,xbeg),min(xcenter[i]+fwhm*wings_range,xend),fwhm*wings_step)
            # ywings = fy(xwings)
            # xs.extend([xcenter,xwings])
            # ys.extend([ycenter,ywings])
    # ## return result
    # x = np.concatenate(xs)
    # y = np.concatenate(ys)
    # x,i = np.unique(x,return_index=True)
    # y = y[i]
    # return(x,y)
    # # xmaxima,ymaxima = np.array(xmaxima),np.array(ymaxima)
    # # ## fit line profile
    # # lines = Dynamic_Recarray()
    # # if len(xmaxima)==0:
        # # pass
    # # elif fit_profile is None:
        # # lines.append(x0=xmaxima,y0=ymaxima)
    # # else:
        # # if len(xmaxima)==1:
            # # iedges = [0,len(x)]
        # # else:
            # # iedges = np.concatenate(([0],np.searchsorted(x,0.5*(xmaxima[1:]+xmaxima[:-1])),[len(x)]))
        # # for ibeg,iend in zip(iedges[0:-1],iedges[1:]):
            # # if fit_profile=='fano':
                # # d = fit_fano(x[ibeg:iend],y[ibeg:iend],print_result=False,plot_result=False,)
                # # for key in ('r','p','y','f'): d.pop(key) # remove unnecessary fit data to save memory
            # # else:
                # # raise ImplementationError(f'fit_profile not implemented: {repr(fit_profile)}')
            # # lines.append_row(**d)
    # # return(x,y,lines)

# def auto_fit_lines(
        # x,y,                    
        # fit_profile="peak",       # "peak" , "fano", "lorentzian", "estimate"
        # verbose=False,
        # **kwargs_find_peaks,
# ):
    # """Autodetect lines in experimental spectrum."""
    # if x is None: x = np.arange(len(y),dtype=float) # default x to index
    # ipeaks = my.find_peaks(y,x,**kwargs_find_peaks) # get line peak indices
    # ## fit line profile
    # lines = Dynamic_Recarray()
    # if len(ipeaks)==0: return(lines) # nothing to fit
    # ## compute region around peaks to fit (midpoints of adjacent
    # ## peaks are a boundary)
    # if len(ipeaks)==1:
        # iedges = [0,len(x)]
    # else:
        # iedges = np.concatenate(([0],np.searchsorted(x,0.5*(x[ipeaks[1:]]+x[ipeaks[:-1]])),[len(x)]))
    # ## fit lines independently
    # for ipeak,ibeg,iend in zip(ipeaks,iedges[0:-1],iedges[1:]):
        # if fit_profile=='peak':
            # d = dict(x0=x[ipeak],y0=y[ipeak])
        # elif fit_profile=='fano':
            # try:
                # d = fit_fano(x[ibeg:iend],y[ibeg:iend],print_result=verbose,plot_result=False,)
            # except Exception as err:
                # warnings.warn(f'fit_fano failed with error: {str(err)}')
                # continue
            # d['y0'] = y[ipeak]
            # for key in ('x','yf','r','p','y','f'):
                # d.pop(key) # remove unnecessary fit data to save memory
        # elif fit_profile=='lorentzian':
            # try:
                # p,yf = fit_lorentzian(x[ibeg:iend],y[ibeg:iend],)
                # d = dict(x0=p['x0'],S=p['S'],Γ=p['Γ'],y0=y[ipeak],x=x[ibeg:iend],yf=yf)
            # except Exception as err:
                # warnings.warn(f'fit_lorentzian failed with error: {str(err)}')
                # d = dict(x0=x[ipeak],S=np.nan,Γ=np.nan,y0=y[ipeak],x=x[ibeg:iend],yf=np.full(x[ibeg:iend].shape,np.nan)),
                # continue
        # elif fit_profile=='estimate':
            # x0 = x[ipeak] # estimate center of line as peak
            # y0 = y[ipeak] # peak height
            # ## estimate width of line
            # xi,yi = x[ibeg:iend],y[ibeg:iend]
            # hm = (y0+yi.min())/2. # half-maximum
            # ihm = np.argwhere(((yi[1:]-hm)*(yi[:-1]-hm))<0) # find indices nearest half-maximum
            # Γ = 2*np.min(np.abs(xi[ihm]-x0))                # estimated full-width half-maximum
            # S = 2*y0*Γ #  estimated integrated strength of line assuming triangular
            # d = dict(x0=x0,y0=y0,S=S,Γ=Γ)
        # else:
            # raise Exception(f'fit_profile not implemented: {repr(fit_profile)}')
        # lines.append_row(**d)
    # return(lines)

# def auto_estimate_lines(x,y,**kwargs_find_peaks):
    # """Autodetect lines in experimental spectrum."""
    # warnings.warn("Deprecated: Use auto_fit_lines(x,y,fit_profile='estimate',**kwargs_find_peaks)")
    # return(auto_fit_lines(x,y,fit_profile='estimate',**kwargs_find_peaks))

# def gaussian_spectrum(
        # x,                      # frequency scale (cm-1)
        # x0,                     # line centers (cm-1)
        # S,                      # line strengths
        # Γ,                     # Gaussian linewidth (cm-1 FWHM)
        # nfwhm=10.,             # Number of Gaussian full-width half-maxima to compute
        # Smin=None,
        # method='python'
# ):
    # """Convert some lines into a spectrum."""
    # ## no x, nothing to do
    # if len(x)==0 or len(x0)==0:
        # return(np.zeros(x.shape,dtype=float))
    # ## strip lines that are too weak
    # if Smin is not None:
        # i = np.abs(S)>Smin
        # x0,S,Γ = x0[i],S[i],Γ[i]
    # ## remove lines outside of fwhm calc range
    # if not nfwhm is None:
        # i = ((x0+nfwhm*Γ)<x[0]) | ((x0-nfwhm*Γ)>x[-1])
        # x0,S,Γ = x0[~i],S[~i],Γ[~i]
    # ## test if nothing left
    # if len(x0)==0:
        # return(np.full(x.shape,0.))
    # ## calc single process
    # if method=='python':
        # y = np.zeros(x.shape,dtype=float)
        # for (x0i,Si,Γi) in zip(x0,S,Γ):
            # i,j = np.searchsorted(x,[x0i-Γi*nfwhm,x0i+Γi*nfwhm])
            # gaussian(x[i:j],x0i,Si,Γi,y=y[i:j])
    # elif method=='fortran':
        # y = np.zeros(x.shape,dtype=float)
        # _fortran_tools.calculate_gaussian_spectrum(x0,S,Γ,x.astype(float),y,nfwhm)
    # elif method=='fortran stepwise':
        # y = np.zeros(x.shape,dtype=float)
        # _fortran_tools.calculate_stepwise_gaussian_spectrum(x0,S,Γ,x.astype(float),y,nfwhm)
    # else:
        # raise Exception(f'Unknown gaussian_method: {repr(method)}')
    # return y 
