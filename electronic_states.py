from copy import copy
from pprint import pprint
from functools import lru_cache
import itertools

import numpy as np
from numpy import nan,array
# import sympy
# from scipy import linalg

# from . import levels,lines
# from . import quantum_numbers
# from . import tools
# from .optimise import Optimiser,P,auto_construct_method
# from .kinetics import get_species,Species



def calc_unbound_wavefunction(R,Rstep,V,μ,E,method='fortran'):
    N = V.shape[0]
    n = V.shape[1]
    ## get coupled wavefunctions 
    if method=='python':
        ## from johnson1978
        I = np.zeros(V.shape,dtype=float)
        for i in range(n): I[:,i,i] = 1 # speed this up
        _constant0 = 0.029660161 # assumes input units of a.m.u/cm-1/A 1e-20*constants.atomic_mass*constants.speed_of_light*100.*constants.h/constants.hbar**2
        T = -Rstep**2/12*(2*μ)*(E*I-V)*_constant0
        U = 12*np.linalg.inv(I-T) - 10*I
        ## calculate RR outwards
        RR = np.zeros(V.shape,dtype=float) # variable RR already taken for radial coordinate
        RRinv = np.zeros(V.shape,dtype=float) # variable RR already taken for radial coordinate
        RR[0,:,:] = U[0,:,:]
        RRinv[0,:,:] = np.linalg.inv(RR[0,:,:])
        for i in range(1,V.shape[0]):
            RR[i,:,:] = U[i,:,:]-RRinv[i-1,:,:]
            RRinv[i,:,:] = np.linalg.inv(RR[i,:,:])
        ## calculate F inwards
        F  = np.zeros(V.shape,dtype=float)
        for i in range(n):
            if V[-1,i,i]<E: F[-1,i,i] = 1 # unbound diagonal 1, else zero
        for i in range(N-2,0,-1):
            F[i,:,:] = np.dot(RRinv[i+1,:,:],F[i+1,:,:])
        ## calc χ from F
        χ  = np.zeros(V.shape,dtype=float)
        for i in range(1,V.shape[0]):
            χ[i,:,:] = np.dot(np.linalg.inv(I[i,:,:]-T[i,:,:]),F[i,:,:]) # take out of loop and vectorise?
    elif method=='fortran':
        pass
        V_fortran = np.array(V,order='F')
        χ = np.zeros(V.shape,dtype=float,order='F')
        _fortran_tools.calculate_multi_channel_wavefunction(V_fortran,E,Rstep,μ,χ)
    ## discards closed channel solutions, matrix is (ntotal,nopen)
    iopen = my.find([V[-1,i,i]<E for i in range(n)]) # find unbound solutions with good aymptotic properties
    if len(iopen)==0: return(np.full((N,n,n),np.nan)) # no open channels return empty array
    χ = χ[:,:,iopen]                                           # discard output closed channels
    ## wavenumber of open channels, matrix is (nopen,nopen)
    _constant1 = 2*μ*constants.atomic_mass/constants.hbar**2 # assumes input units of a.m.u/cm-1/A 1e-20*constants.atomic_mass*constants.speed_of_light*100.*constants.h/constants.hbar**2
    k = np.sqrt(_constant1*my.k2J(E-np.diagonal(V[-1][:,iopen][iopen,:])))
    k = np.column_stack([k for t in k]) # REALLY???
    ## determine asymptotic Bessel function coefficients, χ = A*J
    ## + B*N. 1 is 2nd to last gridpoint, 2 is the last
    J1,J2 = (scipy.special.jv(1,k*my.A2m(R[-2])), scipy.special.jv(1,k*my.A2m(R[-1]))) # Bessel function of the 1st kind
    N1,N2 = (scipy.special.yn(1,k*my.A2m(R[-2])), scipy.special.yn(1,k*my.A2m(R[-1]))) # Bessel function of the 2nd kind
    χ1,χ2 = χ[-2,iopen,:],χ[-1,iopen,:] # last two points of open channel wavefunctions
    A = (χ1/N1-χ2/N2)/(J1/N1-J2/N2)
    B = (χ1/J1-χ2/J2)/(N1/J1-N2/J2)
    ## transform to pure asymptotic states with correct normalisation
    U = np.sqrt(2)*np.linalg.inv(-1j*A+B) # (nopen×nopen)
    _constant2 = constants.hbar**2*constants.pi/(2*μ*constants.atomic_mass) # assumes input units of a.m.u/cm-1/A 1e-20*constants.atomic_mass*constants.speed_of_light*100.*constants.h/constants.hbar**2
    χ = _constant2*np.dot(χ,U)       # (ntotal×nopen)
    ## get a  full size closed channels indicated bynans
    χretval = np.full((N,n,n),np.nan+1j*np.nan)
    χretval[:,:,iopen] = χ
    return(χretval)


def find_single_channel_bound_levels_in_energy_range(
        V,                      # Potential energy curve (cm-1)
        dR,                     # Internuclear-distance grid step (Å)
        μ,                      # Reduced mass (amu)
        Emin=-np.inf, # Beginning of energy range, defaults to equilibrium energy (cm-1)
        Emax=np.inf, # End of energy range, defaults to dissociation energy (cm-1)
        ΔE=100, # Step size for initial search -- should be less than vibrational separation
        δE=1e-3,                # Energy localisation tolerance
):
    """Find all bound levels of a provided potential-potential energy
    curve in a given energy range."""
    ## rationalise ranges
    Emax = min(Emax,V[-1])
    Emin = max(Emin,V.min())
    assert Emin<Emax
    ## get bound levels -- may well miss some. Uses Fortran so
    ## some messing about with inputs/outputs required
    v = np.zeros(1000,dtype=int)
    E = np.zeros(1000,dtype=float)
    χ = np.zeros((1000,len(V)),dtype=float,order='F')
    nfound = np.array([0],dtype=int)
    _fortran_tools.find_single_channel_bound_levels(v,E,χ,nfound,V,dR,μ,Emin,Emax,ΔE,δE,len(V))
    nfound = int(nfound)
    v,E,χ = v[:nfound],E[:nfound],χ[:nfound,:]
    return(v,E,χ)  


def find_single_channel_bound_levels(
        v,                      # vibrational levels to find, defined by node counting.
        V,                      # Potential energy curve (cm-1)
        dR,                     # Internuclear-distance grid step (Å), must be a uniform grid
        μ,                      # Reduced mass (amu)
        Emin=-np.inf, # Beginning of energy range, defaults to equilibrium energy (cm-1), Can be expanded during searching.
        Emax=np.inf, # End of energy range, defaults to dissociation energy (cm-1). Can be expanded during searching.
        ΔE=100, # Step size for initial search -- should be less than vibrational separation
        δE=1e-3,                # Energy localisation tolerance
        raise_error_on_v_not_found=False, # if no error then returns np.nan for the energy
):
    """Find requested vibrational levels of a provided potential-energy
    curve with an initial estimated energy range."""
    Efound,vfound,χfound = np.zeros(0),np.zeros(0),np.zeros((0,len(V)))
    Vmin,Vmax = V.min(),V[-1] # plausible limits of energy given potential
    Emin = max(Emin,Vmin+ΔE*0.2)
    Emax = min(Emax,Vmax-ΔE*0.2)
    vnotfound = []              # record which v-searches failed
    ## search
    for vi in v:
        for safety_stop in range(1000):
            ## add new levels
            if vi in vfound: break      # already found
            vfound,i = np.unique(vfound,return_index=True)
            Efound,χfound = Efound[i],χfound[i]
            i = np.searchsorted(vfound,vi)
            if len(vfound)==0:
                ## increase search range up and down
                Emin,Emax = max(Vmin,Emin-5*ΔE),min(Vmax,Emax+5*ΔE)
            elif vi<vfound.min():
                ## v lies below current domain. Search below. Use
                ## lowest 2 v levels to extrapolate to wanted energy,
                ## or bisect potential below.
                if Emin<(Vmin+ΔE*0.1):
                    message = f"Could not find v={vi} and already at potential minimum. Lowest v found is {min(vfound)}"
                    if raise_error_on_v_not_found:
                        raise Exception(message)
                    else:
                        warnings.warn(message)
                        vnotfound.append(vi)
                        break
                if i>1:
                    Epredicted = Efound[0]-(Efound[1]-Efound[0])/(vfound[1]-vfound[0])*(vfound[0]-vi)
                    Emin,Emax = max(Vmin,min(Emin,Epredicted)-2*ΔE),Emin+2*ΔE
                else:
                    Emin,Emax = Emin-(Emax-Emin),Emin+2*ΔE
            elif vi>vfound.max():       # increase Emax
                ## v lies above current domain. Search abvove. Use
                ## highest 2 v levels to predict wanted energy, or
                ## bisect potential above.
                if Emax>(Vmax-ΔE*0.1):
                    message = f"Could not find v={vi} and already at potential maximum. Highest v found is {max(vfound)}"
                    if raise_error_on_v_not_found:
                        raise Exception(message)
                    else:
                        warnings.warn(message)
                        vnotfound.append(vi)
                        break
                if i>1:
                    Epredicted = Efound[-1]+(Efound[-1]-Efound[-2])/(vfound[-1]-vfound[-2])*(vi-vfound[-1])
                    Emin,Emax = Emax-2*ΔE,min(Vmax,max(Emax,Epredicted)+2*ΔE)
                else:
                    Emin,Emax = Emax-2*ΔE,Emax+(Emax-Emin)
            elif (Efound[i]-Efound[i-1])<(4*ΔE):               # interstitial v missing, need to decrease ΔE
                Emin,Emax,ΔE = Efound[i-1],Efound[i],ΔE/2
            else:          # interstitial v missing but a large gap, search the region with current ΔE
                Emin,Emax= Efound[i-1],Efound[i]
            ## search new region
            tv,tE,tχ = tools.find_single_channel_bound_levels_in_energy_range(V,dR,μ,Emin,Emax,ΔE,δE)
            vfound,Efound,χfound = np.concatenate((vfound,tv)),np.concatenate((Efound,tE)),np.concatenate((χfound,tχ)),
            # for Ei,vi in zip(Efound,vfound):
                # self._previous_bound_levels[(species,J,Σ,vi)] = Ei # add to cache
        else:
            message = f"Safety stop reached and level not found: v={vi}"
            if raise_error_on_v_not_found:
                raise Exception(message)
            else:
                warnings.warn(message)
                vnotfound.append(vi)
                break
    ## final sort, NaN energies for v not found and get rid of any duplicates
    if len(vnotfound)>0:
        vfound = np.concatenate((vfound,vnotfound))
        Efound = np.concatenate((Efound,[np.nan for t in vnotfound]))
        χfound = np.concatenate((χfound,[np.full(len(V),np.nan) for t in vnotfound]))
    vfound,i = np.unique(vfound,return_index=True)
    Efound,χfound = Efound[i],χfound[i]
    return(vfound,Efound,χfound)



