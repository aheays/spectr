## standard libraries
import itertools 
import functools
from copy import copy,deepcopy
import re
from pprint import pprint

## nonstandard libraries
from scipy import constants
import numpy as np
from numpy import nan,array

## this module
from .dataset import Dataset
from . import convert
from .tools import vectorise,cache,file_to_dict,cast_abs_float_array
from . import tools
from . import database
from . import kinetics
from .exceptions import InferException,MissingDataException
from .optimise import optimise_method

prototypes = {}

prototypes['notes'] = dict(description="Notes regarding this line" , kind='U' ,infer=[])
prototypes['author'] = dict(description="Author of data or printed file" ,kind='U' ,infer=[])
prototypes['reference'] = dict(description="Reference",kind='U',infer=[])
prototypes['date'] = dict(description="Date data collected or printed" ,kind='U' ,infer=[])
prototypes['species'] = dict(description="Chemical species with isotope specification" ,kind='U' ,infer=[])
@vectorise(cache=True,vargs=(1,))
def _f0(self,species):
    species_object = kinetics.get_species(species)
    return species_object['chemical_name']

prototypes['chemical_species'] = dict(description="Chemical species without isotope specification" ,kind='U' ,infer=[('species',_f0)])

@vectorise(cache=True,vargs=(1,))
def _f0(self,species):
    try:
        return kinetics.get_species(species).point_group
    except:
        raise InferException
prototypes['point_group']  = dict(description="Symmetry point group of species.", kind='U',fmt='s', infer=[(('species',),_f0)])

@vectorise(vargs=(1,),cache=True)
def _f0(self,species):
    return kinetics.get_species(species)['mass']
@vectorise(vargs=(1,),cache=True)
def _f1(self,species):
    try:
        return database.get_species_property(species,'mass')
    except MissingDataException as err:
        raise InferException(str(err))
prototypes['mass'] = dict(description="Mass (amu)",kind='f', fmt='<11.4f', infer=[(('species',), _f0),])
prototypes['reduced_mass'] = dict(description="Reduced mass (amu)", kind='f', fmt='<11.4f', infer=[(('species','database',), lambda self,species: _get_species_property(species,'reduced_mass'))])
prototypes['E'] = dict(description="Level energy relative to the least",units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('Ee','ZPE'),lambda self,Ee,ZPE: Ee-ZPE),],default_step=1e-3)
prototypes['Ee'] = dict(description="Level energy relative to equilibrium geometry at J=0 (and neglecting spin for linear molecules)" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','ZPE'),lambda self,E,ZPE: E+ZPE),],default_step=1e-3)
prototypes['ZPE'] = dict(description="Zero-point energy of the lowest level relative to Ee" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[('species',lambda self,species: database.get_species_property(species,'ZPE')),],default_step=1e-3)
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind='f',infer=[])
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind='f',fmt='<10.5g', infer=[(('mass','Ttr','ν'), lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν)])

def _f0(self,label,v,Σ,ef,J,E):
    """Compute separate best-fit reduced energy levels for each
    sublevel rotational series."""
    order = 3
    Ereduced = np.full(E.shape,0.0)
    for di,i in tools.unique_combinations_mask(label,v,Σ,ef):
        labeli,vi,Σi,efi = di
        Ji,Ei = [],[]
        for Jj,j in tools.unique_combinations_mask(J[i]):
            Ji.append(Jj[0])
            Ei.append(E[i][j][0])
        Ji,Ei = array(Ji),array(Ei)
        pi = np.polyfit(Ji*(Ji+1),Ei,min(order,len(Ei)-1))
        if self.verbose:
            print(f'{label=} {v=} {Σ=} {ef=} {pi=}')
        Ereduced[i] = E[i] - np.polyval(pi,J[i]*(J[i]+1))
    return Ereduced

def _df0(self,Ereduced,label,dlabel,v,dv,Σ,dΣ,ef,ddef,J,dJ,E,dE):
    """Uncertainty calculation to go with _f0."""
    if dE is None:
        raise InferException()
    dEreduced = dE
    return dEreduced

prototypes['E_reduced'] = dict(description="Reduced level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('label','v','Σ','ef','J','E'),(_f0,_df0)),],)

@vectorise(cache=True,vargs=(1,))
def _f0(self,point_group):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if point_group in ('K','C∞v'):
        return 1.
    else:
        raise InferException

@vectorise(cache=True,vargs=(1,2,3))
def _f1(self,point_group,Inuclear,sa):
    """Calculate homonuclear diatomic molecule level degeneracy."""
    if point_group in ('D∞h'):
        ## get total number of even or odd exchange combinations
        ntotal = (2*Inuclear+1)**2
        neven = 2*Inuclear+1 + (ntotal-(2*Inuclear+1))/2
        nodd = ntotal - neven
        if Inuclear%1==0:
            ## fermion
            if sa==+1:
                return neven
            else:
                return nodd
        else:
            ## boson
            if sa==+1:
                return nodd
            else:
                return neven
    else:
        raise InferException()

prototypes['conf'] = dict(description="Electronic configuration.", kind='U', fmt='10s', infer=[])
prototypes['gnuclear'] = dict(description="Nuclear spin level degeneracy (relative only)." , kind='i' , infer=[(('point_group',),_f0),( ('point_group','Inuclear','sa'),_f1),])
prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind='i' , infer=[(('J','gnuclear'),lambda self,J,gnuclear: (2*J+1)*gnuclear,)])
# prototypes['pm'] = dict(description="Total inversion symmetry" ,kind='i' ,infer=[])
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind='f', fmt='<10.5g', infer=[(('A',),lambda self,τ: 5.309e-12*A,)])
prototypes['τ'] = dict(description="Total decay lifetime (s)", kind='f', infer=[(('A',), lambda self,A: 1/A,)])       
prototypes['A'] = dict(description="Total decay rate (s-1)", kind='f', infer=[(('Γ',),lambda self,Γ: Γ/5.309e-12,)])
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind='f',fmt='>0.1f',infer=[])
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind='f', infer=[(('J','SR'),lambda self,J,SR: J-SR,)])
prototypes['S'] = dict(description="Total electronic spin quantum number", kind='f',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'S'),)])
# prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum.",units='cm-1', kind='f',infer=[((),lambda self,: 0.,)])
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind='f', fmt='0.2f', infer=[],cast=cast_abs_float_array)
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq)],cast=cast_abs_float_array)
# prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind='U', infer=[(('database',), lambda self,: 'database',)])

# @vectorise(cache=True,vargs=(1,2))
def _f5(self,species,Tex):
    if self['Zsource'] != 'HITRAN':
        raise InferException(f'Zsource not "HITRAN".')
    from . import hitran
    return hitran.get_partition_function(species,Tex)
def _f3(self,species,Tex,E,g):
    """Compute partition function from data in self."""
    if self['Zsource'] != 'self':
        raise InferException(f'Zsource not "self".')
    retval = np.full(species.shape,nan)
    for (speciesi,Texi),i in tools.unique_combinations_mask(species,Tex):
        kT = convert.units(constants.Boltzmann,'J','cm-1')*Texi
        retval[i] = np.sum(g[i]*np.exp(-E[i]/kT))
    return retval
prototypes['Z'] = dict(description="Partition function.", kind='f', fmt='<11.3e', infer=[
    # (('species','Tex','E','g'),_f3),
    (('species','Tex'),_f5),
])
prototypes['α'] = dict(description="State population", kind='f', fmt='<11.4e', infer=[(('Z','E','g','Tex'), lambda self,Z,E,g,Tex : g*np.exp(-E/(convert.units(constants.Boltzmann,'J','cm-1')*Tex))/Z,)])
prototypes['Nself'] = dict(description="Column density (cm2)",kind='f',fmt='<11.3e', infer=[])
prototypes['label'] = dict(description="Label of electronic state", kind='U',infer=[])
prototypes['v'] = dict(description="Vibrational quantum number", kind='i',infer=[])
prototypes['ν1'] = dict(description="Vibrational quantum number for mode 1", kind='i',infer=[])
prototypes['ν2'] = dict(description="Vibrational quantum number for mode 2", kind='i',infer=[])
prototypes['ν3'] = dict(description="Vibrational quantum number for mode 3", kind='i',infer=[])
prototypes['ν4'] = dict(description="Vibrational quantum number for mode 4", kind='i',infer=[])
prototypes['l2'] = dict(description="Vibrational angular momentum 2", kind='i',infer=[])
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'Λ'))])
prototypes['L'] = dict(description="Total orbital angular momentum", kind='i',infer=[])
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind='i',infer=[])
prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'s'))])
@vectorise(cache=True,vargs=(1,2))
def _f0(self,ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['i'] = dict(description="Total parity.", kind='i',infer=[])
prototypes['σv'] = dict(description="Symmetry with respect to σv reflection.", kind='i',infer=[(('ef','J'),_f0,)])
prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind='i',infer=[])
prototypes['sa'] = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind='i',infer=[(('σv','gu'),lambda self,σv,gu: σv*gu,)])

def _f0(self,S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef'] = dict(description="e/f symmetry", kind='i',infer=[(('S','Λ','s'),_f0,)],fmt='+1d')

def _f0(S,SR,Λ):
    F = np.full(S.shape,np.nan)
    ## sort according to SR
    i = Λ>0
    F[i] = S[i]-SR[i]+1.
    ## special case Σ± states -- reverse order
    i = ~i
    F[i] = S[i]+SR[i]+1.
    if np.any(np.isnan(F)): raise InferException('Failed to computed F')
    return(F)
prototypes['Fi'] = dict(description="Spin multiplet index", kind='i',infer=[
    ('sublevel',lambda self,sublevel: [int(t[:-1]) for t in sublevel]),
    (('S','SR'),lambda self,S,SR: S-SR+1,)])

prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind='f',fmt='g',infer=[(('Λ','Σ'),lambda self,Λ,S: Λ+S,)])

def _f3(self,S):
    Σ = np.full(S.shape,np.nan)
    Σ[S==0] = 0             # singlet state
    if np.any(np.isnan(Σ)): raise InferException('Could not determine Σ solely from S.')
    return(Σ)
def _f6(self,S,Λ):
    Σ = np.full(S.shape,np.nan)
    Σ[S==0] = 0             # singlet state
    Σ[(S==0.5)&(Λ==0)] = 0.5         # 2Σ± state
    if np.any(np.isnan(Σ)): raise InferException('Could not determine Σ solely from S.')
    return(Σ)
def _f4(self,Λ,S,SR,LSsign):
    if np.any((Λ==0)&(S>1)): raise Exception("Not implemented")
    Σ = np.full(Λ.shape,np.nan)
    Σ[S==0] = 0.
    ## special case Σ states since sign of Σ is not well defined
    Σ[(Λ==0)&(S==1/2)] = 1/2
    Σ[(Λ==0)&(S==1)&(SR==0)] = 1
    Σ[(Λ==0)&(S==1)&(SR==+1)&(LSsign==+1)] = 0 # regular 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==+1)&(LSsign==-1)] = 1 # regular 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==-1)&(LSsign==+1)] = 1 # inverted 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==-1)&(LSsign==-1)] = 0 # inverted 3Σ states
    ## general case -- frequently wrong
    i = np.isnan(Σ)
    Σ[i] = -SR[i]*LSsign[i]
    # Σ[i] = SR[i]*LSsign[i]
    return(Σ)
def _f5(self,Λ,Ω,S):
    Σ = np.full(Λ.shape,np.nan)
    i = Λ==0
    Σ[i] = np.abs(Ω[i])
    i = S<=Λ
    Σ[i] = Ω[i]-Λ[i]
    if np.any(np.isnan(Σ)): raise InferException()
    return(Σ)
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind='f',
                       infer=[
                           (('S',), _f3),                  # trivial S==0
                           (('S','Λ'), _f6),               # trivial S==0 or 2Σ
                           (('Λ','Ω','S'), _f5),           # mostly works
                           (('Λ','S','SR','LSsign'), _f4), # general case
                       ],fmt='+g',)


def _f6(self,Λ,S,Σ,s,ef,LSsign):
    if np.any((Λ==0)&(S>1)): raise Exception("Not implemented")
    SR = np.full(Λ.shape,np.nan)
    SR[S==0] = 0                # trivial case
    ## special cases Σ states since sign of Σ is not well defined
    SR[(Λ==0)&(S==1/2)&(s==0)&(ef=='e')] = 1/2
    SR[(Λ==0)&(S==1/2)&(s==1)&(ef=='f')] = 1/2
    SR[(Λ==0)&(S==1/2)&(s==0)&(ef=='f')] = -1/2
    SR[(Λ==0)&(S==1/2)&(s==1)&(ef=='e')] = -1/2
    i = (Λ==0)&(S==1)&(s==0)&(Σ==0)&(ef=='f'); SR[i] = +1*LSsign[i] # 3Σ+(Σ=0,f)
    i = (Λ==0)&(S==1)&(s==0)&(Σ==1)&(ef=='e'); SR[i] = 0 # 3Σ+(Σ=1,e)
    i = (Λ==0)&(S==1)&(s==0)&(Σ==1)&(ef=='f'); SR[i] = -1*LSsign[i] # 3Σ+(Σ=1,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==0)&(ef=='e'); SR[i] = +1*LSsign[i] # 3Σ+(Σ=0,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==1)&(ef=='f'); SR[i] = 0 # 3Σ-(Σ=1,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==1)&(ef=='e'); SR[i] = -1*LSsign[i] # 3Σ+(Σ=1,f)
    ## general case
    i = np.isnan(SR)
    SR[i] = -Σ[i]*LSsign[i]
    return(SR)
def _f5(self,S):
    if not np.all(S==0): raise InferException()
    return(np.zeros(S.shape))
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind='f',infer=[
    (('S',), _f5), # trivial case, S=0
    (('J','N'),lambda self,J,N: J-N),
    (('S','F'),lambda self,S,F: S-F+1),                   # Fi ordering follows decreasing SR
    (('Λ','S','Σ','s','ef','LSsign'), _f6), # most general case
])

prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind='f',infer=[])


## Effective Hamiltonian parameters
prototypes['Tv']  = dict(description='Term origin' ,units='cm-1',kind='f',fmt='0.6f',infer=[])
# prototypes['dTv'] = dict(description='Uncertainty in Term origin (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Bv']  = dict(description='Rotational constant' ,units='cm-1',kind='f',fmt='0.8f',infer=[])
# prototypes['dBv'] = dict(description='Uncertainty in rotational constant (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Dv']  = dict(description='Centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dDv'] = dict(description='Uncertainty in centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Hv']  = dict(description='Third order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dHv'] = dict(description='Uncertainty in thrid order centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='3g',infer=[])
prototypes['Lv']  = dict(description='Fourth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dLv'] = dict(description='Uncertainty in fourth order centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='3g',infer=[])
prototypes['Av']  = dict(description='Spin-orbit energy' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dAv'] = dict(description='Uncertainty in spin-orbit energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['ADv'] = dict(description='Spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dADv']= dict(description='Uncertainty in spin-orbit centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['AHv'] = dict(description='Higher-order spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dAHv']= dict(description='Uncertainty in higher-order spin-orbit centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['λv']  = dict(description='Spin-spin energy',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dλv'] = dict(description='Uncertainty in spin-spin energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['λDv'] = dict(description='Spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dλDv']= dict(description='Uncertainty in spin-spin centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['λHv'] = dict(description='Higher-order spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dλHv']= dict(description='Uncertainty in higher-order spin-spin centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['γv']  = dict(description='Spin-rotation energy' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dγv'] = dict(description='Uncertainty in spin-rotation energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['γDv'] = dict(description='Spin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dγDv']= dict(description='Uncertainty in spin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.2g',infer=[])
prototypes['γHv'] = dict(description='Higher-orders pin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dγHv']= dict(description='Uncertainty in higher-order spin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.2g',infer=[])
prototypes['ov']  = dict(description='Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dov'] = dict(description='Uncertainty in Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oDv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['doDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oHv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['doHv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oLv']  = dict(description='Ligher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['doLv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['pv']  = dict(description='Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dpv'] = dict(description='Uncertainty in Λ-doubling constant p (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['pDv']  = dict(description='Higher-order Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dpDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant p (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['qv']  = dict(description='Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dqv'] = dict(description='Uncertainty in Λ-doubling constant q (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['qDv']  = dict(description='Higher-order Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
# prototypes['dqDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant q (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])

prototypes['Γv'] = dict(description="Total electronic-vibrational linewidth (cm-1 FWHM)", kind='f',  fmt='<10.5g', strictly_positive=True, infer=[(('τ',),lambda self,τ: 5.309e-12/τ,)]) # tau=1/2/pi/gamma/c
# prototypes['dΓv'] = dict(description="Uncertainty in total electronic-vibrational linewidth (cm-1 FWHM 1σ)", kind='f',  fmt='<10.5g', infer=[(('Γ','τ','dτ'), lambda self,Γ,τ,dτ: dτ*Γ/τ,)])
prototypes['τv'] = dict(description="Total electronic-vibrational decay lifetime (s)", kind='f',  fmt='<10.5g', infer=[(('Γv',), lambda self,Γv: 5.309e-12/Γv),( ('Atv',), lambda self,Atv: 1/Atv,)]) 
# prototypes['dτv'] = dict(description="Uncertainty in total electronic-vibrational decay lifetime (s 1σ)", kind='f',  fmt='<10.5g', infer=[(('Γ','dΓ','τ'), lambda self,Γ,dΓ,τ: dΓ/Γ*τ),(('At','dAt','τ'), lambda self,At,dAt,τ: dAt/At*τ,)])
prototypes['Atv'] = dict(description="Total electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('τv',), lambda self,τv: 1/τv), (('Adv','Ave'), lambda self,Adv,Aev: Adv+Aev,), (('Aev',), lambda self,Aev: Aev),( ('Adv',), lambda self,Adv: Adv,)])# Test for Ad and Ae, if failed then one or the other is undefined/zero
# prototypes['dAtv']= dict(description="Uncertainty in total electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[(('τ','dτ','At'), lambda self,τ,dτ,At: dτ/τ*At,)])
prototypes['Adv'] = dict(description="Nonradiative electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('At','Ae'), lambda self,At,Ae: At-Ae,)])
# prototypes['dAdv']= dict(description="Uncertainty in nonradiative electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['Aev'] = dict(description="Radiative electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('At','Ad'), lambda self,At,Ad: At-Ad,)])
# prototypes['dAev']= dict(description="Uncertainty in radiative electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['ηdv'] = dict(description="Fractional probability of electronic-vibrational level decaying nonradiatively (dimensionless)", kind='f',  fmt='<10.5g', infer=[(('At','Ad'),lambda self,At,Ad:Ad/A,)])
# prototypes['dηdv']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying by dissociation by any channels (dimensionless, 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['ηev'] = dict(description="Fractional probability of electronic-vibrational level decaying radiatively (dimensionless)", kind='f',  fmt='<10.5g', infer=[(('At','Ae'),lambda self,At,Ae:Ae/A,)])
# prototypes['dηev']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying radiatively (dimensionless, 1σ)", kind='f',  fmt='<10.5g', infer=[])

## v+1 reduced versions of these
def _vibrationally_reduce(
        self,
        reduced_quantum_number,
        reduced_polynomial_order,
        val,dval=None,
):
    """Reduce a variable."""
    reduced = np.full(val.shape,np.nan)
    ## loop through unique rotational series
    for keyvals in self.unique_dicts(*self.qn_defining_independent_vibrational_progressions):
        i = self.match(**keyvals)
        x = self[reduced_quantum_number][i]*(self[reduced_quantum_number][i]+1)
        y = val[i]
        ## do an unweighted mean if uncertainties are not set or all the same
        if dval is None:
            dy = None
        else:
            dy = dval[i]
        p = my.polyfit(x,y,dy,order=max(0,min(reduced_polynomial_order,sum(i)-2)),error_on_missing_dy=False)
        reduced[i] = p['residuals']
    return(reduced)

prototypes['Tvreduced'] = dict(description="Vibrational term value reduced by a polynomial in (v+1/2)",units='cm-1', kind='f',  fmt='<11.4f',
    infer=[(('self','reduced_quantum_number','reduced_polynomial_order','Tv','dTv'), _vibrationally_reduce), # dTv is known -- use in a weighted mean
           (('self','reduced_quantum_number','reduced_polynomial_order','Tv'), _vibrationally_reduce,)]) # dTv is not known
prototypes['dTvreduced'] = dict(description="Uncertainty in vibrational term value reduced by a polynomial in (v+1/2) (cm-1 1σ)", kind='f',  fmt='<11.4f', infer=[(('dT',),lambda self,dT: dT,)])
prototypes['Tvreduced_common'] = dict(description="Term values reduced by a common polynomial in (v+1/2)",units='cm-1', kind='f',  fmt='<11.4f', infer=[(('v','Tv','Tvreduced_common_polynomial'), lambda self,v,Tv,Tvreduced_common_polynomial: Tv-np.polyval(Tvreduced_common_polynomial,v+0.5)),(('v','Tv'), lambda self,v,Tv: Tv-np.polyval(np.polyfit(v+0.5,Tv,3),v+0.5),)])
prototypes['dTvreduced_common'] = dict(description="Uncertaintty in term values reduced by a common polynomial in (v+1/2) (cm-1 1σ)", kind='f',  fmt='<11.4e', infer=[(('dTv',), lambda self,dTv: dTv)])
prototypes['Tvreduced_common_polynomial'] = dict(description="Polynomial in terms of (v+1/2) to reduce all term values commonly",units='cm-1', kind='o', infer=[])
prototypes['Bv_μscaled']  = dict(description='Rotational constant scaled by reduced mass to an isotopologue-independent value' ,units='cm-1', kind='f',fmt='0.8f', infer=[(('Bv','reduced_mass'),lambda self,Bv,reduced_mass: Bv*reduced_mass,)])
prototypes['dBv_μscaled'] = dict(description='Uncertainty in Bv_μscaled (1σ, cm-1)' ,kind='f',fmt='0.2g', infer=[(('Bv','dBv','Bv_μscaled'),lambda self,Bv,dBv,Bv_μscaled:dBv/Bv*Bv_μscaled,)])

def _collect_prototypes(*keys):
    retval = {key:prototypes[key] for key in keys}
    return retval

class Base(Dataset):
    """Common stuff for for lines and levels."""
    default_prototypes = _collect_prototypes()
    default_attributes = Dataset.default_attributes | {'Zsource':None,}

    def __init__(self,*args,**kwargs):
        kwargs.setdefault('permit_nonprototyped_data',False)
        Dataset.__init__(self,*args,**kwargs)

    @optimise_method(
        add_construct_function=False,
        add_format_input_function=True,
        format_single_line=True,
        execute_now= True)
    def set_by_qn(self,**kwargs):
        """Set some data to fixed values or optimised parameters, limiting
        setting to matching defining quantum numbers, all given as key word
        arguments."""
        ## collect quantum numbers and set data
        qn,p = {},{}
        for key,val in kwargs.items():
            if key in self.defining_qn:
                qn[key] = val
            else:
                p[key] = val
        ## set data
        for key,val in p.items():
            self.set_parameter(key,val,match=qn)
            self.pop_format_input_function()

    def sort(self,*sort_keys,reverse_order=False):
        if len(sort_keys) == 0:
            sort_keys = [key for key in self.defining_qn if self.is_known(key)]
        Dataset.sort(self,*sort_keys,reverse_order=reverse_order)

class Generic(Base):
    """A generic level."""
    default_prototypes = _collect_prototypes(
        'reference',
        'species','chemical_species',
        'label',
        'point_group',
        'E','Ee','ZPE','E_reduced',
        'Γ','ΓD',
        'J','N','S',
        'g','gnuclear',
        'Teq','Tex','Z','α',
        'Nself',
    )
    defining_qn = ('species','label','ef','J')
    default_xkey = 'J'
    default_zkeys = ('species','label','ef')

class Atomic(Generic):
    default_prototypes = _collect_prototypes(
        *Generic.default_prototypes,
        'conf','L','gu',
    )
    defining_qn = ('species','conf','J','S',)
    default_zkeys = ('species',)

class Linear(Generic):
    default_prototypes = _collect_prototypes(
        *Generic.default_prototypes,
        'Λ','s','Σ','SR','Ω','Fi',
        'i','gu','σv','sa','ef',
    )
    defining_qn = ('species','label','ef','J')
    default_zkeys = ('species','label','ef','Σ')

class LinearTriatomic(Linear):
    """A generic level."""
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'ν1','ν2','ν3','l2',
    )
    defining_qn = ('species','label','ef','ν1','ν2','ν3','l2','J')
    defining_zkeys = ('species','label','ef','ν1','ν2','ν3','l2')

class LinearDiatomic(Linear):
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'v',
        'Γv','τv','Atv','Adv','Aev',
        'ηdv','ηev',
        'Tv','Bv','Dv','Hv','Lv',
        'Av','ADv','AHv',
        'λv','λDv','λHv',
        'γv','γDv','γHv',
        'ov','oDv','oHv','oLv',
        'pv','qv',
        'pDv','qDv',
        'Tvreduced','Tvreduced_common',
        'Bv_μscaled',
    )
    defining_qn = ('species','label','v','Σ','ef','J')
    default_zkeys = ('species','label','v','Σ','ef')

    def load_from_duo(self,filename):
        """Load an output level list computed by DUO (yurchenko2016)."""
        data = file_to_dict(filename)
        if len(data) == 11:
            for icolumn,key in enumerate(('index','E','g','J','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
        elif len(data) == 12:
            for icolumn,key in enumerate(('index','E','g','J','E_residual','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
            data.pop('E_residual') # not currently used
        if len(data)==0:
            print(f'warning: no data found in {repr(filename)}')
            return
        ## get Λ/Σ/Ω sign conventions to match mine: Λ is +, Σ + or -,
        ## Ω is absolute
        i = data['Λ'] < 0
        data['Σ'][i] *= -1
        data['Λ'] = np.abs(data['Λ'])
        data['Ω'] = np.abs(data['Ω'])
        ## translate +/- to +1/-1
        i = data['s']=='-'
        data['s'] = np.full(len(data['s']),+1,dtype=int)
        data['s'][i] = -1
        ## translate e/f to +1/-1
        i = data['ef']=='f'
        data['ef'] = np.full(len(data['ef']),+1,dtype=int)
        data['ef'][i] = -1
        data.pop('index')
        self.extend(**data)

    def load_from_spectra(self,filename):
        """Old filetype. Incomplete"""
        data = file_to_dict(filename)
        ## keys to translate
        for key_old,key_new in (
                ('T','E'),
                ('Tref','ZPE'),
                ):
            if key_old in data:
                assert key_new not in data
                data[key_new] = data.pop(key_old)
        ## data to modify
        if 'ef' in data:
            i = data['ef']=='f'
            data['ef'] = np.full(len(data['ef']),+1,dtype=int)
            data['ef'][i] = -1
        ## data to ignore
        for key in (
                'level_transition_type',
                'partition_source',
                ):
            data.pop(key)
        self.extend(**data)

Diatomic = LinearDiatomic
