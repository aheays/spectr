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
from . import quantum_numbers
from .exceptions import InferException,MissingDataException
from .optimise import optimise_method,Parameter

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

# prototypes['name'] = dict(description="Quantum numbers encoded into a string" ,kind='U' ,infer=[('species',_f0)])

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
prototypes['mass'] = dict(description="Mass",units="amu",kind='f', fmt='<11.4f', infer=[(('species',), _f0),])
prototypes['reduced_mass'] = dict(description="Reduced mass",units="amu", kind='f', fmt='<11.4f', infer=[(('species','database',), lambda self,species: _get_species_property(species,'reduced_mass'))])

@vectorise(vargs=(1,2,3,4,5,6),cache=True)
def _f0(self,species,label,v,Σ,ef,J):
    """Get diatomic molecule level energies from database."""
    level = database.get_level(species)
    if level['classname'] != 'levels.Diatomic':
        raise InferException('Not level.Diatomic')
    i = tools.find(level.match(species=species,label=label,v=v,Σ=Σ,ef=ef,J=J))
    if len(i) == 0:
        raise InferException('no match found')
    if len(i) > 1:
        raise InferException('multiple matches found')
    return level['E'][i][0]

prototypes['E'] = dict(description="Level energy relative to the least",units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('Ee','ZPE'),lambda self,Ee,ZPE: Ee-ZPE),(('species','label','v','Σ','ef','J'),_f0)],default_step=1e-3)
prototypes['Ee'] = dict(description="Level energy relative to equilibrium geometry at J=0 and neglecting spin" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','ZPE'),lambda self,E,ZPE: E+ZPE),],default_step=1e-3)
prototypes['Eref'] = dict(description="Reference level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[])
prototypes['Eres'] = dict(description="Residual error of level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','Eref'),lambda self,E,Eref: Eref-E)])
prototypes['ZPE'] = dict(description="Zero-point energy of the lowest level relative to Ee" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[('species',lambda self,species: database.get_species_property(species,'ZPE')),],default_step=1e-3)
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind='f',infer=[])
prototypes['ΓD'] = dict(description="Gaussian Doppler width",units="cm-1 FWHM",kind='f',fmt='<10.5g', infer=[(('mass','Ttr','ν'), lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν)])

def _f0(self,species,label,v,Σ,ef,J,E):
    """Compute separate best-fit reduced energy levels for each
    sublevel rotational series."""
    order = 3
    Ereduced = np.full(E.shape,0.0)
    for di,i in tools.unique_combinations_masks(species,label,v,Σ,ef):
        speciesi,labeli,vi,Σi,efi = di
        Ji,Ei = [],[]
        for Jj,j in tools.unique_combinations_masks(J[i]):
            Ji.append(Jj[0])
            Ei.append(E[i][j][0])
        Ji,Ei = array(Ji),array(Ei)
        pi = np.polyfit(Ji*(Ji+1),Ei,min(order,len(Ei)-1))
        if self.verbose:
            print(f'{species=} {label=} {v=} {Σ=} {ef=} {pi=}')
        Ereduced[i] = E[i] - np.polyval(pi,J[i]*(J[i]+1))
    return Ereduced
def _df0(self,Ereduced,species,dspecies,label,dlabel,v,dv,Σ,dΣ,ef,ddef,J,dJ,E,dE):
    """Uncertainty calculation to go with _f0."""
    if dE is None:
        raise InferException()
    dEreduced = dE
    return dEreduced
prototypes['Ereduced'] = dict(description="Reduced level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('species','label','v','Σ','ef','J','E'),(_f0,_df0)),],)

def _f0(self,J,E):
    """Compute separate best-fit reduced energy levels for each
    sublevel rotational series."""
    p = np.polyfit(J*(J+1),E,min(3,len(np.unique(J))-1))
    p[-1] = 0 
    Ereduced_common = E - np.polyval(p,J*(J+1))
    return Ereduced_common
def _df0(self,Ereduced_common,J,dJ,E,dE):
    """Uncertainty calculation to go with _f0."""
    if dE is None:
        raise InferException()
    dEreduced_common = dE
    return dEreduced_common
prototypes['Ereduced_common'] = dict(description="Reduced level energy common to all bands." ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('J','E'),(_f0,_df0)),],)

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
prototypes['gnuclear'] = dict(description="Nuclear spin level degeneracy (relative only)" , kind='i' , infer=[(('point_group',),_f0),( ('point_group','Inuclear','sa'),_f1),])

prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind='f',infer=[(('species',), lambda self,species: database.get_species_property(species,'Inuclear'))])
prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind='i' , infer=[(('J','gnuclear'),lambda self,J,gnuclear: (2*J+1)*gnuclear,)])
# prototypes['pm'] = dict(description="Total inversion symmetry" ,kind='i' ,infer=[])
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition" ,units="cm-1 FWHM",kind='f',cast=cast_abs_float_array,fmt='<10.5g', infer=[(('A',),lambda self,τ: 5.309e-12*A,)])
prototypes['Γref'] = dict(description="Reference level natural linewidth" ,units='cm-1.FWHM',kind='f' ,fmt='<14.7f' ,infer=[])
prototypes['Γres'] = dict(description="Residual error of level natural linewidth" ,units='cm-1.FWHM',kind='f' ,fmt='<14.7f' ,infer=[(('Γ','Γref'),lambda self,Γ,Γref: Γ-Γref)])
prototypes['τ'] = dict(description="Total decay lifetime",units="s", kind='f', infer=[(('A',), lambda self,A: 1/A,)])       
prototypes['A'] = dict(description="Total decay rate",units="s-1", kind='f', infer=[(('Γ',),lambda self,Γ: Γ/5.309e-12,)])
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind='f',fmt='>0.1f',infer=[])
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind='f', infer=[(('J','SR'),lambda self,J,SR: J-SR,)])
prototypes['S'] = dict(description="Total electronic spin quantum number", kind='f',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'S'),)])
# prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum.",units='cm-1', kind='f',infer=[((),lambda self,: 0.,)])
prototypes['Teq'] = dict(description="Equilibriated temperature",units="K", kind='f', fmt='0.2f', infer=[],cast=cast_abs_float_array)
prototypes['Tex'] = dict(description="Excitation temperature",units="K", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq)],cast=cast_abs_float_array)
prototypes['conf'] = dict(description="Electronic configuration", kind='U', fmt='10s', infer=[])

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
    for (speciesi,Texi),i in tools.unique_combinations_masks(species,Tex):
        kT = convert.units(constants.Boltzmann,'J','cm-1')*Texi
        retval[i] = np.sum(g[i]*np.exp(-E[i]/kT))
    return retval
def _f4(self,species,Tex):
    """Compute partition function from data in self."""
    if self['Zsource'] != 'database':
        raise InferException(f'Zsource not "database"')
    return database.get_partition_function(species,Tex,self['Eref'])
prototypes['Z'] = dict(description="Partition function.", kind='f', fmt='<11.3e', infer=[
    (('species','Tex','E','g'),_f3),
    (('species','Tex'),_f5),
    (('species','Tex'),_f4),
])
prototypes['α'] = dict(description="State population", kind='f', fmt='<11.4e', 
                       infer=[
                           (('Z','E','g','Tex'), lambda self,Z,E,g,Tex : g*np.exp(-E/(convert.units(constants.Boltzmann,'J','cm-1')*Tex))/Z,),
                       ])
prototypes['Nself'] = dict(description="Column density",units="cm2",kind='f',fmt='<11.3e', infer=[])
prototypes['label'] = dict(description="Label of electronic state", kind='U',infer=[])
prototypes['v'] = dict(description="Vibrational quantum number", kind='i',infer=[])
prototypes['ν1'] = dict(description="Vibrational quantum number for mode 1", kind='i',infer=[])
prototypes['ν2'] = dict(description="Vibrational quantum number for mode 2", kind='i',infer=[])
prototypes['ν3'] = dict(description="Vibrational quantum number for mode 3", kind='i',infer=[])
prototypes['ν4'] = dict(description="Vibrational quantum number for mode 4", kind='i',infer=[])
prototypes['l2'] = dict(description="Vibrational angular momentum 2", kind='i',infer=[])
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'Λ'))]) # 
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

prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'gu'))])
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
    Fi = np.full(S.shape,np.nan)
    ## sort according to SR
    i = Λ>0
    Fi[i] = S[i]-SR[i]+1.
    ## special case Σ± states -- reverse order
    i = ~i
    Fi[i] = S[i]+SR[i]+1.
    if np.any(np.isnan(Fi)):
        raise InferException('Failed to computed Fi')
    return(Fi)
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
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z axis", kind='f',infer=[
    (('S',), _f5), # trivial case, S=0
    (('J','N'),lambda self,J,N: J-N),
    (('S','Fi'),lambda self,S,Fi: S-Fi+1),                   # Fi ordering follows decreasing SR
    (('Λ','S','Σ','s','ef','LSsign'), _f6), # most general case
])

def _f0(self):
    for key in self.defining_qn:
        if not self.is_known(key):
            raise InferException(f'Cannot infer _qnhash because {key} not known')
    _qnhash = np.empty(len(self),dtype=int)
    for i,qn in enumerate(zip(*[self[key] for key in self.defining_qn])):
        _qnhash[i] = hash(qn)
    self._set_value('_qnhash',_qnhash,dependencies=self.defining_qn)
    return None
prototypes['_qnhash'] = dict(description="Hash of defining quantum numbers", kind='i',infer=[((),_f0),])


## Effective Hamiltonian parameters
prototypes['Tv']  = dict(description='Term origin' ,units='cm-1',kind='f',fmt='0.6f',infer=[])
prototypes['Bv']  = dict(description='Rotational constant' ,units='cm-1',kind='f',fmt='0.8f',infer=[])
prototypes['Dv']  = dict(description='Centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['Hv']  = dict(description='Third order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['Lv']  = dict(description='Fourth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['Mv']  = dict(description='Fifth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['Av']  = dict(description='Spin-orbit energy' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['ADv'] = dict(description='Spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['AHv'] = dict(description='Higher-order spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['λv']  = dict(description='Spin-spin energy',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['λDv'] = dict(description='Spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['λHv'] = dict(description='Higher-order spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['γv']  = dict(description='Spin-rotation energy' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['γDv'] = dict(description='Spin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['γHv'] = dict(description='Higher-orders pin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['ov']  = dict(description='Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['oDv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['oHv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['oLv']  = dict(description='Ligher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['pv']  = dict(description='Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['pDv']  = dict(description='Higher-order Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['qv']  = dict(description='Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',infer=[])
prototypes['qDv']  = dict(description='Higher-order Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',infer=[])

prototypes['Γv'] = dict(description="Total electronic-vibrational linewidth",units="cm-1 FWHM", kind='f',  fmt='<10.5g', strictly_positive=True, infer=[(('τ',),lambda self,τ: 5.309e-12/τ,)]) # tau=1/2/pi/gamma/c
prototypes['τv'] = dict(description="Total electronic-vibrational decay lifetime",units="s", kind='f',  fmt='<10.5g', infer=[(('Γv',), lambda self,Γv: 5.309e-12/Γv),( ('Atv',), lambda self,Atv: 1/Atv,)]) 
prototypes['Atv'] = dict(description="Total electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('τv',), lambda self,τv: 1/τv), (('Adv','Ave'), lambda self,Adv,Aev: Adv+Aev,), (('Aev',), lambda self,Aev: Aev),( ('Adv',), lambda self,Adv: Adv,)])# Test for Ad and Ae, if failed then one or the other is undefined/zero
prototypes['Adv'] = dict(description="Nonradiative electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('At','Ae'), lambda self,At,Ae: At-Ae,)])
prototypes['Aev'] = dict(description="Radiative electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('At','Ad'), lambda self,At,Ad: At-Ad,)])
prototypes['ηdv'] = dict(description="Fractional probability of electronic-vibrational level decaying nonradiatively",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ad'),lambda self,At,Ad:Ad/A,)])
prototypes['ηev'] = dict(description="Fractional probability of electronic-vibrational level decaying radiatively",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ae'),lambda self,At,Ae:Ae/A,)])

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
prototypes['Tvreduced'] = dict(description="Vibrational term value reduced by a polynomial in v+1/2",units='cm-1', kind='f',  fmt='<11.4f',
    infer=[(('self','reduced_quantum_number','reduced_polynomial_order','Tv','dTv'), _vibrationally_reduce), # dTv is known -- use in a weighted mean
           (('self','reduced_quantum_number','reduced_polynomial_order','Tv'), _vibrationally_reduce,)]) # dTv is not known
prototypes['Tvreduced_common'] = dict(description="Term values reduced by a common polynomial in v+1/2",units='cm-1', kind='f',  fmt='<11.4f', infer=[(('v','Tv','Tvreduced_common_polynomial'), lambda self,v,Tv,Tvreduced_common_polynomial: Tv-np.polyval(Tvreduced_common_polynomial,v+0.5)),(('v','Tv'), lambda self,v,Tv: Tv-np.polyval(np.polyfit(v+0.5,Tv,3),v+0.5),)])
prototypes['Tvreduced_common_polynomial'] = dict(description="Polynomial in terms of v+1/2 to reduce all term values commonly",units='cm-1', kind='o', infer=[])
prototypes['Bv_μscaled']  = dict(description='Rotational constant scaled by reduced mass to an isotopologue-independent value' ,units='cm-1', kind='f',fmt='0.8f', infer=[(('Bv','reduced_mass'),lambda self,Bv,reduced_mass: Bv*reduced_mass,)])

def _collect_prototypes(*keys):
    retval = {key:prototypes[key] for key in keys}
    return retval

class Base(Dataset):
    """Common stuff for for lines and levels."""
    default_prototypes = _collect_prototypes()
    default_attributes = Dataset.default_attributes | {'Zsource':None,'Eref':0.0,}

    def __init__(self,*args,**kwargs):
        kwargs.setdefault('permit_nonprototyped_data',False)
        Dataset.__init__(self,*args,**kwargs)

    # @optimise_method(format_lines='single')
    # def set_by_name(self,name,_cache=None,**parameters):
        # """Set parameters to all data matching the quantum numbers
        # encoded in name."""
        # if len(_cache) == 0:
            # _cache['i'] = self.match(self.decode_qn(name))
        # i = _cache['i']
        # for key,val in parameters.items():
            # self.set(key,val,index=i)

    @optimise_method(format_lines='single')
    def set_by_qn(self,encoded_qn=None,_cache=None,**defining_qn_and_parameters):
        """Set parameters to all data matching quantum numbers."""
        if self._clean_construct:
            qn,p = {},{}
            for key,val in defining_qn_and_parameters.items():
                if key in self.defining_qn:
                    qn[key] = val
                else:
                    p[key] = val
            if encoded_qn is not None:
                qn = qn | self.decode_qn(encoded_qn)
            i = self.match(qn)
            _cache['p'],_cache['i'] = p,i
        p,i = _cache['p'],_cache['i']
        for key,val in p.items():
            self.set(key,val,index=i)

    # @optimise_method(add_construct_function=False,add_format_input_function=True,format_lines='single',execute_now=True)
    # def set_by_qn(self,**kwargs):
        # """Set some data to fixed values or optimised parameters, limiting
        # setting to matching defining quantum numbers, all given as key word
        # arguments."""
        # ## collect quantum numbers and set data
        # qn,p = {},{}
        # for key,val in kwargs.items():
            # if key in self.defining_qn:
                # qn[key] = val
            # else:
                # p[key] = val
        # ## set data
        # for key,val in p.items():
            # self.set_parameter(key,val,match=qn)
            # self.pop_format_input_function()

    def assert_unique_qn(self,verbose=False):
        t,i,c = np.unique(self['_qnhash'],return_index=True,return_counts=True)
        if len(i) < len(self):
            j = [ti for ti,tc in zip(i,c) if tc > 1]
            if verbose or self.verbose:
                print('\nNon-unique levels:\n')
                print(self[j])
                print()
            raise Exception(f"There are {len(j)} sets of quantum numbers that are repeated (set verbose=True to print).")

        

    def sort(self,*sort_keys,reverse_order=False):
        """Overload sort to include automatic keys."""
        if len(sort_keys) == 0:
            sort_keys = [key for key in self.defining_qn if self.is_known(key)]
        Dataset.sort(self,*sort_keys,reverse_order=reverse_order)

    def match(self,keys_vals=None,**kwargs):
        """Overload Dataset.match to handle 'encoded_qn'."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwargs 
        if 'encoded_qn' in keys_vals:
            for key,val in self.decode_qn(keys_vals.pop('encoded_qn')).items():
                keys_vals.setdefault(key,val)
        return Dataset.match(self,keys_vals)

class Generic(Base):
    """A generic level."""
    default_prototypes = _collect_prototypes(
        'reference','_qnhash',
        'species','chemical_species',
        'label',
        'point_group',
        'E','Ee','ZPE','Ereduced','Ereduced_common','Eref','Eres',
        'Γ','ΓD','Γref','Γres',
        'J','N','S',
        'g','gnuclear','Inuclear',
        'Teq','Tex','Z','α',
        'Nself',
    )
    defining_qn = ('species','label','ef','J')
    default_xkey = 'J'
    default_zkeys = ('species','label','ef')
    encode_qn = lambda self,qn: quantum_numbers.decode_linear_level(qn)
    decode_qn = lambda self,name: quantum_numbers.encode_linear_level(name)
    default_zlabel_format_function = encode_qn
    
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
        'Λ','s','Σ','SR','Ω','Fi','LSsign',
        'i','gu','σv','sa','ef',
    )
    defining_qn = ('species','label','ef','J')
    default_zkeys = ('species','label','ef','Σ')
    encode_qn = lambda self,qn: quantum_numbers.encode_linear_level(qn)
    decode_qn = lambda self,name: quantum_numbers.decode_linear_level(name)
    default_zlabel_format_function = encode_qn

class LinearTriatomic(Linear):
    """A generic level."""
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'ν1','ν2','ν3','l2',
    )
    defining_qn = ('species','label','ef','ν1','ν2','ν3','l2','J')
    defining_zkeys = ('species','label','ef','ν1','ν2','ν3','l2')

class Diatomic(Linear):
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'v',
        'Γv','τv','Atv','Adv','Aev',
        'ηdv','ηev',
        'Tv','Bv','Dv','Hv','Lv','Mv',
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

