import itertools
import functools
from copy import copy,deepcopy

from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import tools
from .exceptions import InferException

###########################################
## prototypes of data defined for levels ##
###########################################

prototypes = {}

prototypes['classname'] = dict( description="Type of levels of lines object.",kind=str ,infer={})
prototypes['description'] = dict( description="",kind=str ,infer={})
prototypes['notes'] = dict(description="Notes regarding this line" , kind=str ,infer={})
prototypes['author'] = dict(description="Author of data or printed file" ,kind=str ,infer={})
prototypes['reference'] = dict(description="Published reference" ,kind=str ,infer={})
prototypes['date'] = dict(description="Date data collected or printed" ,kind=str ,infer={})
prototypes['species'] = dict(description="Chemical species" ,kind=str ,infer={})
prototypes['mass'] = dict(description="Mass (amu)",kind=float, fmt='<11.4f', infer={('species',): lambda species: database.get_mass(species),})
prototypes['reduced_mass'] = dict(description="Reduced mass (amu)", kind=float, fmt='<11.4f', infer={('species','database',): lambda species: _get_species_property(species,'reduced_mass')})
prototypes['E'] = dict(description="Level energy (cm-1)" ,kind=float ,fmt='<14.7f' ,infer={})
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind=float,infer={})

def _f0(classname,J):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if classname == 'DiatomicCinfv':
        return 2*J+1
    else:
        raise InferException('Only valid of DiatomicCinfv')

@tools.vectorise_function
@functools.lru_cache
def _f1(classname,J,Inuclear,sa):
    """Calculate homonuclear diatomic molecule level degeneracy."""
    ## get total number of even or odd exchange combinations
    ntotal = (2*Inuclear+1)**2
    neven = 2*Inuclear+1 + (ntotal-(2*Inuclear+1))/2
    nodd = ntotal - neven
    if Inuclear%1==0:
        ## fermion
        if sa==+1:
            return (2*J+1)*neven
        else:
            return (2*J+1)*nodd
    else:
        ## boson
        if sa==+1:
            return (2*J+1)*nodd
        else:
            return (2*J+1)*neven

prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind=int , infer={('classname','J'):_f0, ('classname','J','Inuclear','sa'):_f1,})

prototypes['pm'] = dict(description="Total inversion symmetry" ,kind=int ,infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g', infer={('A',):lambda τ: 5.309e-12*A,})
prototypes['τ'] = dict(description="Total decay lifetime (s)", kind=float, infer={ ('A',): lambda A: 1/A,})       
prototypes['A'] = dict(description="Total decay rate (s-1)", kind=float, infer={('Γ',): lambda Γ: Γ/5.309e-12,})
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,fmt='>0.1f',infer={})
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR') : lambda J,SR: J-SR,})
prototypes['S'] = dict(description="Total electronic spin quantum number", kind=float,infer={})
prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={() :lambda : 0.,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={'Teq':lambda Teq:Teq})
prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda : 'database',})

@tools.vectorise_function_in_chunks()
def _f5(partition_source,species,Tex):
    from . import hitran
    if partition_source!='HITRAN':
        raise InferException(f'Partition source not "HITRAN".')
    return hitran.get_partition_function(species,Tex)

prototypes['partition'] = dict(description="Partition function.", kind=float, fmt='<11.3e', infer={('partition_source','species','Tex'):_f5,})

prototypes['α'] = dict(description="State population", kind=float, fmt='<11.4e', infer={('partition','E','g','Tex'): lambda partition,E,g,Tex : g*np.exp(-E/(convert(constants.Boltzmann,'J','cm-1')*Tex))/partition,})
prototypes['Nself'] = dict(description="Column density (cm2)",kind=float,fmt='<11.3e', infer={})
prototypes['label'] = dict(description="Label of electronic state", kind=str,infer={})
prototypes['v'] = dict(description="Vibrational quantum number", kind=int,infer={})
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={})
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={})
prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={})

@tools.vectorise_arguments
def _f0(ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['σv'] = dict(description="Symmetry with respect to σv reflection.", kind=int,infer={('ef','J'):_f0,})

prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind=int,infer={})
prototypes['sa'] = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={('σv','gu'):lambda σv,gu: σv*gu,})

@tools.vectorise_arguments
def _f0(S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef'] = dict(description="e/f symmetry", kind=int,infer={('S','Λ','s'):_f0,})

prototypes['Fi'] = dict(description="Spin multiplet index", kind=int,infer={})
prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={})
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={})
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={})
prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind=float,infer={})


class _BaseLinesLevels(Dataset):
    """Common stuff for for lines and levels."""

    prototypes = {key:copy(prototypes[key]) for key in ['description','notes','author','reference','date','classname',]}
    
    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs ca be
        scalar data, further default values of vector data, or if vetors
        themselves will populate data arrays."""
        if name is None:
            name = type(self).__name__
            name = name[0].lower()+name[1:]
        Dataset.__init__(self,name=name)
        self['classname'] = type(self).__name__
        self.permit_nonprototyped_data = False
        self._cache = {}
        for key,val in keys_vals.items():
            self[key] = val
        self.pop_format_input_function()
        self.automatic_format_input_function(limit_to_args=('name',))

class Base(_BaseLinesLevels):
    """A generic level."""

    prototypes = copy(_BaseLinesLevels.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
        'species','E','J','g','pm','Γ','J','N','S','Eref','Teq','Tex','partition_source','partition','α','Nself',]})
        


class DiatomicCinfv(Base):
    """A heteronuclear diatomic molecule."""

    prototypes = copy(Base.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in ['label','v','Λ','LSsign','s','σv','sa','ef','Fi','Ω','Σ','SR',]})

class DiatomicDinfh(Base):
    """A homonuclear diatomic molecule."""
    
    prototypes = deepcopy(DiatomicCinfv.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in ['Inuclear','gu',]})

    def _f(J,Inuclear,sa):
        """Calculate nuclear spin degeneracy of a homonuclear diatomic molecule."""
        g = 2*J+1
        for groupi,Ii in my.unique_combinations(group,I):
            if groupi=='C∞v':
                pass
            elif groupi=='D∞h':
                ## get total number of even or odd exchange combinations
                ntotal = (2*Ii+1)**2
                neven = 2*Ii+1 + (ntotal-(2*Ii+1))/2
                nodd = ntotal - neven
                if Ii%1==0:
                    ## fermion
                    g[(group==groupi)&(I==Ii)&(sa==+1)] *= neven
                    g[(group==groupi)&(I==Ii)&(sa==-1)] *= nodd
                else:
                    ## boson
                    g[(group==groupi)&(I==Ii)&(sa==+1)] *= nodd
                    g[(group==groupi)&(I==Ii)&(sa==-1)] *= neven
            else:
                raise Exception(f"Not implemented for this group: {repr(groupi)}")
        return g



class TriatomicDinfh(Base):
    """Triatomic moleculein the D∞h point group."""
    
    prototypes = deepcopy(Base.prototypes)
    prototypes.update({
        'ν1':dict(description='Vibrational quantum number symmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'ν2':dict(description='Vibrational quantum number bending' ,kind=int,fmt='<3d',infer={}),
        'ν3':dict(description='Vibrational quantum number asymmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'l' :dict(description='Quantum number' ,kind=str,fmt='<3',infer={}),
        })



