import itertools
from copy import copy,deepcopy
from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import tools
from .exceptions import InferException



################
## prototypes ##
################

prototypes = {
    'level_type':dict( description="Type of level.",kind=str ,infer={}) ,
    'description':dict( description="",kind=str ,infer={}) ,
    'notes' :dict(description="Notes regarding this line" , kind=str ,infer={}) ,
    'author' :dict(description="Author of data or printed file" ,kind=str ,infer={}) ,
    'reference' :dict(description="Published reference" ,kind=str ,infer={}) ,
    'date' :dict(description="Date data collected or printed" ,kind=str ,infer={}) ,
    'species' :dict(description="Chemical species" ,kind=str ,infer={}) ,
    'E' :dict(description="Level energy (cm-1)" ,kind=float ,fmt='<14.7f' ,infer={}) ,
    'J' :dict(description="Total angular momentum quantum number excluding nuclear spin" , kind=float,infer={}) ,
    'g' :dict(description="Level degeneracy including nuclear spin statistics" , kind=int , infer={}) ,
    'pm' :dict(description="Total inversion symmetry" ,kind=int ,infer={}) ,
    'Γ' :dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g', infer={('τ',):lambda τ: 5.309e-12/τ,}),
    'J' :dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,fmt='>0.1f',infer={}),
    'N' :dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR') : lambda J,SR: J-SR,}),
    'S' :dict(description="Total electronic spin quantum number", kind=float,infer={}),
    'Eref' :dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={() :lambda : 0.,}),
    'Teq':dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={}),
    'Tex':dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={'Teq':lambda Teq:Teq}),
    'partition_source':dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda : 'database',}),
    'partition':dict(description="Partition function.", kind=float, fmt='<11.3e', infer={}),
    'α':dict(description="State population", kind=float, fmt='<11.4e', infer={('partition','E','g','Tex'): lambda partition,E,g,Tex : g*np.exp(-E/(convert(constants.Boltzmann,'J','cm-1')*Tex))/partition,}),
    'Nself':dict(description="Column density (cm2)",kind=float,fmt='<11.3e', infer={}),
    'label' :dict(description="Label of electronic state", kind=str,infer={}),
    'v' :dict(description="Vibrational quantum number", kind=int,infer={}),
    'Λ' :dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={}),
    'LSsign' :dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={}),
    's' :dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={}),
    'σv' :dict(description="Symmetry with respect to σv reflection.", kind=int,infer={}),
    'gu' :dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind=int,infer={}),
    'sa' :dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={}),
    'ef' :dict(description="e/f symmetry", kind=int,infer={}),
    'Fi' :dict(description="Spin multiplet index", kind=int,infer={}),
    'Ω' :dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={}),
    'Σ' :dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={}),
    'SR' :dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={}),
    'Inuclear' :dict(description="Nuclear spin of individual nuclei.", kind=float,infer={}),
}

@tools.vectorise_arguments
def _f(S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef']['infer']['S','Λ','s'] = _f

@tools.vectorise_arguments
def _f(ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['σv']['infer']['ef','J'] = _f

@tools.vectorise_arguments
def _f(σv,gu):
    return σv*gu
prototypes['sa']['infer']['σv','gu'] = _f

def _f(level_type,J):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if level_type == 'DiatomicCinfv':
        return 2*J+1
    else:
        raise InferException('Only valid of DiatomicCinfv')
prototypes['g']['infer']['level_type','J'] = _f

@tools.vectorise_function
def _f(level_type,J,Inuclear,sa):
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
prototypes['g']['infer']['level_type','J','Inuclear','sa'] = _f


###################
## Level classes ##
###################

class _BaseLinesLevels(Dataset):
    """Init for Lines and Levels"""

    prototypes = {key:copy(prototypes[key]) for key in ['description','notes','author','reference','date','level_type',]}
    
    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs ca be
        scalar data, further default values of vector data, or if vetors
        themselves will populate data arrays."""
        if name is None:
            name = type(self).__name__
            name = name[0].lower()+name[1:]
        Dataset.__init__(self,name=name)
        self['level_type'] = type(self).__name__
        self.permit_nonprototyped_data = False
        self._cache = {}
        for key,val in keys_vals.items():
            self[key] = val
        self.pop_format_input_function()
        self.automatic_format_input_function(limit_to_args=('name',))

class Base(_BaseLinesLevels):

    prototypes = copy(_BaseLinesLevels.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
        'species','E','J','g','pm','Γ','J','N','S','Eref','Teq','Tex','partition_source','partition','α','Nself',]})
        
    def _f5(partition_source,species,Tex):
        if partition_source!='HITRAN':
            raise InferException(f'Partition source not "HITRAN".')
        return hitran.get_partition_function(species,Tex)

class DiatomicCinfv(Base):
    """A generic level."""
    
    prototypes = copy(Base.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
        'label','v','Λ','LSsign','s','σv','sa','ef','Fi','Ω','Σ','SR',
        'Eref','Teq','Tex','partition_source','partition','α','Nself',]})

class DiatomicDinfh(Base):
    """A generic level."""
    
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
    """A generic level."""
    
    prototypes = deepcopy(Base.prototypes)
    prototypes.update({
        'ν1':dict(description='Vibrational quantum number symmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'ν2':dict(description='Vibrational quantum number bending' ,kind=int,fmt='<3d',infer={}),
        'ν3':dict(description='Vibrational quantum number asymmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'l' :dict(description='Quantum number' ,kind=str,fmt='<3',infer={}),
        })



