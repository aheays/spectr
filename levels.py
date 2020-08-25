import itertools
from copy import copy,deepcopy
from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
# from .data_prototypes import prototypes


class _BaseLinesLevels(Dataset):
    """Init for Lines and Levels"""
    
    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        if name is None:
            name = type(self).__name__
            name = name[0].lower()+name[1:]
        Dataset.__init__(self,name=name)
        self.permit_nonprototyped_data = False
        self._cache = {}
        for key,val in keys_vals.items():
            self[key] = val

        self.pop_format_input_function()
        self.add_format_input_function(lambda:f'{self.name} = {type(self).__name__}()')

class Base(_BaseLinesLevels):

    prototypes = {
        # 'class' :dict(description="Dataset subclass" ,kind='str' ,infer={}) ,
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
        'J' :dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,infer={}),
        'N' :dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR') : lambda J,SR: J-SR,}),
        'S' :dict(description="Total electronic spin quantum number", kind=float,infer={}),
        'Eref' :dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={() :lambda : 0.,}),
        'Teq':dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={}),
        'Tex':dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={'Teq':lambda Teq:Teq}),
        'Zsource':dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda : 'database',}),
        'Z':dict(description="Partition function.", kind=float, fmt='<11.3e', infer={}),
        'α':dict(description="State population", kind=float, fmt='<11.4e',
                          infer={('Z','E','g','Tex'): lambda Z,E,g,Tex : g*np.exp(-E/(convert(constants.Boltzmann,'J','cm-1')*Tex))/Z,}),
        'Nself':dict(description="Column density (cm2)",kind=float,fmt='<11.3e', infer={}),

    }

    def _f5(Zsource,species,Tex):
        if Zsource!='HITRAN':
            raise InferException(f'Partition source not "HITRAN".')
        return hitran.get_partition_function(species,Tex)


class DiatomicCinfv(Base):
    """A generic level."""
    
    prototypes = deepcopy(Base.prototypes)
    prototypes.update({
        'label' :dict(description="Label of electronic state", kind=str,infer={}),
        'v' :dict(description="Vibrational quantum number", kind=int,infer={}),
        'Λ' :dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={}),
        'LSsign' :dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={}),
        's' :dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={}),
        'σv' :dict(description="Symmetry with respect to σv reflection.", kind=int,infer={}),
        'gu' :dict(description="Gerade / ungerade symmetry.", kint=int,infer={}),
        'sa' :dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={}),
        'ef' :dict(description="e/f symmetry", kind=int,infer={}),
        'Fi' :dict(description="Spin multiplet index", kind=int,infer={}),
        'Ω' :dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={}),
        'Σ' :dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={}),
        'SR' :dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={}),
        })

    prototypes['g']['infer'] = {'J':lambda J:2*J+1,}

class TriatomicDinfh(Base):
    """A generic level."""
    
    prototypes = deepcopy(Base.prototypes)
    prototypes.update({
        'label' :dict(description="Label of electronic state", kind=str,infer={}),
        'ν1':dict(description='Vibrational quantum number symmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'ν2':dict(description='Vibrational quantum number bending' ,kind=int,fmt='<3d',infer={}),
        'ν3':dict(description='Vibrational quantum number asymmetric stretching' ,kind=int,fmt='<3d',infer={}),
        'l':dict(description='Quantum number' ,kind=str,fmt='<3',infer={}),
        })


