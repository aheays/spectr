import itertools
import functools
from copy import copy,deepcopy

from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import tools
from . import database
from .exceptions import InferException
from .levels_prototypes import *


class BaseLinesLevels(Dataset):
    """Common stuff for for lines and levels."""

    prototypes = {key:copy(prototypes[key]) for key in [
        'description','notes','author','reference','date','classname',]}
    
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



parent = BaseLinesLevels
class HeteronuclearDiatomicElectronicLevel(parent):
    """A generic level."""

    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
        'species','label',
        'Λ','s','S','LSsign','Eref',
    ]})

parent = HeteronuclearDiatomicElectronicLevel
class HeteronuclearDiatomicVibrationalLevel(parent):
    """A generic level."""

    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
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
    ]})


parent = HeteronuclearDiatomicVibrationalLevel
class HeteronuclearDiatomicRotationalLevel(parent):
    """Rotational levels of a heteronuclear diatomic molecule."""

    default_zkeys = ('label','v','Σ','ef')

    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in [
        'E','J','g','pm','Γ','N','S',
        'Teq','Tex','partition_source','partition','α','Nself',
        'σv','sa','ef','Fi','Ω','Σ','SR',
    ]})

parent = HeteronuclearDiatomicElectronicLevel
class HomonuclearDiatomicElectronicLevel(parent):
    """A generic level."""
    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in ['Inuclear','gu',]})

parent = HeteronuclearDiatomicVibrationalLevel
class HomonuclearDiatomicVibrationalLevel(parent):
    """A generic level."""
    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in ['Inuclear','gu',]})

parent = HeteronuclearDiatomicRotationalLevel
class HomonuclearDiatomicRotationalLevel(parent):
    """A generic level."""
    prototypes = copy(parent.prototypes)
    prototypes.update(**{key:copy(prototypes[key]) for key in ['Inuclear','gu',]})


# class TriatomicDinfh(Base):
    # """Rotational levels of a triatomic molecule in the D∞h point group."""

    # prototypes = deepcopy(Base.prototypes)
    # prototypes.update({
        # 'ν1':dict(description='Vibrational quantum number symmetric stretching' ,kind=int,fmt='<3d',infer={}),
        # 'ν2':dict(description='Vibrational quantum number bending' ,kind=int,fmt='<3d',infer={}),
        # 'ν3':dict(description='Vibrational quantum number asymmetric stretching' ,kind=int,fmt='<3d',infer={}),
        # 'l' :dict(description='Quantum number' ,kind=str,fmt='<3',infer={}),
        # })


######################################
## convenient access by point group ##
######################################
        
rotational_level_by_point_group = {}
rotational_level_by_point_group['C∞v'] = HeteronuclearDiatomicRotationalLevel
rotational_level_by_point_group['D∞h'] = HomonuclearDiatomicRotationalLevel

vibrational_level_by_point_group = {}
vibrational_level_by_point_group['C∞v'] = HeteronuclearDiatomicVibrationalLevel
vibrational_level_by_point_group['D∞h'] = HomonuclearDiatomicVibrationalLevel
