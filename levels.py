import itertools
import functools
from copy import copy,deepcopy
import re

from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import tools
from . import database
from .exceptions import InferException
from .levels_prototypes import prototypes


def _unique(x):
    """Take a list and return unique element.s"""
    return tools.unique(x,preserve_ordering= True)

class Base(Dataset):
    """Common stuff for for lines and levels."""

    _init_keys = ['description','notes','author','reference','date','classname',]
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}
    
    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs ca be
        scalar data, further default values of vector data, or if vetors
        themselves will populate data arrays."""
        if name is None:
            name = type(self).__name__
            name = re.sub(r'(.)([A-Z])',r'\1_\2',name).lower()
        Dataset.__init__(self,name=name)
        prototypes['classname']['infer'][()]: lambda: type(self).__name__
        # self['classname'] = type(self).__name__
        self.permit_nonprototyped_data = False
        self._cache = {}
        for key,val in keys_vals.items():
            self[key] = val
        self.pop_format_input_function()
        self.automatic_format_input_function(limit_to_args=('name',))


class GenericLevel(Base):
    """A generic level."""
    _init_keys = Base._init_keys + ['species','E','Eref','Γ','ΓD',]
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HeteronuclearDiatomicElectronicLevel(Base):
    _init_keys = GenericLevel._init_keys +['label', 'Λ','s','S','LSsign',]
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HeteronuclearDiatomicVibrationalLevel(Base):
    _init_keys =  HeteronuclearDiatomicElectronicLevel._init_keys + [
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
        'Bv_μscaled',]
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}


class HeteronuclearDiatomicRotationalLevel(Base):
    """Rotational levels of a heteronuclear diatomic molecule."""
    _init_keys = _unique(HeteronuclearDiatomicVibrationalLevel._init_keys + [
        'E','J','g','pm','Γ','N','S',
        'Teq','Tex','partition_source','partition','α','Nself',
        'σv','sa','ef','Fi','Ω','Σ','SR',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}
    default_zkeys = ('label','v','Σ','ef')

class HomonuclearDiatomicElectronicLevel(HeteronuclearDiatomicElectronicLevel):
    _init_keys = _unique(HeteronuclearDiatomicElectronicLevel._init_keys + ['Inuclear','gu',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HomonuclearDiatomicVibrationalLevel(HeteronuclearDiatomicVibrationalLevel):
    _init_keys = _unique(HeteronuclearDiatomicVibrationalLevel._init_keys + ['Inuclear','gu',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HomonuclearDiatomicRotationalLevel(HeteronuclearDiatomicRotationalLevel):
    _init_keys = _unique(HeteronuclearDiatomicRotationalLevel._init_keys + ['Inuclear','gu',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}


# # class TriatomicDinfh(Base):
    # # """Rotational levels of a triatomic molecule in the D∞h point group."""

    # # prototypes = deepcopy(Base.prototypes)
    # # prototypes.update({
        # # 'ν1':dict(description='Vibrational quantum number symmetric stretching' ,kind=int,fmt='<3d',infer={}),
        # # 'ν2':dict(description='Vibrational quantum number bending' ,kind=int,fmt='<3d',infer={}),
        # # 'ν3':dict(description='Vibrational quantum number asymmetric stretching' ,kind=int,fmt='<3d',infer={}),
        # # 'l' :dict(description='Quantum number' ,kind=str,fmt='<3',infer={}),
        # # })
