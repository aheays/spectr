import itertools
from copy import copy,deepcopy

from spectr.dataset import DataSet
from spectr.data_prototypes import prototypes

class Levels(DataSet):
    """A generic level."""
    
    _prototypes = {key:prototypes[key] for key in (
        'class', 'description', 'notes',
        'author', 'reference', 'date',
        'species',
        'J','S','N',
        'g', 'pm',  
        'E','Γ',
        'partition_source',
    )}

    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        DataSet.__init__(self)
        self.permit_nonprototyped_data = False
        self['class'] = self.__class__.__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val



class HeteronuclearDiatomicLevels(Levels):
    """A generic level."""
    
    _prototypes = copy(Levels._prototypes)
    _prototypes.update({
        key:prototypes[key] for key in (
            'label', 'v',
            'Λ', 'Fi', 'Ω', 'Σ', 'SR',  's',
            'LSsign',
            'σv', 'gu', 'sa', 'ef', 'pm',
            'E','Γ',
            'Eref', 
        )})

    _prototypes['g']['infer']['J'] = lambda J:2*J+1


# class Cinfv(Levels):
    # """Rotational levels of a C∞v diatomic molecule"""

    # _prototypes = {key:prototypes[key] for key in (
        # 'class', 'description', 'notes',
        # 'author', 'reference', 'date',
        # 'species', 'label',
        # 'v', 'J', 'N', 'S',
        # 'Λ', 'Fi', 'Ω', 'Σ', 'SR',  's',
        # 'LSsign','g', 
        # 'σv', 'gu', 'sa', 'ef', 'pm',
        # 'E','Γ',
        # 'Eref', 'partition_source',
    # )}

    # ## additional infer functions
    # _prototypes['g']['infer']['J'] = lambda J:2*J+1


