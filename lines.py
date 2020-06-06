
import itertools
from copy import copy
from pprint import pprint

from spectr.dataset import Dataset
from spectr.levels import Levels

def expand_level_keys(level_class):
    retval = {}
    for key,val in level_class._prototypes.items():
        retval[key+'p'] = copy(val)
        retval[key+'pp'] = copy(val)
    return(retval)


class Lines(Dataset):
    """For now rotational lines."""

    _prototypes = {
        'class':{'description':"What kind of data this is.",'kind':'str',},
        'description':{'kind':str,'description':"",},
        'notes':{'description':"Notes regarding this line.", 'kind':str, },
        'author':{'description':"Author of data or printed file", 'kind':str, },
        'reference':{'description':"", 'kind':str, },
        'date':{'description':"Date data collected or printed", 'kind':str, },
        'branch':dict(description="Rotational branch ΔJ.Fp.Fpp.efp.efpp", dtype='8U', cast=str, fmt='<10s'),
        'ν':dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>13.6f', infer={}),
    }
    _prototypes.update(expand_level_keys(Levels))
    _prototypes['ν']['infer']['Ep','Epp'] = lambda Ep,Epp: Ep-Epp
    _prototypes['Epp']['infer']['Ep','ν'] = lambda Ep,ν: Ep-ν
    _prototypes['Ep']['infer']['Epp','ν'] = lambda Epp,ν: Epp+ν

    def __init__(
            self,
            name=None,
            **keys_vals,
    ):
        Dataset.__init__(self)
        self.permit_nonprototyped_data = False
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val
