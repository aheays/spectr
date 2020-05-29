import itertools

from spectr.dataset import Dataset
from spectr.prototypes import prototypes
from spectr.infer_functions import infer_functions



class Levels(Dataset):
    """For now a rotational level."""


    _prototypes = {
        'class':{'description':"What kind of data this is.",'kind':'str',},
        'description':{'kind':str,'description':"",},
        'notes':{'description':"Notes regarding this line.", 'kind':str, },
        'author':{'description':"Author of data or printed file", 'kind':str, },
        'reference':{'description':"", 'kind':str, },
        'date':{'description':"Date data collected or printed", 'kind':str, },
        'species':{'description':"Chemical species",},
        'E':dict(description="Level energy (cm-1)",kind=float,fmt='<14.7f',infer={}),
        # 'branch':dict(description="Rotational branch ΔJ.Fp.Fpp.efp.efpp", dtype='8U', cast=str, fmt='<10s'),
        # 'ν':dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>13.6f'),
    }

    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        Dataset.__init__(self)
        self.permit_nonprototyped_data = False
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val
