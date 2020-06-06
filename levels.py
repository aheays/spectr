import itertools
from copy import deepcopy

from spectr.dataset import Dataset

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
        'J':dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float),
        'g':dict(description="Level degeneracy including nuclear spin statistics.", kind=int,infer={}),

        'pm',
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



class Cinfv(Levels):
    """For a."""

    _prototypes = deepcopy(Levels._prototypes)
    _prototypes['g']['infer']['J'] = lambda J:2*J+1


    'label','v', 'S','Λ','LSsign','s','gu','Ihomo', 'group','term_symbol', 'Σ','ef','J',,
'reduced_mass',
                                 'Γv','τv','Atv','Adv','Aev',
                                 'ηdv','ηev',
                                 'Tv','Bv','Dv','Hv',
                                 'Av','ADv','AHv',
                                 'λv','λDv','λHv',
                                 'γv','γDv','γHv',
                                 'ov','oDv','oHv','oLv',
                                 'pv','qv',
                                 'pDv','qDv',
                                 'sublevel','N','F','Ω','SR','sa','g','σv',
                                 'T','Γ',
                                 'At','Ae','Ad','ηd',


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


