import itertools

from spectr.dataset import Dataset
from spectr.prototypes import prototypes
from spectr.infer_functions import infer_functions



class Levels(Dataset):
    """For now a rotational level."""

    other_keys = ['class',
                  'description','notes','author','reference','date',
    ]
    data_keys = [
        'E',
    ]
    defining_quantum_numbers = [
        'species',
        # 'label','v','Σ','ef','J'
    ]
    other_quantum_numbers = [
        # 'S','Λ','LSsign','s','gu','Ihomo', 'group','term_symbol'
        # 'sublevel','N','F','Ω','SR','sa','pm','g','σv','encoded'
    ]

    def __init__(self,name=None,**keys_vals,):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        Dataset.__init__(self)
        ## get all prototypes
        for key in itertools.chain(
                self.other_keys,
                self.defining_quantum_numbers,
                self.other_quantum_numbers,
                self.data_keys):
            self._prototypes[key] = prototypes[key]
        self._infer_functions = infer_functions
        self.permit_nonprototyped_data = False
        ## set data
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val
