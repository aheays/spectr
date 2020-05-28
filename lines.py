# from spectra.levels import Level
# from spectr import infer

import itertools
from pprint import pprint

from spectr.dataset import Dataset
from spectr.levels import Levels
from spectr.prototypes import prototypes
from spectr.infer_functions import infer_functions


class Lines(Dataset):
    """For now rotational lines."""

    other_keys = ['class', 'description','notes','author','reference','date',]
    data_keys = ['ν',]
    levels_class = Levels
    defining_quantum_numbers = []
    other_quantum_numbers = ['branch']

    def __init__(
            self,
            name=None,
            **keys_vals,
    ):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
            
        Dataset.__init__(self)
        ## 
        for key in itertools.chain(
                self.other_keys,
                self.defining_quantum_numbers,
                self.other_quantum_numbers,
                self.data_keys):
            self._prototypes[key] = prototypes[key]
        ## get level keys
        for key in (self.levels_class.other_keys
                    + self.levels_class.data_keys
                    + self.levels_class.defining_quantum_numbers
                    + self.levels_class.other_quantum_numbers):
            self._prototypes[key+'p'] = prototypes[key]
            self._prototypes[key+'pp'] = prototypes[key]
        ## set data
        self._infer_functions = infer_functions
        self.permit_nonprototyped_data = False
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val
