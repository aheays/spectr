# import re
# from copy import copy,deepcopy
# from pprint import pprint

import numpy as np
# from numpy import nan

from spectr import tools
# from spectr.tools import AutoDict
# from spectr.exceptions import InferException


class Datum:
    """A scalar or array value, possibly with an uncertainty."""

    _kind_defaults = {
        'f': {'cast':float     ,'fmt'   :'+12.8e','description':'float'        ,'step':1e-8},
        'i': {'cast':int       ,'fmt'   :'+8d'   ,'description':'int'          ,'step':None},
        'b': {'cast':bool      ,'fmt'   :'g'     ,'description':'bool'         ,'step':None},
        'U': {'cast':str       ,'fmt'   :'<10s'  ,'description':'str'          ,'step':None},
        'O': {'cast':lambda x:x,'fmt'   :''      ,'description':'object'       ,'step':None},
    }

    def __init__(
            self,
            value,         # if it has an associated value stored in the type itself
            uncertainty=None,         # if it has an associated value stored in the type itself
            vary=None,
            step=None,
            kind=None,
            cast=None,
            description=None,   # long string
            units=None,
            fmt=None,
    ):
        if kind is not None:
            self.kind = np.dtype(kind).kind
        elif value is not None:
            self.kind = np.dtype(type(value)).kind
            if self.kind=='i' and uncertainty is not None:
                self.kind = 'f'
        else:
            self.kind = 'f'
        d = self._kind_defaults[self.kind]
        self.description = (description if description is not None else d['description'])
        self.fmt = (fmt if fmt is not None else d['fmt'])
        self.cast = (cast if cast is not None else d['cast'])
        self.step = (step if step is not None else d['step'])
        self.vary = vary
        self.units = units
        self.value = value
        self.uncertainty = uncertainty

    def set_value(self,value):
        self._value = self.cast(value)

    def get_value(self):
        return(self._value)

    value = property(get_value,set_value)

    def set_uncertainty(self,uncertainty):
        if uncertainty is not None:
            assert self.kind == 'f'
            self._uncertainty = float(uncertainty)
        else:
            self._uncertainty = None

    def get_uncertainty(self):
        return(self._uncertainty)

    uncertainty = property(get_uncertainty,set_uncertainty)

    def has_uncertainty(self):
        return(self._uncertainty is not None)

    def __str__(self):
        if self.has_uncertainty():
            return(format(self.value,self.fmt)+' ± '+format(self.uncertainty,'0.2g'))
        else:
            return(format(self.value,self.fmt))

    # def make_data(self,length):
        # """Turn current scalar data intor array data of the given
        # length."""
        # assert self.is_scalar(),'Already an array'
        # self.set(np.full(length,self._value),
                 # (np.full(length,self._uncertainty)
                  # if self.has_uncertainty() else None))

    # def deepcopy(self,index=None):
        # retval = Datu(
            # key=self.key,
            # kind=self.kind,
            # description=self.description,
            # # default_differentiation_stepsize=self.default_differentiation_stepsize,
            # fmt=copy(self.fmt),
        # )
        # if self.has_uncertainty():
            # if index is None:
                # retval.set(self.get_value(),self.get_uncertainty())
            # else:
                # retval.set(self.get_value()[index],self.get_uncertainty()[index])
        # else:
            # if index is None:
                # retval.set(self.get_value())
            # else:
                # retval.set(self.get_value()[index])
        # return(retval)

