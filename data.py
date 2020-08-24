import time
import numpy as np

class Data:
    """A scalar or array value, possibly with an uncertainty."""


    _kind_defaults = {
        'f': {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt'   :'+12.8e','description':'float' ,'step':1e-8,},
        'i': {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt'   :'d'     ,'description':'int'   ,'step':None,},
        'b': {'cast':lambda x:np.asarray(x,dtype=bool)  ,'fmt'   :'g'     ,'description':'bool'  ,'step':None,},
        'U': {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt'   :'s'     ,'description':'str'   ,'step':None,},
        'O': {'cast':lambda x:np.asarray(x,dtype=object),'fmt'   :''      ,'description':'object','step':None,},
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
            self.kind = np.dtype(type(value[0])).kind
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

    def _set_value(self,value):
        self._value = self.cast(value)
        self._length = len(self._value)

    def _get_value(self):
        return(self._value[:len(self)])

    value = property(_get_value,_set_value)

    def _set_uncertainty(self,uncertainty):
        if uncertainty is not None:
            assert self.kind == 'f'
            self._uncertainty = np.empty(self._value.shape,dtype=float)
            self._uncertainty[:len(self)] = uncertainty
        else:
            self._uncertainty = None

    def _get_uncertainty(self):
        return(self._uncertainty[:len(self)])

    uncertainty = property(_get_uncertainty,_set_uncertainty)

    def has_uncertainty(self):
        return(self._uncertainty is not None)


    def format_values(self):
        """Get a list of strings representing all values."""
        return [format(t,self.fmt) for t in self.value]

    def format_uncertainties(self):
        """Get a list of strings representing all uncertainties."""
        return [format(t,'0.2g') for t in self.uncertainty]

    def __str__(self):
        """Get a string representation of all data."""
        if self.has_uncertainty():
            return('\n'.join([f'{t0} ± {t1}' for t0,t1 in
                              zip(self.format_values(),self.format_uncertainties())]))
        else:
            return('\n'.join(self.format_values()))

    def __len__(self):
        return(self._length)

    def __iter__(self):
        if self.has_uncertainty():
            for value,uncertainty in zip(
                    self.value,self.uncertainty):
                yield value,uncertainty
        else:
            for value in self.value:
                yield value

    def _extend_length_if_necessary(self,new_length):
        """Change size of internal array to be big enough for new
        data."""
        old_length = self._length
        over_allocate_factor = 2
        if new_length>len(self._value):
            self._value = np.concatenate((
                self._value[:old_length],
                np.empty(int(new_length*over_allocate_factor-old_length),dtype=self.kind)))
            if self.has_uncertainty():
                self._uncertainty = np.concatenate((
                    self._uncertainty[:old_length],
                    np.empty(int(new_length*over_allocate_factor-old_length),dtype=self.kind)))
        self._length = new_length

    # def make_scalar(self):
        # assert not self.is_scalar(),'Already scalar data.'
        # assert np.unique(self.get_value()),'Non-unique data, cannot make scalar.'
        # assert not self.has_uncertainty() or np.unique(self.get_uncertainty()),'Non-unique uncertainty, cannot make scalar.'
        # self.set(self.get_value()[0],
                 # (self.get_uncertainty()[0] if self.has_uncertainty() else None))

    def index(self,index):
        """Set self to index"""
        if self.has_uncertainty():
            self.value,self.uncertainty = self.value[index],self.uncertainty[index]
        else:
            self.value = self.value[index]

    def append(self,value,uncertainty=None):
        if (not self.has_uncertainty() and uncertainty is not None):
            raise Exception('Existing data has uncertainty and appended data does not')
        if (self.has_uncertainty() and uncertainty is None):
            raise Exception('Appended data has uncertainty and existing data does not')
        new_length = len(self)+1
        self._extend_length_if_necessary(new_length)
        self._value[new_length-1] = value
        if self.has_uncertainty():
            self._uncertainty[new_length-1] = uncertainty

    def extend(self,value,uncertainty=None):
        if (not self.has_uncertainty() and uncertainty is not None):
            raise Exception('Existing data has uncertainty and extending data does not')
        if (self.has_uncertainty() and uncertainty is None):
            raise Exception('Extending data has uncertainty and existing data does not')
        old_length = len(self)
        new_length = len(self)+len(value)
        self._extend_length_if_necessary(new_length)
        self._value[old_length:new_length] = value
        if uncertainty is not None:
            self._uncertainty[old_length:new_length] = uncertainty

            
