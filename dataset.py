import numpy as np
from numpy import nan,array

class Data:
    """A scalar or array value, possibly with an uncertainty."""

    _kind_defaults = {
        'f': {'default_value':nan       ,'cast':float ,'fmt'   :'+12.8e'    ,'name':'float','description':'float' ,}                    ,
        'i': {'default_value':999999    ,'cast':int   ,'fmt'   :'+8d'       ,'name':'int'  ,'description':'int'   ,}                    ,
        'b': {'default_value':True      ,'cast':bool  ,'fmt'   :'g'         ,'name':'bool' ,'description':'bool'  ,}                    ,
        'U': {'default_value':''        ,'cast':str   ,'fmt'   :'<10s'      ,'name':'str'  ,'description':'str'   ,}                    ,
        'O': {'default_value':None      ,'cast':str   ,'fmt'   :None        ,'name':'object'  ,'description':'object'   ,}                    ,
    }
    
    def __init__(
            self,
            name=None,          # short string
            value=None,         # if it has an associated value stored in the type itself
            uncertainty=None,         # if it has an associated value stored in the type itself
            kind=None,
            description=None,   # long string
            default_value=None, # a scalar default value for missing data
            default_differentiation_stepsize=None, # if it requires a default value for missing data
            fmt=None,
    ):
        ## determine kind
        if kind is not None:
            self.kind = np.dtype(kind).kind
        elif value is not None:
            if np.isscalar(value):
                self.kind = np.dtype(type(value)).kind
            else:
                self.kind = np.dtype(type(value[0])).kind
        elif default_value is not None:
            self.kind = np.dtype(type(default_value)).kind
        else:
            self.kind = 'f'
        ## determine cast etc from args or kind
        d = self._kind_defaults[self.kind]
        self.name = (name if name is not None else d['name'])
        self.description = (description if description is not None else d['description'])
        self.fmt = (fmt if fmt is not None else d['fmt'])
        self.cast = d['cast']
        ## set value and uncertainty
        self.value = value
        self.uncertainty = uncertainty
        self.default_differentiation_stepsize = default_differentiation_stepsize


    is_scalar = property(lambda self:self._length is None)

    def _set_value(self,value,uncertainty=None):
        if value is None:
            self._value = None
            self._length = None
        elif np.isscalar(value):
            self._value = self.cast(value)
            self._length = None
        else:
            self._value = np.asarray(value,dtype=self.kind)
            self._length = len(value)
        self.uncertainty = None

    def _set_uncertainty(self,uncertainty):
        if uncertainty is None:
            self._uncertainty = None
        else:
            assert self.kind == 'f','Uncertainty only defined for float kind'
            if self.is_scalar:
                self._uncertainty = float(uncertainty)
            else:
                if np.isscalar(uncertainty):
                    self._uncertainty = np.full(len(self),uncertainty,dtype=float)
                else:
                    assert len(uncertainty)==len(self)
                    self._uncertainty = np.asarray(uncertainty,dtype=float)

    def _get_value(self,index=None):
        if self.is_scalar:
            assert index is None
            return(self._value)
        else:
            if index is None:
                return(self._value[:len(self)])
            else:
                return(self._value[:len(self)][index])

    def _get_uncertainty(self):
        if self._uncertainty is None:
            return(self._uncertainty)
        elif self.is_scalar:
            return(self._uncertainty)
        else:
            return(self._uncertainty[:len(self)])

    value = property(_get_value,_set_value)
    uncertainty = property(_get_uncertainty,_set_uncertainty)

    def __len__(self):
        assert not self.is_scalar, 'Scalar'
        return(self._length)

    def __str__(self):
        if self.uncertainty is None:
            return(str(self.value))
        else:
            return(str(self.value)+' ± '+str(self.uncertainty))

    def __iter__(self):
        for val in self.value:
            yield val

    # def index(self,index):
        # """Return a copy of self with indexed array data."""
        # assert not self.is_scalar
        # retval = copy(self)
        # retval.value = self.value[index]
        # if self.uncertainty is not None:
            # retval.uncertainty = self.uncertainty[index]
        # return(retval)

    def _set_length(self,new_length):
        assert not self.is_scalar
        old_length = self._length
        over_allocate_factor = 2
        if new_length>len(self.value):
            new_value = np.empty(int(new_length*over_allocate_factor),dtype=self.kind)
            new_value[:old_length] = self._value[:old_length]
            self._value = np.concatenate((
                self._value[:old_length],
                np.empty(int(new_length*over_allocate_factor-old_length),dtype=self.kind
                )))
            if self.uncertainty is not None:
                self._uncertainty = np.concatenate((
                    self._uncertainty[:old_length],
                    np.empty(int(new_length*over_allocate_factor-old_length),dtype=self.kind
                    )))
        self._length = new_length

    def make_array(self,length):
        assert self.is_scalar,'Already an array'
        if self.uncertainty is not None:
            self.uncertainty = np.full(length,self.uncertainty)
        self.value = np.full(length,self.value)

    def make_scalar(self):
        if self._length is None:
            return              # nothing to be done
        assert np.unique(self.value)
        value = self.value[0]
        if self.uncertainty is not None:
            assert np.unique(self.uncertainty)
            uncertainty = self.uncertainty[0]
        self.value = value
        self.uncertainty = uncertainty

    def asarray(self,length=None,return_uncertainty=False):
        if self.is_scalar:
            if return_uncertainty:
                return(np.full(length,self.value))
            else:
                return(np.full(length,self.value),np.full(length,self.uncertainty))
        else:
            assert length is None or length==len(self)
            if return_uncertainty:
                return(self.value,self.uncertainty)
            else:
                return(self.value)

    def append(self,value,uncertainty=None):
        assert not self.is_scalar,'Cannot append to scalar'
        if (self.uncertainty is None and uncertainty is not None):
            raise Exception('Existing value has uncertainty and appended value does not')
        if (self.uncertainty is not None and uncertainty is None):
            raise Exception('Append value has uncertainty and existing value does not')
        self._set_length(len(self)+1)
        self.value[-1] = value
        if uncertainty is not None:
            self.uncertainty[-1] = uncertainty

    def extend(self,value,uncertainty=None):
        assert not self.is_scalar,'Cannot extend scalar'
        if (self.uncertainty is None and uncertainty is not None):
            raise Exception('Existing value has uncertainty and appended value does not')
        if (self.uncertainty is not None and uncertainty is None):
            raise Exception('Append value has uncertainty and existing value does not')
        old_length = len(self)
        self._set_length(len(self)+len(value))
        self.value[old_length:] = value
        if uncertainty is not None:
            self.uncertainty[old_length:] = uncertainty

            
class Dataset():

    """A collection of scalar or array values, possibly with uncertainties."""

    def __init__(self,**keys_vals):
        self._data = dict()
        self._infer_functions = dict()
        self._inferences = dict()
        self._inferred_from = dict()
        self._length = None
        for key,val in keys_vals.items():
            self.set(key,val)

    def __len__(self):
        assert self._length is not None,'Dataset has no length because all data is scalar'
        return(self._length)

    

    def __setitem__(self,key,value):
        self.set(key,value)

    def set(self,key,value,uncertainty=None):
        """Set a value and possibly its uncertainty."""
        self._data[key] = Data(name=key,value=value,uncertainty=uncertainty)
        if not self._data[key].is_scalar:
            if self._length == None:
                ## first array data, use this to define the length of self
                self._length = len(self._data[key])
            else:
                assert len(self._data[key])==len(self),'Length does not match existing data.'
        ## since this has been set/reset, clean up the record of what
        ## is inferred from what
        if key in self._inferred_from:
            self._inferred_from.pop(key)
        if key in self._inferences:
            for inferred_key in self._inferences[key]:
                self._data.pop(inferred_key)

    def get_value(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].value)

    def get_uncertainty(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].uncertainty)

    def add_infer_function(
            self,
            key,
            dependencies,
            value_function,
            uncertainty_function=None,
    ):
        if key not in self._infer_functions:
            self._infer_functions[key] = []
        self._infer_functions[key].append((
            tuple(dependencies),
            value_function,
            uncertainty_function))

    def index(self,index):
        """Index all array data in place."""
        for key in self:
            if not self.is_scalar(key):
                value = self.get_value(key)[index]
                uncertainty = self.get_uncertainty(key)
                if uncertainty is not None:
                    uncertainty = uncertainty[index]
                self._data[key].value = value
                self._data[key].uncertainty = uncertainty
                new_length = len(self._data[key])
        self._length = new_length

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if keys is None:
            keys = self.keys()
        retval = self.__class__()
        for key in keys:
            value = self.get_value(key)
            uncertainty = self.get_uncertainty(key)
            if not self.is_scalar(key) and index is not None:
                value = value[index]
                if uncertainty is not None:
                    uncertainty = uncertainty[index]
            retval.set(key,value,uncertainty)
        return(retval)

    def match(self,**keys_vals):
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            if np.isscalar(val):
                if val is np.nan:
                    i &= np.isnan(self[key])
                else:
                    i &= (self[key]==val)
            else:
                i &= np.any([
                    (np.isnan(self[key]) if vali is np.nan else self[key]==vali)
                            for vali in val],axis=0)
        return(i)

    def matches(self,**keys_vals):
        """Returns a copy reduced to matching values."""
        return(self.copy(index=self.match(**keys_vals)))

    def unique(self,key):
        """Return unique values of one key."""
        return(np.unique(self[key]))

    def unique_combinations(self,*keys):
        """Return a list of all unique combination of keys."""
        return(my.unique_combinations(*[self[key] for key in keys]))

    def unique_dicts(self,*keys):
        """Return an iterator where each element is a unique set of keys as a
        dictionary."""
        retval = [{key:val for key,val in zip(keys,vals)} for vals in self.unique_combinations(*keys)]
        retval = sorted(retval, key=lambda t: [t[key] for key in keys])
        return(retval)

    def unique_dicts_match(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a boolean array matching this
        combination."""
        if len(keys)==0:
            return((({},np.array([],dtype=bool)),))
        return([(d,self.match(**d)) for d in self.unique_dicts(*keys)])
 
    def unique_dicts_matches(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a copy of self reduced
        to matching values."""
        if len(keys)==0: return((({},self),)) # nothing to do
        return([(d,self.matches(**d)) for d in self.unique_dicts(*keys)])

    def __getitem__(self,arg):
        if isinstance(arg,str):
            return(self.get_value(arg))
        elif np.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            return(self.copy(keys=arg))
        else:
            return(self.copy(index=arg))

    def _infer(self,key,already_attempted=None):
        if key in self:
            return
        if already_attempted is None:
            already_attempted = []
        if key in already_attempted:
            raise InferException(f"Already unsuccessfully attempted to infer key: {repr(key)}")
        already_attempted.append(key) 
        ## Loop through possible methods of inferences.
        for dependencies,value_fcn,uncertainty_fcn in self._infer_functions[key]:
            try:
                for dependency in dependencies:
                    self._infer(dependency,copy(already_attempted)) # copy of already_attempted so it will not feed back here
                    already_attempted.append(dependency) # in case it comes up again at this level
                ## Compute key. There is trivial case of no vector to
                ## add to, and whether or not self is an argument.
                value = value_fcn(*[self.get_value(key) for key in dependencies])
                if uncertainty_fcn is not None:
                    uncertainty = uncertainty_fcn(
                        *[self.get_value(key) for key in dependencies],
                        *[self.get_uncertainty(key) for key in dependencies])
                else:
                    uncertainty = None
                self.set(key,value,uncertainty)
                ## if we get this far without an InferException then success!
                self._inferred_from[key] = dependencies
                for dependency in dependencies:
                    if dependency not in self._inferences:
                        self._inferences[dependency] = set()
                    self._inferences[dependency].add(key)
                break           
            ## some kind of InferException 
            except InferException as err:
                continue      
        ## complete failure to infer
        else:
            raise InferException(f"Could not infer key: {repr(key)}")

    def __iter__(self):
        for key in self._data:
            yield key

    def keys(self):
        return(list(self._data.keys()))

    def format_data(self,keys):
        header,columns = [],{}
        for key in keys:
            if self._data[key].is_scalar:
                if self.get_uncertainty(key) is None:
                    header.append(f'# {key} = {repr(self.get_value(key))}')
                else:
                    header.append(f'# {key} = {repr(self.get_value(key))} ± {repr(self.get_uncertainty(key))}')
            else:
                columns[key]  = self.get_value(key)
                if self.get_uncertainty(key) is not None:
                    columns['d'+key]  = self.get_uncertainty(key)
        retval = '\n'.join(header)
        if len(columns)>0:
            retval += my.format_columns(columns)
        return(retval)

    def __str__(self):
        return(self.format_data(self.keys()))

    def is_scalar(self,key=None):
        """Return boolean whether data for key is scalar or not. If key not
        provided return whether all data is scalara or not."""
        if key is None:
            if self._length is None:
                return(True)
            else:
                return(False)
        else:
            return(self._data[key].is_scalar)

    def make_array(self,key):
        return(self._data[key].make_array(len(self)))

    def make_scalar(self):
        for key in self:
            self._data[key].make_scalar()
        if all([self._data[key].is_scalar for key in self]):
            self._length = None

    def append(self,**new_keys_vals):
        """Append a single row of data from new scalar values."""
        new_dataset = self.__class__()
        for key in new_keys_vals:
            new_dataset[key] = [new_keys_vals[key]]
        self.concatenate(new_dataset)

    def extend(self,**new_keys_vals):
        """Extend data from input vector values (scalar values are
        broadcast)."""
        new_dataset = self.__class__()
        for key in new_keys_vals:
            new_dataset[key] = new_keys_vals[key]
        self.concatenate(new_dataset)

    def concatenate(self,new_dataset):
        """Concatenate data from another Dataset object to this one."""
        assert isinstance(new_dataset,Dataset)
        ## get keys to copy, and check fro problems
        all_keys = set(list(self.keys())+list(new_dataset.keys()))
        for key in copy(all_keys):
            if key not in self:
                raise Exception(f"Key not in existing dataset: {repr(key)}")
            if key not in new_dataset:
                raise Exception(f"Key not in concatenating dataset: {repr(key)}")
            if ((self.get_uncertainty(key) is None)
                is not (new_dataset.get_uncertainty(key) is None)):
                raise Exception(f"Uncertainty must be both set or not set in existing and concatenating dataset for key: {repr(key)}")
            if (self.is_scalar(key) and new_dataset.is_scalar(key) # both scalar
                and self.get_value(key) == new_dataset.get_value(key)                  # values match
                and my.equal_or_none(
                    self.get_uncertainty[key],
                    new_dataset.get_uncertainty[key])): # uncertainties match
                all_keys.remove(key)  # keep existing scalar value
        if new_dataset.is_scalar() or len(new_dataset)==0:
            ## nothing to add
            return
        elif self.is_scalar() or len(self)==0:
            ## currently no data, just add the new stuff
            for key in all_keys:
                self[key] = new_dataset[key]
                self._length = len(new_dataset)
        else:
            ## extend data key by key
            for key in all_keys:
                if self.is_scalar(key):
                    self.make_array(key)
                if new_dataset.is_scalar(key):
                    new_dataset.make_array(key)
                self._data[key].extend(
                    new_dataset.get_value(key),
                    new_dataset.get_uncertainty(key))
            self._length += len(new_dataset)


