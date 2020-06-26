import re
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from numpy import nan

from spectr import tools
from spectr.tools import AutoDict
from spectr.exceptions import InferException

class Data:
    """A scalar or array value, possibly with an uncertainty."""

    _kind_defaults = {
        'f': {'key':'float'        ,'cast':float     ,'fmt'   :'+12.8e','description':'float'        ,},
        'i': {'key':'int'          ,'cast':int       ,'fmt'   :'+8d'   ,'description':'int'          ,},
        'b': {'key':'bool'         ,'cast':bool      ,'fmt'   :'g'     ,'description':'bool'         ,},
        'U': {'key':'string'       ,'cast':str       ,'fmt'   :'<10s'  ,'description':'str'          ,},
        'S': {'key':'scalar_object','cast':lambda x:x,'fmt'   :None    ,'description':'scalar object',},
        'O': {'key':'vector_object','cast':lambda x:x,'fmt'   :None    ,'description':'vector object',},
    }

    def __init__(
            self,
            key=None,          # short string
            value=None,         # if it has an associated value stored in the type itself
            uncertainty=None,         # if it has an associated value stored in the type itself
            kind=None,
            description=None,   # long string
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
            ## if inferred kind is an int, but a uncertainty is given,
            ## then make kind a float instead
            if self.kind=='i' and uncertainty is not None:
                self.kind = 'f'
        else:
            self.kind = 'f'
        ## figure out scalar or vector object data
        if self.kind=='O':
            if value is None:
                self.kind = 'S'
            else:
                if tools.isiterable(value):
                    self.kind = 'O'
                else:
                    self.kind = 'S'
        ## determine cast etc from args or kind
        d = self._kind_defaults[self.kind]
        self.key = (key if key is not None else d['key'])
        self.description = (description if description is not None else d['description'])
        self.fmt = (fmt if fmt is not None else d['fmt'])
        self.cast = d['cast']
        ## set the data
        self.set(value,uncertainty)

    def set(self,value,uncertainty=None):
        """Set the value of this data and optionally its
        uncertainty."""
        if self.kind == 'S':
            ## special case scalar_object
            self._value = value
            self._length = None
            assert uncertainty is None,'Cannot assign uncertainty to scalar_object'
            self._uncertainty = None
        elif value is None:
            ## blank / empty 
            self._value = None
            self._length = None
            assert uncertainty is None,'Cannot assign uncertainty to when value is None'
            self._uncertainty = None
        elif np.ndim(value)==0:
            ## scalar data
            self._value = self.cast(value)
            self._length = None
            if uncertainty is not None:
                assert self.kind=='f','Uncertainty is only defined for float kind.'
                self._uncertainty = float(uncertainty)
            else:
                self._uncertainty = None
        else:
            ## vector data
            self._value = np.asarray(value,dtype=self.kind)
            self._length = len(value)
            if uncertainty is not None:
                assert self.kind=='f','Uncertainty is only defined for float kind.'
                if np.ndim(uncertainty)==0:
                    ## broadcast uncertainty
                    self._uncertainty = np.full(len(self),uncertainty,dtype=float)
                else:
                    assert len(uncertainty)==len(self)
                    self._uncertainty = np.asarray(uncertainty,dtype=float)
            else:
                self._uncertainty = None

    def get_value(self):
        if self.is_scalar():
            return(self._value)
        else:
            return(self._value[:self._length])

    def get_uncertainty(self):
        # assert self.has_uncertainty(),'Uncertainty not set.'
        if not self.has_uncertainty():
            return None
        if self.is_scalar():
            return(self._uncertainty)
        return(self._uncertainty[:self._length])

    def is_scalar(self):
        """True if data is scalar, else False."""
        return(self._length is None)

    def has_uncertainty(self):
        """If uncertainty is set or not."""
        return(self._uncertainty is not None)

    def __len__(self):
        assert not self.is_scalar(), 'Scalar data has no length.'
        return(self._length)

    def __str__(self):
        if self.has_uncertainty():
            return(str(self.get_value())+' ± '+str(self.get_uncertainty()))
        else:
            return(str(self.get_value()))

    def __iter__(self):
        if self.has_uncertainty():
            for value,uncertainty in zip(
                    self.get_value(),
                    self.get_uncertainty()):
                yield value,uncertainty
        else:
            for value in self.get_value():
                yield value

    def _extend_length_if_necessary(self,new_length):
        """Change size of internal array to be big enough for new
        data."""
        assert not self.is_scalar()
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

    def make_array(self,length):
        """Turn current scalar data intor array data of the given
        length."""
        assert self.is_scalar(),'Already an array'
        self.set(np.full(length,self._value),
                 (np.full(length,self._uncertainty)
                  if self.has_uncertainty() else None))

    def make_scalar(self):
        assert not self.is_scalar(),'Already scalar data.'
        assert np.unique(self.get_value()),'Non-unique data, cannot make scalar.'
        assert not self.has_uncertainty() or np.unique(self.get_uncertainty()),'Non-unique uncertainty, cannot make scalar.'
        self.set(self.get_value()[0],
                 (self.get_uncertainty()[0] if self.has_uncertainty() else None))

    def index(self,index):
        """Set self to index"""
        assert not self.is_scalar(), 'Cannot index scalar data.'
        if self.has_uncertainty():
            self.set(
                self.get_value()[index],
                self.get_uncertainty()[index])
        else:
            self.set(self.get_value()[index])

    def copy(self,index=None):
        """Make a deep copy of self, possibly indexing array data."""
        assert index is None or not self.is_scalar(), 'Cannot index scalar data.'
        retval = Data(
            key=self.key,
            kind=self.kind,
            description=self.description,
            # default_differentiation_stepsize=self.default_differentiation_stepsize,
            fmt=copy(self.fmt),
        )
        if self.has_uncertainty():
            if index is None:
                retval.set(self.get_value(),self.get_uncertainty())
            else:
                retval.set(self.get_value()[index],self.get_uncertainty()[index])
        else:
            if index is None:
                retval.set(self.get_value())
            else:
                retval.set(self.get_value()[index])
        return(retval)

    def append(self,value,uncertainty=None):
        assert not self.is_scalar(),'Cannot append to scalar'
        if (not self.has_uncertainty() and uncertainty is not None):
            raise Exception('Existing value has uncertainty and appended value does not')
        if (self.has_uncertainty() and uncertainty is None):
            raise Exception('Append value has uncertainty and existing value does not')
        new_length = len(self)+1
        self._extend_length_if_necessary(new_length)
        self._value[new_length-1] = value
        if self.has_uncertainty():
            self._uncertainty[new_length-1] = uncertainty

    def extend(self,value,uncertainty=None):
        assert not self.is_scalar(),'Cannot extend scalar'
        if (not self.has_uncertainty() and uncertainty is not None):
            raise Exception('Existing value has uncertainty and appended value does not')
        if (self.has_uncertainty() and uncertainty is None):
            raise Exception('Append value has uncertainty and existing value does not')
        old_length = len(self)
        new_length = len(self)+len(value)
        self._extend_length_if_necessary(new_length)
        self._value[old_length:new_length] = value
        if uncertainty is not None:
            self._uncertainty[old_length:new_length] = uncertainty

            

class Dataset():

    """A collection of scalar or array values, possibly with uncertainties."""

    def __init__(self, **keys_vals):
        self._data = dict()
        self._length = None
        if not hasattr(self,'_prototypes'):
            ## derived classes might set this in class definition, so
            ## do not overwrite here
            self._prototypes = {}
        self._inferences = AutoDict([])
        self._inferred_from = AutoDict([])
        self.permit_nonprototyped_data =  True
        for key,val in self._get_keys_values_uncertainties(**keys_vals).items():
            self.set(key,*val)
        self.verbose = True

    def __len__(self):
        assert self._length is not None,'Dataset has no length because all data is scalar'
        return(self._length)

    def _get_keys_values_uncertainties(self,**keys_vals):
        """Match keys for value and uncertainties, e.g., 'x' and 'dx'."""
        keys_values_uncertainties = {}
        keys = list(keys_vals.keys())
        while len(keys)>0:
            key = keys[0]
            if key[0] == 'd':
                assert key[1:] in keys,f'Uncertainty {repr(key)} in data but {repr(key[1:])} is not.'
                keys_values_uncertainties[key[1:]] = (keys_vals[key[1:]],keys_vals[key])
                keys.remove(key)
                keys.remove(key[1:])
            elif 'd'+key in keys:
                keys_values_uncertainties[key] = (keys_vals[key],keys_vals['d'+key])
                keys.remove(key)
                keys.remove('d'+key)
            else:
                keys_values_uncertainties[key] = (keys_vals[key],None)
                keys.remove(key)
        return(keys_values_uncertainties)

    def __setitem__(self,key,value):
        """Shortcut to set, cannot set uncertainty this way."""
        self.set(key,value)

    def set(self,key,value,uncertainty=None,**data_kwargs):
        """Set a value and possibly its uncertainty."""
        ## unset anything inferred from this key now that it has been
        ## changed
        for inferred_key in self._inferences[key]:
            self.unset(inferred_key)
        ## delete any record of inferences to do with this key
        self.unset_inferences(key)
        if not self.permit_nonprototyped_data and key not in self._prototypes:
            raise Exception(f'New data is not in prototypes: {repr(key)}')
        ## if not previously set then get perhaps get a prototype
        if key not in self and key in self._prototypes:
            for tkey,tval in self._prototypes[key].items():
                if tkey == 'infer':
                    continue # not a Data kwarg
                data_kwargs.setdefault(tkey,copy(tval))
        ## set the data
        self._data[key] = Data(key=key,value=value,uncertainty=uncertainty,**data_kwargs)
        if not self._data[key].is_scalar():
            if self.is_scalar():
                ## first array data, use this to define the length of self
                self._length = len(self._data[key])
            else:
                assert len(self._data[key])==len(self),'Length does not match existing data.'

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unset_inferences(key)
        self._data.pop(key)

    def unset_inferences(self,key):
        """Delete any record of inferences to or from this key."""
        for inferred_from_key in self._inferred_from[key]:
            self._inferences[inferred_from_key].remove(key)
            self._inferred_from[key].remove(inferred_from_key)
        for inferred_key in self._inferences[key]:
            self._inferred_from[inferred_key].remove(key)
            self._inferences[key].remove(inferred_key)

    def get_value(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].get_value())

    def get_uncertainty(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].get_uncertainty())

    def has_uncertainty(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].has_uncertainty())

    def add_prototype(self,key,infer=None,**Data_kwargs):
        if infer is None:
            infer = {}
        self._prototypes[key] = dict(infer=infer,**Data_kwargs)

    def add_infer_function(self,key,dependencies,value_function,uncertainty_function=None):
        if key not in self._prototypes:
            self.add_prototype(key)
        self._prototypes[key]['infer'][dependencies] = (value_function,uncertainty_function)

    def index(self,index):
        """Index all array data in place."""
        assert not self.is_scalar()
        for data in self._data.values():
            if not data.is_scalar():
                data.index(index)
        self._length = len(data) # take length from last processed

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if keys is None:
            keys = self.keys()
        retval = self.__class__() # new version of self
        for key in keys:
            if self.is_scalar(key):
                retval._data[key] = self._data[key]
            else:
                retval._data[key] = self._data[key].copy(index)
        return(retval)

    def match(self,**keys_vals):
        """Return boolean array of data matching all key==val."""
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            if np.ndim(val)==0:
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
        return(tools.unique_combinations(*[self[key] for key in keys]))

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
            return((({},ndarray([],dtype=bool)),))
        return([(d,self.match(**d)) for d in self.unique_dicts(*keys)])
 
    def unique_dicts_matches(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a copy of self reduced
        to matching values."""
        if len(keys)==0: return((({},self),)) # nothing to do
        return([(d,self.matches(**d)) for d in self.unique_dicts(*keys)])

    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            if arg[0]=='d':
                return(self.get_uncertainty(arg[1:]))
            else:
                return(self.get_value(arg))
        elif tools.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            return(self.copy(keys=arg))
        else:
            return(self.copy(index=arg))

    def _infer(self,key,already_attempted=None):
        """Get data, or try and compute it."""
        if key in self:
            return
        if already_attempted is None:
            already_attempted = []
        if key in already_attempted:
            raise InferException(f"Already unsuccessfully attempted to infer key: {repr(key)}")
        already_attempted.append(key) 
        ## Loop through possible methods of inferences.
        if key not in self._prototypes or 'infer' not in self._prototypes[key]:
                raise InferException(f"No infer functions for: {repr(key)}")
        for dependencies,value_function in self._prototypes[key]['infer'].items():
            ## sometimes dependencies end up as a string instead of a list of strings
            if isinstance(dependencies,str):
                dependencies = (dependencies,)
            ## may value_function might actually include uncertainty_function
            if tools.isiterable(value_function):
                value_function,uncertainty_function = value_function
            else:
                uncertainty_function = None
            if self.verbose:
                print(f'Attempting to infer {repr(key)} from {repr(dependencies)}')
            try:
                for dependency in dependencies:
                    self._infer(dependency,copy(already_attempted)) # copy of already_attempted so it will not feed back here
                    already_attempted.append(dependency) # in case it comes up again at this level
                ## Compute key. There is trivial case of no vector to
                ## add to, and whether or not self is an argument.
                value = value_function(*[self.get_value(key) for key in dependencies])

                ## compute uncertainties by linearisation

                ## find keys that are floats and have uncertainties
                ## compute finite-difference of them
                ## add into one uncertainty
                ## use explicit functin as override for speed?
                ## Use internal variable in Data for linearisation step size?

                if uncertainty_function is not None:
                    uncertainty = uncertainty_function(
                        *[self.get_value(key) for key in dependencies],
                        *[self.get_uncertainty(key) for key in dependencies])
                else:
                    uncertainty = None


                self.set(key,value,uncertainty)
                ## if we get this far without an InferException then
                ## success!.  Record inference dependencies.
                self._inferred_from[key].extend(dependencies)
                for dependency in dependencies:
                    self._inferences[dependency].append(key)
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

    def is_known(self,*keys):
        for key in keys:
            try:
                self.get_value(key)
            except InferException:
                return(False)
        return(True)

    def assert_known(self,*keys):
        for key in keys:
            self.get_value(key)

    def sort(self,first_key,*more_keys):
        """Sort rows according to key or keys."""
        if self.is_scalar() or len(self)==0:
            return
            
        i = np.argsort(self[first_key])
        for key in more_keys:
            i = i[np.argsort(self[key][i])]
        for key in self:
            if not self.is_scalar(key):
                self._data[key].index(i)

    def format(self,keys=None,comment='# ',delimiter=' '):
        if keys is None:
            keys = self.keys()
        header,columns = [],{}
        for key in keys:
            if self.is_scalar(key):
                if self.has_uncertainty(key):
                    header.append(f'# {key} = {repr(self.get_value(key))} ± {repr(self.get_uncertainty(key))}')
                else:
                    header.append(f'# {key} = {repr(self.get_value(key))}')
            else:
                columns[key]  = self.get_value(key)
                if self.get_uncertainty(key) is not None:
                    columns[self.uncertainty_prefix+key]  = self.get_uncertainty(key)
        retval = '\n'.join(header)
        if len(columns)>0:
            retval += '\n'+tools.format_columns(columns,delimiter=delimiter)
        return(retval)

    def __str__(self):
        return(self.format(self.keys()))

    def format_description(self):
        """Get a string listing data keys and descriptions."""
        return('\n'.join([
            f'# {data.key}: {data.description}'
            for data in self._data.values()]))
            
    def save(self,filename,keys=None,**format_kwargs,):
        """Save some or all data to a text file."""
        if keys is None:
            keys = self.keys()
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(
                filename,
                **{key:self[key] for key in keys},
                **{self.uncertainty_prefix+key:self.get_uncertainty(key) for key in keys if self.has_uncertainty(key)})
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            d = {key:self[key] for key in keys}
            d.update({self.uncertainty_prefix+key:self.get_uncertainty(key) for key in keys if self.has_uncertainty(key)})
            tools.dict_to_hdf5(filename,d)
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                format_kwargs.setdefault('delimiter',', ')
            elif re.match(r'.*\.rs',filename):
                format_kwargs.setdefault('delimiter',' ␞ ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))

    def load(self,filename,comment='#',delimiter=None):
        '''Load data from a text file in standard format generated by
        save_to_file.'''
        ## load common data in file header if a text file
        if re.match(r'.*\.(h5|hdf5)',filename):
            ## hdf5 archive
            data =  tools.hdf5_to_dict(filename)
        elif re.match(r'.*\.npz',filename):
            ## numpy npz archive.  get as scalar rather than
            ## zero-dimensional numpy array
            data = {}
            for key,val in np.load(filename).items():
                if val.ndim == 0:
                    val = val.item()
                data[key] = val
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                delimiter = ','
            elif re.match(r'.*\.rs',filename):
                delimiter = '␞'
            assert comment not in ['',' '], "Not implemented"
            filename = tools.expand_path(filename)
            data = {}
            ## load header
            with open(filename,'r') as fid:
                for line in fid:
                    ## test for end of header
                    if not re.match(r'^ *'+comment,line):
                        break
                    ## looking to match:  "# key = 'string'"
                    if r := re.match(r'^ *'+comment+f' *([^= ]+) *= *["\'](.+)["\'] *',line): 
                        data[r.group(1)] = r.group(2)
                    ## looking to match:  "# key = number"
                    elif r := re.match(r'^ *'+comment+f' *([^= ]+) *= *(.+) *',line): 
                        data[r.group(1)] = tools.string_to_number_if_possible(r.group(2))
            ## load array data
            data.update(tools.txt_to_dict(filename, delimiter=delimiter, comment_regexp=comment,))
        ## Set data in self. Match up uncerainties with data if both
        ## are present.
        keys = list(data)
        while len(keys)>0:
            key = keys[0]
            if key[0] == 'u':
                tkey = key[1:]
                assert tkey in keys,f'Uncertainy {repr(key)} in data but {repr(tkey)} is not.'
                self.set(tkey,data[tkey],data[key])
                keys.remove(tkey)
                keys.remove(key)
            elif 'u'+key in keys:
                self.set(key,data[key],data['u'+key])
                keys.remove(key)
                keys.remove('u'+key)
            else:
                self.set(key,data[key])
                keys.remove(key)

    def is_scalar(self,key=None):
        """Return boolean whether data for key is scalar or not. If key not
        provided return whether all data is scalara or not."""
        if key is None:
            return(self._length is None)
        else:
            return(self._data[key].is_scalar())

    def make_array(self,key):
        return(self._data[key].make_array(len(self)))

    def make_scalar(self):
        """Make array data that has a unique value scalar."""
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
        ## no existing data, just copy fom new
        if len(self.keys()) == 0:
            self._data = deepcopy(new_dataset._data)
            self._length = new_dataset._length
            return
        ## get keys to copy, and check know to both
        all_keys = set(list(self.keys())+list(new_dataset.keys()))
        self.assert_known(*all_keys)
        new_dataset.assert_known(*all_keys)
        ## do not do anything for keys that are both scalar and equal
        for key in copy(all_keys):
            if (self.is_scalar(key) and new_dataset.is_scalar(key) 
                and self.get_value(key) == new_dataset.get_value(key)                  # values match
                and self.get_uncertainty(key) == new_dataset.get_uncertainty(key)): # uncertainties match
                all_keys.remove(key)
        if len(all_keys)==0:
            return
        ## extend data key by keyi
        old_length = (1 if self.is_scalar() else len(self))
        new_length = (1 if new_dataset.is_scalar() else len(new_dataset))
        for key in all_keys:
            if self.is_scalar(key):
                self._data[key].make_array(old_length+new_length)
            self._data[key]._extend_length_if_necessary(new_length+old_length)
            self._data[key]._value[old_length:old_length+new_length] = new_dataset.get_value(key)
            if self.has_uncertainty(key):
                self._data[key]._uncertainty[old_length:old_length+new_length] = new_dataset.get_uncertainty(key)
        self._length = old_length+new_length

    def plot(
            self,
            xkey,
            ykeys,
            zkeys=(),
            fig=None,           # otherwise automatic
            ax=None,            # otherwise automatic
            ynewaxes=True,
            znewaxes=False,
            legend=True,
            zlabel_format_function=None, # accept key=val pairs, defaults to printing them
            plot_errorbars=True, # if uncertainty available
            xscale='linear',     # 'log' or 'linear'
            yscale='linear',     # 'log' or 'linear'
            show=True,
            **plot_kwargs,      # e.g. color, linestyle, label etc
    ):
        """Plot a few standard values for looking at. If ax is set then all
        keys will be printed on that axes, otherwise new ones will be appended
        to figure."""
        from matplotlib import pyplot as plt
        from spectr import plotting
        if len(self)==0:
            return
        if ax is not None:
            ynewaxes,znewaxes = False,False
            fig = ax.figure
        if fig is None:
            fig = plt.gcf()
            fig.clf()
        if xkey is None:
            assert 'index' not in self.keys()
            self['index'] = np.arange(len(self),dtype=int)
            xkey = 'index'
        zkeys = [t for t in tools.ensure_iterable(zkeys) if t not in ykeys and t!=xkey] # remove xkey and ykeys from zkeys
        ykeys = [key for key in tools.ensure_iterable(ykeys) if key not in [xkey]+zkeys]
        ymin = {}
        self.assert_known(xkey,*ykeys,*zkeys)
        for iy,ykey in enumerate(tools.ensure_iterable(ykeys)):
            ylabel = ykey
            for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                z.sort(xkey)
                if zlabel_format_function is None:
                    zlabel = tools.dict_to_kwargs(dz)
                else:
                    zlabel = zlabel_format_function(**dz)
                if ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz+len(zkeys)*iy,fig=fig)
                    color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = None
                    title = ylabel+' '+zlabel
                elif ynewaxes and not znewaxes:
                    ax = plotting.subplot(n=iy,fig=fig)
                    color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = (zlabel if len(zkeys)>0 else None) 
                    title = ylabel
                elif not ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz,fig=fig)
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = ylabel
                    title = zlabel
                elif not ynewaxes and not znewaxes:
                    ax = fig.gca()
                    color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(iy),plotting.newlinestyle(iy)
                    # color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = ylabel+' '+zlabel
                    title = None
                kwargs = copy(plot_kwargs)
                kwargs.setdefault('marker',marker)
                kwargs.setdefault('ls',linestyle)
                kwargs.setdefault('mew',1)
                kwargs.setdefault('markersize',7)
                kwargs.setdefault('color',color)
                kwargs.setdefault('mec',kwargs['color'])
                x = z[xkey]
                y = z.get_value(ykey)
                if label is not None:
                    kwargs.setdefault('label',label)
                if plot_errorbars and self.has_uncertainty(ykey):
                    ## plot errorbars
                    kwargs.setdefault('mfc','none')
                    dy = z.get_uncertainty(ykey)
                    ax.errorbar(x,y,dy,**kwargs)
                    ## plot zero/undefined uncertainty data as filled symbols
                    i = np.isnan(dy)|(dy==0)
                    if np.any(i):
                        kwargs['mfc'] = kwargs['color']
                        kwargs['fillstyle'] = 'full'
                        if 'ls' in kwargs:
                            kwargs['ls'] = ''
                        else:
                            kwargs['linestyle'] = ''
                        kwargs['label'] = None
                        ax.plot(z[xkey][i],z[ykey][i],**kwargs)
                else:
                    kwargs.setdefault('mfc',kwargs['color'])
                    kwargs.setdefault('fillstyle','full')
                    ax.plot(x,y,**kwargs)
                if title is not None: ax.set_title(title)
                if legend and 'label' in kwargs:
                    plotting.legend(fontsize='x-small')
                ax.set_xlabel(xkey)
                ax.grid(True,color='gray',zorder=-5)
                ax.set_yscale(yscale)
                ax.set_xscale(xscale)
        if show:
            plotting.show()
        return(fig)

        


# if __name__=='__main__':

    # t = Dataset(x=1,y=2)
    # t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    # print( t._infer_functions)
    # assert t['z'] == 3
    # # t.add_infer_function('w',('y','z'),lambda y,z:y*z)
    # # assert t['w'] == 6
    # # t = Dataset(x=[1,2,3],y=2)
    # # t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    # # assert list(t['z']) == [3,4,5]
    # # t = Dataset()
    # # t.set('x',1.,0.1)
    # # t.set('y',2.,0.5)
    # # t.add_infer_function('z',('x','y'),lambda x,y:x+y,lambda x,y,dx,dy:np.sqrt(dx**2+dy**2))
    # # assert t['z'] == 3
    # # assert t.get_uncertainty('z') == np.sqrt(0.1**2+0.5**2)
    # # t = Dataset()
    # # t.set('x',1.,0.1)
    # # t.set('y',2.,0.5)
    # # t.add_infer_function('z',('x','y'),lambda x,y:x+y,lambda x,y,dx,dy:np.sqrt(dx**2+dy**2))
    # # t['z']
    # # assert 'z' in t
    # # t.set('x',2.,0.2)
    # # assert 'z' not in t
    # # t['z']
    # # print( t._data['z']._inferred_from)
    # # t['z'] = 5
    # # t.set('x',2.,0.2)
    # # assert 'z' in t
