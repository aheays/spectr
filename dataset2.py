import re
import time
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from numpy import nan

from . import tools
from .tools import AutoDict
from .exceptions import InferException
from . import optimise


class Data:
    """A scalar or array value, possibly with an uncertainty."""


    _kind_defaults = {
        'f': {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt'   :'+12.8e','description':'float' ,'missing':np.nan},
        'i': {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt'   :'d'     ,'description':'int'   ,'missing':-999},  
        'b': {'cast':lambda x:np.asarray(x,dtype=bool)  ,'fmt'   :''      ,'description':'bool'  ,'missing':False}, 
        'U': {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt'   :'s'     ,'description':'str'   ,'missing':''},    
        'O': {'cast':lambda x:np.asarray(x,dtype=object),'fmt'   :''      ,'description':'object','missing':None},  
    }

    def __init__(
            self,
            value,         # if it has an associated value stored in the type itself
            uncertainty=None,         # if it has an associated value stored in the type itself
            kind=None,
            cast=None,
            description=None,   # long string
            units=None,
            fmt=None,
            # missing=None,
    ):
        if kind is not None:
            self.kind = np.dtype(kind).kind
        elif value is not None and len(value)>0:
            self.kind = np.dtype(type(value[0])).kind
            if self.kind=='i' and uncertainty is not None:
                ## treat integers with uncertainties as floats
                self.kind = 'f'
            elif self.kind=='S':
                ## convert bytes string to unicodetreat integers with uncertainties as floats
                self.kind = 'U'
        else:
            self.kind = 'f'
        d = self._kind_defaults[self.kind]
        self.description = (description if description is not None else d['description'])
        self.fmt = (fmt if fmt is not None else d['fmt'])
        self.cast = (cast if cast is not None else d['cast'])
        # self.missing = (missing if missing is not None else d['missing'])
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

    def _change_dtype_if_necessary(self,new_value):
        """Changes datatype of existing data array if adding new_value will require it."""
        if self.kind == 'U':
            ## increase unicode dtype strength length if new strings
            ## are longer than the current dtype
            old_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self._value.dtype)))
            if np.isscalar(new_value):
                new_len = len(str(new_value))
            else:
                ## this is a really hacky way to get the length of string in a numpy array!!!
                new_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(new_value).dtype)))
            if new_len>old_len:
                ## reallocate array with new dtype with overallocation
                over_allocate_factor = 2
                t = np.empty(len(self._value),dtype=f'<U{new_len*over_allocate_factor}')
                t[:len(self)] = self._value[:len(self)]
                self._value = t

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
        self._change_dtype_if_necessary(value)
        self._extend_length_if_necessary(new_length)
        self._value[new_length-1] = value
        if self.has_uncertainty():
            self._uncertainty[new_length-1] = uncertainty

        ## check if the value is an optimise Parameter, if so set an
        ## update hook
        if isinstance(value,optimise.P):
            assert self.kind == 'f','Can onlh optimise float.'
            def _f(value):
                self._value[new_length-1] = value
            value.set_value_functions.append(_f)


    def extend(self,value,uncertainty=None):
        if (not self.has_uncertainty() and uncertainty is not None):
            raise Exception('Existing data has uncertainty and extending data does not')
        if (self.has_uncertainty() and uncertainty is None):
            raise Exception('Extending data has uncertainty and existing data does not')
        old_length = len(self)
        new_length = len(self)+len(value)
        self._change_dtype_if_necessary(value)
        self._extend_length_if_necessary(new_length)
        self._value[old_length:new_length] = value
        if uncertainty is not None:
            self._uncertainty[old_length:new_length] = uncertainty

            

class Dataset(optimise.Optimiser):

    """A collection of scalar or array values, possibly with uncertainties."""

    default_zkeys = []

    def __init__(
            self,
            name='dataset',
            load_from_filename=None,
            **kwargs):
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        self.add_format_input_function(lambda: 'not implemented')
        self._data = dict()
        self._length = 0
        if not hasattr(self,'prototypes'):
            ## derived classes might set this in class definition, so
            ## do not overwrite here
            self.prototypes = {}
        self._inferences = AutoDict([])
        self._inferred_from = AutoDict([])
        self._defaults = {}
        self.permit_nonprototyped_data =  True
        self.permit_reference_breaking = True
        # self.permit_missing = True # add missing data if required
        self.uncertainty_prefix = 'd_' # a single letter to prefix uncertainty keys
        self.verbose = False
        for key,val in kwargs.items():
            self[key] = val
        if load_from_filename is not None:
            self.load(load_from_filename)
            
    def __len__(self):
        return self._length

    def __setitem__(self,key,value):
        """Shortcut to set, cannot set uncertainty this way."""
        if (value_key:=self._get_key_without_uncertainty(key)) is not None:
            self.set_uncertainty(value_key,value)
        else:
            self.set(key,value)

    def set(
            self,
            key,
            value,
            uncertainty=None,
            **data_kwargs,
    ):
        """Set a value and possibly its uncertainty. """
        assert self.permit_nonprototyped_data or key in self.prototypes, f'New data is not in prototypes: {repr(key)}'
        # assert self.permit_reference_breaking or key not in self, f'Attempt to assign {key=} but {self.permit_reference_breaking=}'
        ## if a scalar value is given then set as default, and set
        ## data to this value
        if np.isscalar(value):
            self._defaults[key] = value
            self[key] = np.full(len(self),value)
            return
        ## if not previously set then get perhaps get a prototype
        if key not in self and key in self.prototypes:
            for tkey,tval in self.prototypes[key].items():
                if tkey == 'infer':
                    continue # not a Data kwarg
                data_kwargs.setdefault(tkey,copy(tval))
        
        ## If this is the data set other than defaults then add to set
        ## length of self and add corresponding data for any defaults
        ## set.
        if len(self) == 0 and len(value) > 0:
            self._length = len(value)
            for tkey,tvalue in self._defaults.items():
                print( len(self),tkey)
                self[tkey] = np.full(len(self),tvalue)
        else:
            assert len(value) == len(self),f'Length of new data {repr(key)} is {len(data)} and does not match the length of existing data: {len(self)}.'

        self._data[key] = Data(value=value,uncertainty=uncertainty,**data_kwargs)

        self.unset_inferences(key)

    # def set_missing(self,key,index=None):
        # """"""
        # if key not in self:
            # self[key] = 1.     
        # if self.is_scalar(key):
            # self._data[key].value = self._data[key].missing
        # else:
            # if index is None:
                # ## always use an index to prevent reference breaking
                # index = slice(0,len(self))
            # self._data[key].value[index] = self._data[key].missing
        
    def set_uncertainty(self,key,uncertainty):
        """Set a the uncertainty of an existing value."""
        # assert self.permit_reference_breaking or key not in self, f'Attemp to assign {key=} but {self.permit_reference_breaking=}'
        self.unset_inferences(key)
        assert key in self,f'Value must exist before setting uncertainty: {repr(key)}'
        self._data[key].uncertainty  = uncertainty

    def clear(self):
        for key in self.keys():
            self.unset(key)

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unset_inferences(key)
        if key in self._data:
            ## might already be gone if this is called recursively
            self._data.pop(key)

    def is_inferred_from(self,is_this_key,inferred_from_this_key):
        """Test if key is inferred from another."""
        return inferred_from_this_key in self._inferred_from[is_this_key]

    def unset_inferences(self,key):
        """Delete any record of inferences to or from this key and any data
        inferred from it."""
        for inferred_from_key in self._inferred_from[key]:
            self._inferences[inferred_from_key].remove(key)
            self._inferred_from[key].remove(inferred_from_key)
        for inferred_key in self._inferences[key]:
            self._inferred_from[inferred_key].remove(key)
            self._inferences[key].remove(inferred_key)
            self.unset(inferred_key)

    def get_value(self, key, index=None,):
        """Get value of data. Optionally index. """
        if key not in self._data:
            self._infer(key)
        if index is None:
            return self._data[key].value
        else:
            return self._data[key].value[index]

    def get_uncertainty(self,key,index=None):
        self.assert_known(key)
        if not self.has_uncertainty(key):
            return None
        elif index is None:
            return self._data[key].uncertainty
        else:
            return self._data[key].uncertainty[index]

    def get_unique_value(self,key,**matching_keys_vals):
        """Return value of key that is the uniquely matches
        matching_keys_vals."""
        i = tools.find(self.match(**matching_keys_vals))
        assert len(i)==1,f'Non-unique matches for {matching_keys_vals=}'
        return self.get_value(key,i)

    def has_uncertainty(self,key):
        self.assert_known(key)
        return(self._data[key].has_uncertainty())

    def add_prototype(self,key,infer=None,**data_kwargs):
        if infer is None:
            infer = {}
        self.prototypes[key] = dict(infer=infer,**data_kwargs)

    def add_infer_function(self,key,dependencies,function):
        if key not in self.prototypes:
            self.add_prototype(key)
        self.prototypes[key]['infer'][dependencies] = function

    def index(self,index):
        """Index all array data in place."""
        for data in self._data.values():
            data.index(index)
            self._length = len(data)

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if keys is None:
            keys = self.keys()
        retval = self.__class__() # new version of self
        for key in keys:
            if index is None:
                retval[key] = self[key]
            else:
                retval[key] = deepcopy(self[key][index])
        return retval

    def match(self,**keys_vals):
        """Return boolean array of data matching all key==val.\n\nIf key has
        suffix '_min' or '_max' then match anything greater/lesser
        or equal to this value"""
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            if len(key)>4 and key[-4:] == '_min':
                i &= (self[key[:-4]] >= val)
            elif len(key)>4 and key[-4:] == '_max':
                i &= (self[key[:-4]] <= val)
            elif np.ndim(val)==0:
                if val is np.nan:
                    i &= np.isnan(self[key])
                else:
                    i &= (self[key]==val)
            else:
                i &= np.any([
                    (np.isnan(self[key]) if vali is np.nan else self[key]==vali)
                            for vali in val],axis=0)
        return i

    def matches(self,**keys_vals):
        """Returns a copy reduced to matching values."""
        return(self.copy(index=self.match(**keys_vals)))

    def limit_to_match(self,**keys_vals):
        self.index(self.match(**keys_vals))

    def remove_match(self,**keys_vals):
        self.index(~self.match(**keys_vals))

    def unique(self,key):
        """Return unique values of one key."""
        return np.unique(self[key])

    def unique_combinations(self,*keys):
        """Return a list of all unique combination of keys."""
        return tools.unique_combinations(*[self[key] for key in keys])

    def unique_dicts(self,*keys):
        """Return an iterator where each element is a unique set of keys as a
        dictionary."""
        retval = [{key:val for key,val in zip(keys,vals)} for vals in self.unique_combinations(*keys)]
        retval = sorted(retval, key=lambda t: [t[key] for key in keys])
        return retval 

    def unique_dicts_match(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a boolean array matching this
        combination."""
        if len(keys)==0:
            return((({},ndarray([],dtype=bool)),))
        return [(d,self.match(**d)) for d in self.unique_dicts(*keys)]

    def unique_dicts_matches(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a copy of self reduced
        to matching values."""
        if len(keys)==0: return((({},self),)) # nothing to do
        return [(d,self.matches(**d)) for d in self.unique_dicts(*keys)]

    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            if (value_key:=self._get_key_without_uncertainty(arg)) is not None:
                return(self.get_uncertainty(value_key))
            else:
                return self.get_value(arg)
        elif tools.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            return(self.copy(keys=arg))
        else:
            return(self.copy(index=arg))

    def get_index(self,key,index):
        """Get key. Index by index if vector data, else return
        scalar. Somehow fold into __getitem__?"""
        return self[key][index]
        
    def _infer(self,key,already_attempted=None):
        """Get data, or try and compute it."""
        if key in self:
            return
        if already_attempted is None:
            already_attempted = []
        # print( key,already_attempted)
        if key in already_attempted:
            raise InferException(f"Already unsuccessfully attempted to infer key: {repr(key)}")
        already_attempted.append(key) 
        ## Loop through possible methods of inferences.
        if (key not in self.prototypes
            or 'infer' not in self.prototypes[key]
            or len(self.prototypes[key]['infer'])==0):
                raise InferException(f"No infer functions for: {repr(key)}")
        for dependencies,function in self.prototypes[key]['infer'].items():
            if isinstance(dependencies,str):
                ## sometimes dependencies end up as a string instead of a list of strings
                dependencies = (dependencies,)
            if self.verbose:
                print(f'Attempting to infer {repr(key)} from {repr(dependencies)}')
            try:
                for dependency in dependencies:
                    self._infer(dependency,copy(already_attempted)) # copy of already_attempted so it will not feed back here
                ## compute value if dependencies successfully inferred
                self[key] = function(self,*[self[dependency] for dependency in dependencies])
                ## compute uncertainties by linearisation
                squared_contribution = []
                value = self[key]
                parameters = [self[t] for t in dependencies]
                for i,dependency in enumerate(dependencies):
                    if self.has_uncertainty(dependency):
                        dparameters = deepcopy(parameters) # slow?
                        diffstep = 1e-10*self[dependency]
                        dparameters[i] += diffstep
                        dvalue = value - function(self,*dparameters)
                        data = self._data[dependency]
                        squared_contribution.append((data.uncertainty*dvalue/diffstep)**2)
                if len(squared_contribution)>0:
                    self.set_uncertainty(key,np.sqrt(sum(squared_contribution)))
                ## if we get this far without an InferException then
                ## success!.  Record inference dependencies.
                self._inferred_from[key].extend(dependencies)
                for dependency in dependencies:
                    self._inferences[dependency].append(key)
                break           
            ## some kind of InferException, try next set of dependencies
            except InferException as err:
                if self.verbose:
                    print('    InferException: '+str(err))
                continue      
        ## complete failure to infer
        else:
            raise InferException(f"Could not infer key: {repr(key)}")

    def __iter__(self):
        for key in self._data:
            yield key

    def as_dict(self,index=None):
        """Data in row index as a dict of scalars."""
        if index is None:
            return {key:self[key]for key in self}
        else:
            return {key:self[key][index] for key in self}
        
    def rows(self):
        """Iterate over data row by row, returns as a dictionary of
        scalar values."""
        for i in range(len(self)):
            yield(self.as_dict(i))

    def matching_row(self,**matching_keys_vals):
        """Return uniquely-matching row."""
        i = tools.find(self.match(**matching_keys_vals))
        if len(i)==0:
            raise Exception(f'No matching row found: {matching_keys_vals=}')
        if len(i)>1:
            raise Exception(f'Multiple matching rows found: {matching_keys_vals=}')
        return self.as_dict(i[0])

    def keys(self):
        return list(self._data.keys())

    def assert_known(self,*keys):
        for key in keys:
            self[key]

    def is_known(self,*keys):
        try:
            self.assert_known(*keys)
            return True 
        except InferException:
            return False

    def sort(self,*sort_keys):
        """Sort rows according to key or keys."""
        # if len(self)==0:
            # return
        i = np.argsort(self[sort_keys[0]])
        for key in sort_keys[1:]:
            i = i[np.argsort(self[key][i])]
        self.index(i)

    def format(self,keys=None,comment='# ',delimiter=' | '):
        """Format data into a string representation."""
        if keys is None:
            keys = self.keys()
        ## collect table data
        header,columns = [],[]
        for key in keys:
            if len(self)==0:
                continue
            ## two passes required on all data to align column
            ## widths
            vals = self._data[key].format_values()
            tkey = (comment+key if len(columns)==0 else key)
            width = str(max(len(tkey),np.max([len(t) for t in vals])))
            columns.append([format(tkey,width)]+[format(t,width) for t in vals])
            if self.get_uncertainty(key) is not None:
                vals = self._data[key].format_uncertainties()
                tkey = self.uncertainty_prefix + key
                width = str(max(len(tkey),np.max([len(t) for t in vals])))
                columns.append([format(tkey,width)]+[format(t,width) for t in vals])
        retval = ''
        if header != []:
            retval = '\n'.join(header)+'\n'
        if columns != []:
            retval += '\n'.join([delimiter.join(t) for t in zip(*columns)])+'\n'
        return retval

    def __str__(self):
        return self.format(self.keys())

    def format_description(self):
        """Get a string listing data keys and descriptions."""
        return '\n'.join([
            f'# {data.key}: {data.description}'
            for data in self._data.values()]) 

    def save(self,filename,keys=None,**format_kwargs,):
        """Save some or all data to a text file."""
        if keys is None:
            keys = self.keys()
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(
                filename,
                **{key:self[key] for key in keys},
                **{self.uncertainty_prefix+key:self.get_uncertainty(key)
                   for key in keys if self.has_uncertainty(key)})
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            d = {key:self[key] for key in keys}
            d.update({self.uncertainty_prefix+key:self.get_uncertainty(key)
                      for key in keys if self.has_uncertainty(key)})
            tools.dict_to_hdf5(filename,d)
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                format_kwargs.setdefault('delimiter',', ')
            elif re.match(r'.*\.rs',filename):
                format_kwargs.setdefault('delimiter',' ␞ ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))

    def load(
            self,
            filename,
            comment='#',
            delimiter=None,
            table_name=None,
            translate_keys=None, # from key in file to key in self, None for skip
            **set_keys_vals   # set this data after loading is done
    ):
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
        elif re.match(r'.*\.org',filename):
            data = tools.org_table_to_dict(filename,table_name)
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                delimiter = ','
            elif re.match(r'.*\.rs',filename):
                delimiter = '␞'
            elif re.match(r'.*\.psv',filename):
                delimiter = '|'
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
        ## translate keys
        if translate_keys is not None:
            for from_key,to_key in translate_keys.items():
                if from_key in data:
                    if to_key is None:
                        data.pop(from_key)
                    else:
                        data[to_key] = data.pop(from_key)
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
                self[key] = data[key]
                keys.remove(key)
        ## set data at the end
        for key,val in set_keys_vals.items():
            self[key] = val

    def append(self,**kwargs):
        """Append a single row of data from new scalar values."""
        self.extend(**{key:[val] for key,val in kwargs.items()})

    def extend(self,**kwargs):
        """Append a single row of data from new vector given as kwargs."""
        ## ensure enough data is provided
        for key in self:
            assert key in kwargs or key in self._defaults, f'Key not in extending data and has not default: {key=}'
        ## get length of new data
        newlen = 0
        for key,val in kwargs.items():
            if newlen == 0:
                newlen = len(val)
            else:
                assert newlen == len(val),f'Inconsistent length for {key=}'
        if newlen == 0:
            ## no data to extend
            return
        ## include defaults values if needed
        for key in self._defaults:
            if key not in kwargs:
                kwargs[key] = np.full(newlen,self._defaults[key])
        ## add data
        if len(self) == 0:
            ## currently no data -- set extending values
            for key,val in kwargs.items():
                self[key] = val
        else:
            for key,val in kwargs.items():
                assert key in self, f'Unknown extending key: {key=}'
                self._data[key].extend(val)
            self._length += newlen

    # def concatenate(self,new_dataset):
        # """Concatenate data from another Dataset object to this one. All
        # existing data in current Dataset must be known to the new
        # Dataset, but the reverse is not enforced.  If the existing
        # Dataset is scalar then its existing data is not vectorised
        # before concatenation."""
        # assert self.permit_reference_breaking, f'Attemp to assign {key=} but {self.permit_reference_breaking=}'
        # if len(new_dataset) == 0:
            # ## only concatenate if vector data present
            # return
        # if len(self) == 0:
            # ## just copy everything, keeping scalar data, but maybe
            # ## overwritten by new_dataset
            # for key in new_dataset:
                # self[key] = new_dataset[key]
            # return
        # ## enfore similarity
        # for key in self:
            # if not new_dataset.is_known(key):
                # if self.permit_missing:
                    # new_dataset.set_missing(key)
                # else:
                    # raise Exception(f"Key unknown to new_dataset: {key=}")
        # for key in new_dataset:
            # if not self.is_known(key):
                # if self.permit_missing:
                    # self.set_missing(key)
                # else:
                    # raise Exception(f"Key unknown to self: {key=}")
        # ## sort out scalar and vector keys
        # for key in self:
            # if self.is_scalar(key):
                # if key in new_dataset:
                    # if new_dataset.is_scalar(key):
                        # if new_dataset[key] == self[key]:
                            # ## matching scalar data in existing and new data -- UNCERTAINTIES!!!!
                            # continue
                        # else:
                            # ## non-matching data, make vector
                            # self.make_vector(key)
                            # new_dataset.make_vector(key)
                    # else:
                        # ## make all vector
                        # self.make_vector(key)
                # else:
                    # ## keep existing scalar data as scalar since its not in new data
                    # continue
            # else:
                # assert new_dataset.is_known(key), f"Key missing in new_dataset: {repr(key)}"
                # if new_dataset.is_scalar(key):
                    # ## make new data vector
                    # new_dataset.make_vector(key)
                # else:
                    # ## everything already vector
                    # pass
            # ## copy new_data to self
            # if self.is_scalar(key):
                # continue
            # self._data[key].extend(
                # new_dataset.get_value(key),
                # new_dataset.get_uncertainty(key),)
        # self._length = len(self) + len(new_dataset)

    def plot(
            self,
            xkey,
            ykeys,
            zkeys=None,
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
        if zkeys is None:
            zkeys = self.default_zkeys
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

    def _get_key_without_uncertainty(self,key):
        if len(key) <= len(self.uncertainty_prefix):
            return None
        elif key[:len(self.uncertainty_prefix)] != self.uncertainty_prefix:
            return None
        else:
            return key[len(self.uncertainty_prefix):]

    def optimise_value(
            self,
            match=None,
            **parameter_keys_vals
    ):
        parameters = self.add_parameter_set(**parameter_keys_vals)
        cache = {'first':True,}
        def f():
            i = (None if match is None else self.match(**match))
            ## set data
            for key,p in parameters.items():
                if i is None:
                    ## all values
                    if cache['first'] or self[key] != p.value:
                        self.set(key,p.value,p.uncertainty)
                else:
                    ## some indexed values
                    if cache['first'] or np.any(self[key][i] != p.value):
                        value,uncertainty = self.get_value(key),self.get_uncertainty(key)
                        value[i] = p.value
                        uncertainty[i] = p.uncertainty
                        self.set(key,value,uncertainty)
            cache['first'] = False 
        self.add_construct_function(f)
        def f():
            retval = f'{self.name}.optimise_value('
            retval += parameters.format_as_kwargs()
            if match is not None:
                retval =  ',match='+tools.dict_to_kwargs(match)
            return retval + ')'
        self.add_format_input_function(f)

    # def optimise_scale(
            # self,
            # key,                # to scale
            # scale=1,              # optimisable parameter
            # reset=False, # if True then reset to scale original value every iteration, otherwise scaling may be cumulative
            # match=None,
    # ):
        # """Scale (or optimise) values of key."""
        # p = self.add_parameter('scale_value', ensure_iterable(scale),)
        # cache = {}
        # def f():
            # if len(cache)==0:
                # cache['i'] = self.match(**limit_to_matches)
                # if reset:
                    # cache['original_value'] = self[key][cache['i']]
            # i = cache['i']
            # if reset:
                # self[key][i] = cache['original_value']*p.p
            # else:
                # self[key][i] *= p.p 
            # self.unset_inferences(key)
        # self.construct_functions.append(f)
        # self.format_input_functions.append(
            # lambda: f'{self.name}.scale_value({repr(key)},scale={repr(p)},reset={int(reset)},{my.dict_to_kwargs(limit_to_matches)})')
