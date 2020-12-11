import re
import time
import ast
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from numpy import nan

from . import tools
from .tools import AutoDict
from .exceptions import InferException
from . import optimise

class Dataset(optimise.Optimiser):

    """A collection of scalar or array values, possibly with uncertainties."""

    default_zkeys = []

    ## perhaps better as instance variable?
    _kind_defaults = {
        'f': {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt'   :'+12.8e','description':'float' ,},
        'i': {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt'   :'d'     ,'description':'int'   ,},
        'b': {'cast':lambda x:np.asarray(x,dtype=bool)  ,'fmt'   :''      ,'description':'bool'  ,},
        'U': {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt'   :'s'     ,'description':'str'   ,},
        'O': {'cast':lambda x:np.asarray(x,dtype=object),'fmt'   :''      ,'description':'object',},
    }

    def __init__(
            self,
            name=None,
            load_from_filename=None,
            description=None,
            **kwargs):
        self.classname = self.__class__.__name__
        if name is None:
            name = self.classname.lower()
        optimise.Optimiser.__init__(self,name=name)
        self.description = description # A string describing this dataset
        self.pop_format_input_function()
        self.add_format_input_function(lambda: 'not implemented')
        self._data = dict()
        self._length = 0
        self._over_allocate_factor = 2
        if not hasattr(self,'prototypes'):
            ## derived classes might set this in class definition, so
            ## do not overwrite here
            self.prototypes = {}
        self.permit_nonprototyped_data =  True
        self.permit_reference_breaking = True
        # self.permit_missing = True # add missing data if required
        # self.uncertainty_prefix = 'd_' # a single letter to prefix uncertainty keys
        self.verbose = False
        for key,val in kwargs.items():
            self[key] = val
        if load_from_filename is not None:
            self.load(load_from_filename)
            
    def __len__(self):
        return self._length

    def set(self,key,value,index=None):
        """Set a value"""
        self._modify_time = time.time()
        ## if value is a parameter
        if isinstance(value,optimise.P):
            self.set('d_'+key,value.uncertainty,index)
            value = value.value
        ## delete inferences since data has changed
        if key in self:
            self.unset_inferences(key)
        ## set data differently depending on whether an index is
        ## provided
        if index is None:
            ## decide whether to permit if non-prototyped
            if not self.permit_nonprototyped_data and key not in self.prototypes:
                if self._get_value_key_without_prefix(key) is not None:
                    ## is an uncertainty, vary, stepsize or something
                    pass
                else:
                    raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = dict()
            ## get any prototype data
            if key in self.prototypes:
                data.update(self.prototypes[key])
            ## if a scalar value is given then set as default, and set
            ## data to this value
            if np.isscalar(value):
                data['default'] = value
                value = np.full(len(self),value)
            ## this is an uncertainty of another key
            if (tkey:=self._get_value_key_without_prefix(key)) is not None:
                if tkey == 'd_':
                    data['kind'] = 'f'
                    data['description'] = f'Uncertainty of {tkey}'
                elif tkey == 'v_':
                    data['kind'] = 'b'
                    data['description'] = f'Optimise {tkey}'
                elif tkey == 's_':
                    data['kind'] = 'f'
                    data['description'] = f'Differentiation stepsize for {tkey}'
            ## infer kind
            if 'kind' not in data:
                ## use data to infer kind
                value = np.asarray(value)
                data['kind'] = value.dtype.kind
            # else:
                # ## get from prototypes -- also convert e.g. float to 'f'
                # ## using np.dtype
                # data['kind'] =  np.dtype(data['kind']).kind
                # print('DEBUG:', data['kind'])
            ## convert bytes string to unicode
            if data['kind'] == 'S':
                self.kind = 'U'
            ## some other prototype data
            for tkey in ('description','fmt','cast',):
                if tkey not in data:
                    data[tkey] = self._kind_defaults[data['kind']][tkey]
            ## set data
            data['value'] = data['cast'](value)
            ## initialise inference lists if they do not already exists
            if 'inferred_from' not in data:
                data['inferred_from'] = []
            if 'inferred_to' not in data:
                data['inferred_to'] = []
            ## If this is the data set other than defaults then add to set
            ## length of self and add corresponding data for any defaults
            ## set.
            if len(self) == 0 and len(value) > 0:
                self._length = len(value)
                for tkey,tdata in self._data.items():
                    if tkey == key:
                        continue
                    assert 'default' in tdata,f'Need default for key={tkey}'
                    tdata['value'] = tdata['cast'](np.full(len(self),tdata['default']))
            else:
                assert len(value) == len(self),f'Length of new data {repr(key)} is {len(data)} and does not match the length of existing data: {len(self)}.'
            ## set data
            self._data[key] = data
        else:
            assert key in self,f'Cannot set data with index for unknown {key=}'
            self._data[key]['value'][:self._length][index] = self._data[key]['cast'](value)

    def get(self,key,index=None):
        if index is not None:
            return self.get(key)[index]
        if key not in self._data:
            self._infer(key)
        return self._data[key]['value'][:self._length]

    def get_value(self,key,index=None):
        assert self.is_value_key(key),'Not a value key'
        return self.get(key,index)

    def set_value(self,key,value,index=None):
        assert self.is_value_key(key),'Not a value key'
        self.set(key,value,index)

    def get_uncertainty(self,key,index=None):
        assert self.is_value_key(key),'Not a value key'
        if 'd_'+key in self:
            return self.get('d_'+key,index)
        else:
            return None

    def set_uncertainty(self,key,value,index=None):
        assert self.is_value_key(key),'Not a value key'
        self.set('d_'+key,value,index)

    def get_vary(self,key,index=None):
        assert self.is_value_key(key),'Not a value key'
        if 'v_'+key in self:
            return self.get('v_'+key,index)
        else:
            return None

    def set_vary(self,key,value,index=None):
        assert self.is_value_key(key),'Not a value key'
        self.set('v_'+key,value,index)

    def get_differentiation_step(self,key,index=None):
        assert self.is_value_key(key),'Not a value key'
        if 's_'+key in self:
            return self.get('s_'+key,index)
        else:
            return None

    def set_differentiation_step(self,key,value,index=None):
        assert self.is_value_key(key),'Not a value key'
        self.set('s_'+key,value,index)

    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            return self.get(arg)
        elif tools.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            return self.copy(keys=arg)
        else:
            return self.copy(index=arg)

    def __setitem__(self,key,value):
        """Set a value"""
        self.set(key,value)
        
    def clear(self):
        self._data.clear()

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unset_inferences(key)
        # if key in self._data:
            # ## might already be gone if this is called recursively
        self._data.pop(key)

    def unset_inferences(self,key):
        """Delete any record of inferences to or from this key and any data
        inferred from it."""
        for inferred_from in self._data[key]['inferred_from']:
            self._data[inferred_from]['inferred_to'].remove(key)
            self._data[key]['inferred_from'].remove(inferred_from)
        for inferred_to in self._data[key]['inferred_to']:
            self._data[inferred_to]['inferred_from'].remove(key)
            self._data[key]['inferred_to'].remove(inferred_to)
            self.unset(inferred_to)

    def is_inferred_from(self,key_to,key_from):
        return key_from in self._data[key_to]['inferred_from']

    def set_prototype(self,key,kind,**kwargs):
        """Set prototype data."""
        assert kind in self._kind_defaults,'Unknown kind: {kind=}'
        self.prototypes[key] = dict(kind=kind,infer={},**kwargs)
        for tkey,tval in self._kind_defaults[kind].items():
            self.prototypes[key].setdefault(tkey,tval)

    def add_infer_function(self,key,dependencies,function):
        assert key in self.prototypes
        self.prototypes[key]['infer'][dependencies] = function

    def index(self,index):
        """Index all array data in place."""
        original_length = len(self)
        for data in self._data.values():
            data['value'] = data['value'][:original_length][index]
            self._length = len(data['value'])

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if keys is None:
            keys = self.keys()
        retval = self.__class__() # new version of self
        retval.permit_nonprototyped_data = self.permit_nonprototyped_data
        for key in keys:
            if index is None:
                retval[key] = self[key]
            else:
                retval[key] = deepcopy(self[key][index])
        return retval

    def find(self,**matching_keys_vals):
        """Return an array of indices matching key_vals."""
        length = 0
        for val in matching_keys_vals.values():
            if not np.isscalar(val):
                if length == 0:
                    length = len(val)
                else:
                    assert len(val) == length
        retval = np.empty(length,dtype=int)
        for j in range(length):
            i = tools.find(
                self.match(
                    **{key:(val if np.isscalar(val) else val[j])
                       for key,val in matching_keys_vals.items()}))
            if len(i)==0:
                raise Exception(f'No matching row found: {matching_keys_vals=}')
            if len(i)>1:
                raise Exception(f'Multiple matching rows found: {matching_keys_vals=}')
            retval[j] = i
        return retval

    def match(self,**keys_vals):
        """Return boolean array of data matching all key==val.\n\nIf key has
        suffix '_min' or '_max' then match anything greater/lesser
        or equal to this value"""
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            if len(key) > 4 and key[-4:] == '_min':
                i &= (self[key[:-4]] >= val)
            elif len(key) > 4 and key[-4:] == '_max':
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

    def get_unique_value(self,key,**matching_keys_vals):
        """Return value of key that is the uniquely matches
        matching_keys_vals."""
        i = tools.find(self.match(**matching_keys_vals))
        assert len(i)==1,f'Non-unique matches for {matching_keys_vals=}'
        return self.get_value(key,i)

    def _infer(self,key,already_attempted=None):
        """Get data, or try and compute it."""
        if key in self:
            return
        ## avoid getting stuck in a cycle
        if already_attempted is None:
            already_attempted = []
        if key in already_attempted:
            raise InferException(f"Already unsuccessfully attempted to infer key: {repr(key)}")
        already_attempted.append(key)
        if key not in self.prototypes:
            raise InferException(f"No prototype for {key=}")
        ## loop through possible methods of inferences.
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
                    if (tuncertainty:=self.get_uncertainty(dependency)) is not None:
                        diffstep = 1e-10*self[dependency]
                        parameters[i] = self[dependency] + diffstep # shift one
                        dvalue = value - function(self,*parameters)
                        parameters[i] = self[dependency] # put it back
                        data = self._data[dependency]
                        squared_contribution.append((tuncertainty*dvalue/diffstep)**2)
                if len(squared_contribution)>0:
                    self.set_uncertainty(key,np.sqrt(sum(squared_contribution)))
                ## if we get this far without an InferException then
                ## success!.  Record inference dependencies.
                self._data[key]['inferred_from'].extend(dependencies)
                for dependency in dependencies:
                    self._data[dependency]['inferred_to'].append(key)
                break           
            ## some kind of InferException, try next set of dependencies
            except InferException as err:
                if self.verbose:
                    print('    InferException: '+str(err))
                continue      
        ## complete failure to infer
        else:
            raise InferException(f"Could not infer key: {repr(key)}")

    # def _get_value_key_from_uncertainty(self,key):
        # """Get value key from uncertainty key, or return None."""
        # if len(key) > 2 and key[:2] == 'd_':
            # return key[2:]
            # assert self.permit_nonprototyped_data or key[2:] in self.prototypes,f'Uncertain key with non-prototyped value key: {key=}'
        # else:
            # return None

    def _get_value_key_without_prefix(self,key):
        """Get value key from uncertainty key, or return None."""
        if r:=re.match(r'^(s_|d_|v_)(.+)$',key):
            return r.group(1)
        else:
            return None

    def is_value_key(self,key):
        value_key = self._get_value_key_without_prefix(key)
        if value_key is None:
            return True
        else:
            return False 
        
    def __iter__(self):
        for key in self._data:
            yield key

    def as_dict(self,keys=None,index=None):
        """Data in row index as a dict of scalars."""
        if keys is None:
            keys = self.keys()
        if index is None:
            return {key:self[key]for key in keys}
        else:
            return {key:self[key][index] for key in keys}
        
    def rows(self,keys=None):
        """Iterate over data row by row, returns as a dictionary of
        scalar values."""
        if keys is None:
            keys = self.keys()
        for i in range(len(self)):
            yield(self.as_dict(keys=keys,index=i))

    def row_data(self,keys=None,index=None):
        """Iterate rows, returning data in a tuple."""
        if keys is None:
            keys = self.keys()
        if index is None:
            index = slice(0,len(self))
        for t in zip(*[self[key][index] for key in keys]):
            yield t

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

    def value_keys(self):
        return [key for key in self._data.keys() if self.is_value_key(key)]
    
    def optimised_keys(self):
        return [key for key in self._data.keys() if 'v_'+key in self]

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

    def format(
            self,
            keys=None,
            delimiter=' | ',
            automatically_add_uncertainties=True,
            unique_values_in_header=True,
    ):
        """Format data into a string representation."""
        if len(self)==0:
            return ''
        if keys is None:
            keys = self.keys()
        if automatically_add_uncertainties:
            tkeys = []
            for key in keys:
                tkeys.append(key)
                if key in tkeys:
                    continue
                if self._get_uncertainty(key) is not None:
                    tkeys.append(self.uncertainty_prefix + key)
            keys = tkeys
        ## collect table data
        header = [f'classname = {repr(self.classname)}']
        if self.description is not None:
            header.append(f'description = {repr(self.description)}')
        columns = []
        for key in keys:
            if unique_values_in_header and len(tval:=self.unique(key)) == 1:
                header.append(f'{key} = {repr(tval[0])}')
            else:
                ## two passes required on all data to align column
                ## widths
                vals = [format(t,self._data[key]['fmt'])
                        for t in self._data[key]['value'][:len(self)]]
                width = str(max(len(key),np.max([len(t) for t in vals])))
                columns.append([format(key,width)]+[format(t,width) for t in vals])
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
            data = {key:self[key] for key in keys}
            for attrkey in ('classname','description'):
                if (attrval:=getattr(self,attrkey)) is not None:
                    data[attrkey] = attrval
            np.savez(filename,**data)
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            data = {key:self[key] for key in keys}
            for attrkey in ('classname','description'):
                if (attrval:=getattr(self,attrkey)) is not None:
                    data[attrkey] = attrval
            tools.dict_to_hdf5(filename,data)
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
            comment='',
            delimiter=None,
            table_name=None,
            translate_keys=None, # from key in file to key in self, None for skip
            **set_keys_vals   # set this data after loading is done
    ):
        '''Load data from a text file in standard format generated by
        save_to_file.'''
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
            # assert comment not in ['',' '], "Not implemented"
            filename = tools.expand_path(filename)
            data = {}
            ## load header
            with open(filename,'r') as fid:
                for iline,line in enumerate(fid):
                    if r:=re.match(r'^ *'+comment+f' *([^= ]+) *= *(.+) *',line):
                        key,val = r.groups()
                        data[key] = ast.literal_eval(val)
                    else:
                        ## end of header
                        break
            ## load array data
            data.update(tools.txt_to_dict(
                filename,
                delimiter=delimiter,
                labels_commented=False,
                skiprows=iline,))
        ## translate keys
        if translate_keys is not None:
            for from_key,to_key in translate_keys.items():
                if from_key in data:
                    if to_key is None:
                        data.pop(from_key)
                    else:
                        data[to_key] = data.pop(from_key)
        ## Set data in self and selected attributes
        for key,val in data.items():
            if key in ('classname','description',):
                setattr(self,key,val)
            else:
                self[key] = val

    def load_from_string(self,string,delimiter='|'):
        """Load data from a string."""
        ## Write a temporary file and then uses the regular file load
        tmpfile = tools.tmpfile()
        tmpfile.write(string.encode())
        tmpfile.flush()
        tmpfile.seek(0)
        self.load(tmpfile.name,delimiter='|')

    def append(self,**kwargs):
        """Append a single row of data from kwarg scalar values."""
        self.extend(**{key:[val] for key,val in kwargs.items()})

    def extend(self,**kwargs):
        """Extend self with data given as kwargs."""
        ## ensure enough data is provided
        for key in self:
            assert key in kwargs or 'default' in self._data[key], f'Key not in extending data and has no default: {key=}'
        ## get length of new data
        new_data_length = 0
        for key,val in kwargs.items():
            if np.isscalar(val):
                pass
            elif new_data_length == 0:
                new_data_length = len(val)
            else:
                assert new_data_length == len(val),f'Inconsistent length for {key=}'
        ## test if any data to extend with
        if new_data_length == 0:
            return
        ## extend scalar kwargs to full new length
        for key,val in list(kwargs.items()):
            if np.isscalar(val):
                kwargs[key] = np.full(new_data_length,kwargs[key])
        ## include default values if needed
        for key in self:
            if key not in kwargs:
                kwargs[key] = np.full(new_data_length,self._data[key]['default'])
        ## add data
        if len(self) == 0:
            ## currently no data -- simply set extending values
            for key,val in kwargs.items():
                self[key] = val
        else:
            total_length = len(self) + new_data_length
            for key,val in kwargs.items():
                assert key in self, f'Unknown extending key: {key=}'
                ## increase unicode dtype length if new strings are
                ## longer than the current
                if self._data[key]['kind'] == 'U':
                    ## this is a really hacky way to get the length of string in a numpy array!!!
                    old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self._data[key]['value'].dtype)))
                    new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(val).dtype)))
                    if new_str_len > old_str_len:
                        ## reallocate array with new dtype with overallocation
                        t = np.empty(len(self._data[key]['value']),dtype=f'<U{new_str_len*self._over_allocate_factor}')
                        t[:len(self)] = self._data[key]['value'][:len(self)]
                        self._data[key]['value'] = t

                ## reallocate and lengthen value array if necessary
                if total_length > len(self._data[key]['value']):
                    self._data[key]['value'] = np.concatenate((
                        self[key],
                        np.empty(
                            int(total_length*self._over_allocate_factor-len(self)),
                            dtype=self._data[key]['value'].dtype)))
                ## set extending data
                self._data[key]['value'][len(self):total_length] = val
            self._length = total_length

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
        zkeys = [t for t in tools.ensure_iterable(zkeys) if t not in ykeys and t!=xkey and self.is_known(t)] # remove xkey and ykeys from zkeys
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
                y = z[ykey]
                if label is not None:
                    kwargs.setdefault('label',label)
                if plot_errorbars and (dy:=z.get_uncertainty(ykey)) is not None:
                    ## plot errorbars
                    kwargs.setdefault('mfc','none')
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

def find_common(x,y,*keys,verbose=False):
    """Return indices of two Datasets that have uniquely matching
    combinations of keys."""
    ## if empty list then nothing to be done
    if len(x)==0 or len(y)==0:
        return(np.array([]),np.array([]))
    # ## Make a list of default keys if not provided as inputs. If a
    # ## Level or Transition object (through a hack) then use
    # ## defining_qn, else use all set keys known to both.
    # if len(keys)==0:
        # if hasattr(x,'defining_qn'):
            # keys = [t for t in x.defining_qn if x.is_known(t) and y.is_known(t)]
        # else:
            # keys = [t for t in x.keys() if x.is_known(t) and y.is_known(t)]
    if verbose:
        print('keys:',keys)
    x.assert_known(keys)
    y.assert_known(keys)
    ## sort by first calculating a hash of sort keys
    xhash = np.array([hash(t) for t in x.row_data(keys=keys)])
    yhash = np.array([hash(t) for t in y.row_data(keys=keys)])
    ## get sorted hashes, checking for uniqueness
    xhash,ixhash = np.unique(xhash,return_index=True)
    assert len(xhash) == len(x), f'Non-unique combinations of keys in x: {repr(keys)}'
    yhash,iyhash = np.unique(yhash,return_index=True)
    assert len(yhash) == len(y), f'Non-unique combinations of keys in y: {repr(keys)}'
    ## use np.searchsorted to find one set of hashes in the other
    iy = np.arange(len(yhash))
    ix = np.searchsorted(xhash,yhash)
    ## remove y beyond max of x
    i = ix<len(xhash)
    ix,iy = ix[i],iy[i]
    ## requires removing hashes that have no searchsorted partner
    i = yhash[iy]==xhash[ix]
    ix,iy = ix[i],iy[i]
    ## undo the effect of the sorting above
    ix,iy = ixhash[ix],iyhash[iy]
    ## sort by index of first array -- otherwise sorting seems to be arbitrary
    i = np.argsort(ix)
    ix,iy = ix[i],iy[i]
    return ix,iy

def get_common(x,y,*keys,**limit_to_matches):
    """A short cut to find the common levels of a Dynamic_Recarrays object
    and return subset copies that are sorted to match each other."""
    if limit_to_matches is not None:
        x = x.matches(**limit_to_matches)
        y = y.matches(**limit_to_matches)
    i,j = find_common(x,y,*keys)
    # if len(i)==0:
        # ## No matches return copy, but empty of data
        # return(
            # x[np.full(len(x),False)] if len(x)>0 else x[:],
            # y[np.full(len(y),False)] if len(y)>0 else y[:],
        # ) 
    return x[i],y[j]

def load(filename):
    """Load a Dataset or one of its subclasses."""
    ## load once, determine classname, then load again into correct
    ## class if not Dataset
    d = Dataset()
    d.load(filename)
    if d.classname != 'Dataset':
        from . import lines,levels
        for module in (levels,lines):
            if hasattr(module,d.classname):
                d = getattr(module,d.classname)()
                d.load(filename)
                break
    return d


    
