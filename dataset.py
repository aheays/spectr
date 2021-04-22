import re
from time import perf_counter as timestamp
import ast
from copy import copy,deepcopy
from pprint import pprint,pformat
import importlib
import warnings

import numpy as np
from numpy import nan
from immutabledict import immutabledict as idict

from . import tools
from .tools import AutoDict,convert_to_bool_vector_array
from .exceptions import InferException
from . import convert
from . import optimise
from .optimise import optimise_method,Parameter,Fixed



class Dataset(optimise.Optimiser):

    """A set of data vectors of common length."""


    ## basic properties for keys that have no or incomplete prototypes
    default_kinds = {

        'f':    {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt':'+12.8e','default':nan   ,'description':'float' ,},
        'i':    {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt':'d'     ,'default':-999  ,'description':'int'   ,},
        'b':    {'cast':convert_to_bool_vector_array       ,'fmt':''      ,'default':True  ,'description':'bool'  ,},
        'U':    {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt':'s'     ,'default':''    ,'description':'str'   ,},
        'O':    {'cast':lambda x:np.asarray(x,dtype=object),'fmt':''      ,'default':None  ,'description':'object',},
        'unc' : {'cast':lambda x:np.abs(x,dtype=float)     ,'fmt':'8.2e'  ,'default':nan   ,'description':'Uncertainty'                         ,},
        'step': {'cast':lambda x:np.abs(x,dtype=float)     ,'fmt':'8.2e'  ,'default':1e-8  ,'description':'Numerical differentiation step size' ,},
        # 'vary': {'cast':lambda x:np.asarray(x,dtype=bool)  ,'fmt':'5'     ,'default':False ,'description':'Whether to vary during optimisation.',},
        'vary': {'cast':convert_to_bool_vector_array       ,'fmt':'5'     ,'default':False ,'description':'Whether to vary during optimisation.',},

    }

    ## always available
    default_attributes = {
        'classname':None,
        'description':None,
    }

    ## prototypes on instantiation
    default_prototypes = {}

    ## used for plotting and sorting perhaps
    default_zkeys = ()

    def __init__(
            self,
            name=None,
            permit_nonprototyped_data = True,
            auto_defaults = False, # if no data or way to infer it, then set to default prototype value if this data is needed
            prototypes = None,  # a dictionary of prototypes
            load_from_file = None,
            load_from_string = None,
            copy_from = None,
            limit_to_match=None, # dict of things to match
            **kwargs):
        ## basic internal variables
        self._data = {} # table data and its properties stored here
        self._length = 0    # length of data
        self._over_allocate_factor = 2 # to speed up appending to data
        self.attributes = {} # applies to all data
        self._last_modify_data_time = timestamp() # when data is changed this is update
        self.permit_nonprototyped_data = permit_nonprototyped_data # allow the addition of data not in self.prototypes
        self.auto_defaults = auto_defaults # set default values if necessary automatically
        self.verbose = False                             # print extra information at various places
        self.prototypes = {}                            # predefined data keys 
        ## set default prototypes
        self.prototypes = copy(self.default_prototypes)
        ## set prototypes given as an argument
        if prototypes is not None:
            for key,val in prototypes.items():
                self.set_prototype(key,**val)
        ## initialise default attributes
        for key,val in self.default_attributes.items():
            self.attributes[key] = val
        ## classname to identify type of Dataset
        self.attributes['classname'] = re.sub(
            r"<class 'spectr.(.+)'>",
            r'\1',
            str(self.__class__))
        ## default name is a valid symbol
        if name is None:
            name = tools.make_valid_python_symbol_name(
                self.attributes['classname'].lower())
        ## init as optimiser, make a custom form_input_function, save
        ## some extra stuff if output to directory
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        ## new format input function
        def format_input_function():
            retval = f'{self.name} = {self.attributes["classname"]}({repr(self.name)},'
            if load_from_file is not None:
                retval += f'load_from_file={repr(load_from_file)},'
            if len(kwargs)>0:
                retval += '\n'
            for key,val in kwargs.items():
                retval += f'    {key}={repr(val)},\n'
            retval += ')'
            return retval
        self.add_format_input_function(format_input_function)
        self.add_save_to_directory_function(lambda directory: self.save(f'{directory}/dataset.h5'))
        ## copy data from another dataset provided as argument
        if copy_from is not None:
            self.copy_from(copy_from)
        ## load data from a file path provide as an argument
        if load_from_file is not None:
            self.load(load_from_file)
        ## load data from an encode tabular string provided as an
        ## argument
        if load_from_string is not None:
            self.load_from_string(load_from_string)
        ## kwargs set data somehow
        for key,val in kwargs.items():
            if key in self.attributes:
                ## set attribute
                self.attributes[key] = val
            elif isinstance(val,optimise.Parameter):
                ## an optimisable parameter (input function already
                ## handled)
                self.set_parameter(key,val)
                self.pop_format_input_function()
            else:
                self[key] = val
        ## limit to matching data somehow loaded above
        if limit_to_match is not None:
            self.limit_to_match(**limit_to_match)

    def __len__(self):
        return self._length

    def set(
            self,
            key,
            value,
            associated_data=None,
            index=None,
            _inferred=False,    # internal use -- mark if this is set call is due to inference
            match=None,
            **prototype_kwargs
    ):
        """Set a value"""
        if match is not None:
            assert index is None
            index = self.match(match)
        ## check errors in 'associated_data'
        if associated_data is None:
            root_key = None
        else:
            root_key,key = key,f'{key}_{associated_data}'
            if not self.is_known(root_key):
                raise Exception(f"Value for {root_key=} must be set before {key=}")
            if self.get_kind(root_key) != 'f':
                raise Exception(f'Unc/step/vary only defined for kind="f" floating-point data and {root_key=} has kind={self.get_kind(root_key)}')
            if associated_data == 'vary':
                if not self.is_set(f'{root_key}_unc'):
                    self.set(root_key,self.default_kinds['unc']['default'],'unc')
                if not self.is_set(f'{root_key}_step'):
                    if 'default_step' in self._data[root_key]:
                        self.set(root_key,self._data[root_key]['default_step'],'step')
                    else:
                        self.set(root_key,self.default_step['step']['default'],'step')
            prototype_kwargs = self.default_kinds[associated_data] | prototype_kwargs
        ## update modification if externally set, not if it is inferred
        if self.verbose:
            print(f'{self.name}: setting {key} inferred={_inferred}')
        if not _inferred:
            self._last_modify_data_time = timestamp()
        ## delete inferences since data has changed
        if key in self:
            self.unlink_inferences(key)
        ## if an index is provided then data must already exist, set
        ## new indeed data and return
        if index is not None:
            if key not in self and 'default' in prototype_kwargs:
                if associated_data is None:
                    self.set(key,prototype_kwargs['default'])
                else:
                    self.set(root_key,prototype_kwargs['default'],associated_data)
            data = self._data[key]
            if key not in self:
                raise Exception(f'Cannot set data by index for unset key: {key}')
            data['value'][:self._length][index] = data['cast'](value)
        ## set full array -- does not hvae to existin in advacne
        else:
            ## create entire data dict
            ## decide whether to permit if non-prototyped
            if not self.permit_nonprototyped_data:
                if associated_data is None:
                    if key not in self.prototypes:
                        raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = dict()
            self._data[key] = data
            data['associated_data'] = {}
            ## get any prototype data
            if key in self.prototypes:
                for tkey,tval in self.prototypes[key].items():
                    if tkey not in ('default',):
                        data[tkey] = tval
            ## apply prototype kwargs
            for tkey,tval in prototype_kwargs.items():
                data[tkey] = tval
            ## get information from default_kinds if associated data
            if associated_data is not None:
                assert associated_data in self.default_kinds
                data['kind'] = associated_data
            ## if a scalar value is then expand to full length, if
            ## vector then cast as a new array.  Do not use asarray
            ## but instead make a copy -- this will prevent mysterious
            ## bugs where assigned arrays feedback.
            if 'kind' in data and data['kind'] == 'O':
                raise ImplementationError()
            ## use data to infer kind if necessary
            if 'kind' not in data:
                value = np.asarray(value)
                data['kind'] = value.dtype.kind
            ## convert bytes string to unicode
            if data['kind'] == 'S':
                self.kind = 'U'
            ## some other prototype data
            for tkey in ('description','fmt','cast'):
                if tkey not in data:
                    data[tkey] = self.default_kinds[data['kind']][tkey]
            if data['kind']=='f' and 'default_step' not in data:
                data['default_step'] = 1e-8
            ## infer function dict,initialise inference lists, and units if needed
            for tkey,tval in (('infer',[]),
                              ('inferred_from',[]),
                              ('inferred_to',[]),):
                if tkey not in data:
                    data[tkey] = tval
            ## deal with this being or having associated data
            if root_key is not None:
                self._data[root_key]['associated_data'][associated_data] = data
                data['root_data'] = self._data[root_key]
            ## if a scalar expand to lenght of self and also set as
            ## default
            if not tools.isiterable(value):
                data['default'] = value
                value = np.full(len(self),value)
            ## If this is the data set other than defaults then add to set
            ## length of self and add corresponding data for any defaults
            ## set.
            if len(self) == 0 and len(value) > 0:
                self._length = len(value)
                for tkey,tdata in self._data.items():
                    if tkey == key:
                        continue
                    if 'default' in tdata:
                        tdata['value'] = tdata['cast'](np.full(len(self),tdata['default']))
                    else:
                        raise Exception(f'Need default for key {tkey}')
            elif len(value) != len(self):
                raise Exception(f'Length of new data {repr(key)} is {len(value)} and does not match the length of existing data: {len(self)}.')
            ## cast and set data
            data['value']  = data['cast'](value)

    def get(
            self,
            key,
            associated_data=None,
            index=None,
            units=None,
    ):
        """Get value."""
        if key in self.attributes:
            ## return if an attribute
            retval = self.attributes[key]
        else:
            ## return vector data
            if key not in self._data:
                try:
                    ## attempt to infer
                    self._infer(key)
                except InferException as err:
                    if self.auto_defaults:
                        ## try an autodefault if inference fails
                        if key in self.prototypes:
                            self[key] = self.prototypes[key]['default']
                        else:
                            raise Exception(r"No prototype for autodefault: {repr(key)}")
                    else:
                        raise err
            if index is None:
                ## get deafult entire index
                index = slice(0,len(self))
            if associated_data is None:
                data = self._data[key]
            else:
                data = self._data[key]['associated_data'][associated_data]
            retval = data['value'][:self._length][index]
        if units is not None:
            ## convert units before setting
            retval = convert.units(retval,self.get_units(key),units)
        return retval

    def set_default(self,key,value):
        self._data[key]['default'] = value
        if key not in self:
            self[key] = value

    def get_default(self,key):
        if 'default' not in self._data[key]:
            return None
        return self._data[key]['default']

    def get_units(self,key):
        if 'units' not in self._data[key]:
            return None
        return self._data[key]['units']

    def cast(self,key,value):
        """Returns value cast appropriately for key."""
        if 'cast' not in self._data[key]:
            return np.asarray(value)
        else:
            return self._data[key]['cast'](value)

    def set_prototype(self,key,kind,infer=None,**kwargs):
        """Set prototype data."""
        if kind is str:
            kind = 'U'
        elif kind is float:
            kind = 'f'
        elif kind is bool:
            kind = 'b'
        elif kind is object:
            kind = 'O'
        if infer is None:
            infer = []
        self.prototypes[key] = dict(kind=kind,infer=infer,**kwargs)
        for tkey,tval in self.default_kinds[kind].items():
            self.prototypes[key].setdefault(tkey,tval)

    def get_prototype(self,key):
        """Get prototype as a dictionary."""
        if key not in self.prototypes:
            return None
        return self.prototypes[key]

    def get_kind(self,key):
        return self._data[key]['kind']

    @optimise_method(format_single_line=True)
    def set_parameter(
            self,
            key,
            value,          # a scalar or Parameter
            index=None,         # only apply to these indices
            match=None,
            # **prototype_kwargs,
    ):
        """Set a value and it will be updated every construction and possible
        optimised."""
        if match is not None:
            index = self.match(**match)
        ## if not a parameter then treat as a float -- could use set(
        ## instead and branch there, requiring a Parameter here
        if isinstance(value,Parameter):
            ## only reconstruct for the following reasons
            if (
                    key not in self.keys() # key is unknown -- first run
                    or value._last_modify_value_time > self._last_construct_time # parameter has been set
                    or np.any(self.get(key,index=index) != value.value) # data has changed some other way and differs from parameter
                    or (self.is_set(key,'unc')
                        and not np.isnan(value.unc)
                        and (np.any(self.get(key,'unc',index=index) != value.unc))) # data has changed some other way and differs from parameter
                ):
                self.set(key,value.value,index=index)
                self.set(key,value.unc,'unc',index=index)
                self.set(key,value.step,'step',index=index)
        elif key in self.attributes:
            self.attributes[key] = value
        else:
            ## only reconstruct for the following reasons
            if (
                    key not in self.keys() # key is unknown -- first run
                    or np.any(self.get(key,index=index) != value) # data has changed some other way and differs from parameter
                ):
                self.set(key,value=value,index=index)

    @optimise_method()
    def set_spline(self,xkey,ykey,knots,order=3,match=None,index=None,_cache=None):
        """Set ykey to spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...]. If index or a match dictionary
        given, then only set these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        xspline,yspline = zip(*knots)
        if index is not None:
            i = index
        elif match is not None:
            i = self.match(**match)
        else:
            i = None
        self.set(
            ykey,
            value=tools.spline(xspline,yspline,self.get(xkey,index=i),order=order),
            index=i,
        )
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            if i is None:
                self.set(ykey,'unc')
            else:
                self.set(ykey,'unc',index=i)

    def keys(self):
        return list(self._data.keys())

    def optimised_keys(self):
        return [key for key in self.keys() if self.is_set(key,'vary')]

    def root_keys(self):
        return [key for key in self.keys() if 'root_data' not in self._data[key]]

    def __iter__(self):
        for key in self._data:
            yield key

    def items(self):
        """Iterate over set keys and their values."""
        for key in self:
            yield key,self[key]

    def pop(self,key):
        """Pop data in key."""
        value = self[key]
        self.unset(key)
        return value

    def is_set(self,key,associated_data=None):
        if key in self._data:
            if associated_data is None:
                return True
            if associated_data in self._data[key]['associated_data']:
                return True
        return False

    def assert_known(self,*keys):
        for key in keys:
            self[key]

    def is_known(self,*keys):
        try:
            self.assert_known(*keys)
            return True 
        except InferException:
            return False

    # def _decode_key_with_associated_data(self,key_with_associated_data):
        # """ Split associated associated_data key, e.g., abs_unc, into 'abs' and
        # 'unc'"""
        # for associated_data in self.associated_associated_datas:
            # n = len(associated_data)
            # if (len(key_with_associated_data) > (n+1)
                # and key_with_associated_data[-n-1] == '_'
                # and key_with_associated_data[-n:] == associated_data):
                # return key_with_associated_data[:-n-1],key_with_associated_data[-n:]
        # else:
            # return key_with_associated_data,'value'
            
    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            return self.get(arg)
        elif tools.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            ## make a copy of a list of keys
            return self.copy(keys=arg)
        else:
            ## make an indexed copy
            return self.copy(index=arg)

    def __setitem__(self,key,value):
        """Set a key to value. If key_unc then set uncertainty. If key_vary or
        key_step then set optimisation parameters"""
        if key in self.attributes:
            self.attributes[key] = value
        else:
            if isinstance(value,optimise.P):
                self.set_parameter(key,value)
            else:
                for associated_data in ('unc','step','vary'):
                    n = len(associated_data)
                    if len(key) > (n+1) and key[-n:] == associated_data:
                        self.set(key[:-n-1],value,associated_data=associated_data)
                        break
                else:
                    self.set(key,value)
       
    def clear(self):
        """Clear all data"""
        self._last_modify_value_time = timestamp()
        self._length = 0
        self._data.clear()

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unlink_inferences(key)
        data = self._data[key]
        if 'root_key' in data:
            self._data[data['root_key']['associated_data']].pop(key)
        else:
            for associated_key in data['associated_data']:
                self.pop(f'{key}_{associated_key}')
        self._data.pop(key)

    def is_inferred(self,key):
        if len(self._data[key]['inferred_from']) > 0:
            return True
        else:
            return False
   
    def unset_inferred(self):
        """Delete all inferred data."""
        for key in list(self):
            if key in self and self.is_inferred(key):
                self.unlink_inferences(key)
                self.unset(key)
   
    def unlink_inferences(self,*keys,unset_inferred=True):
        """Delete any record of inferences to or from the given keys and
        delete anything inferred from these keys (but not if it is  among
        keys itself)."""
        self.assert_known(*keys)
        for key in keys:
            if key in self:     # test this since key might have been unset earlier in this loop
                for inferred_from in list(self._data[key]['inferred_from']):
                    ## delete record of having been inferred from
                    ## something else
                    self._data[inferred_from]['inferred_to'].remove(key)
                    self._data[key]['inferred_from'].remove(inferred_from)
                for inferred_to in list(self._data[key]['inferred_to']): #
                    if inferred_to not in self._data:
                        ## this inferred_to has already been taking
                        ## care of in a previous loop somewhere
                        continue
                    ## delete record of having led to seomthing else
                    ## begin inferred
                    self._data[inferred_to]['inferred_from'].remove(key)
                    self._data[key]['inferred_to'].remove(inferred_to)
                    ## delete inferred data if not an argument key
                    if inferred_to not in keys:
                        self.unset(inferred_to)

    def add_infer_function(self,key,dependencies,function):
        """Add a new method of data inference."""
        self.prototypes[key]['infer'].append((dependencies,function))

    def index(self,index):
        """Index all array data in place."""
        original_length = len(self)
        for data in self._data.values():
            data['value'] = data['value'][:original_length][index]
            self._length = len(data['value'])

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        retval = self.__class__() # new version of self
        retval.copy_from(self,keys,index)
        return retval

    @optimise_method()
    def copy_from(
            self,
            source,             # Dataset to copy
            keys=None,          # keys to copy
            index=None,         # indices to copy
            match=None,         # copy matching {key:val,...} 
            copy_opt=False,
    ):
        """Copy all values and uncertainties from source Dataset and update if
        source changes during optimisation."""
        self.clear()            # total data reset
        if keys is None:
            keys = source.keys()
        if not copy_opt:
            keys = [key for key in keys if not re.match(r'.+_(vary|step)$',key)]
        self.permit_nonprototyped_data = source.permit_nonprototyped_data
        ## get matching indices
        if match is not None:
            if index is None:
                index = source.match(**match)
            else:
                ## requires index be a boolean mask array -- will fail
                ## on index array or slice, could add logic for those
                ## cases
                index &= source.match(**match)
        for key in keys:
            self[key] = source.get(key,index=index)
        ## copy all attributes
        for key in source.attributes:
            self[key] = source[key]

    def find(self,**keys_vals):
        """Return an array of indices matching key_vals."""
        length = 0
        for val in keys_vals.values():
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
                       for key,val in keys_vals.items()}))
            if len(i)==0:
                raise Exception(f'No matching row found: {keys_vals=}')
            if len(i)>1:
                raise Exception(f'Multiple matching rows found: {keys_vals=}')
            retval[j] = i
        return retval

    def match(self,keys_vals=idict(),**kwarg_keys_vals):
        """Return boolean array of data matching all key==val.\n\nIf key has
        suffix '_min' or '_max' then match anything greater/lesser
        or equal to this value"""
        keys_vals = keys_vals | kwarg_keys_vals
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

    def find(self,**keys_vals):
        """Find unique indices matching keys_vals which contains one or more
        vector matches or the same length."""
        ## SLOW IMPLEMENTATION -- REPLACE WITH HASH MATCHING?
        ## separate vector and scalar match data
        vector_keysvals = {}
        scalar_keysvals = {}
        vector_length = None
        for key,val in keys_vals.items():
            if np.isscalar(val):
                scalar_keysvals[key] = val
            else:
                vector_keysvals[key] = val
                if vector_length == None:
                    vector_length = len(val)
                elif vector_length != len(val):
                    raise Exception('All vector matching data must be the same length')
        ## get data matching scalar keys_vals
        iscalar = tools.find(self.match(**scalar_keysvals))
        ## find vector_key matches one by one
        i = np.empty(vector_length,dtype=int)
        for ii in range(vector_length):
            ti = np.all([self[key][iscalar]==val[ii] for key,val in vector_keysvals.items()],0)
            ti = tools.find(ti)
            if len(ti) == 0:
                raise Exception("No match: {vector_key}={repr(vector_vali)} and {repr(keys_vals)}")
            if len(ti) > 1:
                raise Exception("Non-unique match: {vector_key}={repr(vector_vali)} and {repr(keys_vals)}")
            i[ii] = iscalar[ti]
        return i

    def matches(self,*args,**kwargs):
        """Returns a copy reduced to matching values."""
        return self.copy(index=self.match(*args,**kwargs))

    def limit_to_match(self,**keys_vals):
        self.index(self.match(**keys_vals))

    def remove_match(self,**keys_vals):
        self.index(~self.match(**keys_vals))

    def unique(self,key):
        """Return unique values of one key."""
        self[key]
        if self.get_kind(key) == 'O':
            return self[key]
        else:
            return np.unique(self[key])

    def unique_combinations(self,*keys):
        """Return a list of all unique combination of keys."""
        return tools.unique_combinations(*[self[key] for key in keys])

    def unique_dataset(self,*keys):
        """Return a dataset summarising unique combination of keys."""
        retval = self.__class__()
        for data in self.unique_dicts(*keys):
            retval.append(**data)
        return retval

    def unique_dicts(self,*keys):
        """Return an iterator where each element is a unique set of keys as a
        dictionary."""
        if len(keys)==0:
            return ({},)
        retval = [{key:val for key,val in zip(keys,vals)} for vals in self.unique_combinations(*keys)]
        retval = sorted(retval, key=lambda t: [t[key] for key in keys])
        return retval 

    def unique_dicts_match(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a boolean array matching this
        combination."""
        retval = []
        for d in self.unique_dicts(*keys):
            retval.append((d,self.match(**d)))
        return retval

    def unique_dicts_matches(self,*keys):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a copy of self reduced
        to matching values."""
        retval = []
        for d in self.unique_dicts(*keys):
            retval.append((d,self.matches(**d)))
        return retval
                          
    def _infer(self,key,already_attempted=None,depth=0):
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
        for dependencies,function in self.prototypes[key]['infer']:
            ## if function is a tuple of two functions then the second
            ## is for computing uncertainties
            if tools.isiterable(function):
                function,unc_function = function
            else:
                unc_function = None
            if isinstance(dependencies,str):
                ## sometimes dependencies end up as a string instead
                ## of a list of strings
                dependencies = (dependencies,)
            if self.verbose:
                print(''.join(['    ' for t in range(depth)])
                      +f'Attempting to infer {repr(key)} from {repr(dependencies)}')
            try:
                for dependency in dependencies:
                    self._infer(dependency,copy(already_attempted),depth=depth+1) # copy of already_attempted so it will not feed back here
                ## compute value if dependencies successfully inferred
                    
                self.set(key,function(self,*[self[dependency] for dependency in dependencies]),_inferred=True)
                ## compute uncertainties by linearisation
                if unc_function is None:
                    squared_contribution = []
                    value = self[key]
                    parameters = [self[t] for t in dependencies]
                    for i,dependency in enumerate(dependencies):
                        if self.is_set(dependency,'unc'):
                            if 'step' in self._data[key]['associated_data']:
                                step = self.get(dependency,'step')
                            else:
                                step = 1e-8    
                            parameters[i] = self[dependency] + step # shift one
                            dvalue = value - function(self,*parameters)
                            parameters[i] = self[dependency] # put it back
                            squared_contribution.append((self.get(dependency,'unc')*dvalue/step)**2)
                    if len(squared_contribution)>0:
                        self.set(key,np.sqrt(np.sum(squared_contribution,axis=0)),associated_data='unc')
                else:
                    ## args for uncertainty_function.  First is the
                    ## result of calculating keys, after that paris of
                    ## dependencies and their uncertainties, if they
                    ## have no uncertainty then None is substituted.
                    args = [self,self[key]]
                    for dependency in dependencies:
                        if self.is_set(dependency,'unc'):
                            args.extend((self[dependency],self.get(dependency,'unc')))
                        else:
                            args.extend((self[dependency],None))
                    try:
                        self.set(key,unc_function(*args),'unc')
                    except InferException:
                        pass
                ## if we get this far without an InferException then
                ## success!.  Record inference dependencies.
                ## replace this block with
                self._add_dependency(key,dependencies)
                break           
            ## some kind of InferException, try next set of dependencies
            except InferException as err:
                if self.verbose:
                    print(''.join(['    ' for t in range(depth)])
                          +'    InferException: '+str(err))
                continue      
        ## complete failure to infer
        else:
            raise InferException(f"Could not infer key: {repr(key)}")

    def _add_dependency(self,key_inferred_to,keys_inferred_from):
        """Set a dependence connection between a key and its
        dependencies."""
        self._data[key_inferred_to]['inferred_from'].extend(keys_inferred_from)
        for key in keys_inferred_from:
            self._data[key]['inferred_to'].append(key_inferred_to)

    def as_dict(self,keys=None,index=None,include_opt=False):
        """Return as a dict of arrays, including uncertainties."""
        if keys is None:
            keys = self.keys()
        retval = {}
        for key in keys:
            retval[key] = self.get(key,index=index)
        return retval
        
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

    def find_unique(self,**matching_keys_vals):
        """Return index of a uniquely matching row."""
        i = tools.find(self.match(**matching_keys_vals))
        if len(i) == 0:
            raise Exception(f'No matching row found: {matching_keys_vals=}')
        if len(i) > 1:
            raise Exception(f'Multiple matching rows found: {matching_keys_vals=}')
        return i[0]

    def matching_row(self,return_index=False,**matching_keys_vals):
        """Return uniquely-matching row as a dictionary."""
        i = self.find_unique(**matching_keys_vals)
        d = self.as_dict(index=i)
        if return_index:
            return d,i
        else:
            return d

    def matching_value(self,key,**keys_vals):
        """Return value of key from a row that uniquely matches
        keys_vals."""
        i = self.find_unique(**matching_keys_vals)
        value = self.get(key,index=i)
        return value

    def sort(self,*sort_keys,reverse_order=False):
        """Sort rows according to key or keys."""
        i = np.argsort(self[sort_keys[0]])
        if reverse_order:
            i = i[::-1]
        for key in sort_keys[1:]:
            i = i[np.argsort(self[key][i])]
        self.index(i)

    def format(
            self,
            keys=None,
            delimiter=' | ',
            format_opt=True,
            unique_values_in_header=True,
            include_description=True,
            include_attributes=True,
            quote_strings=False,
            quote_keys=False,
    ):
        """Format data into a string representation."""
        if len(self)==0:
            return ''
        if keys is None:
            keys = self.keys()
        ## data to store in header
        ## collect columns of data -- maybe reducing to unique values
        columns = []
        header_values = {}
        for key in keys:
            formatted_key = ( "'"+key+"'" if quote_keys else key )
            if (unique_values_in_header
                and len(tval:=self.unique(key)) == 1
                ):
                ## format value for header
                header_values[key] = tval[0]
            else:
                ## two passes required on all data to align column
                ## widths
                vals = [format(t,self._data[key]['fmt']) for t in self.get(key)]
                if quote_strings and self._data[key]['kind'] == 'U':
                    vals = ["'"+val+"'" for val in vals]
                width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
        ## construct header before table
        header = []
        ## add attributes to header
        if include_attributes:
            for key,val in self.attributes.items():
                if val is not None:
                    header.append(f'{key:12} = {repr(val)}')
        if include_description:
            ## include description of keys
            for key in self:
                line = f'{key:12}'
                if key in header_values:
                    line += f' = {repr(header_values[key]):20}'
                else:
                    line += f'{"":23}'    
                line += f' # '+self._data[key]['description']
                if (units:=self.get_units(key)) is not None:
                    line += f' [{units}]'
                header.append(line)
        else:
            for key,val in header_values.items():
                header.append(f'{key:12} = {repr(val)}')
        ## make full formatted string
        retval = ''
        if header != []:
            retval = 'header\n'+'\n'.join(header)+'\ndata\n'
        if columns != []:
            retval += '\n'.join([delimiter.join(t) for t in zip(*columns)])+'\n'
        return retval

    def format_as_list(self):
        """Form as a valid python list of lists."""
        retval = f'[ \n'
        data = self.format(
            delimiter=' , ',
            format_unc=True,
            format_vary=True,
            format_step= True,
            unique_values_in_header=False,
            include_description=False,
            include_attributes=False,
            quote_strings=True,
            quote_keys=True,
        )
        for line in data.split('\n'):
            if len(line)==0:
                continue
            retval += '    [ '+line+' ],\n'
        retval += ']'
        return retval

    def __str__(self):
        return self.format(
            delimiter=' | ',
            # format_unc=True,
            # format_vary=True,
            # format_step=True,
            unique_values_in_header=True,
            include_description=False,
        )

    def _save_to_hdf5(self,filename,keys):
        """Save self to hdf5 format"""
        import h5py
        filename = tools.expand_path(filename)
        tools.mkdir(tools.dirname(filename)) # make leading directories if not currently there
        with h5py.File(filename,mode='w') as fid:
            ## add global attributes
            for key,val in self.attributes.items():
                if val is not None:
                    fid.attrs.create(key,val)
            ## add data 
            for key in self:
                val = self[key]
                if not np.isscalar(val) and val.dtype.kind=='U':
                    val = np.array(val, dtype=h5py.string_dtype(encoding='utf-8'))
                dset = fid.create_dataset(key,data=val)
                ## add selected prototypes of data as attributes
                data = self._data[key]
                for tkey in ('description','units'):
                    if tkey in data:
                        dset.attrs.create(tkey,data[tkey])
        
    def save(
            self,
            filename,
            keys=None,
            **format_kwargs,
    ):
        """Save some or all data to a file."""
        if keys is None:
            keys = self.keys()
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(
                filename,
                **self.as_dict(),
                **{key:val for key,val in self.attributes.items() if val is not None}
            )
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            self._save_to_hdf5(filename,keys)
            # tools.dict_to_hdf5(
                # filename,
                # self.as_dict(),
                # attributes={key:val for key,val in self.attributes.items() if val is not None})
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                format_kwargs.setdefault('delimiter',', ')
            elif re.match(r'.*\.rs',filename):
                format_kwargs.setdefault('delimiter',' ␞ ')
            elif re.match(r'.*\.psv',filename):
                format_kwargs.setdefault('delimiter',' | ')
            else:
                format_kwargs.setdefault('delimiter',' ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))

    def load(
            self,
            filename,
            comment='',
            keys=None,          # load only this data
            delimiter=None,
            table_name=None,
            translate_keys=None, # from key in file to key in self, None for skip
            return_classname_only=False, # do not load the file -- just try and load the classname and return it
            vector_data_labels_commented=False,
            **set_keys_vals   # set this data after loading is done
    ):
        '''Load data from a file.'''
        if re.match(r'.*\.(h5|hdf5)',filename):
            ## hdf5 archive, load data then top-level attributes
            data =  tools.hdf5_to_dict(filename)
            import h5py
            with h5py.File(tools.expand_path(filename),'r') as fid:
                for key,val in fid.attrs.items():
                    data[key] = val
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
            elif re.match(r'.*\.tsv',filename):
                delimiter = '\t'
            # assert comment not in ['',' '], "Not implemented"
            filename = tools.expand_path(filename)
            data = {}
            ## load header
            escaped_comment = re.escape(comment)
            blank_line_re = re.compile(r'^ *$')
            description_line_re = re.compile(f'^ *{escaped_comment} *([^# ]+) *# *(.+) *') # no value in line
            unique_value_line_re = re.compile(f'^ *{escaped_comment} *([^= ]+) *= *([^#]*[^ #\\n])') # may also contain description
            beginning_of_header_re = re.compile(f'^ *{escaped_comment} *header *\\n$') 
            beginning_of_data_re = re.compile(f'^ *{escaped_comment} *data *\\n$') 
            with open(filename,'r') as fid:
                for iline,line in enumerate(fid):
                    if re.match(blank_line_re,line):
                        continue
                    elif r:=re.match(description_line_re,line):
                        continue
                    elif r:=re.match(beginning_of_header_re,line):
                        continue
                    elif r:=re.match(unique_value_line_re,line):
                        key,val = r.groups()
                        data[key] = ast.literal_eval(val)
                    elif r:=re.match(beginning_of_data_re,line):
                        ## end of header
                        iline += 1 
                        break
                    else:
                        ## end of header
                        break
            ## load array data
            data.update(tools.txt_to_dict(
                filename,
                delimiter=delimiter,
                labels_commented=vector_data_labels_commented,
                skiprows=iline))

        ## temp hack delete
        for key in list(data.keys()): #  HACK
            if 'HT_HITRAN' in key:    #  HACK
                data[key.replace('HT_HITRAN','HITRAN_HT')] = data.pop(key) #  HACK
        ## end of temp hack delete

        ## translate keys
        if translate_keys is not None:
            for from_key,to_key in translate_keys.items():
                if from_key in data:
                    if to_key is None:
                        data.pop(from_key)
                    else:
                        data[to_key] = data.pop(from_key)
        ## test for a matching classname, return if requested or make
        ## sure it matches self
        if return_classname_only:
            if 'classname' in data:
                return data['classname']
            else:
                return None
        if 'classname' in data and data['classname'] != self.attributes['classname']:
            warnings.warn(f'The loaded classname, {repr(data["classname"])}, does not match self, {repr(self["classname"])}, and it will be ignored.')
            data.pop('classname')
        ## Set data in self and selected attributes
        if keys is None:
            keys = data.keys()
        for key in keys:
            if key in self.attributes:
                self.attributes[key] = data[key]
            else:
                self[key] = data[key]

    def load_from_string(
            self,
            string,             # multi line string in the format expected by self.load
            delimiter='|',      # column delimiter
            **load_kwargs       # other kwargs passed to self.load
    ):     
        """Load data from a delimter and newline separated string."""
        ## Write a temporary file and then uses the regular file load
        tmpfile = tools.tmpfile()
        tmpfile.write(string.encode())
        tmpfile.flush()
        tmpfile.seek(0)
        self.load(tmpfile.name,delimiter=delimiter,**load_kwargs)

    def load_from_lists(self,keys,*values):
        """Add many lines of data efficiently, with values possible
        optimised."""
        cache = {}
        if len(cache) == 0:
            ## first construct
            cache['ibeg'] = len(self)
            cache['keys_vals'] = {key:[t[j] for t in values] for j,key in enumerate(keys)}
        for key,val in cache['keys_vals'].items():
            self[key] = val
        def format_input_function():
            retval = self.format_as_list()
            retval = f'{self.name}.load_from_lists(' + retval[1:-1] + ')'
            return retval
        self.add_format_input_function(format_input_function)

    def append(self,keys_vals_as_dict=idict(),keys='all',**keys_vals_as_kwargs):
        """Append a single row of data from kwarg scalar values."""
        keys_vals = dict(**keys_vals_as_dict,**keys_vals_as_kwargs)
        for key in keys_vals:
            keys_vals[key] = [keys_vals[key]]
        self.extend(keys_vals,keys=keys)

    def extend(
            self,
            keys_vals_as_dict_or_dataset=idict(),
            keys='all',         # 'old','new','all'
            **keys_vals_as_kwargs
    ):
        """Extend self with new_data.  Keys must be present in both new and
        old data.  If keys='old' then extra keys in new data are
        ignored. If keys='new' then extra keys in old data are unset.
        If 'all' then keys must match exactly.  If key=='new' no data
        currently present then just add this data."""  
        ## select keys, delete all inferred data first
        self.unset_inferred()
        if isinstance(keys_vals_as_dict_or_dataset, Dataset):
            keys_vals_as_dict_or_dataset.unset_inferred()
        ## get a list of keys, or shortcut if currently no data
        if len(self) == 0 and keys in ('all','new'):
            ## copy new data directly if no existing data
            if keys == 'new':
                ## delete unnecessary data in self
                keys = list(keys_vals_as_dict_or_dataset) + list(keys_vals_as_kwargs)
                for key in self:
                    if key not in keys:
                        self.unset(key)
            for key,val in keys_vals_as_dict_or_dataset.items():
                self[key] = val
            for key,val in keys_vals_as_kwargs.items():
                self[key] = val
            return
        else:
            if keys == 'all':
                keys = {*self,*keys_vals_as_dict_or_dataset,*keys_vals_as_kwargs}
            elif keys == 'old':
                keys = {*self}
            elif keys == 'new':
                keys = {*keys_vals_as_dict_or_dataset,*keys_vals_as_kwargs}
                ## delete unnecessary data in self
                for key in list(self):
                    if key not in keys:
                        self.unset(key)
            else:
                raise Exception(f"valid keys options: 'all', 'new', 'old'")
        ## search for needed keys in self -- infer if necessary, end
        ## by deleting all temporarily inferred data
        for key in keys:
            if not self.is_known(key):
                raise Exception(f"Extending key not in existing data: {repr(key)}")
            else:
                self.unlink_inferences(key)
        self.unset_inferred()
        ## collect keys from the new data
        new_data = {}
        for key in keys:
            if key in keys_vals_as_dict_or_dataset:
                new_data[key] = keys_vals_as_dict_or_dataset[key]
            elif key in keys_vals_as_kwargs:
                new_data[key] = keys_vals_as_kwargs[key]
            elif (isinstance(keys_vals_as_dict_or_dataset,Dataset)
                  and keys_vals_as_dict_or_dataset.is_known(key)):
                new_data[key] = keys_vals_as_dict_or_dataset[key]
            elif self.get_default(key) is not None:
                new_data[key] = self.get_default(key)
            ## could add logic for auto defaults based on kind as below
            else:
                raise Exception(f'Extending key missing in new data: {repr(key)}')
        ## determine length of data
        original_length = len(self)
        extending_length = None
        for key,val in new_data.items():
            if tools.isiterable(val):
                if extending_length is None:
                    extending_length = len(val)
                elif extending_length != len(val):
                    raise Exception(f'Mismatched lengths in extending data')
        if extending_length is None:
            raise Exception("No vector data in new data")
        total_length = original_length + extending_length
        ## add new data to old
        for key in keys:
            if key not in self:
                self.set(key,new_data[key])
            else:
                ## the object in self to extend
                data = self._data[key] 
                ## increase unicode dtype length if new strings are
                ## longer than the current
                if self.get_kind(key) == 'U':
                    ## this is a really hacky way to get the length of string in a numpy array!!!
                    old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self.get(key).dtype)))
                    new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(val).dtype)))
                    if new_str_len > old_str_len:
                        ## reallocate array with new dtype with overallocation
                        t = np.empty(
                            len(self)*self._over_allocate_factor,
                            dtype=f'<U{new_str_len*self._over_allocate_factor}')
                        t[:len(self)] = self.get(key)
                        data['value'] = t
                ## reallocate and lengthen arrays if necessary
                if total_length > len(data['value']):
                    data['value'] = np.concatenate((
                        self[key],
                        np.empty(int(total_length*self._over_allocate_factor-original_length),
                                 dtype=data['value'].dtype)))
                ## set extending data
                data['value'][original_length:total_length] = data['cast'](new_data[key])
        ## finalise new length
        self._length = total_length

    def __add__(self,other):
        """Adding dataset concatenates data in all keys."""
        retval = self.copy()
        retval.extend(other)
        return retval

    def __radd__(self,other):
        """Adding dataset concatenates data in all keys."""
        retval = self.copy()
        retval.extend(other)
        return retval

    def plot(
            self,
            xkey,               # key to use for x-axis data
            ykeys,              # list of keys to use for y-axis data
            zkeys=None,         # plot x-y data separately for unique combinations of zkeys
            fig=None,           # otherwise automatic
            ax=None,            # otherwise automatic
            ynewaxes=True,      # plot y-keys on separates axes -- else as different lines
            znewaxes=False,     # plot z-keys on separates axes -- else as different lines
            legend=True,        # plot a legend or not
            annotate_lines=False, # annotate lines with their labels
            zlabel_format_function=None, # accept key=val pairs, defaults to printing them
            label_prefix=None, # put this before label otherwise generated
            plot_errorbars=True, # if uncertainty available
            xscale='linear',     # 'log' or 'linear'
            yscale='linear',     # 'log' or 'linear'
            ncolumns=None,       # number of columsn of subplot -- None to automatically select
            show=False,          # show figure after issuing plot commands
            **plot_kwargs,      # e.g. color, linestyle, label etc
    ):
        """Plot data."""
        from matplotlib import pyplot as plt
        from spectr import plotting
        if len(self)==0:
            return
        ## re-use or make a new figure/axes
        if ax is not None:
            ynewaxes,znewaxes = False,False
            fig = ax.figure
        if fig is None:
            fig = plt.gcf()
            fig.clf()
        ## xkey, ykeys, zkeys
        if xkey == 'index':
            if 'index'  in self.keys():
                raise Exception("Index already exists")
            self['index'] = np.arange(len(self),dtype=int)
            xkey = 'index'
        if zkeys is None:
            zkeys = self.default_zkeys
        zkeys = [t for t in tools.ensure_iterable(zkeys) if t not in ykeys and t!=xkey and self.is_known(t)] # remove xkey and ykeys from zkeys
        ykeys = [key for key in tools.ensure_iterable(ykeys) if key not in [xkey]+zkeys]
        self.assert_known(xkey,*ykeys,*zkeys)
        ## plot each 
        ymin = {}
        for iy,ykey in enumerate(tools.ensure_iterable(ykeys)):
            ylabel = ykey
            for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                z.sort(xkey)
                if zlabel_format_function is None:
                    # if len(dz) == 1:
                        # zlabel = str(dz[list(dz.keys())[0]])
                    # else:
                        # zlabel = tools.dict_to_kwargs(dz)
                    zlabel = tools.dict_to_kwargs(dz)
                else:
                    zlabel = zlabel_format_function(**dz)
                if ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz+len(zkeys)*iy,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = None
                    title = ylabel+' '+zlabel
                elif ynewaxes and not znewaxes:
                    ax = plotting.subplot(n=iy,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = (zlabel if len(zkeys)>0 else None) 
                    title = ylabel
                elif not ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = ylabel
                    title = zlabel
                elif not ynewaxes and not znewaxes:
                    ax = fig.gca()
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    # color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = ylabel+' '+zlabel
                    title = None
                if label_prefix is not None and label is not None:
                    label = label_prefix + label
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
                if plot_errorbars and z.is_set(ykey,'unc'):
                    dy = z.get(ykey,'unc')
                    ## plot errorbars
                    kwargs.setdefault('mfc','none')
                    dy[np.isnan(dy)] = 0.
                    ax.errorbar(x,y,dy,**kwargs)
                    ## plot zero/undefined uncertainty data as filled symbols
                    i = np.isnan(dy)|(dy==0)
                    if np.any(i):
                        kwargs['mfc'] = kwargs['color']
                        if 'fillstyle' not in kwargs:
                            kwargs['fillstyle'] = 'full'
                        if 'ls' in kwargs:
                            kwargs['ls'] = ''
                        else:
                            kwargs['linestyle'] = ''
                        kwargs['label'] = None
                        line = ax.plot(z[xkey][i],z[ykey][i],**kwargs)
                else:
                    kwargs.setdefault('mfc',kwargs['color'])
                    kwargs.setdefault('fillstyle','full')
                    line = ax.plot(x,y,**kwargs)
                if title is not None:
                    ax.set_title(title)
                if 'label' in kwargs:
                    if legend:
                        plotting.legend(fontsize='x-small')
                    if annotate_lines:
                        plotting.annotate_line(line=line)
                ax.set_xlabel(xkey)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True,color='gray',zorder=-5)
                if self.get_kind(xkey) == 'U':
                    plotting.set_tick_labels_text(x, axis='x', ax=ax, rotation=70,)
        if show:
            plotting.show()
        return fig

    def polyfit(self,xkey,ykey,index=None,**polyfit_kwargs):
        return tools.polyfit(
            self.get(xkey,index=index),
            self.get(ykey,index=index),
            self.get(ykey,'unc',index),
            **polyfit_kwargs)

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
    return x[i],y[j]

def _get_class(classname):
    """Find a class matching class name."""
    if classname == 'dataset.Dataset':
        return Dataset
    else:
        module,subclass = classname.split('.')
        if module == 'lines':
            from . import lines
            return getattr(lines,subclass)
        elif module == 'levels':
            from . import levels
            return getattr(levels,subclass)
    raise Exception(f'Could not find a class matching {classname=}')
    
def make(classname='dataset.Dataset',*args,**kwargs):
    """Make an instance of the this classname."""
    return _get_class(classname)(*args,**kwargs)

def load(
        filename,
        classname=None,
        # translate_keys=None,
        # vector_data_labels_commented=False,
        **load_kwargs
):
    """Load a Dataset.  Attempts to automatically find the correct
    subclass if it is not provided as an argument, but this requires
    loading the file twice."""
    if classname is None:
        d = Dataset()
        classname = d.load(filename,return_classname_only=True,**load_kwargs)
        if classname is None:
            classname = 'dataset.Dataset'
    retval = make(classname)
    retval.load(filename,**load_kwargs)
    return retval

def copy_from(dataset,*args,**kwargs):
    """Make a copy of dataset with additional initialisation args and
    kwargs."""
    classname = dataset['classname'] # use the same class as dataset
    retval = make(classname,*args,copy_from=dataset,**kwargs)
    return retval
