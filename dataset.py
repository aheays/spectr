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
import h5py

from . import tools
from .tools import AutoDict,convert_to_bool_vector_array
from .exceptions import InferException
from . import convert
from . import optimise
from .optimise import optimise_method,Parameter,Fixed



class Dataset(optimise.Optimiser):

    """A set of data vectors of common length."""


    ## basic kinds of data
    kinds = {
        'f':    {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt':'+12.8e','default':nan   ,'description':'float'                                  ,},
        'i':    {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt':'d'     ,'default':-999  ,'description':'int'                                    ,},
        'b':    {'cast':convert_to_bool_vector_array       ,'fmt':''      ,'default':True  ,'description':'bool'                                   ,},
        'U':    {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt':'s'     ,'default':''    ,'description':'str'                                    ,},
        'O':    {'cast':lambda x:np.asarray(x,dtype=object),'fmt':''      ,'default':None  ,'description':'object'                                 ,},
    }

    ## associated data
    associated_kinds = {
        'unc'      : {'description':'Uncertainty'                         , 'kind':'f' , 'valid_kinds':('f',), 'cast':lambda x:np.abs(x,dtype=float)     ,'fmt':'8.2e'  ,'default':0.0   ,},
        'step'     : {'description':'Numerical differentiation step size' , 'kind':'f' , 'valid_kinds':('f',), 'cast':lambda x:np.abs(x,dtype=float)     ,'fmt':'8.2e'  ,'default':1e-8  ,},
        'vary'     : {'description':'Whether to vary during optimisation' , 'kind':'b' , 'valid_kinds':('f',), 'cast':convert_to_bool_vector_array       ,'fmt':''      ,'default':False ,},
        'residual' : {'description':'Residual error'                      , 'kind':'f' , 'valid_kinds':('f',), 'cast':lambda x:np.asarray(x,dtype=float) ,'fmt':'8.2e'  ,'default':nan   ,},
        'ref'      : {'description':'Source reference'                    , 'kind':'U' , 'valid_kinds':('f',), 'cast':lambda x:np.asarray(x,dtype='U20') ,'fmt':'s'     ,'default':nan   ,},
    }

    ## always available
    # default_attributes = {
        # 'classname':None,
        # 'description':None,
        # 'default_step':1e-8,
    # }

    ## prototypes on instantiation
    default_prototypes = {}

    ## used for plotting and sorting perhaps
    default_zkeys = ()
    default_zlabel_format_function = tools.dict_to_kwargs


    def __init__(
            self,
            name=None,
            permit_nonprototyped_data = True,
            permit_indexing = True,
            auto_defaults = False, # if no data or way to infer it, then set to default prototype value if this data is needed
            prototypes = None,  # a dictionary of prototypes
            load_from_file = None,
            load_from_string = None,
            copy_from = None,
            limit_to_match=None, # dict of things to match
            description='',
            **kwargs):
        ## basic internal variables
        self._data = {} # table data and its properties stored here
        self._length = 0    # length of data
        self._over_allocate_factor = 2 # to speed up appending to data
        # self.attributes = {} # applies to all data
        self.description = description
        self._last_modify_data_time = timestamp() # when data is changed this is update
        self.permit_nonprototyped_data = permit_nonprototyped_data # allow the addition of data not in self.prototypes
        self.permit_indexing = permit_indexing # Data can be added to the end of arrays, but not removal or rearranging of data
        self.auto_defaults = auto_defaults # set default values if necessary automatically
        self.verbose = False                             # print extra information at various places
        ## get prototypes from defaults and then input argument
        self.prototypes = copy(self.default_prototypes)
        if prototypes is not None:
            self.prototypes |= prototypes
        ## classname to identify type of Dataset
        self.classname = re.sub(
            r"<class 'spectr.(.+)'>",
            r'\1',
            str(self.__class__))
        ## default name is a valid symbol
        if name is None:
            name = tools.make_valid_python_symbol_name(
                self.classname.lower())
        ## init as optimiser, make a custom form_input_function, save
        ## some extra stuff if output to directory
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        ## new format input function
        def format_input_function():
            retval = f'{self.name} = {self.classname}({repr(self.name)},'
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
            if isinstance(val,optimise.Parameter):
                ## an optimisable parameter (input function already
                ## handled)
                self.set_and_optimise(key,val)
                self.pop_format_input_function()
            elif np.isscalar(val):
                ## set default value
                self.set_default(key,val)
            else:
                ## set data
                self[key] = val
        ## limit to matching data somehow loaded above
        if limit_to_match is not None:
            self.limit_to_match(**limit_to_match)

    def __len__(self):
        return self._length

    def set(self,key_assoc,value,index=None,match=None,**match_kwargs):
        """Set value of key or (key,assoc)"""
        key,assoc = self._separate_key_assoc(key_assoc)
        match = {**({} if match is None else match),**match_kwargs}
        ## set value or associated data
        if assoc is None:
            return self._set_value(key,value,index,match)
        else:
            return self._set_associated_data(key,assoc,value,index,match)
        
    def _set_value(self,key,value,index=None,match=None,dependencies=None,**prototype_kwargs):
        """Set a value"""
        ## determine index
        if match is not None and len(match) > 0:
            if index is None:
                index = self.match(match)
            else:
                index &= self.match(match)
        ## update modification if externally set, not if it is inferred
        if self.verbose:
            print(f'{self.name}: setting {key}')
        ## explicitly set data
        if dependencies is None:
            ## self has changed explicitly
            self._last_modify_data_time = timestamp()
            ## if data already existed delete anything inferred from it
            if key in self:
                self.unlink_inferences(key)
        ## if an index is provided then data must already exist, set
        ## new indeed data and return
        if index is not None:
            if key not in self:
                if 'default' in prototype_kwargs:
                    self.set(key,value=prototype_kwargs['default'])
                else:
                    raise Exception(f'Setting {repr(key)} for (possible) partial indices but it is not already set')
            data = self._data[key]
            if key not in self:
                raise Exception(f'Cannot set data by index for unset key: {key}')
            data['value'][:self._length][index] = data['cast'](value)
        ## set full array -- does not hvae to exist in in advance
        else:
            ## create entire data dict
            ## decide whether to permit if non-prototyped
            if not self.permit_nonprototyped_data and key not in self.prototypes:
                raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = {'assoc':{},##'default':None,'units':None,
                    'infer':[],'inferred_from':[],'inferred_to':[]}
            self._data[key] = data
            data['assoc'] = {}
            ## get any prototype data
            if key in self.prototypes:
                for tkey,tval in self.prototypes[key].items():
                    if tkey == 'default' and not self.auto_defaults:
                        ## do not set default from prototypes -- only 
                        continue
                    data[tkey] = tval
            ## apply prototype kwargs
            for tkey,tval in prototype_kwargs.items():
                data[tkey] = tval
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
                data['kind'] = 'U'
            ## some other prototype data based on kind
            for tkey in ('description','fmt','cast'):
                if tkey not in data:
                    data[tkey] = self.kinds[data['kind']][tkey]
            if data['kind']=='f' and 'default_step' not in data:
                data['default_step'] = 1e-8
            ## if a scalar expand to length of self
            ## default
            if not tools.isiterable(value):
                ## data['default'] = value
                value = np.full(len(self),value)
            ## If this is the nonzero-length data set then increase
            ## length of self and set any other keys with defaults to
            ## their default values
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
            data['value'] = data['cast'](value)
        ## If this is inferred data then record dependencies
        if dependencies is not None:
            self._set_dependency(key,dependencies)

    def _set_associated_data(self,key,assoc,value,index=None,match=None):
        """Set associated data for key to value."""
        ## basic error checks
        if assoc not in self.associated_kinds:
            raise Exception(f'Invalid associated data: {repr(assoc)}')
        if not self.is_known(key):
            raise Exception(f"Key {repr(key)} must be set before setting associated data {repr(assoc)}")
        if self.get_kind(key) not in self.associated_kinds[assoc]['valid_kinds']:
            raise Exception(f"Key {repr(key)} is not a valid kind for associated data {repr(assoc)}")
        if self.associated_kinds[assoc]['kind'] == 'O':
            raise ImplementationError()
        if self.verbose:
            print(f'{self.name}: setting ({key},{assoc})')
        ## determine index
        if match is not None and len(match) > 0:
            if index is None:
                index = self.match(match)
            else:
                index &= self.match(match)
        ## set data
        if index is None:
            ## set entire array
            if not tools.isiterable(value):
                ## expand scalar input
                value = np.full(len(self),value)
            elif len(value) != len(self):
                raise Exception(f'Length of new ({key},{assoc}) ({len(value)} does not match existing data ({len(self)})')
            ## set data
            self._data[key]['assoc'][assoc] = self.associated_kinds[assoc]['cast'](value)
        else:
            ## set part of array by index
            if assoc not in self._data[key]['assoc']:
                ## set missing data outside indexed range to a default
                ## value using the get method
                self._get_associated_data(key,assoc)
            ## set indexed data
            self._data[key]['assoc'][assoc][:len(self)][index] = self.associated_kinds[assoc]['cast'](value)

    def _set_dependency(self,key,dependencies):
        """Set a dependence connection between a key and its
        dependencies."""
        if self.verbose:
            print(f'{self.name}: Setting dependencies for {repr(key)}: {repr(list(dependencies))}')
        self._data[key]['inferred_from'].extend(dependencies)
        for dependency in dependencies:
            self._data[dependency]['inferred_to'].append(key)

    def get(self,key,index=None,units=None,match=None,**match_kwargs):
        """Get value for key or (key,assoc)."""
        match = {**({} if match is None else match),**match_kwargs}
        if len(match) > 0:
            if index is None:
                index = self.match(match)
            else:
                index &= self.match(match)
        if isinstance(key,str):
            return self._get_data(key,index,units)
        else:
            key,assoc = key
            return self._get_associated_data(key,assoc,index,units)

    def _get_data(self,key,index=None,units=None):
        """Get value for key."""
        ## return vector data
        if key not in self._data:
            try:
                ## attempt to infer
                self._infer(key)
            except InferException as err:
                if key in self.prototypes and 'default' in self.prototypes[key]:
                    self[key] = self.prototypes[key]['default']
                else:
                    raise err
        if index is None:
            ## get default entire index
            index = slice(0,len(self))
        data = self._data[key]
        retval = data['value'][:self._length][index]
        if units is not None:
            ## convert units before setting
            retval = convert.units(retval,self._data[key]['units'],units)
        return retval

    def _get_associated_data(self,key,assoc,index=None,units=None):
        """Get associatedf value."""
        ## basic error checks
        if assoc not in self.associated_kinds:
            raise Exception(f'Invalid associated data: {repr(assoc)}')
        if not self.is_known(key):
            raise Exception(f"Key {repr(key)} must be set before getting associated data {repr(assoc)}")
        if self.get_kind(key) not in self.associated_kinds[assoc]['valid_kinds']:
            raise Exception(f"Key {repr(key)} is not a valid kind for associated data {repr(assoc)}")
        ## get default index
        if index is None:
            index = slice(0,len(self))
        ## set default if value if necessary, look for default_assoc
        ## from prototypes or use associated_kinds default
        if not self.is_set((key,assoc)):
            if (tkey:=f'default_{assoc}') in self._data[key]:
                default_value = self._data[key][tkey]
            else:
                default_value = self.associated_kinds[assoc]['default']
            self._set_associated_data(key,assoc,default_value)
        ## return data
        retval = self._data[key]['assoc'][assoc][0:len(self)][index]
        if units is not None:
            retval = convert.units(retval,self._data[key]['units'],units)
        return retval

    def set_default(self,key=None,value=None,**keys_values):
        """Set default value for key, and set existing data to this value if
        not already set."""
        if key is not None:
            keys_values[key] = value
        for key,value in keys_values.items():
            if key not in self:
                self[key] = value
            self._data[key]['default'] = value

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
        self.prototypes[key] = dict(kind=kind,**kwargs)
        for tkey,tval in self.kinds[kind].items():
            self.prototypes[key].setdefault(tkey,tval)

    def get_kind(self,key):
        return self._data[key]['kind']

    def _get_combined_index(self,index,match,**match_kwargs):
        """Combined specified index with match arguments as boolean mask. If
        no data given the return None"""
        if index is None and match is None and len(match_kwargs)==0:
            retval = None
        else:
            if index is None:
                retval = np.full(len(self),True)
            elif np.isscalar(index):
                retval = np.full(len(self),bool(index))
            elif isinstance(index,slice):
                retval = np.full(len(self),False)
                retval[index] = True
            else:
                retval = np.asarray(index)
                if retval.dtype == bool:
                    pass
                elif retval.dtype == int:
                    retval = tools.find_inverse(retval,len(self))
            retval &= self.match(match,**match_kwargs)
        return retval

    @optimise_method(format_lines='single')
    def set_and_optimise(
            self,
            key,
            value,          # a scalar or Parameter
            index=None,         # only apply to these indices
            match=None,
            **match_kwargs
            # **prototype_kwargs,
    ):
        """Set a value and it will be updated every construction and possible
        optimised."""
        index = self._get_combined_index(index,match,**match_kwargs)
        ## if not a parameter then treat as a float -- could use set(
        ## instead and branch there, requiring a Parameter here
        if isinstance(value,Parameter):
            ## only reconstruct for the following reasons
            if (
                    key not in self.keys() # key is unknown -- first run
                    or value._last_modify_value_time > self._last_construct_time # parameter has been set
                    or np.any(self.get(key,index=index) != value.value) # data has changed some other way and differs from parameter
                    or (self.is_set((key,'unc'))
                        and not np.isnan(value.unc)
                        and (np.any(self.get((key,'unc'),index=index) != value.unc))) # data has changed some other way and differs from parameter
                ):
                self.set(key,value.value,index=index)
                self.set((key,'unc'),value.unc,index=index)
                self.set((key,'step'),value.step,index=index)
        else:
            ## only reconstruct for the following reasons
            if (key not in self.keys() # key is unknown -- first run
                or np.any(self.get(key,index=index) != value)): # data has changed some other way and differs from parameter
                self.set(key,value=value,index=index)

    @optimise_method(format_lines='single')
    def set_spline(self,xkey,ykey,knots,order=3,default=None,
                   match=None,index=None,_cache=None,**match_kwargs):
        """Set ykey to spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...]. If index or a match dictionary
        given, then only set these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        if len(_cache) == 0: 
            xspline,yspline = zip(*knots)
            ## get index limit to defined xkey range
            index = self._get_combined_index(index,match,**match_kwargs)
            if index is None:
                index = (self[xkey]>=np.min(xspline)) & (self[xkey]<=np.max(xspline))
            else:
                index &= (self[xkey]>=np.min(xspline)) & (self[xkey]<=np.max(xspline))
            _cache['index'] = index
            _cache['xspline'],_cache['yspline'] = xspline,yspline
        ## get cached data
        index,xspline,yspline = _cache['index'],_cache['xspline'],_cache['yspline']
        ## set data
        if not self.is_known(ykey):
            if default is None:
                raise Exception(f'Setting {repr(ykey)} to spline but it is not known and no default value if provided')
            else:
                self[ykey] = default                                                                                   
            
        self.set(ykey,value=tools.spline(xspline,yspline,self.get(xkey,index=index),order=order),index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set((ykey,'unc')):
            self.set((ykey,'unc'),nan,index=index)

    def keys(self):
        return list(self._data.keys())

    def limit_to_keys(self,keys):
        """Unset all keys except these."""
        keys = tools.ensure_iterable(keys)
        self.assert_known(keys)
        self.unlink_inferences(keys)
        self.unset([key for key in self if key not in keys])

    def optimised_keys(self):
        return [key for key in self.keys() if self.is_set((key,'vary'))]

    def explicitly_set_keys(self):
        return [key for key in self if not self.is_inferred(key)]

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

    def _separate_key_assoc(self,key_assoc):
        if isinstance(key_assoc,str):
            return key_assoc,None
        else:
            if len(key_assoc) != 2:
                raise Exception
            if key_assoc[1] not in self.associated_kinds:
                raise Exception(f'Unknown associated kind: {repr(key_assoc[1])}')
            return key_assoc

    def is_set(self,key_assoc):
        key,assoc = self._separate_key_assoc(key_assoc)
        if key in self._data:
            if assoc is None:
                return True
            if assoc in self._data[key]['assoc']:
                return True
        return False

    def assert_known(self,*key_assoc):
        """Check is known by trying to get item."""
        for t in key_assoc:
            self[t]

    def is_known(self,*key_assoc):
        """Test if key is known."""
        try:
            self.assert_known(*key_assoc)
            return True 
        except InferException:
            return False
            
    def __getitem__(self,index):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(index,str):
            ## a key -- return data
            return self.get(index)
        elif isinstance(index,slice):
            ## a slice -- return indexed copy
            return self.copy(index=index)
        elif isinstance(index,int):
            ## an index -- return as flat dict containing scalar data
            return self.as_flat_dict(index=index)
        elif tools.isiterable(index):
            if len(index) == 0:
                ## empty index, make an empty copy of self
                return self.copy(index=index)
            elif isinstance(index[0],str):
                if isinstance(index,tuple):
                    if len(index) == 2:
                        if  isinstance(index[1],str):
                            ## (key,assoc) tuple, return data
                            return self.get(index)
                        else:
                            ## (key,index) tuple, return data
                            return self.get(index[0],index=index[1])
                    elif len(index) == 3:
                        ## (key,assoc,index) tuple, return data
                        return self.get(index[:2],index=index[2])
                    else:
                        raise Exception
                else:
                    ## list of keys, make a copy containing these
                    return self.copy(keys=index)
            else:
                ## array index, make an index copy of self
                return self.copy(index=index)
        else:
            raise Exception(f"Cannot interpret index: {repr(index)}")

    def __setitem__(self,key,value):
        """Set a key to value. If (key,assoc) then set associated data. If
        (key,index/slice) or (key,assoc,index/slice) then set only that
        index/slice."""
        ## look for index
        index = None
        if tools.isiterable(key):
            if not isinstance(key[-1],str):
                ## must be an index/slice
                index = key[-1]
                if len(key) == 3:
                    key = key[0:2]
                elif len(key) == 2:
                    key = key[0]
        ## set
        if isinstance(value,optimise.P):
            self.set_and_optimise(key,value,index=index)
        else:
            self.set(key,value,index=index)
       
    def clear(self):
        """Clear all data"""
        if not self.permit_indexing:
            raise Exception('Cannot clear dataset with not permit_indexing.')
        self._last_modify_value_time = timestamp()
        self._length = 0
        self._data.clear()

    def unset(self,*keys):
        """Delete data.  Also clean up inferences."""
        for key in keys:
            key,assoc = self._separate_key_assoc(key)
            if key not in self:
                continue
            if assoc is None:
                self.unlink_inferences(key)
                data = self._data[key]
                self._data.pop(key)
            else:
                if assoc in self._data[key]['assoc']:
                    self._data[key]['assoc'].pop(assoc)

    def pop(self,key):
        """Return data and unset key."""
        retval = self[key]
        self.unset(key)
        return retval
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
   
    def unlink_inferences(self,keys,unset_inferred=True):
        """Delete any record of inferences to or from the given keys and
        delete anything inferred from these keys (but not if it is  among
        keys itself)."""
        keys = tools.ensure_iterable(keys)
        for t in keys:
            self.assert_known(t)
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
        if not self.permit_indexing:
            raise Exception('Indexing not permitted')
        original_length = len(self)
        # for data in self._data.values():
        for key,data in self._data.items():
            data['value'] = data['value'][:original_length][index]
            for key,value in data['assoc'].items():
                data['assoc'][key] = value[:original_length][index]
            self._length = len(data['value'])

    def remove(self,index):
        """Remove boolean indices."""
        self.index(~index)

    def copy(
            self,
            keys=None,
            index=None,
            name=None,
            match=None,
            copy_assoc=False,
            copy_inferred_data=False,
            **match_kwargs
    ):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if name is None:
            name = f'copy_of_{self.name}'
        retval = self.__class__(name=name) # new version of self
        retval.copy_from(
            self,keys,index,
            copy_assoc=copy_assoc,
            copy_inferred_data=copy_inferred_data,
            match=match,
            **match_kwargs)
        retval.pop_format_input_function()
        return retval

    def copy_from(
            self,
            source,
            keys=None,
            index=None,
            match=None,
            copy_assoc=False,
            copy_inferred_data=False,
            **match_kwargs
    ):
        """Copy all values and uncertainties from source Dataset and update if
        source changes during optimisation."""
        self.clear()            # total data reset
        if keys is None:
            if copy_inferred_data:
                keys = source.keys()
            else:
                keys = source.explicitly_set_keys()
        self.permit_nonprototyped_data = source.permit_nonprototyped_data
        ## get matching indices
        index = source._get_combined_index(index,match,**match_kwargs)
        ## copy data
        for key in keys:
            self[key] = source[key][index]
            if copy_assoc:
                ## copy associated data
                for assoc in source._data[key]['assoc']:
                    self[key,assoc] = source[key,assoc][index]

    @optimise_method()
    def copy_from_and_optimise(
            self,
            source,
            keys=None,
            skip_keys=(),
            index=None,
            match=None,
            copy_assoc=False,
            copy_inferred_data=False,
            _cache=None,
            **match_kwargs
    ):
        """Copy all values and uncertainties from source Dataset and update if
        source changes during optimisation."""
        ## get keys and indices to copy
        if self._clean_construct:
            # self.clear()            # total data reset
            if keys is None:
                if copy_inferred_data:
                    keys = source.keys()
                else:
                    keys = source.explicitly_set_keys()
            keys = [key for key in keys if key not in skip_keys]
            index = source._get_combined_index(index,match,**match_kwargs)
            _cache['keys'],_cache['index'] = keys,index
        else:
            keys,index = _cache['keys'],_cache['index']
        ## copy data
        self.permit_nonprototyped_data = source.permit_nonprototyped_data
        for key in keys:
            self[key] = source[key][index]
            if copy_assoc:
                ## copy associated data
                for assoc in source._data[key]['assoc']:
                    self[key,assoc] = source[key,assoc][index]

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

    def match(self,keys_vals=None,**kwarg_keys_vals):
        """Return boolean array of data matching all key==val.\n\nIf key has
        suffix '_min' or '_max' then match anything greater/lesser
        or equal to this value"""
        if keys_vals is None:
            keys_vals = {}
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
        return self.copy(
            index=self.match(*args,**kwargs),
            copy_assoc=True,
            copy_inferred_data= True,
        )

    def limit_to_match(self,**keys_vals):
        self.index(self.match(**keys_vals))

    def remove_match(self,*match_args,**match_keys_vals):
        self.index(~self.match(*match_args,**match_keys_vals))

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
                function,uncertainty_function = function
            else:
                uncertainty_function = None
            if isinstance(dependencies,str):
                ## sometimes dependencies end up as a string instead
                ## of a list of strings
                dependencies = (dependencies,)
            if self.verbose:
                print(f'{self.name}:',
                      ''.join(['    ' for t in range(depth)])
                      +f'Attempting to infer {repr(key)} from {repr(dependencies)}')
            try:
                for dependency in dependencies:
                    ## use a  copy of already_attempted so it will not feed back here
                    self._infer(dependency,copy(already_attempted),depth=depth+1)
                ## compute value if dependencies successfully
                ## inferred.  If value is None then the data and
                ## dependencies are set internally in the infer
                ## function.
                value = function(self,*[self[dependency] for dependency in dependencies])
                if value is not None:
                    self._set_value(key,value,dependencies=dependencies)
                if self.verbose:
                    print(f'{self.name}: Sucessfully inferred: {repr(key)}')
                ## compute uncertainties by linearisation
                if uncertainty_function is None:
                    squared_contribution = []
                    value = self[key]
                    parameters = [self[t] for t in dependencies]
                    for i,dependency in enumerate(dependencies):
                        if self.is_set((dependency,'unc')):
                            step = self.get((dependency,'step'))
                            parameters[i] = self[dependency] + step # shift one
                            dvalue = value - function(self,*parameters)
                            parameters[i] = self[dependency] # put it back
                            squared_contribution.append((self.get((dependency,'unc'))*dvalue/step)**2)
                    if len(squared_contribution)>0:
                        uncertainty = np.sqrt(np.sum(squared_contribution,axis=0))
                        self._set_associated_data(key,'unc',uncertainty)
                        if self.verbose:
                            print(f'{self.name}: Inferred uncertainty: {repr(key)}')
                else:
                    ## args for uncertainty_function.  First is the
                    ## result of calculating keys, after that paris of
                    ## dependencies and their uncertainties, if they
                    ## have no uncertainty then None is substituted.
                    args = [self,self[key]]
                    for dependency in dependencies:
                        if self.is_set((dependency,'unc')):
                            t_uncertainty = self.get((dependency,'unc'))
                        else:
                            t_uncertainty = None
                        args.extend((self[dependency],t_uncertainty))
                    try:
                        self.set((key,'unc'),uncertainty_function(*args))
                    except InferException:
                        pass
                break           # success
            ## some kind of InferException, try next set of dependencies
            except InferException as err:
                if self.verbose:
                    print(f'{self.name}:',
                          ''.join(['    ' for t in range(depth)])
                          +'    InferException: '+str(err))
                continue     
        else:
            ## not set and cannot infer
            if key in self.prototypes and 'default' in self.prototypes[key]:
                ## use default value. Include empty dependencies so
                ## this is not treated as explicitly set data
                self._set_value(key,self.prototypes[key]['default'],dependencies=())
            else:
                ## complete failure to infer
                raise InferException(f"Could not infer key: {repr(key)}")

    def as_flat_dict(self,keys=None,index=None):
        """Return as a dict of arrays, including uncertainties."""
        if keys is None:
            keys = self.keys()
        retval = {}
        for key in keys:
            retval[key] = self.get(key,index=index)
            if self.is_set((key,'unc')):
                retval[f'{key}_unc'] = self.get((key,'unc'),index=index)
        return retval

    def as_dict(self,keys=None,index=None):
        """Return as a structured dict."""
        ## default to all data
        if keys is None: 
            keys = list(self.keys())
        ## add data
        retval = {}
        retval['classname'] = self.classname
        retval['dataset_description'] = self.description
        for key in keys:
            retval[key] = {}
            for tkey,tval in self._data[key].items():
                if tkey == 'value':
                    ## data
                    retval[key]['value'] = self.get(key,index)
                elif tkey == 'assoc':
                    ## associated data
                    retval[key]['assoc'] = {}
                    for assoc in tval:
                        retval[key]['assoc'][assoc] = self.get((key,assoc),index)
                elif tkey in ('kind','units','description','fmt'):
                    ## attributes
                    retval[key][tkey] = tval
                else:
                    ## do not save anything else
                    pass
        return retval
        
    def rows(self,keys=None):
        """Iterate over data row by row, returns as a dictionary of
        scalar values."""
        if keys is None:
            keys = self.keys()
        for i in range(len(self)):
            yield(self.as_flat_dict(keys=keys,index=i))

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
        d = self.as_flat_dict(index=i)
        if return_index:
            return d,i
        else:
            return d

    def matching_value(self,key,**matching_keys_vals):
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
            unique_values_in_header=True,
            include_description=True,
            include_classname=True,
            include_key_description=True,
            include_assoc=True,
            include_keys_with_leading_underscore=False,
            quote_strings=False,
            quote_keys=False,
    ):
        """Format data into a string representation."""
        if keys is None:
            keys = self.keys()
            if not include_keys_with_leading_underscore:
                keys = [key for key in keys if key[0]!='_']
        ##
        self.assert_known(*keys)
        ## data to store in header
        ## collect columns of data -- maybe reducing to unique values
        columns = []
        header_values = {}
        for key in keys:
            if len(self) == 0:
                break
            formatted_key = ( "'"+key+"'" if quote_keys else key )
            if (
                    unique_values_in_header # input parameter switch
                    and (not include_assoc or len(self._data[key]['assoc'])==0) # neglect if there is associated data (convenience)
                    and len(tval:=self.unique(key)) == 1 # data is unique
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
            ## do everything again for associated data
            if include_assoc:
                for assoc,assoc_value in self._data[key]['assoc'].items():
                    assoc_key = f'{key},{assoc}'
                    vals = [format(t,self.associated_kinds[assoc]['fmt']) for t in assoc_value]
                    if quote_strings and self.associated_kinds[assoc]['kind'] == 'U':
                        vals = ["'"+val+"'" for val in vals]
                    formatted_assoc_key = ( f"'{assoc_key}'" if quote_keys else assoc_key )
                    width = str(max(len(formatted_assoc_key),np.max([len(t) for t in vals])))
                    columns.append([format(formatted_assoc_key,width)]+[format(t,width) for t in vals])
        ## construct header before table
        header = []
        ## add attributes to header
        if include_key_description:
            ## include description of keys
            for key in self:
                line = f'{key:12}'
                if key in header_values:
                    line += f' = {repr(header_values[key]):20}'
                else:
                    line += f'{"":23}'    
                line += f' # '+self._data[key]['description']
                if ('units' in self._data[key]
                    and (units:=self._data[key]['units']) is not None):
                    line += f' [{units}]'
                header.append(line)
        else:
            for key,val in header_values.items():
                header.append(f'{key:12} = {repr(val)}')
        ## make full formatted string
        retval = ''
        if include_classname:
            retval += f'[classname]\n{self.classname}\n'
        if include_description:
            retval += f'[description]\n{self.description}\n'
        if header != []:
            retval += '[keys]\n'+'\n'.join(header)
        if columns != []:
            if len(retval) > 0:
                retval += '\n[data]\n'
            retval += '\n'.join([delimiter.join(t) for t in zip(*columns)])+'\n'
        return retval

    def format_as_list(self):
        """Form as a valid python list of lists."""
        retval = f'[ \n'
        data = self.format(
            delimiter=' , ',
            format_assoc=True,
            unique_values_in_header=False,
            include_description=False,
            quote_strings=True,
            quote_keys=True,
        )
        for line in data.split('\n'):
            if len(line)==0:
                continue
            retval += '    [ '+line+' ],\n'
        retval += ']'
        return retval

    def print_data(self,delimiter=' | '):
        """Print flat data"""
        print(self.format(
            delimiter=delimiter,
            unique_values_in_header=False,
            include_description=False,
            include_assoc=True,
            include_keys_with_leading_underscore=False,
            quote_strings=False,
            quote_keys=False,
        ))


    def __str__(self):
        return self.format(
            delimiter=' | ',
            include_assoc=False,
            unique_values_in_header=False,
            include_description=False,)

    def save(self,filename,keys=None,**format_kwargs):
        """Save some or all data to a file."""
        if keys is None:
            keys = self.keys()
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(filename,self.as_dict())
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            tools.dict_to_hdf5(filename,self.as_dict(),verbose=False)
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
            table_name=None,
            translate_keys=None, # from key in file to key in self, None for skip
            return_classname_only=False, # do not load the file -- just try and load the classname and return it
            labels_commented=False,
            delimiter=None,
            load_assoc=True,
            txt_to_dict_kwargs=None,
            **set_keys_vals   # set this data after loading is done
    ):
        '''Load data from a file.'''
        if re.match(r'.*\.(h5|hdf5)',filename):
            ## hdf5 archive, load data then top-level attributes
            data = tools.hdf5_to_dict(filename)
            ## hack to get flat data or not
            for val in data.values():
                if isinstance(val,dict):
                    data_is_flat = False
                    break
            else:
                data_is_flat = True
        elif re.match(r'.*\.npz',filename):
            ## numpy npz archive.  get as scalar rather than
            ## zero-dimensional numpy array
            data = {}
            for key,val in np.load(filename).items():
                if val.ndim == 0:
                    val = val.item()
                data[key] = val
            data_is_flat = True
        elif re.match(r'.*\.org',filename):
            ## org to dict -- no header
            data = tools.org_table_to_dict(filename,table_name)
            data_is_flat = True
        else:
            data = self.load_from_text(
                filename=filename,
                comment=comment,
                labels_commented=labels_commented,
                delimiter=delimiter,
                txt_to_dict_kwargs=txt_to_dict_kwargs,
            )
            data_is_flat = True
        ## build structured data from flat data by associated keys
        if data_is_flat:
            flat_data = data
            data = {}
            assoc_data = []
            while len(flat_data)>0:
                key = list(flat_data.keys())[0]
                val = flat_data.pop(key)
                if np.isscalar(val):
                    ## attribute
                    data[key] = val
                elif r:=re.match(r'([^,]+),([^,]+)',key):
                    ## save associated data and set after all values
                    key,suffix = r.groups()
                    assoc_data.append((key,suffix,val))
                else:
                    ## value data
                    data[key] = {'value':val,'assoc':{}}
            ## set associated data
            for (key,suffix,val) in assoc_data:
                data[key]['assoc'][suffix] = val
        ## TEMP HACK DELETE 
        for key in list(data.keys()): #  HACK
            if 'HT_HITRAN' in key:    #  HACK
                data[key.replace('HT_HITRAN','HITRAN_HT')] = data.pop(key) #  HACK
        ## END OF TEMP HACK DELETE
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
        if 'classname' in data:
            if data['classname'] != self.classname:
                warnings.warn(f'The loaded classname, {repr(data["classname"])}, does not match self, {repr(self["classname"])}, and it will be ignored.')
            data.pop('classname')
        ## 2021-06-11 HACK TO ACCOUNT FOR DEPRECATED ATTRIBUTES DELETE ONE DAY
        if 'default_step' in data: # HACK
            data.pop('default_step') # HACK
        ## END OF HACK
        ## description is saved in data
        if 'dataset_description' in data:
            self.description = str(data.pop('dataset_description'))
        ## Set data in self and selected attributes
        scalar_data = {}
        for key,val in data.items():
            ## only load requested keys
            if keys is not None and key not in keys:
                continue
            ## vector data but given as a scalar -- defer loading until after vector data so the length of data is known
            elif np.isscalar(val):
                scalar_data[key] = val
            ## vector data
            else:
                self[key] = val['value']
                for tkey,tval in val['assoc'].items():
                    self[(key,tkey)] = tval
        ## load scalar data
        for key,val in scalar_data.items():
            self[key] = val 

    def load_from_text(
            self,
            filename,
            comment='',
            labels_commented=False,
            delimiter=None,
            txt_to_dict_kwargs=None,
    ):
        """Load data from a text-formatted file."""
        ## text table to dict with header
        if txt_to_dict_kwargs is None:
            txt_to_dict_kwargs = {}
        txt_to_dict_kwargs |= {'delimiter':delimiter,'labels_commented':labels_commented}
        if txt_to_dict_kwargs['delimiter'] is None:
            if re.match(r'.*\.csv',filename):
                txt_to_dict_kwargs['delimiter'] = ','
            elif re.match(r'.*\.rs',filename):
                txt_to_dict_kwargs['delimiter'] = '␞'
            elif re.match(r'.*\.psv',filename):
                txt_to_dict_kwargs['delimiter'] = '|'
            elif re.match(r'.*\.tsv',filename):
                txt_to_dict_kwargs['delimiter'] = '\t'
        # assert comment not in ['',' '], "Not implemented"
        filename = tools.expand_path(filename)
        data = {}
        ## load header
        escaped_comment = re.escape(comment)
        blank_line_re = re.compile(r'^ *$')
        beginning_of_classname_re = re.compile(f'^ *{escaped_comment} *\\[classname\\] *\\n$') 
        beginning_of_dataset_description_re = re.compile(f'^ *{escaped_comment} *\\[description\\] *\\n$') 
        beginning_of_keys_re = re.compile(f'^ *{escaped_comment} *\\[keys\\] *\\n$') 
        beginning_of_data_re = re.compile(f'^ *{escaped_comment} *\\[data\\] *\\n$')
        key_line_re = re.compile(f'^ *{escaped_comment} *([^# ]+) *# *(.+) *') # no value in line
        key_line_with_value_re = re.compile(f'^ *{escaped_comment} *([^= ]+) *= *([^#]*[^ #\\n])') # may also contain description
        current_section = 'none'
        with open(filename,'r') as fid:
            for iline,line in enumerate(fid):
                ## identify which section of the file this is
                if r:=re.match(beginning_of_classname_re,line):
                    current_section = 'classname'
                    continue
                elif r:=re.match(beginning_of_dataset_description_re,line):
                    current_section = 'description'
                    data['dataset_description'] = ""
                    continue
                elif re.match(beginning_of_keys_re,line):
                    current_section = 'keys'
                    continue
                elif r:=re.match(beginning_of_data_re,line):
                    current_section = 'data'
                    iline += 1 
                    break
                ## process data depending on section
                elif current_section == 'none':
                    if re.match(blank_line_re,line):
                        continue
                elif current_section == 'classname':
                    data['classname'] = line[:-1]
                elif current_section == 'description':
                    data['dataset_description'] += line
                elif current_section == 'keys':
                    if re.match(blank_line_re,line):
                        continue
                    elif re.match(key_line_re,line):
                        continue
                    elif r:=re.match(key_line_with_value_re,line):
                        key,val = r.groups()
                        data[key] = ast.literal_eval(val)
        ## remove trailing newline from dataset_description
        if 'dataset_description' in data and len(data['dataset_description'])>0:
            data['dataset_description'] = data['dataset_description'][:-1]
        ## load array data
        data.update(tools.txt_to_dict(filename,skiprows=iline,**txt_to_dict_kwargs))
        return data

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

    def concatenate(self,new_dataset,keys='old'):
        """Extend self by new_dataset using keys existing in self. New data updated
        on optimisaion if new_dataset changes."""
        ## limit to keys
        if keys == 'old':
            keys = list(self.explicitly_set_keys())
        elif keys == 'new':
            keys = list(new_dataset.explicitly_set_keys())
        elif keys == 'all':
            keys = {*self.explicitly_set_keys(),*new_dataset.explicitly_set_keys()}
        ## make sure necessary keys are known
        self.assert_known(*keys)
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)
        ## set new data
        old_length = len(self)
        new_length = len(new_dataset)
        total_length = len(self) + len(new_dataset)
        self._reallocate(total_length)
        ## set extending data and associated data known to either
        ## new or old
        for key,data in self._data.items():
            if new_dataset.is_known(key):
                new_val = new_dataset[key]
                new_assocs = new_dataset._data[key]['assoc']
            else:
                if 'default' in self._data[key]:
                    new_val = self._data[key]['default']
                    new_assocs = {}
                else:
                    raise Exception(f'Key {repr(key)} not known to concatenated data.')
            data['value'][old_length:total_length] = data['cast'](new_val)
            for assoc in data['assoc'] | new_assocs:
                self.assert_known((key,assoc))
                if new_dataset.is_known(key):
                    new_val = new_dataset[key,assoc]
                else:
                    new_val = self.associated_kinds[assoc]['default']
                cast = self.associated_kinds[assoc]['cast']
                data['assoc'][assoc][old_length:total_length] = cast(new_val)

    @optimise_method(add_construct_function= True)
    def concatenate_and_optimise(self,new_dataset,keys='old',_cache=None):
        """Extend self by new_dataset using keys existing in self. New data updated
        on optimisaion if new_dataset changes."""
        if self._clean_construct and 'total_length' not in _cache:
            ## concatenate data if it hasn't been done before
            self.permit_indexing = False
            # new_dataset.permit_indexing = False
            ## limit to keys
            if keys == 'old':
                keys = list(self.explicitly_set_keys())
            elif keys == 'new':
                keys = list(new_dataset.explicitly_set_keys())
            elif keys == 'all':
                keys = {*self.explicitly_set_keys(),*new_dataset.explicitly_set_keys()}
            old_length = len(self)
            new_length = len(new_dataset)
            total_length = len(self) + len(new_dataset)
            self.concatenate(new_dataset,keys)
            _cache['keys'],_cache['old_length'],_cache['new_length'],_cache['total_length'] = keys,old_length,new_length,total_length
        else:
            ## update data in place
            index = slice(_cache['old_length'],_cache['total_length'])
            for key in _cache['keys']:
                self.set(key,new_dataset[key],index)
                for assoc in self._data[key]['assoc']:
                    self.set((key,assoc),new_dataset[key,assoc],index)

    def append(self,keys_vals_as_dict=None,keys='all',**keys_vals_as_kwargs):
        """Append a single row of data from kwarg scalar values."""
        if keys_vals_as_dict is None:
            keys_vals_as_dict = {}
        keys_vals = dict(**keys_vals_as_dict,**keys_vals_as_kwargs)
        for key in keys_vals:
            keys_vals[key] = [keys_vals[key]]
        self.extend(keys_vals,keys=keys)

    def extend(
            self,
            keys_vals_as_dict_or_dataset=idict(),
            keys='old',         # 'old','new','all'
            **keys_vals_as_kwargs
    ):
        """Extend self with new_data.  Keys must be present in both new and
        old data.  If keys='old' then extra keys in new data are
        ignored. If keys='new' then extra keys in old data are unset.
        If 'all' then keys must match exactly.  If key=='new' no data
        currently present then just add this data."""
        ## get preset lists of keys to extend
        if keys in ('old','all','new'):
            tkeys = set()
            if keys in ('old','all'):
                tkeys = tkeys.union(self.explicitly_set_keys())
            if keys in ('new','all'):
                tkeys = tkeys.union(keys_vals_as_kwargs)
                if isinstance(keys_vals_as_dict_or_dataset,Dataset):
                    tkeys = tkeys.union(keys_vals_as_dict_or_dataset.explicitly_set_keys())
                else:
                    tkeys = tkeys.union(keys_vals_as_dict_or_dataset)
            keys = tkeys
        ## ensure all keys are present in new and old data, and limit
        ## old data to these
        new_data = {}
        for key in keys:
            ## ensure keys are present in existing data
            if len(self) == 0 and key not in self:
                self[key] = []
            elif not self.is_known(key):
                raise Exception(f"Extending key not in existing data: {repr(key)}")
            ## collect new data
            if key in keys_vals_as_dict_or_dataset:
                new_data[key] = keys_vals_as_dict_or_dataset[key]
            elif key in keys_vals_as_kwargs:
                new_data[key] = keys_vals_as_kwargs[key]
            elif (isinstance(keys_vals_as_dict_or_dataset,Dataset)
                  and keys_vals_as_dict_or_dataset.is_known(key)):
                new_data[key] = keys_vals_as_dict_or_dataset[key]
            elif (default:=self._data[key]['default']) is not None:
                new_data[key] = default
            ## could add logic for auto defaults based on kind as below
            else:
                raise Exception(f'Extending key missing in new data: {repr(key)}')
        ## limit self to keys and mark not inferred
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)
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
        ## reallocate and lengthen arrays if necessary
        self._reallocate(total_length)
        ## add new data to old
        for key in keys:
            ## the object in self to extend
            data = self._data[key]
            new_val = new_data[key]
            ## increase unicode dtype length if new strings are
            ## longer than the current
            if self.get_kind(key) == 'U':
                ## this is a really hacky way to get the length of string in a numpy array!!!
                old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self.get(key).dtype)))
                new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(new_val).dtype)))
                if new_str_len > old_str_len:
                    ## reallocate array with new dtype with overallocation
                    t = np.empty(
                        len(self)*self._over_allocate_factor,
                        dtype=f'<U{new_str_len*self._over_allocate_factor}')
                    t[:len(self)] = self.get(key)
                    data['value'] = t
            ## set extending data and associated data
            data['value'][original_length:total_length] = data['cast'](new_val)
        ## finalise new length

    def _reallocate(self,new_length):
        """Lengthen data array and associated data"""
        for data in self._data.values():
            old_length = len(data['value'])
            if new_length > old_length:
                data['value'] = np.concatenate(
                    (data['value'],
                     np.empty(int(new_length*self._over_allocate_factor-old_length),
                              dtype=data['value'].dtype)))
            for key in data['assoc']:
                old_length = len(data['assoc'][key])
                if new_length > old_length:
                    data['assoc'][key] = np.concatenate(
                        (data['assoc'][key],
                         np.full(int(new_length*self._over_allocate_factor-old_length),
                                 self.associated_kinds[key]['default'],
                                  dtype=data['assoc'][key].dtype)))
        self._length = new_length

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
            legend_loc='best',
            annotate_lines=False, # annotate lines with their labels
            zlabel_format_function=None, # accept key=val pairs, defaults to printing them
            label_prefix='', # put this before label otherwise generated
            plot_errorbars=True, # if uncertainty available
            xscale='linear',     # 'log' or 'linear'
            yscale='linear',     # 'log' or 'linear'
            ncolumns=None,       # number of columsn of subplot -- None to automatically select
            show=False,          # show figure after issuing plot commands
            ylim=None,
            title=None,
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
        # if xkey == 'index':
            # if 'index'  in self.keys():
                # raise Exception("Index already exists")
            # self['index'] = np.arange(len(self),dtype=int)
            # xkey = 'index'
        if zkeys is None:
            zkeys = self.default_zkeys
        zkeys = [t for t in tools.ensure_iterable(zkeys) if t not in ykeys and t!=xkey and self.is_known(t)] # remove xkey and ykeys from zkeys
        ykeys = [key for key in tools.ensure_iterable(ykeys) if key not in [xkey]+zkeys]
        for t in [xkey,*ykeys,*zkeys]:
            self.assert_known(t)
        ## plot each 
        ymin = {}
        for iy,ykey in enumerate(tools.ensure_iterable(ykeys)):
            ylabel = ykey
            for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                z.sort(xkey)
                if zlabel_format_function is None:
                    zlabel_format_function = self.default_zlabel_format_function
                zlabel = zlabel_format_function(dz)
                if ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz+len(zkeys)*iy,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = None
                    if title is None:
                        title = ylabel+' '+zlabel
                elif ynewaxes and not znewaxes:
                    ax = plotting.subplot(n=iy,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = (zlabel if len(zkeys)>0 else None) 
                    if title is None:
                        title = ylabel
                elif not ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = ylabel
                    if title is None:
                        title = zlabel
                elif not ynewaxes and not znewaxes:
                    ax = fig.gca()
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    # color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = ylabel+' '+zlabel
                ## plotting kwargs
                kwargs = copy(plot_kwargs)
                kwargs.setdefault('marker',marker)
                kwargs.setdefault('ls',linestyle)
                kwargs.setdefault('mew',1)
                kwargs.setdefault('markersize',7)
                kwargs.setdefault('color',color)
                kwargs.setdefault('mec',kwargs['color'])
                if label is not None:
                    kwargs.setdefault('label',label_prefix+label)
                ## plotting data
                x = z[xkey]
                y = z[ykey]
                if plot_errorbars and z.is_set((ykey,'unc')):
                    dy = z.get((ykey,'unc'))
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
                        plotting.legend(fontsize='x-small',loc=legend_loc)
                    if annotate_lines:
                        plotting.annotate_line(line=line)
                if ylim is not None:
                    if ylim == 'data':
                        t,t,ybeg,yend = plotting.get_data_range(ax)
                        ax.set_ylim(ybeg,yend)
                    elif tools.isiterable(ylim) and len(ylim) == 2:
                        ybeg,yend = ylim
                        if ybeg == 'data':
                            t,t,ybeg,t = plotting.get_data_range(ax)
                        if yend == 'data':
                            t,t,t,yend = plotting.get_data_range(ax)
                    ax.set_ylim(ybeg,yend)
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
            self.get((ykey,'unc'),index),
            **polyfit_kwargs)

def find_common(x,y,keys=None,verbose=False):
    """Return indices of two Datasets that have uniquely matching
    combinations of keys."""
    ## if empty list then nothing to be done
    if len(x)==0 or len(y)==0:
        return(np.array([],dtype=int),np.array([],dtype=int))
    ## use quantum numbers as default keys -- could use _qnhash instead
    if keys is None:
        if hasattr(x,'defining_qn') and hasattr(y,'defining_qn'):
            keys = list(getattr(x,'defining_qn'))
        else:
            raise Exception("No keys provided and defining_qn unavailable x.")
    # ## Make a list of default keys if not provided as inputs. If a
    # ## Level or Transition object (through a hack) then use
    # ## defining_qn, else use all set keys known to both.
    # if len(keys)==0:
        # if hasattr(x,'defining_qn'):
            # keys = [t for t in x.defining_qn if x.is_known(t) and y.is_known(t)]
        # else:
            # keys = [t for t in x.keys() if x.is_known(t) and y.is_known(t)]
    if verbose:
        print('find_commmon keys:',keys)
    for key in keys:
        x.assert_known(key)
        y.assert_known(key)
    ## sort by first calculating a hash of sort keys
    xhash = np.array([hash(t) for t in x.row_data(keys=keys)])
    yhash = np.array([hash(t) for t in y.row_data(keys=keys)])
    ## get sorted hashes, checking for uniqueness
    xhash,ixhash,inv_xhash,count_xhash = np.unique(xhash,return_index=True,return_inverse=True,return_counts=True)
    if len(xhash) != len(x):
        if verbose:
            print("Duplicate key combinations in x:")
            for i in tools.find(count_xhash>1):
                print(f'    count = {count_xhash[i]},',repr({key:x[key][i] for key in keys}))
        raise Exception(f'There is {len(x)-len(xhash)} duplicate key combinations in x: {repr(keys)}. Set verbose=True to list them.')
    yhash,iyhash = np.unique(yhash,return_index=True)
    if len(yhash) != len(y):
        if verbose:
            print("Duplicate key combinations in y:")
            for i in tools.find(count_yhash>1):
                print(f'    count = {count_yhash[i]},',repr({key:y[key][i] for key in keys}))
        raise Exception(f'Non-unique combinations of keys in y: {repr(keys)}')
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
    ix = np.asarray(ix,dtype=int)
    iy = np.asarray(iy,dtype=int)
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

def load(filename,classname=None,**load_kwargs):
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
