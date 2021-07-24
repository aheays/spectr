import re
from time import perf_counter as timestamp
import ast
from copy import copy,deepcopy
from pprint import pprint,pformat
import importlib
import warnings

import numpy as np
from numpy import nan,arange,linspace,array
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

    ## The kind of data that 'value' contains.  Influences which subkinds are relevant.
    data_kinds = {
        'f':    {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt':'+12.8e','description':'float' },
        'i':    {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt':'d'     ,'description':'int'   },
        'b':    {'cast':convert_to_bool_vector_array       ,'fmt':''      ,'description':'bool'  },
        'U':    {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt':'s'     ,'description':'str'   },
        'O':    {'cast':lambda x:np.asarray(x,dtype=object),'fmt':''      ,'description':'object'},
        'h':    {'cast':lambda x:np.asarray(x,dtype='S20') ,'fmt':''      ,'description':'SHA1 hash'},
    }

    ##  Kinds of subdata that are vectors
    vector_subkinds = {
        'value'        : {'description' : 'Value of this data'},
        'unc'          : {'description' : 'Uncertainty'                         , 'kind'         : 'f' , 'valid_kinds'                : ('f',), 'cast' : lambda x                                  : np.abs(x,dtype=float)     ,'fmt' : '8.2e'  ,'default' : 0.0   ,},
        'step'         : {'description' : 'Default numerical differentiation step size' , 'kind' : 'f' , 'valid_kinds'                : ('f',), 'cast' : lambda x                                  : np.abs(x,dtype=float)     ,'fmt' : '8.2e'  ,'default' : 1e-8  ,},
        'vary'         : {'description' : 'Whether to vary during optimisation' , 'kind'         : 'b' , 'valid_kinds'                : ('f',), 'cast' : convert_to_bool_vector_array       ,'fmt' : ''      ,'default'               : False ,},
        'ref'          : {'description' : 'Source reference'                    , 'kind'         : 'U' ,                       'cast' : lambda x       : np.asarray(x,dtype='U20') ,'fmt'          : 's'     ,'default'               : nan   ,},
    }

    ##  Kinds of subdata that are single valued but maybe complex objects
    scalar_subkinds = {
        'infer'          : {'description':'List of infer functions',},
        'kind'           : {'description':'Kind of data in value, corresponds to keys of data_kinds',},
        'cast'           : {'description':'Vectorised function to cast data',},
        'fmt'            : {'description':'Format string for printing',},
        'description'    : {'description':'Description of data',},
        'units'          : {'description':'Units of data',},
        'default'        : {'description':'Default value',},
        'default_step'   : {'description':'Default differentiation step size','valid_kinds':('f',)},
        '_inferred_to'   : {'description':'List of keys inferred from this data',},
        '_inferred_from' : {'description':'List of keys used to infer this data',},
        '_modify_time'   : {'description':'When this data was last modified',},
    }

    ## all subdata kinds in one dictionary
    all_subkinds = vector_subkinds | scalar_subkinds

    ## prototypes on instantiation
    default_prototypes = {}
    default_permit_nonprototyped_data = False

    ## used for plotting and sorting perhaps
    default_zkeys = ()
    default_zlabel_format_function = tools.dict_to_kwargs

    def __init__(
            self,
            name=None,
            permit_nonprototyped_data = True,
            permit_indexing = True,
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
        self.description = description
        self._row_modify_time = np.array([],dtype=float,ndmin=1) # record modification time of each explicitly set row
        self._global_modify_time = timestamp() # record modification time of any explicit change
        ## whether to allow the addition of data not in self.prototypes
        if permit_nonprototyped_data is None:
            self.permit_nonprototyped_data = self.default_permit_nonprototyped_data
        else:
            self.permit_nonprototyped_data = permit_nonprototyped_data
        self.permit_indexing = permit_indexing # Data can be added to the end of arrays, but not removal or rearranging of data
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
            else:
                ## set data
                self[key] = val
        ## limit to matching data somehow loaded above
        if limit_to_match is not None:
            self.limit_to_match(**limit_to_match)

    def __len__(self):
        return self._length

    def set(
            self,
            key,                # "key" or ("key","subkey")
            subkey,
            value,
            index=None,         # set these indices only
            match=None,         # set these matches only
            set_changed_only=False, # only set data if it differs from value
            kind=None,
            **match_kwargs
    ):
        """Set value of key or (key,data)"""
        ## check for invalid key
        forbidden_character_regexp = r'.*([\'"=#,:]).*' 
        if r:=re.match(forbidden_character_regexp,key):
            raise Exception(f"Forbidden character {repr(r.group(1))} in key {repr(key)}. Forbidden regexp: {repr(forbidden_character_regexp)}")
        ## make array copies
        value = copy(value)
        ## scalar subkind — set and return, not cast
        if subkey in self.scalar_subkinds:
            self._data[key][subkey] = value
        elif subkey in self.vector_subkinds:
            ## combine indices -- might need to sort value if an index array is given
            combined_index = self._get_combined_index(index,match,**match_kwargs)
            ## reduce index and value to changed data only
            if set_changed_only and self.is_set(key,subkey):
                index_changed = self[key,subkey,combined_index] != value
                if combined_index is None:
                    combined_index = index_changed
                else:
                    combined_index = combined_index[index_changed]
                if tools.isiterable(value):
                    value = np.array(value)[index_changed]
            ## set value or other subdata
            if subkey == 'value':
                self._set_value(key,value,combined_index,kind=kind)
            else:
                self._set_subdata(key,subkey,value,combined_index)
        else:
            raise Exception('Invalid subkey: {repr(subkey)}')
            
    @optimise_method(format_multi_line=99)
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
            index = self._get_combined_index(index,match,return_bool=True,**match_kwargs)
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
        self.set(ykey,'value',value=tools.spline(xspline,yspline,self.get(xkey,index=index),order=order),index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            self.set(ykey,'unc',nan,index=index)
        ## set vary to False if set, but only on the first execution
        if 'not_first_execution' not in _cache:
            if 'vary' in self._data[ykey]:
                self.set(ykey,'vary',False,index=index)
            _cache['not_first_execution'] = True

    def _set_value(self,key,value,index=None,dependencies=None,kind=None):
        """Set a value"""
        ## turn Parameter into its floating point value
        if isinstance(value,Parameter):
            value = float(value)
        ## if key is already set then delete anything previously
        ## inferred from it, and previous things it is inferred 
        if key in self:
            self.unlink_inferences(key)
        ## if an index is provided then data must already exist, set
        ## new indeed data and return
        if index is not None:
            if not self.is_known(key):
                if key in self.prototypes and 'default' in self.prototypes[key]:
                    self.set(key,'value',value=self.prototypes[key]['default'])
                else:
                    raise Exception(f'Setting {repr(key)} for (possible) partial indices but it is not already set')
            data = self._data[key]
            if key not in self:
                raise Exception(f'Cannot set data by index for unset key: {key}')
            ## reallocate with increased unicode dtype length if new
            ## strings are longer than the current array dtype
            if self.get_kind(key) == 'U':
                ## this is a really hacky way to get the length of string in a numpy array!!!
                old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self.get(key).dtype)))
                new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(value).dtype)))
                if new_str_len > old_str_len:
                    ## reallocate array with new dtype with overallocation
                    t = np.empty(
                        len(self)*self._over_allocate_factor,
                        dtype=f'<U{new_str_len*self._over_allocate_factor}')
                    t[:len(self)] = self.get(key)
                    data['value'] = t
            ## set indexed data
            data['value'][:self._length][index] = data['cast'](value)
        else:
            ## set full array -- does not hvae to exist in in advance
            ##
            ## create entire data dict
            ## decide whether to permit if non-prototyped
            if not self.permit_nonprototyped_data and key not in self.prototypes:
                raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = {'infer':[],'_inferred_to':[]}
            ## get any prototype data
            if key in self.prototypes:
                for tkey,tval in self.prototypes[key].items():
                    data[tkey] = tval
            ## object kind not implemented
            if 'kind' in data and data['kind'] == 'O':
                raise ImplementationError()
            ## use data to infer kind if necessary
            if kind is not None:
                data['kind'] = kind
            if 'kind' not in data:
                value = np.asarray(value)
                data['kind'] = value.dtype.kind
            ## convert bytes string to unicode
            if data['kind'] == 'S':
                data['kind'] = 'U'
            ## some other prototype data based on kind
            data = self.data_kinds[data['kind']] | data
            ## if a scalar expand to length of self
            ## and set as default value
            if not tools.isiterable(value):
                data['default'] = value
                value = np.full(len(self),value)
            ## if this is the first data then allocate an initial
            ## length to match
            if len(self) == 0:
                self._reallocate(len(value))
                ## If this is the first nonzero-length data set then increase
                ## length of self and set any other keys with defaults to
                ## their default values
                if len(value) > 0:
                    for tkey,tdata in self._data.items():
                        if tkey == key:
                            continue
                        if 'default' in tdata:
                            tdata['value'] = tdata['cast'](np.full(len(self),tdata['default']))
                        else:
                            raise Exception(f'Need default for key {tkey}')
            if len(value) != len(self):
                raise Exception(f'Length of new data {repr(key)} is {len(value)} and does not match the length of existing data: {len(self)}.')
            ## cast and set data
            data['value'] = data['cast'](value)
            ## add to self
            self._data[key] = data
        ## If this is inferred data then record dependencies
        if dependencies is not None:
            self._data[key]['_inferred_from'] = list(dependencies)
            for dependency in dependencies:
                self._data[dependency]['_inferred_to'].append(key)
        ## Record key, global, row modify times if this is an explicit
        ## change.
        tstamp = timestamp()
        if dependencies is None or len(dependencies) == 0:
            self._global_modify_time = tstamp
            if index is None:
                self._row_modify_time[:self._length] = tstamp
            else:
                self._row_modify_time[:self._length][index] = tstamp
            self._data[key]['_modify_time'] = tstamp
        else:
            ## If inferred data record modification time of the most
            ## recently modified dependency
            self._data[key]['_modify_time'] = max(
                [self[tkey,'_modify_time'] for tkey in dependencies])

    def _set_subdata(self,key,subkey,value,index=None):
        """Set vector subdata."""
        subkind = self.vector_subkinds[subkey]
        if not self.is_known(key):
            raise Exception(f"Value of key {repr(key)} must be set before setting subkey {repr(subkey)}")
        data = self._data[key]
        if (subkind['valid_kinds'] is not None and self.get_kind(key) not in subkind['valid_kinds']):
            raise Exception(f"The value kind of {repr(key)} is {repr(data['kind'])} and is invalid for setting {repr(subkey)}")
        if subkind['kind'] == 'O':
            raise ImplementationError()
        if self.verbose:
            print(f'{self.name}: setting ({key}:{subkey})')
        ## set data
        if index is None:
            ## set entire array
            if not tools.isiterable(value):
                ## expand scalar input
                value = np.full(len(self),value)
            elif len(value) != len(self):
                raise Exception(f'Length of new subdata {repr(subkey)} for key {repr(key)} ({len(value)} does not match existing data length ({len(self)})')
            ## set data
            data[subkey] = subkind['cast'](value)
        else:
            ## set part of array by index
            if subkey not in data:
                ## set missing data outside indexed range to a default
                ## value using the get method
                self.get(key,subkey)
            ## set indexed data
            data[subkey][:len(self)][index] = subkind['cast'](value)

    row_modify_time = property(lambda self:self._row_modify_time[:self._length])
    global_modify_time = property(lambda self:self._global_modify_time)

    def get(self,key,subkey='value',index=None,units=None,match=None,**match_kwargs):
        """Get value for key or (key,subkey)."""
        index = self._get_combined_index(index,match,**match_kwargs)
        ## ensure data is known
        if key not in self._data:
            try:
                ## attempt to infer
                self._infer(key)
            except InferException as err:
                if key in self.prototypes and 'default' in self.prototypes[key]:
                    self[key] = self.prototypes[key]['default']
                else:
                    raise err
        ## get relevant data
        data = self._data[key]
        ## check subdata exists
        subkind = self.all_subkinds[subkey] 
        ## test that this subkind is valid
        if 'valid_kinds' in subkind and data['kind'] not in subkind['valid_kinds']:
            raise Exception(f"Key {repr(key)} of kind {data['kind']} is not a valid kind for subdata {repr(subkey)}")
        ## if data is not set then set default if possible
        if not self.is_set(key,subkey):
            assert subkey != 'value','should be inferred above'
            if subkey == 'step' and 'default_step' in data:
                ## special shortcut case for specied default step
                self.set(key,subkey,data['default_step'])
            elif 'default' in subkind:
                self.set(key,subkey,subkind['default'])
            else:
                raise InferException(f'Could not determine default value for subkey {repr(subkey)})')
        ## return data
        if subkey in self.scalar_subkinds:
            ## scalar subdata
            return data[subkey]
        elif subkey in self.vector_subkinds:
            ## return indexed data
            retval = data[subkey][:len(self)]
            if index is not None:
                retval = retval[index]
            if units is not None:
                retval = convert.units(retval,data['units'],units)
            return retval
        else:
            raise Exception(f'Invalid subkey: {repr(subkey)}')

    def set_default(self,key=None,value=None,**more_keys_values):
        """Set default value for key, and set existing data to this value if
        not already set."""
        if key is not None:
            more_keys_values[key] = value
        for key,value in more_keys_values.items():
            if key not in self:
                self[key] = value
            self._data[key]['default'] = value

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
        for tkey,tval in self.data_kinds[kind].items():
            self.prototypes[key].setdefault(tkey,tval)
            
    def get_kind(self,key):
        return self._data[key]['kind']

    def _has_attribute(self,key,subkey,attribute):
        """Test if key,subkey has a certain attribute."""
        self.assert_known(key,subkey)
        if subkey == 'value':
            return (attribute in self._data[key])
        else:
            return (attribute in self.all_subkinds[subkey])

    def _get_attribute(self,key,subkey,attribute):
        """Get data from data_kinds or all_subkinds"""
        self.assert_known(key,subkey)
        if subkey == 'value':
            return self[key,attribute]
        else:
            return self.all_subkinds[subkey][attribute]
            
    def _get_combined_index(self,index=None,match=None,return_bool=False,**match_kwargs):
        """Combined specified index with match arguments as integer array. If
        no data given the return None"""
        if index is None and match is None and len(match_kwargs)==0:
            ## no indices at all
            retval = None
        elif np.isscalar(index):
            ## single index
            retval = index
            if match is not None and len(match_kwargs) != 0:
                raise Exception("Single index cannot be addtionally matched.")
            if return_bool:
                raise Exception("Single index cannot be returned as a boolean array.")
        else:
            ## get index into a bool or index array
            if index is None:
               retval = np.arange(len(self))
            elif isinstance(index,slice):
                ## slice
                retval = np.arange(len(self))
                retval = retval[index]
            else:
                retval = np.array(index,ndmin=1)
                if retval.dtype == bool:
                    retval = tools.find(retval)
            ## reduce by matches if given
            if match is not None or len(match_kwargs) > 0:
                imatch = self.match(match,**match_kwargs)
                retval = retval[tools.find(imatch[retval])]
            if return_bool:
                ## convert to boolean array
                t = np.full(len(self),False)
                t[retval] = True
                retval = t
        return retval

    @optimise_method(format_multi_line=99)
    def set_value(
            self,
            key,
            value,          # a scalar or Parameter
            index=None,         # only apply to these indices
            match=None,
            _cache=None,
            **match_kwargs
    ):
        """Set a value and it will be updated every construction and possible
        optimised."""
        if self._clean_construct:
            ## cache matching indices
            _cache['index'] = self._get_combined_index(index,match,**match_kwargs)
        index = _cache['index']
        self.set(key,'value',value,index=index,set_changed_only= True)
        if self._clean_construct and isinstance(value,Parameter):
            self.set(key,'unc',value.unc,index=index)
            self.set(key,'step',value.step,index=index)
        ## set vary to False if set, but only on the first execution
        if 'not_first_execution' not in _cache:
            if 'vary' in self._data[key]:
                self.set(key,'vary',False,index=index)
            _cache['not_first_execution'] = True

    set_and_optimise = set_value

    def keys(self):
        return list(self._data.keys())

    def limit_to_keys(self,keys):
        """Unset all keys except these."""
        keys = tools.ensure_iterable(keys)
        for key in keys:
            self.assert_known(key)
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)

    def optimised_keys(self):
        return [key for key in self.keys() if self.is_set(key,'vary')]

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


    def is_set(self,key,subkey='value'):
        if key in self._data and subkey in self._data[key]:
            return True
        else:
            return False

    def assert_known(self,key,subkey='value'):
        """Check is known by trying to get item."""
        self[key,subkey]

    def is_known(self,key,subkey='value'):
        """Test if key is known."""
        try:
            self.assert_known(key,subkey)
            return True 
        except InferException:
            return False
            
    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an arg, return an arged copy of self."""
        if isinstance(arg,str):
            ## a non indexed key
            return self.get(key=arg)
        elif isinstance(arg,(int,np.int64)):
            ## single indexed row
            return self.row(index=arg)
        elif isinstance(arg,slice):
            ## index by slice
            return self.copy(index=arg)
        elif isinstance(arg,np.ndarray):
            ## an index array
            return self.copy(index=arg)
        elif not tools.isiterable(arg):
            ## no more single valued arguments defined
            raise Exception(f"Cannot interpret getitem argument: {repr(arg)}")
        elif isinstance(arg[0],(int,np.int64,bool)):
            ## index
            return self.copy(index=arg)
        elif len(arg) == 1:
            if isinstance(arg[0],str):
                ## copy with keys
                return self.copy(keys=arg)
            else:
                ## copy with index
                return self.copy(index=arg[0])

            
        elif len(arg) == 2:
            if isinstance(arg[0],str):
                if isinstance(arg[1],str):
                    if arg[1] in self.all_subkinds:
                        ## return key,subkey
                        return self.get(key=arg[0],subkey=arg[1])
                    else:
                        ## copy keys
                        return self.copy(keys=arg)
                else:
                    ## return key,index
                    return self.get(key=arg[0],index=arg[1])

            else:
                ## copy with keys and index
                return self.copy(keys=arg[0],index=arg[1])
            

        elif len(arg) == 3:
            if arg[1] in self.all_subkinds:
                ## get key,subkey,index
                return self.get(key=arg[0],subkey=arg[1],index=arg[2])
            else:
                ## copy keys
                return self.copy(keys=arg)

        else:
            ## all args are keys for copying
            return self.copy(keys=arg)

            
        # elif tools.isiterable(arg[0]):
            # ## make a copy of self possibly with keys and indices
            # ## specified
            # if len(arg) == 1:
                # if isinstance(arg[0][0],str):
                    # ## a list of keys
                    # return self.copy(key=arg[0])
                # else:
                    # ## an index of some kind
                    # return self.copy(index=arg[0])
            # elif len(arg) == 2:
                # ## list of keys and index
                # return self.copy(key=arg[0],index=arg[1])

        # else:
            # raise Exception(f"Cannot interpret getitem argument: {repr(arg)}")

        # elif len(arg) == 1:
            # ## a single vector argument – return copy
            # if isinstance(arg,str):
                # return self.get(key=arg[0])
            # else:
                # return self.copy(index=arg[0])
        # elif len(arg) == 2:
            # if not isinstance(arg[0],str):
                # return self.copy(index=arg)
            # if isinstance(arg[1],str):
                # return self.get(key=arg[0],subkey=arg[1])
            # else:
                # return self.get(key=arg[0],index=arg[1])
        # elif len(arg) == 3:
            # if not isinstance(arg[0],str):
                # return self.copy(index=arg)
            # else:
                # return self.get(key=arg[0],subkey=arg[1],index=arg[2])
        # else:
            # raise Exception(f"Cannot interpret key: {repr(arg)}")

    def __setitem__(self,key,value):
        """Set key, (key,subkey), (key,index), (key,subkey,index) to
        value."""
        if isinstance(key,str):
            key,subkey,index = key,'value',None
        elif len(key) == 1:
            key,subkey,index = key[0],'value',None
        elif len(key) == 2:
            if isinstance(key[1],str):
                key,subkey,index = key[0],key[1],None
            else:
                key,subkey,index = key[0],'value',key[1]
        elif len(key) == 3:
                key,subkey,index = key[0],key[1],key[2]
        self.set(key,subkey,value,index)
       
    def clear(self):
        """Clear all data"""
        if not self.permit_indexing:
            raise Exception('Cannot clear dataset with not permit_indexing.')
        self._last_modify_value_time = timestamp()
        self._length = 0
        self._data.clear()

    def unset(self,key,subkey='value'):
        """Delete data.  Also clean up inferences."""
        if key in self:
            if subkey == 'value':
                self.unlink_inferences(key)
                self._data.pop(key)
            else:
                data = self._data[key]
                if subkey in data:
                    data.pop(subkey)

    def pop(self,key):
        """Return data and unset key."""
        retval = self[key]
        self.unset(key)
        return retval

    def is_inferred(self,key):
        """Test whether this key is inferred (or explicitly set)."""
        if '_inferred_from' in self._data[key]:
            return True
        else:
            return False
   
    def unset_inferred(self):
        """Delete all inferred data."""
        for key in list(self):
            if key in self and self.is_inferred(key):
                self.unlink_inferences(key)
                self.unset(key)
   
    def unlink_inferences(self,keys):
        """Delete any record these keys begin inferred. Also recursively unset
        any key inferred from these that is not among keys itself."""
        keys = tools.ensure_iterable(keys)
        for key in keys:
            self.assert_known(key)
        ## no longer marked as inferred from something else
        for key in keys:
            if key not in self:
                continue
            if self.is_inferred(key):
                for tkey in self._data[key]['_inferred_from']:
                    if tkey in self._data:
                        if key in self._data[tkey]['_inferred_to']:
                            self._data[tkey]['_inferred_to'].remove(key)
                self._data[key].pop('_inferred_from')
            ## recursively delete everything inferred to
            for tkey in copy(self._data[key]['_inferred_to']):
                if tkey not in keys and tkey in self:
                    self.unset(tkey)


    def add_infer_function(self,key,dependencies,function):
        """Add a new method of data inference."""
        self.prototypes[key]['infer'].append((dependencies,function))

    def index(self,index):
        """Index all array data in place."""
        if not self.permit_indexing:
            raise Exception('Indexing not permitted')
        original_length = len(self)
        for key,data in self._data.items():
            for subkey in data:
                if subkey in self.vector_subkinds:
                    data[subkey] = data[subkey][:original_length][index]
            self._length = len(data['value'])

    def remove(self,index):
        """Remove indices."""
        index = self._get_combined_index(index,return_bool=True)
        self.index(~index)

    def copy(self,*args_copy_from,name=None,**kwargs_copy_from):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if name is None:
            name = f'copy_of_{self.name}'
        retval = self.__class__(name=name) # new version of self
        retval.copy_from(self,*args_copy_from,**kwargs_copy_from)
        retval.pop_format_input_function()
        return retval

    def copy_from(
            self,
            source,
            keys=None,
            index=None,
            match=None,
            subkeys=None,
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
        ## copy data and selected prototype data
        for key in keys:
            # self.set(key,'value',source[key,'value',index])
            self.set(key,'value',source[key,'value',index])
            ## get a list of subkeys to copy for this key, ignore those beginning with '_'
            if subkeys is None:
                tsubkeys = [subkey for subkey in source.all_subkinds if source.is_set(key,subkey) and subkey[0]!='_']
            else:
                tsubkeys  = subkeys
            ## copy subdata, 'value' already copied
            for subkey in tsubkeys:
                if subkey == 'value':
                    continue
                if subkey in self.vector_subkinds:
                    self.set(key,subkey,source[key,subkey,index])
                else:
                    self.set(key,subkey,source[key,subkey])

    @optimise_method()
    def copy_from_and_optimise(
            self,
            source,
            keys=None,
            skip_keys=(),
            index=None,
            match=None,
            subkeys=('value','unc','description','units','fmt'),
            copy_inferred_data=False,
            _cache=None,
            **match_kwargs
    ):
        """Copy all values and uncertainties from source Dataset and update if
        source changes during optimisation."""
        ## get keys and indices to copy
        if self._clean_construct:
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
        ## copy data and selected prototype data
        for key in keys:
            for subkey in subkeys:
                if (source.is_set(key,subkey)
                    and (not self.is_set(key,subkey)
                         or source[key,'_modify_time'] > self[key,'_modify_time'])):
                    if subkey in self.vector_subkinds:
                        self.set(key,subkey,source[key,subkey,index],set_changed_only= True)
                    else:
                        self.set(key,subkey,source[key,subkey])

    def match_regexp(self,keys_vals=None,**kwarg_keys_vals):
        """Match string keys to regular expressions."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwarg_keys_vals
        retval = np.full(len(self),True)
        for key,regexp in keys_vals.items():
            retval &= np.array([re.match(regexp,val) for val in self[key]],dtype=bool)
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

    # def find(self,**keys_vals):
        # """Find unique indices matching keys_vals which contains one or more
        # vector matches or the same length."""
        # ## SLOW IMPLEMENTATION -- REPLACE WITH HASH MATCHING?
        # ## separate vector and scalar match data
        # vector_keysvals = {}
        # scalar_keysvals = {}
        # vector_length = None
        # for key,val in keys_vals.items():
            # if np.isscalar(val):
                # scalar_keysvals[key] = val
            # else:
                # vector_keysvals[key] = val
                # if vector_length == None:
                    # vector_length = len(val)
                # elif vector_length != len(val):
                    # raise Exception('All vector matching data must be the same length')
        # ## get data matching scalar keys_vals
        # iscalar = tools.find(self.match(**scalar_keysvals))
        # ## find vector_key matches one by one
        # i = np.empty(vector_length,dtype=int)
        # for ii in range(vector_length):
            # ti = np.all([self[key][iscalar]==val[ii] for key,val in vector_keysvals.items()],0)
            # ti = tools.find(ti)
            # if len(ti) == 0:
                # raise Exception("No match: {vector_key}={repr(vector_vali)} and {repr(keys_vals)}")
            # if len(ti) > 1:
                # raise Exception("Non-unique match: {vector_key}={repr(vector_vali)} and {repr(keys_vals)}")
            # i[ii] = iscalar[ti]
        # return i

    def matches(self,*args,**kwargs):
        """Returns a copy reduced to matching values."""
        return self.copy(index=self.match(*args,**kwargs),copy_inferred_data=True)

    def limit_to_match(self,*match_args,**match_kwargs):
        self.index(self.match(*match_args,**match_kwargs))

    def remove_match(self,*match_args,**match_keys_vals):
        self.index(~self.match(*match_args,**match_keys_vals))

    def unique(self,key,subkey='value'):
        """Return unique values of one key."""
        self.assert_known(key,subkey)
        if self.get_kind(key) == 'O':
            raise ImplementationError()
            return self[key]
        else:
            return np.unique(self[key,subkey])

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
            raise InferException(f"No prototype for key: {repr(key)}")
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
                ## compute uncertainties by linearisation
                if uncertainty_function is None:
                    squared_contribution = []
                    value = self[key]
                    parameters = [self[t] for t in dependencies]
                    for i,dependency in enumerate(dependencies):
                        if self.is_set(dependency,'unc'):
                            step = self.get(dependency,'step')
                            parameters[i] = self[dependency] + step # shift one
                            dvalue = value - function(self,*parameters)
                            parameters[i] = self[dependency] # put it back
                            squared_contribution.append((self.get(dependency,'unc')*dvalue/step)**2)
                    if len(squared_contribution)>0:
                        uncertainty = np.sqrt(np.sum(squared_contribution,axis=0))
                        self._set_subdata(key,'unc',uncertainty)
                        if self.verbose:
                             print(f'{self.name}: Inferred uncertainty: {repr(key)}')
                else:
                    ## args for uncertainty_function.  First is the
                    ## result of calculating keys, after that paris of
                    ## dependencies and their uncertainties, if they
                    ## have no uncertainty then None is substituted.
                    args = [self,self[key]]
                    for dependency in dependencies:
                        if self.is_set(dependency,'unc'):
                            t_uncertainty = self.get(dependency,'unc')
                        else:
                            t_uncertainty = None
                        args.extend((self[dependency],t_uncertainty))
                    try:
                        self.set(key,'unc',uncertainty_function(*args))
                    except InferException:
                        pass
                ## success
                if self.verbose:
                    print(f'{self.name}:',''.join(['    ' for t in range(depth)])+f'Sucessfully inferred: {repr(key)}')
                break           
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
            if self.is_set(key,'unc'):
                retval[f'{key}_unc'] = self.get(key,'unc',index=index)
        return retval

    def as_dict(
            self,
            keys=None,
            index=None,
            subkeys=('value','unc','description','units'),
    ):
        """Return as a structured dict."""
        ## default to all data
        if keys is None: 
            keys = list(self.keys())
        ## add data
        retval = {}
        retval['classname'] = self.classname
        retval['description'] = self.description
        for key in keys:
            retval[key] = {}
            for subkey in subkeys:
                if self.is_set(key,subkey):
                    retval[key][subkey] = self.get(key,subkey)
        return retval
        
    def row(self,index,keys=None):
        """Iterate value data row by row, returns as a dictionary of
        scalar values."""
        if keys is None:
            keys = self.keys()
        return {key:self.get(key,'value',int(index)) for key in keys}
        
        
    def rows(self,keys=None):
        """Iterate value data row by row, returns as a dictionary of
        scalar values."""
        if keys is None:
            keys = self.keys()
        for i in range(len(self)):
            yield {key:self.get(key,'value',i) for key in keys}

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
            subkeys=('value','unc','vary','step','ref','description','units'),
            include_description=True,
            include_classname=True,
            include_key_description=True,
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
        for key in keys:
            self.assert_known(key)
        ## data to store in header
        ## collect columns of data -- maybe reducing to unique values
        columns = []
        header_values = {}
        for key in keys:
            self.assert_known(key)
            if len(self) == 0:
                break
            formatted_key = ( "'"+key+"'" if quote_keys else key )
            if (unique_values_in_header # input parameter switch
                and not np.any([self.is_set(key,subkey) for subkey in subkeys]) # no other subdata 
                and self.unique(key) == 1): # value is unique
                ## format value for header
                header_values[key] = tval[0]
            else:
                ## format columns
                for subkey in subkeys:
                    if self.is_set(key,subkey) and subkey in self.vector_subkinds:
                        if subkey == 'value':
                            formatted_key = (f'"{key}"' if quote_keys else f'{key}')
                        else:
                            formatted_key = (f'"{key}:{subkey}"' if quote_keys else f'{key}:{subkey}')
                        fmt = self._get_attribute(key,subkey,'fmt')
                        kind = self._get_attribute(key,subkey,'kind')
                        if quote_strings and kind == 'U':
                            vals = ['"'+format(t,fmt)+'"' for t in self[key,subkey]]
                        else:
                            vals = [format(t,fmt) for t in self[key,subkey]]
                        width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                        columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
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
        if include_description and self.description is not None:
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

    def format_flat(self,delimiter=' | '):
        """Print flat data"""
        return self.format(
            delimiter=delimiter,
            unique_values_in_header=False,
            include_description=False,
            include_keys_with_leading_underscore=False,
            include_key_description=False,
            include_classname=False,
            quote_strings=False,
            quote_keys=False,
        )

    def __str__(self):
        return self.format(
            delimiter=' | ',
            unique_values_in_header= True,
            include_description= True,
            include_classname=False,
            include_key_description=False,
            include_keys_with_leading_underscore=False,
            quote_strings=False,
            quote_keys=False,
        )
        # return self.format_flat()

    def __repr__(self):
        if len(self)>50:
            return(self.name)
        return f"{self.classname}(load_from_string='''\n{self.format_flat()}''')"
            
    def save(
            self,
            filename,
            keys=None,
            subkeys=None,
            **format_kwargs,
    ):
        """Save some or all data to a file."""
        if keys is None:
            keys = self.keys()
        if subkeys is None:
            ## get a list of default subkeys, ignore those beginning
            ## with "_"
            subkeys = [subkey for subkey in self.vector_subkinds if subkey[0] != '_']
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(filename,self.as_dict(keys=keys,subkeys=subkeys))
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            tools.dict_to_hdf5(filename,self.as_dict(keys=keys,subkeys=subkeys),verbose=False)
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
            subkeys = None,     # what to load, None for all
            txt_to_dict_kwargs=None,
            translate_from_anh_spectrum=False, # HACK to translate keys from spectrum module
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
        ## build structured data from flat data 
        if data_is_flat:
            flat_data = data
            data = {}
            for key,val in flat_data.items():
                if key == 'classname':
                    ## classname attribute
                    data['classname'] = val
                    # self.classname = val
                elif key == 'description':
                    ## description attribute
                    self.description = val
                else:
                    ## if r:=re.match(r'([^:]+)[:]([^:]+)',key): # proper regexp
                    if r:=re.match(r'([^:,]+)[:,]([^:,]+)',key): # HACK TO INCLUDE , SEPARATOR, REMOVE THIS ONE DAY 2021-06-22
                        key,subkey = r.groups()
                    else:
                        subkey = 'value'
                    if key not in data:
                        data[key] = {}
                    data[key][subkey] = val
        ## translate keys
        if translate_keys is None:
            translate_keys = {}
        if translate_from_anh_spectrum:
            translate_keys.update({
                'Jp':'J_u', 'Sp':'S_u', 'Tp':'E_u',
                'labelp':'label_u', 'sp':'s_u',
                'speciesp':'species_u', 'Λp':'Λ_u', 'vp':'v_u',
                'column_densityp':'Nself_u', 'temperaturep':'Teq_u',
                'Jpp':'J_l', 'Spp':'S_l', 'Tpp':'E_l',
                'labelpp':'label_l', 'spp':'s_l',
                'speciespp':'species_l', 'Λpp':'Λ_l', 'vpp':'v_l',
                'column_densitypp':'Nself_l', 'temperaturepp':'Teq_l',
                'Treduced_common_polynomialp':None, 'Tref':'Eref',
                'branch':'branch', 'dfv':None,
                'level_transition_type':None, 'partition_source':None,
                'Γ':'Γ','df':None,
            })
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
                warnings.warn(f'The loaded classname, {repr(data["classname"])}, does not match self, {repr(self.classname)}, and it will be ignored.')
            data.pop('classname')
        ## 2021-06-11 HACK TO ACCOUNT FOR DEPRECATED ATTRIBUTES DELETE ONE DAY
        if 'default_step' in data: # HACK
            data.pop('default_step') # HACK
        ## END OF HACK
        ## description is saved in data
        if 'description' in data:
            self.description = str(data.pop('description'))
        ## HACK REMOVE ASSOC 2021-06-21 DELETE ONE DAY
        for key in data:
            if 'assoc' in data[key]:
                for subkey in data[key]['assoc']:
                    data[key][subkey] = data[key]['assoc'][subkey]
                data[key].pop('assoc')
        ## END OF HACK
        ## Set data in self and selected attributes
        scalar_data = {}
        for key in data:
            ## only load requested keys
            if keys is not None and key not in keys:
                continue
            ## vector data but given as a scalar -- defer loading
            ## until after vector data so the length of data is known
            elif 'value' not in data[key]:
                raise Exception
            elif np.isscalar(data[key]['value']):
                scalar_data[key] = data[key]
            ## vector data
            else:
                self[key,'value'] = data[key].pop('value')
                for subkey in data[key]:
                    self[key,subkey] = data[key][subkey]
        ## load scalar data
        for key in scalar_data:
            self[key,'value'] = scalar_data[key].pop('value')
            for subkey in scalar_data[key]:
                self[key,subkey] = scalar_data[key][subkey]

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
        filename = tools.expand_path(filename)
        data = {}
        ## load header
        escaped_comment = re.escape(comment)
        blank_line_re = re.compile(r'^ *$')
        commented_line_re = re.compile(f'^ *{escaped_comment} */(.*)$')
        beginning_of_section_re = re.compile(f'^ *{escaped_comment} *\\[([^]]+)\\] *$') 
        key_line_without_value_re = re.compile(f'^ *{escaped_comment} *([^# ]+) *# *(.+) *') # no value in line
        key_line_with_value_re = re.compile(f'^ *{escaped_comment} *([^= ]+) *= *([^#]*[^ #])') # may also contain description
        current_section = 'data'
        valid_sections = ('classname','description','keys','data')
        section_iline = 0       # how many lines read in this section
        classname = None
        description = None
        with open(filename,'r') as fid:
            for iline,line in enumerate(fid):
                ## remove newline
                line = line[:-1]
                ## check for bad section title
                if current_section not in valid_sections:
                    raise Exception(f'Invalid data section: {repr(current_section)}. Valid sections: {repr(valid_sections)}')
                ## remove comment character unless in data section —
                ## then skip the line, or description then keep it in
                ## plac
                if r:=re.match(commented_line_re,line):
                    if current_section == 'data':
                        continue
                    elif current_section == 'description':
                        pass
                    else:
                        line = r.match(1)
                ## skip blank lines unless in the description
                elif re.match(blank_line_re,line) and current_section != 'description':
                    continue
                ## moving forward in this section
                section_iline += 1
                ## process data from this line
                if r:=re.match(beginning_of_section_re,line):
                    ## new section header line
                    current_section = r.group(1)
                    section_iline = 0
                    if current_section == 'description':
                        description = ''
                    continue
                elif current_section == 'classname':
                    ## save classname 
                    if section_iline > 1:
                        raise Exception("Invalid classname section")
                    classname = line
                elif current_section == 'description':
                    ## add to description
                    description += '\n'+line
                elif current_section == 'keys':
                    ## add value of key if value given
                    if r:=re.match(key_line_without_value_re,line):
                        continue
                    elif r:=re.match(key_line_with_value_re,line):
                        data[r.group(1)] = ast.literal_eval(r.group(2))
                elif current_section == 'data':
                    ## remainder of data is data, no more header to
                    ## process
                    break
        ## load array data
        data.update(tools.txt_to_dict(filename,skiprows=iline,**txt_to_dict_kwargs))
        if classname is not None:
            data['classname'] = classname
        if description is not None:
            data['description'] = description
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
        if len(self.keys()) == 0:
            self.copy_from(new_dataset)
            return 
        ## limit to keys
        if keys == 'old':
            keys = list(self.explicitly_set_keys())
        elif keys == 'new':
            keys = list(new_dataset.explicitly_set_keys())
        elif keys == 'all':
            keys = {*self.explicitly_set_keys(),*new_dataset.explicitly_set_keys()}
        else:
            keys = keys
        ## make sure necessary keys are known
        for key in keys:
            self.assert_known(key)
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)
        ## set new data
        old_length = len(self)
        new_length = len(new_dataset)
        total_length = len(self) + len(new_dataset)
        self._reallocate(total_length)
        ## set extending data 
        for key,data in self._data.items():
            for subkey in data:
                if subkey in self.vector_subkinds:
                    if new_dataset.is_known(key,subkey):
                        new_val = new_dataset[key,subkey]
                    elif self._has_attribute(key,subkey,'default'):
                        new_val = self._get_attribute(key,subkey,'default')
                    else:
                        raise Exception(f'Unknown to concatenated data: ({repr(key)},{repr(subkey)})')
                    data[subkey][old_length:total_length] = self._get_attribute(key,subkey,'cast')(new_val)

    def join(self,new_dataset):
        """Join keys form new data set onto this one.  No overlap allowed."""
        ## error checks
        if len(self) != len(new_dataset):
            raise Exception(f'Length mismatch between self and new dataset: {len(self)} and {len(new_dataset)}')
        i,j = tools.common(self.keys(),new_dataset.keys())
        if len(i)>0:
            raise Exception(f'Overlapping keys between self and new dataset: {repr(self.keys()[i])}')
        ## add from new_dataset
        for key in new_dataset:
            if key in self.prototypes:
                self[key] = new_dataset[key]
            else:
                if not self.permit_nonprototyped_data:
                    raise Exception(f'Key from new dataset is not prototyped in self: {repr(key)}')
                self._data[key] = deepcopy(new_dataset._data[key])

    @optimise_method()
    def concatenate_and_optimise(self,new_dataset,keys='old',_cache=None):
        """Extend self by new_dataset using keys existing in self. New data updated
        on optimisaion if new_dataset changes."""
        if self._clean_construct and 'total_length' not in _cache:
            ## concatenate data if it hasn't been done before
            self.permit_indexing = False
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
                self.set(key,'value',new_dataset[key],index,set_changed_only=True)
                if self.is_set(key,'unc'):
                    if self.is_set(key,'unc'):
                        self.set(key,'unc',new_dataset[key,'unc'],index)
                    else:
                        self.set(key,'unc',self.vector_subkinds['unc']['default'],index)
                if self.is_set(key,'vary'):
                    self.set(key,'vary',False,index)

    def append(self,keys_vals=None,keys='all',**keys_vals_as_kwargs):
        """Append a single row of data from kwarg scalar values."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals |= keys_vals_as_kwargs
        for key in keys_vals:
            keys_vals[key] = [keys_vals[key]]
        self.extend(keys_vals,keys=keys)

    def extend(
            self,
            keys_vals=None,
            keys='old',         # 'old','new','all'
            **keys_vals_as_kwargs
    ):
        """Extend self with new_data.  Keys must be present in both new and
        old data.  If keys='old' then extra keys in new data are
        ignored. If keys='new' then extra keys in old data are unset.
        If 'all' then keys must match exactly.  If key=='new' no data
        currently present then just add this data."""
        ## get preset lists of keys to extend
        if keys_vals is None:
            keys_vals = {}
        keys_vals |= keys_vals_as_kwargs

        ## separate subkeys
        subkeys_vals = {}
        for key in list(keys_vals):
            if not isinstance(key,str):
                tkey,tsubkey = key
                if tsubkey == 'value':
                    ## no need to store subkey
                    keys_vals[tkey] = keys_vals.pop(key)
                else:
                    subkeys_vals[tkey,tsubkey] = keys_vals.pop(key)
        ## collect value keys
        if keys in ('old','all','new'):
            tkeys = set()
            if keys in ('old','all'):
                tkeys = tkeys.union(self.explicitly_set_keys())
            if keys in ('new','all'):
                tkeys = tkeys.union(keys_vals)
            keys = tkeys
        ## ensure all keys are present in new and old data, and limit
        ## old data to these
        new_data = {}
        for key in keys:
            ## collect new data
            if key not in keys_vals:
                keys_vals[key] = self._data[key]['default']
            # ## could add logic for auto defaults based on kind as below
            # else:
                # raise Exception(f'Extending key missing in new data: {repr(key)}')
            ## ensure keys are present in existing data, if kind not
            ## in prototypes then infer from the new data, if no new
            ## data assume float
            if len(self) == 0 and not self.is_known(key):
                if key in self.prototypes:
                    kind = None
                elif len(keys_vals[key]) > 0:
                    kind = array(keys_vals[key]).dtype.kind
                else:
                    kind ='f'
                self.set(key,'value',[],kind=kind)
            elif not self.is_known(key):
                raise Exception(f"Extending key not in existing data: {repr(key)}")
        ## limit self to keys and mark not inferred
        self.unlink_inferences(keys)
        for key in list(self):
            if key not in keys:
                self.unset(key)
        ## determine length of data
        original_length = len(self)
        extending_length = None
        for key,val in keys_vals.items():
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
        ## add new data to old, set values first then other subdata
        ## afterwards
        if original_length == 0:
            index = None
        else:
            index = slice(original_length,total_length)
        for key in keys:
            self.set(key,'value',keys_vals[key],index)
        for (key,subkey),val in subkeys_vals.items():
            self.set(key,subkey,val,index)

    def _reallocate(self,new_length):
        """Lengthen data arrays."""
        for key in self:
            for subkey in self._data[key]:
                if subkey in self.vector_subkinds:
                    val = self._data[key][subkey]
                    old_length = len(val)
                    if new_length > old_length:
                        self._data[key][subkey] = np.concatenate(
                            (val,
                             np.empty(int(new_length*self._over_allocate_factor-old_length),
                                      dtype=val.dtype)))
        self._length = new_length
        ## increase length of modify time array
        if len(self._row_modify_time) < new_length:
            self._row_modify_time = np.concatenate((
                self._row_modify_time,
                np.full(new_length*self._over_allocate_factor
                        -len(self._row_modify_time),timestamp())))

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
            xsort=True,
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
        if zkeys is None:
            zkeys = self.default_zkeys
        zkeys = [t for t in tools.ensure_iterable(zkeys) if t not in ykeys and t!=xkey and self.is_known(t)] # remove xkey and ykeys from zkeys
        ykeys = [key for key in tools.ensure_iterable(ykeys) if key not in [xkey]+zkeys]
        for t in [xkey,*ykeys,*zkeys]:
            self.assert_known(t)
        ## set xlabel
        xlabel = xkey
        if self.is_known(xkey,'units'):
            xlabel += ' ('+self[xkey,'units']+')'
        ## plot each 
        ymin = {}
        for iy,ykey in enumerate(tools.ensure_iterable(ykeys)):
            ylabel = ykey
            if self.is_known(ykey,'units'):
                ylabel += ' ('+self[ykey,'units']+')'
            for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                if xsort:
                    z.sort(xkey)
                if zlabel_format_function is None:
                    zlabel_format_function = self.default_zlabel_format_function
                zlabel = zlabel_format_function(dz)
                if ynewaxes and znewaxes:
                    ax = plotting.subplot(n=iz+len(zkeys)*iy,fig=fig,ncolumns=ncolumns)
                    color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    label = None
                    if title is None:
                        title = zlabel
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
                    ylabel = None
                    if title is None:
                        title = zlabel
                elif not ynewaxes and not znewaxes:
                    ax = fig.gca()
                    color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = ylabel+' '+zlabel
                    ylabel = None
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
                if self[xkey,'kind'] == 'U':
                    ## if string xkey then ensure different plots are aligned on the axis
                    xkey_unique_strings = self.unique(xkey)
                    x = tools.findin(z[xkey],xkey_unique_strings)
                else:
                    x = z[xkey]
                y = z[ykey]
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
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
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
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                ax.grid(True,color='gray',zorder=-5)
                if self[xkey,'kind'] == 'U':
                    plotting.set_tick_labels_text(xkey_unique_strings,axis='x',ax=ax,rotation=70,)
        if show:
            plotting.show()
        return fig

    # def plot_bar(self,xlabelkey=None,ykeys=None):
        # from matplotlib import pyplot as plt
        # from spectr import plotting
        # ax = plotting.gca()
        # if ykeys is None:
            # ykeys = self.keys()
            # if xlabelkey is not None:
                # ykeys.remove(xlabelkey)
        # x = arange(len(self))
        # if xlabelkey is None:
            # xlabels = x
        # else:
            # xlabels = self[xlabelkey]
        # labels = []
        # for iykey,ykey in enumerate(ykeys):
            # ax.bar(
                # x=x+0.1*(iykey-(len(ykeys)-1)/2),
                # height=self[ykey],
                # width=-0.1,
                # tick_label=[format(t) for t in xlabels],
                # color=plotting.newcolor(iykey),
            # )
            # labels.append(dict(color=plotting.newcolor(iykey),label=ykey))
        # plotting.legend(*labels)
        # for t in ax.xaxis.get_ticklabels():
            # t.set_size('small')
            # t.set_rotation(-45)
        

    def polyfit(self,xkey,ykey,index=None,**polyfit_kwargs):
        return tools.polyfit(
            self.get(xkey,index=index),
            self.get(ykey,index=index),
            self.get(ykey,'unc',index=index),
            **polyfit_kwargs)

def find_common(x,y,keys=None,verbose=False):
    """Return indices of two Datasets that have uniquely matching
    combinations of keys."""
    ## if empty list then nothing to be done
    if len(x)==0 or len(y)==0:
        return(np.array([],dtype=int),np.array([],dtype=int))
    ## use quantum numbers as default keys -- could use qnhash instead
    if keys is None:
        from . import levels
        if isinstance(x,levels.Base):
            ## a hack to use defining_qn for levels/lines as defalt
            ## match keys
            keys = [key for key in x.defining_qn if x.is_known(key)]
        else:
            raise Exception("No keys provided and defining_qn unavailable x.")
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

def get_common(x,y,keys=None,**limit_to_matches):
    """A short cut to find the common levels of a Dynamic_Recarrays object
    and return subset copies that are sorted to match each other."""
    if limit_to_matches is not None:
        x = x.matches(**limit_to_matches)
        y = y.matches(**limit_to_matches)
    i,j = find_common(x,y,keys)
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
    classname = dataset.classname # use the same class as dataset
    retval = make(classname,*args,copy_from=dataset,**kwargs)
    return retval
