import re
import ast
from copy import copy,deepcopy
from pprint import pprint,pformat
import importlib
import warnings
import ast

import numpy as np
from numpy import nan,arange,linspace,array

from . import tools
from .tools import timestamp
from .exceptions import InferException,NonUniqueValueException
from . import convert
from . import optimise
from .optimise import optimise_method,Parameter,Fixed



class Dataset(optimise.Optimiser):

    """A set of data vectors of common length."""

    ## The kind of data that 'value' associated with a key contains.  Influences which subkinds are relevant.
    data_kinds = {
        'f' : {'cast' : lambda x                                        : np.asarray(x,dtype=float) ,'fmt' : '+12.8e','description' : 'float' },
        'a' : {'cast' : lambda x                                        : np.asarray(x,dtype=float) ,'fmt' : '+12.8e','description' : 'positive float' },
        'i' : {'cast' : lambda x                                        : np.asarray(x,dtype=int)   ,'fmt' : 'd'     ,'description' : 'int'   },
        'b' : {'cast' : tools.convert_to_bool_vector_array       ,'fmt' : ''      ,'description'           : 'bool'  },
        'U' : {'cast' : lambda x                                        : np.asarray(x,dtype=str)   ,'fmt' : 's'     ,'description' : 'str'   },
        'O' : {'cast' : lambda x                                        : np.asarray(x,dtype=object),'fmt' : ''      ,'description' : 'object'},
        'h' : {'cast' : lambda x                                        : np.asarray(x,dtype='S20') ,'fmt' : ''      ,'description' : 'SHA1 hash'},
    }

    ##  Kinds of subdata associatd with a key that are vectors of the same length as 'value'
    vector_subkinds = {
        'value'        : {'description' : 'Value of this data'},
        'unc'          : {'description' : 'Uncertainty'                         , 'kind'         : 'f' , 'valid_kinds'                : ('f','a'), 'cast' : lambda x                                  : np.abs(x,dtype=float)     ,'fmt' : '8.2e'  ,'default' : 0.0   ,},
        'step'         : {'description' : 'Default numerical differentiation step size' , 'kind' : 'f' , 'valid_kinds'                : ('f','a'), 'cast' : lambda x                                  : np.abs(x,dtype=float)     ,'fmt' : '8.2e'  ,'default' : 1e-8  ,},
        'vary'         : {'description' : 'Whether to vary during optimisation' , 'kind'         : 'b' , 'valid_kinds'                : ('f','a'), 'cast' : tools.convert_to_bool_vector_array       ,'fmt' : ''      ,'default'               : False ,},
        'ref'          : {'description' : 'Source reference'                    , 'kind'         : 'U' ,                       'cast' : lambda x       : np.asarray(x,dtype='U20') ,'fmt'          : 's'     ,'default'               : nan   ,},
    }

    ##  Kinds of subdata associatd with a key that are single valued but (potentially complex) objects
    scalar_subkinds = {
        'infer'          : {'description':'List of infer functions',},
        'kind'           : {'description':'Kind of data in value corresponding to a key in data_kinds',},
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

    ## all subdata kinds in one convenient dictionary
    all_subkinds = vector_subkinds | scalar_subkinds

    ## prototypes automatically set on instantiation
    default_prototypes = {}
    default_permit_nonprototyped_data = False

    ## used for plotting and sorting the data
    default_zkeys = ()
    default_zlabel_format_function = tools.dict_to_kwargs

    def __init__(
            self,
            name=None,          # name of this Dataset
            permit_nonprototyped_data = True, # allow addition of data that is not prototyped -- kind will be guessed
            permit_indexing = True, # allow the dataset to shrink or grow -- after any init data added
            prototypes = None,      # a dictionary of prototypes
            load_from_file = None,  # load from a file -- guess type
            load_from_string = None, # load from formatted string
            copy_from = None,        # copy form another dataset
            limit_to_match=None,     # dict of things to match
            description='',          # description of this Dataset
            data=None,               # load keys_vals into self, or set with set_value if they are Parameters
            global_attributes=None,  # keys and values of this dictionary are copied to global_attributes in self
            **data_kwargs,           # added to data
    ):
        ## init as optimiser, make a custom form_input_function, save
        ## some extra stuff if output to directory
        optimise.Optimiser.__init__(self)
        self.pop_format_input_function()
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
        self.permit_indexing = True # Data can be added to the end of arrays, but not removal or rearranging of data
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
        self.name = name
        name = self.name
        ## new format input function
        def format_input_function():
            retval = f'{self.name} = {self.classname}({repr(self.name)},'
            if load_from_file is not None:
                retval += f'load_from_file={repr(load_from_file)},'
            if len(data_kwargs)>0:
                retval += '\n'
            for key,val in data_kwargs.items():
                retval += f'    {key}={repr(val)},\n'
            retval += ')'
            return retval
        self.add_format_input_function(format_input_function)
        ## save self to directory
        self.add_save_to_directory_function(self._save_to_directory)
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
        ## load input data
        if data is None:
            data = {}
        data |= data_kwargs
        for key,val in data.items():
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
        ## possible set non-permission for indexing after initial data
        ## added
        self.permit_indexing = permit_indexing
        ## dictionary to store global attributes
        self.global_attributes = {}
        if global_attributes is not None:
            self.global_attributes |= global_attributes

    ## name is adjusted to be proper python symbol when set
    def _set_name(self,name):
        self._name = tools.make_valid_python_symbol_name(name)
    name = property(lambda self:self._name,_set_name)

    def _save_to_directory(self,directory):
        """Save data in directory as a directory, also save as psv
        file if data is not too much."""
        self.save(f'{directory}/data',filetype='directory')
        if len(self)*len(self.keys()) < 10000:
            self.save(f'{directory}/data.psv')

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
            combined_index = self.get_combined_index(index,match,**match_kwargs)
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
            raise Exception(f'Invalid subkey: {repr(subkey)}')
    
    def _guess_kind_from_value(self,value):
        """Guess what kind of data this is from a provided scalar or
        vector value."""
        dtype = np.asarray(value).dtype
        kind = dtype.kind
        return kind

    def set_new(self,key,value,kind=None,**other_metadata):
        """Set key to value with other kinds of subkey metadata also set. Will
        create a prototype first."""
        if key in self:
            raise Exception(f"set_new but key already exists: {repr(key)}")
        if key in self.prototypes:
            raise Exception(f"set_new but key already in prototypes: {repr(key)}")
        if kind is None:
            kind = self._guess_kind_from_value(value)
        self.set_prototype(key,kind=kind,**other_metadata)
        self.set(key,'value',value)

    def set_default(self,key,val,kind=None):
        """Set a default value if not already set."""
        if key not in self:
            self.set_new(key,val,kind)
        self[key,'default'] = val

    def set_prototype(self,key,kind,infer=None,**kwargs):
        """Set a prototype."""
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

    @optimise_method(format_multi_line=3)
    def set_spline(
            self,
            xkey,
            ykey,
            knots,
            order=3,
            default=None,
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Set ykey to spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...]. If index or a match dictionary
        given, then only set these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        if len(_cache) == 0: 
            ## set data
            if not self.is_known(ykey):
                if default is None:
                    raise Exception(f'Setting {repr(ykey)} to spline but it is not known and no default value if provided')
                else:
                    self[ykey] = default
            xspline,yspline = zip(*knots)
            ## get index limit to defined xkey range
            get_combined_index_kwargs |= {f'min_{xkey}':np.min(xspline),f'max_{xkey}':np.max(xspline)}
            _cache['index'] = self.get_combined_index(**get_combined_index_kwargs)
            _cache['xspline'],_cache['yspline'] = xspline,yspline
        ## get cached data
        index,xspline,yspline = _cache['index'],_cache['xspline'],_cache['yspline']
        self.set(ykey,'value',value=tools.spline(xspline,yspline,self.get(xkey,index=index),order=order),index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            self.set(ykey,'unc',nan,index=index)
        ## set vary to False if set, but only on the first execution
        if 'not_first_execution' not in _cache:
            if 'vary' in self._data[ykey]:
                self.set(ykey,'vary',False,index=index)
            _cache['not_first_execution'] = True

    @optimise_method(format_multi_line=3)
    def add_spline(
            self,
            xkey,
            ykey,
            knots,
            order=3,
            default=None,
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Compute a spline function of xkey defined by knots at
        [(x0,y0),(x1,y1),(x2,y2),...] and add to current value of
        ykey. If index or a match dictionary given, then only set
        these."""
        ## To do: cache values or match results so only update if
        ## knots or match values have changed
        if len(_cache) == 0:
            self.assert_known(xkey)
            self.assert_known(ykey)
            xspline,yspline = zip(*knots)
            ## get index limit to defined xkey range
            get_combined_index_kwargs |= {f'min_{xkey}':np.min(xspline),f'max_{xkey}':np.max(xspline)}
            _cache['index'] = self.get_combined_index(**get_combined_index_kwargs)
            _cache['xspline'],_cache['yspline'] = xspline,yspline
        ## get cached data
        index,xspline,yspline = _cache['index'],_cache['xspline'],_cache['yspline']
        ## add to ykey
        ynew = self.get(ykey,'value',index=index)
        yspline = tools.spline(xspline,yspline,self.get(xkey,index=index),order=order)
        self.set(ykey,'value',value=ynew+yspline,index=index)
        ## set previously-set uncertainties to NaN
        if self.is_set(ykey,'unc'):
            self.set(ykey,'unc',nan,index=index)
        ## set vary to False if set, but only on the first execution
        ## (for some reason?!?) 
        if 'not_first_execution' not in _cache:
            if 'vary' in self._data[ykey]:
                self.set(ykey,'vary',False,index=index)
            _cache['not_first_execution'] = True

    @optimise_method(format_multi_line=3)
    def multiply(
            self,
            key,                # key to multiply
            factor,             # factor to multiply by (optimisable)
            from_original_value=False,        # if true then multiply original value on method call during optimisation, else multiply whatever is currenlty there
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Scale key by optimisable factor."""
        ## get index of values to adjsut
        if self._clean_construct:
            index = self.get_combined_index(**get_combined_index_kwargs)
            if index is None:
                index = slice(0,len(self))
            _cache['index'] = index
            if from_original_value:
                original_value = self[key,index]
                _cache['original_value'] = original_value
        ## multiply value
        index = _cache['index']
        if from_original_value:
            value = _cache['original_value']*factor
        else:
            value = self.get(key,'value',index=index)*factor
        self.set(key,'value',value=value,index=index)
        ## not sure how to handle uncertainty -- unset it for now
        self.unset(key,'unc')

    def _increase_char_length_if_necessary(self,key,subkey,new_data):
        """reallocate with increased unicode dtype length if new
        strings are longer than the current array dtype"""
        ## test if (key,subkey is set actually a string data
        if (self.is_set(key,subkey)
            and ((subkey == 'value' and self[key,'kind'] == 'U')
                 or (key in self.vector_subkinds and self.vector_subkinds(subkey,'kind')=='U'))):
            old_data = self[key,subkey]
            ## this is a really hacky way to get the length of string in a numpy array!!!
            old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(old_data.dtype)))
            new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(new_data).dtype)))
            ## reallocate array with new dtype with overallocation if necessary
            if new_str_len > old_str_len:
                t = np.empty(len(self)*self._over_allocate_factor,
                             dtype=f'<U{new_str_len*self._over_allocate_factor}')
                t[:len(self)] = old_data
                self._data[key][subkey] = t

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
            ## reallocate string arrays if needed
            self._increase_char_length_if_necessary(key,'value',value)
            # if self[key,'kind'] == 'U':
                # ## this is a really hacky way to get the length of string in a numpy array!!!
                # old_str_len = int(re.sub(r'[<>]?U([0-9]+)',r'\1', str(self.get(key).dtype)))
                # new_str_len =  int(re.sub(r'^[^0-9]*([0-9]+)$',r'\1',str(np.asarray(value).dtype)))
                # if new_str_len > old_str_len:
                    # ## reallocate array with new dtype with overallocation
                    # t = np.empty(
                        # len(self)*self._over_allocate_factor,
                        # dtype=f'<U{new_str_len*self._over_allocate_factor}')
                    # t[:len(self)] = self.get(key)
                    # data['value'] = t
            ## cast scalar data correctly
            if np.isscalar(value):
                value = self[key,'cast'](array([value]))
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
                raise NotImplementedError()
            ## use data to infer kind if necessary
            if kind is not None:
                data['kind'] = kind
            if 'kind' not in data:
                value = np.asarray(value)
                data['kind'] = self._guess_kind_from_value(value)
                # data['kind'] = value.dtype.kind
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
        if ('valid_kinds' in subkind and self.get_kind(key) not in subkind['valid_kinds']):
            raise Exception(f"The value kind of {repr(key)} is {repr(data['kind'])} and is invalid for setting {repr(subkey)}")
        if subkind['kind'] == 'O':
            raise ImplementationError()
        # if self.verbose:
            # print(f'{self.name}: setting ({key}:{subkey})')
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
            ## reallocate string arrays if needed
            self._increase_char_length_if_necessary(key,subkey,value)
            ## set indexed data
            data[subkey][:len(self)][index] = subkind['cast'](value)

    row_modify_time = property(lambda self:self._row_modify_time[:self._length])
    global_modify_time = property(lambda self:self._global_modify_time)

    def get(self,key,subkey='value',index=None,units=None,match=None,**match_kwargs):
        """Get value for key or (key,subkey). This is the data in place, not a
        copy."""
        index = self.get_combined_index(index,match,**match_kwargs)
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
            

    def get_kind(self,key):
        return self._data[key]['kind']

    def modify(self,key,rename=None,**new_metadata):
        """Modify metadata of a key, change its units, or rename it."""
        self.assert_known(key)
        ## rename key by adding a new one and unsetting the original --
        ## breaking inferences
        if rename is not None:
            for subkey in self.all_subkinds:
                if self.is_set(key,subkey):
                    self[rename,subkey] = self[key,subkey]
            self.unset(key)
            key = rename
        for subkey,val in new_metadata.items():
            if subkey == 'description':
                self[key,subkey] = val
            elif subkey == 'kind':
                if self[key,subkey] != val:
                    new_data = {subkey:self[key,subkey] for subkey in all_subkinds if self.is_set(key,subkey)}
                    new_data['kind'] = val
                    self.unset(key)
                    self.set_new(key,**new_data)
            elif subkey == 'units':
                if self.is_set(key,'units'):
                    ## convert units of selected subkeys, convert all
                    ## first before setting
                    new_data = {}
                    for tsubkey in ('value','unc','step',):
                        if self.is_set(key,tsubkey):
                            new_data[tsubkey] = convert.units(self[key,tsubkey],self[key,'units'],val)
                    for tsubkey,tval in new_data.items():
                            self[key,tsubkey] = tval
                self[key,'units'] = val
            elif subkey == 'fmt':
                self[key,'fmt'] = val
            else:
                raise NotImplementedError(f'Modify {subkey}')
        
    # def change_units(self,key,new_units):
        # """Change units of key to new_units."""
        # old_units = self[key,'units']
        # for subkey in ('value','unc','step'):
            # if self.is_set(key,subkey):
                # self[key,subkey] = convert.units(self[key,subkey],old_units,new_units)
        # self[key,'units'] = new_units

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
            
    def get_combined_index(self,index=None,match=None,**match_kwargs):
        """Combined specified index with match arguments as integer array. If
        no data given the return None"""
        ## combine match dictionaries
        if match is None:
            match = {}
        if match_kwargs is not None:
            match |= match_kwargs
        if index is None and len(match) == 0:
            ## no indices at all
            retval = None
        elif np.isscalar(index):
            ## single index
            retval = index
            if len(match) > 0:
                raise Exception("Single index cannot be addtionally matched.")
        else:
            ## get index array
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
            if len(match) > 0:
                imatch = self.match(match)
                retval = retval[tools.find(imatch[retval])]
        return retval
            
    def get_combined_index_bool(self,**get_combined_index_kwargs):
        """Combined specified index with match arguments as integer array. If
        no data given the return None"""
        index = get_combined_index(**get_combined_index_kwargs)
        if index is None:
            raise Exception('Cannot return bool array combined index if None.')
        if np.isscalar(index):
            raise Exception("Cannot return bool array for Single index.")
        ## convert to boolean array
        retval = np.full(len(self),False)
        retaval[match] = True
        return retval

    @optimise_method(format_multi_line=99)
    def set_value(
            self,
            key,
            value,
            default=None,
            _cache=None,
            **get_combined_index_kwargs
    ):
        """Set a value and it will be updated every construction and may be a
        Parameter for optimisation."""
        ## cache matching indices
        if self._clean_construct:
            _cache['index'] = self.get_combined_index(**get_combined_index_kwargs)
            ## set a default value if key is not currently known
            if not self.is_known(key) and default is not None:
                self[key] = default
        index = _cache['index']
        ## set the data
        self.set(key,'value',value,index=index,set_changed_only= True)
        if isinstance(value,Parameter):
            # if value.vary == True: #  DEBUG
                # print('DEBUG:', value)
            self.set(key,'unc' ,value.unc ,index=index)
            if self._clean_construct:
                self.set(key,'step',value.step,index=index)
                self.set(key,'vary',False     ,index=index)
        # ## set vary to False if set, but only on the first execution
        # if 'not_first_execution' not in _cache:
            # if 'vary' in self._data[key]:
                # self.set(key,'vary',False,index=combined_index)
            # _cache['not_first_execution'] = True

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

    def match_keys(self,regexp=None,beg=None,end=None,):
        """Return a list of keys matching any of regex or beginning/ending string beg/end."""
        keys = []
        if regexp is not None:
            keys += [key for key in self if re.match(regexp,key)]
        if beg is not None:
            keys += [key for key in self if len(key)>=len(beg) and key[:len(beg)] == beg]
        if end is not None:
            keys += [key for key in self if len(key)>=len(end) and key[-len(end):] == end]
        return keys
            
    def match_keys_matches(self,regexp):
        """Return a list of keys matching any of regexp or beginning/ending string beg/end."""
        retval = []
        for key in self:
            if r:=re.match(regexp,key):
                retval.append((key,*r.groups()))
        return retval 

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

    def __setitem__(self,key,value):
        """Set key, (key,subkey), (key,index), (key,subkey,index) to
        value. If key='key:subkey' then decode this."""
        if isinstance(key,str):
            tkey,tsubkey,tindex = key,'value',None
        elif len(key) == 1:
            tkey,tsubkey,tindex = key[0],'value',None
        elif len(key) == 2:
            if isinstance(key[1],str):
                tkey,tsubkey,tindex = key[0],key[1],None
            else:
                tkey,tsubkey,tindex = key[0],'value',key[1]
        elif len(key) == 3:
                tkey,tsubkey,tindex = key[0],key[1],key[2]
        # ## maybe a key:subkey encoded key
        # if tsubkey == 'value' and ':' in tkey:
            # tkey,tsubkey = tkey.split(':')
        self.set(tkey,tsubkey,value,tindex)
       
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

    def unlink_all_inferences(self):
        """Mark all data as not inferred."""
        self.unlink_inferences(self.keys())

    def add_infer_function(self,key,dependencies,function):
        """Add a new method of data inference."""
        self.prototypes[key]['infer'].append((dependencies,function))

    def index(self,index):
        """Index all array data in place."""
        if not self.permit_indexing:
            raise Exception('Indexing not permitted')
        if isinstance(index,(int,np.int64)):
            ## allow single integer indexing
            index = [index]
        original_length = len(self)
        for key,data in self._data.items():
            for subkey in data:
                if subkey in self.vector_subkinds:
                    data[subkey] = data[subkey][:original_length][index]
            self._length = len(data['value'])

    def remove(self,index):
        """Remove indices."""
        index = self.get_combined_index_bool(index)
        self.index(~index)

    def __deepcopy__(self,memo):
        """Manually controlled deepcopy which does seem to be faster than the
        default for some reason. Relies on all mutable attributes
        being included in attr_to_deepcopy."""
        retval = copy(self)
        memo[id(self)] = retval # add this in case of circular references to it below
        for attr_to_deepcopy in (
                '_data',
                '_row_modify_time',
                'prototypes',
                'global_attributes',
                '_construct_functions',
                '_post_construct_functions',
                '_plot_functions',
                '_monitor_functions',
                '_save_to_directory_functions',
                '_format_input_functions',
                '_suboptimisers',
                'residual',
                'combined_residual',
                'store',
        ):
            setattr(retval,attr_to_deepcopy, deepcopy(getattr(self,attr_to_deepcopy), memo))
        return retval

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
            keys_re=None,
            subkeys=None,
            copy_global_attributes=True,
            copy_inferred_data=False,
            **get_combined_index_kwargs
    ):
        """Copy all values and uncertainties from source Dataset."""
        self.clear()            # total data reset
        if keys_re is not None:
            if keys is None:
                keys = []
            else:
                keys = list(keys)
            keys.extend(source.match_keys(regexp=keys_re))
        if keys is None:
            if copy_inferred_data:
                keys = source.keys()
            else:
                keys = source.explicitly_set_keys()
        self.permit_nonprototyped_data = source.permit_nonprototyped_data
        ## get matching indices
        index = source.get_combined_index(**get_combined_index_kwargs)
        ## copy data and selected prototype data
        for key in keys:
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
        ## copy global_attributes
        if copy_global_attributes:
            self.global_attributes = deepcopy(source.global_attributes)

    @optimise_method()
    def copy_from_and_optimise(
            self,
            source,
            keys=None,
            skip_keys=(),
            subkeys=('value','unc','description','units','fmt'),
            copy_inferred_data=False,
            _cache=None,
            **get_combined_index_kwargs
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
            index = source.get_combined_index(**get_combined_index_kwargs)
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

    def match_re(self,keys_vals=None,**kwarg_keys_vals):
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
        suffix 'min_' or 'max_' then match anything greater/lesser or
        equal to this value.  If key has suffix _not then match not
        equal."""
        ## combine all match keys/vals
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwarg_keys_vals
        ## update match by key/val
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            if len(key) > 4 and key[:4] == 'not_' and not self.is_known(key):
                ## negate match
                i &= ~self.match({key[4:]:val})
            elif not np.isscalar(val):
                if len(key) > 6 and key[:6] == 'range_' and not self.is_known(key):
                    ## find all values in a the range of a pair
                    if len(val) != 2:
                        raise Exception(r'Invalid range: {val!r}')
                    i &= (self[key[6:]] >= val[0]) & (self[key[6:]] <= val[1])
                else:
                    ## multiple possibilities to match against
                    i &= np.any([self.match({key:vali}) for vali in val],0)
            else:
                ## a single value to match against
                if self.is_known(key):
                    ## a simple equality
                    if val is np.nan:
                        ## special case for equality with nan
                        i &= np.isnan(self[key])
                    else:
                        ## simple equality
                        i &= self[key]==val
                elif len(key) > 4 and key[:4] == 'min_':
                    ## find all larger values
                    i &= (self[key[4:]] >= val)
                elif len(key) > 4 and key[:4] == 'max_':
                    ## find all smaller values
                    i &= (self[key[4:]] <= val)
                elif len(key) > 3 and key[:3] == 're_':
                    ## recursively get reverse match for this key
                    i &= self.match_re({key[3:]:val})
                else:
                    ## total failure
                    raise InferException(f'Could not match key: {repr(key)}')
        return i

    def find(self,*match_args,**match_kwargs):
        """Return as a array of matching indices."""
        i = tools.find(self.match(*match_args,**match_kwargs))
        return i

    def matches(self,*args,**kwargs):
        """Returns a copy reduced to matching values."""
        return self.copy(index=self.match(*args,**kwargs),copy_inferred_data=False)

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

    def unique_dicts_match(self,*keys,return_bool=True):
        """Return pairs where the first element is a dictionary of unique
        combinations of keys and the second is a boolean array matching this
        combination. If return_bool=False then returns an array of matching indices instead."""
        retval = []
        for d in self.unique_dicts(*keys):
            if return_bool:
                retval.append((d,self.match(**d)))
            else:
                retval.append((d,self.find(**d)))
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
        self.prototypes[key].setdefault('infer',[])
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
                            print(f'{self.name}:',
                                  ''.join(['    ' for t in range(depth)])
                                  +f'{self.name}: Inferred uncertainty: {repr(key)}')
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
                default_value = self.prototypes[key]['default']
                if self.verbose:
                    print(f'{self.name}:',''.join(['    ' for t in range(depth)])+f'Cannot infer {repr(key)} and setting to default value: {repr(default_value)}')
                self._set_value(key,default_value,dependencies=())
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
            subkeys=None,
    ):
        """Return as a structured dict."""
        ## default to all data
        if keys is None: 
            keys = list(self.keys())
        if subkeys is None:
            subkeys = ('value','unc','description','units')
        ## add data
        retval = {}
        retval['classname'] = self.classname
        retval['description'] = self.description
        if len(self.global_attributes) > 0:
            retval['global_attributes'] = self.global_attributes
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
        keys = tools.ensure_iterable(keys)
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
        i = self.find(**matching_keys_vals)
        if len(i) == 0:
            raise NonUniqueValueException(f'No matching row found: {matching_keys_vals=}')
        if len(i) > 1:
            raise NonUniqueValueException(f'Multiple matching rows found: {matching_keys_vals=}')
        return i[0]

    def unique_row(self,return_index=False,**matching_keys_vals):
        """Return uniquely-matching row as a dictionary."""
        i = self.find_unique(**matching_keys_vals)
        d = self.as_flat_dict(index=i)
        if return_index:
            return d,i
        else:
            return d

    def unique_value(self,key,**matching_keys_vals):
        """Return value of key from a row that uniquely matches
        keys_vals."""
        i = self.find_unique(**matching_keys_vals)
        value = self.get(key,index=i)
        return value

    def sort(self,sort_keys,reverse=False):
        """Sort rows according to key or keys."""
        if isinstance(sort_keys,str):
            sort_keys = [sort_keys]
        i = np.argsort(self[sort_keys[0]])
        if reverse:
            i = i[::-1]
        for key in sort_keys[1:]:
            i = i[np.argsort(self[key][i])]
        self.index(i)

    def format(
            self,
            keys=None,
            keys_re=None,
            delimiter=' | ',
            line_ending='\n',
            simple=False,       # print data in a table
            unique_values_as_default=False,
            subkeys=('value','unc','vary','step','ref','default','units','fmt','kind','description',),
            exclude_keys_with_leading_underscore=True, # if no keys specified, do not include those with leading underscores
            exclude_inferred_keys=False, # if no keys specified, do not include those which are not explicitly set
            quote=False,
    ):
        """Format data into a string representation."""
        if keys is None:
            if keys_re is None:
                keys = self.keys()
                if exclude_keys_with_leading_underscore:
                    keys = [key for key in keys if key[0]!='_']
                if exclude_inferred_keys:
                    keys = [key for key in keys if not self.is_inferred(key)]
            else:
                keys = []
        if keys_re is not None:
            keys = {*keys,*self.match_keys(regexp=keys_re)}
        ##
        for key in keys:
            self.assert_known(key)
        ## collect columns of data -- maybe reducing to unique values
        columns = []
        header_values = {}
        for key in keys:
            self.assert_known(key)
            if len(self) == 0:
                break
            formatted_key = ( f'"{key}"' if quote else key )
            if (
                    not simple and unique_values_as_default # input parameter switch
                    and not np.any([self.is_set(key,subkey) for subkey in subkeys if subkey in self.vector_subkinds and subkey != 'value']) # no other vector subdata 
                    and len((tval:=self.unique(key))) == 1 # q unique value
            ): 
                ## value is unique, format value for header
                header_values[key] = tval[0] 
            else:
                ## format columns
                for subkey in subkeys:
                    if self.is_set(key,subkey) and subkey in self.vector_subkinds:
                        if subkey == 'value':
                            formatted_key = (f'"{key}"' if quote else f'{key}')
                        else:
                            formatted_key = (f'"{key}:{subkey}"' if quote else f'{key}:{subkey}')
                        fmt = self._get_attribute(key,subkey,'fmt')
                        kind = self._get_attribute(key,subkey,'kind')
                        if quote and kind == 'U':
                            vals = ['"'+format(t,fmt)+'"' for t in self[key,subkey]]
                        else:
                            vals = [format(t,fmt) for t in self[key,subkey]]
                        width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                        columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
        ## format key metadata
        formatted_metadata = []
        if not simple:
            ## include description of keys
            for key in keys:
                metadata = {}
                if key in header_values:
                    metadata['default'] = header_values[key]
                ## include much metadata in description. If it is
                ## dictionary try to semi-align the keys
                for subkey in subkeys:
                    if subkey in self.scalar_subkinds and self.is_set(key,subkey):
                        metadata[subkey] = self[key,subkey]
                if isinstance(metadata,dict):
                    line = f'{key:20} = {{ '
                    for tkey in subkeys:
                        if tkey not in self.scalar_subkinds:
                            continue
                        if tkey not in metadata:
                            continue
                        tval = metadata[tkey]
                        ## description length for alignment is in
                        ## 40-char quanta, else 15 char
                        if tkey == 'description':
                            tfmt = str(40*((len(tval)-1)//40+1))
                        else:
                            tfmt= '25'
                        line += format(f'{tkey!r}: {tval!r}, ',tfmt)
                    line += '}'
                else:
                    line = f'{key:20} = {metadata!r}'
                formatted_metadata.append(line)
        ## make full formatted string
        retval = ''
        if not simple:
            retval += f'[classname]\n{self.classname}\n'
        if not simple and self.description not in (None,''):
            retval += f'[description]\n{self.description}\n'
        if len(self.global_attributes) > 0:
            retval += f'[global_attributes]\n'+'\n'.join([f'{repr(tkey):20} : {repr(tval)}' for tkey,tval in self.global_attributes.items()])+'\n'
        if formatted_metadata != []:
            retval += '[metadata]\n'+'\n'.join(formatted_metadata)
        if columns != []:
            if len(retval) > 0:
                retval += '\n[data]\n'
            retval += line_ending.join([delimiter.join(t) for t in zip(*columns)])+line_ending
        return retval

    def format_metadata(
            self,
            keys=None,
            keys_re=None,
            subkeys=('default','description','units','fmt','kind',),
            exclude_keys_with_leading_underscore=True, # if no keys specified, do not include those with leading underscores
            exclude_inferred_keys=False, # if no keys specified, do not include those which are not explicitly set
    ):
        """Format metadata into a string representation."""
        ## determine keys to include
        if keys is None:
            if keys_re is None:
                keys = self.keys()
                if exclude_keys_with_leading_underscore:
                    keys = [key for key in keys if key[0]!='_']
                if exclude_inferred_keys:
                    keys = [key for key in keys if not self.is_inferred(key)]
            else:
                keys = []
        if keys_re is not None:
            keys = {*keys,*self.match_keys(regexp=keys_re)}
        for key in keys:
            self.assert_known(key)
        ## format key metadata
        metadata = {}
        for key in keys:
            ## include much metadata in description. If it is
            ## dictionary try to semi-align the keys
            metadata[key] = {}
            for subkey in subkeys:
                if subkey in self.scalar_subkinds and self.is_set(key,subkey):
                    metadata[key][subkey] = self[key,subkey]
        retval = tools.format_dict(metadata,newline_depth=0,enclose_in_braces=False)
        return retval

    def print_metadata(self,*args_format_metadata,**kwargs_format_metadata):
        """Print a string representation of metadata"""
        string = self.format_metadata(*args_format_metadata,**kwargs_format_metadata)
        print( string)

    def format_as_list(self):
        """Form as a valid python list of lists."""
        retval = f'[ \n'
        data = self.format(
            delimiter=' , ',
            simple=True,
            quote=True,
        )
        for line in data.split('\n'):
            if len(line)==0:
                continue
            retval += '    [ '+line+' ],\n'
        retval += ']'
        return retval

    def __str__(self):
        return self.format(simple=True)

    def __repr__(self):
        # if len(self)>50:
        return self.name
        # return f"{self.classname}(load_from_string='''\n{self.format_flat()}''')"
            
    def save(
            self,
            filename,
            keys=None,
            subkeys=None,
            filetype=None,           # 'text' (default), 'hdf5', 'directory'
            **format_kwargs,
    ):
        """Save some or all data to a file."""
        if filetype is None:
            ## if not provided as an input argument then get save
            ## format form filename, or default to text
            filetype = tools.infer_filetype(filename)
            if filetype == None:
                filetype = 'text'
        if keys is None:
            keys = self.keys()
        if subkeys is None:
            ## get a list of default subkeys, ignore those beginning
            ## with "_" and some specific keys
            # subkeys = [subkey for subkey in self.vector_subkinds if subkey[0] != '_']
            subkeys = [subkey for subkey in self.all_subkinds if
                       (subkey[0] != '_') and subkey not in ('infer','cast')]
        if filetype == 'hdf5':
            ## hdf5 file
            tools.dict_to_hdf5(filename,self.as_dict(keys=keys,subkeys=subkeys),verbose=False)
        elif filetype == 'npz':
            ## numpy archive
            np.savez(filename,self.as_dict(keys=keys,subkeys=subkeys))
        elif filetype == 'directory':
            ## directory of npy files
            tools.dict_to_directory(filename,self.as_dict(keys=keys,subkeys=subkeys),repr_strings=True)
        elif filetype == 'text':
            ## space-separated text file
            format_kwargs.setdefault('delimiter',' ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))
        elif filetype == 'rs':
            ## ␞-separated text file
            format_kwargs.setdefault('delimiter',' ␞ ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))
        elif filetype == 'psv':
            ## |-separated text file
            format_kwargs.setdefault('delimiter',' | ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))
        elif filetype == 'csv':
            ## comma-separated text file
            format_kwargs.setdefault('delimiter',', ')
            tools.string_to_file(filename,self.format(keys,**format_kwargs))
        else:
            raise Exception(f'Do not know how save to {filetype=}')
            
    def load(self,filename,filetype=None,**load_method_kwargs):
        '''Load data from a file.'''
        if filetype is None:
            ## if not provided as an input argument then get save
            ## format form filename, or default to text
            filetype = tools.infer_filetype(filename)
            if filetype == None:
                filetype = 'text'
        if filetype == 'hdf5':
            self._load_from_hdf5(filename,**load_method_kwargs)
        elif filetype == 'directory':
            self._load_from_directory(filename,**load_method_kwargs)
        elif filetype == 'npz':
            self._load_from_npz(filename,**load_method_kwargs)
        elif filetype == 'org':
            self._load_from_org(filename,**load_method_kwargs)
        elif filetype == 'text':
            self._load_from_text(filename,**load_method_kwargs,delimiter=' ')
        elif filetype == 'rs':
            self._load_from_text(filename,**load_method_kwargs,delimiter='␞')
        elif filetype == 'psv':
            self._load_from_text(filename,**load_method_kwargs,delimiter='|')
        elif filetype == 'csv':
            self._load_from_text(filename,**load_method_kwargs,delimiter=',')
        else:
            raise Exception(f"Unrecognised data filetype: {filename=} {filetype=}")

    def load_from_dict(
            self,
            data,
            keys=None,          # load only this data
            flat=False,
            metadata=None,
            translate_keys=None, # from key in file to key in self, None for skip
            translate_keys_regexp=None, # a list of (regexp,subs) pairs to translate keys -- operate successively on each key
            translate_from_anh_spectrum=False, # HACK to translate keys from spectrum module
            load_classname_only=False,
    ):
        """Load from a structured dictionary as produced by as_dict."""
        ## translate key with direct substitutions
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
        ## translate keys with regexps
        if translate_keys_regexp is not None:
            for key in list(data.keys()):
                original_key = key
                for match_re,sub_re in translate_keys_regexp:
                    key = re.sub(match_re,sub_re,key)
                if key != original_key:
                    data[key] = data.pop(original_key)
        ## hack -- sometimes classname is quoted, so remove them
        if 'classname' in data:
            data['classname'] = str(data['classname'])
            if ((data['classname'][0] == "'" and data['classname'][-1] == "'")
                or  (data['classname'][0] == '"' and data['classname'][-1] == '"')):
                data['classname'] = data['classname'][1:-1]
        ## if load_classname_only then return it do nothing else. None
        ## if unknown. This is a hack, makes self pretty unusable
        ## apart from the classname attribute
        if load_classname_only:
            if 'classname' in data:
                self.classname =  data['classname']
            else:
                self.classname =  None
                ## test loaded classname matches self
        if 'classname' in data:
            if data['classname'] != self.classname:
                warnings.warn(f'The loaded classname, {repr(data["classname"])}, does not match self, {repr(self.classname)}, and it will be ignored.')
            data.pop('classname')
        ## build structured data from flat data 
        if flat:
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
        ## update metadata
        if metadata is not None:
            for key,info in metadata.items():
                for subkey,val in info.items():
                    data[key][subkey] = val
        ## 2021-06-11 HACK TO ACCOUNT FOR DEPRECATED ATTRIBUTES DELETE ONE DAY
        if 'default_step' in data: # HACK
            data.pop('default_step') # HACK
        ## END OF HACK
        ## description is saved in data
        if 'description' in data:
            self.description = str(data.pop('description'))
        ## global_attributes are saved in data, try to evalute as literal, or keep as string on fail
        if 'global_attributes' in data:
            for key,val in data.pop('global_attributes').items():
                try:
                    self.global_attributes[key] = ast.literal_eval(val)
                except ValueError:
                    self.global_attributes[key] = val
        ## HACK REMOVE ASSOC 2021-06-21 DELETE ONE DAY
        for key in data:
            if 'assoc' in list(data[key]):
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
            ## no data
            if 'value' not in data[key]:
                raise Exception(f'No "value" subkey in data {repr(key)} with subkeys {repr(list(data[key]))}')
            ## if kind is then add a prototype (or replace
            ## existing if the kinds do not match)
            if 'kind' in data[key]:
                kind = data[key].pop('kind')
                if key not in self.prototypes or self.prototypes[key]['kind'] != kind:
                    self.set_prototype(key,kind)
            ## vector data but given as a scalar -- defer loading
            ## until after vector data so the length of data is known
            if np.isscalar(data[key]['value']):
                scalar_data[key] = data[key]
            ## vector data -- add value and subkeys
            else:
                self[key,'value'] = data[key].pop('value')
                for subkey in data[key]:
                    self[key,subkey] = data[key][subkey]
        ## load scalar data
        for key in scalar_data:
            self[key,'value'] = scalar_data[key].pop('value')
            for subkey in scalar_data[key]:
                self[key,subkey] = scalar_data[key][subkey]

    def load_from_parameters_dict(self,parameters):
        """Load a dict recursively into a flat scalar list. Only scalars,
            Parameters, and dictionaries with string keys are
            added. Everything else is ignored."""
        def recursively_flatten_scalar_dict(data,prefix=''):
            from .optimise import Parameter
            retval = {}
            for key,val in data.items():
                if isinstance(key,str):
                    key = prefix + key
                    if np.isscalar(val):
                        retval[key] = val
                    elif isinstance(val,Parameter):
                        retval[key] = val.value
                        retval[key+':unc'] = val.unc
                    elif isinstance(val,dict):
                        tdata = recursively_flatten_scalar_dict(val,prefix=key+'_')
                        for tkey,tval in tdata.items():
                            retval[tkey] = tval
            return retval
        ## load columns
        data = {}
        for i,(keyi,datai) in enumerate(parameters.items()):
            ## new data point
            datai = recursively_flatten_scalar_dict(datai)
            datai['key'] = keyi
            ## ensure key consistency
            for key in datai:
                if key not in data:
                    if i==0:
                        data[key] = []
                    else:
                        data[key] = [nan for i in range(i)]
            for key in data:
                if key not in datai:
                    datai[key] = nan
            for key,val in datai.items():
                data[key].append(val)
        self.load_from_dict(data,flat=True)


    def _load_from_directory(self,filename,**load_from_dict_kwargs):
        """Load data stored in a structured directory tree."""
        data = tools.directory_to_dict(filename)
        self.load_from_dict(data,**load_from_dict_kwargs)

    def _load_from_hdf5(self,filename,load_attributes=True,**load_from_dict_kwargs):
        """Load data stored in a structured or unstructured hdf5 file."""
        data = tools.hdf5_to_dict(filename,load_attributes=load_attributes)
        ## hack to get flat data or not
        for val in data.values():
            if isinstance(val,dict):
                flat = False
                break
        else:
            flat = True
        self.load_from_dict(data,flat=flat,**load_from_dict_kwargs)

    def _load_from_npz(self,filename,**load_from_dict_kwargs):
        """numpy npz archive.  get as scalar rather than
        zero-dimensional numpy array"""
        data = {}
        for key,val in np.load(filename).items():
            if val.ndim == 0:
                val = val.item()
            data[key] = val
        self.load_from_dict(data,flat=True,**load_from_dict_kwargs)

    def _load_from_org(self,filename,**load_from_dict_kwargs):
        """Load form org table"""
        data = tools.org_table_to_dict(filename,table_name)
        self.load_from_dict(data,flat=True,**load_from_dict_kwargs)

    def _load_from_text(
            self,
            filename,
            comment='#',
            labels_commented=False,
            delimiter=' ',
            txt_to_dict_kwargs=None,
            load_classname_only=False,
            **load_from_dict_kwargs
    ):
        """Load data from a text-formatted file."""
        ## text table to dict with header
        if txt_to_dict_kwargs is None:
            txt_to_dict_kwargs = {}
        txt_to_dict_kwargs |= {'delimiter':delimiter,'labels_commented':labels_commented}
        filename = tools.expand_path(filename)
        data = {}
        metadata = {}
        ## load header
        escaped_comment = re.escape(comment)
        blank_line_re = re.compile(r'^ *$')
        commented_line_re = re.compile(f'^ *{escaped_comment} *(.*)$')
        beginning_of_section_re = re.compile(f'^ *\\[([^]]+)\\] *$') 
        key_line_without_value_re = re.compile(f'^ *([^# ]+) *# *(.+) *') # no value in line
        key_line_with_value_re = re.compile(f'^ *([^= ]+) *= *([^#]*[^ #])') # may also contain description
        current_section = 'data'
        valid_sections = ('classname','description','keys','data','metadata','global_attributes')
        section_iline = 0       # how many lines read in this section
        classname = None
        description = None
        global_attributes = []
        with open(filename,'r') as fid:
            for iline,line in enumerate(fid):
                ## remove newline
                line = line[:-1]
                ## check for bad section title
                if current_section not in valid_sections:
                    raise Exception(f'Invalid data section: {repr(current_section)}. Valid sections: {repr(valid_sections)}')
                ## remove comment character unless in data section —
                ## then skip the line, or description then keep it in
                ## place
                if r:=re.match(commented_line_re,line):
                        continue
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
                    if load_classname_only:
                        ## shortcut, load classname ignore the rest of the data
                        self.classname = classname
                        return
                elif current_section == 'description':
                    ## add to description
                    description += '\n'+line
                elif current_section == 'global_attributes':
                    ## add global attribute
                    global_attributes.append(line)
                elif current_section == 'keys':
                    ## decode key line getting key, value, and any metadata
                    r = re.match(
                        f'^(?:{escaped_comment}| )*([^#= ]+) *(?:= *([^ #]+))? *(?:# *(.* *))?',
                        line)
                    key = None
                    value = None
                    info = None
                    if r:
                        if r.group(1) is not None:
                            key = r.group(1)
                        if r.group(2) is not None:
                            value = ast.literal_eval(r.group(2))
                        if r.group(3) is not None:
                            try:
                                info = ast.literal_eval(r.group(3))
                                if not isinstance(info,dict):
                                    info = {'description':r.group(3)}
                            except:
                                info = {'description':r.group(3)}
                        if value is not None:
                            data[key] = value
                        if info is not None:
                            metadata[key] = info
                elif current_section == 'metadata': 
                    ## decode key line getting key, value, and any
                    ## metadata from an python-encoded dictionary e.g.,
                    ## key={'description':"abd",kind='f',value=5.0,...}.
                    ## Or key=description_string.
                    r = re.match(f'^(?:{escaped_comment}| )*([^= ]+)(?: *= *(.+))?',line) 
                    if r:
                        key = r.group(1)
                        if r.group(2) is None:
                            key_metadata = None
                        else:
                            try:
                                key_metadata = ast.literal_eval(r.group(2))
                            except:
                                raise Exception(f"Invalid metadata encoding for {repr(key)}: {repr(r.group(2))}")
                            if isinstance(key_metadata,dict):
                                pass
                            elif isinstance(key_metadata,str):
                                key_metadata = {'description':key_metadata}
                            else:
                                raise Exception(f'Could not decode key metadata for {key}: {repr(key_metadata)}')
                            # if 'value' in key_metadata:
                            #     data[key] = key_metadata.pop('value')
                            if 'default' in key_metadata:
                                data[key] = key_metadata['default']
                            metadata[key] = key_metadata
                elif current_section == 'data':
                    ## remainder of data is data, no more header to
                    ## process
                    break
        ## load array data
        data.update(
            tools.txt_to_dict(
                filename,
                skiprows=iline,
                try_cast_numeric=False,
                **txt_to_dict_kwargs))
        # ## a blank key with all nan data indicates a leading or trailing delimiter -- delete it
        # if '' in data and np.all(np.isnan(data[''])):
        #     data.pop('')
        ## a blank key indicates a leading or trailing delimiter -- delete it
        if '' in data:
            data.pop('')
        ## load classname and description
        if classname is not None:
            data['classname'] = classname
        if description is not None:
            data['description'] = description
        if len(global_attributes) > 0:
            tdict = '{'+','.join(global_attributes)+'}'
            self.global_attributes |= ast.literal_eval(tdict)
        ## if there is no kind for this key then try and cast to numeric data
        for key in data:
            if key not in metadata or 'kind' not in metadata[key]:
                data[key] = tools.try_cast_to_numeric_array(data[key])
        ## load into self
        self.load_from_dict(data,metadata=metadata,flat=True,**load_from_dict_kwargs)

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
        self._load_from_text(tmpfile.name,delimiter=delimiter,**load_kwargs)

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

    def concatenate(self,new_dataset,keys=None,defaults=None):
        """Extend self by new_dataset using keys existing in self. New data updated
        on optimisaion if new_dataset changes."""
        ## process defaults, keys that are missing a subkey are
        ## converted to (key,'value').
        if defaults is None:
            defaults = {}
        for key in list(defaults):
            if isinstance(key,str):
                defaults[key,'value'] = defaults.pop(key)
        ## test if there is currently any data
        if len(self.keys()) == 0:
            ## if currently no data at all then copy from new_dataset and return
            if keys is None:
                keys = new_dataset.keys()
            self.copy_from(new_dataset,keys)
            return
        else:
            ## else concatenate to existing data
            if keys is None:
                ## get combined key list if not given
                keys = list({*self.explicitly_set_keys(),*new_dataset.explicitly_set_keys()})
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
            # for key,data in self._data.items():
            #     for subkey in data:
            #         if subkey in self.vector_subkinds:
            #             if new_dataset.is_known(key,subkey):
            #                 new_val = new_dataset[key,subkey]
            #             elif self._has_attribute(key,subkey,'default'):
            #                 new_val = self._get_attribute(key,subkey,'default')
            #             elif (key,subkey) in defaults:
            #                 new_val = defaults[key,subkey]
            #             else:
            #                 raise Exception(f'Unknown to concatenated data: ({repr(key)},{repr(subkey)})')
            #             ## increase char-length of string arrays if needed and insert new data
            #             self._increase_char_length_if_necessary(key,subkey,new_val)
            #             data[subkey][old_length:total_length] = self._get_attribute(key,subkey,'cast')(new_val)
            for key,data in self._data.items():
                for subkey in self.vector_subkinds:
                    if self.is_set(key,subkey) or new_dataset.is_set(key,subkey):

                        self.assert_known(key,subkey)
                        new_dataset.assert_known(key,subkey)

                        if self.is_known(key,subkey) and new_dataset.is_known(key,subkey):
                            new_val = new_dataset[key,subkey]
                        # elif self._has_attribute(key,subkey,'default'):
                            # new_val = self._get_attribute(key,subkey,'default')
                        elif (key,subkey) in defaults:
                            new_val = defaults[key,subkey]
                        else:
                            raise Exception(f'Unknown to concatenated data: ({repr(key)},{repr(subkey)})')

                        ## increase char-length of string arrays if needed and insert new data
                        self._increase_char_length_if_necessary(key,subkey,new_val)
                        data[subkey][old_length:total_length] = self._get_attribute(key,subkey,'cast')(new_val)

    @optimise_method()
    def concatenate_and_optimise(self,new_dataset,keys=None,_cache=None):
        """Extend self by new_dataset using keys existing in self. New data updated
        on optimisaion if new_dataset changes."""
        if self._clean_construct and 'new_length' not in _cache:
            _cache['old_length'] = len(self)
            _cache['new_length'] = len(self) + len(new_dataset)
            self.concatenate(new_dataset,keys)
            self.permit_indexing = False
            _cache['keys'] = [*self.keys()]
            
        else:
            ## update data in place
            index = slice(_cache['old_length'],_cache['new_length'])
            for key in _cache['keys']:
                for subkey in ('value','unc'):
                    if self.is_set(key,subkey):
                        self.set(key,subkey,new_dataset[key,subkey],index,set_changed_only=True)
                if self.is_set(key,'vary'):
                    self.set(key,'vary',False,index)

    def append(self,keys_vals=None,**keys_vals_as_kwargs):
        """Append a single row of data from kwarg scalar values."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals |= keys_vals_as_kwargs
        for key in keys_vals:
            keys_vals[key] = [keys_vals[key]]
        self.extend(keys_vals)

    def extend(self,keys_vals=None,**keys_vals_as_kwargs):
        """Extend self with new data.  All keys or their defaults must be set
        in both new and old data."""
        ## get lists of new data
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
        ## new keys
        keys = list(keys_vals.keys())
        ## ensure new data includes all explicitly set keys in self
        for key in self.explicitly_set_keys():
            if key not in keys:
                if self.is_set(key,'default'):
                    keys_vals[key] = self[key,'default']
                    keys.append(key)
                else:
                    raise Exception(f'Extending data missing key: {repr(key)}')
        ## Ensure all new keys are existing data, unless the current
        ## Dataset is zero length, then add them.
        for key in keys:
            if not self.is_known(key):
                if len(self) == 0:
                    if key in self.prototypes:
                        kind = None
                    elif len(keys_vals[key]) > 0:
                        kind = array(keys_vals[key]).dtype.kind
                    else:
                        kind ='f'
                    self.set(key,'value',[],kind=kind)
                else:
                    raise Exception(f"Extending key not in existing data: {repr(key)}")
        ## determine length of new data
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

    def join(self,data):
        """Join keys from a new dataset set onto this one.  No key overlap allowed."""
        ## add data if it is a Dataset
        if isinstance(data,Dataset):
            ## error checks
            if len(self) != len(data):
                raise Exception(f'Length mismatch between self and new dataset: {len(self)} and {len(data)}')
            i,j = tools.common(self.keys(),data.keys())
            if len(i) > 0:
                raise Exception(f'Overlapping keys between self and new dataset: {repr(self.keys()[i])}')
            ## add from data
            for key in data:
                if key in self.prototypes:
                    self[key] = data[key]
                else:
                    if not self.permit_nonprototyped_data:
                        raise Exception(f'Key from new dataset is not prototyped in self: {repr(key)}')
                    self._data[key] = deepcopy(data._data[key])
        ## add data if it is a dictionary
        elif isinstance(data,dict):
            for key in data:
                if self.is_set(key):
                    raise Exception(f'Key already present: {key!r}')
                self[key] = data[key]

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
            xkeys=None,         # key to use for x-axis data
            ykeys=None,         # list of keys to use for y-axis data
            zkeys=None,         # plot x-y data separately for unique combinations of zkeys
            ykeys_re=None,
            fig=None,           # otherwise automatic
            ax=None,            # otherwise automatic
            xnewaxes=True,      # plot x-keys on separates axes -- else as different lines
            ynewaxes=True,      # plot y-keys on separates axes -- else as different lines
            znewaxes=False,     # plot z-keys on separates axes -- else as different lines
            legend=True,        # plot a legend or not
            legend_loc='best',
            annotate_lines=False, # annotate lines with their labels
            zlabel_format_function=None, # accept key=val pairs, defaults to printing them
            label_prefix='', # put this before label otherwise generated
            plot_errorbars=True, # if uncertainty available
            xlog=False,
            ylog=False,
            ncolumns=None,       # number of columsn of subplot -- None to automatically select
            show=False,          # show figure after issuing plot commands
            xlim=None,
            ylim=None,
            title=None,
            xsort=True,         # True sort by xkey, False, do not sort, or else a key or list of keys to sort by
            annotate_points_keys=None,
            **plot_kwargs,      # e.g. color, linestyle, label etc
    ):
        """Plot data."""
        from matplotlib import pyplot as plt
        from spectr import plotting
        if len(self)==0:
            return
        ## re-use or make a new figure/axes
        if ax is not None:
            xnewaxes = ynewaxes = znewaxes = False
            fig = ax.figure
        if fig is None:
            fig = plt.gcf()
            fig.clf()
        ## xkey, ykeys, zkeys
        xkeys = list(tools.ensure_iterable(xkeys))
        if ykeys is None:
            ykeys = []
        ykeys = list(tools.ensure_iterable(ykeys))
        if ykeys_re is not None:
            ykeys += [key for key in self if re.match(ykeys_re,key)]
        if zkeys is None:
            zkeys = self.default_zkeys
        zkeys = list(tools.ensure_iterable(zkeys))
        zkeys = [key for key in zkeys if key not in xkeys+ykeys and self.is_known(key)] # remove xkey and ykeys from zkeys
        for t in xkeys+ykeys+zkeys:
            self.assert_known(t)
        ## total number of subplots in figure
        nsubplots = 1
        if xnewaxes:
            nsubplots *= len(xkeys)
        if ynewaxes:
            nsubplots *= len(ykeys)
        if znewaxes:
            nsubplots *= len(zkeys)
        ## plot each xkey/ykey/zkey combination
        for ix,xkey in enumerate(xkeys):
            for iy,ykey in enumerate(ykeys):
                for iz,(dz,z) in enumerate(self.unique_dicts_matches(*zkeys)):
                    ## sort data
                    if xsort == True:
                        z.sort(xkey)
                    elif xsort == False:
                        pass
                    else:
                        z.sort(xsort)
                    ## get axes
                    if ax is None:
                        isubplot = 0
                        subplot_multiplier = 1
                        if xnewaxes:
                            isubplot += subplot_multiplier*ix
                            subplot_multiplier *= len(xkeys)
                        if ynewaxes:
                            isubplot += subplot_multiplier*iy
                            subplot_multiplier *= len(ykeys)
                        if znewaxes:
                            isubplot += subplot_multiplier*ix
                            subplot_multiplier *= len(zkeys)
                        tax = plotting.subplot(n=isubplot,fig=fig,ncolumns=ncolumns,ntotal=nsubplots)
                    else:
                        tax = ax 
                    ## get axis labels and perhaps convert them to legend labesl
                    label = ''
                    ## x-axis
                    xlabel = xkey
                    if self.is_known(xkey,'units'):
                        xlabel += ' ('+self[xkey,'units']+')'
                    if not xnewaxes:
                        label += f' {xlabel}'
                        xlabel = None
                    ## y-axis
                    ylabel = ykey
                    if self.is_known(ykey,'units'):
                        ylabel += ' ('+self[ykey,'units']+')'
                    if not ynewaxes:
                        label += f' {ylabel}'
                        ylabel = None
                    ## z-axis
                    if zlabel_format_function is None:
                        # zlabel_format_function = self.default_zlabel_format_function
                        zlabel_format_function = tools.dict_to_kwargs
                    zlabel = zlabel_format_function(dz)
                    if not znewaxes:
                        label += f' {zlabel}'
                        zlabel = None
                    ## get color/marker/linestyle
                    if xnewaxes and ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(0),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif not xnewaxes and ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(ix),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif xnewaxes and not ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif xnewaxes and ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(0),plotting.newlinestyle(0)
                    elif not xnewaxes and not ynewaxes and znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(ix),plotting.newlinestyle(iy)
                    elif not xnewaxes and ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(ix),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    elif xnewaxes and not ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    elif not xnewaxes and not ynewaxes and not znewaxes:
                        color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(ix),plotting.newlinestyle(iz)
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
                    if plot_errorbars and (z.is_set(ykey,'unc') or z.is_set(xkey,'unc')):
                        ## get uncertainties if they are known
                        if z.is_set(xkey,'unc'):
                            dx = z.get(xkey,'unc')
                            dx[np.isnan(dx)] = 0.
                        else:
                            dx = np.full(len(z),0.)
                        if z.is_set(ykey,'unc'):
                            dy = z.get(ykey,'unc')
                            dy[np.isnan(dy)] = 0.
                        else:
                            dy = np.full(len(z),0.)
                        ## plot errorbars
                        kwargs.setdefault('mfc','none')
                        i = ~np.isnan(x) & ~np.isnan(y)
                        tax.errorbar(x[i],y[i],dy[i],dx[i],**kwargs)
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
                            line = tax.plot(x[i],z[ykey][i],**kwargs)
                    else:
                        kwargs.setdefault('mfc',kwargs['color'])
                        kwargs.setdefault('fillstyle','full')
                        line = tax.plot(x,y,**kwargs)
                    if annotate_points_keys is not None:
                        ## annotate all points with the value of this key
                        # for li,xi,yi in zip(z[annotate_points_keys],x,y): 
                        #     if ~np.isnan(xi) and ~np.isnan(yi):
                        #         plt.annotate(format(li),(xi,yi),fontsize='x-small',in_layout=False)
                        for i in range(len(z)):
                            if ~np.isnan(x[i]) and ~np.isnan(y[i]):
                                annotation = '\n'.join([tkey+'='+format(z[tkey][i],self[tkey,'fmt']) for tkey in tools.ensure_iterable(annotate_points_keys)])
                                # annotation = pformat({tkey:format(z[tkey][i]) for tkey in tools.ensure_iterable(annotate_points_keys)})
                                plt.annotate(annotation,(x[i],y[i]),fontsize='x-small',in_layout=False)
                    if title is not None:
                        tax.set_title(title)
                    # elif auto_title is not None:
                        # tax.set_title(auto_title)
                    if ylabel is not None:
                        tax.set_ylabel(ylabel)
                    if xlabel is not None:
                        tax.set_xlabel(xlabel)
                    if 'label' in kwargs:
                        if legend:
                            plotting.legend(fontsize='small',loc=legend_loc,show_style=True,ax=tax)
                        if annotate_lines:
                            plotting.annotate_line(line=line)
                    if xlim is not None:
                        tax.set_xlim(*xlim)
                    if xlog:
                        tax.set_xscale('log')
                    if ylog:
                        tax.set_yscale('log')
                    tax.grid(True,color='gray',zorder=-5)
                    if self[xkey,'kind'] == 'U':
                        plotting.set_tick_labels_text(
                            xkey_unique_strings,
                            axis='x',
                            ax=tax,
                            rotation=70,
                            fontsize='x-small',
                            ha='right',
                        )
                ## set ylim for all axes
                if ylim is not None:
                    for tax in fig.axes:
                        if ylim == 'data':
                            t,t,ybeg,yend = plotting.get_data_range(tax)
                            tax.set_ylim(ybeg,yend)
                        elif tools.isiterable(ylim) and len(ylim) == 2:
                            ybeg,yend = ylim
                            if ybeg is not None:
                                if ybeg == 'data':
                                    t,t,ybeg,t = plotting.get_data_range(tax)
                                tax.set_ylim(ymin=ybeg)
                            if yend is not None:
                                if yend == 'data':
                                    t,t,t,yend = plotting.get_data_range(tax)
                                tax.set_ylim(ymax=yend)
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


    def __ior__(self,other_dict_like):
        """In place addition of key or substitution like a dictionary using |=."""
        for key in other_dict_like.keys():
            self[key] = other_dict_like[key]
        return self

    def __iadd__(self,other_dataset):
        """Concatenate self with another dataset using +=."""
        self.concatenate(other_dataset)
        return self

    ## other possible in place operators: __iadd__ __isub__ __imul__
    ##  __imatmul__ __itruediv__ __ifloordiv__ __imod__ __ipow__
    ##  __ilshift__ __irshift__ __iand__ __ixor__ __ior__
    


def find_common(x,y,keys=None,verbose=False):
    """Return indices of two Datasets that have uniquely matching
    combinations of keys."""
    keys = tools.ensure_iterable(keys)
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
    ## hack -- old classnames
    if classname == 'levels.LinearDiatomic':
        classname = 'levels.Diatomic'
    if classname == 'lines.LinearDiatomic':
        classname = 'lines.Diatomic'
    ## end of hack
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
        elif module == 'spectrum':
            from . import spectrum
            return getattr(spectrum,subclass)
        elif module == 'atmosphere':
            from . import atmosphere
            return getattr(atmosphere,subclass)
    raise Exception(f'Could not find a class matching {classname=}')
    
def make(classname='dataset.Dataset',*init_args,**init_kwargs):
    """Make an instance of the this classname."""
    class_object = _get_class(classname)
    dataset = class_object(*init_args,**init_kwargs)
    return dataset

def load(
        filename,
        classname=None,
        prototypes=None,
        permit_nonprototyped_data=None,
        name=None,
        **load_kwargs):
    """Load a Dataset.  Attempts to automatically find the correct
    subclass if it is not provided as an argument, but this requires
    loading the file twice."""
    ## get classname
    if classname is None:
        d = Dataset()
        d.load(filename,load_classname_only=True,**load_kwargs)
        classname = d.classname
    ## make Dataset
    init_kwargs = {}
    if prototypes is not None:
        init_kwargs['prototypes'] = prototypes
    if permit_nonprototyped_data is not None:
        init_kwargs['permit_nonprototyped_data'] = permit_nonprototyped_data
    if name is not None:
        init_kwargs['name'] = name
    retval = make(classname,**init_kwargs)
    retval.load(filename,**load_kwargs)
    return retval

def copy_from(dataset,*args,**kwargs):
    """Make a copy of dataset with additional initialisation args and
    kwargs."""
    classname = dataset.classname # use the same class as dataset
    retval = make(classname,*args,copy_from=dataset,**kwargs)
    return retval


