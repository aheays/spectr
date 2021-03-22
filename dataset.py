import re
from time import perf_counter as timestamp
import ast
from copy import copy,deepcopy
from pprint import pprint
import importlib
import warnings

import numpy as np
from numpy import nan

from . import tools
from .tools import AutoDict
from .exceptions import InferException
from . import convert
from . import optimise
from .optimise import optimise_method,Parameter,Fixed



class Dataset(optimise.Optimiser):

    """A set of data."""


    ## perhaps better as instance variable?
    _kind_defaults = {
        'f': {'cast':lambda x:np.asarray(x,dtype=float) ,'fmt'   :'+12.8e','description':'float' ,'default':nan,},
        'i': {'cast':lambda x:np.asarray(x,dtype=int)   ,'fmt'   :'d'     ,'description':'int'   ,'default':-999,},
        'b': {'cast':lambda x:np.asarray(x,dtype=bool)  ,'fmt'   :''      ,'description':'bool'  ,'default':True,},
        'U': {'cast':lambda x:np.asarray(x,dtype=str)   ,'fmt'   :'s'     ,'description':'str'   ,'default':'',},
        'O': {'cast':lambda x:np.asarray(x,dtype=object),'fmt'   :''      ,'description':'object','default':None,},
    }

    default_attributes = ('classname','description',)
    default_zkeys = []

    def __init__(
            self,
            name=None,
            permit_nonprototyped_data = True,
            # permit_reference_breaking = True,
            permit_auto_defaults = False,
            prototypes = None,  # a dictionary of prototypes
            load_from_file = None,
            load_from_string = None,
            copy_from = None,
            limit_to_match=None, # dict of things to match 
            **kwargs):
        ## deal with arguments
        self._data = dict()
        self.attributes = dict()
        self._length = 0
        self._over_allocate_factor = 2
        self._last_modify_data_time = timestamp()  # used for triggering optimise construct
        self.permit_nonprototyped_data = permit_nonprototyped_data # allow the addition of data not in self._prototypes
        # self.permit_reference_breaking = permit_reference_breaking  # not implemented -- I think
        self.permit_auto_defaults = permit_auto_defaults         # set default values if necessary automatically
        # self.permit_missing = True # add missing data if required
        self.verbose = False
        ## init prototypes
        self._prototypes = {}
        if prototypes is not None:
            for key,val in prototypes.items():
                self.set_prototype(key,**val)
        ## initialise attributes
        for key in self.default_attributes:
            self.attributes[key] = None
        ## classname to identify type of Dataset
        # self.classname = self.__class__.__name__
        self.attributes['classname'] = re.sub(r"<class 'spectr.(.+)'>",r'\1',str(self.__class__))
        ## default name is snake version of camel object name
        if name is None:
            # name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.classname.lower())
            name = re.sub(r'[<!^.]', '_', self.attributes['classname'].lower())
        ## init as optimiserg, make a custom form_input_function, save
        ## some extra stuff if output to directory
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        def format_input_function():
            retval = f'{self.name} = {self.attributes["classname"]}({self.name},'
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
        ## copy from another dataset
        if copy_from is not None:
            self.copy_from(copy_from)
        ## load data from a file
        if load_from_file is not None:
            self.load(load_from_file)
        ## load data from an encode tabular string
        if load_from_string is not None:
            self.load_from_string(load_from_string)
        ## kwargs to default values if scale, data values if vector,
        ## set_parameter if Paramters
        for key,val in kwargs.items():
            if key in self.attributes:
                self[key] = val
            elif isinstance(val,optimise.Parameter):
                self.set_parameter(key,val)
                self.pop_format_input_function() # input function customised above
            elif tools.isiterable(val):
                self[key] = val
            else:
                self.set_default(key,val)
        ## limit_to_match of data inserted above
        if limit_to_match is not None:
            self.limit_to_match(**limit_to_match)

    def __len__(self):
        return self._length

    def set(
            self,
            key,
            value,
            index=None,
            uncertainty=None,
            vary=None,
            step=None,
            _inferred=False,    # internal use -- mark if this is set call is due to inference
            **prototype_kwargs
    ):
        """Set a value"""
        if self.verbose:
            print(f'{self.name}: setting {key} inferred={_inferred}')
        ## update modification if externally set, not if it is inferred
        if not _inferred:
            self._last_modify_data_time = timestamp()
        ## delete inferences since data has changed
        if key in self:
            self.unset_inferences(key)
        ## set data differently depending on whether an index is
        ## provided
        if index is None:
            ## decide whether to permit if non-prototyped
            if not self.permit_nonprototyped_data and self.get_prototype(key) is None:
                raise Exception(f'New data is not in prototypes: {repr(key)}')
            ## new data
            data = dict()
            ## get any prototype data
            if key in self._prototypes:
                data.update(self._prototypes[key])
            ## apply prototype kwargs
            for tkey,tval in prototype_kwargs.items():
                data[tkey] = tval
            ## if a scalar value is then expand to full length, if
            ## vector then cast as a new array.  Do not use asarray
            ## but instead make a copy -- this will prevent mysterious
            ## bugs where assigned arrays feedback.
            if np.isscalar(value):
                value = np.full(len(self),value)
            else:
                value = np.array(value)
            ## infer kind
            if 'kind' not in data:
                ## use data to infer kind
                # value = np.asarray(value)
                data['kind'] = value.dtype.kind
            ## convert bytes string to unicode
            if data['kind'] == 'S':
                self.kind = 'U'
            ## some other prototype data
            for tkey in ('description','fmt','cast'):
                if tkey not in data:
                    data[tkey] = self._kind_defaults[data['kind']][tkey]
            if 'default' not in data and self.permit_auto_defaults:
                data['default'] = self._kind_defaults[data['kind']]['default']
            if data['kind']=='f' and 'default_step' not in data:
                data['default_step'] = 1e-8
            ## infer function dict,initialise inference lists, and units if needed
            for tkey,tval in (
                    ('infer',[]),
                    ('inferred_from',[]),
                    ('inferred_to',[]),
            ):
                if tkey not in data:
                    data[tkey] = tval
            ## set data
            data['value'] = data['cast'](value)
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
                raise Exception(f'Length of new data {repr(key)} is {len(data)} and does not match the length of existing data: {len(self)}.')
            ## set data
            self._data[key] = data
        else:
            if key not in self:
                raise Exception(f'Cannot set data with index for unknown {key=}')
            self._data[key]['value'][:self._length][index] = self._data[key]['cast'](value)
        ## set uncertainty / variability / differentiation stepsize
        if uncertainty is not None:
            self.set_uncertainty(key,uncertainty,index=index)
        if vary is not None:
            self.set_vary(key,vary,index=index)
        if step is not None:
            self.set_step(key,step,index=index)

    def get(self,key,index=None,units=None):
        """Get value."""
        if index is not None:
            return self.get(key)[index]
        if key not in self._data:
            self._infer(key)
        retval = self._data[key]['value'][:self._length]
        if units is not None:
            retval = convert.units(retval,self.get_units(key),units)
        return retval

    def has_uncertainty(self,key):
        return 'uncertainty' in self._data[key]
        
    def get_uncertainty(self,key,index=None,units=None):
        if 'uncertainty' not in self._data[key]:
            return None
        if index is not None:
            return self.get_uncertainty(key,units=units)[index]
        retval = self._data[key]['uncertainty'][:self._length]
        if units is not None:
            retval = convert.units(retval,self.get_units(key),units)
        return retval

    def set_uncertainty(self,key,value,index=None):
        if self._data[key]['kind'] != 'f':
            raise Exception('Uncertainty only defined for kind="f" floating-point data and {key=} has kind={self._data[key]["kind"]}')
        if value is None:
            ## unset uncertainty
            self._data[key]['uncertainty'] = None
        elif index is not None:
            ## set some uncertainties -- if not defined set others to
            ## NaN
            if self.get_uncertainty(key) is None:
                self.set_uncertainty(key,nan)
            self.get_uncertainty(key)[index] = value
        else:
            ## set all uncertainties
            if np.isscalar(value):
                self._data[key]['uncertainty'] = np.full(len(self),value,dtype=float)
            else:
                if len(value) != len(self):
                    raise Exception()
                self._data[key]['uncertainty'] = np.asarray(value,dtype=float)

    def get_step(self,key,index=None,return_default_if_necessary=False):
        if 'step' not in self._data[key]:
            if return_default_if_necessary:
                return self._data[key]['default_step']
            else:
                return None
        if index is not None:
            return self.get_step(key)[index] 
        return self._data[key]['step'][:len(self)]

    def set_step(self,key,step,index=None):
        if index is not None:
            get_step(key)[index] = step
        if np.isscalar(step):
            self._data[key]['step'] = np.full(len(self),step,dtype=float)
        else:
            if len(step) != len(self):
                raise Exception()
            self._data[key]['step'] = np.asarray(step,dtype=float)

    def get_vary(self,key,index=None):
        if 'vary' not in self._data[key]:
            return None
        if index is not None:
            return self.get_vary(key)[index] 
        return self._data[key]['vary'][:len(self)]

    def set_vary(self,key,vary,index=None):
        """Set boolean vary for optimisation. Strings 'True', 'False', and
        'None' also accepted."""
        ## interpret string represnation  of "True" and "False"
        if vary is True:
            vary = True
        elif vary is False or vary is None or vary is Fixed: 
            vary = False
        elif len(vary) > 0 and isinstance(vary[0],str):
            tvary = vary
            vary = []
            for val in tvary:
                if val in ('True',):
                    val = True
                if val in ('False','None'): 
                    val = False
                vary.append(val)
        ## set if indexed values
        if index is not None:
            if self.get_vary(key) is None:
                self.set_vary(key,False)
            self.get_vary(key)[index] = vary
            return
        ## set if scalar
        if np.isscalar(vary):
            self._data[key]['vary'] = np.full(len(self),vary,dtype=bool)
        ## set if vector
        else:
            if len(vary) != len(self):
                raise Exception()
            self._data[key]['vary'] = np.asarray(vary,dtype=bool)
        ## set a default step size and uncertainty
        if self.get_step(key) is None:
            if (prototype:=self.get_prototype(key)) is not None and 'default_step' in prototype:
                self.set_step(key,prototype['default_step'])
            else:
                step = np.full(len(self),1e-5)
                i = self[key] != 0
                step[i] = 10**np.round(np.log10(1e-5*np.abs(self[key][i])))
                self.set_step(key,step)
        if self.get_uncertainty(key) is None:
            self.set_uncertainty(key,nan)

    def get_units(self,key):
        if 'units' not in self._data[key]:
            return None
        return self._data[key]['units']

    def get_description(self,key):
        return self._data[key]['description']

    def set_default(self,key=None,value=None,**more_keys_values):
        """Set a value that will be used if otherwise missing"""
        if key is not None:
            more_keys_values[key] = value
        for key,value in more_keys_values.items():
            if not self.is_known(key):
                self.set(key,value)
            self._data[key]['default'] = value

    def get_default(self,key):
        """Set a value that will be used if otherwise missing"""
        if 'default' not in self._data[key]:
            return None
        return self._data[key]['default']

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
        self._prototypes[key] = dict(kind=kind,infer=infer,**kwargs)
        for tkey,tval in self._kind_defaults[kind].items():
            self._prototypes[key].setdefault(tkey,tval)

    def get_prototype(self,key):
        """Get prototype as a dictionary."""
        if key not in self._prototypes:
            return None
        return self._prototypes[key]

    def get_kind(self,key):
        return self._data[key]['kind']

    @optimise_method(format_single_line=True)
    def set_parameter(
            self,
            key,
            parameter,          # a scalar or Parameter
            index=None,         # only apply to these indices
            **prototype_kwargs,
    ):
        """Set a value to be optimised."""
        ## if not a parameter then treat as a float -- could use set(
        ## instead and branch there, requiring a Parameter here
        if isinstance(parameter,Parameter):
            ## only reconstruct for the following reasons
            if (
                    key not in self.keys() # key is unknown -- first run
                    or parameter._last_modify_value_time > self._last_construct_time # parameter has been set
                    or np.any(self.get(key,index=index) != parameter.value) # data has changed some other way and differs from parameter
                    or ((not np.isnan(parameter.uncertainty)) and (np.any(self.get_uncertainty(key,index=index) != parameter.uncertainty))) # data has changed some other way and differs from parameter
                ):
                self.set(
                    key,
                    value=parameter.value,
                    uncertainty=parameter.uncertainty,
                    step=parameter.step,
                    index=index,
                    **prototype_kwargs)
        else:
            ## only reconstruct for the following reasons
            if (
                    key not in self.keys() # key is unknown -- first run
                    or np.any(self.get(key,index=index) != parameter) # data has changed some other way and differs from parameter
                ):
                self.set(key,value=parameter,index=index,**prototype_kwargs)

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
        if self.get_uncertainty(ykey) is not None:
            if i is None:
                self.set_uncertainty(ykey,None)
            else:
                self.set_uncertainty(ykey,nan,index=i)

    def keys(self):
        return list(self._data.keys())

    def items(self):
        """Iterate over set keys and their values."""
        for key in self:
            yield key,self[key]

    def pop(self,key):
        """Pop data in key."""
        value = self[key]
        self.unset(key)
        return value

    def optimised_keys(self):
        return [key for key in self.keys() if self.get_vary(key) is not None]

    def assert_known(self,*keys):
        for key in keys:
            self[key]

    def is_known(self,*keys):
        try:
            self.assert_known(*keys)
            return True 
        except InferException:
            return False

    def __getitem__(self,arg):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            if len(arg) > 4 and arg[-4:] == '_unc':
                return self.get_uncertainty(arg[:-4])
            elif len(arg) > 5 and arg[-5:] == '_vary':
                return self.get_vary(arg[:-5])
            elif len(arg) > 5 and arg[-5:] == '_step':
                return self.get_step(arg[:-5])
            elif arg in self.attributes:
                return self.attributes[arg]
            else:
                return self.get(arg)
        elif tools.isiterable(arg) and len(arg)>0 and isinstance(arg[0],str):
            return self.copy(keys=arg)
        else:
            return self.copy(index=arg)

    def __setitem__(self,key,value):
        """Set a key to value. If key_unc then set uncertainty. If key_vary or
        key_step then set optimisation parameters"""
        if len(key) > 4 and key[-4:] == '_unc':
            self.set_uncertainty(key[:-4],value)
        elif len(key) > 5 and key[-5:] == '_vary':
            self.set_vary(key[:-5],value)
        elif len(key) > 5 and key[-5:] == '_step':
            self.set_step(key[:-5],value)
        elif isinstance(value,optimise.P):
            self.set_parameter(key,value)
        elif key in self.attributes:
            self.attributes[key] = value
        else:
            self.set(key,value)
        
    def clear(self):
        """Clear all data"""
        self._last_modify_value_time = timestamp()
        self._length = 0
        self._data.clear()

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unset_inferences(key)
        self._data.pop(key)

    def unset_inferences(self,*keys):
        """Delete any record of inferences to or from the given
        keys. If no keys given then delete all inferred data."""
        if len(keys) == 0:
            ## default to uninfer everything inferred
            keys = list(self.keys()) 
        for key in keys:
            if key in self:     # test this since key might have been unset earlier in this loop
                for inferred_from in list(self._data[key]['inferred_from']):
                    self._data[inferred_from]['inferred_to'].remove(key)
                    self._data[key]['inferred_from'].remove(inferred_from)
                for inferred_to in list(self._data[key]['inferred_to']):
                    self._data[inferred_to]['inferred_from'].remove(key)
                    self._data[key]['inferred_to'].remove(inferred_to)
                    self.unset(inferred_to)

    def is_inferred_from(self,key_to,key_from):
        """Check whether data is calculated from other dats."""
        return key_from in self._data[key_to]['inferred_from']

    def add_infer_function(self,key,dependencies,function):
        """Add a new method of data inference."""
        assert key in self._prototypes
        self._prototypes[key]['infer'].append((dependencies,function))

    def index(self,index):
        """Index all array data in place."""
        original_length = len(self)
        for data in self._data.values():
            ## index value
            data['value'] = data['value'][:original_length][index]
            ## index any ancillary data
            for key in 'uncertainty','vary','step':
                if key in data and data[key] is not None:
                    data[key] = data[key][:original_length][index]
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
            copy_uncertainty= True, #
            copy_step=False,
            copy_vary=False,
    ):
        """Copy all values and uncertainties from source Dataset and update if
        source changes during optimisation."""
        self.clear()            # total data reset
        if keys is None:
            keys = source.keys()
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
            self.set(key,source.get(key,index=index))
            if copy_uncertainty and (t:=source.get_uncertainty(key,index=index)) is not None:
                self.set_uncertainty(key,t)
            if copy_step and (t:=source.get_step(key,index=index)) is not None:
                self.set_step(key,t)
            if copy_vary and (t:=source.get_vary(key,index=index)) is not None:
                self.set_vary(key,t)
        ## copy all attributes
        for key in source.attributes:
            self[key] = source[key]

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
        if self.get_kind(key) == 'O':
            return self[key]
        else:
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
        """Return value of key from a row that uniquely matches
        matching_keys_vals."""
        i = tools.find(self.match(**matching_keys_vals))
        if len(i)!=1:
            raise Excecption(f'Non-unique ({len(i)}) matches for {matching_keys_vals=}')
        return self.get_value(key,i)

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
        if key not in self._prototypes:
            raise InferException(f"No prototype for {key=}")
        ## loop through possible methods of inferences.
        for dependencies,function in self._prototypes[key]['infer']:
            if isinstance(dependencies,str):
                ## sometimes dependencies end up as a string instead of a list of strings
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
                squared_contribution = []
                value = self[key]
                parameters = [self[t] for t in dependencies]
                for i,dependency in enumerate(dependencies):
                    if (tuncertainty:=self.get_uncertainty(dependency)) is not None:
                        step = self.get_step(dependency,return_default_if_necessary=True)
                        parameters[i] = self[dependency] + step # shift one
                        dvalue = value - function(self,*parameters)
                        parameters[i] = self[dependency] # put it back
                        data = self._data[dependency]
                        squared_contribution.append((tuncertainty*dvalue/step)**2)
                if len(squared_contribution)>0:
                    self.set_uncertainty(key,np.sqrt(np.sum(squared_contribution,axis=0)))
                ## if we get this far without an InferException then
                ## success!.  Record inference dependencies.
                self._data[key]['inferred_from'].extend(dependencies)
                for dependency in dependencies:
                    self._data[dependency]['inferred_to'].append(key)
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

    def _get_value_key_without_prefix(self,key):
        """Get value key from uncertainty key, or return None."""
        if '_' in key:
            return key.split('_',1)
        else:
            return None,None
        
    def __iter__(self):
        for key in self._data:
            yield key

    def as_dict(self,keys=None,index=None,):
        """Return as a dict of arrays."""
        if keys is None:
            keys = self.keys()
        retval = {}
        for key in keys:
            retval[key] = self.get(key,index=index)
            if self.has_uncertainty(key):
                retval[key+'_unc'] = self.get_uncertainty(key,index=index)
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

    def matching_row(self,return_index=False,**matching_keys_vals):
        """Return uniquely-matching row as a dictionary."""
        i = tools.find(self.match(**matching_keys_vals))
        if len(i)==0:
            raise Exception(f'No matching row found: {matching_keys_vals=}')
        if len(i)>1:
            raise Exception(f'Multiple matching rows found: {matching_keys_vals=}')
        d = self.as_dict(index=i[0])
        if return_index:
            return d,i
        else:
            return d

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
            format_uncertainty=True,
            format_vary=True,
            format_step= True,
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
                and ( (not format_uncertainty) or self.get_uncertainty(key) is None)
                and ( (not format_vary) or self.get_vary(key) is None)
                and ( (not format_step) or self.get_step(key) is None)
                ):
                ## format value for header
                # header_values[key] = format(tval[0],self._data[key]['fmt'])
                header_values[key] = tval[0]
            else:
                ## two passes required on all data to align column
                ## widths
                vals = [format(t,self._data[key]['fmt']) for t in self.get(key)]
                if quote_strings and self._data[key]['kind'] == 'U':
                    vals = ["'"+val+"'" for val in vals]
                width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
                ## add uncertinaties / vary /step
                if format_uncertainty and self.get_uncertainty(key) is not None:
                    formatted_key = ( "'"+key+"_unc'" if quote_keys else key+"_unc" )
                    vals = [format(t,"0.1e") for t in self.get_uncertainty(key)]
                    width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                    columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
                if format_vary and self.get_vary(key) is not None:
                    formatted_key = ( "'"+key+"_vary'" if quote_keys else key+"_vary" )
                    vals = [repr(t) for t in self.get_vary(key)]
                    width = str(max(len(formatted_key),np.max([len(t) for t in vals])))
                    columns.append([format(formatted_key,width)]+[format(t,width) for t in vals])
                if format_step and self.get_step(key) is not None:
                    formatted_key = ( "'"+key+"_step'" if quote_keys else key+"_step" )
                    vals = [format(t,"0.1e") for t in self.get_step(key)]
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
                line += f' # {self.get_description(key)}'
                if (units:=self.get_units(key)) is not None:
                    line += f' [{units}]'
                header.append(line)
        else:
            for key,val in header_values.items():
                header.append(f'{key:12} = {repr(val)}')
        ## make full formatted string
        retval = ''
        if header != []:
            retval = '\n'.join(header)+'\n'
        if columns != []:
            retval += '\n'.join([delimiter.join(t) for t in zip(*columns)])+'\n'
        return retval

    def format_as_list(self):
        """Form as a valid python list of lists."""
        retval = f'[ \n'
        data = self.format(
            delimiter=' , ',
            format_uncertainty=True,
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
            format_uncertainty=True,
            format_vary=True,
            format_step=True,
            unique_values_in_header=True,
            include_description=False,
        )

    def save(self,filename,keys=None,**format_kwargs,):
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
            tools.dict_to_hdf5(
                filename,
                self.as_dict(),
                attributes={key:val for key,val in self.attributes.items() if val is not None})
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
            delimiter=None,
            table_name=None,
            translate_keys=None, # from key in file to key in self, None for skip
            return_classname_only=False, # do not load the file -- just try and load the classname and return it
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
            # assert comment not in ['',' '], "Not implemented"
            filename = tools.expand_path(filename)
            data = {}
            ## load header
            blank_line_re = re.compile(r'^ *$')
            description_line_re = re.compile(r'^ *'+comment+f' *([^# ]+) *# *(.+) *') # no value in line
            unique_value_line_re = re.compile(r'^ *'+comment+f' *([^= ]+) *= *([^#]*[^ #\n])') # may also contain description
            with open(filename,'r') as fid:
                for iline,line in enumerate(fid):
                    if re.match(blank_line_re,line):
                        continue
                    if r:=re.match(description_line_re,line):
                        pass
                    elif r:=re.match(unique_value_line_re,line):
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
        for key,val in data.items():
            if key in self.attributes:
                self.attributes[key] = val
            elif not tools.isiterable(val):
                self.set_default(key,val)
            else:
                self[key] = val

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
                                  

        # ## add data from name to common_keys_vals if it is provided
        # keys_vals = collections.OrderedDict() # will contain line data
        # parameters = []                       # save optimisation parameters in a list
        # value_lists = [list(t) for t in value_lists]       # make mutable
        # for ivals,vals in enumerate(value_lists):              # loop through all lines
            # for ikey,(key,val) in enumerate(zip(keys,vals)):
                # if key not in keys_vals: keys_vals[key] = [] # add to keys if first
                # if np.isscalar(val):                         
                    # keys_vals[key].append(val) # add data to append
                # else:
                    # p = self.add_parameter(key,*val) # add to optimiser
                    # vals[ikey] = p
                    # parameters.append((ivals+len(self),p)) # record which line in self will be after appending data
                    # keys_vals[key].append(p.p)   # add data to append
        # ## append lines to self
        # self.append(**keys_vals,**common_keys_vals)
        # ## add optimisation function
        # def f():
            # for i,p in parameters:
                # ## update parameters and uncertainty
                # self[p.name][i] = self.vector_data[p.name].cast(p.p) 
                # if not self.is_set('d'+p.name):
                    # self['d'+p.name] = np.nan
                # self['d'+p.name][i] = self.vector_data['d'+p.name].cast(p.dp) 
                # self.unset_inferences(p.name)
        # self.add_construct(f)
        # ## add format input function
        # def f():
            # retval = f'{self.name}.add_lines(\n    {repr(keys)},'
            # retval += '\n    '+",\n    ".join([repr(vals) for vals in value_lists])+",\n    "
            # if len(common_keys_vals)>0:
                # retval += my.dict_to_kwargs(common_keys_vals)+','
            # retval +=  ')'
            # return(retval)
        # self.format_input_functions.append(f)


    def append(self,**kwargs):
        """Append a single row of data from kwarg scalar values."""
        self.extend(**{key:[val] for key,val in kwargs.items()})

    def extend(self,**kwargs):
        """Extend self with data given as kwargs."""
        keys = np.unique(self.keys() + list(kwargs.keys()))
        ## get data lengths 
        original_length = len(self)
        extending_length = 0
        for val in kwargs.values():
            if not np.isscalar(val):
                extending_length = max(extending_length,len(val))
        total_length = original_length + extending_length
        for key in keys:
            ## make sure key is in _data
            # if not self.is_known(key):
            if key not in self:
                if original_length == 0:
                    self.set(key,[])
                elif (self.permit_auto_defaults
                      and ((prototype:=self.get_prototype(key)) is not None)):
                    self.set_default(key,prototype['default'])
                else:
                    raise Exception()
            ## get a default value for missing extended data
            if key not in kwargs:
                if (default:=self.get_default(key)) is not None:
                    kwargs[key] = default
                elif (self.permit_auto_defaults
                      and (prototype:=self.get_prototype(key)) is not None):
                    kwargs[key] = prototype['default']
                else:
                    raise Exception()
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
                    self._data[key]['value'] = t
            ## reallocate and lengthen value array if necessary
            if total_length > len(self._data[key]['value']):
                self._data[key]['value'] = np.concatenate((
                    self[key],
                    np.empty(
                        int(total_length*self._over_allocate_factor-original_length),
                        dtype=self._data[key]['value'].dtype)))
            ## set extending data
            self._data[key]['value'][original_length:total_length] = kwargs[key]
        self._length = total_length
        
    def __add__(self,other):
        """Adding dataset combines each key."""
        retval = self.copy()
        retval.extend(**other)
        return retval

    def __radd__(self,other):
        """Adding dataset combines each key."""
        retval = self.copy()
        retval.extend(**other)
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
                    if len(dz) == 1:
                        zlabel = str(dz[list(dz.keys())[0]])
                    else:
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
                    color,marker,linestyle = plotting.newcolor(iz),plotting.newmarker(iz),plotting.newlinestyle(iy)
                    # color,marker,linestyle = plotting.newcolor(iy),plotting.newmarker(iz),plotting.newlinestyle(iz)
                    label = ylabel+' '+zlabel
                    title = None
                if label_prefix is not None:
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
                if plot_errorbars and (dy:=z.get_uncertainty(ykey)) is not None:
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
                        ax.plot(z[xkey][i],z[ykey][i],**kwargs)
                else:
                    kwargs.setdefault('mfc',kwargs['color'])
                    kwargs.setdefault('fillstyle','full')
                    ax.plot(x,y,**kwargs)
                if title is not None:
                    ax.set_title(title)
                if legend and 'label' in kwargs:
                    plotting.legend(fontsize='x-small')
                ax.set_xlabel(xkey)
                ax.grid(True,color='gray',zorder=-5)
                ax.set_yscale(yscale)
                ax.set_xscale(xscale)
        if show:
            plotting.show()
        return(fig)

    def polyfit(self,xkey,ykey,index=None,**polyfit_kwargs):
        return tools.polyfit(
            self.get(xkey,index),
            self.get(ykey,index),
            self.get_uncertainty(ykey,index),
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

def load(filename,classname=None,**kwargs):
    """Load a Dataset.  Attempts to automatically find the correct
    subclass if it is not provided as an argument, but this requires
    loading the file twice."""
    if classname is None:
        d = Dataset()
        classname = d.load(filename,return_classname_only=True)
        if classname is None:
            classname = 'dataset.Dataset'
    retval = make(classname,load_from_file=filename,**kwargs)
    return retval

def copy_from(dataset,*args,**kwargs):
    """Make a copy of dataset with additional initialisation args and
    kwargs."""
    classname = dataset['classname'] # use the same class as dataset
    retval = make(classname,*args,copy_from=dataset,**kwargs)
    return retval

    

