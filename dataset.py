import re
from copy import copy,deepcopy

import numpy as np
from numpy import nan

from spectr import tools

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
            # if isinstance(value,np.ndarray):
                # self.kind = value.dtype.kind
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
        self._inferences = []
        self._inferred_from = []

    is_scalar = property(lambda self:self._length is None)

    def _set_value(self,value):
        if value is None:
            self._value = None
            self._length = None
        elif np.ndim(value)==0:
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
            if self.is_scalar:
                self._uncertainty = float(uncertainty)
            else:
                if np.ndim(uncertainty)==0:
                    self._uncertainty = np.full(len(self),uncertainty,dtype=float)
                else:
                    assert len(uncertainty)==len(self)
                    self._uncertainty = np.asarray(uncertainty,dtype=float)

    def _get_value(self):
        if self.is_scalar:
            return(self._value)
        else:
            return(self._value[:self._length])

    def _get_uncertainty(self):
        if self._uncertainty is None:
            return(self._uncertainty)
        elif self.is_scalar:
            return(self._uncertainty)
        else:
            return(self._uncertainty[:self._length])

    value = property(_get_value,_set_value)
    uncertainty = property(_get_uncertainty,_set_uncertainty)

    def __len__(self):
        assert not self.is_scalar, 'Scalar'
        return(self._length)

    def index(self,index):
        """Set self to index"""
        assert not self.is_scalar, 'Cannot index because it is scalar'
        value = self.value[index]
        if self.uncertainty is None:
            uncertainty  = None
        else:
            uncertainty = self.uncertainty[index]
        self.value,self.uncertainty = value,uncertainty

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
        ## If the key exists it is being reset. Then clean up the record of what
        ## is inferred from it, and what it is inferred from.
        if key in self._data:
            for inferred_from_key in self._data[key]._inferred_from:
                self._data[inferred_from_key]._inferences.remove(key)
            self._data[key]._inferred_from = []
            for inferred_key in self._data[key]._inferences:
                self._data.pop(inferred_key)
            self._data[key]._inferences = []
        ## set the data
        self._data[key] = Data(name=key,value=value,uncertainty=uncertainty)
        if not self._data[key].is_scalar:
            if self._length == None:
                ## first array data, use this to define the length of self
                self._length = len(self._data[key])
            else:
                assert len(self._data[key])==len(self),'Length does not match existing data.'

    def get_value(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].value)

    def get_uncertainty(self,key):
        if key not in self._data:
            self._infer(key)
        return(self._data[key].uncertainty)

    def has_uncertainty(self,key):
        if key not in self._data:
            self._infer(key)
        if self._data[key].uncertainty is None:
            return(False)
        else:
            return( True)

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
                self._data[key]._inferred_from = dependencies
                for dependency in dependencies:
                    self._data[dependency]._inferences.append(key)
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
            self._data[key].index(i)

    def format(self,keys=None,comment='# ',delimiter=' '):
        if keys is None:
            keys = self.keys()
        header,columns = [],{}
        for key in keys:
            if self._data[key].is_scalar:
                if self.has_uncertainty(key):
                    header.append(f'# {key} = {repr(self.get_value(key))} ± {repr(self.get_uncertainty(key))}')
                else:
                    header.append(f'# {key} = {repr(self.get_value(key))}')
            else:
                columns[key]  = self.get_value(key)
                if self.get_uncertainty(key) is not None:
                    columns['d'+key]  = self.get_uncertainty(key)
        retval = '\n'.join(header)
        if len(columns)>0:
            retval += '\n'+tools.format_columns(columns,delimiter=delimiter)
        return(retval)

    def __str__(self):
        return(self.format(self.keys()))


    def save(self,filename,keys=None,**format_kwargs,):
        """Save some or all data to a text file."""
        if keys is None:
            keys = self.keys()
        if re.match(r'.*\.npz',filename):
            ## numpy archive
            np.savez(
                filename,
                **{key:self[key] for key in keys},
                **{'d'+key:self.get_uncertainty(key) for key in keys if self.has_uncertainty(key)})
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            d = {key:self[key] for key in keys}
            d.update({'d'+key:self.get_uncertainty(key) for key in keys if self.has_uncertainty(key)})
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
        # # filename = tools.expand_path(filename)
        # ## text file
        # data = tools.file_to_dict(filename,**file_to_recarray_kwargs)
        ## load common data in file header if a text file
        if re.match(r'.*\.(h5|hdf5)',filename):
            ## hdf5 archive
            for key,val in tools.hdf5_to_dict(filename).items():
                self[key] = val
        elif re.match(r'.*\.npz',filename):
            ## numpy npz archive
            for key,val in np.load(filename).items():
                if val.ndim == 0:
                    val = val.item() # get as scalar rather than zero-dimensional numpy array
                self[key] = val
        else:
            ## text file
            if re.match(r'.*\.csv',filename):
                delimiter = ','
            elif re.match(r'.*\.rs',filename):
                delimiter = '␞'
            ## load header
            assert comment not in ['',' '], "Not implemented"
            filename = tools.expand_path(filename)
            with open(filename,'r') as fid:
                for line in fid:
                    ## test for end of header
                    if not re.match(r'^ *'+comment,line):
                        break
                    if r := re.match(r'^ *'+comment+f' *([^= ]+) *=(.+)',line): # looking to match:  "# variable = value : description"
                        key = r.group(1)
                        value = tools.string_to_number_if_possible(r.group(2))
                        self[key] = value

                d = tools.txt_to_dict(filename, delimiter=delimiter, comment_regexp=comment,)
                for key,val in d.items():
                    self[key] = val
                # if r: # key,val,description = r.group(1).strip(),r.group(2).strip(),r.group(3).strip()
                        # ## check if given as a quoted string or not
                        # if (len(val)>=2
                            # and ((val[0]=='"' and val[-1]=='"')
                                 # or (val[0]=="'" and val[-1]=="'"))):
                            # val_quoted = True
                            # val = val[1:-1].strip() # remove quotes and excess white space
                        # else:
                            # val_quoted = False
                        # ## HACK TO DEAL WITH CHANIGN DATATYPE NAMES
                        # if hasattr(self,'_load_from_file_with_translation_dict'): # HACK
                            # if key in self._load_from_file_with_translation_dict:
                                # key = self._load_from_file_with_translation_dict[key]
                        # ## set if scalar data
                        # if key in self.scalar_data:
                            # if self.scalar_data[key].dtype.kind != 'O':
                                # data.setdefault(key,self.scalar_data[key].cast(val))
                            # else:
                                # ## THIS IS THE PLACE TO LOAD SCALAR DATA OBJECTS, NOT IMPLEMENTED
                                # pass
                        # ## default value for vector data
                        # elif key in self.vector_data:
                            # if self.vector_data[key].dtype.kind != 'O':
                                # data.setdefault(key,self.vector_data[key].cast(val))
                        # ## else unknown raise error
                        # elif not self.permit_new_keys:
                            # warnings.warn("Unknown header key: "+repr(key)+' in file '+repr(filename))
                        # ## or add with inferred type
                        # else:
                            # if val_quoted:
                                # data[key] = val
                            # else:
                                # data[key] = my.string_to_number(val) # convert to numerical value
    # ## add data to self
    # if len(data)==0: return # no data
    # ## translate input keys if provided as an argument, or if
    # ## default values reside in
    # t = copy(self._load_from_file_with_translation_dict)
    # if label_translation_dict is not None: t.update(label_translation_dict)
    # for key in list(data.keys()):
        # if key in t:
            # data[t[key]] = data.pop(key)
    # ## leave out keys already known to be objects -- these cannot
    # ## be loaded from a file
    # for key in list(data.keys()):
        # if key in self.all_keys() and self.get_data_type(key).dtype.kind == 'O':
            # data.pop(key)
    # # ## add data
    # # new_dynamic_recarray = self.__class__(**data,permit_new_keys=True)
    # # if keys is not None:
        # # new_dynamic_recarray.limit_to_keys(*keys)
    # # if limit_to_matches is not None:
        # # new_dynamic_recarray.limit_to_matches(**limit_to_matches)
    # # ## add new data -- this is quiet slow but provides a means for loading from multiple files
    # # self.extend(new_dynamic_recarray,permit_new_keys=permit_new_keys)
    # self.append(**data,error_on_mismatching_scalar_data=error_on_mismatching_scalar_data)


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
            if self.has_uncertainty(key) is not new_dataset.has_uncertainty(key):
                raise Exception(f"Uncertainty must be both set or not set in existing and concatenating dataset for key: {repr(key)}")
            if (self.is_scalar(key) and new_dataset.is_scalar(key) # both scalar
                and self.get_value(key) == new_dataset.get_value(key)                  # values match
                and tools.equal_or_none(
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

        


if __name__=='__main__':
    t = Dataset()
    t.set('a','So be it.')
    t.set('b',1.32e25)
    t.set('x',[1,2,3])
    t.set('y',[1,2,3],[0.1,0.2,0.3])
    t.set('z',[2,4,5])
    print( t.format(delimiter=' ␞ '))
    t.save_to_file('t0')
    t.load_from_file('t0')
    # t.plot('x',('y','z'))
   #  
