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
            name='dataset',
            load_from_filename=None,
            **kwargs):
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        self.add_format_input_function(lambda: 'not implemented')
        self._data = dict()
        self._length = 0
        self._over_allocate_factor = 2
        if not hasattr(self,'prototypes'):
            ## derived classes might set this in class definition, so
            ## do not overwrite here
            self.prototypes = {}
        # self._inferences = AutoDict([])
        # self._inferred_from = AutoDict([])
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

    def __setitem__(self,key,value):
        """Set a value"""
        assert (self.permit_nonprototyped_data
                or key in self.prototypes
                or self._get_value_key_from_uncertainty(key) is not None # is an uncertainty
                ), f'New data is not in prototypes: {repr(key)}'
        ## delete inferences since data has changed
        if key in self:
            self.unset_inferences(key)
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
        ## uncertainty stuff
        if (tkey:=self._get_value_key_from_uncertainty(key)) is not None:
            data['kind'] = 'f'
            data['description'] = f'Uncertainty of {tkey}'
        ## infer kind
        if 'kind' not in data:
            value = np.asarray(value)
            data['kind'] = value.dtype.kind
        ## convert bytes string to unicode
        if data['kind'] == 'S':
            self.kind = 'U'
        ## some other data
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

    def get_unique_value(self,key,**matching_keys_vals):
        """Return value of key that is the uniquely matches
        matching_keys_vals."""
        i = tools.find(self.match(**matching_keys_vals))
        assert len(i)==1,f'Non-unique matches for {matching_keys_vals=}'
        return self.get_value(key,i)

    def add_prototype(self,key,**kwargs):
        self.prototypes[key] = dict(**kwargs)

    def add_infer_function(self,key,dependencies,function):
        if key not in self.prototypes:
            self.add_prototype(key,infer={})
        self.prototypes[key]['infer'][dependencies] = function

    def index(self,index):
        """Index all array data in place."""
        original_length = len(self)
        for data in self._data.values():
            data['value'] = data['value'][:original_length][index]
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

    def __getitem__(self,arg,index=None):
        """If string 'x' return value of 'x'. If "ux" return uncertainty
        of x. If list of strings return a copy of self restricted to
        that data. If an index, return an indexed copy of self."""
        if isinstance(arg,str):
            if arg not in self._data:
                self._infer(arg)
            if index is None:
                return self._data[arg]['value'][:self._length]
            else:
                return self._data[arg]['value'][:self._length][index]
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
                    if (tuncertainty:=self._get_uncertainty(dependency)) is not None:
                        diffstep = 1e-10*self[dependency]
                        parameters[i] = self[dependency] + diffstep # shift one
                        dvalue = value - function(self,*parameters)
                        parameters[i] = self[dependency] # put it back
                        data = self._data[dependency]
                        squared_contribution.append((tuncertainty*dvalue/diffstep)**2)
                if len(squared_contribution)>0:
                    self._set_uncertainty(key,np.sqrt(sum(squared_contribution)))
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

    def _get_value_key_from_uncertainty(self,key):
        """Get value key from uncertainty key, or return None."""
        if len(key) > 2 and key[:2] == 'd_':
            return key[2:]
            assert self.permit_nonprototyped_data or key[2:] in self.prototypes,f'Uncertain key with non-prototyped value key: {key=}'
        else:
            return None

    def _get_uncertainty(self,key):
        if 'd_'+key in self:
            return self['d_'+key]
        else:
            return None

    def _set_uncertainty(self,key,value):
        self['d_'+key] = value

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

    def format(self,keys=None,delimiter=' | ',automatically_add_uncertainties=True):
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
        header,columns = [],[]
        for key in keys:
            if len(tval:=self.unique(key)) == 1:
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
            np.savez(filename, **{key:self[key] for key in keys})
        elif re.match(r'.*\.h5',filename):
            ## hdf5 file
            d = {key:self[key] for key in keys}
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
                        if r:=re.match(r'"(.*)"|\'(.*)\'',val):
                            ## value is a string
                            val = r.group(1)
                        else:
                            ## else try cast as number
                            val = tools.string_to_number_if_possible(val)
                        data[key] = val
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
            if key in ('description',):
                setattr(self,key,val)
            else:
                self[key] = val

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
                y = z[ykey]
                if label is not None:
                    kwargs.setdefault('label',label)
                if plot_errorbars and (dkey:=self._get_uncertainty(ykey)) is not None:
                    ## plot errorbars
                    kwargs.setdefault('mfc','none')
                    dy = self[dkey]
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
                        value,uncertainty = self.get_value(key),self._get_uncertainty(key)
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

