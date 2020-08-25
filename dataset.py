import re
import time
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from numpy import nan

from spectr import tools
# from spectr.data import Data
# from spectr.datum import Datum
from spectr.tools import AutoDict
from spectr.exceptions import InferException



from . import optimise
from .datum import Datum
from .data import Data

class Dataset(optimise.Optimiser):

    """A collection of scalar or array values, possibly with uncertainties."""

    def __init__(
            self,
            name='dataset',
            load_from_filename=None,
            **keys_vals):
        optimise.Optimiser.__init__(self,name=name)
        self.pop_format_input_function()
        self.add_format_input_function(lambda: 'not implemented')
        self._data = dict()
        self._length = None
        if not hasattr(self,'prototypes'):
            ## derived classes might set this in class definition, so
            ## do not overwrite here
            self.prototypes = {}
        self._inferences = AutoDict([])
        self._inferred_from = AutoDict([])
        self.permit_nonprototyped_data =  True
        self.uncertainty_prefix = 'd_' # a single letter to prefix uncertainty keys
        self.verbose = False
        for key,val in keys_vals.items():
            self[key] = val
        if load_from_filename is not None:
            self.load(load_from_filename)

            
    def __len__(self):
        assert self._length is not None,'Dataset has no length because all data is scalar'
        return(self._length)

    # def _get_keys_values_uncertainties(self,**keys_vals):
        # """Match keys for value and uncertainties, e.g., 'x' and 'dx'."""
        # keys_values_uncertainties = {}
        # keys = list(keys_vals.keys())
        # while len(keys)>0:
            # key = keys[0]
            # if key[0] == self.uncertainty_prefix:
                # assert key[1:] in keys,f'Uncertainty {repr(key)} in data but {repr(key[1:])} is not.'
                # keys_values_uncertainties[key[1:]] = (keys_vals[key[1:]],keys_vals[key])
                # keys.remove(key)
                # keys.remove(key[1:])
            # elif self.uncertainty_prefix+key in keys:
                # keys_values_uncertainties[key] = (keys_vals[key],keys_vals[self.uncertainty_prefix+key])
                # keys.remove(key)
                # keys.remove(self.uncertainty_prefix+key)
            # else:
                # keys_values_uncertainties[key] = (keys_vals[key],None)
                # keys.remove(key)
        # return(keys_values_uncertainties)

    def __setitem__(self,key,value):
        """Shortcut to set, cannot set uncertainty this way."""
        if (value_key:=self._get_key_without_uncertainty(key)) is not None:
            self.set_uncertainty(value_key,value)
        else:
            self.set(key,value)

    # def set_value(self,key,value,is_scalar=None,**data_kwargs):
    #     """Set a value and possibly its uncertainty."""
    #     self.unset_inferences(key)
    #     assert self.permit_nonprototyped_data or key in self.prototypes, f'New data is not in prototypes: {repr(key)}'
    #     ## if not previously set then get perhaps get a prototype
    #     if key not in self and key in self.prototypes:
    #         for tkey,tval in self.prototypes[key].items():
    #             if tkey == 'infer':
    #                 continue # not a Data kwarg
    #             data_kwargs.setdefault(tkey,copy(tval))
    #     ## set the data
    #     if is_scalar or np.isscalar(value):
    #         self._data[key] = Datum(value=value,**data_kwargs)
    #     else:
    #         self._data[key] = Data(value=value,**data_kwargs)
    #         if self.is_scalar():
    #             ## first array data, use this to define the length of self
    #             self._length = len(self._data[key])
    #         else:
    #             assert len(self._data[key])==len(self),f'Length of data {repr(key)} does not match existing data.'

    def set(
            self,
            key,
            value,
            uncertainty=None,
            is_scalar=None,
            **data_kwargs,
    ):
        """Set a value and possibly its uncertainty. Set is_scalar=True to set
        a scalar Object type that is iterable."""
        assert self.permit_nonprototyped_data or key in self.prototypes, f'New data is not in prototypes: {repr(key)}'
        ## if not previously set then get perhaps get a prototype
        if key not in self and key in self.prototypes:
            for tkey,tval in self.prototypes[key].items():
                if tkey == 'infer':
                    continue # not a Data kwarg
                data_kwargs.setdefault(tkey,copy(tval))
        ## set the data
        if is_scalar or np.isscalar(value):
            data = Datum(value=value,uncertainty=uncertainty,**data_kwargs)
        else:
            data = Data(value=value,uncertainty=uncertainty,**data_kwargs)
            if self.is_scalar():
                ## first array data in self, use this to define the
                ## length of self
                self._length = len(data)
            else:
                assert len(data)==len(self),f'Length of new data {repr(key)} is {len(data)} and does not match the length of existing data: {len(self)}.'
        self._data[key] = data
        self.unset_inferences(key)

    def set_uncertainty(self,key,uncertainty):
        """Set a the uncertainty of an existing value."""
        self.unset_inferences(key)
        assert key in self,f'Value must exist before setting uncertainty: {repr(key)}'
        self._data[key].uncertainty  = uncertainty

    def unset(self,key):
        """Delete data.  Also clean up inferences."""
        self.unset_inferences(key)
        self._data.pop(key)

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

    def get_value(self,key,ensure_vector=False):
        if key not in self._data:
            self._infer(key)
        if ensure_vector and self.is_scalar(key):
            assert len(self) is not None
            return np.full(len(self),self[key])
        else:
            return self._data[key].value

    def get_uncertainty(self,key):
        self.assert_known(key)
        if self.has_uncertainty(key):
            return self._data[key].uncertainty
        else:
            return None

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
        assert not self.is_scalar(),'Cannot index, all data is scalar.'
        for data in self._data.values():
            if isinstance(data,Data):
                data.index(index)
                self._length = len(data) # take length from last processed

    def copy(self,keys=None,index=None):
        """Get a copy of self with possible restriction to indices and
        keys."""
        if keys is None:
            keys = self.keys()
        retval = self.__class__() # new version of self
        for key in keys:
            if self.is_scalar(key) or index is None:
                retval[key] = self[key]
            else:
                retval[key] = deepcopy(self[key][index])
        return(retval)

    def match(self,**keys_vals):
        """Return boolean array of data matching all key==val."""
        i = np.full(len(self),True,dtype=bool)
        for key,val in keys_vals.items():
            ## if key has suffix 'min' or 'max' then match to range the match anything greater
            ## or equal to this value
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
        return(i)

    def matches(self,**keys_vals):
        """Returns a copy reduced to matching values."""
        return(self.copy(index=self.match(**keys_vals)))
    def limit_to_matches(self,**keys_vals):
        self.index(self.match(**keys_vals))
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
            if (value_key:=self._get_key_without_uncertainty(arg)) is not None:
                return(self.get_uncertainty(value_key))
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
                    # already_attempted.append(dependency) # in case it comes up again at this level
                ## compute value if dependencies successfully inferred
                self[key] = function(*[self[dependency] for dependency in dependencies])
                ## compute uncertainties by linearisation
                squared_contribution = []
                value = self[key]
                parameters = [self[t] for t in dependencies]
                for i,dependency in enumerate(dependencies):
                    if self.has_uncertainty(dependency):
                        dparameters = copy(parameters)
                        diffstep = self[dependency]*1e-10
                        dparameters[i] += diffstep
                        dvalue = value - function(*dparameters)
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
            ## some kind of InferException 
            except InferException as err:
                continue      
        ## complete failure to infer
        else:
            raise InferException(f"Could not infer key: {repr(key)}")

    def __iter__(self):
        for key in self._data:
            yield key

    def rows(self):
        d = {key:self[key] for key in self if self.is_scalar(key)}
        vector_keys = [key for key in self if not self.is_scalar(key)]
        for i in range(len(self)):
            d.update({key:self[key][i] for key in vector_keys})
            yield d
            
    def keys(self):
        return(list(self._data.keys()))

    # def is_known(self,*keys):
        # for key in keys:
            # try:
                # self.get_value(key)
            # except InferException:
                # return(False)
        # return(True)

    def assert_known(self,*keys):
        for key in keys:
            self[key]

    def is_known(self,*keys):
        try:
            self.assert_known(*keys)
            return True 
        except InferException:
            return False

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
        """Format data into a string representation."""
        if keys is None:
            keys = self.keys()
        ## collect table data
        header,columns = [],[]
        for key in keys:
            if self.is_scalar(key):
                header.append(f'{comment}{key} = {str(self._data[key])}')
            else:
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
                self[key] = data[key]
                keys.remove(key)

    def is_scalar(self,key=None):
        """Return boolean whether data for key is scalar or not. If key not
        provided return whether all data is scalara or not."""
        if key is None:
            return(self._length is None)
        else:
            return(isinstance(self._data[key],Datum))

    def make_vector(self,key=None):
        if key is None:
            for key in self:
                self.make_vector(key)
            return
        data = self._data[key]
        if self.is_scalar():
            value = [data.value]
            self._length = 1
        else:
            value = np.full(len(self),data.value)
        if isinstance(data,Datum):
            self._data[key] = Data(
                value=value,
                uncertainty=data.uncertainty,
                kind=data.kind,
                description=data.description,
                units=data.units)

    def make_scalar(self,key=None):
        """Make array data that has a unique value scalar."""
        if key is None:
            for key in self:
                self.make_scalar(key)
            return
        data = self._data[key]
        if (isinstance(data,Data)
            and len(np.unique(data.value))==1
            and (not data.has_uncertainty()
                 or len(np.unique(data.uncertainty))==1)):
            self._data[key] = Datum(
                value=data.value[0],
                uncertainty=(data.uncertainty[0] if data.has_uncertainty() else None),
                kind=data.kind,
                description=data.description,
                units=data.units)

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
        """Concatenate data from another Dataset object to this one. All
        existing data in current Dataset must be known to the new
        Dataset, but the reverse is not enforced.  If the existing
        Dataset is scalar then its existing data is not vectorised
        before concatenation."""
        if new_dataset.is_scalar():
            ## only concatenate if vector data present
            return
        if self.is_scalar():
            ## just copy everything, keeping scalar data, but maybe
            ## overwritten by new_dataset
            for key in new_dataset:
                self[key] = new_dataset[key]
            return
        ## sort out scalar and  vector keys
        for key in self:
            if self.is_scalar(key):
                if key in new_dataset:
                    if new_dataset.is_scalar(key):
                        if new_dataset[key] == self[key]:
                            ## matching scalar data in existing and new data -- UNCERTAINTIES!!!!
                            continue
                        else:
                            ## non-matching data, make vector
                            self.make_vector(key)
                            new_dataset.make_vector(key)
                    else:
                        ## make all vector
                        self.make_vector(key)
                else:
                    ## keep existing scalar data as scalar since its not in new data
                    continue
            else:
                assert new_dataset.is_known(key), f"Key missing in new_dataset: {repr(key)}"
                if new_dataset.is_scalar(key):
                    ## make new data vector
                    new_dataset.make_vector(key)
                else:
                    ## everything already vector
                    pass
            ## copy new_data to self
            if self.is_scalar(key):
                continue
            self._data[key].extend(
                new_dataset.get_value(key),
                new_dataset.get_uncertainty(key),)
        self._length = len(self) + len(new_dataset)

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
            **optimise_keys_vals
    ):
        parameters = self.add_parameter_set(**optimise_keys_vals)
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
