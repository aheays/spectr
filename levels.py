from spectr.dataset import Dataset

class Level(Dataset):

    # (_class_key_data, _class_scalar_data, _class_vector_data) = _generate_keys(
        # more_vector_other_keys=('index','name',),
        # more_scalar_other_keys = ['level_transition_type','description',
                                  # 'date', 'author', 'reference',
                                  # 'data_identifier','Tref','dTref',
                                  # 'partition_source',],)

    # # defining_qn = all_qn        # minimum list of unique quantum numbers
    # default_sorting_order = None
    # default_plot_xkey  = None                    # if not specified, plot this on x-axis
    # default_plot_ykeys = (None,)                 # if not specified, plot these on y-axis
    # default_plot_zkeys = ()                 # if not specified, plot these on separate subplot
    # ## Key translation used by load_from_file_with_translation. If
    # ## an element is a list then the 2nd element of implied data.
    # _load_from_file_with_translation_dict = dict(
        # # ex2=('example2',{'example1':52.}),
    # )
    # _decode_name_function = lambda self,name: {'example1':45} # Return a dictionary of quantum numbers decoded from one name.
    # verbose = False                  # True to turn on extra output
    
    def __init__(
            self,
            name=None, # capitalised because 'name' refers to line data 'name'
            # decode_names=True, # whether or not to infer quantum numbers from level/transition names
            # permit_new_keys=False, # whether new kesy can be added by assignment
            # load_from_filename=None, # loads data from this if given
            # copy_data_from=None,     # base data on this Level_Transition of the same type
            # set_all_defaults=False,
            # description=None,   # set if a key but also passed to Optimiser -- confusing!
            # **kwargs
    ):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
        pass
        # ## make internal data array
        # Dynamic_Recarray.__init__(
            # self,
            # *deepcopy(list(self._class_vector_data.values())),
            # scalar=deepcopy(self._class_scalar_data), 
            # default_sorting_order=self.default_sorting_order,
            # permit_new_keys=permit_new_keys,
            # extend_data_when_first_set= True,
        # )
        # ## make an instance variable
        # self.key_data = deepcopy(self._class_key_data) # does not exist in regular Dynamic_Recarray
        # ## set type
        # self['level_transition_type'] = self.__class__.__name__
        # ## perhaps remove these
        # self.all_qn = list(self.key_data['qn']) # DEPRECATED
        # self.defining_qn = list(self.key_data['defining_qn']) # DEPRECATED
        # ## deal with inputs
        # kwargs_original = copy(kwargs)
        # self.name = (Name if Name is not None else self['level_transition_type']) # generic object name used by optimiser
        # self.decode_names = decode_names # whether or not to infer quantum numbers from level/transition names
        # ## special case of name in kwargs if decode it then abort silently on error
        # if self.decode_names and 'name' in kwargs and np.isscalar(kwargs['name']):
            # t = kwargs.pop('name')
            # try:
                # kwargs.update(self._decode_name_function(t).items())
            # except InvalidEncodingException:
                # warnings.warn(f"Error decoding name {repr(t)}")
        # ## handle scalar and defaults in kwargs
        # for key in list(kwargs):
            # ## if scalar data then set this value
            # if key in self.scalar_data:
                # self[key] = kwargs.pop(key)
            # ## set as default data
            # elif np.isscalar(kwargs[key]) and key in self.vector_data:
                # self.vector_data[key].default_value = kwargs.pop(key) 
                # self.vector_data[key].is_set = True
                # self[key] = self.vector_data[key].default_value # add to internal array
        # ## setting remaining vector kwargs initial data
        # self.append(**kwargs)
        # ## some things cache data, here it is initialised. self.clear_cache() to delete data
        # self._cache = {}
        # ## create an optimiser
        # Optimiser.__init__(self,name=self.name)
        # self.format_input_functions = [] # no formatted input
        # ## new input line
        # def format_input():
            # retval = f'{self.name} = {self["level_transition_type"]}(Name={repr(self.name)}'
            # if load_from_filename is not None:
                # retval += f',load_from_filename={repr(load_from_filename)}'
            # if copy_data_from is not None:
                # retval += f',copy_data_from={copy_data_from.name}'
            # if len(kwargs_original)>0:
                # retval += f',{my.dict_to_kwargs(kwargs_original)}'
            # return(retval+')')
        # self.format_input_functions.append(format_input)
        # ## output to directory
        # def f(directory):
            # filename = f'{directory}/{self["level_transition_type"]}.h5'
            # self.unset_all_inferences()
            # self.save_to_file(filename) 
        # self.output_to_directory_functions.append(f) 
        # ## set description string
        # if description is not None:
            # self.set_description(description)
            # if self.has_key('description'):
                # self['description'] = description
        # ## set default data if requested
        # if set_all_defaults:
            # self.set_default(*self.vector_data)
        # ## load from file if requested, handle the case where this is
        # ## is possibly a list of filenames
        # if load_from_filename is not None:
            # for filename in my.ensure_iterable(load_from_filename):
                # self.load_from_file(filename)
                # self.format_input_functions.pop(-1) # since handled above
        # ## copy data from another object
        # if copy_data_from is not None:
            # self.copy_data_from(copy_data_from)
            # self.format_input_functions.pop(-1) # since handled above


    def load_from_file(self,*args,**kwargs):
        """Thin wrapper on Dynamic_Recarray.load_from_file to set
        format_input_function."""
        self.format_input_functions.append(
            lambda: f'{self.name}.load_from_file({my.repr_args_kwargs(*args,**kwargs)})')
        Dynamic_Recarray.load_from_file(self,*args,**kwargs)

    def format_table(
            self,
            *keys,     # some valid data keys or key_classes in key_data, e.g., 'defining_qn','vector_data','scalar_data'
            status='set',       # 'set' or 'known'
            skip_all_nan=False,
            **format_table_kwargs,
    ):
        """Like in Dynamic_Recarray except has some convenient options for
        including quantum numbers more easily."""
        new_keys = []
        ## expand key classes
        for key in keys:
            if key in self.key_data:
                for tkey in self.key_data[key]:
                    if tkey in keys: continue # do not repeat
                    if ((status=='set' and  self.is_set(tkey))
                        or (status=='known' and  self.is_known(tkey))):
                        new_keys.append(tkey)
            else:
                new_keys.append(key)
        ## add uncertainties
        keys = []
        for key in new_keys:
            if key in keys:
                continue
            keys.append(key)
            if (key in self.key_data['data'] and self.is_known('d'+key) and 'd'+key not in keys):
                keys.append('d'+key)
        ##
        if skip_all_nan:
            for key in copy(keys):
                try:
                    if np.all(np.isnan(self[key])):
                        keys.remove(key)
                except TypeError:
                    pass
        return(Dynamic_Recarray.format_table(self,*keys,**format_table_kwargs))

    def unset(self,*keys,**kwargs):
        """Like in Dynamic_Recarray but automatically unset uncertainties."""
        for key in keys:
            Dynamic_Recarray.unset(self,key,**kwargs)
            if key in self.key_data['vector_data']+self.key_data['scalar_data']:
                Dynamic_Recarray.unset(self,'d'+key,**kwargs)

    def copy_data_from(self,source,*keys):
        """Copy all data from source and update if source changes during
        optimisation."""
        assert type(self)==type(source)
        assert len(self)==0
        cache = {}
        def f():
            ## woefully inadequate error check
            if len(cache)==0:
                assert len(self)==0
                ## prepare empty self of correct length
                self.set_length(len(source))
                ## default list of keys in none providedi
                if len(keys)>0:
                    cache['keys'] = keys
                else:
                    cache['keys'] = []
                    for key in source.set_keys():
                        if len(source.get_data_type(key).inferred_from)==0:
                            cache['keys'].append(key)
                ## add all keys to internal array -- faster than
                ## adding them one at a time as data is set below
                self._add_keys_to_data_recarray(
                    *[key for key in cache['keys'] if key in self.vector_data])
                ## set all data
                for key in cache['keys']:
                    self[key] = source[key]
            else:
                assert len(self)==len(source)
            ## reset data only if changed
            for key in cache['keys']:
                if ((key in self.key_data['scalar'] and self[key]!=source[key])
                    or (key in self.key_data['vector'] and np.any(self[key]!=source[key]))):
                        self[key] = source[key]
        self.suboptimisers.append(source)
        self.construct_functions.append(f)
        self.format_input_functions.append(lambda: f'{self.name}.copy_data_from({source.name})')

    def plot(self,xkey=None,ykeys=None,zkeys=None,**kwargs):
        """Thin wrapper on Dynamic_Recarray.plot to set defaults."""
        if len(ykeys)==0:
            ykeys = [t for t in self.default_plot_ykeys if self.is_known(t)]
        if xkey is None:
            xkey = self.default_plot_xkey
        if zkeys is None:
            zkeys = [t for t in self.default_plot_zkeys if self.is_known(t)]
        kwargs.setdefault('zlabel_format_function',self._encode_name_function)
        Dynamic_Recarray.plot(self,xkey=xkey,ykeys=ykeys,zkeys=zkeys,**kwargs)

    def add_line(self,**keys_vals):
        """Add a line of data, or one record, with keys_vals that can
        be optimised."""
        ## if name='...' given then use it to decode quantum numbesr
        if 'name' in keys_vals: # I'm not sure this is necessary -- it might be done automatically by append
            for key,val in self._decode_name_function(keys_vals['name']).items():
                keys_vals.setdefault(key,val)
        ## divide keys into optimised and fixed
        keys_vals_fixed,keys_vals_varied = {},{}
        for key,val in keys_vals.items():
            if np.isscalar(val):
                keys_vals_fixed[key] = val
            else:
                keys_vals_varied[key] = val
        ## optimisable data
        parameters = self.add_parameter_set(
            f'add_line {repr(keys_vals_fixed)}',
            **{key:val for key,val in keys_vals_varied.items()})
        ## change fmt to sensible
        for key in keys_vals_varied:
            parameters.param_dict[key].pfmt = self.vector_data[key].fmt
        ## make line
        self.append(**{p.name:p.p for p in parameters},**keys_vals_fixed)
        ## update function
        i = len(self)-1         # record index for updating
        def f():
            for p in parameters:
                self[p.name][i] = p.p # update
        self.add_construct(f)
        ## new input line
        self.format_input_functions.append(
            lambda: f'{self.name}.add_line({parameters.format_input()},{my.dict_to_kwargs(keys_vals_fixed)})'.replace('(,','('))

    def add_lines(
            self,
            keys,               # list of keys 
            *value_lists,       # list of 2D data where each row matches length of keys. Elements can be strings, scalar, or Parameter inputs.
            **common_keys_vals   # some other common data to set along with the value lists -- not optimisable
    ):
        """Add many lines of data efficiently, with values possible optimised."""
        ## add data from name to common_keys_vals if it is provided
        if 'name' in common_keys_vals:
            for key,val in self._decode_name_function(common_keys_vals.pop('name')).items():
                common_keys_vals.setdefault(key,val)
        keys_vals = collections.OrderedDict() # will contain line data
        parameters = []                       # save optimisation parameters in a list
        value_lists = [list(t) for t in value_lists]       # make mutable
        for ivals,vals in enumerate(value_lists):              # loop through all lines
            for ikey,(key,val) in enumerate(zip(keys,vals)):
                if key not in keys_vals: keys_vals[key] = [] # add to keys if first
                if np.isscalar(val):                         
                    keys_vals[key].append(val) # add data to append
                else:
                    p = self.add_parameter(key,*val) # add to optimiser
                    vals[ikey] = p
                    parameters.append((ivals+len(self),p)) # record which line in self will be after appending data
                    keys_vals[key].append(p.p)   # add data to append
        ## append lines to self
        self.append(**keys_vals,**common_keys_vals)
        ## add optimisation function
        def f():
            for i,p in parameters:
                self[p.name][i] = self.vector_data[p.name].cast(p.p) # update
                self.unset_inferences(p.name)
        self.add_construct(f)
        ## add format input function
        def f():
            retval = f'{self.name}.add_lines(\n    {repr(keys)},'
            retval += '\n    '+",\n    ".join([repr(vals) for vals in value_lists])+",\n    "
            if len(common_keys_vals)>0:
                retval += my.dict_to_kwargs(common_keys_vals)+','
            retval +=  ')'
            return(retval)
        self.format_input_functions.append(f)

    def add_lines_from_kwargs(self,**keys_vals):
        """Add new lines of data to self by keyword argument. keys_vals are
        manipulated and passed to add_lines. Scalar vals are
        common_keys_vals in add_lines and vector are the optimisable
        parameters, and must all be the same length."""
        ## parse keys_vals
        vector_data = collections.OrderedDict()
        scalar_data = collections.OrderedDict()
        vector_length = None
        for key,val in keys_vals.items():
            if np.isscalar(val):
                scalar_data[key] = val
            else:
                vector_data[key] = val
                if vector_length is None:
                    vector_length = len(val)
                assert len(val)==vector_length,'Nonconstant lengths of vector data.'
        ## build value lists
        self.add_lines(list(vector_data.keys()), *zip(*vector_data.values()), **scalar_data)

    def add_lines_from_level_transition(
            self,
            source,             # a level_transition of the same type as self
            *keys,              # list of keys or (key,vary,step) to indicate what to add to self and what to optimise
            **common_keys_vals, # added globally to self, if name is included it will be decoed for quantum numbers
    ):
        """Collect data from a level_transition to self, optimising some of it
        and pass to add_lines so it can be added to self."""
        ## determine vary and step from keys given as tuples
        keys = list(keys)
        vary,step = {},{}
        for i,key in enumerate(copy(keys)):
            if not np.isscalar(key):
                keys[i] = key[0]
                vary[key[0]] = key[1] if len(key)>1 else False
                step[key[0]] = key[2] if len(key)>2 else None
        ## basic error checks
        assert type(self)==type(source)
        assert len(self)==0
        source.assert_known(*keys)
        ## build a list of input lines for add_lines
        values_lists = []
        for sourcei in source:
            values_list = []
            for key in keys:
                if key in vary:
                    values_list.append((sourcei[key],vary[key],step[key]))
                else:
                    values_list.append(sourcei[key])
            values_lists.append(values_list)
        ## call add_lines
        self.add_lines(keys,*values_lists,**common_keys_vals)
        

    def set_value(
            self,
            unset=None,         # optional key or keys to unset every iteration
            **qn_and_data, # quantum numbers are used to match levels/transitions and parameters are set and/or optimised
    ):
        """Change data in this object. If datakey=val is a
        Parmeter/Optimsier or iterable then it will be optimsied.  If
        scalar then just set. qn are used to only set matching
        recrods. example:\n
        x.set_value(T=(1000,True,1e-2),Tpp=0,Λp=0,labelp=X,vp=0,Jp=[14,15],unset=('ν',)) """
        ## unset can be input as a single key or a list
        if unset is not None:
            unset = my.ensure_iterable(unset)
        ## find kwargs that are qn to match on and fixed data values
        qn,data = {},{}
        for key in list(qn_and_data):
            assert key in self.all_keys(),f'Unknown key: {repr(key)}'
            if key in self.key_data['qn']:
                qn[key] = qn_and_data.pop(key)
            elif np.isscalar(qn_and_data[key]):
                data[key] = qn_and_data.pop(key)
        ## interpete remaining kwargs as data
        t = self.add_parameter_set(note='set_value '+repr(qn),**qn_and_data)
        data.update(t.param_dict)
        ## make a new input line
        def f():
            retval = [f'{self.name}.set_value(']
            for key,val in qn.items():
                retval.append(f'{key}={repr(val)},')
            if unset is not None:
                retval.append(f'unset={repr(unset)},')
            for key,val in data.items():
                if len(data)>1:
                    retval.append(f'\n    {key}={repr(val)},')
                else:
                    retval.append(f'{key}={repr(val)},')
            retval.append(')')
            return(''.join(retval))
        self.format_input_functions.append(f)
        ## optimisation function
        cache = {}
        def f():
            ## if first run, find matching lines
            if len(cache)==0:
                cache['i'] = (None if len(qn)==0 else self.match(**qn))
            ## set data
            for key,val in data.items():
                self.setval(key,val,cache['i'])
            ## unset anything explicity specified to be unest
            if unset is not None:
                self.unset(*unset)
        ## add to optimiser
        self.add_construct(f)

    def setval(
            self,
            key,                # key being set
            val,                # data
            index=None,             # optional index
            dval=None,          # optional uncertainty
    ):
        '''Set key to value, and optional dkey to dval. If index given set
        only those indices. If val is a Parameter or Optimised_Parameter get
        val and dval from this. Not optimised, done instantly.'''
        ## get values out of Parameter or Optimised_Parameter
        if (isinstance(val,optimise_model.Parameter)
            or isinstance(val,optimise_model.Optimised_Parameter)):
            val,dval = val.p,val.dp
        assert key in self.all_keys(),f'Unknown key {repr(key)}'
        assert dval is None or 'd'+key in self.all_keys(),f'Unknown key {repr("d"+key)}'
        ## in case scalar value
        if key in self.scalar_data:
            assert index is None,r'Cannot use index for scalar data {repr(key)}'
            self[key] = val
            if dval is not None:
                self['d'+key] = dval
        ## in case vector value
        else:
            assert index is None or self.is_known(key),f'Cannot set {repr(key)} for some indices when the others are not known.'
            assert dval is None or index is None or self.is_known('d'+key),f'Cannot set {repr("d"+key)} for some indices when the others are not known.'
            ## set all data
            if index is None:
                self[key] = val
                if dval is not None:
                    self['d'+key] = val
            ## limit to given index
            else:
                self[key][index] = val
                self.unset_inferences(key)
                if dval is not None:
                    self['d'+key][index] = val
                    self.unset_inferences('d'+key)


    def set_values(
            self,
            keys,               # list of keys that are varied line by line in "lines"
            *lines,                         # first element is a dict of quantum numbers, remaining elements match keys
            **parameters_and_quantum_numbers, # list of common parametres and quantum numbers adjusting print matching data
    ):
        lines = [list(t) for t in lines]
        ## sort kwargs into quantum numbers and possibly variable parameters
        qncommon,pcommon = {},collections.OrderedDict()
        for key,val in parameters_and_quantum_numbers.items():
            if key in self.key_data['qn']:
                qncommon[key] = val
            elif key in self.key_data['data']:
                p = self.add_parameter(key,*val)
                parameters_and_quantum_numbers[key] = val
                pcommon[key] = val
            else:
                raise KeyError(f'Not implmeneted key: {repr(key)}')
        def f():
            retval =  f'{self.name}.set_values('
            retval += f'\n    {repr(keys)},'
            retval += f'\n    {repr(parameters_and_quantum_numbers)},'
            retval += '\n    '+',\n    '.join([repr(line) for line in lines])+','
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        ## add all varied data to self.optimiser
        for iline,line in enumerate(lines):
            for ikey,(key,val) in enumerate(zip(keys,line[1:])):
                line[ikey+1] = self.add_parameter(key,*val)
        ## optimisation function
        cache = {}              # permanently store indices
        def f():
            if len(self)==0: return # nothing to do
            if 'irows' not in cache:
                icommon = my.find(self.match(**qncommon))
                t = self[icommon]
                cache['irows'] = tuple([icommon[t.match(**line[0])] for line in lines])
                ## make sure uncertainties set
                for key in keys:
                    if not self.is_set(key):
                        self[key] = np.nan
                    if not self.is_set('d'+key):
                        self['d'+key] = np.nan
            ## delete anything dependent calculated from varied parameter,
            self.unset_inferences(*keys)
            for p in pcommon:
                self[p.name] = p.p
                self['d'+p.name] = p.dp
            ## update data line-by-line
            for iline,line in enumerate(lines):
                for ikey,(key,val) in enumerate(zip(keys,line[1:])):
                    irows = cache['irows'][iline]
                    p = line[ikey+1]
                    self[key][irows] = p.p
                    self['d'+key][irows] = p.dp
                    if self.vector_data[p.name].strictly_positive:
                        self[p.name][irows] = np.abs(self[p.name][irows]) # make positive if required
        ## add to optimiser
        self.add_construct(f)

    def shift_value(
            self,
            key,                # to shift
            shift,              # optimisable parameter
            unset=(),           # unset these when optimising
            **limit_to_matches,
    ):
        """Shift (or optimise) values of key."""
        p = self.add_parameter('shift_value',*my.ensure_iterable(shift))
        previous_shift = [0.]
        def do_shift():
            i = self.match(**limit_to_matches)
            self[key][i] += p - previous_shift[0]
            previous_shift[0] = float(p)
        do_shift()
        self.construct_functions.append(do_shift)
        self.format_input_functions.append(
            lambda: f'{self.name}.shift_value({repr(key)},shift={repr(p)},unset={repr(unset)},{my.dict_to_kwargs(limit_to_matches)})')

    def scale_value(
            self,
            key,                # to scale
            scale,              # optimisable parameter
            **limit_to_matches,
    ):
        """Scale (or optimise) values of key."""
        p = self.add_parameter('scale',*my.ensure_iterable(scale),note=f'scale_value: {repr(limit_to_matches)}')
        cache = {}
        def f():
            if len(cache)==0:
                cache['i'] = self.match(**limit_to_matches)
            i = cache['i']
            self[key][i] *= p.p 
            self.unset_inferences(key)
        self.construct_functions.append(f)
        self.format_input_functions.append(
            lambda: f'{self.name}.scale_value({repr(key)},scale={repr(p)},{my.dict_to_kwargs(limit_to_matches)})')

    def set_all_values_to_current_values(self,*keys,unset=(),**matching_qn):
        """set_value_to_current_value fo key for all matching_qn individually."""
        for i in my.find(self.match(**matching_qn)):
            self.set_value(
                unset=unset,
                **{qn:self[qn][i] for qn in self.defining_qn if self.is_known(qn)}, # quantum numbers for this level
                **{key:self[key][i] for key in keys},) # variable parameters

    def set_value_to_current_value(self,*keys,unset=(),**matching_qn):
        """Create a "set_value" call variable key restricted
        to quantum numbers qn. The initial value is set to the current
        value. If multiple matches the first value is used."""
        i = self.match(**matching_qn)
        self.set_value(unset=unset,**matching_qn,**{key:self[key][i][0] for key in keys})
        
    def set_polynomial_value(
            self,
            xkey,               # e.g., J or Np(Np+1)
            unset=(),           # unset these when optimising
            **parameters_and_quantum_numbers, # quantum numbers select level/transition to set value for. Parmaeters are e.g. f0=(...) or Γp3=(...) for zeroth and 3rd order f-values and widths
    ):
        xkey_in = xkey
        ## sort kwargs into quantum numbers and possibly variable parameters
        qn,parameters = {},collections.OrderedDict()
        for key,val in parameters_and_quantum_numbers.items():
            if key in self.key_data['qn']:
                qn[key] = val
            else:
                parameters[key] = val
        ## a convenience function to find matching quantum numbers, with cache
        local_cache = dict(previous_length=None,i=None)
        def find_matching_qn():
            ## if length changed calc a new i -- IF ORDER OF LINES CHANGES WE HAVE A PROBLEM
            if local_cache['previous_length'] is None or local_cache['previous_length']!=len(self):
                local_cache['previous_length'] = len(self)
                if len(qn)==0:
                    local_cache['i'] = slice(0,len(self)) # all lines
                else:
                    local_cache['i'] = self.match(**qn)   # find matching lines
            return(local_cache['i'])
        ## add optimisable parameters
        parameters = self.add_parameter_set(f'set_value {repr(parameters_and_quantum_numbers)} unset={repr(unset)}',**parameters) 
        self.format_input_functions.append(lambda: f'{self.name}.set_polynomial_value({parameters.format_input()},xkey={repr(xkey_in)},unset={repr(unset)},{my.dict_to_kwargs(qn)},)')
        ## modified keys
        modified_keys = set()
        for p in parameters:
            ## t0,t1= re.match(r'(.*[^[0-9])([-+eEfF0-9]+)',p.name).groups()
            t0,t1= re.match(r'(.*[^[0-9])([0-9]+)',p.name).groups()
            modified_keys.add(t0)
            assert t0 in self.all_keys(),f'Unknown key {repr(t0)}'
            p.name_without_order,p.order = t0,int(t1)
        ## determine xkey, if given as e.g., Jp(Jp+1) then incorporate this fact
        xkey_type = 'linear'
        t = re.match(r'(?P<xkey>.+)\((?P=xkey)\+1\)',xkey) # regex search for xkey(xkey+1)
        if t: xkey,xkey_type = t.group(1),'x(x+1)'
        ## optimisation function
        def f():   
            if len(self)==0: return # nothing to do
            ## compute spline values
            i = find_matching_qn()   
            for key in modified_keys:
                self[key][i] = 0.
                ## recalculate anything dependent on the varied parameter,
                ## this might require manual internation by specifying som
                ## extra unset variables
                self.unset_inferences(key)
            self.unset(*unset,unset_inferences=True)
            for p in parameters:
                if xkey_type=='linear':
                    x = self[xkey][i]
                elif xkey_type=='x(x+1)':
                    x = self[xkey][i]*(self[xkey][i]+1)
                self[p.name_without_order][i] += p.p*x**p.order
                ## make positive if required
                if self.vector_data[key].strictly_positive:  self[key][i] = np.abs(self[key][i]) 
        self.add_construct(f) # add to optimiser

    def set_spline_value(
            self,
            xkey,               # e.g., 'J' or 'Np(Np+1)'
            ykey,               # e.g., 'f' or 'Γpp'
            *spline_values,     # (x,y) spline points, y can be optimisable e.g., (0,2.5),(10,(3.7,True,1e-3),...
            order=3,            # spline order
            **matching_qn,      # limit matching quantum numbers
    ):
        """
        Set ykey value of data matching matching_qn to a spline
        function given by spline_values in terms of xkey, e.g.,:
           o.set_spline_value('Jp(Jp+1)','Γp',(0,(1,True,1e-3)),(50,0.5,True,1e-3),species='14N2')
        """
        ## convert yvalues to Parameters if given as a list
        spline_values = list(spline_values)
        for i,(xval,yval) in enumerate(spline_values):
            if not np.isscalar(yval):
                spline_values[i] = (xval,self.add_parameter(f'{xkey}{xval}',*yval,note=f'set_spline_value, xkey={xkey}, ykey={ykey}'))
        ## new input line function
        def f():
            retval = [f'{self.name}.set_spline_value(\n     {repr(xkey)},{repr(ykey)}']
            for xval_yval in spline_values:
                retval.append(repr(xval_yval))
            if order!=3:
                retval.append(f'order={repr(spline_order)}')
            if len(matching_qn)>0:
                retval.append(my.dict_to_kwargs(matching_qn))
            return(',\n    '.join(retval)+')')
        self.format_input_functions.append(f)
        ## add optimisation function
        cache = {}
        def f():
            ## first run -- find matching lines and cache x-values
            if len(cache)==0:
                cache['i'] = (slice(0,len(self)) if len(matching_qn)==0 else self.match(**matching_qn))
                cache['ys'] = [t[1] for t in spline_values] # will change as parameters optimised
                if xkey in self.all_keys():
                    cache['xs'] = [t[0] for t in spline_values]
                    cache['x'] = self[xkey][cache['i']] # assumes doesn't change ever
                else:
                    r = re.match(r'(?P<xkey>.+)\((?P=xkey)\+1\)',xkey) # regex search for xkey(xkey+1)
                    if r:
                        cache['xs'] = [t[0]*(t[0]+1) for t in spline_values]
                        txkey = r.group(1)
                        cache['x'] = (self[txkey]*(self[txkey]+1))[cache['i']]
                    else:
                        raise Exception(f"Could not decode xkey {repr(xkey)}. Expecting a valid key, or key(key+1).")
            ## calculate spline
            self[ykey][cache['i']] = my.spline(cache['xs'],cache['ys'],cache['x'],order=order)
            self.unset_inferences(ykey)
        self.add_construct(f)
        
    def set_Tref(self,Tref):
        """Change the reference energy. Can be 'T0-Te' for database value."""
        assert self.is_known('Tref'),'Cannot change Tref because the current value is unknown.'
        ## get database T0
        if Tref=='T0-Te':
            if isinstance(self,Base_Level):
                species_key = 'species'
            elif isinstance(self,Base_Transition):
                ## ASSUMES SPECIESPP==SPECIESP
                species_key = 'speciespp'
            else:
                raise ImplementationError()
            assert len(self.unique(species_key))==1,'Not implemented mixed species'
            Tref = database.get_species_property(self[species_key][0],'T0-Te')
        for key in ('T','Tp','Tpp','Tv','Tvp','Tvpp'):
            if key in self.all_keys() and self.is_set(key):
                self[key] += self['Tref'] - Tref
        self['Tref'] = Tref

    def exchange_quantum_numbers(self,iname=None,jname=None,**qn):
        """Exchange all quantum numbers for matching levels i and j. All keys
        in qn that are quantum numbers will be used to match both i
        and j. Quantum numbers with "i" appended will match i-levesl
        and if "j" appended will match j-levels. If multiple levels
        are swapped then they are aligned according to whatever
        current sorting of data is. iname and jname are decoded to get
        quantum numbesr, that will be overwritten by **qn """
        ## interpret quantum numbers
        if iname is not None:
            qni = decode_level(iname)
        else:
            qni = {}
        if jname is not None:
            qnj = decode_level(jname)
        else:
            qnj = {}
        for key,val in qn.items():
            if key in self.key_data['qn']:
                qni[key] = qnj[key] = val
            elif len(key)>1 and key[-1]=='i' and key[:-1] in self.key_data['qn']:
                qni[key[:-1]] = val
            elif len(key)>1 and key[-1]=='j' and key[:-1] in self.key_data['qn']:
                qnj[key[:-1]] = val
            else:
                raise Exception(f"Could not interpret quantum number: {repr(key)}")
        i = self.match(**qni)
        j = self.match(**qnj)
        assert sum(i)==sum(j)
        for key in self.key_data['qn']:
            if self.is_set(key):
                self[key][i],self[key][j] = self[key][j],self[key][i]
        # self.format_input_functions.append(lambda: f'{self.name}.exchange_quantum_numbers({repr(qni)},{repr(qnj)},{my.repr_args_kwargs(qn_common)})')

    def clear_cache(self,*keys):
        """Delete entire cache, or specific caches if keys provided."""
        if len(keys)==0: keys = list(self._cache.keys())
        for key in keys:
            t = self._cache.pop(key); del(t)
        
    def limit_to_keys(self,*keys,keep_quantum_numbers=False):
        """Require all these keys to be set, and no others. Optionally keep
        all quantum numbers."""
        if keep_quantum_numbers:
            keys = set(list(keys)+list([t for t in self.key_data['qn'] if self.is_known(t)]))
        Dynamic_Recarray.limit_to_keys(self,*keys)

    # def all_data_keys(self):
        # return([key for key in self.all_keys() if self.is_data_key(key)])

    # def all_uncertainty_keys(self):
        # return([key for key in self.all_keys() if self.is_uncertainty_key(key)])

    # def is_quantum_number_key(self,key):
        # """Key is for a quantum number."""
        # return(key in self.key_data['qn'])
   #  
    # def is_uncertainty_key(self,key):
        # """If this key is the uncertainty of a vector data value."""
        # if key in self.scalar_keys():
            # return(False)
        # elif self.is_quantum_number_key(key):
            # return(False)
        # elif len(key)==1:
            # return(False)
        # elif key[0]!='d':
            # return(False)
        # elif key[1:] not in self.all_keys():
            # return(False)
        # else:
            # return(True)

    # def is_data_key(self,key):
        # """A vector data value (not an uncertainty)."""
        # if key in self.scalar_keys():
            # return(False)
        # elif self.is_quantum_number_key(key):
            # return(False)
        # elif self.is_uncertainty_key(key):
            # return(False)
        # else:
            # return(True)
            
    def average(
            self,
            data_keys=None, # what to average, defaults to all common data that is not a quantum number
            qn_keys=None, # common values are averaged, defaults to defining quantum number that exist in all data
            weighted_average=True, # uncertainties required
    ):
        """Look for identical levels and average keys that have
        uncertainties. Averages over common
        defining_qn. If uncertainties are defined they
        cannot be nan."""
        if len(self)==0: return # nothing to be done
        ## get quantum numbers defining common data, data keys to
        ## average and their uncertainties
        if data_keys is None:
            data_keys = [key for key in self.key_data['vector_data'] if self.is_known(key)]
        else:
            for key in data_keys:
                assert self.is_known(key),f'Unknown data key {repr(key)}'
        if qn_keys is None:
            qn_keys = [key for key in self.key_data['qn'] if self.is_known(key)]
        else:
            for key in qn_keys:
                assert self.is_known(key),f'Unknown quantum number key {repr(key)}'
        if weighted_average:
            uncertainty_keys = []
            for key in data_keys:
                dkey = 'd'+key
                assert self.is_known(dkey),f'Unknown uncertainty key {repr(dkey)}, cannot computed weighted_average.'
                uncertainty_keys.append(dkey)
        ## unset all other data
        for key in self.set_keys():
            if (key in self.scalar_keys()
                or key in qn_keys
                or key in data_keys
                or (weighted_average and key in uncertainty_keys)):
                continue
            else:
                self.unset(key)
        ## loop through sets of common quantum numbers
        for d in self.unique_dicts(*qn_keys):
            ## look for groups of matching quantum number
            j = self.match(**d) # bool array
            i = my.find(j)      # indices
            if len(i)<2: continue # does not need averaging
            ## average each variable requested -- if uncertainties are
            ## known use a weighted mean. HACK: IF UNCERTAINTIES ARE ALL ZERO THEN DO UNWEIGHTED MEAN
            for key in data_keys:
                dkey = 'd'+key
                if weighted_average:
                    assert self.is_known(dkey),          f'Uncertainty not known: {repr(dkey)}'
                    assert np.all(~np.isnan(self[dkey])),f'Uncertainty contains NaN: {repr(dkey)}'
                    assert not np.any(self[dkey]<0),         f'Uncertainty negative: {repr(dkey)}'
                    ## if all errors are zero unweighted mean, if
                    ## partially zero raise error 
                    if np.all(self[dkey]>0):
                        self[i[0]][key],self[i[0]][dkey] = my.weighted_mean(self[key][i], self[dkey][i], error_on_nan_zero= True)
                    else:
                        if np.any(self[dkey]==0):
                            assert np.all(self[dkey]==0),    f'Some uncertainty zero, but not all: {repr(dkey)}'
                            self[i[0]][key] = np.mean(self[key][i])
                            self[i[0]][dkey] = 0.
                else:
                    self[i[0]][key] = np.mean(self[key][i])
            ## delete levels other than the newly averaged one
            j[i[0]] = False
            self.remove(j)
        # ## clear data on averagable data not to be averaged -- that is
        # ## if it has an uncertainty but is not listed in
        # ## keys_to_average
        # for key in self.all_keys():
            # if key in keys_to_average: continue # should be averaged
            # if 'd'+key not in self.all_keys(): continue # must be a quantum number -- do not delete
            # if not self.is_set(key): continue # not set anyway
            # if key[0]=='d' and len(key)>1 and key[1:] in keys_to_average: continue # uncertainty should be averaged
            # # self[key] = self.vector_data[key]['default']
            # # self.vector_data[key]['is_set'] = False
            # self.unset(key)

    def reduce_common_data(self,*keys):
        """If there are multiple entries with identical quantum numbers, then
        reduce to the first occurrence. Defaults to matching
        defining_qn."""
        if len(keys)==0:
            keys = [key for key in self.defining_qn if self.is_known(key)] 
        for d in self.unique_dicts(*keys):
            ## look for groups of matching quantum number
            j = self.match(**d) # bool array
            i = my.find(j)      # indices
            if len(i)<2: continue # does not need reduction
            ## reduce to first occurence
            j[i[0]] = False
            self.remove(j)

    def _cast_data_identifier(self,x):
        """Cast data_identifier. If string convert into a list of strings,
        e.g., ['101203-F','160203-M',...]. Else cast to list."""
        if isinstance(x,str):
            retval = [t for t in x.replace('[','').replace(']','').replace('\'','').replace('"','').replace(' ','').split(',')]
        else:
            retval = list(x)
        return(retval)

