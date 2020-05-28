from spectr.dataset import Dataset
from spectr import infer

class Level(Dataset):
    """For now a rotational level."""

    _class_prototypes = {
        'class':{'description':"What kind of data this is.",'kind':'str', 'infer_functions':(('self',lambda self:self.__class__.__name__),),},
        'description':{'kind':str,'description':"",},
        'notes':{'description':"Notes regarding this line.", 'kind':str, },
        'author':{'description':"Author of data or printed file", 'kind':str, },
        'reference':{'description':"", 'kind':str, },
        'date':{'description':"Date data collected or printed", 'kind':str, },
        'species':{'description':"Chemical species",
                   'infer':((('encoded',),infer.species_from_encoded_level,None),)},
        
        }

    quantum_numbers = ('species','label','v','Σ','ef','J',
                        'S','Λ','LSsign','s','gu','Ihomo', 'group','term_symbol'
                        'sublevel','N','F','Ω','SR','sa','pm','g','σv','encoded')
    
    defining_quantum_numbers = ('species','label','v','Σ','ef','J')



        # more_other_qn_keys=['S','Λ','LSsign','s','gu','Ihomo', 'group','term_symbol'],
        # more_vector_data_keys = ['reduced_mass',
                                 # # 'Telec',
        # ],
        # more_scalar_other_keys = ['R'],)

        # more_defining_qn_keys=['v',],
        # more_vector_data_keys = ['Γv','τv','Atv','Adv','Aev',
                                 # 'ηdv','ηev',
                                 # 'Tv','Bv','Dv','Hv',
                                 # 'Av','ADv','AHv',
                                 # 'λv','λDv','λHv',
                                 # 'γv','γDv','γHv',
                                 # 'ov','oDv','oHv','oLv',
                                 # 'pv','qv',
                                 # 'pDv','qDv',
                                 # # 'Tvib',
                                 # 'Tvreduced','Tvreduced_common',],
        # more_vector_other_keys = ['χ',],
        # more_scalar_other_keys = ['Tvreduced_common_polynomial'])
    # default_plot_xkey = 'v'                    # if not specified, plot this on x-axis
    # default_plot_ykeys=('Tv','Bv')                 # if not specified, plot these on y-axis
    # default_sorting_order = ('species','label','v')
    # qn_defining_independent_vibrational_progressions = ('species','label') # make scalar data?


        # more_defining_qn_keys=['Σ','ef','J',],
        # more_other_qn_keys=['sublevel','N','F','Ω','SR','sa','pm','g','σv',],
        # more_vector_data_keys = ['T','Γ',
                                 # 'Treduced','Treduced_common','Λdoubling',
                                 # 'At','Ae','Ad','ηd',
                                 # # 'Trot',
        # ],
        # more_scalar_other_keys = ['Treduced_common_polynomial',])
    # default_plot_xkey = 'J'                    # if not specified, plot this on x-axis
    # default_plot_ykeys=('T','Γ')                 # if not specified, plot these on y-axis
    # default_plot_zkeys= ('species','label','v','Σ','Ω','ef')
    # default_sorting_order = ('species','label','v','Ω','ef','J')
    # qn_defining_independent_rotational_progressions = ('species','label','v','Σ','ef')

   #  
    # # (_class_key_data, _class_scalar_data, _class_vector_data) = _generate_keys(
        # # more_vector_other_keys=('index','name',),
        # # more_scalar_other_keys = ['level_transition_type','description',
                                  # # 'date', 'author', 'reference',
                                  # # 'data_identifier','Tref','dTref',
                                  # # 'partition_source',],)

    # # # defining_qn = all_qn        # minimum list of unique quantum numbers
    # # default_sorting_order = None
    # # default_plot_xkey  = None                    # if not specified, plot this on x-axis
    # # default_plot_ykeys = (None,)                 # if not specified, plot these on y-axis
    # # default_plot_zkeys = ()                 # if not specified, plot these on separate subplot
    # # ## Key translation used by load_from_file_with_translation. If
    # # ## an element is a list then the 2nd element of implied data.
    # # _load_from_file_with_translation_dict = dict(
        # # # ex2=('example2',{'example1':52.}),
    # # )
    # # _decode_name_function = lambda self,name: {'example1':45} # Return a dictionary of quantum numbers decoded from one name.
    # # verbose = False                  # True to turn on extra output

    
    def __init__(
            self,
            name=None,
            **dataset_kwargs,
    ):
        """Default_name is decoded to give default values. Kwargs can be
        scalar data, further default values of vector data, or if vectors
        themselves will populate data arrays."""
            
        Dataset.__init__(self,**dataset_kwargs)
        self['class'] = type(self).__name__.lower()
        self.name = (name if name is not None else self['class'])
        self._prototypes = self._class_prototypes

    # def load_from_file(self,*args,**kwargs):
        # """Thin wrapper on Dynamic_Recarray.load_from_file to set
        # format_input_function."""
        # self.format_input_functions.append(
            # lambda: f'{self.name}.load_from_file({my.repr_args_kwargs(*args,**kwargs)})')
        # Dynamic_Recarray.load_from_file(self,*args,**kwargs)

    # def exchange_quantum_numbers(self,iname=None,jname=None,**qn):
        # """Exchange all quantum numbers for matching levels i and j. All keys
        # in qn that are quantum numbers will be used to match both i
        # and j. Quantum numbers with "i" appended will match i-levesl
        # and if "j" appended will match j-levels. If multiple levels
        # are swapped then they are aligned according to whatever
        # current sorting of data is. iname and jname are decoded to get
        # quantum numbesr, that will be overwritten by **qn """
        # ## interpret quantum numbers
        # if iname is not None:
            # qni = decode_level(iname)
        # else:
            # qni = {}
        # if jname is not None:
            # qnj = decode_level(jname)
        # else:
            # qnj = {}
        # for key,val in qn.items():
            # if key in self.key_data['qn']:
                # qni[key] = qnj[key] = val
            # elif len(key)>1 and key[-1]=='i' and key[:-1] in self.key_data['qn']:
                # qni[key[:-1]] = val
            # elif len(key)>1 and key[-1]=='j' and key[:-1] in self.key_data['qn']:
                # qnj[key[:-1]] = val
            # else:
                # raise Exception(f"Could not interpret quantum number: {repr(key)}")
        # i = self.match(**qni)
        # j = self.match(**qnj)
        # assert sum(i)==sum(j)
        # for key in self.key_data['qn']:
            # if self.is_set(key):
                # self[key][i],self[key][j] = self[key][j],self[key][i]
        # # self.format_input_functions.append(lambda: f'{self.name}.exchange_quantum_numbers({repr(qni)},{repr(qnj)},{my.repr_args_kwargs(qn_common)})')

    # def clear_cache(self,*keys):
        # """Delete entire cache, or specific caches if keys provided."""
        # if len(keys)==0: keys = list(self._cache.keys())
        # for key in keys:
            # t = self._cache.pop(key); del(t)
       #  
    # def limit_to_keys(self,*keys,keep_quantum_numbers=False):
        # """Require all these keys to be set, and no others. Optionally keep
        # all quantum numbers."""
        # if keep_quantum_numbers:
            # keys = set(list(keys)+list([t for t in self.key_data['qn'] if self.is_known(t)]))
        # Dynamic_Recarray.limit_to_keys(self,*keys)

    # # def all_data_keys(self):
        # # return([key for key in self.all_keys() if self.is_data_key(key)])

    # # def all_uncertainty_keys(self):
        # # return([key for key in self.all_keys() if self.is_uncertainty_key(key)])

    # # def is_quantum_number_key(self,key):
        # # """Key is for a quantum number."""
        # # return(key in self.key_data['qn'])
   # #  
    # # def is_uncertainty_key(self,key):
        # # """If this key is the uncertainty of a vector data value."""
        # # if key in self.scalar_keys():
            # # return(False)
        # # elif self.is_quantum_number_key(key):
            # # return(False)
        # # elif len(key)==1:
            # # return(False)
        # # elif key[0]!='d':
            # # return(False)
        # # elif key[1:] not in self.all_keys():
            # # return(False)
        # # else:
            # # return(True)

    # # def is_data_key(self,key):
        # # """A vector data value (not an uncertainty)."""
        # # if key in self.scalar_keys():
            # # return(False)
        # # elif self.is_quantum_number_key(key):
            # # return(False)
        # # elif self.is_uncertainty_key(key):
            # # return(False)
        # # else:
            # # return(True)
           #  
    # def average(
            # self,
            # data_keys=None, # what to average, defaults to all common data that is not a quantum number
            # qn_keys=None, # common values are averaged, defaults to defining quantum number that exist in all data
            # weighted_average=True, # uncertainties required
    # ):
        # """Look for identical levels and average keys that have
        # uncertainties. Averages over common
        # defining_qn. If uncertainties are defined they
        # cannot be nan."""
        # if len(self)==0: return # nothing to be done
        # ## get quantum numbers defining common data, data keys to
        # ## average and their uncertainties
        # if data_keys is None:
            # data_keys = [key for key in self.key_data['vector_data'] if self.is_known(key)]
        # else:
            # for key in data_keys:
                # assert self.is_known(key),f'Unknown data key {repr(key)}'
        # if qn_keys is None:
            # qn_keys = [key for key in self.key_data['qn'] if self.is_known(key)]
        # else:
            # for key in qn_keys:
                # assert self.is_known(key),f'Unknown quantum number key {repr(key)}'
        # if weighted_average:
            # uncertainty_keys = []
            # for key in data_keys:
                # dkey = 'd'+key
                # assert self.is_known(dkey),f'Unknown uncertainty key {repr(dkey)}, cannot computed weighted_average.'
                # uncertainty_keys.append(dkey)
        # ## unset all other data
        # for key in self.set_keys():
            # if (key in self.scalar_keys()
                # or key in qn_keys
                # or key in data_keys
                # or (weighted_average and key in uncertainty_keys)):
                # continue
            # else:
                # self.unset(key)
        # ## loop through sets of common quantum numbers
        # for d in self.unique_dicts(*qn_keys):
            # ## look for groups of matching quantum number
            # j = self.match(**d) # bool array
            # i = my.find(j)      # indices
            # if len(i)<2: continue # does not need averaging
            # ## average each variable requested -- if uncertainties are
            # ## known use a weighted mean. HACK: IF UNCERTAINTIES ARE ALL ZERO THEN DO UNWEIGHTED MEAN
            # for key in data_keys:
                # dkey = 'd'+key
                # if weighted_average:
                    # assert self.is_known(dkey),          f'Uncertainty not known: {repr(dkey)}'
                    # assert np.all(~np.isnan(self[dkey])),f'Uncertainty contains NaN: {repr(dkey)}'
                    # assert not np.any(self[dkey]<0),         f'Uncertainty negative: {repr(dkey)}'
                    # ## if all errors are zero unweighted mean, if
                    # ## partially zero raise error 
                    # if np.all(self[dkey]>0):
                        # self[i[0]][key],self[i[0]][dkey] = my.weighted_mean(self[key][i], self[dkey][i], error_on_nan_zero= True)
                    # else:
                        # if np.any(self[dkey]==0):
                            # assert np.all(self[dkey]==0),    f'Some uncertainty zero, but not all: {repr(dkey)}'
                            # self[i[0]][key] = np.mean(self[key][i])
                            # self[i[0]][dkey] = 0.
                # else:
                    # self[i[0]][key] = np.mean(self[key][i])
            # ## delete levels other than the newly averaged one
            # j[i[0]] = False
            # self.remove(j)
        # # ## clear data on averagable data not to be averaged -- that is
        # # ## if it has an uncertainty but is not listed in
        # # ## keys_to_average
        # # for key in self.all_keys():
            # # if key in keys_to_average: continue # should be averaged
            # # if 'd'+key not in self.all_keys(): continue # must be a quantum number -- do not delete
            # # if not self.is_set(key): continue # not set anyway
            # # if key[0]=='d' and len(key)>1 and key[1:] in keys_to_average: continue # uncertainty should be averaged
            # # # self[key] = self.vector_data[key]['default']
            # # # self.vector_data[key]['is_set'] = False
            # # self.unset(key)

    # def reduce_common_data(self,*keys):
        # """If there are multiple entries with identical quantum numbers, then
        # reduce to the first occurrence. Defaults to matching
        # defining_qn."""
        # if len(keys)==0:
            # keys = [key for key in self.defining_qn if self.is_known(key)] 
        # for d in self.unique_dicts(*keys):
            # ## look for groups of matching quantum number
            # j = self.match(**d) # bool array
            # i = my.find(j)      # indices
            # if len(i)<2: continue # does not need reduction
            # ## reduce to first occurence
            # j[i[0]] = False
            # self.remove(j)

    # def _cast_data_identifier(self,x):
        # """Cast data_identifier. If string convert into a list of strings,
        # e.g., ['101203-F','160203-M',...]. Else cast to list."""
        # if isinstance(x,str):
            # retval = [t for t in x.replace('[','').replace(']','').replace('\'','').replace('"','').replace(' ','').split(',')]
        # else:
            # retval = list(x)
        # return(retval)

