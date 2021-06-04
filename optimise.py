from time import perf_counter as timestamp
import inspect
import re
import os
# import sys
from pprint import pprint,pformat
from copy import copy,deepcopy
import warnings
import types
import multiprocessing
import functools

import numpy as np
from numpy import nan,inf,array,arange,linspace
from scipy import optimize,linalg
from immutabledict import immutabledict as idict

# from . import dataset
from . import tools
from .tools import ensure_iterable
from . import plotting

class CustomBool():
    """Create Boolean alternatives to True and False."""
    def __init__(self,name,value):
        self.name = name
        self.value = bool(value)
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name
    def __bool__(self):
        return self.value

## False, and not intended to be changed
Fixed = CustomBool('Fixed',False)

def _collect_parameters_and_optimisers(x):
    """Iteratively and recursively collect Parameter and Optimiser objects
    from x descending into any iterable children."""
    maximum_length_for_searching = 1000
    parameters,optimisers = [],[]
    if isinstance(x,Parameter):
        parameters.append(x)
    elif isinstance(x,Optimiser):
        optimisers.append(x)
    elif isinstance(x,list) and len(x) < maximum_length_for_searching:
        for t in x:
            tp,to = _collect_parameters_and_optimisers(t)
            parameters.extend(tp)
            optimisers.extend(to)
    elif isinstance(x,dict):
        for t in x.values():
            tp,to = _collect_parameters_and_optimisers(t)
            parameters.extend(tp)
            optimisers.extend(to)
    return parameters,optimisers

def optimise_method(
        add_construct_function=True,
        add_format_input_function=True,
        execute_now= True,
        format_lines='auto',    # 'auto','single','multi
):
    """A decorator factory for automatically adding parameters,
    construct_function, and input_format_function to a method in an
    Optimiser subclass.  The undecorated method must return any
    resiudal that you want added to the Optimsier. Parameters and
    suboptimisers are picked out of the input kwargs.\n\n Input
    arguments format_single_line and format_multi_line override
    default format_input_function properties. The only reason to write
    a factory is to accomodate these arguments, and maybe future ones.
    Not that new_function does not actually run function!  Instead it
    is run by add_construct_function and construct."""
    def actual_decorator(function):
        @functools.wraps(function)
        def new_function(self,*args,**kwargs):
            ## this block subtitutes into kwargs with keys taken from
            ## the function signature.  get signature arguments -- skip
            ## first "self"
            signature_keys = list(inspect.signature(function).parameters)[1:]
            for iarg,(arg,signature_key) in enumerate(zip(args,signature_keys)):
                if signature_key in kwargs:
                    raise Exception(f'Positional argument also appears as keyword argument {repr(signature_key)} in function {repr(function.__name__)}.')
                kwargs[signature_key] = arg
            ## add parameters suboptimisers in args to self
            parameters,suboptimisers =  _collect_parameters_and_optimisers(kwargs)
            for t in parameters:
                self.add_parameter(t)
            for t in suboptimisers:
                self.add_suboptimiser(t)
                self.pop_format_input_function()
            ## if '_cache' is a kwarg of the function then initialise
            ## as an empty dictionary
            if '_cache' in signature_keys:
                kwargs['_cache'] = {}
            ## if '_parameters' or _suboptimisers are kwarg of the
            ## function then provide as lists
            if '_parameters' in signature_keys:
                kwargs['_parameters'] = parameters
            if '_suboptimisers' in signature_keys:
                kwargs['_suboptimisers'] = suboptimisers
            ## make a construct function which returns the output of
            ## the original method            if add_construct_function:
            if add_construct_function:
                construct_function = lambda kwargs=kwargs: function(self,**kwargs)
                construct_function.__name__ = function.__name__+'_optimisation_method'
                self.add_construct_function(construct_function)
            ## execute the function now
            if execute_now:
                function(self,**kwargs)
            ## make a format_input_function
            if add_format_input_function:
                def f():
                    kwargs_to_format = {key:val for key,val in kwargs.items() if key[0] != '_' and val is not None}
                    if format_lines=='single' or (format_lines=='auto' and len(kwargs_to_format)<2):
                        formatted_kwargs = ','.join([f"{key}={repr(val)}" for key,val in kwargs_to_format.items()])
                        return f'{self.name}.{function.__name__}({formatted_kwargs},)'
                    elif format_lines=='multi' or (format_lines=='auto' and len(kwargs_to_format)>=2):
                        formatted_kwargs = ',\n    '.join([f"{key:10} = {repr(val)}" for key,val in kwargs_to_format.items()])
                        return f'{self.name}.{function.__name__}(\n    {formatted_kwargs},\n)'
                    else:
                        raise Exception(f"Bad format_line ({repr(format_line)}) valid values are 'auto', 'single', 'multi'")
                self.add_format_input_function(f)
            ## returns all args as a dictionary except added
            ## optimsation variables
            retval = {key:val for key,val in kwargs.items() if key not in ('_cache','_parameters','_suboptimisers')}
            return retval
        return new_function
    return actual_decorator

class Optimiser:
    """Defines adjustable parameters and model-building functions which
    may return a residual error. Then optimise the parameters with
    respect to these residuals. Can contain suboptimisers which are
    simultaneously optimised and dependencies of this one. """
    
    def __init__(
            self,
            name='optimiser',           # used to generate evaluable python code
            *suboptimisers,     # Optimisers
            verbose=False,
            description=None,
            **named_parameters,
    ):
        """Suboptimisers can be other Optimiser that are also
        optimised with this one."""
        assert isinstance(name,str),'Name must be a string.'
        # self.name = tools.regularise_string_to_symbol(name) # for generating evaluatable references to self
        self.name = name # for generating evaluatable references to self
        self.residual_scale_factor = 1 # scale all resiudal by this amount -- useful when combining with other optimisers
        self.parameters = []           # Data objects
        self._ncpus = 1
        self._construct_functions = {} # list of function which collectively construct the model and optionally return a list of fit residuals
        self._post_construct_functions = {} # run in reverse order after other construct functions
        self._plot_functions = [] 
        self._monitor_functions = [] # call these functions at special times during optimisation
        self._monitor_frequency = 'every iteration' # when to call monitor fuctions: 'never', 'every iteration', 'rms decrease', 'significant rms decrease'
        self._monitor_frequency_significant_rms_decrease_fraction = 1e-2
        self._monitor_parameters = False            # print out each on monitor
        self._save_to_directory_functions = [] # call these functions when calling save_to_directory (input argument is the directory)
        self._format_input_functions = {}        # call these functions (return strings) to build a script recreating an optimisation
        self._suboptimisers = list(suboptimisers) # Optimiser objects optimised with this one, their construct functions are called first
        self.verbose = verbose                     # True to activate various informative outputs
        self._clean_construct = True                 # a flag to construct functions with caches to rebuild them 
        self.residual = None # array of residual errors generated by construct functions
        self.combined_residual = None # array of residual errors generated by construct functions of this and its suboptimisers
        self.description = description
        self._last_construct_time = timestamp() # when last constructed
        ## add named parameters addressable as optimiser[key]
        self._named_parameters = {}
        for key,val in named_parameters.items():
            if isinstance(val,Parameter):
                val = (val.value,val.vary,val.step)
            self.add_named_parameter(key,*ensure_iterable(val))
        ## make an input line
        def f():
            retval = f'{self.name} = Optimiser(name={repr(self.name)},'
            if len(suboptimisers)>0:
                retval += ','.join([repr(t) for t in suboptimisers])+','
            if self.verbose:
                retval += f'verbose={repr(self.verbose)},'
            if self.description is not None:
                retval += f'description={repr(self.description)},'
            if len(self._named_parameters) > 0:
                retval += '\n'
                for key,val in self._named_parameters.items():
                    retval += f'    {key}={str(val)},\n'
            retval += ')'
            return retval
        self.add_format_input_function(f)

    def __repr__(self):
        """No attempt to represent this data but its name may be used in place
        of it."""
        return self.name

    def add_format_input_function(self,*functions):
        """Add a new format input function."""
        for function in functions:
            self._format_input_functions[timestamp()] = function

    def pop_format_input_function(self):
        """Delete last format input function added."""
        key = list(self._format_input_functions.keys())[-1]
        return self._format_input_functions.pop(key)

    def automatic_format_input_function(self,limit_to_args=None,multiline=False):
        """Try to figure one out from any non None variables. Could
            easily fail."""
        caller_locals = inspect.currentframe().f_back.f_locals
        ## get e.g., spectrum.Model, keeping only last submodule name
        class_name =  caller_locals['self'].__class__.__name__
        submodule_name = re.sub('.*[.]','', caller_locals['self'].__class__.__module__)
        def f():
            args = []
            for key,val in caller_locals.items():
                if ((key == 'self')
                    or (val is None)
                    or (limit_to_args is not None and key not in limit_to_args)):
                    continue
                args.append(f'{key}={repr(val)}')
            if multiline:
                retval = f'{caller_locals["name"]} = {submodule_name}.{class_name}(\n    '+'\n    '.join(args)+'\n)'
            else:
                retval = f'{caller_locals["name"]} = {submodule_name}.{class_name}('+','.join(args)+')'
            return retval
        self.add_format_input_function(f)

    def add_suboptimiser(self,*suboptimisers,add_format_function=True):
        """Add one or suboptimisers."""
        ## add only if not already there
        for t in suboptimisers:
            if t not in self._suboptimisers:
                self._suboptimisers.append(t)
        if add_format_function:
            self.add_format_input_function(
                lambda: f'{self.name}.add_suboptimiser({",".join(t.name for t in suboptimisers)})')

    suboptimisers = property(lambda self: self._suboptimisers)

    def add_parameter(self,parameter,*args,**kwargs):
        """Add one parameter. Return a reference to it. Args are as in
        pP or one are is a P."""
        if (isinstance(parameter,Named_Parameter)
            and parameter.optimiser not in self._suboptimisers):
            self.add_suboptimiser(parameter.optimiser)
        if not isinstance(parameter,Parameter):
            parameter = Parameter(*tools.ensure_iterable(parameter),*args,**kwargs)
        self.parameters.append(parameter)
        return parameter

    def add_named_parameter(self,key,value,*args,**kwargs):
        """Add one parameter. Return a reference to it. Args are as in
        Parameter."""
        parameter = Named_Parameter(self,key,value,*args,**kwargs)
        self.parameters.append(parameter)
        self._named_parameters[key] = parameter
        return parameter

    def __getitem__(self,key):
        return self._named_parameters[key]

    def __setitem__(self,key,val):
        self._named_parameters[key].value = val

    def __iter__(self):
        for key in self._named_parameters:
            yield key

    def keys():
        return self._named_parameters.keys()

    def add_construct_function(self,*functions):
        """Add one or more functions that are called each iteration when the
        model is optimised. Optionally these may return an array that is added
        to the list of residuals."""
        for f in functions:
            self._construct_functions[timestamp()] = f
        self._clean_construct = True 

    def add_post_construct_function(self,*functions):
        """Add one or more functions that is run after normal construct
        functions. Run in the reverse order in which they were added."""
        for f in functions:
            self._post_construct_functions[timestamp()] = f
        self._clean_construct = True 

    def add_monitor_function(self,*functions):
        """Add one or more functions that are called when a new minimum
        is found in the optimisation. This might be useful for saving
        intermediate fits in case of failure or abortion."""
        self._monitor_functions.extend(functions)
        
    def add_save_to_directory_function(self,*functions):
        self._save_to_directory_functions.extend(functions)
        
    def add_plot_function(self,*functions):
        """Add a new format input function."""
        self._plot_functions.extend(functions)

    def get_all_suboptimisers(self,_already_recursed=None,include_self=True):
        """Return a list of all suboptimisers including self and without double
        counting. Ordered so suboptimisers always come first."""
        if _already_recursed is None:
            _already_recursed = [self] # first
        else:
            _already_recursed.append(self)
        retval = []
        for optimiser in self._suboptimisers:
            if optimiser in _already_recursed:
                continue
            else:
                _already_recursed.append(optimiser)
                retval.extend(optimiser.get_all_suboptimisers(_already_recursed))
        if include_self:
            retval.append(self)     # put self last
        return retval 

    def get_all_parameters(self):
        """Return a list of all parameter objects."""
        parameters = []
        for o in self.get_all_suboptimisers():
            for p in o.parameters:
                if p not in parameters:
                    parameters.append(p)
        return parameters

    def format_input(self,match_lines_regexp=None):
        """Join strings which should make an exectuable python script for
        repeating this optimisation with updated parameters. Each element of
        self.format_input_functions should be a string or a function of no
        arguments evaluating to a string."""
        ## collect all format_input_functions
        timestamps,functions,suboptimisers = [],[],[]
        for optimiser in self.get_all_suboptimisers():
            timestamps.extend(optimiser._format_input_functions.keys())
            functions.extend(optimiser._format_input_functions.values())
            suboptimisers.extend([optimiser for t in optimiser._format_input_functions])
        ## evaluate input lines sorted by timestamp
        lines = []
        lines.append('from spectr import *\n') # general import at beginning of formatted input
        previous_suboptimiser = None
        for i in np.argsort(timestamps):
            ## separate with newlines if a new suboptimser
            if (previous_suboptimiser is not None
                and suboptimisers[i] is not previous_suboptimiser):
                lines.append('')
            lines.append(functions[i]())
            previous_suboptimiser = suboptimisers[i]
        retval = '\n'.join(lines)
        return retval

    def print_input(self,match_lines_regexp=None):
        """Print recreated input function. Filter lines by regexp if
        desired."""
        lines = self.format_input()
        if match_lines_regexp is None:
            print(lines)
        else:
            for line in lines.split('\n'):
                if re.match(match_lines_regexp,line):
                    print(line)

    def save_input(self,filename=None):
        """Save recreated input function to a file."""
        tools.string_to_file(filename,self.format_input())

    def __str__(self):
        return self.format_input()

    def save_to_directory(self,directory):
        """Save results of model and optimisation to a directory. WILL
        FIRST DELETE CONTENTS OF DIRECTORY!"""
        directory = tools.expand_path(directory)
        tools.mkdir(directory,trash_existing=True)
        ## output self and all suboptimisers into a flat subdirectory
        ## structure
        names = []              # all names thus far
        for optimiser in self.get_all_suboptimisers():
            ## uniquify name -- COLLISION STILL POSSIBLE if name for
            ## three optimisers is xxx, xxx, and xxx_2. There will end
            ## up being two xxx_2 subdirectories.
            if optimiser.name in names:
                optimiser._unique_name = optimiser.name+'_'+str(names.count(optimiser.name)+1)
                print(f'Repeated optimiser name {repr(optimiser.name)} replaced with {repr(optimiser._unique_name)}')
            else:
                optimiser._unique_name = optimiser.name
            names.append(optimiser.name)
            ## output this suboptimiser
            subdirectory = directory+'/'+optimiser._unique_name
            tools.mkdir(subdirectory,trash_existing=False)
            tools.string_to_file(subdirectory+'/input.py',optimiser.format_input())
            if optimiser.residual is not None:
                tools.array_to_file(subdirectory+'/residual' ,optimiser.residual,fmt='%+0.4e')
            if optimiser.description is not None:
                tools.string_to_file(subdirectory+'/README' ,str(optimiser.description))
            for f in optimiser._save_to_directory_functions:
                f(subdirectory)
        ## symlink suboptimsers into subdirectories
        for optimiser in self.get_all_suboptimisers():
            for suboptimiser in optimiser._suboptimisers:
                tools.mkdir(f'{directory}/{optimiser._unique_name}/suboptimisers/')
                os.symlink(
                    f'../../{suboptimiser._unique_name}',
                    f'{directory}/{optimiser._unique_name}/suboptimisers/{suboptimiser._unique_name}',
                    target_is_directory=True)

    def plot_residual(self,ax=None,**plot_kwargs):
        """Plot residual error."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()
        plot_kwargs.setdefault('marker','o')
        previous_xend = 0
        for suboptimiser in self.get_all_suboptimisers():
            residual = suboptimiser.residual
            if residual is None or len(residual) == 0:
                continue
            x = np.arange(previous_xend,previous_xend+len(residual))
            ax.plot(x, residual, label=suboptimiser.name, **plot_kwargs)
            previous_xend = x[-1]+1
            plotting.legend(ax=ax)

    def plot(self,first_figure_number=1):
        """Plot all plot functions in separate figures."""
        import matplotlib.pyplot as plt
        for suboptimiser in self.get_all_suboptimisers():
            for function in suboptimiser._plot_functions:
                fig = plt.figure(first_figure_number)
                fig.clf()
                function()
                first_figure_number += 1
        
    def monitor(self):
        """Run monitor functions."""
        for optimiser in self.get_all_suboptimisers():
            for f in optimiser._monitor_functions:
                f()

    def _get_parameters(self):
        """Collect all varied parameters for optimisation. These are either in
        self.parameters, in self._data if this is also a Dataset or found in
        suboptimiser."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        value,unc,step = [],[],[]
        unique_parameters = []  # to prevent the same parmeter being loaded twice from suboptimisers
        for optimiser in self.get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if id(parameter) not in unique_parameters and parameter.vary:
                    value.append(parameter.value)
                    unc.append(parameter.step)
                    step.append(parameter.step)
                    unique_parameters.append(id(parameter))
            if isinstance(optimiser,Dataset):
                for key in optimiser.optimised_keys():
                    vary = optimiser.get((key,'vary'))
                    value.extend(optimiser[key][vary])
                    unc.extend(optimiser.get((key,'unc'),index=vary))
                    step.extend(optimiser.get((key,'step'),index=vary))
        return value,step,unc

    def _set_parameters(self,p,dp=None,rescale=False):
        """Set assign elements of p and dp from optimiser to the
        correct Parameters."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        p = list(p)
        ## shift and scale optimised parameter back to its original scale
        if rescale:
            p = [tp*tsi/1e-8+tpi for tp,tpi,tsi in zip(p,self._initial_p,self._initial_step)]
            if dp is not None:
                dp = [tdp*tsi/1e-8 for tdp,tsi in zip(dp,self._initial_step)]
        ## print parameters
        if self._monitor_parameters:
            print('    monitor parameters:  ['+' ,'.join([format(t,'+#15.13e') for t in p])+' ]')
        ## deal with missing dp
        if dp is None:
            dp = [np.nan for pi in p]
        else:
            dp = list(dp)
        already_set = []
        for optimiser in self.get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if id(parameter) not in already_set and parameter.vary:
                    parameter.value = p.pop(0)
                    parameter.unc = dp.pop(0)
                    already_set.append(id(parameter))
            if isinstance(optimiser,Dataset):
                for key in optimiser.optimised_keys():
                    vary = optimiser.get((key,'vary'))
                    ## could speed up using slice rather than pop?
                    for i in tools.find(vary):
                        optimiser.set(key,p.pop(0),index=i)
                        optimiser.set((key,'unc'),dp.pop(0),index=i)

    construct_functions = property(lambda self: list(self._construct_functions.values()) + list(self._post_construct_functions.values())[::-1])

    def construct(self,clean_construct=False):
        """Run all construct functions and return collected residuals."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        ## collect residuals from suboptimisers and self
        combined_residual = []  # from self and suboptimisers
        for o in self.get_all_suboptimisers():
            if clean_construct:
                o._clean_construct = True
            ## construct optimiser for one of the following reasons
            if (
                    ## clean construct
                    o._clean_construct
                    ## one of its parameters has changed
                    or any([p._last_modify_value_time > o._last_construct_time for p in o.parameters])
                    ## it is a Dataset and its data has changed
                    or (isinstance(o,Dataset) and o._last_modify_data_time > o._last_construct_time)
                    ## one of its suboptimisers has been reconstructed
                    or any([to._last_construct_time > o._last_construct_time for to in o.get_all_suboptimisers(include_self=False)])
            ):
                if self.verbose:
                    print(f'constructing optimiser: {o.name}')
                ## collect all construct functions, if there are any,
                ## then run them, save the residual, and mark the
                ## construct time
                o.residual = []
                construct_functions = o.construct_functions
                if len(construct_functions) > 0:
                    for f in construct_functions:
                        t = f()                # run construction function, get residual if there is one
                        if t is None:
                            continue
                        if np.isscalar(t):
                            t = [t]
                        if len(t)==0:
                            continue 
                        o.residual.append(t) # add residual to overall residual
                    ## combine construct function residuals into one
                    if o.residual is not None and len(o.residual)>0:
                        o.residual = np.concatenate(o.residual)
                ## record time of construction
                o._last_construct_time = timestamp()
            ## remove any clean construct mark
            o._clean_construct = False
            ## add residual to return value for optimisation, possibly rescaling it
            if o.residual is not None:
                combined_residual.append(o.residual_scale_factor*np.array(o.residual))
        combined_residual = np.concatenate(combined_residual)  # includes own residual and for all suboptimisers
        self.combined_residual = combined_residual # this includes residuals from construct_functions combined with suboptimisers
        return combined_residual

    def _optimisation_function(self,p,rescale=True):
        """Internal function used by optimise routine. p is a list of varied
        parameters."""
        self._number_of_optimisation_function_calls += 1
        ## update parameters in internal model
        self._set_parameters(p,rescale=rescale)
        ## rebuild model and calculate residuals
        residuals = self.construct()
        if len(residuals) == 0:
            raise Exception("No residuals to optimise.")
        ## monitor
        rms = tools.rms(residuals)
        if np.isinf(rms) or np.isnan(rms):
            raise Exception(f'rms is {rms}')
        if (
                ## every iteration
                (self._monitor_frequency=='every iteration')
                ## every time rms decreases
                or (self._monitor_frequency=='rms decrease'
                    and rms < self._rms_minimum)
                ## every time rms decreases by at least self._monitor_frequency_significant_rms_decrease_fraction
                or (self._monitor_frequency=='significant rms decrease'
                    and abs(self._rms_minimum-rms)/rms > self._monitor_frequency_significant_rms_decrease_fraction)
        ):
            self.monitor()
        ## print rms
        current_time = timestamp()
        if self._monitor_iterations:
            print(f'call: {self._number_of_optimisation_function_calls:>6d} time: {current_time-self._previous_time:<7.2e} rms: {rms:<12.8e}')
        self._previous_time = current_time
        if rms < self._rms_minimum:
            self._rms_minimum = rms
        return residuals

    @optimise_method(add_construct_function=False)
    def optimise(
            self,
            # compute_unc_only=False, # do not optimise -- just compute uncertainty at current position, actually does one iteration
            # rms_noise=None,
            verbose=True,
            # normalise_suboptimiser_residuals=False,
            method=None,
            least_squares_options=idict(),
            ncpus=1,            # Controls the use of multiprocessing of the Jacobian
            monitor_iterations=True, # print rms evey iteration
            monitor_parameters=False, # print parameters every iteration
            monitor_frequency='significant rms decrease', # run monitor functions 'never', 'end', 'every iteration', 'rms decrease', 'significant rms decrease'
    ):
        """Optimise parameters."""
        self._ncpus = ncpus
        # if normalise_suboptimiser_residuals:
            # ## normalise all suboptimiser residuals before handing to
            # ## the least-squares routine
            # for suboptimiser in self.get_all_suboptimisers():
                # self.construct(reconstruct_everything=True)
                # if suboptimiser.residual is not None and len(suboptimiser.residual)>0:
                    # if suboptimiser.residual_scale_factor is None:
                        # suboptimiser.residual_scale_factor = 1/tools.rms(suboptimiser.residual)
                    # else:
                        # suboptimiser.residual_scale_factor /= tools.rms(suboptimiser.residual)
        ## get initial values and reset uncertainties
        self._initial_p,self._initial_step,self._initial_dp = self._get_parameters()
        ## full reconstruct -- then set to efficient partial reconstruct
        self.construct(clean_construct=True)
        ## some communication variables between methods to do with
        ## monitoring the optimisation
        self._monitor_frequency = monitor_frequency
        self._monitor_parameters = monitor_parameters
        self._monitor_iterations = monitor_iterations
        valid_monitor_frequency = ('never','end','every iteration','rms decrease','significant rms decrease')
        if monitor_frequency not in valid_monitor_frequency:
            raise Exception(f'Invalid monitor_frequency, choose from: {repr(valid_monitor_frequency)}')
        self._rms_minimum,self._previous_time = inf,timestamp()
        self._number_of_optimisation_function_calls = 0
        ## describe the upcoming optimisation
        if verbose or self.verbose:
            print(f'\n{self.name}: optimising')
            print(f'number of varied parameters: {len(self._initial_p)}')
        ## the optimisation
        if len(self._initial_p)>0:
            least_squares_options = {
                'fun':self._optimisation_function,
                'x0':[0. for t in self._initial_p],
                'diff_step':np.full(len(self._initial_p),1e-8),
                ## 'x0':copy(self._initial_p),
                ## 'diff_step':copy(self._initial_step),
                ## 'diff_step':1e-8,
                # 'xtol':(1e-10 if xtol is None else xtol),
                # 'ftol':(1e-10 if ftol is None else ftol),
                # 'gtol':(1e-10 if gtol is None else gtol),
                # 'max_nfev':max_nfev,
                'method':None,
                'jac':'2-point',
                } | least_squares_options
            if least_squares_options['method'] is None:
                ## get a default method
                if len(self._initial_p) == 1:
                    # least_squares_options['method'] = 'lm'
                    least_squares_options['method'] = 'trf'
                else:
                    least_squares_options['method'] = 'trf'
            if least_squares_options['method'] == 'lm':
                ## use 'lm' Levenberg-Marquadt
                pass
            elif least_squares_options['method'] == 'trf':
                ## use 'trf' -- trust region
                least_squares_options['x_scale'] = 'jac'
                least_squares_options['loss'] = 'linear'
                if len(self._initial_p) < 5:
                    least_squares_options['tr_solver'] = 'exact'
                else:
                    least_squares_options['tr_solver'] = 'lsmr'
            else:
                raise Exception(f'Unknown optimsiation method: {repr(least_squares_options["method"])}')
            # least_squares_options['jac'] = self._calculate_jacobian
            # if self._ncpus > 1:
                # least_squares_options['jac'] = self._calculate_jacobian
            ## call optimisation routine -- KeyboardInterrupt possible
            try:
                if verbose or self.verbose:
                    print('method:',least_squares_options['method'])
                    # print("Optimisation parameters:")
                    # print('    '+pformat(least_squares_options).replace('\n','\n    '))
                result = optimize.least_squares(**least_squares_options)
                self._optimisation_function(result['x'])
                if verbose or self.verbose:
                    print('optimisation complete')
                    print('    number parameters:    ',len(result['x']))
                    print('    number of evaluations:',self._number_of_optimisation_function_calls)
                    print('    number of iterations: ',result['nfev'])
                    print('    termination reason:   ',result['message'])
            except KeyboardInterrupt:
                pass
            ## calculate uncertainties -- KeyboardInterrupt possible
            try:
                self.calculate_uncertainty()
            except KeyboardInterrupt:
                pass
        ## recalculate final solution
        residual = self.construct()
        if self._monitor_frequency != 'never':
            self.monitor() 
        ## describe result
        if verbose or self.verbose:
            print('total RMS:',np.sqrt(np.mean(np.array(self.combined_residual)**2)))
            for suboptimiser in self.get_all_suboptimisers():
                if (suboptimiser.residual is not None and len(suboptimiser.residual)>0):
                    print(f'suboptimiser {suboptimiser.name} RMS:',tools.rms(suboptimiser.residual))
        return residual

    def __deepcopy__(self,memo):
        """Deep copies paramters and construct_functions, preserving
        internal links of construct_function default arguments which
        are parameters to the copied parameters.  I HAVE NOT BOTHERED
        TO PROPERLY DEEPCOPY INPUT_FORMAT_FUNCTION,
        SAVE_TO_DIRECTORY_FUNCTIONS ETC. THIS COULD BE SIMILARLY TO
        CONSTRUCT_FUNCTIONS."""
        ## The top level optimiser is used to get a copy of all
        ## parameters. Suboptimisers get a copy of this list -- ONLY
        ## WORKS IF TOP LEVEL DEEPCOPY IS AN OPTIMISER
        if len(memo) == 0:
            self._copied_parameters = {}
            self._copied_suboptimisers = {}
            for o in self.get_all_suboptimisers():
                o._copied_parameters = self._copied_parameters
                o._copied_suboptimisers = self._copied_suboptimisers
        ## shallow copy everything
        retval = copy(self)
        ## deepcopy parameters
        for i,t in enumerate(self.parameters):
            if id(t) not in self._copied_parameters:
                self._copied_parameters[id(t)] = deepcopy(t,memo)
            self.parameters[i] = self._copied_parameters[id(t)]
        ## deepcopy suboptimisers
        for i,t in enumerate(self._suboptimisers):
            if id(t) not in self._copied_suboptimisers:
                self._copied_suboptimisers[id(t)] = deepcopy(t,memo)
            self._suboptimisers[i] = self._copied_suboptimisers[id(t)]
        ## copy construct functions while updating references to parameters and suboptimisers
        translate_defaults = self._copied_parameters | self._copied_suboptimisers
        for key,function in self._construct_functions.items():
            self._construct_functions[key] = _deepcopy_function(function,translate_defaults)
        # for key,function in self._post_construct_functions.items():
            # self._post_construct_functions[key] = _deepcopy_function(function,translate_defaults)
        return retval

    def _calculate_jacobian(self,p,rescale=True):
        """Compute 1σ uncertainty by first computing forward-difference
        Jacobian.  Only accurate for a well-optimised model."""
        ## compute model at p
        self._set_parameters(p,rescale=True)
        residual = self.construct()
        rms = tools.rms(residual)
        ## compute Jacobian by forward finite differencing, if a
        ## difference is too small then it cannot be used to compute
        ## an uncertainty -- so only build a jacobian with differences
        ## above machine precisison.  ALTHERNATIVELY, COULD TRY AND
        ## INCREASE STEP SIZE AND RECALCULATE.
        # print(f'    Calculate Jacobian for {len(p)} parameters and {len(residual)} residual points.')
        if self._ncpus == 1:
            ## single thread
            jacobian = [] # jacobian columns and which parameter they correspond to
            pnew = list(p)
            step = 1e-8 # must agree with optimise
            for i in range(len(p)):
                timer = timestamp()
                pnew[i] += step
                self._set_parameters(pnew,rescale=True)
                residualnew = self.construct()
                dresidual = (residualnew-residual)
                jacobian.append(dresidual/step)
                pnew[i] = p[i] # change it back
                rmsnew = tools.rms(residualnew)
                if rms == rmsnew:
                    message = 'parameter has no effect'
                else:
                    message = ''
                if self._monitor_iterations:
                    print(f'jcbn: {i:>6d} time: {timestamp()-timer:>7.2e} rms: {rmsnew:>12.8e} {message}')
            jacobian = np.transpose(jacobian)
        else:
            ## multiprocessing, requires serialisation
            import dill
            manager = multiprocessing.Manager()
            shared_namespace = manager.Namespace()
            shared_namespace.dill_pickle = dill.dumps(self)
            shared_namespace.residual = residual
            with multiprocessing.Pool(self._ncpus) as pool:
                ibeg = 0
                results = []
                ishuffle = arange(len(p))
                np.random.shuffle(ishuffle)
                for n in range(self._ncpus):
                    # i = range(n,len(p),self._ncpus)
                    i = ishuffle[n::self._ncpus]
                    results.append(
                        pool.apply_async(
                            _calculate_jacobian_multiprocessing_worker,
                            args=(shared_namespace,p,i)))
                ## run proceses, tidy keyboard interrupt, sum to give full spectrum
                try:
                    pool.close()
                    pool.join()
                except KeyboardInterrupt as err:
                    pool.terminate()
                    pool.join()
                    raise err
                jacobian = np.empty((len(p),len(residual)))
                for n in range(self._ncpus):
                    # i = range(n,len(p),self._ncpus) 
                    i = ishuffle[n::self._ncpus]
                    jacobian[i,:] = results[n].get()
                jacobian = np.transpose(jacobian)
        ## set back to best fit
        self._set_parameters(p,rescale=True) # set param
        self.construct()
        # ## set zeros/nans to very small to prevent matrix inversion problems
        # i = (jacobian == 0)
        # if np.any(i):
            # print('Some parameters have no effect on the model' )
            # jacobian[i] = 1e-50
        # i = np.isnan(jacobian)
        # if np.any(i):
            # print('Some parameters nan' )
            # jacobian[i] = 1e-50
        ## return
        return jacobian

    def calculate_uncertainty(self,rms_noise=None,verbose=True,ncpus=None):
        """Compute 1σ uncertainty by first computing forward-difference
        Jacobian.  Only accurate for a well-optimised model."""
        ## whether or not to multiprocess
        if ncpus is not None:
            self._ncpus = ncpus
        ## get residual at current solugion
        self.construct(clean_construct=True)
        residual = self.combined_residual
        ## get current parameter
        self._initial_p,self._initial_step,self._initial_dp = self._get_parameters()
        ## get rescaled parameters -- all zero
        p = np.full(len(self._initial_p),0.0) 
        if verbose or self.verbose:
            print(f'{self.name}: computing uncertainty for {len(p)} parameters')
        ## get jacobian using rescaled parameters
        jacobian = self._calculate_jacobian(p)
        min_valid = 0
        while True:
            try: 
                inonzero = np.any(np.abs(jacobian)>min_valid,axis=0)
                t = np.sum(~inonzero)
                if t>0:
                    print(f'Jacobian is not invertible so discarding {t} our of {len(inonzero)} columns with no values greater than {min_valid:0.0e}.')
                ## compute 1σ uncertainty from Jacobian
                unc = np.full(len(p),nan)
                if len(jacobian) == 0 or np.sum(inonzero) == 0:
                    print('All parameters have no effect, uncertainties not calculated')
                else:
                    t = jacobian[:,inonzero]
                    covariance = linalg.inv(np.dot(t.T,t))
                    if rms_noise is None:
                        chisq = np.sum(residual**2)
                        dof = len(residual)-len(p)+1
                        rms_noise = np.sqrt(chisq/dof)
                    unc[inonzero] = np.sqrt(covariance.diagonal())*rms_noise
                break
            except linalg.LinAlgError as error:
                if min_valid == 0:
                    min_valid = 1e-10
                else:
                    min_valid *= 10
        self._set_parameters(p,unc,rescale=True)
        self.construct()

    def _get_rms(self):
        """Compute root-mean-square error."""
        if self.residual is None or len(self.residual)==0:
            return(None)
        retval = tools.rms(self.residual)
        return(retval)

    rms = property(_get_rms)


class Parameter():
    """An adjustable parameter."""

    def __init__(
            self,
            value=1.,
            vary=False,
            step=None,
            unc=0.0,
            fmt='0.12g',
            description='parameter',
    ):
        self._value = float(value)
        self.vary = vary
        self.fmt = fmt
        self.unc = float(unc)
        self.description = description
        if step is not None:
            self.step = abs(float(step))
        else:
            if self.value != 0:
                self.step = self.value*1e-4
            else:
                self.step = 1e-4
        self._last_modify_value_time = timestamp()

    def _get_value(self):
        return self._value

    def _set_value(self,value):
        """Set new parameter value and set modify time if it has changed."""
        if value == self._value:
            ## if parameter is not changed then do nothing
            return
        self._value = value
        self._last_modify_value_time = timestamp()

    value = property(_get_value,_set_value)

    def __repr__(self):
        return ('P(' +format(self.value,self.fmt)
                +','+repr(self.vary)
                +','+format(self.step,'0.2g')
                +','+format(self.unc,'0.2g')
                +')')

    def __str__(self):
        return repr(self)

    def __neg__(self): return(-self.value)
    def __float__(self): return(float(self.value))
    def __pos__(self): return(+self.value)
    def __abs__(self): return(abs(self.value))
    def __add__(self,other): return(self.value+other)
    def __radd__(self,other): return(self.value+other)
    def __sub__(self,other): return(self.value-other)
    def __rsub__(self,other): return(other-self.value)
    def __truediv__(self,other): return(self.value/other)
    def __rtruediv__(self,other): return(other/self.value)
    def __mul__(self,other): return(self.value*other)
    def __rmul__(self,other): return(other*self.value)
    def __pow__(self,other): return(self.value**other)
    def __rpow__(self,other): return(other**self.value)
    def __eq__(self,other): return(self.value == other)
    def __ne__(self,other): return(self.value != other)
    def __req__(self,other): return(self.value == other)
    def __gt__(self,other): return(self.value>other)
    def __lt__(self,other): return(self.value<other)
    def __ge__(self,other): return(self.value>=other)
    def __le__(self,other): return(self.value<=other)

P = Parameter                   # an abbreviation

class Named_Parameter(P):
    """Like a Parameter but has a name and knows which Optimiser it
    originally belongs to."""

    def __init__(self,optimiser,name,value,*args,**kwargs):
        self.optimiser = optimiser # back reference to optimiser where this is defined
        self.name = name
        P.__init__(self,value,*args,**kwargs)

    def __repr__(self):
        return f"{self.optimiser.name}['{self.name}']"

    __str__ = Parameter.__repr__

def _substitute_parameters_and_optimisers(x,translate):
    """Search for Parameters and Optimisers in x and substitude by id from
    translate dict.  x must be a list or a dict.  Recursively searchs
    lists or dicts in x."""
    maximum_list_length_for_recursive_substitution = 1000
    if isinstance(x,list):
        for i,val in enumerate(x):
            if isinstance(val,(Parameter,Optimiser)) and id(val) in translate:
                x[i] = translate[id(val)]
            elif isinstance(val,(list,dict)) and i < maximum_list_length_for_recursive_substitution:
                x[i] = _substitute_parameters_and_optimisers(val,translate)
    elif isinstance(x,dict):
        for key,val in x.items():
            if isinstance(val,(Parameter,Optimiser)) and id(val) in translate:
                x[key] = translate[id(val)]
            elif isinstance(val,(list,dict)):
                x[key] = _substitute_parameters_and_optimisers(val,translate)
    return x

def _deepcopy_function(function,translate_defaults=None):
    """Make a new function copy.  Substitute default parameters by id
    from those in translate_defaults."""
    ## get default -- including substitutions
    defaults = function.__defaults__
    if translate_defaults is not None:
        ## copy all Paraemter/Optimiser references
        defaults = _substitute_parameters_and_optimisers(list(defaults),translate_defaults)
        defaults = tuple(defaults)
    ## make new function
    fnew = types.FunctionType(
        function.__code__.replace(),
        function.__globals__,
        function.__name__,
        defaults,
        function.__closure__,)
    fnew.__qualname__= function.__name__
    return fnew

def _calculate_jacobian_multiprocessing_worker(shared_namespace,p,i):
    """Calculate part of a jacobian."""
    import dill
    from time import perf_counter as timestamp
    import numpy as np
    optimiser = dill.loads(shared_namespace.dill_pickle)
    residual = shared_namespace.residual
    rms = np.sqrt(np.mean(residual**2))
    pnew = copy(p)
    jacobian = []
    step = 1e-8                 # nmust be consistent abovce
    for ii in i:
        timer = timestamp()
        pnew[ii] += step
        optimiser._set_parameters(pnew,rescale=True)
        residualnew = optimiser.construct()
        rmsnew = np.sqrt(np.mean(residualnew**2))
        dresidual = (residualnew-residual)
        jacobian.append((residualnew-residual)/step)
        pnew[ii] = p[ii]
        if rms == rmsnew:
            message = 'parameter has no effect'
        else:
            message = ''    
        print(f'jcbn: {ii:>6d} time: {timestamp()-timer:>7.2e} rms: {rmsnew:>12.8e} {message}')
    return jacobian

