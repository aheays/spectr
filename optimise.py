from time import perf_counter as timestamp
import inspect
import re
import os
# import sys
from pprint import pprint
from copy import copy,deepcopy
import warnings

import numpy as np
from numpy import nan,inf
from scipy import optimize,linalg

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
    """Iteratively collect Parameter and Optimiser objects from x
    descending into any iterable children."""
    maximum_length_for_searching = 1000
    parameters,optimisers = [],[]
    if isinstance(x,Parameter):
        parameters.append(x)
    elif isinstance(x,Optimiser):
        optimisers.append(x)
    elif tools.isiterable(x) and len(x)<maximum_length_for_searching:
        if isinstance(x,dict):
            x = x.values()
        retval = []
        for y in x:
            tp,to = _collect_parameters_and_optimisers(y)
            parameters.extend(tp)
            optimisers.extend(to)
    return parameters,optimisers

def auto_construct_method(
        function_name,
        format_single_line=None,
        format_multi_line=None,
):
    """A decorator factory for automatically adding parameters,
    construct_function, and input_format_function from a decorated
    method.  function_name required to make the
    input_format_function.  The undecorated method must return a
    construct_function. Parameters are picked out of the input
    kwargs. POSITIONAL ARGUMENTS NOT ALLOWED for simplicity."""
    warnings.warn('deprecated')
    def actual_decorator(function):
        def new_function(self,*args,**kwargs):
            ## this block subtitutes into kwargs with keys taken from
            ## the function signature.  get signature argumets -- skip
            ## first "self"
            signature_keys = list(inspect.signature(function).parameters)[1:]
            for iarg,(arg,signature_key) in enumerate(zip(args,signature_keys)):
                if signature_key in kwargs:
                    raise Exception(f'Positional argument also appears as keyword argument {repr(signature_key)} in function {repr(function_name)}.')
                kwargs[signature_key] = arg
            ## add parameters suboptimisers in args to self
            parameters,suboptimisers =  _collect_parameters_and_optimisers(kwargs)
            for t in parameters:
                self.add_parameter(t)
            for t in suboptimisers:
                self.add_suboptimiser(t)
            ## make a construct function
            construct_function = function(self,**kwargs)
            self.add_construct_function(construct_function)
            ## make a foramt_input_function
            def f():
                if (len(kwargs)<2 or format_single_line) and not format_multi_line:
                    formatted_kwargs = ','.join([f"{key}={repr(val)}" for key,val in kwargs.items() if val is not None])
                    return f'{self.name}.{function_name}({formatted_kwargs},)'
                else:
                    formatted_kwargs = ',\n    '.join([f"{key}={repr(val)}" for key,val in kwargs.items() if val is not None])
                    return f'{self.name}.{function_name}(\n    {formatted_kwargs},\n)'
            self.add_format_input_function(f)
            ## return kwargs
            return kwargs
        return new_function
    return actual_decorator

def optimise_method(format_single_line=None,format_multi_line=None):
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
        def new_function(self,*args,**kwargs):
            ## this block subtitutes into kwargs with keys taken from
            ## the function signature.  get signature arguments -- skip
            ## first "self"
            signature_keys = list(inspect.signature(function).parameters)[1:]
            for iarg,(arg,signature_key) in enumerate(zip(args,signature_keys)):
                if signature_key in kwargs:
                    raise Exception(f'Positional argument also appears as keyword argument {repr(signature_key)} in function {repr(function_name)}.')
                kwargs[signature_key] = arg
            ## add parameters suboptimisers in args to self
            parameters,suboptimisers =  _collect_parameters_and_optimisers(kwargs)
            for t in parameters:
                self.add_parameter(t)
            for t in suboptimisers:
                self.add_suboptimiser(t)
            ## make a construct function which returns the output of
            ## the original method, add an empty _cache if '_cache' in
            ## signature kwargs but not provided in arguments
            if '_cache' in signature_keys:
                kwargs.setdefault('_cache',{})
            self.add_construct_function(lambda: function(self,**kwargs))
            ## make a foramt_input_function
            def f():
                kwargs_to_format = {key:val for key,val in kwargs.items() if key != '_cache' and val is not None}
                if (len(kwargs_to_format)<2 or format_single_line) and not format_multi_line:
                    formatted_kwargs = ','.join([f"{key}={repr(val)}" for key,val in kwargs_to_format.items()])
                    return f'{self.name}.{function.__name__}({formatted_kwargs},)'
                else:
                    formatted_kwargs = ',\n    '.join([f"{key}={repr(val)}" for key,val in kwargs_to_format.items()])
                    return f'{self.name}.{function.__name__}(\n    {formatted_kwargs},\n)'
            self.add_format_input_function(f)
            ## returns all args as a dictionary -- but not _cache
            retval = copy(kwargs)
            if "_cache" in retval:
                retval.pop('_cache')
            return retval
        return new_function
    return actual_decorator

def format_input_method(format_single_line=None,format_multi_line=None):
    """Add function to format_input_functions and run it."""
    def actual_decorator(function):
        def new_function(self,*args,**kwargs):
            ## this block subtitutes into kwargs with keys taken from
            ## the function signature.  get signature arguments -- skip
            ## first "self"
            signature_keys = list(inspect.signature(function).parameters)[1:]
            for iarg,(arg,signature_key) in enumerate(zip(args,signature_keys)):
                if signature_key in kwargs:
                    raise Exception(f'Positional argument also appears as keyword argument {repr(signature_key)} in function {repr(function_name)}.')
                kwargs[signature_key] = arg
            ## make a format_input_function
            def f():
                kwargs_to_format = {key:val for key,val in kwargs.items() if key != '_cache' and val is not None}
                if (len(kwargs_to_format)<2 or format_single_line) and not format_multi_line:
                    formatted_kwargs = ','.join([f"{key}={repr(val)}" for key,val in kwargs_to_format.items()])
                    return f'{self.name}.{function.__name__}({formatted_kwargs},)'
                else:
                    formatted_kwargs = ',\n    '.join([f"{key}={repr(val)}" for key,val in kwargs_to_format.items()])
                    return f'{self.name}.{function.__name__}(\n    {formatted_kwargs},\n)'
            self.add_format_input_function(f)
            ## run the function
            function(self,*args,**kwargs)
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
        self._construct_functions = {} # list of function which collectively construct the model and optionally return a list of fit residuals
        self._post_construct_functions = {} # run in reverse order after other construct functions
        self._plot_functions = [] 
        self._monitor_functions = [] # call these functions at special times during optimisation
        self.monitor_frequency = 'rms decrease' # when to call monitor fuctions: 'never', 'every iteration', 'rms decrease'
        self._save_to_directory_functions = [] # call these functions when calling save_to_directory (input argument is the directory)
        self._format_input_functions = {}        # call these functions (return strings) to build a script recreating an optimisation
        self._suboptimisers = list(suboptimisers) # Optimiser objects optimised with this one, their construct functions are called first
        self.verbose = verbose                     # True to activate various informative outputs
        self.residual = None # array of residual errors generated by construct functions
        self.combined_residual = None # array of residual errors generated by construct functions of this and its suboptimisers
        self.description = description
        self._last_construct_time = timestamp() # when last constructed
        self._last_add_construct_function_time = timestamp()    # when model was last modified by adding a construct function
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
        f()                  # run function now
        # self._last_construct_time = timestamp()
        self._last_add_construct_function_time = timestamp()

    def add_post_construct_function(self,*functions):
        for f in functions:
            self._post_construct_functions[timestamp()] = f
        self._last_add_construct_function_time = timestamp()

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

    def _get_all_suboptimisers(self,_already_recursed=None,include_self=True):
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
                retval.extend(optimiser._get_all_suboptimisers(_already_recursed))
        if include_self:
            retval.append(self)     # put self last
        return retval 

    def format_input(self,match_lines_regexp=None):
        """Join strings which should make an exectuable python script for
        repeating this optimisation with updated parameters. Each element of
        self.format_input_functions should be a string or a function of no
        arguments evaluating to a string."""
        ## collect all format_input_functions
        timestamps,functions,suboptimisers = [],[],[]
        for optimiser in self._get_all_suboptimisers():
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
        ## limit to matching lines if requested
        if match_lines_regexp is not None:
            lines = [t for t in lines if re.match(match_lines_regexp,t)]
        return('\n'.join(lines))

    def print_input(self,match_lines_regexp=None):
        """Print recreated input function. Filter lines by regexp if
        desired."""
        # t = repr(match_lines_regexp) if match_lines_regexp is not None else ''
        self.add_format_input_function(lambda: f'{self.name}.print_input({repr(match_lines_regexp) if match_lines_regexp is not None else ""})')
        print(self.format_input(match_lines_regexp=match_lines_regexp))

    def save_input(self,filename=None,match_lines_regexp=None):
        """Save recreated input function to a file."""
        tools.string_to_file(filename,self.format_input(match_lines_regexp))

    def __str__(self):
        return self.format_input()


    # @format_input_method()
    def save_to_directory(self,directory,trash_existing=False):
        """Save results of model and optimisation to a directory."""
        directory = tools.expand_path(directory)
        tools.mkdir(directory,trash_existing=trash_existing)
        ## output self and all suboptimisers into a flat subdirectory
        ## structure
        used_subdirectories = []
        for optimiser in self._get_all_suboptimisers():
            subdirectory = directory+'/'+optimiser.name+'/'
            tools.mkdir(subdirectory,trash_existing=True)
            if subdirectory in used_subdirectories:
                raise Exception(f'Non-unique optimiser names producting subdirectory: {repr(subdirectory)}')
            used_subdirectories.append(subdirectory)
            tools.string_to_file(subdirectory+'/input.py',optimiser.format_input())
            if optimiser.residual is not None:
                tools.array_to_file(subdirectory+'/residual' ,optimiser.residual,fmt='%+0.4e')
            if optimiser.description is not None:
                tools.string_to_file(subdirectory+'/README' ,str(optimiser.description))
            for f in optimiser._save_to_directory_functions:
                f(subdirectory)
        ## symlink suboptimsers into subdirectories
        for optimiser in self._get_all_suboptimisers():
            for suboptimiser in optimiser._suboptimisers:
                tools.mkdir(f'{directory}/{optimiser.name}/suboptimisers/')
                os.symlink(
                    f'../../{suboptimiser.name}',
                    f'{directory}/{optimiser.name}/suboptimisers/{suboptimiser.name}',
                    target_is_directory=True)

    def plot_residual(self,ax=None,**plot_kwargs):
        """Plot residual error."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()
        plot_kwargs.setdefault('marker','o')
        previous_xend = 0
        for suboptimiser in self._get_all_suboptimisers():
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
        for suboptimiser in self._get_all_suboptimisers():
            for function in suboptimiser._plot_functions:
                fig = plt.figure(first_figure_number)
                fig.clf()
                function()
                first_figure_number += 1
        
    def monitor(self):
        """Run monitor functions."""
        for optimiser in self._get_all_suboptimisers():
            for f in optimiser._monitor_functions:
                f()

    def _get_parameters(self):
        """Collect all varied parameters for optimisation. These are either in
        self.parameters, in self._data if this is also a Dataset or found in
        suboptimiser."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        value,uncertainty,step = [],[],[]
        unique_parameters = []  # to prevent the same parmeter being loaded twice from suboptimisers
        for optimiser in self._get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if id(parameter) not in unique_parameters and parameter.vary:
                    value.append(parameter.value)
                    uncertainty.append(parameter.step)
                    step.append(parameter.step)
                    unique_parameters.append(id(parameter))
            if isinstance(optimiser,Dataset):
                for key in optimiser.optimised_keys():
                    vary = optimiser.get_vary(key)
                    value.extend(optimiser[key][vary])
                    if optimiser.get_uncertainty(key,vary) is None:
                        optimiser.set_uncertainty(key,nan)
                    uncertainty.extend(optimiser.get_uncertainty(key,vary))
                    step.extend(optimiser.get_step(key,vary))
        return value,step,uncertainty

    def _set_parameters(self,p,dp=None):
        """Set assign elements of p and dp from optimiser to the
        correct Parameters."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        p = list(p)
        if dp is None:
            dp = [np.nan for pi in p]
        else:
            dp = list(dp)
        already_set = []
        for optimiser in self._get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if id(parameter) not in already_set and parameter.vary:
                    parameter.value = p.pop(0)
                    parameter.uncertainty = dp.pop(0)
                    already_set.append(id(parameter))
            if isinstance(optimiser,Dataset):
                for key in optimiser.optimised_keys():
                    vary = optimiser.get_vary(key)
                    ## could speed up using slice rather than pop?
                    for i in tools.find(vary):
                        optimiser.set(key,value=p.pop(0),uncertainty=dp.pop(0),index=i)

    def has_changed(self):
        """Whether the construction of this optimiser has been changed or any
        parameters have changed since its last construction."""
        ## test if new construct function added recently
        if self._last_add_construct_function_time > self._last_construct_time:
            return True 
        ## test if parameter in self has changed recently
        for p in self.parameters:
            if p._last_modify_value_time > self._last_construct_time:
                return True
        ## test if self is a Dataset and has been modified recently
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        if isinstance(self,Dataset) and self._last_modify_data_time > self._last_construct_time:
            return True
        ## test if any suboptimiser has changed, or been constructed
        ## recently
        for o in self._get_all_suboptimisers(include_self=False):
            if o.has_changed():
                return True 
            if  o._last_construct_time > self._last_construct_time:
                return True 
        ## no changes
        return False

    construct_functions = property(lambda self: list(self._construct_functions.values()) + list(self._post_construct_functions.values())[::-1])

    def construct(
            self,
            recompute_all=False, # recompute Optimiser even if has not changed since last construct
    ):
        """Run all construct functions and return collected residuals."""
        ## collect residuals from suboptimisers and self
        combined_residual = []  # from self and suboptimisers
        for optimiser in self._get_all_suboptimisers():
            if optimiser.has_changed() or recompute_all:
                if self.verbose:
                    print(f'constructing optimiser: {optimiser.name}')
                ## collect all construct functions, if there are any,
                ## then run them, save the residual, and mark the
                ## construct time
                optimiser.residual = []
                construct_functions = optimiser.construct_functions
                if len(construct_functions) > 0:
                    for f in construct_functions:
                        t = f()                # run construction function, get residual if there is one
                        if t is None:
                            continue
                        if np.isscalar(t):
                            t = [t]
                        if len(t)==0:
                            continue 
                        optimiser.residual.append(t) # add residual to overall residual
                    ## combine construct function residuals into one
                    if optimiser.residual is not None and len(optimiser.residual)>0:
                        optimiser.residual = np.concatenate(optimiser.residual)
                ## record time of construction
                optimiser._last_construct_time = timestamp()
            ## add resisudal to return value for optimisation, possibly rescaling it
            if optimiser.residual is not None:
                combined_residual.append(optimiser.residual_scale_factor*np.array(optimiser.residual))
        combined_residual = np.concatenate(combined_residual)  # includes own residual and for all suboptimisers
        self.combined_residual = combined_residual # this includes residuals from construct_functions combined with suboptimisers
        return combined_residual

    def _optimisation_function(self,p):
        """Internal function used by optimise routine. p is a list of varied
        parameters."""
        self._number_of_optimisation_function_calls += 1
        ## update parameters in internal model
        self._set_parameters(p)
        ## rebuild model and calculate residuals
        residuals = self.construct()
        ## monitor
        if residuals is not None and len(residuals)>0 and self.monitor_frequency!='never':
            rms = tools.nanrms(residuals)
            assert not np.isinf(rms),'rms is inf'
            assert not np.isnan(rms),'rms is nan'
            if (self.monitor_frequency=='every iteration'
                or (self.monitor_frequency=='rms decrease' and rms<self._rms_minimum)):
                current_time = timestamp()
                print(f'call: {self._number_of_optimisation_function_calls:<6d} time: {current_time-self._previous_time:<10.3g} RMS: {rms:<12.8e}')
                self.monitor()
                self._previous_time = current_time
                if rms<self._rms_minimum: self._rms_minimum = rms
        return residuals

    @format_input_method()
    def optimise(
            self,
            # compute_uncertainty_only=False, # do not optimise -- just compute uncertainty at current position, actually does one iteration
            rms_noise=None,
            monitor_frequency='every iteration', # 'rms decrease', 'never'
            verbose=True,
            normalise_suboptimiser_residuals=False,
            max_nfev=None,         # max number of iterations
            method=None,
    ):
        """Optimise parameters."""
        # if compute_uncertainty_only:
            # max_nfev = 1
        if normalise_suboptimiser_residuals:
            ## normalise all suboptimiser residuals before handing to
            ## the least-squares routine
            for suboptimiser in self._get_all_suboptimisers():
                self.construct(recompute_all=True)
                if suboptimiser.residual is not None and len(suboptimiser.residual)>0:
                    if suboptimiser.residual_scale_factor is None:
                        suboptimiser.residual_scale_factor = 1/tools.rms(suboptimiser.residual)
                    else:
                        suboptimiser.residual_scale_factor /= tools.rms(suboptimiser.residual)
        ## get initial values and reset uncertainties
        p,s,dp = self._get_parameters()
        self.monitor_frequency = monitor_frequency
        assert monitor_frequency in ('rms decrease','every iteration','never'),f"Valid monitor_frequency: {repr(('rms decrease','every iteration','never'))}"
        self._rms_minimum,self._previous_time = inf,timestamp()
        self._number_of_optimisation_function_calls = 0
        if verbose or self.verbose:
            print(f'{self.name}: optimising')
            print('Number of varied parameters:',len(p))
        if len(p)>0:
            if method == 'lm' or len(p) == 1:
                ## use 'lm' Levenberg-Marquadt
                warnings.warn('lm options not optimised')
                kwargs = dict(
                    method='lm',
                    jac='2-point',
                    # ## xtol=1e-10,
                    # ## ftol=,
                    # ## gtol=,
                    # ## bounds=(-inf,inf),
                    # ## x_scale=s,
                    # ## diff_step=1e-21,
                    # ## diff_step=[(si/pi if pi!=0 else 1/si) for si,pi in zip(s,p)],
                    # diff_step=np.asarray(s,dtype=float),
                    # x_scale='jac',
                    # ## x_scale=[t*100 for t in s],
                    # ## loss='soft_l1',
                    # loss='linear',
                    # ## tr_solver='exact',
                    # tr_solver='lsmr',
                    # max_nfev=max_nfev,
                    # ## jac_sparsity=None, 
                )
            else:
                ## use 'trf' -- trust region
                kwargs = dict(
                    method='trf',
                    jac='2-point',
                    ## xtol=1e-10,
                    ## ftol=,
                    ## gtol=,
                    ## bounds=(-inf,inf),
                    ## x_scale=s,
                    ## diff_step=1e-21,
                    ## diff_step=[(si/pi if pi!=0 else 1/si) for si,pi in zip(s,p)],
                    diff_step=s,
                    x_scale='jac',
                    ## x_scale=[t*100 for t in s],
                    ## loss='soft_l1',
                    loss='linear',
                    ## tr_solver='exact',
                    tr_solver='lsmr',
                    max_nfev=max_nfev,
                    ## jac_sparsity=None, 
                )
            try:
                ## call optimisation routine
                if verbose or self.verbose:
                    print('Method:',kwargs['method'])
                result = optimize.least_squares(self._optimisation_function,p,**kwargs)
                if verbose or self.verbose:
                    print('Optimisation complete')
                    print('    Number parameters:    ',len(p))
                    print('    Number of evaluations:',self._number_of_optimisation_function_calls)
                    print('    Number of iterations: ',result['nfev'])
                    print('    Termination reason:   ',result['message'])
                p = result['x']
                ## compute 1σ standard error
                try:
                    jacobian = result['jac']
                    covariance = linalg.inv(
                        np.dot(np.transpose(jacobian),jacobian))
                    if rms_noise is None:
                        chisq=np.sum(result["fun"]*result["fun"])
                        dof=len(result["fun"])-len(result['x'])+1        # degrees of freedom
                        rms_noise = np.sqrt(chisq/dof)
                    dp = np.sqrt(covariance.diagonal())*rms_noise
                except linalg.LinAlgError as err:
                    print(f'warning: failed to computed uncertainty: {err}')
                    dp = None
                ## update parameters and uncertainties
                self._set_parameters(p,dp)
            except KeyboardInterrupt:
                pass
        residual = self.construct(recompute_all=True) # run at least once, recompute_all to get uncertainties
        self.monitor() # run monitor functions after optimisation
        if verbose or self.verbose:
            print('total RMS:',np.sqrt(np.mean(np.array(self.combined_residual)**2)))
            for suboptimiser in self._get_all_suboptimisers():
                if (suboptimiser.residual is not None
                    and len(suboptimiser.residual)>0):
                    print(f'suboptimiser {suboptimiser.name} RMS:',tools.rms(suboptimiser.residual))
        return residual

    def calculate_uncertainty(self,p=None,rms_noise=None,verbose=True):
        """Compute 1σ uncertainty by first computing forward-difference
        Jacobian.  Only accurate for a well-optimised model."""
        ## get parameter and uncertainties
        if p is not None:
            self._set_parameters(p)
        value,step,uncertainty = self._get_parameters()
        ## compute model at p
        self._number_of_optimisation_function_calls = 0
        self._previous_time = timestamp()
        self._rms_minimum = np.inf
        self.monitor_frequency = 'every iteration'
        residual = self._optimisation_function(value)
        ## compute Jacobian by forward finite differencing
        tvalue = deepcopy(value)
        jacobian = np.full((len(residual),len(value)),0.0)
        for i,(valuei,stepi) in enumerate(zip(value,step)):
            tvalue[i] = valuei+stepi
            residuali = self._optimisation_function(tvalue)
            jacobian[:,i] = (residuali-residual)/stepi
            tvalue[i] = valuei # change it back
        ## compute 1σ uncertainty from Jacobian
        covariance = linalg.inv(
            np.dot(np.transpose(jacobian),jacobian))
        if rms_noise is None:
            chisq = np.sum(residual**2)
            dof = len(residual)-len(value)+1
            rms_noise = np.sqrt(chisq/dof)
        uncertainty = np.sqrt(covariance.diagonal())*rms_noise
        ## set back to best fit
        self._set_parameters(value,uncertainty) # set param
        self._optimisation_function(value)      # construct
        self._set_parameters(value,uncertainty) # reset uncertainties
        return uncertainty

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
            uncertainty=0.0,
            fmt='0.12g',
            description='parameter',
    ):
        self._value = float(value)
        self.vary = vary
        self.fmt = fmt
        self.uncertainty = float(uncertainty)
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
                +','+format(self.uncertainty,'0.2g')
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

