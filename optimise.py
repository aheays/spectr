import time
import inspect
import re
import os
# import sys

import numpy as np
from numpy import nan

# from . import dataset
from . import tools
from .tools import ensure_iterable
from . import plotting

def _collect_parameters(x):
    """Iteratively collect Parametero / P objects from x descending
    into any iterable children."""
    maximum_length_for_searching_for_parameters = 100
    if isinstance(x,P):
        return [x]
    elif tools.isiterable(x) and len(x)<maximum_length_for_searching_for_parameters:
        if isinstance(x,dict):
            x = x.values()
        retval = []
        for y in x:
            retval.extend(_collect_parameters(y))
        return retval
    else:
        return []

def auto_construct_method(function_name):
    """A decorator factory for automatically adding parameters,
    construct_function, and input_format_function from a decorated
    method.  function_name required to make the
    input_format_function.  The undecorated method must return a
    construct_function. Parameters are picked out of the input
    kwargs. POSITIONAL ARGUMENTS NOT ALLOWED for simplicity."""
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
            ## add parameters in kwargs
            self.parameters.extend(_collect_parameters(kwargs))
            ## make a construct function
            construct_function = function(self,**kwargs)
            self.add_construct_function(construct_function)
            ## make a foramt_input_function
            def f():
                if len(kwargs)<2:
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
            description='',
            **named_parameters,
    ):
        """Suboptimisers can be other Optimiser that are also
        optimised with this one."""
        assert isinstance(name,str),'Name must be a string.'
        self.name = name # for generating evaluatable references to self
        self.residual_scale_factor = None # scale all resiudal by this amount -- useful when combining with other optimisers
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
        self._construct_time = time.time() # when last constructed
        self._modify_time = time.time()    # when model was last modified by adding a construct function
        self._named_parameters = {}         # addresable as optimiser[key]
        for key,val in named_parameters.items():
            self._named_parameters[key] = self.add_parameter(val)
        ## make an input line
        kwargs = dict(
            name=self.name,
            suboptimisers=self._suboptimisers,
            verbose=self.verbose,
            description=self.description,
            )
        if len(self._suboptimisers) == 0:
            kwargs.pop('suboptimisers')
        if self.verbose == False:
            kwargs.pop('verbose')
        if self.description == '':
            kwargs.pop('description')
        def f():
            retval = f'{self.name} = Optimser(name={self.name},)'
            if len(self._suboptimisers)>0:
                retval += ','.join([repr(t) for t in self._suboptimisers])+','
            if self.verbose:
                retval += f'verbose={repr(self.verbose)},'
            if self.description != '':
                retval += f'description={repr(self.description)},'
            # if len(parameters)>0:
                # retval += ','.join([f'name={repr(p)}' for name,p in parameters.items()])
            return retval
        self.add_format_input_function(f)

    def __repr__(self):
        """No attempt to represent this data but its name may be used in place
        of it."""
        return self.name

    def add_format_input_function(self,*functions):
        """Add a new format input function."""
        for function in functions:
            self._format_input_functions[time.time()] = function

    def pop_format_input_function(self):
        """Delete last format input function added."""
        key = list(self._format_input_functions.keys())[-1]
        self._format_input_functions.pop(key)

    def automatic_format_input_function(self,limit_to_args=None):
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
            return f'{caller_locals["name"]} = {submodule_name}.{class_name}('+','.join(args)+')'
        self.add_format_input_function(f)

    def add_suboptimiser(self,*suboptimisers,add_format_function=False,):
        """Add one or suboptimisers."""
        ## add only if not already there
        for t in suboptimisers:
            if t not in self._suboptimisers:
                self._suboptimisers.append(t)
        if add_format_function:
            self.format_input_functions.append(
                f'{self.name}.add_suboptimiser({",".join(t.name for t in suboptimisers)},{repr(add_format_function)})')

    def add_parameter(self,parameter,*args,**kwargs):
        """Add one parameter. Return a reference to it. Args are as in
        pP or one are is a P."""
        if not isinstance(parameter,P):
            parameter = P(*tools.ensure_iterable(parameter),*args,**kwargs)
        self.parameters.append(parameter)
        return parameter

    def __getitem__(self,key):
        return self._named_parameters[key]

    def __setitem__(self,key,val):
        self._named_parameters[key].value = val

    def add_construct_function(self,*functions):
        """Add one or more functions that are called each iteration when the
        model is optimised. Optionally these may return an array that is added
        to the list of residuals."""
        for f in functions:
            self._construct_functions[time.time()] = f
        self._modify_time = time.time()

    def add_post_construct_function(self,*functions):
        for f in functions:
            self._post_construct_functions[time.time()] = f
        self._modify_time = time.time()

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

    # def set_residual_scale_factor(self,factor):
        # """Residual is scaled by this factor, useful where this is a
        # suboptimiser of something else to select their relative
        # importance."""
        # self.residual_scale_factor = factor
        # self.format_input_functions.append(f'{self.name}.set_residual_scale_factor({repr(factor)})')

    def _get_all_suboptimisers(self,_already_recursed=None):
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
        retval.append(self)     # put self last
        return(retval)

    # def get_parameter_dataset(self):
        # """Compose parameters into a DataSet"""
        # from .dataset import Dataset
        # data = Dataset()
        # data.prototypes = {
            # 'description':dict(kind=str,description='Description of this parameter.'),
            # 'value':dict(kind=float,description='Parameter value.'),
            # 'uncertainty':dict(kind=float,description='Parameter uncertainty.',fmt='0.2g'),
            # 'vary':dict(kind=bool,description='Optimised or not.'),
            # 'step':dict(kind=float, description='Linearisation step size.', fmt='0.2g'),
        # }
        # parameters = self.get_parameters()
        # data['description'] = [t.description for t in parameters]
        # data['value'] = [t.value for t in parameters]
        # data['uncertainty'] = [t.uncertainty for t in parameters]
        # data['vary'] = [t.vary for t in parameters]
        # data['step'] = [t.step for t in parameters],
        # return data

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
        t = repr(match_lines_regexp) if match_lines_regexp is not None else ''
        self.add_format_input_function(lambda: f'{self.name}.print_input({repr(match_lines_regexp) if match_lines_regexp is not None else ""})')
        print(self.format_input(match_lines_regexp=match_lines_regexp))

    def save_input(self,filename=None,match_lines_regexp=None):
        """Save recreated input function to a file."""
        tools.string_to_file(filename,self.format_input(match_lines_regexp))

    def __str__(self):
        # return self.get_parameter_dataset().format()
        return self.format_input()

    def save_to_directory(
            self,
            directory,
            trash_existing=False, # delete existing data, even if false overwriting may occur
    ):
        """Save results of model and optimisation to a directory."""
        ## new input line
        self.add_format_input_function(
            lambda directory=directory: f'{self.name}.save_to_directory({repr(directory)},trash_existing={repr(trash_existing)})')
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
            tools.string_to_file(
                subdirectory+'/parameters.psv',
                optimiser.get_parameter_dataset().format(delimiter=' | '))
            tools.string_to_file(subdirectory+'/input.py',optimiser.format_input())
            if optimiser.residual is not None:
                tools.array_to_file(subdirectory+'/residual' ,optimiser.residual,fmt='%+0.4e')
            # else:
                # tools.array_to_file(subdirectory+'/residual' ,[])
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

    # def get_parameters(self):
        # """Return a list of parameter objects in this optimiser and all
        # suboptimisers."""
        # retval = []
        # unique_ids = []
        # for optimiser in self._get_all_suboptimisers():
            # for parameter in optimiser.parameters:
                # if id(parameter) not in unique_ids:
                    # retval.append(parameter)
                    # unique_ids.append(id(parameter))
        # return(retval)

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
        """Set output of optimiser p-vectro to Parmeter objects, self._data if
        a Dataset or in suboptimisers."""
        from .dataset import Dataset # import here to avoid a circular import when loading this model with dataset.py
        unique_parameters = []
        p = list(p)
        if dp is not None:
            dp = list(dp)
        for optimiser in self._get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if parameter not in unique_parameters and parameter.vary:
                    parameter.value = p.pop(0)
                    if dp is not None:
                        parameter.dp = dp.pop(0)
                    unique_parameters.append(parameter)
            if isinstance(optimiser,Dataset):
                for key in optimiser.optimised_keys():
                    vary = optimiser.get_vary(key)
                    for i in tools.find(vary):
                        optimiser.set(key,p.pop(0),i)
                        if dp is not None:
                            optimiser.set_uncertainty(key,dp.pop(0),i)
                        else:
                            optimiser.set_uncertainty(key,np.nan,i)

    def has_changed(self):
        """Whether the construction of this optimiser has been changed or any
        parameters have changed since its last construction."""
        ## search self and all suboptimisers
        for o in self._get_all_suboptimisers():
            ## search for internally specified modification
            if o._modify_time > self._construct_time:
                return True 
            ## search for change parameter
            for p in o.parameters:
                if p._modify_time > self._construct_time:
                    return True
            ## search for the addition of  new construct fucntions
            for t,f in (list(o._construct_functions.items())
                        + list(o._post_construct_functions.items())[::-1]):
                if t > self._construct_time:
                    return True 
        ## no changes
        return False

    def construct(
            self,
            recompute_all=False, # recompute Optimiser even if has not changed since last construct
    ):
        """Run all construct functions and return collected residuals."""
        ## nothing to be done, return cached residual immediately
        if not self.has_changed() and not recompute_all:
            return self.combined_residual
        ## collect residuals from suboptimisers and self
        combined_residual = []  # from self and suboptimisers
        for optimiser in self._get_all_suboptimisers():
            if optimiser.has_changed() or recompute_all:
                if self.verbose:
                    print(f'constructing optimiser: {optimiser.name}')
                optimiser.residual = []
                for f in (list(optimiser._construct_functions.values())
                          + list(optimiser._post_construct_functions.values())[::-1]):
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
                optimiser._construct_time = time.time()
            ## add resisudal to return value for optimisation, possibly rescaling it
            if optimiser.residual is not None:
                if optimiser.residual_scale_factor is not None:
                    combined_residual.append(optimiser.residual_scale_factor*np.array(optimiser.residual))
                else:
                    combined_residual.append(np.array(optimiser.residual))
        combined_residual = np.concatenate(combined_residual)  # includes own residual and for all suboptimisers
        self.combined_residual = combined_residual # this includes residuals from construct_functions combined with suboptimisers
        return(combined_residual)

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
                current_time = time.time()
                print(f'call: {self._number_of_optimisation_function_calls:<6d} time: {current_time-self._previous_time:<10.3g} RMS: {rms:<12.8e}')
                self.monitor()
                self._previous_time = current_time
                if rms<self._rms_minimum: self._rms_minimum = rms
        return(residuals)           

    def optimise(
            self,
            compute_final_uncertainty=False, # single Hessian computation with normlised suboptimiser residuals
            xtol=1e-14,
            rms_noise=None,
            monitor_frequency='every iteration', # 'rms decrease', 'never'
            verbose=True,
            normalise_suboptimiser_residuals=False,
            data_interpolation_factor=1.,
    ):
        """Optimise parameters."""
        def f(xtol=xtol,
              normalise_suboptimiser_residuals=normalise_suboptimiser_residuals):
            return(f'''{self.name}.optimise(
            xtol={repr(xtol)},
            compute_final_uncertainty={repr(compute_final_uncertainty)},
            rms_noise={repr(rms_noise)},
            monitor_frequency={repr(monitor_frequency)},
            verbose={repr(verbose)},
            data_interpolation_factor={repr(data_interpolation_factor)},
            normalise_suboptimiser_residuals={repr(normalise_suboptimiser_residuals)})''')
        self.add_format_input_function(f)
        if compute_final_uncertainty:
            ## a hack to prevent iteratoin of leastsq, and just get
            ## the estimated error at the starting point
            xtol = 1e16
        if normalise_suboptimiser_residuals:
            ## normalise all suboptimiser residuals to one, only
            ## appropriate if model is finished -- common normalisation
            ## for all construct function outputs in each suboptimiser
            for suboptimiser in self._get_all_suboptimisers():
                self.construct()
                if suboptimiser.residual is not None and len(suboptimiser.residual)>0:
                    if suboptimiser.residual_scale_factor is None:
                        suboptimiser.residual_scale_factor = 1/tools.nanrms(suboptimiser.residual)
                    else:
                        suboptimiser.residual_scale_factor /= tools.nanrms(suboptimiser.residual)
                    suboptimiser.residual = None # mark undone
        ## info
        if self.verbose:
            print(f'{self.name}: optimising')
        self.monitor_frequency = monitor_frequency
        assert monitor_frequency in ('rms decrease','every iteration','never'),f"Valid monitor_frequency: {repr(('rms decrease','every iteration','never'))}"
        self._rms_minimum,self._previous_time = np.inf,time.time()
        self._number_of_optimisation_function_calls = 0
        ## get initial values and reset uncertainties
        p,s,dp = self._get_parameters()
        if verbose:
            print('Number of varied parameters:',len(p))
        if len(p)>0:
            ## 2018-05-08 on one occasion I seemed to be getting
            ## returned p from leastsq which did not correspond to the
            ## best fit!!! So I did not update p from this output and
            ## retained what was set in construct()
            try:
                p,dp = tools.leastsq(self._optimisation_function, p,s,xtol=xtol,rms_noise=rms_noise,)
                if self.verbose:
                    print('Number of evaluations:',self._number_of_optimisation_function_calls)
                ## update parameters and uncertainties
                self._set_parameters(p,dp)
            except KeyboardInterrupt:
                pass
        residual = self.construct(recompute_all=True) # run at least once, recompute_all to get uncertainties
        self.monitor() # run monitor functions after optimisation
        if verbose:
            print('total RMS:',np.sqrt(np.mean(np.array(self.combined_residual)**2)))
            for suboptimiser in self._get_all_suboptimisers():
                if (suboptimiser.residual is not None
                    and len(suboptimiser.residual)>0):
                    print(f'suboptimiser {suboptimiser.name} RMS:',tools.rms(suboptimiser.residual))
        return(residual) # returns residual array

    def _get_rms(self):
        """Compute root-mean-square error."""
        if self.residual is None or len(self.residual)==0:
            return(None)
        retval = tools.rms(self.residual)
        return(retval)

    rms = property(_get_rms)




class P():
    """An adjustable parameter."""

    def __init__(
            self,
            value=1.,
            vary=False,
            step=None,
            uncertainty=np.nan,
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
        self._modify_time = time.time()

    def _get_value(self):
        return self._value

    def _set_value(self,value):
        self._value = value
        self._modify_time = time.time()

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
    def __eq__(self,other): return(self.value == other)
    def __req__(self,other): return(self.value == other)
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



