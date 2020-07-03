import time
import re

import numpy as np

from spectr.datum import Datum
from spectr.dataset import DataSet
from spectr import tools

class Optimiser:
    """Defines adjustable parameters and model-building functions which
    may return a residual error. Then optimise the parameters with
    respect to these residuals. Can contain suboptimisers which are
    simultaneously optimised and dependencies of this one. """
    
    def __init__(
            self,
            name='o',           # used to generate evaluable python code
            *suboptimisers,     # Optimisers
            verbose=False,
            description=None,
            **parameters # To add a parameter set in during initialisation
    ):
        """Suboptimisers can be other Optimiser that are also
        optimised with this one."""
        assert isinstance(name,str),'Name must be a string.'
        self.name = name # for generating evaluatable references to self
        self.residual_scale_factor = None # scale all resiudal by this amount -- useful when combining with other optimisers
        self.parameters = []           # Data objects
        self.construct_functions = [] # list of function which collectively construct the model and optionally return a list of fit residuals
        self.monitor_functions = [] # call these functions at special times during optimisation
        self.monitor_frequency = 'rms decrease' # when to call monitor fuctions: 'never', 'every iteration', 'rms decrease'
        self.output_to_directory_functions = [] # call these functions when calling output_to_directory (input argument is the directory)
        self._format_input_functions = {}        # call these functions (return strings) to build a script recreating an optimisation
        self.suboptimisers = list(suboptimisers) # Optimiser objects optimised with this one, their construct functions are called first
        self.verbose = verbose                     # True to activate various informative outputs
        self.residual = None # array of residual errors generated by construct functions
        self.combined_residual = None # array of residual errors generated by construct functions of this and its suboptimisers
        self.description = description
        self.timestamp = time.time() # when last computed
        ## add given parameters as a Parameter_Set
        if len(parameters)>0:
            parameters = self.add_parameter_set(description=None,**parameters)
        ## make an input line
        def f():
            retval =  f'{self.name} = Optimiser({repr(self.name)}'
            if len(suboptimisers)>0:
                retval += ',\n    '+",".join([t.name for t in self.suboptimisers])+','
            if len(parameters)==0:
                pass
            elif len(parameters)==1:
                retval += ',',+parameters.format_input()
            else:
                retval += ',\n    '+parameters.format_multiline()
            if self.description is not None:
                retval += f',\ndescription={repr(description)},'
            retval += ')'
            return(retval)
        self.add_format_input_function(f)

    def add_format_input_function(self,function):
        """Add a new format input function."""
        self._format_input_functions[time.time()] = function

    ## read only
    format_input_functions = property(lambda self:self._format_input_functions)

    # def set_residual_scale_factor(self,factor):
        # """Residual is scaled by this factor, useful where this is a
        # suboptimiser of something else to select their relative
        # importance."""
        # self.residual_scale_factor = factor
        # self.format_input_functions.append(f'{self.name}.set_residual_scale_factor({repr(factor)})')

    def add_suboptimiser(self,*suboptimisers,add_format_function=None):
        ## add only if not already there
        for t in suboptimisers:
            if t not in self.suboptimisers:
                self.suboptimisers.append(t)
        if add_format_function:
            self.format_input_functions.append(
                f'{self.name}.add_suboptimiser({",".join(t.name for t in suboptimisers)})')

    def add_parameter(self,description,value,vary=None,step=1e-8,):
        p = Datum(
            description=description,
            value=value,vary=vary,step=step,
            kind=float,uncertainty=np.nan,)
        self.parameters.append(p)
        return(p)

    def add_parameter_set(
            self,
            description=None, # added as a description to all parameters
            **keys_args,       # kwargs in the form name=p or name=(Parameter_args)
    ):
        p = ParameterSet(description=description)
        for key,args in keys_args.items():
            p[key] = args
        self.parameters.extend(p.values())
        return(p)

    # def add_parameter_list(self,name_prefix='',p=[],vary=False,step=None,note=None,fmt=None):
        # """Add a variable length list of parameters. vary and step can
        # either be constants, be the same length as p, or be a function
        # of the parameter order. name is a list of defaults to enumerate."""
        # ## expand inputs into lists of the same length
        # if vary is None or np.isscalar(vary):
            # vary = [vary for t in p]
        # elif type(vary)==types.FunctionType:
            # vary = [bool(vary(i)) for i in range(len(p))]
        # if np.isscalar(step) or step is None:
            # step = [step for t in p]
        # elif type(step)==types.FunctionType:
            # step = [float(step(i)) for i in range(len(p))]
        # assert len(p)==len(vary) and len(p)==len(step)
        # name = [name_prefix+str(i) for i in range(len(p))]
        # return(self.add_parameter_set(note=note,**{namei:(pi,varyi,stepi) for (namei,pi,varyi,stepi) in zip(name,p,vary,step)},fmt=fmt))

    def add_construct(self,*construct_functions):
        """Add one or more functions that are called each iteration when the
        model is optimised. Optionally these may return an array that is added
        to the list of residuals."""
        self.construct_functions.extend(construct_functions)

    def add_monitor(self,*monitor_functions):
        """Add one or more functions that are called when a new minimum
        is found in the optimisation. This might be useful for saving
        intermediate fits in case of failure or abortion."""
        self.monitor_functions.extend(monitor_functions)
        
    def get_all_suboptimisers(self,_already_recursed=None):
        """Return a list of all suboptimisers including self and without double
        counting. Ordered so suboptimisers always come first."""
        if _already_recursed is None:
            _already_recursed = [self] # first
        else:
            _already_recursed.append(self)
        retval = []
        for optimiser in self.suboptimisers:
            if optimiser in _already_recursed:
                continue
            else:
                _already_recursed.append(optimiser)
                retval.extend(optimiser.get_all_suboptimisers(_already_recursed))
        retval.append(self)     # put self last
        return(retval)

    def get_parameter_array(self):
        """Compose parameters into a Dynamic_Array"""
        data = DataSet()
        data.set('description',
                 [t.description for t in self.parameters],
                 kind=str,description='Description of this parameter.')
        data.set('value',
                 [t.value for t in self.parameters],
                 [t.uncertainty for t in self.parameters],
                 kind=float,description='Parameter values.')
        data.set('vary',
                 [t.vary for t in self.parameters],
                 kind=bool,description='Optimised or not.')
        data.set('step',
                 [t.step for t in self.parameters],
                 kind=float,description='Linearisation step size.')
        return(data)

    # def format_csv(self,varied_parameters_only=False):
        # """Print a list of parameters in csv format."""
        # lines = []
        # lines.append('{:<20s} , {:<25s} , {:>16s} , {:<5s} , {:>6} , {:>7} , {:<s}'.format('optimiser','name','p','vary','step','dp','note'))
        # # def quoted(string): return('"'+string+'"')
        # for optimiser in self.get_all_suboptimisers():
            # t = '\n'.join([
                # f'{optimiser.name:<20s} , {t.name:<25s} , {t.p:>+16.9e} , {repr(t.vary):<5} , {t.step:>6.0e} , {t.dp:>7.1e} , "{str(t.note)}"'
                # for t in optimiser.parameters if (t.vary or not varied_parameters_only)])
            # if len(t)>0:lines.append(t)
        # return('\n'.join(lines))

    def format_input(self,match_lines_regexp=None):
        """Join strings which should make an exectuable python script for
        repeating this optimisation with updated parameters. Each element of
        self.format_input_functions should be a string or a function of no
        arguments evaluating to a string."""
        ## collect all format_input_functions
        timestamps,functions,suboptimisers = [],[],[]
        
        for optimiser in self.get_all_suboptimisers():
            timestamps.extend(optimiser.format_input_functions.keys())
            functions.extend(optimiser.format_input_functions.values())
            suboptimisers.extend([optimiser for t in optimiser.format_input_functions])
        ## evaluate input lines sorted by timestamp
        lines = []
        lines.append('from spectr import *\n') # general import at beginning of formatted input
        previous_suboptimiser = None
        for i in np.argsort(timestamps):
            lines.append(functions[i]())
            ## separate with newlines if a new suboptimser
            if (previous_suboptimiser is not None
                and suboptimisers[i] is not previous_suboptimiser):
                lines.append('')
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

    # def __str__(self):
        # d = self.get_parameter_array()
        # return(d.format_table(
            # delimiter=' , ',
            # make_header=True,
            # unique_values_in_header=True,
            # quote_strings=True))

    # def output_to_directory(
            # self,
            # directory,
            # trash_existing_directory=False, # delete existing data, even if false overwriting may occur
    # ):
        # """Save results of model and optimisation to a directory."""
        # ## new input line
        # self.format_input_functions.append(
            # lambda directory=directory: f'{self.name}.output_to_directory({repr(directory)})')
        # directory = tools.expand_path(directory)
        # tools.mkdir_if_necessary(directory,trash_existing_directory=trash_existing_directory)
        # ## output self and all suboptimisers into a flat subdirectory
        # ## structure
        # used_subdirectories = []
        # for optimiser in self.get_all_suboptimisers():
            # subdirectory = directory+'/'+optimiser.name+'/'
            # tools.mkdir_if_necessary(subdirectory,trash_existing_directory=True)
            # if subdirectory in used_subdirectories:
                # raise Exception(f'Non-unique optimiser names producting subdirectory: {repr(subdirectory)}')
            # used_subdirectories.append(subdirectory)
            # tools.string_to_file(subdirectory+'/parameters.csv',optimiser.format_csv())
            # tools.string_to_file(subdirectory+'/input.py',optimiser.format_input())
            # if optimiser.residual is not None:
                # tools.array_to_file(subdirectory+'/residual' ,optimiser.residual,fmt='%+0.4e')
            # else:
                # tools.array_to_file(subdirectory+'/residual' ,[])
            # if optimiser.description is not None:
                # tools.string_to_file(subdirectory+'/README' ,str(optimiser.description))
            # for f in optimiser.output_to_directory_functions:
                # f(subdirectory)
        # ## symlink suboptimsers into subdirectories
        # for optimiser in self.get_all_suboptimisers():
            # for suboptimiser in optimiser.suboptimisers:
                # tools.mkdir_if_necessary(f'{directory}/{optimiser.name}/suboptimisers/')
                # os.symlink(
                    # f'../../{suboptimiser.name}',
                    # f'{directory}/{optimiser.name}/suboptimisers/{suboptimiser.name}',
                    # target_is_directory=True)

    # def plot_residual(self,ax=None,**plot_kwargs):
        # """Plot residual error."""
        # import matplotlib.pyplot as plt
        # if ax is None:
            # fig = plt.gcf()
            # ax = fig.gca()
        # plot_kwargs.setdefault('marker','o') 
        # # ax.plot(self._optimisation_function(self.parameters),**plot_kwargs)
        # ax.plot(self.combined_residual,**plot_kwargs)

    def monitor(self):
        """Run monitor functions."""
        for optimiser in self.get_all_suboptimisers():
            for f in optimiser.monitor_functions:
                f()

    def get_parameters(self):
        """Return alist of parameter objects in this optimiser and all
        suboptimisers."""
        retval = []
        unique_ids = []
        for optimiser in self.get_all_suboptimisers():
            for parameter in optimiser.parameters:
                if id(parameter) not in unique_ids:
                    retval.append(parameter)
                    unique_ids.append(id(parameter))
        return(retval)

    def has_changed(self):
        # if self.residual is None:
            # return(True)
        # for p in self.parameters:
        #     if p.timestamp>self.timestamp:
        #         return(True)
        # for o in self.suboptimisers:
        #     if o.has_changed:
        #         return(True)
        for p in self.get_parameters():
            if p.timestamp>self.timestamp:
                return(True)
        return(False)

    def construct(
            self,
            recompute_all=False,
            verbose=False,
    ):
        """Run all construct functions and return collected residuals."""
        ## nothing to be done, return cached residual immediately
        if not self.has_changed() and not recompute_all:
            return(self.combined_residual) # includes own residual and for all suboptimisers
        ## collect residuals from suboptimisers and self
        combined_residual = []  # from self and suboptimisers
        for optimiser in self.get_all_suboptimisers():
            if optimiser.has_changed() or recompute_all:
                if verbose:
                    print(f'constructing optimiser: {optimiser.name}')
                optimiser.residual = []
                for f in optimiser.construct_functions:
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
                ## optimiser.set_unchanged() #  to indicate that now it has been recomputed
                optimiser.timestamp = time.time()
            ## add resisudal to return value for optimisation, possibly rescaling it
            if optimiser.residual is not None:
                if optimiser.residual_scale_factor is not None:
                    combined_residual.append(optimiser.residual_scale_factor*np.array(optimiser.residual))
                else:
                    combined_residual.append(np.array(optimiser.residual))
        combined_residual = np.concatenate(combined_residual)
        self.combined_residual = combined_residual # this includes residuals from construct_functions combined with suboptimisers
        return(combined_residual)

    def _optimisation_function(self,p):
        """Internal function used by optimise routine. p is a list of varied
        parameters."""
        self._number_of_optimisation_function_calls += 1
        ## update parameters in internal model
        p = list(p)
        for t in self.get_parameters():
            if t.vary:
                t.value = p.pop(0)
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
            for suboptimiser in self.get_all_suboptimisers():
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
        p  = []
        for t in self.get_parameters():
            if t.vary:
                p.append(t.value)
            t.uncertainties = np.nan
        if verbose:
            print('Number of varied parameters:',len(p))
        if len(p)>0:
            ## 2018-05-08 on one occasion I seemed to be getting
            ## returned p from leastsq which did not correspond to the
            ## best fit!!! So I did not update p from this output and
            ## retained what was set in construct()
            try:
                p,dp = tools.leastsq(
                    self._optimisation_function,
                    p,
                    [t.step for t in self.get_parameters() if t.vary],
                    xtol=xtol,
                    rms_noise=rms_noise,
                )
                if self.verbose: print('Number of evaluations:',self._number_of_optimisation_function_calls)
                ## update parameters and uncertainties
                p,dp = list(p),list(dp)
                for t in self.get_parameters():
                    if t.vary:
                        t.p = p.pop(0)
                        t.dp = dp.pop(0)*np.sqrt(data_interpolation_factor) # not quite correct degneracy rescale for interpolation
            except KeyboardInterrupt:
                pass
        residual = self.construct(recompute_all=True) # run at least once, recompute_all to get uncertainties
        self.monitor() # run monitor functions after optimisation
        if verbose:
            print('total RMS:',np.sqrt(np.mean(np.array(self.combined_residual)**2)))
            for t in self.get_all_suboptimisers():
                print(f'  suboptimiser {t.name} RMS:',t.get_rms())
        return(residual) # returns residual array

    def get_rms(self):
        """Compute root-mean-square error."""
        if self.residual is None or len(self.residual)==0:
            return(None)
        retval = tools.rms(self.residual)
        # if self.residual_scale_factor is not None:
            # retval *= self.residual_scale_factor
        return(retval)

    # def get_parameter(self,name):
        # """Returns parameter matching_name in this optimiser (not in
        # suboptimisers). Error if name does not exist or is not
        # unique."""
        # retval = None
        # for p in self.parameters:
            # if p.name==name:
                # if retval is None:
                    # retval = p
                # else:
                    # raise Exception(f'The requested parameter name is not unique: {repr(parameter_name)}')
        # if retval is None:
            # raise Exception(f'There is no parameter with the requested name: {repr(parameter_name)}')
        # return(retval)

    # def __float__(self):
        # """If one and only only parameter is defined returns its
        # value."""
        # if len(self.parameters)==1:
            # return(float(self.parameters[0].p))
        # else:
            # raise Exception(f'Float conversion of Optimiser is only possible when it has exactly one parameter.')

    # def keys(self):
        # """Return list of parameter names. Error if the names are not
        # unique."""
        # keys = {p.name for p in self.parameters}
        # assert len(keys)==len(self.parameters),'Non-unique parameters names, cannot give a list of keys.'
        # return(keys)
           #  


class ParameterSet():

    """A collection of scalar or array values, possibly with uncertainties."""

    def __init__(self,description=None,**keys_args):
        self._data = dict()
        self.description = description
        for key,args in keys_args.items():
            self[key] = args

    def __setitem__(
            self,
            key,
            args,               # (value,vary,step,description)
    ):
        args = tools.ensure_iterable(args)
        if key not in self._data:
            ## add new data
            self._data[key] = Datum(
                description = (key
                               + (' '+self.description if self.description is not None else '')
                               + (' '+args[3] if len(args)>3 else '')),
                value=args[0],
                vary=(args[1] if len(args)>1 else None),
                step=(args[2] if len(args)>2 else None),
                kind=float,
                uncertainty=np.nan,)
        else:
            ## update existing data
            assert len(args)==1
            self._data[key].value = args[0]

    def _get_timestamp(self):
        return(max(data.timestamp for data in self._data.values()))

    timestamp = property(_get_timestamp)

    def __getitem__(self,key):
        return(self._data[key])

    def __iter__(self):
        for key in self._data:
            yield key

    def keys(self):
        return(self._data.keys())

    def values(self):
        return(self._data.values())

    def items(self):
        return(self._data.items())

    def __str__(self):
        return(
            tools.format_columns({
                    'description':[self[key].description for key in self],
                    'value':[self[key].value for key in self],
                    'uncertainty':[self[key].uncertainty for key in self],
                    'vary':[self[key].vary for key in self],
                    'step':[self[key].step for key in self],
                }, delimiter=' | ',fmt='>11.5g',comment_string=''))

    def save(self,filename='parameters.psv'):
        tools.string_to_file(filename,str(self))

    
