# from . import *
# from scipy import signal
# import numpy as np
# from copy imoprt
from matplotlib import pyplot as plt

from .dataset import *
from . import optimise
from . import plotting
from . import tools
# from . import optimise,plotting

class Spectrum(optimise.Optimiser):

    def __init__(self,name='spectrum',verbose=None,model_residual_weighting=None): 
        self.name = name        # object string -- warning this is evaluated as a variable name!!!
        self.xexp = None                             # x-data of spectrum to fit
        self.yexp = None                             # y-data of spectrum to fit
        self.xmod = None                             # x-data of spectrum to fit
        self.ymod = None                             # y-data of spectrum to fit
        self.model_residual = None                      # array of residual fit
        self.model_residual_weighting = model_residual_weighting            # weighting pointwise in xexp
        self.model_interpolate_factor = None
        self.verbose = bool(verbose)
        ## initialise an optimiser for the experimental spectrum
        self._exp = optimise.Optimiser(f'{self.name}_exp')
        self._exp.ignore_format_input_functions = True 
        def f():
            self.xexp,self.yexp = None,None
        self._exp.add_construct_function(f)
        ## initialise an optimiser for the model spectrum
        self._mod = optimise.Optimiser(f'{self.name}_mod')
        self._mod.ignore_format_input_functions = True
        self._mod.add_suboptimiser(self._exp)
        def f():
            assert self.xmod is not None
            self.ymod = np.zeros(self.xmod.shape)
        self._mod.add_construct_function(f)
        ## residual optimiser
        optimise.Optimiser.__init__(self,self.name)
        self.add_suboptimiser(self._mod,self._exp)
        def format_input_function():
            retval = f'{self.name} = Spectrum(name={repr(self.name)}'
            if verbose is not None:
                retval += f',verbose={repr(self.verbose)}'
            if model_residual_weighting is not None:
                retval += f',model_residual_weighting={repr(self.model_residual_weighting)}'
            retval += ')'
            return(retval)
        self.add_format_input_function(format_input_function)
        self.add_construct_function(self.construct_residual)
        self.add_save_to_directory_function(self.output_data_to_directory)
        self._figure = None
        self.verbose = verbose
        self.experimental_parameters = {} # a dictionary containing any additional known experimental parameters
        self._cache = {}

    def __len__(self):
        if self.xexp is not None:
            return(len(self.xexp))
        elif self.xmod is not None:
            return(len(self.xmod))
        else:
            raise Exception("No experimental or model spectrum to define length.")

    def set_experimental_spectrum(self,x,y,xbeg=None,xend=None):
        """Set x and y as the experimental spectrum. With various safety
        checks. Not optimisable, no format_input_function."""
        x,y = np.array(x),np.array(y)
        i = np.argsort(x); x,y = x[i],y[i] # sort
        ## set range
        if xbeg is None:
            xbeg = x[0]
        else:
            assert x[0]<=xbeg,'xbeg is outside range of spectrum: '+repr(xbeg)+' , '+repr(x[0])
            i = np.array(x>=xbeg)
            x,y = x[i],y[i]
        if xend is None:
            xend = x[-1]
        else:
            assert x[-1]>=xend,'xend is outside range of spectrum: '+repr(xend)
            i = np.array(x<=xend)
            x,y = x[i],y[i]
        self.experimental_parameters['xbeg'] = xbeg 
        self.experimental_parameters['xend'] = xend
        ## check for regular x grid
        t0,t1 = np.diff(x).min(),np.diff(x).max()
        assert (t1-t0)/t1<1e-3, 'Experimental data must be on an uniform grid.' # within a factor of 1e3
        ## add to self
        def f():
            self.xexp,self.yexp = copy(x),copy(y) # make copy -- more memory but survive other changes
            self.xmod = copy(self.xexp)
        self._exp.add_construct_function(f)
        if self.verbose:
            print('experimental_parameters:')
            pprint(self.experimental_parameters)

    def set_experimental_spectrum_from_object(
            self,
            spectrum_object,
            xbeg=None,xend=None,
    ):
        """Load a spectrum to fit from an x,y file."""
        self.format_input_functions.append(self.name+f'.set_experimental_spectrum_from_object(experimental_spectrum.name,xbeg={xbeg:g},xend={xend:g})')
        self.suboptimisers.append(spectrum_object)
        i = tools.inrange(spectrum_object.x,xbeg,xend) # constant length
        def f():
            self.set_experimental_spectrum(
                spectrum_object.x[i],
                spectrum_object.y[i])
        f()

    def set_experimental_spectrum_from_file(self,filename,xbeg=None,xend=None,**file_to_array_kwargs):
        """Load a spectrum to fit from an x,y file."""
        self.format_input_functions.append(self.name+f'.set_experimental_spectrum_from_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)},{tools.dict_to_kwargs(file_to_array_kwargs)})')
        x,y = tools.file_to_array_unpack(filename,**file_to_array_kwargs)
        self.experimental_parameters['filename'] = filename
        self.set_experimental_spectrum(x,y,xbeg,xend)

    def set_experimental_spectrum_from_opus_file(self,filename,xbeg=None,xend=None):
        """Load a spectrum in an Bruker opus binary file."""
        x,y,d = lib_molecules.load_bruker_opus_spectrum(filename)
        self.format_input_functions.append(self.name+f'.set_experimental_spectrum_from_opus_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)})')
        self.experimental_parameters['filename'] = filename
        d = lib_molecules.load_bruker_opus_data(filename)
        self.experimental_parameters['interpolation_factor'] = float(d['Fourier Transformation']['ZFF'])
        if d['Fourier Transformation']['APF'] == 'B3':
            self.experimental_parameters['apodisation_function'] = 'Blackman-Harris 3-term'
        else:
            warnings.warn(f"Unknown opus apodisation function: {repr(d['Fourier Transformation']['APF'])}")
        self.set_experimental_spectrum(x,y,xbeg,xend)

    def scale_xexp(self,scale=(1,True,1e-8)):
        """Rescale experimental spectrum x-grid."""
        p = self.add_parameter_set(scale=scale,note='scale_xexp')
        self.format_input_functions.append(lambda: f"{self.name}.scale_xexp({p.format_input()})")
        def construct_function():
            self.xexp *= p['scale']
        self._exp.add_construct_function(construct_function)
        
    def set_jh_inst_spectrum_from_file(self,filename_or_scan_index,xbeg=None,xend=None):
        """Load spectrum from the Jaroslav Heyrovský institute, or by index in summary_of_scans -- probably out of date."""
        if isinstance(filename_or_scan_index,str):
            self.set_experimental_spectrum_from_file(filename_or_scan_index,xbeg,xend) 
            self.format_input_functions.pop(-1) # since defined here
            self.format_input_functions.append(lambda: f'{self.name}.set_jh_inst_spectrum_from_file({repr(filename_or_scan_index)},xbeg={repr(xbeg)},xend={repr(xend)})')
        elif isinstance(filename_or_scan_index,int):
            if 'jh_inst_summary_of_scans' not in self._cache:
                self._cache['jh_inst_summary_of_scans'] = Dynamic_Recarray(load_from_filename='~/exp/jh-inst/PALS_HR_FTIR/scans/summary_of_scans.rs')
            decoded_filename_or_scan_index = '/home/heays/exp/jh-inst/PALS_HR_FTIR/scans/'+self._cache['jh_inst_summary_of_scans'].get_unique_value('filename',index=filename_or_scan_index)
            self.set_experimental_spectrum_from_file(decoded_filename_or_scan_index,xbeg,xend) 
            self.format_input_functions.pop(-1) # since defined here
            self.format_input_functions.append(lambda: f'{self.name}.set_jh_inst_spectrum_from_file({repr(filename_or_scan_index)},xbeg={repr(xbeg)},xend={repr(xend)}) # decoded filename: {decoded_filename_or_scan_index}')
        else:
            raise Exception

    def interpolate_spectrum(self,dx):
        """Interpolate experimental spectrum to a grid of width dx, may change
        position of xend."""
        self.format_input_functions.append(self.name+f'.interpolate_spectrum({repr(dx)})')
        xnew = np.arange(self.xexp[0],self.xexp[-1],dx)
        ynew = tools.spline(self.xexp,self.yexp,xnew)
        if self.verbose: print(f"Interpolating to grid ({repr(xnew[0])},{repr(xnew[-1])},{dx}) from grid ({repr(self.xexp[0])},{repr(self.xexp[-1])},{self.xexp[1]-self.xexp[0]})")
        self.xexp,self.yexp = xnew,ynew

    def interpolate_model_spectrum(self,dx):
        """When calculating model set to dx grid (or less to achieve
        overlap with experimental points."""
        self.format_input_functions.append(f'{self.name}.interpolate_model_spectrum({dx})')
        def f():
            xstep = (self.xexp[-1]-self.xexp[0])/(len(self.xexp)-1)
            self.model_interpolate_factor = int(np.ceil(xstep/dx))
            self.xmod = np.linspace(self.xexp[0],self.xexp[-1],1+(len(self.xexp)-1)*self.model_interpolate_factor)
        self._exp.add_construct_function(f)

    def add_absorption_cross_section_from_file(
            self,
            name,               # for new input line
            filename,           # the data filename, loaded with file_to_dict if xkey/ykey given else file_to_array
            column_density=1e16,              # to compute optical depth
            xshift=None, yshift=None,             # shiftt the data
            xscale=None, yscale=None, # scale the data
            xbeg=None, xend=None, # limits to add
            xkey=None, ykey=None, # in case file is indexable by keys
            xtransform=None,      # modify x data with this function
            resample_interval=None, # resample for some reason
            **file_to_dict_or_array_kwargs,
    ):
        """Load a cross section from a file. Interpolate this to experimental
        grid. Add absorption according to given column density, which
        can be optimised."""
        ## add adjustable parameters for optimisation
        p = self.add_parameter_set(
            note=f'add_absorption_cross_section_from_file name={name} file={filename}',
            column_density=column_density, xshift=xshift,
            yshift=yshift, xscale=xscale, yscale=yscale,
            step_scale_default={'column_density':0.01, 'xshift':1e-3, 'yshift':1e-3,
                                'xscale':1e-8, 'yscale':1e-3,})
        ## new input line
        def f(xbeg=xbeg,xend=xend):
            retval = f'{self.name}.add_absorption_cross_section_from_file({repr(name)},{repr(filename)}'
            if xbeg is not None:
                retval += f',xbeg={repr(xbeg)}'
            if xend is not None:
                retval += f',xend={repr(xend)}'
            if xkey is not None:
                retval += f',xkey={repr(xkey)}'
            if ykey is not None:
                retval += f',ykey={repr(ykey)}'
            if xtransform is not None:
                retval += f',xtransform={repr(xtransform)}'
            if len(p)>0:
                retval += f',{p.format_input()}'
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        if xkey is None and ykey is None:
            xin,σin = tools.file_to_array_unpack(filename,**file_to_dict_or_array_kwargs)
        elif xkey is not None and ykey is not None:
            t = tools.file_to_dict(filename,**file_to_dict_or_array_kwargs)
            xin,σin = t[xkey],t[ykey]
        else:
            raise Exception('Forbidden case.')
        if xtransform is not None:
            xin = getattr(my,xtransform)(xin)
        ## resample if requested to remove noise
        if resample_interval is not None:
            xt = np.arange(xin[0],xin[-1],resample_interval)
            xin,σin = xt,lib_molecules.resample_data(xin,σin,xt)
        ## set range if specified
        if xbeg is not None:
            i = xin>=xbeg
            xin,σin = xin[i],σin[i]
        if xend is not None:
            i = xin<=xend
            xin,σin = xin[i],σin[i]
        ## add to model
        cache = {}
        def f():
            if len(xin)==0:
                return # nothing to add
            ## compute transmission if necessary
            if 'transmission' not in cache or p.timestamp>self.timestamp:
                cache['transmission'] = np.exp(
                    -p['column_density']
                    *tools.spline(
                        xin*(p['xscale'] if xscale is not None else 1) - (p['xshift'] if xshift is not None else 0),
                        σin*(p['yscale'] if yscale is not None else 1) + (p['yshift'] if yshift is not None else 0),
                        self.xmod))
                for key in p.keys():
                    cache[key]  = p[key]
            self.ymod *= cache['transmission'] # add to model
        self._mod.add_construct_function(f)
    add_cross_section_from_file = add_absorption_cross_section_from_file # deprecated name

    def add_absorption_cross_section(
            self,
            cross_section_object,
            column_density=1e16,
            # xshift=None,
            # xscale=None,
            # xbeg=None,
            # xend=None,
    ):
        """Load a cross section from a file. Interpolate this to experimental
        grid. Add absorption according to given column density, which
        can be optimised."""
        p = self.add_parameter_set(
            note=f'add_absorption_cross_section {cross_section_object.name} in {self.name}',
            column_density=column_density,# xshift=xshift, xscale=xscale,
            step_scale_default={'column_density':0.01, 'xshift':1e-3, 'xscale':1e-8,})
        def f():
            retval = f'{self.name}.add_absorption_cross_section({cross_section_object.name}'
            # if xbeg is not None:
                # retval += f',xbeg={xbeg}'
            # if xend is not None:
                # retval += f',xend={xend}'
            if len(p)>0:
                retval += f',{p.format_input()}'
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        self.suboptimisers.append(cross_section_object)
        cache = {}
        def f():
            ## update if necessary
            if ('transmission' not in cache 
                or (p.timestamp>self.timestamp)
                or (cross_section_object.timestamp>self.timestamp)):
                # cache['transmission'] = np.full(self.xmod.shape,1.0)
                # i = np.full(self.xmod.shape,
                # if xbeg is not None:
                    # i &= self.xmod>=xbeg
                # if xend is not None:
                    # i &= self.xmod<=xend
                # cache['i'] = i
                cache['transmission'] = np.exp(-p['column_density']*cross_section_object.σ(self.xmod))
                for key in p.keys():
                    cache[key] = p[key]
            ## add to model
            self.ymod *= cache['transmission']
            # self.optical_depths[cross_section_object.name] = cache['τ']
        self._mod.add_construct_function(f)
    add_cross_section = add_absorption_cross_section # deprecated name

    def add_transition_from_file(
            self,
            filename, # must be a "transitions format" file
            vν=False, # vary frequency
            vf=False, # vary fvalue
            vΓ=False,
            νshift=0.,  # shift uniformly -- WARNING conflicts with vν
            fscale=1.,  # scale uniformly -- WARNING conflicts with vf
            **file_to_transition_kwargs,
    ):
        """Add all lines in a transition file."""
        p = self.add_parameter_set('add_transition_from_file',νshift=νshift,fscale=fscale,step_scale_default=1e-2)
        self.format_input_functions.append(lambda: f'{self.name}.add_transition_from_file({repr(filename)},vν={repr(vν)},vf={repr(vf)},vΓ={repr(vΓ)},{repr(p)},{tools.dict_to_kwargs(file_to_transition_kwargs)})')
        ## load transition
        transition = spectra.file_to_transition( 
            filename,
            temperaturepp=300., # default value
            column_densitypp=1., # default value
            dν=np.nan, df=np.nan, dΓ=np.nan, # ensure set
            **file_to_transition_kwargs
        )
        self.ftransition.append(transition) #  add to self
        j = slice(len(self.ftransition)-len(transition),len(self.ftransition)) # indices in transition of this new data
        ## deal with uniform shift and scale parameters
        p.νcache,p.fcache = copy(transition['ν']),copy(transition['f'])
        p.νshift_cache,p.fscale_cache = np.nan,np.nan
        def f():
            if p.νshift_cache!=p['νshift'] or p.fscale_cache!=p['fscale']:
                p.νshift_cache,p.fscale_cache = p['νshift'],p['fscale']
                self.ftransition['ν'][j] = p.νcache-p['νshift']
                self.ftransition['f'][j] = p.fcache*np.abs(p['fscale'])
        self._mod.add_construct_function(f)        
        ## add varied parameters to optimiser, get functions to update on optimisation
        i = (transition['ν'][j]>=self.xexp[0])&(transition['ν'][j]<=self.xexp[-1]) # in range for useful variation
        if vν:
            p = self.add_parameter_list(filename,transition['ν'][j][i],vν,step=0.01)
            def f(p=p): self.ftransition['ν'][j][i],self.ftransition['dν'][j][i] = p.plist,p.dplist
            self._mod.add_construct_function(f)
        if vf:
            p = self.add_parameter_list(filename,transition['f'][i],vf,step=transition['f'][i]*1e-2)
            def f(p=p): self.ftransition['f'][j][i],self.ftransition['df'][j][i] = np.abs(p.plist),p.dplist
            self._mod.add_construct_function(f)
        if vΓ:
            p = self.add_parameter_list(filename,transition['Γ'][j][i],vΓ,step=0.01)
            def f(p=p): self.ftransition['Γ'][j][i],self.ftransition['dΓ'][j][i] = np.abs(p.plist),p.dplist
            self._mod.add_construct_function(f)

    def add_transition(self,transition,name=None,**optimise_keys_vals):
        """Add all lines in a transition file. These are copied locally and
        then whatever is in optimise_keys_vals is applied each iteration, and
        then possibly optimised themselves. I.e., temperaturepp or
        column_densitypp. DEPRECATED, used add_absorption_transition"""
        warnings.warn('add_transition deprecated. Used add_absorption_transition or add_emission_transition')
        ## make local_transition depedent on the data in transition and any optimsied keys_vals
        if name is None:
            name =  f'{transition.name}_in_{self.name}'
        if transition['level_transition_type'] == 'Rotational_Transition':
            local_transition = Rotational_Transition(
                Name=name,
                partition_source=transition['partition_source'],
                # partition_sourcepp=transition['partition_sourcepp'],
                # partition_sourcep=transition['partition_sourcep'],
            )
        elif transition['level_transition_type'] == 'Atomic_Transition':
            local_transition = Atomic_Transition(Name=name)
        else:
            raise Exception("Do not know how to add transition of level_transition_type {repr(transition['level_transition_type'])}")
        local_transition.format_input_functions = [] # do not reproduce an input line for this
        ## add optimisable things
        p = local_transition.add_parameter_set(f'add_transition {transition.name} to {self.name}',**optimise_keys_vals) # optimise these with initial values
        local_transition.suboptimisers.append(transition)
        ## update local transition data from parent transition and
        def f():
            ## delete all data  if length changed -- a crude test of major changes
            if len(local_transition)!=len(transition):
                local_transition.clear()
                local_transition.set_length(len(transition))
            ## no data - do not try to set anything
            if len(transition)==0:
                return
            for key in transition.set_keys():
                if not local_transition.is_set(key) or np.any(local_transition[key]!=transition[key]):
                    local_transition[key] = transition[key]
            for key in p.keys():
                if not local_transition.is_set(key) or np.any(local_transition[key]!=p[key]):
                    local_transition[key] = p[key]
            ## indicate a changed optical depth
            # self.optical_depths[local_transition] = None 
        local_transition.construct_functions.append(f)
        ## add this as a suboptimiser to self
        self.transitions.append(local_transition)
        self.suboptimisers.append(local_transition)
        def f():
            retval = f'{self.name}.add_transition({transition.name},'
            retval += f'name={repr(name)},'
            retval += f'{p.format_input()})'
            return(retval)
        self.format_input_functions.append(f)

    def add_absorption_lines(
            self,
            lines,
            nfwhmL=20,
            nfwhmG=100,
            τmin=None,
            gaussian_method=None,
            voigt_method=None,
            use_multiprocessing=None,
            use_cache= None,
            **optimise_keys_vals):
        ## update optimsied things -- e.g., temperaturepp or
        ## column_densitypp
        name = f'add_absorption_lines {lines.name} to {self.name}'
        # assert name not in self.optical_depths,f'Non-unique name in optical_depths: {repr(name)}'
        # self.lines.append(lines)
        self._mod.add_suboptimiser(lines,add_format_function=False)
        parameter_set = self._mod.add_parameter_set(**optimise_keys_vals,description=name)
        ## update lines data and recompute optical depth if
        ## necessary
        cache = {}
        def f():
            ## first call -- no good, xmod not set yet
            if self.xmod is None:
                return
            if len(lines)==0:
                ## no lines
                return
            ## recompute spectrum if is necessary for somem reason
            if (cache == {}    # currently no spectrum computed
                or self.timestamp < lines.timestamp # lines has changed
                or self.timestamp < parameter_set.timestamp # optimise_keys_vals has changed
                or not (len(cache['xmod']) == len(self.xmod)) # experimental domain has changed
                or not np.all( cache['xmod'] == self.xmod )): # experimental domain has changed
                ## update optimise_keys_vals that have changed
                for key,val in parameter_set.items():
                    if (cache=={} or lines[key][0]!=val): # has been changed elsewhere, or the parameter has changed, or first use
                        lines[key] = val
                x,y = lines.calculate_spectrum(
                    x=self.xmod,
                    ykey='τpa',
                    nfwhmG=(nfwhmG if nfwhmG is not None else 10),
                    nfwhmL=(nfwhmL if nfwhmL is not None else 100),
                    ymin=τmin,
                    ΓG='ΓDoppler',
                    ΓL='Γ',
                    # gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran stepwise'),
                    gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran'),
                    voigt_method=(voigt_method if voigt_method is not None else 'wofz'),
                    # use_multiprocessing=(use_multiprocessing if use_multiprocessing is not None else False),
                    use_multiprocessing=(use_multiprocessing if use_multiprocessing is not None else 4),
                    use_cache=(use_cache if use_cache is not None else True),
                )
                cache['xmod'] = copy(self.xmod)
                cache['absorbance'] = np.exp(-y)
            ## absorb
            self.ymod *= cache['absorbance']
        self._mod.add_construct_function(f)
        ## new input line
        def f():
            retval = [f'{self.name}.add_absorption_lines({lines.name}']
            if len(parameter_set)>0:
                ## retval.append('\n    '+parameter_set.format_input(multiline=True))
                retval.append(parameter_set.format_input(multiline=False))
            if nfwhmL is not None:
                retval.append(f'nfwhmL={repr(nfwhmL)}')
            if nfwhmG is not None:
                retval.append(f'nfwhmG={repr(nfwhmG)}')
            if τmin is not None:
                retval.append(f'τmin={repr(τmin)}')
            if use_multiprocessing is not None:
                retval.append(f'use_multiprocessing={repr(use_multiprocessing)}')
            if use_cache is not None:
                retval.append(f'use_cache={repr(use_cache)}')
            if voigt_method is not None:
                retval.append(f'voigt_method={repr(voigt_method)}')
            if gaussian_method is not None:
                retval.append(f'gaussian_method={repr(gaussian_method)}')
            retval.append(')')
            return(','.join(retval))
        self.add_format_input_function(f)

    def add_emission_transition(
            self,
            transition,
            nfwhmL=None,
            nfwhmG=None,
            Imin=None,
            gaussian_method=None,
            voigt_method=None,
            use_multiprocessing=None,
            use_cache=None,
            **optimise_keys_vals):
        self.suboptimisers.append(transition) # to model rebuilt when transition changed
        self.transitions.append(transition)   
        name = f'add_emission_transition_transition {transition.name} to {self.name}'
        assert name not in self.emission_intensities,f'Non-unique name in emission_intensities: {repr(name)}'
        self.emission_intensities[name] = None
        p = self.add_parameter_set(**optimise_keys_vals,note=name)
        cache = {}
        def construct_function():
            ## first call -- no good, xmod not set yet
            if self.xexp is None:
                self.emission_intensities[name] = None
                return
            ## recompute spectrum
            if (cache =={}
                or self.emission_intensities[name] is None # currently no spectrum computed
                or self.timestamp<transition.timestamp # transition has changed
                or self.timestamp<p.timestamp     # optimise_keys_vals has changed
                or not (len(cache['xexp']) == len(self.xexp)) # experimental domain has changed
                or not np.all( cache['xexp'] == self.xexp )): # experimental domain has changed
                ## update optimise_keys_vals that have changed
                for key,val in p.items():
                    if (not transition.is_set(key)
                        or np.any(transition[key]!=val)):
                        transition[key] = val
                ## that actual computation
                x,y = transition.calculate_spectrum(
                    x=self.xmod,
                    ykey='I',
                    nfwhmG=(nfwhmG if nfwhmG is not None else 10),
                    nfwhmL=(nfwhmL if nfwhmL is not None else 100),
                    ymin=Imin,
                    ΓG='ΓDoppler',
                    ΓL='Γ',
                    # gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran stepwise'),
                    gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran'),
                    voigt_method=(voigt_method if voigt_method is not None else 'wofz'),
                    use_multiprocessing=(use_multiprocessing if use_multiprocessing is not None else False),
                    use_cache=(use_cache if use_cache is not None else True),
                )
                cache['xexp'] = copy(self.xexp)
                cache['intensity'] = y
            ## add emission intensity to the overall model
            self.ymod += cache['intensity']
        self._mod.add_construct_function(construct_function)
        ## new input line
        def f():
            retval = f'{self.name}.add_emission_transition({transition.name}'
            if nfwhmL is not None: retval += f',nfwhmL={repr(nfwhmL)}'
            if nfwhmG is not None: retval += f',nfwhmG={repr(nfwhmG)}'
            if Imin is not None: retval += f',Imin={repr(Imin)}'
            if use_multiprocessing is not None: retval += f',use_multiprocessing={repr(use_multiprocessing)}'
            if use_cache is not None: retval += f',use_cache={repr(use_cache)}'
            if voigt_method is not None: retval += f',voigt_method={repr(voigt_method)}'
            if gaussian_method is not None: retval += f',gaussian_method={repr(gaussian_method)}'
            if len(p)>0: retval += f',{p.format_input()}'
            return(retval+')')
        self.format_input_functions.append(f)
        
    def construct_experiment(self):
        """Load, calibrate, and preprocess experimental before fitting any
        model."""
        self._exp.construct()
        return(self.xexp,self.yexp)

    def construct_model(self,x):
        """Construct a model spectrum."""
        self.xmod = np.asarray(x,dtype=float)
        self._mod.construct()
        return(self.xmod,self.ymod)

    def construct_residual(self):
        self._exp.construct()
        self._mod.construct()
        ## construct experiment
        # self.construct_experiment()
        # # assert self.xexp is not None,'No experimental spectrum.'
        # if self.xexp is None:
            # return
        # ## construct model, on a temporary interpolated grid if necessary
        # if self.model_interpolate_factor!=1:
            # x = np.linspace(self.xexp[0],self.xexp[-1],1+(len(self.xexp)-1)*self.model_interpolate_factor)
        # else:
            # x = copy(self.xexp)
        # self.construct_model()

        # if self.model_interpolate_factor!=1:
            # self.xmod = self.xmod[::self.model_interpolate_factor]
            # self.ymod = self.ymod[::self.model_interpolate_factor]
        ## calculate residual and undo model interpolation for
        ## residual calculation
        if self.model_interpolate_factor is None:
            self.model_residual = self.yexp - self.ymod
        else:
            self.model_residual = self.yexp - self.ymod[::self.model_interpolate_factor]
        ## weight residual pointwise
        if self.model_residual_weighting is not None:
            self.model_residual *= self.model_residual_weighting
        return(self.model_residual)
    
    def load_SOLEIL_spectrum_from_file(
            self,
            filename,
            xbeg=None,
            xend=None,
            xscale=None,
    ):
        """ Load SOLEIL spectrum from file with given path."""
        x,y,header = load_SOLEIL_spectrum_from_file(filename)
        self.experimental_parameters['filename'] = filename
        self.experimental_parameters.update(header)
        p = self.add_parameter_set('load_SOLEIL_spectrum_from_file',xscale=xscale,step_default={'xscale':1e-8},)
        def f():
            retval = f'{self.name}.load_SOLEIL_spectrum_from_file({repr(filename)}'
            if xbeg is not None:
                retval += f',xbeg={copy(xbeg):g}'
            if xend is not None:
                retval += f',xend={copy(xend):g}'
            if xscale is not None:
                retval += f',{p.format_input()}'
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        ## Limit xbeg/xend to fall within limits of actual data. If
        ## there is no data then self.xexp and self.yexp are None
        if xbeg is None:
            xbeg = x.min()
        if xend is None:
            xend = x.max()
        if xbeg>x.max() or xend<x.min():
            self.xexp = self.yexp = None
            warnings.warn(f"SOLEIL spectrum has no data in x-range: {repr(filename)}")
            return
        xbeg,xend = max(xbeg,x.min()),min(xend,x.max())
        self.set_experimental_spectrum(x,y,xbeg=xbeg,xend=xend)
        if xscale is not None:
            i = (x>xbeg-10.)&(x<xend+10.)
            x,y = x[i],y[i]
            yscaled = [None]
            def f():
                if (xscale is not None
                    and (yscaled[0] is None
                         or p.timestamp>self.timestamp)):
                    yscaled[0] = tools.spline(x*p['xscale'],y,x)
                    self.set_experimental_spectrum(x,yscaled[0],xbeg=xbeg,xend=xend)
            self._exp.add_construct_function(f)

    def fit_noise(
            self,
            xbeg,xend,
            n=1,
            figure_number=None,
            interpolation_factor=None,
    ):
        """Estimate the noise level by fitting a polynomial of order n
        between xbeg and xend to the experimental data. Also rescale
        if the experimental data has been interpolated."""
        ## new input line
        self.format_input_functions.append(
            lambda: f'{self.name}.fit_noise({xbeg},{xend},n={n},figure_number={figure_number})')
        self.construct_experiment() # need experimental data to be already loaded
        ## deal with interpolation factor (4 means 3 intervening extra points added)
        if (interpolation_factor is None 
             and 'interpolation_factor' not in self.experimental_parameters):
            interpolation_factor = 1
        elif (interpolation_factor is None 
              and 'interpolation_factor' in self.experimental_parameters):
            interpolation_factor = self.experimental_parameters['interpolation_factor']
        elif (interpolation_factor is not None 
              and 'interpolation_factor' in self.experimental_parameters
              and interpolation_factor != self.experimental_parameters['interpolation_factor']):
            raise Exception(f'interpolation_factor={repr(interpolation_factor)} does not match the value in self.experimental_parameters={self.experimental_parameters["interpolation_factor"]}')
        assert interpolation_factor>=1,'Down sampling will cause problems in this method.'
        if interpolation_factor!=1:
            print(f'warning: {self.name}: RMS rescaled to account for data interpolation_factor = {interpolation_factor}')
        ## compute noise resiudal of polynomial fit
        i = (self.xexp>xbeg)&(self.xexp<xend)
        if sum(i)==0:
            warnings.warn(f'{self.name}: No data in range for fit_noise, not done.')
            return
        x,y = self.xexp[i],self.yexp[i]
        xt = x-x.mean()
        p = np.polyfit(xt,y,n)
        yf = np.polyval(p,xt)
        r = y-yf
        rms = np.sqrt(np.mean(r**2))
        ## set residual scale factor bearing in mind the rms and
        ## amount of model interpolation.  This is a bit of a hack to
        ## cancel out the incorrect sqrt(n) underestimate of
        ## uncertainty where interpolation artificially increases the
        ## apparent degrees of freedom in the fit but without adding
        ## any new independent data.
        self.residual_scale_factor = 1/rms*np.sqrt(interpolation_factor)
        ## plot to check it looks ok
        if figure_number is not None:
            fig,ax = tools.fig(figure_number)
            ax.plot(x,y,label='exp')
            ax.plot(x,yf,label='fit')
            ax.plot(x,r,label=f'residual, rms={rms}')
            ax.set_title(f'fit rms to data\n{self.name}')
            tools.legend(ax=ax)
            ax = tools.subplot(fig=fig)
            ax.plot(tools.autocorrelate(r),marker='o',)
            ax.set_title(f'noise autocorrelation\n{self.name}')
            ax = tools.subplot(fig=fig)
            tools.plot_hist_with_fitted_gaussian(r,ax=ax)
            ax.set_title(f'noise distribution\n{self.name}')
        
    def set_residual_weighting(self,weighting,xbeg=None,xend=None):
        """Set the weighting or residual between xbeg and xend to a
        constant."""
        def f():
            retval = f'{self.name}.set_residual_weighting({repr(weighting)}'
            if xbeg is not None:
                retval += f',{repr(xbeg)}'
            if xend is not None:
                retval += f',{repr(xend)}'
            return retval+')'
        self.format_input_functions.append(f)
        def f():
            if xbeg is None and xend is None:
                self.model_residual_weighting = np.full(self.xexp.shape,weighting,dtype=float)
            else:
                if self.model_residual_weighting is None:
                    self.model_residual_weighting = np.ones(self.xexp.shape,dtype=float)
                self.model_residual_weighting[
                    (self.xexp>=(xbeg if xbeg is not None else -np.inf))
                    &(self.xexp<=(xend if xend is not None else np.inf))
                ] = weighting
        self._mod.add_construct_function(f)

    def set_residual_weighting_over_range(self,xbeg,xend,weighting):
        """Set the weighting or residual between xbeg and xend to a
        constant."""
        warnings.warn("set_residual_weighting_over_range is deprecated, use set_residual_weighting")
        self.set_residual_weighting(weighting,xbeg,xend)
        
    def autodetect_lines(
            self,
            filename=None,            # x,y data showing lines and no background, default to experimental data
            τ = None,                 # fix strength to this
            Γ = 0.,                   # fix width to this
            vν = False,         # whether estimated line is optimised
            vτ = False,         # whether estimated line is optimised
            vΓ = False,         # whether estimated line is optimised
            τscale = 1,            # scale estimated strengths by this amount -- usually necessary for absorption lines
            xmin = 0., # ignore peaks closer together than this -- just estimate one line
            ymin = None, # ignore peaks closer to zero than this, defaults to a fraction of the peak absolute residual
            **qn,        # anything else describing this 
    ):
        """Autodetect lines in experimental spectrum."""
        self.format_input_functions.append('# '+self.name+f'.autodetect_lines(filename={repr(filename)},τ={τ},Γ={Γ},vν={repr(vν)},vτ={repr(vτ)},vΓ={repr(vΓ)},τscale={repr(τscale)},xmin={repr(xmin)},ymin={repr(ymin)},'+tools.dict_to_kwargs(qn)+')')
        ## get something to find lines in
        if filename is None:
            x = copy(self.xexp)
            if self.model_residual is not None: y = copy(self.model_residual) # get from residual
            else:    y = copy(self.yexp-self.yexp.mean()) # get from data after removing mean / background
        else:
            x,y = tools.file_to_array_unpack(filename) # else get from a specified data file
        y = np.abs(y)      # to fit both emission and absorption lines
        i =  list(np.where( (y[1:-1]>y[0:-2]) & (y[1:-1]>y[2:]) )[0]+1) # get local maxima
        ## find equal neighbouring points that make a local maximum
        j = list(np.where(y[1:]==y[:-1])[0])
        while len(j)>0:
            jj = j.pop(0)
            kk = jj + 1
            if kk+1>=len(y): break
            while y[kk+1]==y[jj]:
                j.pop(0)
                kk = kk+1
                if kk==len(y): break
            if jj==0: continue
            if kk==len(y): continue
            if (y[jj]>y[jj-1])&(y[kk]>y[kk+1]):  i.append(int((jj+kk)/2.))
        i = np.sort(np.array(i))
        if ymin is None: ymin = y.max()*0.3 # exclude those that are too weak
        i = i[np.abs(y[i])>ymin]
        ## reject duplicates that are too close together
        if xmin>0:
            while True:
                jj = np.where(np.diff(x[i]) < minX)[0]
                if len(jj)==0: break
                for j in jj:
                    if ys[j]>ys[j+1]:   i[j+1] = -1
                    else:               i[j] = -1
                i = [ii for ii in i if ii!=-1]
        ## estimate line strength and width
        lines = []              # lines to add
        for (i,a,b) in zip(
                i,              # the peak position
                np.ceil(0.5*(i+np.concatenate(([0],i[:-1])))).astype(int), # halfway to peak position below
                np.floor(0.5*(i+np.concatenate((i[1:],[len(x)])))).astype(int), # halfway to peak position above
        ):
            ## estimate center of line as peak
            ν = x[i]
            ## estimate width of line
            xi,yi = x[a:b],y[a:b]
            hm = (yi.max()+yi.min())/2.
            ihm = np.argwhere(((yi[1:]-hm)*(yi[:-1]-hm))<0)
            Γestimate = 2*np.min(np.abs(xi[ihm]-ν))
            if Γ is None: Γ = Γestimate # use estimated total width as natural width
            if τ is None: τ = 2*y[i]*Γestimate # use estimated strength of line
            # self.add_τline(ν=(ν,vν),τ=(τ,vτ),Γ=(Γ,vΓ),**qn) # add to model
            lines.append(dict(ν=(ν,vν),τ=(τ,vτ),Γ=(Γ,vΓ)))
        self.add_lines(*lines,**qn) # add to self via add_lines
        if self.verbose: print("autodetect_lines added",i+1,"lines")

    # def convert_to_emission_intensity(
            # self,
            # widths_cutoff=np.inf,
            # minimum_strength=-np.inf,
            # use_multiprocessing=False,
    # ):
        # """Compute total emission intensitiy."""
        # self.format_input_functions.append(lambda: f'{self.name}.convert_to_emission_intensity(widths_cutoff={repr(widths_cutoff)},minimum_strength={repr(minimum_strength)})')
        # def f():
            # ## Add all emission intensities to ymod, recomputing them if
            # ## necessary.
            # for transition,I in self.emission_intensities.items():
                # if I is None:
                    # ν,I = transition.calculate_spectrum(
                        # self.xmod,
                        # ykey='Ae',
                        # nfwhmG=10,
                        # nfwhmL=widths_cutoff,
                        # ymin=minimum_strength,
                        # method='wofz',
                        # # method='fortran',
                        # use_multiprocessing=use_multiprocessing,
                        # use_cache=False,
                    # )
                    # self.emission_intensities[transition] = I
                # self.ymod += self.emission_intensities[transition]
        # self._mod.add_construct_function(f)

    # def convert_to_optical_depth(
            # self,
            # widths_cutoff=np.inf,
            # minimum_strength=-np.inf,
            # use_multiprocessing=False,
    # ):
        # """Compute optical depth of self.transition."""
        # self.format_input_functions.append(lambda: f'{self.name}.convert_to_optical_depth(widths_cutoff={repr(widths_cutoff)},minimum_strength={repr(minimum_strength)})')
        # def f():
            # ## ensure computed optical depth
            # for t in self.transitions:
                # if t not in self.optical_depths or self.optical_depths[t] is None:
                    # ## Can use Fortran once under-resolved Lorentzian's are accounted for -- it is about 3 times faster
                    # ## ν,τ = t.calculate_spectrum(self.xmod,ykey='τ',nfwhmG=10,nfwhmL=widths_cutoff,ymin=minimum_strength,method='wofz',)
                    # ν,τ = t.calculate_spectrum(
                        # self.xmod,
                        # ykey='τ',
                        # nfwhmG=10,
                        # nfwhmL=widths_cutoff,
                        # ymin=minimum_strength,
                        # voigt_method='wofz',
                        # # method='fortran',
                        # use_multiprocessing=use_multiprocessing,
                        # use_cache=False,
                    # )
                    # self.optical_depths[t] = τ
                # self.ymod += self.optical_depths[t]
        # self._mod.add_construct_function(f)
    
    def convert_to_transmission(self):
        """Convert optical depths calculated thus far to a transmission spectrum."""
        ## create the continuum object
        self.format_input_functions.append(self.name+'.convert_to_transmission()')
        def f():
            self.ymod = np.exp(-self.ymod)
        self._mod.add_construct_function(f)
        
    def scale_by_constant(self,scale=1.0):
        """Scale model by a constant value."""
        scale = self.add_parameter('scale',scale)
        self.format_input_functions.append(lambda: f"{self.name}.scale_by_constant(ν={repr(scale)})")
        def construct_function():
            self.ymod *= scale.p
        self._mod.add_construct_function(construct_function)

    def scale_by_spline(self,ν=50,amplitudes=1,vary=True,step=0.0001,order=3):
        """Scale by a spline defined function."""
        if np.isscalar(ν):           ν = np.arange(self.xexp[0]-ν,self.xexp[-1]+ν*1.01,ν) # default to a list of ν with spacing given by ν
        if np.isscalar(amplitudes):  amplitudes = amplitudes*np.ones(len(ν)) # default amplitudes to list of hge same length
        ν,amplitudes = np.array(ν),np.array(amplitudes)
        p = self.add_parameter_list(f'scale_by_spline',amplitudes,vary,step) # add to optimsier
        def format_input_function():
            retval = f"{self.name}.scale_by_spline("
            retval += f"vary={repr(vary)}"
            retval += f",ν=["+','.join([format(t,'0.0f') for t in ν])+']'
            retval += f",amplitudes=["+','.join([format(t.p,'0.4f') for t in p])+']'
            retval += f",step={repr(step)}"
            retval += f",order={repr(order)}"
            return(retval+')')
        self.format_input_functions.append(format_input_function)
        # self.format_input_functions.append(lambda: f"{self.name}.scale_by_spline(ν={repr(list(ν))},amplitudes={repr(p.plist)},vary={repr(vary)},step={repr(step)},order={repr(order)})") # new input line
        def f():
            i = (self.xmod>=np.min(ν))&(self.xmod<=np.max(ν))
            self.ymod[i] = self.ymod[i]*scipy.interpolate.UnivariateSpline(ν,p.plist,k=min(order,len(ν)-1),s=0)(self.xmod[i])
        self._mod.add_construct_function(f) # multiply spline during construct

    def modulate_by_spline(
            self,
            ν=None,
            amplitude=None,
            phase=None,         # if constant then will be used as a frequency in cm-1
            step_amplitude=1e-3,
            vary_amplitude=False,
            step_phase=1e-3,
            vary_phase=False,
            verbose=False,
            fbeg=-np.inf, fend=-np.inf, # estimate range of frequency for auto fitting
    ):
        """Modulate by 1 + sinusoid."""
        ## if scalar then use as stepsize of a regular grid
        if ν is None:
            ν = np.linspace(self.xexp[0],self.xexp[-1],10)
        elif np.isscalar(ν):
            ν = np.arange(self.xexp[0]-ν/2,self.xexp[-1]+ν/2+1,ν)
        else:
            ν = np.array(ν)
        ## if no amplitude given default to 1%
        if amplitude is None:
            amplitude = np.full(ν.shape,1e-2)
        elif np.isscalar(amplitude):
            amplitude = np.full(ν.shape,amplitude)
        ## if no phase default to frequency of 1 cm-1 if scalar use as
        ## frequency 
        if phase is None:
            if verbose:
                ax = tools.fig(880).gca()
            phase = np.zeros(ν.shape,dtype=float)
            for i in range(len(ν)-1):
                νbeg,νend = ν[i],ν[i+1]
                j = tools.inrange(self.xexp,νbeg,νend)
                tf,tF,r = tools.power_spectrum(
                    self.xexp[j],self.yexp[j],
                    fit_peaks=True,
                    xbeg=fbeg,xend=fend)
                f0 = r['f0'][np.argmax(r['S'])]
                phase[i+1] = phase[i] + 2*constants.pi*f0*(ν[i+1]-ν[i])
                if verbose:
                    print(f'{ν[i]:10.2f} {ν[i+1]:10.2f} {f0:10.4f}')
                    ax.plot([νbeg,νend],[f0,f0],color=plotting.newcolor(0),marker='o')
                    ax.set_xlabel('ν')
                    ax.set_ylabel('f')
        elif np.isscalar(phase):
            phase = 2*constants.pi*(ν-self.xexp[0])/phase
        amplitude = self.add_parameter_list('amplitude', amplitude, vary_amplitude, step_amplitude,note='modulate_by_spline')
        phase = self.add_parameter_list('phase', phase, vary_phase, step_phase,note='modulate_by_spline')
        def format_input_function():
            retval = f"{self.name}.modulate_by_spline("
            retval += f"vary_amplitude={repr(vary_amplitude)}"
            retval += f",vary_phase={repr(vary_phase)}"
            retval += f",ν=["+','.join([format(t,'0.4f') for t in ν])+']'
            retval += f",amplitude=["+','.join([format(p.p,'0.4f') for p in amplitude])+']'
            retval += f",phase=["+','.join([format(p.p,'0.4f') for p in phase])+']'
            retval += f",step_amplitude={repr(step_amplitude)}"
            retval += f",step_phase={repr(step_phase)}"
            return(retval+')')
        self.format_input_functions.append(format_input_function)
        def f():
            self.ymod *= 1. + tools.spline(ν,amplitude.plist,self.xmod)*np.sin(tools.spline(ν,phase.plist,self.xmod))
        self._mod.add_construct_function(f)

    # def modulate(self,*modulations,νbeg=None,νend=None,):
        # """Modulate by 1 + sinusoid."""
        # amplitude,frequency,phase = [],[],[]
        # modulations = list(modulations)
        # for i,m in enumerate(copy(modulations)):
            # assert len(m)==3
            # modulations[i] = (
                # self.add_parameter(f'frequency_{i}',*tools.ensure_iterable(m[0]),note='modulate'),
                # self.add_parameter(f'amplitude_{i}',*tools.ensure_iterable(m[1]),note='modulate'),
                # self.add_parameter(f'phase_{i}'        ,*tools.ensure_iterable(m[2]),note='modulate'))
        # def format_input_function():
            # retval = f"{self.name}.modulate(\n    "
            # retval += f",\n    ".join([repr(m) for m in modulations])+',\n'
            # if νbeg is not None:
                # retval += f"νbeg={νbeg},"
            # if νend is not None:
                # retval += f"νend={νend},"
            # return(retval+')')
        # self.format_input_functions.append(format_input_function)
        # def f():
            # iν = tools.inrange(self.xmod, (-np.inf if νbeg is None else νbeg), (np.inf if νend is None else νend),)
            # ν = self.xmod[iν]
            # modulation = np.zeros(ν.shape,dtype=float)
            # for f,a,p in modulations:
                # modulation += a.p*np.sin(2*constants.pi*f.p*ν+p.p)
            # self.ymod[iν] *= 1. + modulation
        # self._mod.add_construct_function(f)

    def scale_by_polynomial(self,center=None,p0=0,p1=0,p2=0,p3=0):
        """Multiply transmission by a source intensity."""
        ## deal with inputs
        if center is None: center = 0.5*(self.xexp[0]+self.xexp[-1])
        p = self.add_parameter_set('scale_by_polynomial',p0=p0,p1=p1,p2=p2,p3=p3,step_scale_default=0.1)
        if center<(self.xexp[0]-100) or center>(self.xexp[-1]+100): warnings.warn('scale_by_polynomial center far outside of x-range')
        self.format_input_functions.append(lambda: f"{self.name}.scale_by_polynomial(center={center:0.8g},\n{p.format_multiline(neglect_fixed_zeros=True)}\n)")
        def f():
            self.ymod = self.ymod*np.polyval([p['p3'],p['p2'],p['p1'],p['p0']],self.xmod-center)
        self._mod.add_construct_function(f)

    def add_intensity(self,intensity=1):
        """Shift by a spline defined function."""
        intensity = self._mod.add_parameter('intensity',*tools.ensure_iterable(intensity))
        self.add_format_input_function(lambda: f"{self.name}.add_intensity(intensity={repr(intensity)}")
        def f():
            self.ymod += float(intensity)
        self._mod.add_construct_function(f)

    def add_intensity_spline(self,x=50,y=0,vary=True,step=None,order=3):
        """Shift by a spline defined function."""
        if np.isscalar(x):
            self.construct_experiment()
            x = np.arange(self.xexp[0]-x,self.xexp[-1]+x*1.01,x) # default to a list of x with spacing given by x
        if np.isscalar(y):
            y = y*np.ones(len(x)) # default y to list of hge same length
        if step is None:
            if self.yexp is not None:
                step = self.yexp.max()*1e-3
            else:
                step = 1e-3
        x,y = np.array(x),np.array(y)
        p = self.add_parameter_list('shift_by_spline',y,vary,step) # add to optimsier
        self.format_input_functions.append(lambda: f"{self.name}.add_intensity_spline(vary={repr(vary)},x={repr(list(x))},y={repr(p.plist)},step={repr(step)},order={repr(order)})") # new input line
        def f():
            i = (self.xmod>=np.min(x))&(self.xmod<=np.max(x))
            self.ymod[i] += tools.spline(x,p.plist,self.xmod[i],order=order)
        self._mod.add_construct_function(f) # multiply spline during construct

    shift_by_spline = add_intensity_spline # deprecated

    def shift_by_constant(self,shift=0.):
        """Shift by a constant amount."""
        shift = self.add_parameter('shift_by_constant',*tools.ensure_iterable(shift)) # add to optimsier
        self.format_input_functions.append(lambda: f"{self.name}.shift_by_constant({repr(shift)})")
        def f():
            self.ymod += shift.p
        self._mod.add_construct_function(f) # multiply spline during construct

    def shift_by_polynomial(self,center=None,p0=0,p1=0,p2=0,p3=0):
        """Multiply transmission by a source intensity."""
        ## deal with inputs
        if center is None: center = 0.5*(self.xexp[0]+self.xexp[-1])
        p = self.add_parameter_set('shift_by_polynomial',p0=p0,p1=p1,p2=p2,p3=p3,step_scale_default=0.1)
        if center<(self.xexp[0]-100) or center>(self.xexp[-1]+100): warnings.warn('shift_by_polynomial center far outside of x-range')
        self.format_input_functions.append(lambda: f"{self.name}.shift_by_polynomial(center={center:0.8g},\n{p.format_multiline(neglect_fixed_zeros=True)}\n)")
        def f(): self.ymod += np.polyval([p['p3'],p['p2'],p['p1'],p['p0']],self.xmod-center)
        self._mod.add_construct_function(f)

    def scale_by_source_from_file(self,filename,scale_factor=1.):
        p = self.add_parameter_set('scale_by_source_from_file',scale_factor=scale_factor,step_scale_default=1e-4)
        self.format_input_functions.append(lambda: f"{self.name}.scale_by_source_from_file({repr(filename)},{p.format_input()})")
        x,y = tools.file_to_array_unpack(filename)
        scale = tools.spline(x,y,self.xexp)
        def f(): self.ymod *= scale*p['scale_factor']
        self._mod.add_construct_function(f)

    def convolve_with_gaussian(self,width,fwhms_to_include=100):
        """Convolve with gaussian."""
        p = self.add_parameter_set('convolve_with_gaussian',width=width,step_default={'width':0.01})
        self.format_input_functions.append(lambda: f'{self.name}.convolve_with_gaussian({p.format_input()})')
        ## check if there is a risk that subsampling will ruin the convolution
        def f():
            x,y = self.xmod,self.ymod
            width = np.abs(p['width'])
            if self.verbose and width<3*np.diff(self.xexp).min(): warnings.warn('Convolving gaussian with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1)
            padding = np.arange(dx,fwhms_to_include*width+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate gaussian to convolve with
            xconv = np.arange(-fwhms_to_include*width,fwhms_to_include*width,dx)
            if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
            yconv = np.exp(-(xconv-xconv.mean())**2*4*np.log(2)/width**2) # peak normalised gaussian
            yconv = yconv/yconv.sum() # sum normalised
            ## convolve and return, discarding padding
            self.ymod = scipy.signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self._mod.add_construct_function(f)

    def convolve_with_lorentzian(self,width,fwhms_to_include=100):
        """Convolve with lorentzian."""
        p = self.add_parameter_set('convolve_with_lorentzian',width=width,step_default={'width':0.01})
        self.format_input_functions.append(lambda: f'{self.name}.convolve_with_lorentzian({p.format_input()})')
        ## check if there is a risk that subsampling will ruin the convolution
        def f():
            x,y = self.xmod,self.ymod
            width = np.abs(p['width'])
            if self.verbose and width<3*np.diff(self.xexp).min(): warnings.warn('Convolving Lorentzian with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1)
            padding = np.arange(dx,fwhms_to_include*width+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate function to convolve with
            xconv = np.arange(-fwhms_to_include*width,fwhms_to_include*width,dx)
            if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
            yconv = lineshapes.lorentzian(xconv,xconv.mean(),1.,width)
            yconv = yconv/yconv.sum() # sum normalised
            ## convolve and return, discarding padding
            self.ymod = scipy.signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self._mod.add_construct_function(f)

    def convolve_with_sinc(self,width=None,fwhms_to_include=100):
        """Convolve with sinc function, width is FWHM."""
        ## check if there is a risk that subsampling will ruin the convolution
        p = self.add_parameter_set('convolve_with_sinc',width=width)
        if 'sinc_FWHM' in self.experimental_parameters: # get auto width and make sure consistent with what is given
            if width is None: width = self.experimental_parameters['sinc_FWHM']
            if np.abs(np.log(p['width']/self.experimental_parameters['sinc_FWHM']))>1e-3: warnings.warn(f"Input parameter sinc FWHM {repr(p['width'])} does not match experimental_parameters sinc_FWHM {repr(self.experimental_parameters['sinc_FWHM'])}")
        self.format_input_functions.append(lambda: f'{self.name}.convolve_with_sinc({p.format_input()})')
        if self.verbose and p['width']<3*np.diff(self.xexp).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        def f():
            x,y = self.xmod,self.ymod
            width = np.abs(p['width'])
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1) # ASSUMES EVEN SPACED GRID
            padding = np.arange(dx,fwhms_to_include*width+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate sinc to convolve with
            xconv = np.arange(-fwhms_to_include*width,fwhms_to_include*width,dx)
            if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
            yconv = np.sinc((xconv-xconv.mean())/width*1.2)*1.2/width # unit integral normalised sinc
            yconv = yconv/yconv.sum() # sum normalised
            ## convolve and return, discarding padding
            self.ymod = scipy.signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self._mod.add_construct_function(f)

    def convolve_with_instrument_function(
            self,
            sinc_fwhm=0,
            gaussian_fwhm=0,
            lorentzian_fwhm=0,
            signum_magnitude=0,
            sinc_fwhms_to_include=200,
            gaussian_fwhms_to_include=10,
            lorentzian_fwhms_to_include=10,
    ):
        """Convolve with sinc function, width is FWHM."""
        ## check if there is a risk that subsampling will ruin the convolution
        p = self.add_parameter_set(
            'convolve_with_instrument_function',
            sinc_fwhm=sinc_fwhm,
            gaussian_fwhm=gaussian_fwhm,
            lorentzian_fwhm=lorentzian_fwhm,
            signum_magnitude=signum_magnitude,
            step_default={
                'sinc_fwhm':1e-3,
                'gaussian_fwhm':1e-3,
                'lorentzian_fwhm':1e-3,
                'signum_magnitude':1e-4,
            },)
        ## get auto width from experimental data and make sure
        ## consistent with what is given in the input of this function
        if 'sinc_FWHM' in self.experimental_parameters:
            if p['sinc_fwhm']==0:
                p.get_parameter('sinc_fwhm').p = self.experimental_parameters['sinc_FWHM']
            else:
                if np.abs(np.log(p['sinc_fwhm']/self.experimental_parameters['sinc_FWHM']))>1e-3:
                    warnings.warn(f"Input parameter sinc FWHM {repr(p['sinc_fwhm'])} does not match experimental_parameters sinc_FWHM {repr(self.experimental_parameters['sinc_FWHM'])}")
        ## rewrite input line
        def f():
            # retval = f'{self.name}.convolve_with_instrument_function({p.format_input()}'
            retval = f'{self.name}.convolve_with_instrument_function({p.format_multiline()}'
            if p['sinc_fwhm']      !=0: retval += f',sinc_fwhms_to_include={sinc_fwhms_to_include}'
            if p['gaussian_fwhm']  !=0: retval += f',gaussian_fwhms_to_include={gaussian_fwhms_to_include}'
            if p['lorentzian_fwhm']!=0: retval += f',lorentzian_fwhms_to_include={lorentzian_fwhms_to_include}'
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        # if self.verbose and p['width']<3*np.diff(self.xexp).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        instrument_function_cache = dict(y=None,width=None,) # for persistence between optimsiation function calls
        def f():
            dx = (self.xmod[-1]-self.xmod[0])/(len(self.xmod)-1) # ASSUMES EVEN SPACED GRID
            ## get cached instrument function or recompute
            # if instrument_function_cache['y'] is None or p.has_changed():
            if instrument_function_cache['y'] is None or p.timestamp>self.timestamp:
                ## get total extent of instrument function
                width = abs(p['sinc_fwhm'])*sinc_fwhms_to_include + abs(p['gaussian_fwhm'])*gaussian_fwhms_to_include
                instrument_function_cache['width'] = width
                ## if necessary compute instrument function on a reduced
                ## subsampling to ensure accurate convolutions -- this wont
                ## help with an accurate convolution against the actual
                ## data!
                required_points_per_fwhm = 10
                subsample_factor = int(np.ceil(max(
                    1,
                    (required_points_per_fwhm*dx/p['sinc_fwhm'] if p['sinc_fwhm']!=0. else 1),
                    (required_points_per_fwhm*dx/p['gaussian_fwhm'] if p['gaussian_fwhm']!=0. else 1),
                    )))
                ## create the instrument function on a regular grid -- odd length with 0 in the middle
                x = np.arange(0,width+dx/subsample_factor*0.5,dx/subsample_factor,dtype=float)
                x = np.concatenate((-x[-1:0:-1],x))
                imidpoint = int((len(x)-1)/2)
                ## initial function is a delta function
                y = np.full(x.shape,0.)
                y[imidpoint] = 1.
                ## convolve with sinc function
                if p['sinc_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.sinc(x,Γ=abs(p['sinc_fwhm'])),'same')
                ## convolve with gaussian function
                if p['gaussian_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.gaussian(x,Γ=abs(p['gaussian_fwhm'])),'same')
                ## convolve with lorentzian function
                if p['lorentzian_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.lorentzian(x,Γ=abs(p['lorentzian_fwhm'])),'same')
                ## if necessary account for phase correction by convolving with a signum
                if p['signum_magnitude']!=0:
                    ty = 1/x*p['signum_magnitude'] # hyperbolically decaying signum on either side
                    ty[imidpoint] = 1 # the central part  -- a delta function
                    y = signal.convolve(y,ty,'same')
                ## convert back to data grid if it has been subsampled
                if subsample_factor!=1:
                    a = y[imidpoint-subsample_factor:0:-subsample_factor][::-1]
                    b = y[imidpoint::subsample_factor]
                    b = b[:len(a)+1] 
                    y = np.concatenate((a,b))
                ## normalise 
                y = y/y.sum()
                instrument_function_cache['y'] = y
            ## convolve model with instrument function
            padding = np.arange(dx,instrument_function_cache['width']+dx, dx)
            xpad = np.concatenate((self.xmod[0]-padding[-1::-1],self.xmod,self.xmod[-1]+padding))
            ypad = np.concatenate((np.full(padding.shape,self.ymod[0]),self.ymod,np.full(padding.shape,self.ymod[-1])))
            self.ymod = scipy.signal.convolve(ypad,instrument_function_cache['y'],mode='same')[len(padding):len(padding)+len(self.xmod)]
        self._mod.add_construct_function(f)

    def apodise(
            self,
            apodisation_function=None,
            interpolation_factor=None,
            **kwargs_specific_to_apodisation_function,
    ):
        """Apodise the spectrum with a known function. This is done in the
        length-domain so both Fourier and inverse-Fourier transforms are
        required."""
        ## get apodisation_function and interpolation_factor 
        if apodisation_function is not None:
            pass
        elif 'apodisation_function' in self.experimental_parameters:
            apodisation_function = self.experimental_parameters['apodisation_function']
        else:
            raise Exception('apodisation_function not provided and not found in experimental_parameters.')
        # if 'interpolation_factor' not in self.experimental_parameters:
            # warnings.warn("interpolation_factor not found in experimental_parameters, assuming it is 1")
            # interpolation_factor = 1.
        cache = {'interpolation_factor':interpolation_factor,}
        def f():
            if cache['interpolation_factor'] is None:
                interpolation_factor = (self.model_interpolate_factor if self.model_interpolate_factor else 1)
                if 'interpolation_factor' in self.experimental_parameters:
                    interpolation_factor *= self.experimental_parameters['interpolation_factor']
            else:
                interpolation_factor = cache['interpolation_factor']
            ft = scipy.fft.dct(self.ymod) # get Fourier transform
            n = np.linspace(0,interpolation_factor,len(ft),dtype=float)
            if apodisation_function == 'boxcar':
                w = np.ones(lpad.shape); w[abs(lpad)>L/2]  = 0 # boxcar
            elif apodisation_function == 'triangular':
                w = 1-abs(n) # triangular
            elif apodisation_function == 'cos':
                w = np.cos(pi/2*n)       # cos apodisation
            elif apodisation_function == 'Hamming':
                α = 0.54 ;  w = α + (1-α)*np.cos(pi*n) ## Hamming apodisation
            elif apodisation_function == 'Blackman':
                w = 0.42 + 0.50*np.cos(pi*n) + 0.08*np.cos(pi*n*2) # Blackman apodisation
            elif apodisation_function == 'Blackman-Harris 3-term':               # harris1978
                ## Convolve with the coefficients equivalent to a
                ## Blackman-Harris window. Coefficients taken from
                ## harris1978 p. 65.  There are multiple coefficents
                ## given for 3- and 5-Term windows. I use the left.
                ## Includes a boxcar function.
                a0,a1,a2,a3 = 0.42323,0.49755,0.07922,0
                w = (a0
                     + a1*np.cos(constants.pi*n) 
                     + a2*np.cos(constants.pi*n*2)
                     + a3*np.cos(constants.pi*n*3) )
            elif apodisation_function == 'Blackman-Harris 4-term':               # harris1978
                a0,a1,a2,a3 = 0.35875,0.48829,0.14128,0.01168
                w = (a0
                     + a1*np.cos(constants.pi*n) 
                     + a2*np.cos(constants.pi*n*2)
                     + a3*np.cos(constants.pi*n*3) )
            else:
                raise Exception(f'Unknown apodisation_function: {repr(apodisation_function)}')
            w[n>1] = 0          # zero-padded region contributes nothing
            self.ymod = scipy.fft.idct(ft*w) # new apodised spetrum
        self._mod.add_construct_function(f)
        # self.construct_functions.append(f)
        self.format_input_functions.append(lambda: f'{self.name}.apodise({repr(apodisation_function)},{interpolation_factor},{tools.dict_to_kwargs(kwargs_specific_to_apodisation_function)})')

    def convolve_with_blackman_harris(self,terms=3):
        """Convolve with the coefficients equivalent to a Blackman-Harris
        window. Coefficients taken from harris1978 p. 65.  There are
        multiple coefficents given for 3- and 5-Term windows. I use
        the left.  This is faster than apodisation in the
        length-domain."""
        raise ImplementationError('Does not quite work in current form when compared with frequency-domain apodisation.')
        # self.format_input_functions.append(lambda: f'{self.name}.convolve_with_signum(amplitude={repr(amplitude)},xwindow={xwindow})')
        warned_already = False
        def f():
            x,y = self.xmod,self.ymod
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1) # ASSUMES EVEN SPACED GRID
            padding = np.arange(dx,2*(terms-1)+1+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate sinc to convolve with
            if ('interpolation_factor' not in self.experimental_parameters
                and not warned_already):
                warnings.warn("interpolation_factor not found in experimental_parameters, assuming it is 1")
                warned_already = True 
            interpolation_factor = (self.model_interpolate_factor if self.model_interpolate_factor else 1)
            if 'interpolation_factor' in self.experimental_parameters:
                interpolation_factor *= self.experimental_parameters['interpolation_factor']
            assert interpolation_factor%1 == 0,'Blackman-Harris convolution only valid for integer interpolation_factor (current: {interpolation_factor{)'
            interpolation_factor = int(interpolation_factor)
            if terms == 3:
                yconv = np.zeros(4*interpolation_factor+1,dtype=float)
                yconv[2*interpolation_factor] = 0.42323 # harris1978
                yconv[interpolation_factor] = yconv[-interpolation_factor-1] = 0.49755 # harris1978
                yconv[0] = yconv[-1] = 0.07922 # harris1978
            yconv /= yconv.sum()                    # normalised
            self.ymod = scipy.signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self._mod.add_construct_function(f)

    def convolve_with_signum(
            self,
            amplitude,          # amplitude of signume
            xwindow=10,         # half-window for convolutiom
            xbeg=None,
            xend=None,
    ):
        """Convolve with signum function generating asymmetry. δ(x-x0) + amplitude/(x-x0)."""
        amplitude = self.add_parameter('amplitude',*tools.ensure_iterable(amplitude),note='convolve_with_signum')
        def f():
            retval =  f'{self.name}.convolve_with_signum(amplitude={repr(amplitude)},xwindow={xwindow}'
            if xbeg is not None:
                retval += f',xbeg={xbeg}'
            if xend is not None:
                retval += f',xend={xend}'
            return(retval+')')
        self.format_input_functions.append(f)
        def f():
            i = ((self.xmod>=(xbeg if xbeg is not None else -np.inf))
                 &(self.xmod<=(xend if xend is not None else np.inf)))
            x,y = self.xmod[i],self.ymod[i]
            if len(x)==0:
                return
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1) # ASSUMES EVEN SPACED GRID
            padding = np.arange(dx,xwindow+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate sinc to convolve with
            xconv = np.arange(0,xwindow,dx)
            xconv = np.concatenate((-xconv[::-1],xconv[1:]))
            yconv = amplitude.p/xconv/len(xconv)               # signum function
            yconv[int((len(yconv)-1)/2)] = 1.       # add δ function at center
            yconv /= yconv.sum()                    # normalised
            self.ymod[i] = scipy.signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self._mod.add_construct_function(f)

    def convolve_with_SOLEIL_instrument_function(
            self,
            sinc_fwhm=None,
            gaussian_fwhm=None,
            lorentzian_fwhm=(0,None,1e-3),
            signum_magnitude=(0,None,1e-3),
            sinc_fwhms_to_include=200,
            gaussian_fwhms_to_include=10,
            lorentzian_fwhms_to_include=10,
    ):
        """Convolve with SOLEIL instrument function."""
        ## get automatically set values if not given explicitly
        if sinc_fwhm is None:
            sinc_fwhm = (self.experimental_parameters['sinc_FWHM'],None,1e-3)
        if gaussian_fwhm is None:
            gaussian_fwhm = (0.1,None,1e-3)
        p = self.add_parameter_set(
            'convolve_with_instrument_function',
            sinc_fwhm=sinc_fwhm,
            gaussian_fwhm=gaussian_fwhm,
            lorentzian_fwhm=lorentzian_fwhm,
            signum_magnitude=signum_magnitude,
            step_default={'sinc_fwhm':1e-3, 'gaussian_fwhm':1e-3,
                          'lorentzian_fwhm':1e-3, 'signum_magnitude':1e-4,},)
        if abs(self.experimental_parameters['sinc_FWHM']-p['sinc_fwhm'])>(1e-5*p['sinc_fwhm']):
            warnings.warn('sinc FWHM does not match SOLEIL data file header')
        ## rewrite input line
        def f():
            retval = f'{self.name}.convolve_with_SOLEIL_instrument_function({p.format_input()}'
            if p['sinc_fwhm']      !=0: retval += f',sinc_fwhms_to_include={sinc_fwhms_to_include}'
            if p['gaussian_fwhm']  !=0: retval += f',gaussian_fwhms_to_include={gaussian_fwhms_to_include}'
            if p['lorentzian_fwhm']!=0: retval += f',lorentzian_fwhms_to_include={lorentzian_fwhms_to_include}'
            retval += ')'
            return(retval)
        self.format_input_functions.append(f)
        ## generate instrument function and broaden
        cache = dict(y=None,width=None,) # for persistence between optimisation function calls
        def f():
            dx = (self.xexp[-1]-self.xexp[0])/(len(self.xexp)-1) # ASSUMES EVEN SPACED GRID
            assert (abs((self.xexp[-1]-self.xexp[-2])-(self.xexp[1]-self.xexp[0]))
                    <((self.xexp[1]-self.xexp[0])/1e5)),'Experimental x-domain must be regular.' # poor test of grid regularity
            ## get cached instrument function or recompute
            # if cache['y'] is None or p.has_changed():
            if cache['y'] is None or p.timestamp>self.timestamp:
                ## get total extent of instrument function
                width = (abs(p['sinc_fwhm'])*sinc_fwhms_to_include
                         + abs(p['gaussian_fwhm'])*gaussian_fwhms_to_include
                         + abs(p['lorentzian_fwhm'])*lorentzian_fwhms_to_include)
                cache['width'] = width
                ## if necessary compute instrument function on a reduced
                ## subsampling to ensure accurate convolutions -- this wont
                ## help with an accurate convolution against the actual
                ## data!
                required_points_per_fwhm = 10
                subsample_factor = int(np.ceil(max(
                    1,
                    (required_points_per_fwhm*dx/p['sinc_fwhm'] if p['sinc_fwhm']!=0. else 1),
                    (required_points_per_fwhm*dx/p['gaussian_fwhm'] if p['gaussian_fwhm']!=0. else 1),
                    )))
                ## create the instrument function on a regular grid -- odd length with 0 in the middle
                x = np.arange(0,width+dx/subsample_factor*0.5,dx/subsample_factor,dtype=float)
                x = np.concatenate((-x[-1:0:-1],x))
                imidpoint = int((len(x)-1)/2)
                ## initial function is a delta function
                y = np.full(x.shape,0.)
                y[imidpoint] = 1.
                ## convolve with sinc function
                if p['sinc_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.sinc(x,Γ=abs(p['sinc_fwhm'])),'same')
                ## convolve with gaussian function
                if p['gaussian_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.gaussian(x,Γ=abs(p['gaussian_fwhm'])),'same')
                ## convolve with lorentzian function
                if p['lorentzian_fwhm']!=0:
                    y = signal.convolve(y,lineshapes.lorentzian(x,Γ=abs(p['lorentzian_fwhm'])),'same')
                ## if necessary account for phase correction by convolving with a signum
                if p['signum_magnitude']!=0:
                    ty = 1/x*p['signum_magnitude'] # hyperbolically decaying signum on either side
                    ty[imidpoint] = 1 # the central part  -- a delta function
                    y = signal.convolve(y,ty,'same')
                ## convert back to data grid if it has been subsampled
                if subsample_factor!=1:
                    a = y[imidpoint-subsample_factor:0:-subsample_factor][::-1]
                    b = y[imidpoint::subsample_factor]
                    b = b[:len(a)+1] 
                    y = np.concatenate((a,b))
                ## normalise 
                y = y/y.sum()
                cache['y'] = y
            ## convolve model with instrument function
            padding = np.arange(dx,cache['width']+dx, dx)
            xpad = np.concatenate((self.xmod[0]-padding[-1::-1],self.xmod,self.xmod[-1]+padding))
            ypad = np.concatenate((np.full(padding.shape,self.ymod[0]),self.ymod,np.full(padding.shape,self.ymod[-1])))
            self.ymod = scipy.signal.convolve(ypad,cache['y'],mode='same')[len(padding):len(padding)+len(self.xmod)]
        self._mod.add_construct_function(f)

    def add_SOLEIL_double_shifted_delta_function(self,magnitude,shift=(1000,None)):
        """Adds two copies of SOLEIL spectrum onto the model (after
        convolution with instrument function perhaps). One copy shifted up by
        shift(cm-1) and one shifted down. Shifted up copy is scaled by
        magnitude, copy shifted down is scaled by -magnitude. This is to deal
        with periodic errors aliasing the spectrum due to periodic
        vibrations."""
        filename = self.experimental_parameters['filename']
        x,y,header = load_SOLEIL_spectrum_from_file(filename)
        yshifted ={'left':None,'right':None}
        p = self.add_parameter_set('add_SOLEIL_double_shifted_delta_function',
                                   magnitude=magnitude,shift=shift,
                                   step_default={'magnitude':1e-3,'shift':1e-3},)
        previous_shift = [p['shift']]
        self.format_input_functions.append(lambda: f'{self.name}.add_SOLEIL_double_shifted_delta_function({p.format_input()})')
        def f():
            ## +shift from left -- use cached value of splined shifted
            ## spectrum if shift has not changed
            ## positive for this shift
            if yshifted['left'] is None or p['shift']!=previous_shift[0]:
                i = tools.inrange(self.xmod,x-p['shift'])
                j = tools.inrange(x+p['shift'],self.xmod)
                dy = tools.spline(x[j]+p['shift'],y[j],self.xmod[i])
                yshifted['left'] = (i,dy)
            else:
                i,dy = yshifted['left']
            self.ymod[i] += dy*p['magnitude']
            ## -shift from right -- use cached value of splined
            ## shifted spectrum if shift has not changed, magnitude is
            ## negative for this shift
            if yshifted['right'] is None or p['shift']!=previous_shift[0]:
                i = tools.inrange(self.xmod,x+p['shift'])
                j = tools.inrange(x-p['shift'],self.xmod)
                dy = tools.spline(x[j]-p['shift'],y[j],self.xmod[i])
                yshifted['right'] = (i,dy)
            else:
                i,dy = yshifted['right']
            self.ymod[i] += dy*p['magnitude']*-1
        self._mod.add_construct_function(f)
        
    # def set_polynomial_fvalue(self,name_regexp='',p0=1,p1=0,p2=0,p3=0,xkey='Jp',**matching_qn):
        # """Fit f to a polynomial in terms of xkey(xkey+1)."""
        # p = self.add_parameter_set('set_polynomial_fvalue',p0=p0,p1=p1,p2=p2,p3=p3)
        # self.format_input_functions.append(lambda: f'{self.name}.set_polynomial_fvalue({repr(name_regexp)},{repr(p)},xkey={repr(xkey)})')
        # i = self.ftransition.match(name_regexp=name_regexp,**matching_qn) # matching lines
        # def f():
            # self.ftransition['f'][i] = np.polyval([p['p3'],p['p2'],p['p1'],p['p0']],self.ftransition[xkey][i]*(self.ftransition[xkey][i]+1))
        # self._mod.add_construct_function(f)
        
    def plot(
            self,
            figure_number=None,
            ax=None,
            plot_optical_depth=False,
            plot_experiment= True,
            plot_model= True,
            plot_residual= True,
            plot_labels=False,
            plot_branch_heads=False,
            qn_defining_branch=('speciesp','speciespp','labelp','labelpp','vp','vpp','Fp'),
            label_key=None,
            label_match_name_re=None,
            label_match_qn=None,
            minimum_τ_to_label=None, # for absorption lines
            minimum_I_to_label=None, # for emission lines
            # annotate_reference_lines=['32S16O',],
            plot_title= True,
            title='auto',
            plot_legend=False,
            plot_contaminants=True, # whether or not to label locations of reference contaminants
            # contaminants_to_plot=('default',), # what contaminant to label
            contaminants_to_plot=None, # what contaminant to label
            linewidth=1,
            colors=plotting.linecolors_screen,
            shift_residual=0.,
            xlabel=None,ylabel=None,
            invert_model=False,
            **limit_to_qn,
    ):
        """Plot experimental and model spectra."""
        self.add_format_input_function(lambda: f'{self.name}.plot_spectrum(figure_number={repr(figure_number)},label_key={repr(label_key)},plot_labels={repr(plot_labels)},plot_optical_depth={repr(plot_optical_depth)},plot_experiment={repr(plot_experiment)},plot_model={repr(plot_model)},plot_residual={repr(plot_residual)})')
        ## make a figure
        plotting.presetRcParams('a4landscape')
        # tools.presetRcParams('screen')
        from cycler import cycler
        plt.rcParams.update({
            'figure.autolayout':False,
            'axes.prop_cycle'    : cycler('color',colors),
            'lines.linewidth': linewidth, 'lines.markersize': 10.0, 'lines.markeredgewidth': 0, 'patch.edgecolor': 'none',
            'grid.alpha' : 1.0, 'grid.color' : 'gray', 'grid.linestyle' : ':',
            'axes.titlesize'       :14., 'axes.labelsize'       :14.,
            'xtick.labelsize'      :12., 'ytick.labelsize'      :12.,
            'legend.fontsize'      :12., 'legend.handlelength'  :1, 'legend.handletextpad' :0.4, 'legend.labelspacing'  :0., 'legend.numpoints'     :1,
            'font.size'            :14., 'font.family'          :'serif', 'text.usetex'          :False, 'mathtext.fontset'     :'cm',
            'xtick.minor.top': True, 'xtick.minor.bottom': True, 'xtick.minor.visible': True , 'xtick.top': True , 'xtick.bottom': True ,
            'ytick.minor.right': True, 'ytick.minor.left': True, 'ytick.minor.visible': True , 'ytick.right': True , 'ytick.left': True ,
            'toolbar': 'toolbar2',
            'path.simplify'      :  True, # whether or not to speed up plots by joining line segments
            'path.simplify_threshold' :1, # how much to do so
            'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
        })
        def format_coord(x,y):
            if x<1e-5 or x>=1e10: xstr = f'{x:0.18e}'
            else:               xstr = f'{x:0.18f}'
            if y<1e-5 or y>1e5: ystr = f'{y:0.18e}'
            else:               ystr = f'{y:0.18f}'
            return(f'x={xstr:<25s} y={ystr:<25s}')
        ## get axes if not specified
        if figure_number is None: figure_number = plt.gcf().number
        if ax is None:
            fig = plt.figure(figure_number)
            fig.clf()
            ax = fig.gca()
            ax.format_coord = format_coord
        else:
            fig = ax.figure
        ymin,ymax = np.inf,-np.inf
        xmin,xmax = np.inf,-np.inf
        ## plot intensity and residual
        if plot_experiment and self.yexp is not None:
            # ymin,ymax = min(ymin,self.yexp.min()),max(ymax,self.yexp.max())
            ymin,ymax = -0.1*self.yexp.max(),self.yexp.max()*1.1
            xmin,xmax = min(xmin,self.xexp.min()),max(xmax,self.xexp.max())
            ax.plot(self.xexp,self.yexp,color=plotting.newcolor(0), label='Experimental spectrum') # plot experimental spectrum
        if plot_model and self.ymod is not None:
            if invert_model:
                self.ymod *= -1
            ymin,ymax = min(ymin,self.ymod.min(),-0.1*self.ymod.max()),max(ymax,self.ymod.max()*1.1)
            xmin,xmax = min(xmin,self.xmod.min()),max(xmax,self.xmod.max())
            ax.plot(self.xmod,self.ymod,color=plotting.newcolor(1), label='Model spectrum') # plot model spectrum
            if invert_model:
                self.ymod *= -1
        if plot_residual and self.model_residual is not None:
            ymin,ymax = min(ymin,self.model_residual.min()+shift_residual),max(ymax,self.model_residual.max()+shift_residual)
            xmin,xmax = min(xmin,self.xexp.min()),max(xmax,self.xexp.max())
            ax.plot(self.xexp,self.model_residual+shift_residual,color=plotting.newcolor(2),zorder=-1,label='Exp-Mod residual error',) # plot fit residual
        ## plot optical depth of model and individual lines (approx)
        # if plot_optical_depth:
            # yscale = (ymax-ymin)/self.optical_depths['total'].max()
            # ax.plot(self.xmod,ymin-self.optical_depths['total']*yscale,color=plotting.newcolor(3),label='τ',zorder=5) # plot optical depth of model
            # ymin -= ymax
        ## annotate rotational series
        if plot_labels:
            ystep = ymax/20.
            for transition in self.transitions:
                ## limit to qn
                if not transition.is_known(*limit_to_qn):
                    continue
                i = transition.match(**limit_to_qn)
                ## limit to ν-range and sufficiently strong lines
                i &= (transition['ν']>self.xexp[0])&(transition['ν']<self.xexp[-1])
                if minimum_τ_to_label is not None:
                    i &= transition['τ']>minimum_τ_to_label
                if minimum_I_to_label is not None:
                    i &= transition['I']>minimum_I_to_label
                t = transition[i]
                if len(t)==0: continue
                if isinstance(t,Rotational_Transition):
                    zkeys = ('speciesp','labelp','labelpp','vp','vpp','branch',)
                    if label_key==None:
                        label_key = 'Jp'
                elif isinstance(t,Atomic_Transition):
                    zkeys = ('speciesp',)
                    if label_key==None:
                        label_key = 'speciesp'
                else:
                    warnings.warn(f"Label keys not implemented for transition type {repr(type(transition))}")
                branch_annotations = annotate_spectrum_by_branch(
                    t, # only label lines which are strong enough
                    ymax+ystep/2.,
                    ystep,
                    zkeys=zkeys,
                    length=-0.02, # fraction of axes coords
                    color_by=('branch' if 'branch' in zkeys else zkeys),
                    labelsize='xx-small',namesize='x-small', namepos='float',    
                    label_key=label_key,
                    match_name_re=label_match_name_re,
                    match_qn=label_match_qn,
                )
                ymax += ystep*(len(branch_annotations)+1)
        ## plot branch heads
        if plot_branch_heads:
            for transition in self.transitions:
                annotate_branch_heads(transition,qn_defining_branch,match_branch_re=label_match_name_re)
        ## plot contaminant indicators
        if plot_contaminants and contaminants_to_plot is not None:
            contaminant_linelist = database.get_spectral_contaminant_linelist(
                *contaminants_to_plot,
                νbeg=ax.get_xlim()[0],
                νend=ax.get_xlim()[1],)
            for line in contaminant_linelist:
                x,y = line['ν'],ax.get_ylim()[0]/2.
                ax.plot(x,y,ls='',marker='o',color='red',markersize=6)
                ax.annotate(line['name'],(x,1.1*y),ha='center',va='top',color='gray',fontsize='x-small',rotation=90,zorder=-5)
        ## finalise plot
        if plot_title and 'filename' in self.experimental_parameters:
            if title == 'auto': title = tools.basename(self.experimental_parameters['filename'])
            t = ax.set_title(title,fontsize='x-small')
            t.set_in_layout(False)
        if plot_legend:
            # tools.legend_colored_text(loc='lower right')
            tools.legend_colored_text(loc='upper left')
            # tools.legend_colored_text(loc='best')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.grid(True,color='gray')
        plotting.simple_tick_labels(ax=ax)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        self._figure = fig
        return(fig)


    plot_spectrum = plot        # DEPRECATED


    def plot_band_parameters(self,figure_number=None,ykeys=('f','Treducedp','Treduced_commonp','Γ'),xkey='Jp',names_re='.*',ignore_uncertainties=False,**plot_kwargs): 
        """Plot band parameters."""
        self.format_input_functions.append(lambda: f'{self.name}.plot_band_parameters(figure_number={repr(figure_number)},ykeys={repr(ykeys)},xkey={repr(xkey)},names_re={repr(names_re)},ignore_uncertainties={repr(ignore_uncertainties)},{tools.dict_to_kwargs(plot_kwargs)})')
        plt.rcParams.update({
            'figure.autolayout':True ,
            'lines.linewidth': 0.5, 'lines.markersize': 10.0, 'lines.markeredgewidth': 0, 'patch.edgecolor': 'none',
            'grid.alpha' : 1.0, 'grid.color' : 'gray', 'grid.linestyle' : ':',
            'axes.titlesize'       :14., 'axes.labelsize'       :14.,
            'xtick.labelsize'      :14., 'ytick.labelsize'      :14.,
            'legend.fontsize'      :12., 'legend.handlelength'  :1.5, 'legend.handletextpad' :0.4, 'legend.labelspacing'  :0., 'legend.numpoints'     :1,
            'font.size'            :14., 'font.family'          :'serif', 'text.usetex'          :False, 'mathtext.fontset'     :'cm',
            'xtick.minor.top': True, 'xtick.minor.bottom': True, 'xtick.minor.visible': True , 'xtick.top': True , 'xtick.bottom': True ,
            'ytick.minor.right': True, 'ytick.minor.left': True, 'ytick.minor.visible': True , 'ytick.right': True , 'ytick.left': True ,
            'toolbar': 'toolbar2',
            # 'toolbar': 'none',
        })
        if figure_number is None: figure_number = plt.gcf().number
        tools.presetRcParams('a4landscape')
        fig = plt.figure(figure_number)
        if names_re is None:    transition = self.ftransition
        else:                   transition = self.ftransition[tools.find_regexp(names_re,self.ftransition['name'])]
        if not transition.is_known(xkey): return # no data to plot
        ykeys = [t for t in ykeys if transition.is_known(t)]
        if ignore_uncertainties: transition.unset(*['d'+key for key in ykeys])
        transition.plot(xkey=xkey,ykeys=ykeys,**plot_kwargs)

    def output_data_to_directory(
            self,
            directory='td',
            output_model_residual=False,
            output_transition_linelists=False,
            output_individual_optical_depths=False,
    ):
        """Save various files from this optimsiation to a directory."""
        tools.mkdir_if_necessary(directory)
        ## model data
        if self.xexp is not None and self.yexp is not None:
            tools.array_to_file(directory+'/experimental_spectrum.h5',self.xexp,self.yexp)
        if self.xmod is not None and self.ymod is not None:
            tools.array_to_file(directory+'/model_spectrum.h5',self.xmod,self.ymod)
        if output_model_residual and self.xexp is not None and self.model_residual is not None:
            tools.array_to_file(directory+'/model_residual.h5', self.xexp,self.model_residual)
        if self._figure is not None:
            self._figure.savefig(directory+'/figure.png',dpi=300) # save figure
        ## save transition linelists
        if output_transition_linelists:
            tools.mkdir_if_necessary(directory+'/transitions')
            for transition in self.absorption_transitions:
                transition.save_to_file(directory+'/transitions/'+transition.name+'.h5')
        ## save optical deptth spectrum
        # if output_individual_optical_depths:
            # tools.mkdir_if_necessary(directory+'/optical_depths')
            # for key,τ in self.optical_depths.items():
                # if τ is not None and self.xmod is not None:
                    # name = (tools.rootname(key) if isinstance(key,str) else key.name) # key is either a string or Transition object
                    # tools.array_to_file(f'{directory}/optical_depths/{name}.h5', self.xmod,τ)

    def load_from_directory(self,directory):
        """Load internal data from a previous "output_to_directory" model."""
        self.experimental_parameters['filename'] = directory
        directory = tools.expand_path(directory)
        assert os.path.exists(directory) and os.path.isdir(directory),f'Directory does not exist or is not a directory: {repr(directory)}'
        for filename in (
                directory+'/experimental_spectrum', # text file
                directory+'/experimental_spectrum.gz', # in case compressed
                directory+'/experimental_spectrum.h5', # in case compressed
                directory+'/exp',                      # deprecated
        ):
            if os.path.exists(filename):
                self.xexp,self.yexp = tools.file_to_array_unpack(filename)
                break
        for filename in (
                directory+'/model_spectrum',
                directory+'/model_spectrum.gz', 
                directory+'/model_spectrum.h5', 
                directory+'/mod',
        ):
            if os.path.exists(filename):
                self.xmod,self.ymod = tools.file_to_array_unpack(filename)
                break
        for filename in (
                directory+'/model_residual',
                directory+'/model_residual.gz',
                directory+'/model_residual.h5',
                directory+'/residual',
        ):
            if os.path.exists(filename):
                # t,self.model_residual = tools.file_to_array_unpack(filename)
                t = tools.file_to_array(filename)
                if t.ndim==1:
                    self.model_residual = t
                elif t.ndim==2:
                    self.model_residual = t[:,1]
                break
        # for filename in (
                # directory+'/optical_depth',
                # directory+'/optical_depth.gz',
                # directory+'/optical_depth.h5',
        # ):
            # if os.path.exists(filename):
                # t,self.optical_depths['total'] = t
                # break
        # for filename in tools.myglob(directory+'/optical_depths/*'):
            # self.optical_depths[tools.basename(filename)] = tools.file_to_array_unpack(filename)[1]
        # for filename in tools.myglob(directory+'/transitions/*'):
            # self.transitions.append(load_transition(
                # filename,
                # Name=tools.basename(filename),
                # decode_names=False,
                # permit_new_keys=True,
                # # error_on_unknown_key=False, # fault tolerant
            # ))

    # def get_transition(self,name):
        # for t in self.transitions:
            # if t.name==name:
                # return(t)
        # else:
            # raise Exception(f'Transition names {repr(name)} not found.')    

    def show(self):
        """Show plots."""
        self.format_input_functions.append(f'{self.name}.show()')
        plt.show()



# def load_SOLEIL_spectrum_from_file(filename,remove_HeNe=False):
    # """ Load SOLEIL spectrum from file with given path."""
    # ## resolve SOLEIL filename
    # if os.path.exists(tools.expand_path(filename)):
        # ## filename is an actual path to a file
        # filename = tools.expand_path(filename)
    # elif os.path.exists(f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.hdf5'):
        # ## filename is a scan base name in default data directory
        # filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.hdf5'
    # elif os.path.exists(f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.h5'):
        # ## filename is a scan base name in default data directory
        # filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.h5'
    # else:
        # ## else look for unique prefix in scan database
        # t = tools.sheet_to_dict('~/exp/SOLEIL/summary_of_scans.rs',comment='#')
        # i = tools.find_regexp(r'^'+re.escape(filename)+'.*',t['filename'])
        # if len(i)==1:
            # filename = t['filename'][int(i)]
            # filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.hdf5'
        # else:
            # raise Exception(f"Could not find SOLEIL spectrum: {repr(filename)}")
    # extension = os.path.splitext(filename)[1]
    # ## get header data if possible, not possible if an hdf5 file is used.
    # header = dict(filename=filename,header=[])
    # if extension in ('.TXT','.wavenumbers'): 
        # with open(filename,'r',encoding='latin-1') as fid:
            # header['header'] = []
            # while True:
                # line = fid.readline()[:-1]
                # if re.match(r'^ *[0-9.eE+-]+[, ]+[0-9.eE+-]+ *$',line): break # end of header
                # header['header'].append(line) # save all lines to 'header'
                # ## post-processing zero-adding leads to an
                # ## interpolation of data by this factor
                # r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
                # if r:
                    # header['interpolation_factor'] = float(r.group(1))
                # ## the resolution before any interpolation
                # r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
                # if r: header['ds'] = float(r.group(1))
                # ## NMAX parameter indicates that the spectrometer is
                # ## being run at maximum resolution. This is not an
                # ## even power of two. Then the spectrum is zero padded
                # ## to have 2**21 points. This means that there is an
                # ## additional interpolation factor of 2**21/NMAX. This
                # ## will likely be non-integer.
                # r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
                # if r:
                    # header['interpolation_factor'] *= 2**21/float(r.group(1))
                # ## extract pressure from header
                # r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
                # if r:
                    # header['date_time'] = r.group(1)
                    # header['pressure_pirani'] = float(r.group(2))
                    # header['pressure_baratron'] = float(r.group(3))
            # header['header'] = '\n'.join(header['header'])
            # ## compute instrumental resolution, FWHM
    # elif extension in ('.hdf5','.h5'): # expect header stored in 'README'
        # data = tools.hdf5_to_dict(filename)
        # header['header'] = data['README']
        # for line in header['header'].split('\n'):
            # ## post-processing zero-adding leads to an
            # ## interpolation of data by this factor
            # r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
            # if r:
                # header['interpolation_factor'] = float(r.group(1))
            # ## the resolution before any interpolation
            # r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
            # if r: header['ds'] = float(r.group(1))
            # ## NMAX parameter -- see above
            # r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
            # if r:
                # header['interpolation_factor'] *= 2**21/float(r.group(1))
            # ## extract pressure from header
            # r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
            # if r:
                # header['date_time'] = r.group(1)
                # header['pressure_pirani'] = float(r.group(2))
                # header['pressure_baratron'] = float(r.group(3))
    # else:
        # raise Exception(f"bad extension: {repr(extension)}")
    # ## compute instrumental resolution, FWHM
    # header['sinc_FWHM'] = 1.2*header['interpolation_factor']*header['ds'] 
    # ## get spectrum
    # if extension=='.TXT':
        # x,y = [],[]
        # data_started = False
        # for line in tools.file_to_lines(filename,encoding='latin-1'):
            # r = re.match(r'^([0-9]+),([0-9.eE+-]+)$',line) # data point line
            # if r:
                # data_started = True # header is passed
                # x.append(float(r.group(1))),y.append(float(r.group(2))) # data point
            # else:
                # if data_started: break            # end of data
                # else: continue                    # skip header line
        # x,y = np.array(x)*header['ds'],np.array(y)
    # elif extension=='.wavenumbers':
        # x,y = tools.file_to_array(filename,unpack=True,comments='#',encoding='latin-1')
    # elif extension in ('.hdf5','.h5'):
        # data = tools.hdf5_to_dict(filename)
        # x,y = data['data'].transpose()
    # else:
        # raise Exception(f"bad extension: {repr(extension)}")
    # ## process a bit. Sort and remove HeNe line profile and jitter
    # ## estimate. This is done assumign the spectrum comes
    # ## first. and finding the first index were the wavenumber
    # ## scale takes a backward step
    # if remove_HeNe:
        # i = x>31600
        # x,y = x[i],y[i]
    # t = tools.find(np.diff(x)<0)
    # if len(t)>0:
        # i = t[0]+1 
        # x,y = x[:i],y[:i]
    # ## get x range
    # header['xmin'],header['xmax'] = x.min(),x.max()
    # header['xcentre'] = 0.5*(header['xmin']+header['xmax'])
    # return (x,y,header)

    # ## deprecated
    # scale_by_source_spline = scale_by_spline



# def load_spectrum(filename,**kwargs):
    # """Use a heuristic method to load a directory output by
    # Spectrum."""
    # x = Spectrum()
    # x.load_from_directory(filename,**kwargs)
    # return(x)
