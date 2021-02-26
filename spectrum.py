import inspect,re
from copy import copy,deepcopy
from pprint import pprint
import warnings

from matplotlib import pyplot as plt
from scipy import signal,constants,fft,interpolate
import numpy as np


from . import optimise
from .optimise import *
from . import plotting
from . import tools
from . import hitran
from . import bruker
from . import lineshapes
from . import lines
from .dataset import Dataset

class Experiment(Optimiser):
    
    def __init__(
            self,
            name='experiment',
            filename=None,
            x=None,y=None,
            xbeg=None,xend=None,
    ):
        optimise.Optimiser.__init__(self,name)
        self.pop_format_input_function()
        self.automatic_format_input_function()
        # def f():
            # retval = f'{self.name} = spectrum.Experiment({repr(self.name)}'
            # if filename is not None:
                # retval += f',filename={repr(filename)}'
            # return retval+')'
        # self.add_format_input_function(f)
        self.x = None
        self.y = None
        self.experimental_parameters = {} # a dictionary containing any additional known experimental parameters
        if filename is not None:
            self.set_spectrum_from_file(filename,xbeg,xend)
        if x is not None and y is not None:
            self.set_spectrum(x,y,xbeg,xend)
        self.add_save_to_directory_function(
            lambda directory: tools.array_to_file(directory+'/spectrum.h5',self.x,self.y))

    @auto_construct_method('set_spectrum')
    def set_spectrum(self,x,y,xbeg=None,xend=None,**experimental_parameters):
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
        ##
        self.experimental_parameters.update(experimental_parameters)
        ## verbose info
        if self.verbose:
            print('experimental_parameters:')
            pprint(self.experimental_parameters)
        ## construct function
        def f():
            self.x,self.y = copy(x),copy(y) # make copy -- more memory but survive other changes
        return f

    def set_spectrum_from_file(self,filename,xbeg=None,xend=None,**file_to_array_kwargs):
        """Load a spectrum to fit from an x,y file."""
        self.add_format_input_function(lambda:self.name+f'.set_spectrum_from_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)},{tools.dict_to_kwargs(file_to_array_kwargs)})')
        x,y = tools.file_to_array_unpack(filename,**file_to_array_kwargs)
        self.experimental_parameters['filename'] = filename
        self.set_spectrum(x,y,xbeg,xend)

    def set_spectrum_from_dataset(self,filename,xbeg=None,xend=None,xkey='x',ykey='y'):
        """Load a spectrum to fit from an x,y file."""
        d = Dataset()
        d.load(filename)
        experimental_parameters = {key:val for key,val in d.items() if  key not in (xkey,ykey)}
        self.set_spectrum(d[xkey],d[ykey],xbeg,xend,**d.attributes)

    def set_spectrum_from_opus_file(self,filename,xbeg=None,xend=None):
        """Load a spectrum in an Bruker opus binary file."""
        opusdata = bruker.OpusData(filename)
        x,y = opusdata.get_spectrum()
        d = opusdata.data
        self.experimental_parameters['filename'] = filename
        if 'Fourier Transformation' in d:
            self.experimental_parameters['interpolation_factor'] = float(d['Fourier Transformation']['ZFF'])
            if d['Fourier Transformation']['APF'] == 'B3':
                self.experimental_parameters['apodisation_function'] = 'Blackman-Harris 3-term'
            else:
                warnings.warn(f"Unknown opus apodisation function: {repr(d['Fourier Transformation']['APF'])}")
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function() 
        self.add_format_input_function(lambda:self.name+f'.set_spectrum_from_opus_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)})')

    
    def set_spectrum_from_SOLEIL_file(
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
        self.add_format_input_function(f)
        ## Limit xbeg/xend to fall within limits of actual data. If
        ## there is no data then self.x and self.y are None
        if xbeg is None:
            xbeg = x.min()
        if xend is None:
            xend = x.max()
        if xbeg>x.max() or xend<x.min():
            self.x = self.y = None
            warnings.warn(f"SOLEIL spectrum has no data in x-range: {repr(filename)}")
            return
        xbeg,xend = max(xbeg,x.min()),min(xend,x.max())
        self.set_spectrum(x,y,xbeg=xbeg,xend=xend)
        if xscale is not None:
            i = (x>xbeg-10.)&(x<xend+10.)
            x,y = x[i],y[i]
            yscaled = [None]
            def f():
                if (xscale is not None
                    and (yscaled[0] is None
                         or p.timestamp>self.timestamp)):
                    yscaled[0] = tools.spline(x*p['xscale'],y,x)
                    self.set_spectrum(x,yscaled[0],xbeg=xbeg,xend=xend)
            self.add_construct_function(f)

    @auto_construct_method('scalex')
    def scalex(self,scale=1):
        """Rescale experimental spectrum x-grid."""
        def construct_function():
            self.x *= float(scale)
        return construct_function

    # def interpolate(self,dx):
        # """Interpolate experimental spectrum to a grid of width dx, may change
        # position of xend."""
        # self.add_format_input_function(lambda:self.name+f'.interpolate({repr(dx)})')
        # def f():
            # xnew = np.arange(self.x[0],self.x[-1],dx)
            # ynew = tools.spline(self.x,self.y,xnew)
            # if self.verbose:
                # print(f"Interpolating to grid ({repr(xnew[0])},{repr(xnew[-1])},{dx}) from grid ({repr(self.x[0])},{repr(self.x[-1])},{self.x[1]-self.x[0]})")
            # self.x,self.y = xnew,ynew
        # self.add_construct_function(f)

    def __len__(self):
        return len(self.x)

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
        self.add_format_input_function(
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

    def plot(self,ax=None):
        """Plot spectrum."""
        self.construct()
        ## reuse current axes if not specified
        if ax is None:
            ax = plt.gca()
            ax.cla()
            def format_coord(x,y):
                if x<1e-5 or x>=1e10:
                    xstr = f'{x:0.18e}'
                else:
                    xstr = f'{x:0.18f}'
                if y<1e-5 or y>1e5:
                    ystr = f'{y:0.18e}'
                else:
                    ystr = f'{y:0.18f}'
                return(f'x={xstr:<25s} y={ystr:<25s}')
            ax.format_coord = format_coord
            ax.grid(True,color='gray')
            plotting.simple_tick_labels(ax=ax)
        ax.plot(self.x,self.y)

class Model(Optimiser):

    def __init__(
            self,
            name=None,
            experiment=None,
            residual_weighting=None,
            verbose=None,
            xbeg=None,xend=None,
    ):
        self.experiment = experiment
        if name is None:
            if self.experiment is None:
                name = 'model'
            else:
                name = f'model_of_{experiment.name}'
        self.x = None
        self.y = None
        self._xin = None
        self.xbeg = xbeg
        self.xend = xend
        self.residual = None                      # array of residual fit
        self.residual_weighting = residual_weighting            # weighting pointwise in xexp
        self._interpolate_factor = None
        optimise.Optimiser.__init__(self,name)
        self.pop_format_input_function()
        # self.automatic_format_input_function()
        if self.experiment is not None:
            self.add_suboptimiser(self.experiment)
        self.add_construct_function(self._initialise)
        self.add_post_construct_function(self._get_residual,self._remove_interpolation)
        self.add_save_to_directory_function(self.output_data_to_directory)
        self._figure = None

    def _initialise(self):
        """Function run before everything else to set x and y model
        grid."""
        if self._xin is not None:
            ## x is from a call to get_spectrum
            self.x = self._xin
            self.xexp = self.yexp = None
        elif self.experiment is not None:
            ## get domain from experimental data
            self.xexp = self.experiment.x
            self.yexp = self.experiment.y
            if self.xbeg is not None:
                i = self.xexp>=self.xbeg
                self.xexp,self.yexp = self.xexp[i],self.yexp[i]
            if self.xend is not None:
                i = self.xexp<=self.xend
                self.xexp,self.yexp = self.xexp[i],self.yexp[i]
            self.x = copy(self.xexp)
        else:
            raise Exception('Cannot determine x')
        self.y = np.zeros(self.x.shape,dtype=float)
        
    def get_spectrum(self,x):
        """Construct a model spectrum at x."""
        self._xin = np.asarray(x,dtype=float) # needed in _initialise
        self.timestamp = -1     # force reconstruction, but not suboptimisers
        self.construct()
        self._xin = None        # might be needed in next _initialise
        return self.y

    def _get_residual(self):
        """Compute residual error."""
        if self.xexp is None:
            return []
        residual = self.yexp - self.y
        if self.residual_weighting is not None:
            residual *= self.residual_weighting
        return residual

    @optimise_method()
    def interpolate(self,dx):
        """When calculating model set to dx grid (or less to achieve
        overlap with experimental points. DELETES CURRENT Y!"""
        xstep = (self.x[-1]-self.x[0])/(len(self.x)-1)
        self._interpolate_factor = int(np.ceil(xstep/dx))
        self.x = np.linspace(self.x[0],self.x[-1],1+(len(self.x)-1)*self._interpolate_factor)
        self.y = np.zeros(self.x.shape,dtype=float) # delete current y!!

    def _remove_interpolation(self):
        """If the model has been interpolated then restore it to original
        grid."""
        if self._interpolate_factor is not None:
            self.x = self.x[::self._interpolate_factor]
            self.y = self.y[::self._interpolate_factor]

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
        self.add_format_input_function(f)
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
                        self.x))
                for key in p.keys():
                    cache[key]  = p[key]
            self.y *= cache['transmission'] # add to model
        self.add_construct_function(f)

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
        self.add_format_input_function(f)
        self.add_suboptimisers(cross_section_object)
        cache = {}
        def f():
            ## update if necessary
            if ('transmission' not in cache 
                or (p.timestamp>self.timestamp)
                or (cross_section_object.timestamp>self.timestamp)):
                # cache['transmission'] = np.full(self.x.shape,1.0)
                # i = np.full(self.x.shape,
                # if xbeg is not None:
                    # i &= self.x>=xbeg
                # if xend is not None:
                    # i &= self.x<=xend
                # cache['i'] = i
                cache['transmission'] = np.exp(-p['column_density']*cross_section_object.σ(self.x))
                for key in p.keys():
                    cache[key] = p[key]
            ## add to model
            self.y *= cache['transmission']
            # self.optical_depths[cross_section_object.name] = cache['τ']
        self.add_construct_function(f)
    add_cross_section = add_absorption_cross_section # deprecated name

    @optimise_method()
    def add_absorption_lines(
            self,
            lines=None,
            nfwhmL=20,
            nfwhmG=100,
            τmin=None,
            lineshape='voigt',
            ncpus=1,
            _cache=None
    ):
        if self.x is None:
            ## x not set yet
            return


        ## recompute spectrum if is necessary for some reason --
        ## do various tests to see if a cached version is ok
        if (
                'absorbance'  not in _cache # first run
                or self._last_construct_time < lines._last_construct_time # line spectrum has changed
                or self._last_construct_time < self.experiment._last_construct_time # experimental x-domain might have changed
        ):
            ## only include lines in the x-range
            if 'i' not in _cache:
                _cache['i'] = (lines['ν'] > (self.x[0]-1)) & (lines['ν'] < (self.x[-1]+1))
            tlines = lines[_cache['i']]
            ## if previous calculations are cached then find which lines have actually changed -- store in j
            if 'tlines' in _cache:
                j = np.any([ tlines[key] != _cache['tlines'][key] for key in tlines.keys()], axis=0)
            else:
                j = None
            ## compute entire spectrum
            if  'τ' not in _cache or j is None or np.sum(j) > (len(tlines)/2):
                x,y = tlines.calculate_spectrum(
                    x=self.x,xkey='ν',ykey='τ',nfwhmG=nfwhmG,nfwhmL=nfwhmL,
                    ymin=τmin, ncpus=ncpus, lineshape=lineshape,)
            ## compute difference of changed lines
            else:
                xnew,ynew = tlines[j].calculate_spectrum(
                    x=self.x,xkey='ν',ykey='τ',nfwhmG=nfwhmG,nfwhmL=nfwhmL,
                    ymin=τmin, ncpus=ncpus, lineshape=lineshape,)
                xold,yold = _cache['tlines'][j].calculate_spectrum(
                    x=self.x,xkey='ν',ykey='τ',nfwhmG=nfwhmG,nfwhmL=nfwhmL,
                    ymin=τmin, ncpus=ncpus, lineshape=lineshape,)
                y = _cache['τ'] - yold + ynew             
            ## store _cache
            # tlines.unset_inferences()
            _cache['tlines'] = tlines
            _cache['τ'] = y
            _cache['absorbance'] = np.exp(-y)
        ## set absorbance in self
        self.y *= _cache['absorbance']

    @optimise_method()
    def add_rautian_absorption_lines(self,lines,τmin=None,_cache=None,):
        ## x not set yet
        if self.x is None:
            return
        ## first run
        if len(_cache) == 0:
            ## recompute spectrum if is necessary for some reason --
            ## do various tests to see if a cached version is ok
            i = (lines['ν'] > (self.x[0]-1)) & (lines['ν'] < (self.x[-1]+1))
            _cache['i'] = i
        else:
            i = _cache['i']
            
        if (
                'absorbance'  not in _cache
                or self._last_construct_time < lines._last_construct_time # line spectrum has changed
                or self._last_construct_time < self.experiment._last_construct_time # experimental x-domain might have changed
        ):
            τ = lineshapes.rautian_spectrum(
                x=self.x,
                x0=lines['ν'][i],
                S=lines['τ'][i],
                ΓL=lines['Γ'][i],
                ΓG=lines['ΓD'][i],
                νvc=lines['νvc'][i],
                Smin=τmin,
            )
            _cache['absorbance'] = np.exp(-τ)
        self.y *= _cache['absorbance']

    def add_hitran_absorption_lines(self,species=None,**kwargs):
        """Shortcut"""
        return self.add_absorption_lines(lines=hitran.get_lines(species),**kwargs)
                
    @optimise_method()
    def add_noise(self,rms=1):
        """Add normally distributed noise with given rms."""
        self.y += rms*tools.randn(len(self.y))

    def add_emission_lines(
            self,
            lines,
            nfwhmL=None,
            nfwhmG=None,
            Imin=None,
            gaussian_method=None,
            voigt_method=None,
            use_multiprocessing=None,
            use_cache=None,
            **optimise_keys_vals):
        self.add_suboptimiser(lines) # to model rebuilt when transition changed
        # self.transitions.append(transition)   
        name = f'add_emission_lines {lines.name} to {self.name}'
        # assert name not in self.emission_intensities,f'Non-unique name in emission_intensities: {repr(name)}'
        # self.emission_intensities[name] = None
        p = self.add_parameter_set(**optimise_keys_vals,note=name)
        cache = {}
        def construct_function():
            ## first call -- no good, x not set yet
            if self.xexp is None:
                # self.emission_intensities[name] = None
                return
            ## recompute spectrum
            if (cache =={}
                # or self.emission_intensities[name] is None # currently no spectrum computed
                or self.timestamp<lines.timestamp # transition has changed
                or self.timestamp<p.timestamp     # optimise_keys_vals has changed
                or not (len(cache['experiment.x']) == len(self.xexp)) # experimental domain has changed
                or not np.all( cache['experiment.x'] == self.xexp )): # experimental domain has changed
                ## update optimise_keys_vals that have changed
                for key,val in p.items():
                    if (not lines.is_set(key)
                        or np.any(lines[key]!=val)):
                        lines[key] = val
                ## that actual computation
                x,y = lines.calculate_spectrum(
                    x=self.x,
                    ykey='I',
                    nfwhmG=(nfwhmG if nfwhmG is not None else 10),
                    nfwhmL=(nfwhmL if nfwhmL is not None else 100),
                    ymin=Imin,
                    ΓG='ΓD',
                    ΓL='Γ',
                    # gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran stepwise'),
                    gaussian_method=(gaussian_method if gaussian_method is not None else 'fortran'),
                    voigt_method=(voigt_method if voigt_method is not None else 'wofz'),
                    use_multiprocessing=(use_multiprocessing if use_multiprocessing is not None else False),
                    use_cache=(use_cache if use_cache is not None else True),
                )
                cache['experiment.x'] = copy(self.xexp)
                cache['intensity'] = y
            ## add emission intensity to the overall model
            self.y += cache['intensity']
        self.add_construct_function(construct_function)
        ## new input line
        def f():
            retval = f'{self.name}.add_emission_lines({lines.name}'
            if nfwhmL is not None: retval += f',nfwhmL={repr(nfwhmL)}'
            if nfwhmG is not None: retval += f',nfwhmG={repr(nfwhmG)}'
            if Imin is not None: retval += f',Imin={repr(Imin)}'
            if use_multiprocessing is not None: retval += f',use_multiprocessing={repr(use_multiprocessing)}'
            if use_cache is not None: retval += f',use_cache={repr(use_cache)}'
            if voigt_method is not None: retval += f',voigt_method={repr(voigt_method)}'
            if gaussian_method is not None: retval += f',gaussian_method={repr(gaussian_method)}'
            if len(p)>0: retval += f',{p.format_input()}'
            return(retval+')')
        self.add_format_input_function(f)
        

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
        self.add_format_input_function(f)
        def f():
            if xbeg is None and xend is None:
                self.residual_weighting = np.full(self.xexp.shape,weighting,dtype=float)
            else:
                if self.residual_weighting is None:
                    self.residual_weighting = np.ones(self.xexp.shape,dtype=float)
                self.residual_weighting[
                    (self.xexp>=(xbeg if xbeg is not None else -np.inf))
                    &(self.xexp<=(xend if xend is not None else np.inf))
                ] = weighting
        self.add_construct_function(f)

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
        self.add_format_input_function('# '+self.name+f'.autodetect_lines(filename={repr(filename)},τ={τ},Γ={Γ},vν={repr(vν)},vτ={repr(vτ)},vΓ={repr(vΓ)},τscale={repr(τscale)},xmin={repr(xmin)},ymin={repr(ymin)},'+tools.dict_to_kwargs(qn)+')')
        ## get something to find lines in
        if filename is None:
            x = copy(self.xexp)
            if self.residual is not None: y = copy(self.residual) # get from residual
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
        
    def scale_by_constant(self,scale=1.0):
        """Scale model by a constant value."""
        scale = self.add_parameter('scale',scale)
        self.add_format_input_function(lambda: f"{self.name}.scale_by_constant(ν={repr(scale)})")
        def f():
            self.y *= scale.p
        self.add_construct_function(f)

    def scale_by_spline(self,ν=50,amplitudes=1,vary=True,step=0.0001,order=3):
        """Scale by a spline defined function."""
        if np.isscalar(ν):
            ν = np.arange(self.xexp[0]-ν,self.xexp[-1]+ν*1.01,ν) # default to a list of ν with spacing given by ν
        if np.isscalar(amplitudes):
            amplitudes = amplitudes*np.ones(len(ν)) # default amplitudes to list of hge same length
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
        self.add_format_input_function(format_input_function)
        def f():
            i = (self.x>=np.min(ν))&(self.x<=np.max(ν))
            self.y[i] = self.y[i]*interpolate.UnivariateSpline(ν,p.plist,k=min(order,len(ν)-1),s=0)(self.x[i])
        self.add_construct_function(f) # multiply spline during construct

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
        self.add_format_input_function(format_input_function)
        def f():
            self.y *= 1. + tools.spline(ν,amplitude.plist,self.x)*np.sin(tools.spline(ν,phase.plist,self.x))
        self.add_construct_function(f)

    @auto_construct_method('add_intensity')
    def add_intensity(self,intensity=1):
        """Shift by a spline defined function."""
        def f():
            self.y += float(intensity)
        return f 

    def auto_add_intensity_spline(self,xstep=1000.,y=1.):
        """Quickly add an evenly-spaced intensity spline."""
        self.experiment.construct()
        knots = [(x,P(y,True)) for x in
                 np.arange(self.experiment.x[0]-xstep,self.experiment.x[-1]+xstep*1.01,xstep)]
        self.add_intensity_spline(knots=knots)

    @optimise_method()
    def add_intensity_spline(self,knots=None,order=3,_cache=None):
        """Add intensity points defined by a spline."""
        x = np.array([float(xi) for xi,yi in knots])
        y = np.array([float(yi) for xi,yi in knots])
        i = (self.x>=np.min(x))&(self.x<=np.max(x))
        ## calculate spline -- get cached version if inputs have not
        ## changed
        if ('s' not in _cache
            or np.any(_cache['x'] != x)
            or np.any(_cache['y'] != y)
            or np.any(_cache['i'] != i)
            ):
            s = tools.spline(x,y,self.x[i],order=order)
        else:
            s = _cache['s']
        _cache['s'],_cache['x'],_cache['y'],_cache['i'] = s,x,y,i
        self.y[i] += s

    # def scale_by_source_from_file(self,filename,scale_factor=1.):
        # p = self.add_parameter_set('scale_by_source_from_file',scale_factor=scale_factor,step_scale_default=1e-4)
        # self.add_format_input_function(lambda: f"{self.name}.scale_by_source_from_file({repr(filename)},{p.format_input()})")
        # x,y = tools.file_to_array_unpack(filename)
        # scale = tools.spline(x,y,self.xexp)
        # def f(): self.y *= scale*p['scale_factor']
        # self.add_construct_function(f)

    @optimise_method()
    def shift_by_constant(self,shift):
        """Shift by a constant amount."""
        self.y += float(shift)

    @auto_construct_method('convolve_with_gaussian')
    def convolve_with_gaussian(self,width=1,fwhms_to_include=100):
        """Convolve with gaussian."""
        def f():
            x,y = self.x,self.y
            if width == 0:
                ## nothing to be done
                return
            abswidth = abs(width)
            max_width = 1
            if abswidth > 1e2:
                raise Exception(f'Gaussian width > 100')
            if self.verbose and abswidth<3*np.diff(self.xexp).min(): 
                warnings.warn('Convolving gaussian with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
            ## get padded spectrum to minimise edge effects
            dx = (x[-1]-x[0])/(len(x)-1)
            padding = np.arange(dx,fwhms_to_include*abswidth+dx,dx)
            xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            ## generate gaussian to convolve with
            xconv_beg = -fwhms_to_include*abswidth
            xconv_end =  fwhms_to_include*abswidth
            # if ((xconv_end-xconv_beg)/dx) > 1e6:
                # warnings.warn('Convolution domain length very long.')
            xconv = np.arange(xconv_beg,xconv_end,dx)
            if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
            yconv = np.exp(-(xconv-xconv.mean())**2*4*np.log(2)/abswidth**2) # peak normalised gaussian
            yconv = yconv/yconv.sum() # sum normalised
            ## convolve and return, discarding padding
            self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        return f

    def convolve_with_lorentzian(self,width,fwhms_to_include=100):
        """Convolve with lorentzian."""
        p = self.add_parameter_set('convolve_with_lorentzian',width=width,step_default={'width':0.01})
        self.add_format_input_function(lambda: f'{self.name}.convolve_with_lorentzian({p.format_input()})')
        ## check if there is a risk that subsampling will ruin the convolution
        def f():
            x,y = self.x,self.y
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
            self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self.add_construct_function(f)

    def convolve_with_sinc(self,width=None,fwhms_to_include=100):
        """Convolve with sinc function, width is FWHM."""
        ## check if there is a risk that subsampling will ruin the convolution
        p = self.add_parameter_set('convolve_with_sinc',width=width)
        if 'sinc_FWHM' in self.experimental_parameters: # get auto width and make sure consistent with what is given
            if width is None: width = self.experimental_parameters['sinc_FWHM']
            if np.abs(np.log(p['width']/self.experimental_parameters['sinc_FWHM']))>1e-3: warnings.warn(f"Input parameter sinc FWHM {repr(p['width'])} does not match experimental_parameters sinc_FWHM {repr(self.experimental_parameters['sinc_FWHM'])}")
        self.add_format_input_function(lambda: f'{self.name}.convolve_with_sinc({p.format_input()})')
        if self.verbose and p['width']<3*np.diff(self.xexp).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        def f():
            x,y = self.x,self.y
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
            self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self.add_construct_function(f)

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
        self.add_format_input_function(f)
        # if self.verbose and p['width']<3*np.diff(self.xexp).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        instrument_function_cache = dict(y=None,width=None,) # for persistence between optimsiation function calls
        def f():
            dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
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
            xpad = np.concatenate((self.x[0]-padding[-1::-1],self.x,self.x[-1]+padding))
            ypad = np.concatenate((np.full(padding.shape,self.y[0]),self.y,np.full(padding.shape,self.y[-1])))
            self.y = signal.oaconvolve(ypad,instrument_function_cache['y'],mode='same')[len(padding):len(padding)+len(self.x)]
        self.add_construct_function(f)

    def apodise(
            self,
            apodisation_function=None,
            interpolation_factor=None,
            **kwargs_specific_to_apodisation_function,
    ):
        """Apodise the spectrum with a known function. This is done in the
        length-domain so both Fourier and inverse-Fourier transforms are
        required.  MUST INCLUDE ENTIRE SPECTRUM? CORRECTABLE?"""
        ## get apodisation_function and interpolation_factor 
        if apodisation_function is not None:
            pass
        elif 'apodisation_function' in self.experiment.experimental_parameters:
            apodisation_function = self.experiment.experimental_parameters['apodisation_function']
        else:
            raise Exception('apodisation_function not provided and not found in experimental_parameters.')
        # if 'interpolation_factor' not in self.experiment.experimental_parameters:
            # warnings.warn("interpolation_factor not found in experiment.experimental_parameters, assuming it is 1")
            # interpolation_factor = 1.
        cache = {'interpolation_factor':interpolation_factor,}
        def f():
            if cache['interpolation_factor'] is None:
                interpolation_factor = (self._interpolate_factor if self._interpolate_factor else 1)
                if 'interpolation_factor' in self.experiment.experimental_parameters:
                    interpolation_factor *= self.experiment.experimental_parameters['interpolation_factor']
            else:
                interpolation_factor = cache['interpolation_factor']
            ft = fft.dct(self.y) # get Fourier transform
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
            self.y = fft.idct(ft*w) # new apodised spetrum
        self.add_construct_function(f)
        # self.construct_functions.append(f)
        self.add_format_input_function(lambda: f'{self.name}.apodise({repr(apodisation_function)},{interpolation_factor},{tools.dict_to_kwargs(kwargs_specific_to_apodisation_function)})')


    # def convolve_with_blackman_harris(self,terms=3,fwhms_to_include=100):
        # """Convolve with sinc function, width is FWHM."""
        # ## check if there is a risk that subsampling will ruin the convolution
        # p = self.add_parameter_set('convolve_with_sinc',width=width)
        # if 'sinc_FWHM' in self.experimental_parameters: # get auto width and make sure consistent with what is given
            # if width is None: width = self.experimental_parameters['sinc_FWHM']
            # if np.abs(np.log(p['width']/self.experimental_parameters['sinc_FWHM']))>1e-3: warnings.warn(f"Input parameter sinc FWHM {repr(p['width'])} does not match experimental_parameters sinc_FWHM {repr(self.experimental_parameters['sinc_FWHM'])}")
        # self.add_format_input_function(lambda: f'{self.name}.convolve_with_sinc({p.format_input()})')
        # if self.verbose and p['width']<3*np.diff(self.xexp).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        # def f():
            # x,y = self.x,self.y
            # width = np.abs(p['width'])
            # ## get padded spectrum to minimise edge effects
            # dx = (x[-1]-x[0])/(len(x)-1) # ASSUMES EVEN SPACED GRID
            # padding = np.arange(dx,fwhms_to_include*width+dx,dx)
            # xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
            # ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
            # ## generate sinc to convolve with
            # xconv = np.arange(-fwhms_to_include*width,fwhms_to_include*width,dx)
            # if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
            # yconv = np.sinc((xconv-xconv.mean())/width*1.2)*1.2/width # unit integral normalised sinc
            # yconv = yconv/yconv.sum() # sum normalised
            # ## convolve and return, discarding padding
            # self.y = signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        # self.add_construct_function(f)

    @optimise_method()
    def convolve_with_blackman_harris(
            self,
            terms=3,
            interpolation_factor=None,
            fwhms_to_include=10,
            _cache=None,
    ):
        """Convolve with the coefficients equivalent to a Blackman-Harris
        window. Coefficients taken from harris1978 p. 65.  There are
        multiple coefficents given for 3- and 5-Term windows. I use
        the left.  This is faster than apodisation in the
        length-domain."""
        x,y = self.x,self.y
        ## determine how much interpolation there is
        if 'interpolate_factor' not in _cache:
            if interpolation_factor is None:
                ## in experimental spectrum
                if 'interpolation_factor' in self.experiment.experimental_parameters:
                    interpolation_factor = self.experiment.experimental_parameters['interpolation_factor']
                else:
                    interpolation_factor = 1
                    warnings.warn("interpolation_factor not found in experimental_parameters, assuming it is 1")
                ## added in  model
                if self._interpolate_factor is not None:
                    interpolation_factor *= self._interpolate_factor
            if interpolation_factor%1 != 0:
                raise Exception('Blackman-Harris convolution only valid for integer interpolation_factor, not {interpolation_factor}')
            interpolation_factor = int(interpolation_factor)
            _cache['interpolation_factor'] = interpolation_factor
        interpolation_factor = _cache['interpolation_factor']
        ## get padded spectrum to minimise edge effects
        dx = (x[-1]-x[0])/(len(x)-1) # ASSUMES EVEN SPACED GRID
        width = interpolation_factor*dx
        padding = np.arange(dx,fwhms_to_include*width+dx,dx)
        xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
        ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
        ## generate sinc to convolve with
        if 'yconv' not in _cache:
            xconv = np.arange(0,fwhms_to_include*width,dx)
            xconv = np.concatenate((-xconv[-1:0:-1],xconv))
            if terms == 3:
                yconv =  0.42323*np.sinc(xconv/width*1.2)
                yconv += 0.5*0.49755*np.sinc((xconv-width/1.2)/width*1.2)
                yconv += 0.5*0.49755*np.sinc((xconv+width/1.2)/width*1.2)
                yconv += 0.5*0.07922*np.sinc((xconv-2*width/1.2)/width*1.2)
                yconv += 0.5*0.07922*np.sinc((xconv+2*width/1.2)/width*1.2)
            else: 
                raise Exception("Only 3-term version implemented.")
            yconv /= yconv.sum()                    # normalised
            _cache['yconv'] = yconv
        else:
            yconv = _cache['yconv']
        ## self.y = signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]

    @optimise_method()
    def convolve_with_signum(
            self,
            amplitude,          # amplitude of signume
            xwindow=10,         # half-window for convolutiom
            xbeg=None,
            xend=None,
    ):
        """Convolve with signum function generating asymmetry. δ(x-x0) + amplitude/(x-x0)."""
        i = ((self.x>=(xbeg if xbeg is not None else -np.inf))
             &(self.x<=(xend if xend is not None else np.inf)))
        x,y = self.x[i],self.y[i]
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
        yconv = amplitude/xconv/len(xconv)               # signum function
        yconv[int((len(yconv)-1)/2)] = 1.       # add δ function at center
        yconv /= yconv.sum()                    # normalised
        self.y[i] = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]

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
            sinc_fwhm = (self.experiment.experimental_parameters['sinc_FWHM'],None,1e-3)
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
        if abs(self.experiment.experimental_parameters['sinc_FWHM']-p['sinc_fwhm'])>(1e-5*p['sinc_fwhm']):
            warnings.warn('sinc FWHM does not match SOLEIL data file header')
        ## rewrite input line
        def f():
            retval = f'{self.name}.convolve_with_SOLEIL_instrument_function({p.format_input()}'
            if p['sinc_fwhm']      !=0: retval += f',sinc_fwhms_to_include={sinc_fwhms_to_include}'
            if p['gaussian_fwhm']  !=0: retval += f',gaussian_fwhms_to_include={gaussian_fwhms_to_include}'
            if p['lorentzian_fwhm']!=0: retval += f',lorentzian_fwhms_to_include={lorentzian_fwhms_to_include}'
            retval += ')'
            return(retval)
        self.add_format_input_function(f)
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
            xpad = np.concatenate((self.x[0]-padding[-1::-1],self.x,self.x[-1]+padding))
            ypad = np.concatenate((np.full(padding.shape,self.y[0]),self.y,np.full(padding.shape,self.y[-1])))
            self.y = signal.oaconvolve(ypad,cache['y'],mode='same')[len(padding):len(padding)+len(self.x)]
        self.add_construct_function(f)

    def add_SOLEIL_double_shifted_delta_function(self,magnitude,shift=(1000,None)):
        """Adds two copies of SOLEIL spectrum onto the model (after
        convolution with instrument function perhaps). One copy shifted up by
        shift(cm-1) and one shifted down. Shifted up copy is scaled by
        magnitude, copy shifted down is scaled by -magnitude. This is to deal
        with periodic errors aliasing the spectrum due to periodic
        vibrations."""
        filename = self.experiment.experimental_parameters['filename']
        x,y,header = load_SOLEIL_spectrum_from_file(filename)
        yshifted ={'left':None,'right':None}
        p = self.add_parameter_set('add_SOLEIL_double_shifted_delta_function',
                                   magnitude=magnitude,shift=shift,
                                   step_default={'magnitude':1e-3,'shift':1e-3},)
        previous_shift = [p['shift']]
        self.add_format_input_function(lambda: f'{self.name}.add_SOLEIL_double_shifted_delta_function({p.format_input()})')
        def f():
            ## +shift from left -- use cached value of splined shifted
            ## spectrum if shift has not changed
            ## positive for this shift
            if yshifted['left'] is None or p['shift']!=previous_shift[0]:
                i = tools.inrange(self.x,x-p['shift'])
                j = tools.inrange(x+p['shift'],self.x)
                dy = tools.spline(x[j]+p['shift'],y[j],self.x[i])
                yshifted['left'] = (i,dy)
            else:
                i,dy = yshifted['left']
            self.y[i] += dy*p['magnitude']
            ## -shift from right -- use cached value of splined
            ## shifted spectrum if shift has not changed, magnitude is
            ## negative for this shift
            if yshifted['right'] is None or p['shift']!=previous_shift[0]:
                i = tools.inrange(self.x,x+p['shift'])
                j = tools.inrange(x-p['shift'],self.x)
                dy = tools.spline(x[j]-p['shift'],y[j],self.x[i])
                yshifted['right'] = (i,dy)
            else:
                i,dy = yshifted['right']
            self.y[i] += dy*p['magnitude']*-1
        self.add_construct_function(f)
        
    def plot(
            self,
            ax=None,
            fig=None,
            plot_model= True,
            plot_experiment= True,
            plot_residual= True,
            plot_labels=False,
            plot_branch_heads=False,
            qn_defining_branch=('speciesp','speciespp','labelp','labelpp','vp','vpp','Fp'),
            label_key=None,
            label_match_qn=None,
            minimum_τ_to_label=None, # for absorption lines
            minimum_I_to_label=None, # for emission lines
            plot_title= True,
            title='auto',
            plot_legend= True,
            plot_contaminants=True, # whether or not to label locations of reference contaminants
            # contaminants_to_plot=('default',), # what contaminant to label
            contaminants_to_plot=None, # what contaminant to label
            linewidth=1,
            shift_residual=0.,
            xlabel=None,ylabel=None,
            invert_model=False,
            plot_kwargs=None,
    ):
        """Plot experimental and model spectra."""
        if plot_kwargs is None:
            plot_kwargs = {}
        ## get axes if not specified
        if ax is not None:
            fig = ax.figure
        else:
            if fig is None:
                fig = plt.gcf()
            elif isinstance(fig,int):
                fig = plt.figure(fig)
            fig.clf()
            ax = fig.gca()
            def format_coord(x,y):
                if x<1e-5 or x>=1e10:
                    xstr = f'{x:0.18e}'
                else:
                    xstr = f'{x:0.18f}'
                if y<1e-5 or y>1e5:
                    ystr = f'{y:0.18e}'
                else:
                    ystr = f'{y:0.18f}'
                return(f'x={xstr:<25s} y={ystr:<25s}')
            ax.format_coord = format_coord
        self.add_format_input_function(lambda: f'{self.name}.plot_spectrum(fig={repr(fig.number)},label_key={repr(label_key)},plot_labels={repr(plot_labels)},plot_experiment={repr(plot_experiment)},plot_model={repr(plot_model)},plot_residual={repr(plot_residual)})')
        ymin,ymax = np.inf,-np.inf
        xmin,xmax = np.inf,-np.inf
        ## plot intensity and residual
        if plot_experiment and self.experiment is not None and self.yexp is not None:
            # ymin,ymax = min(ymin,self.yexp.min()),max(ymax,self.yexp.max())
            ymin,ymax = -0.1*self.yexp.max(),self.yexp.max()*1.1
            xmin,xmax = min(xmin,self.xexp.min()),max(xmax,self.xexp.max())
            tkwargs = copy(plot_kwargs)
            tkwargs.setdefault('color',plotting.newcolor(0))
            tkwargs.setdefault('label','Experimental spectrum')
            ax.plot(self.xexp,self.yexp,**tkwargs)
        if plot_model and self.y is not None:
            if invert_model:
                self.y *= -1
            ymin,ymax = min(ymin,self.y.min(),-0.1*self.y.max()),max(ymax,self.y.max()*1.1)
            xmin,xmax = min(xmin,self.x.min()),max(xmax,self.x.max())
            tkwargs = copy(plot_kwargs)
            tkwargs.setdefault('color',plotting.newcolor(1))
            tkwargs.setdefault('label','Model spectrum')
            ax.plot(self.x,self.y,**tkwargs)
            if invert_model:
                self.y *= -1
        if plot_residual and self.residual is not None and len(self.residual)>0:
            ymin,ymax = min(ymin,self.residual.min()+shift_residual),max(ymax,self.residual.max()+shift_residual)
            xmin,xmax = min(xmin,self.xexp.min()),max(xmax,self.xexp.max())
            tkwargs = copy(plot_kwargs)
            tkwargs.setdefault('color',plotting.newcolor(2))
            tkwargs.setdefault('label','Exp-Mod residual error')
            ax.plot(self.xexp,self.residual+shift_residual,zorder=-1,**tkwargs) # plot fit residual
        ## annotate rotational series
        if plot_labels:
            ystep = ymax/20.
            for line in self.suboptimisers:
                if not isinstance(line,lines.Generic):
                    continue
                ## limit to qn
                if label_match_qn is None:
                    label_match_qn = {}
                if not line.is_known(*label_match_qn):
                    continue
                i = line.match(**label_match_qn)
                ## limit to ν-range and sufficiently strong line
                i &= (line['ν']>self.xexp[0])&(line['ν']<self.xexp[-1])
                if minimum_τ_to_label is not None:
                    i &= line['τ']>minimum_τ_to_label
                if minimum_I_to_label is not None:
                    i &= line['I']>minimum_I_to_label
                line = line[i]
                if len(line)==0:
                    continue
                zkeys = [key for key in line.defining_qn if line.is_known(key)]
                branch_annotations = plotting.annotate_spectrum_by_branch(
                    line,
                    ymax+ystep/2.,
                    ystep,
                    zkeys=zkeys,
                    length=-0.02, # fraction of axes coords
                    color_by=('branch' if 'branch' in zkeys else zkeys),
                    labelsize='xx-small',namesize='x-small', namepos='float',    
                    label_key=label_key,
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
        if (plot_title
            and hasattr(self.experiment,'experimental_parameters')
            and 'filename' in self.experiment.experimental_parameters):
            if title == 'auto':
                title = tools.basename(self.experiment.experimental_parameters['filename'])
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

    def output_data_to_directory(
            self,
            directory='td',
            output_model_residual=False,
            output_transition_linelists=False,
            output_individual_optical_depths=False,
    ):
        """Save various files from this optimsiation to a directory."""
        tools.mkdir(directory)
        ## model data
        if self.x is not None and self.y is not None:
            tools.array_to_file(directory+'/spectrum.h5',self.x,self.y)
        # if output_residual and self.residual is not None:
            # tools.array_to_file(directory+'/residual.h5', self.xexp,self.residual)
        if self._figure is not None:
            self._figure.savefig(directory+'/figure.png',dpi=300) # save figure
        ## save transition linelists
        if output_transition_linelists:
            tools.mkdir_if_necessary(directory+'/transitions')
            for transition in self.absorption_transitions:
                transition.save_to_file(directory+'/transitions/'+transition.name+'.h5')

    # def load_from_directory(self,directory):
        # """Load internal data from a previous "output_to_directory" model."""
        # self.experimental_parameters['filename'] = directory
        # directory = tools.expand_path(directory)
        # assert os.path.exists(directory) and os.path.isdir(directory),f'Directory does not exist or is not a directory: {repr(directory)}'
        # for filename in (
                # directory+'/experimental_spectrum', # text file
                # directory+'/experimental_spectrum.gz', # in case compressed
                # directory+'/experimental_spectrum.h5', # in case compressed
                # directory+'/exp',                      # deprecated
        # ):
            # if os.path.exists(filename):
                # self.xexp,self.yexp = tools.file_to_array_unpack(filename)
                # break
        # for filename in (
                # directory+'/model_spectrum',
                # directory+'/model_spectrum.gz', 
                # directory+'/model_spectrum.h5', 
                # directory+'/mod',
        # ):
            # if os.path.exists(filename):
                # self.x,self.y = tools.file_to_array_unpack(filename)
                # break
        # for filename in (
                # directory+'/model_residual',
                # directory+'/model_residual.gz',
                # directory+'/model_residual.h5',
                # directory+'/residual',
        # ):
            # if os.path.exists(filename):
                # # t,self.residual = tools.file_to_array_unpack(filename)
                # t = tools.file_to_array(filename)
                # if t.ndim==1:
                    # self.residual = t
                # elif t.ndim==2:
                    # self.residual = t[:,1]
                # break
        # # for filename in (
                # # directory+'/optical_depth',
                # # directory+'/optical_depth.gz',
                # # directory+'/optical_depth.h5',
        # # ):
            # # if os.path.exists(filename):
                # # t,self.optical_depths['total'] = t
                # # break
        # # for filename in tools.myglob(directory+'/optical_depths/*'):
            # # self.optical_depths[tools.basename(filename)] = tools.file_to_array_unpack(filename)[1]
        # # for filename in tools.myglob(directory+'/transitions/*'):
            # # self.transitions.append(load_transition(
                # # filename,
                # # Name=tools.basename(filename),
                # # decode_names=False,
                # # permit_new_keys=True,
                # # # error_on_unknown_key=False, # fault tolerant
            # # ))

    # def show(self):
        # """Show plots."""
        # self.add_format_input_function(f'{self.name}.show()')
        # plt.show()


# class Fit(Optimiser):
    # def __init__(
            # self,
            # experiment,
            # model,
            # name=None,
            # residual_weighting=None,
            # verbose=None,
            # xbeg=None,
            # xend=None,
    # ):
        # self.experiment = experiment
        # self.model = model
        # self.xbeg = xbeg
        # self.xend = xend
        # self.residual = None
        # if name is None:
                # name = f'fit_{model.name}_{experiment.name}'
        # self.residual_weighting = residual_weighting
        # optimise.Optimiser.__init__(self,name)
        # self.pop_format_input_function()
        # self.add_suboptimiser(self.model)
        # self.add_suboptimiser(self.experiment)
        # self.add_construct_function(self.get_residual)

    # def get_residual(self):
        # ## limit to xbeg → xend
        # i = np.full(True,self.experiment.x.shape)
        # if self.xbeg is not None:
            # i &= self.experiment.x>=self.xbeg
        # if self.xend is not None:
            # i &= self.experiment.x<=self.xend
        # ymod = self.model.get_spectrum(self.experiment.x[i])
        # self.residual = self.experiment.yexp - ymod
        # if self.residual_weighting is not None:
            # self.residual *= self.residual_weighting
        # return self.residual

        
def load_spectrum(filename,**kwargs):
    """Use a heuristic method to load a directory output by
    Spectrum."""
    x = Spectrum()
    x.load_from_directory(filename,**kwargs)
    return(x)
