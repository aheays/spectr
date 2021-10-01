import inspect,re
from copy import copy,deepcopy
from pprint import pprint
import warnings
from time import perf_counter as timestamp

from matplotlib import pyplot as plt
from scipy import signal,constants,fft,interpolate
from scipy.constants import pi as π
import numpy as np
from numpy import arange
import h5py
from immutabledict import immutabledict as idict


from . import optimise
from .optimise import *
from . import plotting
from . import tools
from . import hitran
from . import bruker
from . import lineshapes
from . import lines
from . import levels
from . import exceptions
from . import dataset
from . import database
from . import fortran_tools
from .dataset import Dataset

class Experiment(Optimiser):
    
    def __init__(
            self,
            name='experiment',
            filename=None,
            x=None,
            y=None,
            xbeg=None,
            xend=None,
            noise_rms=None,
            store=None,
    ):
        ## initalise optimiser variables
        optimise.Optimiser.__init__(self,name,store=store)
        self.pop_format_input_function() 
        store = self.store
        self.automatic_format_input_function()
        ## initialise data arrays
        self.x = None
        self.y = None
        self.experimental_parameters = {} # a dictionary containing any additional known experimental parameters
        if filename is not None:
            self.set_spectrum_from_file(filename,xbeg=xbeg,xend=xend)
        if x is not None and y is not None:
            self.set_spectrum(x,y,xbeg,xend)
        if noise_rms is not None:
            self.experimental_parameters['noise_rms'] = float(noise_rms)
        self.add_save_to_directory_function(self.output_data_to_directory)
            # lambda directory: tools.array_to_file(directory+'/spectrum.h5',self.x,self.y))
        self._figure = None

    @optimise_method()
    def set_spectrum(self,x,y,xbeg=None,xend=None,_cache=None,**experimental_parameters):
        """Set x and y as the experimental spectrum. With various safety
        checks. Not optimisable, no format_input_function."""
        if self._clean_construct:
            x,y = np.array(x),np.array(y)
            i = np.argsort(x); x,y = x[i],y[i] # sort
            ## set range
            if xbeg is None:
                xbeg = x[0]
            if xend is None:
                xend = x[-1]
            if x[0] > xend or x[-1] < xbeg:
                raise Exception(f'No data in range {xbeg} to {xend}')
            i = (x>=xbeg)&(x<=xend)
            x,y = x[i],y[i]
            self.experimental_parameters['xbeg'] = xbeg 
            self.experimental_parameters['xend'] = xend
            ## check for regular x grid
            t0,t1 = np.diff(x).min(),np.diff(x).max()
            assert (t1-t0)/t1<1e-3, 'Experimental data must be on an uniform grid.' # within a factor of 1e3
            self.experimental_parameters.update(experimental_parameters)
            ## verbose info
            if self.verbose:
                print('experimental_parameters:')
                pprint(self.experimental_parameters)
            _cache['x'],_cache['y'] = x,y
        ## every iteration make a copy -- more memory but survives
        ## in-place other changes
        self.x,self.y = copy(_cache['x']),copy(_cache['y']) 

    def output_data_to_directory(self,directory):
        """Save various files from this optimsiation to a directory."""
        tools.mkdir(directory)
        ## model data
        if self.x is not None and self.y is not None:
            t = Spectrum(ν=self.x,I=self.y,description=f'Experimental spectrum of {self.name}')
            t.save(f'{directory}/spectrum',filetype='directory')
        if self._figure is not None:
            ## svg / pdf are the fastest output formats. Significantly
            ## faster if there is not text on the figure
            # self._figure.savefig(directory+'/figure.png',dpi=300)
            self._figure.savefig(f'{directory}/figure.pdf')
        if len(self.experimental_parameters) > 0:
            tools.save_dict(
                f'{directory}/experimental_parameters.py',
                experimental_parameters=self.experimental_parameters,
                header=f'## experimental parameters of {self.name}\n')


    @optimise_method()
    def set_spectrum_from_file(
            self,
            filename,
            xkey=None,ykey=None,
            xcol=None,ycol=None,
            xbeg=None,xend=None,
            # **load_function_kwargs
            _cache=None,
    ):
        """Load a spectrum to fit from an x,y file."""
        ## only runs once
        if 'has_run' in _cache:
            return
        _cache['has_run'] = True
        # self.add_format_input_function(lambda:self.name+f'.set_spectrum_from_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)},{tools.dict_to_kwargs(load_function_kwargs)})')
        # x,y = tools.file_to_array_unpack(filename,**load_function_kwargs)
        x,y = tools.loadxy(filename, xcol=xcol,ycol=ycol, xkey=xkey,ykey=ykey,)
        self.experimental_parameters['filename'] = filename
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function()

    def set_spectrum_from_dataset(self,filename,xbeg=None,xend=None,xkey='x',ykey='y'):
        """Load a spectrum to fit from an x,y file."""
        d = Dataset()
        d.load(filename)
        experimental_parameters = {key:val for key,val in d.items() if  key not in (xkey,ykey)}
        self.set_spectrum(d[xkey],d[ykey],xbeg,xend,filename=filename,**d.attributes)

    @optimise_method()
    def set_spectrum_from_hdf5(self,filename,xkey='x',ykey='y',xbeg=None,xend=None,_cache=None):
        """Load a spectrum to fit from an x,y file."""
        ## only runs once
        if 'has_run' in _cache:
            return
        _cache['has_run'] = True
        self.experimental_parameters['filename'] = filename
        with h5py.File(tools.expand_path(filename),'r') as data:
            x = np.asarray(data[xkey],dtype=float)
            y = np.asarray(data[ykey],dtype=float)
            ## get any experimental_parameters as attributes
            for key,val in data.attrs.items():
                self.experimental_parameters[key] = val
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function()

    @optimise_method()
    def set_spectrum_from_opus_file(self,filename,xbeg=None,xend=None,_cache=None):
        """Load a spectrum in an Bruker opus binary file."""
        ## only runs once
        if 'has_run' in _cache:
            return
        _cache['has_run'] = True
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
        if 'Acquisition' in d:
            self.experimental_parameters['resolution'] = float(d['Acquisition']['RES'])
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function() 
        # self.add_format_input_function(lambda:self.name+f'.set_spectrum_from_opus_file({repr(filename)},xbeg={repr(xbeg)},xend={repr(xend)})')
    
    @optimise_method()
    def set_spectrum_from_soleil_file(self,filename,xbeg=None,xend=None,_cache=None):
        """ Load soleil spectrum from file with given path."""
        ## only runs once
        if 'has_run' in _cache:
            return
        _cache['has_run'] = True
        x,y,header = load_soleil_spectrum_from_file(filename)
        self.experimental_parameters['filename'] = filename
        self.experimental_parameters.update(header)
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function() 

    @optimise_method()
    def interpolate(self,xstep,order=1):
        """Spline interpolate to a regular grid defined by xtstep."""
        xnew = np.arange(self.x[0],self.x[-1],xstep)
        self.y = tools.spline(self.x,self.y,xnew,order=order)
        self.x = xnew

    @optimise_method()
    def bin_data(self,n=None,width=None,mean=False,_cache=None):
        """Bin to every nth datapoint computing the 'mean' or 'sum' of y
        (according to out) and mean of x."""
        if width is not None:
            dx = (self.x[-1]-self.x[0])/len(self.x)
            n = max(1,np.ceil(width/dx))
        self.x = tools.bin_data(self.x,n,mean=True)
        self.y = tools.bin_data(self.y,n,mean=mean)

    @optimise_method()
    def convolve_with_gaussian(self,width):
        """Convolve spectrum with a gaussian."""
        self.y = tools.convolve_with_gaussian(self.x,self.y,width)

    @optimise_method()
    def set_soleil_sidebands(self,yscale=0.1,shift=1000,signum_magnitude=None,_cache=None):
        """Adds two copies of spectrum onto itself shifted rightwards and
        leftwards by shift (cm-1) and scaled by ±yscale."""
        ## load and cache spectrum
        if self._clean_construct:
            x,y,header = load_soleil_spectrum_from_file(self.experimental_parameters['filename'])
            _cache['x'],_cache['y'] = x,y
        x,y = _cache['x'],_cache['y']
        ## get signum convolution kernel
        if signum_magnitude is not None:
            dx = (x[-1]-x[0])/(len(x)-1)
            nconv = int(10/dx)
            xconv = np.linspace(-nconv,nconv,2*nconv+1)
            yconv = np.full(len(xconv),1.0)
            i = xconv!=0
            yconv[i] = 1/xconv[i]*signum_magnitude
        ## shift from left
        i = ((x+shift) >= self.x[0]) & ((x+shift) <= self.x[-1])
        xi,yi = x[i]+shift,y[i]
        j = (self.x>=xi[0]) & (self.x<=xi[-1])
        sideband = yscale*tools.spline(xi,yi,self.x[j])
        if signum_magnitude is not None:
            sideband = signal.oaconvolve(sideband,yconv,'same')
        self.y[j] += sideband
        ## shift from right
        i = ((x-shift) >= self.x[0]) & ((x-shift) <= self.x[-1])
        xi,yi = x[i]-shift,y[i]
        j = (self.x>=xi[0]) & (self.x<=xi[-1])
        sideband = yscale*tools.spline(xi,yi,self.x[j])
        if signum_magnitude is not None:
            sideband = signal.oaconvolve(sideband,yconv,'same')
        self.y[j] -= sideband
        
    @optimise_method()
    def scalex(self,scale=1):
        """Rescale experimental spectrum x-grid."""
        self.x *= float(scale)

    def __len__(self):
        return len(self.x)

    @format_input_method(format_multi_line=inf)
    def fit_noise(self,xbeg=None,xend=None,xedge=None,n=1,make_plot=False,figure_number=None):
        """Estimate the noise level by fitting a polynomial of order n
        between xbeg and xend to the experimental data. Also rescale
        if the experimental data has been interpolated."""
        x,y = self.x,self.y
        if xedge is not None:
            xbeg = self.x[0]
            xend = self.x[0] + xedge
        if xbeg is not None:
            i = x >= xbeg
            x,y = x[i],y[i]
        if xend is not None:
            i = x <= xend
            x,y = x[i],y[i]
        if len(x) == 0:
            warnings.warn(f'{self.name}: No data in range for fit_noise, not done.')
            return
        xt = x - x.mean()
        p = np.polyfit(xt,y,n)
        yf = np.polyval(p,xt)
        r = y-yf
        rms = np.sqrt(np.sum(r**2)/(len(r)-n+1))
        self.experimental_parameters['noise_rms'] = rms
        self.residual_scale_factor = 1/rms
        ## make optional plot
        if make_plot:
            ax = plotting.qax(n=figure_number)
            ax.plot(x,y,label='exp')
            ax.plot(x,yf,label='fit')
            ax.plot(x,r,label=f'residual, rms={rms:0.3e}')
            ax.set_title(f'fit rms to data\n{self.name}')
            plotting.legend(ax=ax)
            ax = plotting.subplot()
            ax.plot(tools.autocorrelate(r),marker='o',)
            ax.set_title(f'noise autocorrelation\n{self.name}')
            ax = plotting.subplot()
            plotting.plot_hist_with_fitted_gaussian(r,ax=ax)
            ax.set_title(f'noise distribution\n{self.name}')
        return rms

    def plot(self,ax=None):
        """Plot spectrum."""
        self.construct()
        ## reuse current axes if not specified
        if ax is None:
            ax = plotting.gca()
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
        ax.plot(self.x,self.y,label=self.name)
        plotting.legend(ax=ax,loc='upper left')
        self._figure = plotting.gcf()
        return ax

class Model(Optimiser):

    def __init__(
            self,
            name=None,
            experiment=None,
            load_experiment_args=None,
            residual_weighting=None,
            verbose=None,
            xbeg=None,xend=None,
    ):
        ## build an Experiment from method givens as dict of method:args
        if load_experiment_args is not None:
            experiment = Experiment('experiment')
            for method,args in load_experiment_args.items():
                getattr(experiment,method)(*args)
        self.experiment = experiment
        if name is None:
            if self.experiment is None:
                name = 'model'
            else:
                name = f'model_of_{experiment.name}'
        self.x = None
        self.y = None
        self._xin = None
        self._xcache = None
        self.xbeg = xbeg
        self.xend = xend
        self.residual = None                      # array of residual fit
        self.residual_weighting = residual_weighting            # weighting pointwise in xexp
        self._interpolate_factor = None
        optimise.Optimiser.__init__(self,name)
        self.pop_format_input_function()
        self.automatic_format_input_function()
        if self.experiment is not None:
            self.add_suboptimiser(self.experiment)
            self.pop_format_input_function()
        self._initialise()
        self.add_post_construct_function(self.get_residual)
        self.add_save_to_directory_function(self.output_data_to_directory)
        self.add_plot_function(lambda: self.plot(plot_labels=False))
        self._figure = None

    @optimise_method(add_format_input_function=False)
    def _initialise(self,_cache=None):
        """Function run before everything else to set x and y model grid and
        residual_scale_factor if experimental noise_rms is known."""
        ## clean construct if the experimental domain has changed
        if ('xexp' in _cache
            and (len(self.experiment.x) != len(_cache['xexp'])
                 or np.any(self.experiment.x != _cache['xexp']))):
            self._clean_construct = True
        if self._clean_construct:
            ## get new x grid
            self.residual_scale_factor = 1
            if self._xin is not None:
                ## x is from a call to get_spectrum
                self.x = self._xin
                self._xin = None
                self._compute_residual =False
            elif self.experiment is not None:
                if 'iexp' not in _cache:
                    if self.xbeg is None:
                        ibeg = 0
                    else:
                        ibeg = np.searchsorted(self.experiment.x,self.xbeg)
                    if self.xend is None:
                        iend = len(self.experiment.x)
                    else:
                        iend = np.searchsorted(self.experiment.x,self.xend)
                    iexp = slice(int(ibeg),int(iend))
                    _cache['iexp'] = iexp
                iexp = _cache['iexp']
                self.x = self.xexp = self.experiment.x[iexp]
                _cache['xexp'] = self.experiment.x
                self.yexp = self.experiment.y[iexp]
                ## if known use experimental noise RMS to normalise
                ## residuals
                if 'noise_rms' in self.experiment.experimental_parameters:
                    self.residual_scale_factor = 1./self.experiment.experimental_parameters['noise_rms']
                self._compute_residual = True
            elif self.xbeg is not None and self.xend is not None:
                ## xbeg to xend
                self.x = linspace(self.xbeg,self.xend,1000,dtype=float)
                self._compute_residual =False
            else:
                ## default
                self.x = linespace(100,1000,1000,dtype=float)
                self._compute_residual =False
        ## new grid
        self.y = np.zeros(self.x.shape,dtype=float)
        
    def get_spectrum(self,x):
        """Construct a model spectrum at x."""
        self._xin = np.asarray(x,dtype=float) # needed in _initialise
        # self.timestamp = -1     # force reconstruction, but not suboptimisers
        self._clean_construct = True   # force reconstruction
        self.construct()
        self._xin = None        # might be needed in next _initialise
        return self.y

    def get_residual(self):
        """Compute residual error."""
        if self._compute_residual:
            if self._interpolate_factor is not None:
                self.uninterpolate(average=False)
            if self.experiment is None:
                return []
            residual = self.yexp - self.y
            if self.residual_weighting is not None:
                residual *= self.residual_weighting
        else:
            residual = None
        return residual

    @optimise_method()
    def interpolate(self,dx,_cache=None):
        """When calculating model set to dx grid (or less to achieve
        overlap with experimental points. Always an odd number of
        intervals / even number of interstitial points. DELETES
        CURRENT Y!"""
        # if self._clean_construct or self._xchanged:
        if self._clean_construct:
            xstep = (self.x[-1]-self.x[0])/(len(self.x)-1)
            interpolate_factor = int(np.ceil(xstep/dx))
            if interpolate_factor%2 == 0:
                interpolate_factor += 1
            _cache['x'] = np.linspace(self.x[0],self.x[-1],1+(len(self.x)-1)*interpolate_factor)
            _cache['y'] = np.zeros(_cache['x'].shape,dtype=float)
            _cache['interpolate_factor'] = interpolate_factor
        self._interpolate_factor = _cache['interpolate_factor']
        self.x = _cache['x']
        self.y = _cache['y'].copy() # delete current y!!

    @optimise_method()
    def uninterpolate(self,average=None):
        """If the model has been interpolated then restore it to
        original grid. If average=True then average the values in each
        interpolated interval."""
        if self._interpolate_factor is not None:
            self.x = self.x[::self._interpolate_factor]
            if average:
                ## ## reduce to non-interpolated points -- using python
                ## half = int(self._interpolate_factor/2)
                ## shifts = np.arange(1, half+1, dtype=int)
                ## y = self.y[::self._interpolate_factor]
                ## ## add points to right
                ## y[:-1] += np.sum([self.y[shift::self._interpolate_factor] for shift in shifts],0)
                ## ## add points to left
                ## y[1:] += np.sum([self.y[shift+half::self._interpolate_factor] for shift in shifts],0)
                ## ## convert sum to mean
                ## y[0] = y[0] / (self._interpolate_factor/2+1)
                ## y[-1] = y[-1] / (self._interpolate_factor/2+1)
                ## y[1:-1] = y[1:-1] / self._interpolate_factor
                ## self.y = y
                ## reduce to non-interpolated points -- using python
                y = np.empty(self.x.shape,dtype=float)
                fortran_tools.uninterpolate_with_averaging(self.y,y,self._interpolate_factor)
                self.y = y
            else:
                ## reduce to non-interpolated points
                self.y = self.y[::self._interpolate_factor]
            self._interpolate_factor = None

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

    @format_input_method()
    def add_hitran_line(
            self,
            species,
            match=None,
            *args,
            **kwargs,
            
    ):
        line = hitran.get_lines(species,match=match)
        line.clear_format_input_functions()
        self.add_line(line,*args,**kwargs)
        self.pop_format_input_function()

    @optimise_method()
    def add_line(
            self,
            line,               # lines.Generic or a subclass
            kind='absorption',  # 'absorption', 'emission' or else any other key defined in line
            nfwhmL=20,          # number of Lorentzian fwhms to include
            nfwhmG=10,          # number of Gaussian fwhms to include
            ymin=0,             # minimum value of ykey to include
            lineshape=None,     # as in lineshapes.py, or will be automatically selected
            ncpus=1,            # for multiprocessing
            verbose=None,       # print info 
            match=None,         # only include lines matching keys:vals in this dictionary
            _cache=None,
            **set_keys_vals
    ):
        ## nothing to be done
        if len(self.x) == 0 or len(line) == 0:
            return
        if self._clean_construct:
            ## first run — initalise local copy of lines data, do not
            ## set keys if they are in set_keys_vals
            tmatch = {} if match is None else copy(match)
            tmatch.setdefault('min_ν',(self.x[0]-1))
            tmatch.setdefault('max_ν',(self.x[-1]+1))
            imatch = line.match(**tmatch)
            nmatch = np.sum(imatch)
            keys = [key for key in line.explicitly_set_keys() if key not in set_keys_vals]
            line_copy = line.copy(index=imatch,keys=keys)
            if verbose is not None:
                line_copy.verbose = verbose
            if verbose:
                print('add_line: clean construct')
            ## set parameter/constant data
            for key,val in set_keys_vals.items():
                line_copy[key] = val
            ## get ykey
            if kind == 'absorption':
                ykey = 'τ'
            elif kind == 'emission':
                ykey = 'I'
            else:
                ykey = kind
            ## calculate full spectrum
            def _calculate_spectrum(line,index):
                if len(line) == 0:
                    return np.full(len(self.x),0.0)
                x,y = line.calculate_spectrum(
                    x=self.x,xkey='ν',ykey=ykey,nfwhmG=nfwhmG,nfwhmL=nfwhmL,
                    ymin=ymin, ncpus=ncpus, lineshape=lineshape,index=index)
                if kind == 'absorption':
                    y = np.exp(-y)
                return y
            y = _calculate_spectrum(line_copy,None)
            ## important data — only update spectrum if these have
            ## changed -- HACK
            data = {key:line_copy[key] for key in ('ν',ykey,'ΓL','ΓD') if line_copy.is_known(key)}
            ## cache
            (
                _cache['data'],_cache['y'],
                _cache['imatch'],_cache['nmatch'],
                _cache['line_copy'],_cache['ykey'],
                _cache['_calculate_spectrum'],
            ) = (
                data,y,imatch,nmatch,line_copy,ykey,_calculate_spectrum,
            )
        else:
            ## subsequent runs -- maybe only recompute a few line
            ##
            ## load cache
            (data,y,imatch,nmatch,line_copy,ykey,_calculate_spectrum,) = (
                 _cache['data'],_cache['y'],
                 _cache['imatch'],_cache['nmatch'],
                 _cache['line_copy'],_cache['ykey'],
                 _cache['_calculate_spectrum'],
            ) 
            ## nothing to be done
            if nmatch == 0:
                return
            ## set modified data in set_keys_vals if they have changed
            ## from cached values.  Only update Parameters (assume
            ## other types ## cannot change)
            for key,val in set_keys_vals.items():
                if isinstance(val,Parameter):
                    if self._last_construct_time < val._last_modify_value_time:
                        line_copy[key] = val
            ## if the source lines data has changed then update
            ## changed rows and keys in the local copy
            if line.global_modify_time > self._last_construct_time:
                for key in line.explicitly_set_keys():
                    if line[key,'_modify_time'] > self._last_construct_time:
                        line_copy.set(key,'value',line[key,imatch],set_changed_only=True)
            ## update spectrum for any changed lines in the local copy
            if line_copy.global_modify_time > self._last_construct_time:
                ## get indices of local lines that has been changed
                ichanged = line_copy.row_modify_time > self._last_construct_time
                nchanged = np.sum(ichanged)
                if (
                        ## all lines have changed
                        (nchanged == len(ichanged))
                        ## ykey has changed
                        and (line_copy[ykey,'_modify_time'] > self._last_construct_time)
                        ## no key other than ykey has changed
                        and (np.all([line_copy[key,'_modify_time'] < self._last_construct_time for key in data if key != ykey]))
                        ## ykey has changed by a near-constant factor -- RISKY!!!!
                        and _similar_within_fraction(data[ykey],line_copy[ykey])
                        ## if ymin is set then scaling is dangerous -- lines can fail to appear when scaled up
                        and (ymin is None or ymin == 0)
                ):
                    ## constant factor ykey -- scale saved spectrum
                    if ymin is not None and ymin != 0:
                        warnings.warn(f'Scaling spectrum uniformly but ymin is set to a nonzero value, {repr(ymin)}.  This could lead to lines appearing in subsequent model constructions.')
                    if verbose:
                        print('add_line: constant factor scaling all lines')
                    ## ignore zero values
                    i = (line_copy[ykey]!=0)&(data[ykey]!=0)
                    scale = np.mean(line_copy[ykey,i]/data[ykey][i])
                    if kind == 'absorption':
                        y = y**scale
                    else:
                        y = y*scale
                elif nchanged/len(ichanged) > 0.5:
                    ## more than half lines have changed -- full
                    ## recalculation
                    if verbose:
                        print(f'add_line: more than half the lines ({nchanged}/{len(ichanged)}) have changed, full recalculation')
                    y = _calculate_spectrum(line_copy,None)
                else:
                    ## a few lines have changed, update these only
                    if verbose:
                        print(f'add_line: {nchanged} lines have changed, recalculate these')
                    ## temporary object to calculate old spectrum
                    line_old = dataset.make(
                        classname=line_copy.classname,
                        **{key:data[key][ichanged] for key in data})
                    yold = _calculate_spectrum(line_old,None)
                    ## partial recalculation
                    ynew = _calculate_spectrum(line_copy,ichanged)
                    if kind == 'absorption':
                        y = y * ynew / yold
                    else:
                        y = y + ynew - yold
            else:
                ## nothing changed keep old spectrum
                pass
        ## nothing to be done
        if nmatch == 0:
            return
        ## apply to model
        if kind == 'absorption':
            self.y *= y
        else:
            self.y += y
        _cache['y'] = y
        _cache['data'] = {key:line_copy[key] for key in data}
        _cache['set_keys_vals'] = {key:(val.value if isinstance(val,Parameter) else val) for key,val in set_keys_vals.items()}


    @optimise_method()
    def add_absorption_cross_section(self,x,y,N=1,_cache=None):
        if self._clean_construct:
            _cache['ys'] = tools.spline(x,y,self.x)
        ys = _cache['ys']
        self.y *= np.exp(-N*ys)

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
                ΓL=lines['ΓL'][i],
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
                self.residual_weighting = np.full(self.yexp.shape,weighting,dtype=float)
            else:
                if self.residual_weighting is None:
                    self.residual_weighting = np.ones(self.yexp.shape,dtype=float)
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
            if self.residual is not None:
                y = copy(self.residual) # get from residual
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
            ν = np.arange(self.x[0]-ν,self.x[-1]+ν*1.01,ν) # default to a list of ν with spacing given by ν
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

    @optimise_method()
    def add_constant(self,intensity=1):
        """Shift by a spline defined function."""
        self.y += float(intensity)

    def add_intensity(self,*args,**kwargs):
        """Deprecated"""
        self.add_constant(*args,**kwargs)

    @optimise_method()
    def add_from_file(self,filename,scale=1,_cache=None):
        """Spline contents of file to model grid and add to spectrum."""
        if self._clean_construct:
            if 'x' not in _cache:
                x,y = tools.file_to_array_unpack(filename)
                _cache['x'],_cache['y'] = x,y
            x,y = _cache['x'],_cache['y']
            ys = tools.spline(x,y,self.x)
            _cache['ys'] = ys
        ys = _cache['ys']
        self.y += ys * float(scale)
    
    def auto_add_spline(self,xi=100,vary=False,make_plot=False):
        """Quickly add an evenly-spaced intensity spline. If x is a vector
        then define spline at those points."""
        ## boundaries to cater to instrument convolution
        xbeg,xend = self.x[0]-xi/2,self.x[-1]+xi/2 
        ## get experimental spectrum
        self.experiment.construct()
        i = tools.inrange(self.experiment.x,xbeg,xend)
        x,y = self.experiment.x[i],self.experiment.y[i]
        ## get good spline points from median of experimental data
        xs,ys,yf = tools.fit_spline_to_extrema_or_median(x,y,xi=xi,make_plot=make_plot)
        ## ensure end points in included
        if xs[0] > x[0]:
            xs = np.concatenate((x[0:1],xs))
            ys = np.concatenate((yf[0:1],ys))
        if xs[-1] < x[-1]:
            xs = np.concatenate((xs,x[-1:]))
            ys = np.concatenate((ys,yf[-1:]))
        knots = [[xsi,P(ysi,vary)] for (xsi,ysi) in zip(xs,ys)]
        self.add_spline(knots)
        return knots

    @optimise_method()
    def add_spline(self,knots=None,order=3,_cache=None,_parameters=None,):
        """Add a spline."""
        if self._clean_construct:
            xs = [t[0] for t in knots]
            ys = [t[1] for t in knots]
            i = (self.x >= np.min(xs)) & (self.x <= np.max(xs))
            _cache['i'] = i
            _cache['xs'] = xs
            _cache['ys'] = ys
        if (self._clean_construct 
            or np.any([t._last_modify_value_time > self._last_construct_time
                       for t in _parameters])):
            y = tools.spline(_cache['xs'],_cache['ys'],self.x[_cache['i']],order=order)
            _cache['y'] = y
        self.y[_cache['i']] += _cache['y']

    add_intensity_spline = add_spline # deprecated

    def auto_multiply_spline(self,x=1000.,y=None,vary=True,order=3,construct=True):
        """Quickly add an evenly-spaced spline to multiply . If x is a vector
        then define spline at those points. If y is not given then fit to the experiment."""
        self.experiment.construct()
        if np.isscalar(x):
            xbeg,xend = self.x[0]-x/2,self.x[-1]+x+2 # boundaries to cater to instrument convolution
            x = linspace(xbeg,xend,max(2,int((xend-xbeg)/x)))
        ## make knots
        knots = []
        for xi in x:
            if y is None:
                ## get current value of model as y
                i = min(np.searchsorted(self.x,xi),len(self.y)-1)
                if self.y[i] != 0:
                    yi = self.yexp[i]/self.y[i]
                else:
                    yi = 1
                ystep = (yi*1e-3 if yi !=0 else 1e-4)
            else:
                ## fixed y
                yi = y
                ystep = None
            knots.append([xi,P(yi,vary,ystep)])
        if construct:
            self.multiply_spline(knots=knots,order=order)
        return knots
    
    @optimise_method()
    def multiply_spline(self,knots=None,order=3,_cache=None,_parameters=None):
        """Multiple y by a spline function."""
        ## make the spline if necessary
        if 'spline' not in _cache:
            spline = Spline(f'{self.name}_multiply_spline',knots,order)
            self.add_suboptimiser(spline)
            self.pop_format_input_function()
            spline.clear_format_input_functions()
            _cache['spline'] = spline
        spline = _cache['spline'] 
        ## compute the  spline if necessary
        # if self._clean_construct or self._xchanged:
        if self._clean_construct:
            i = (self.x >= np.min(spline.xs)) & (self.x <= np.max(spline.xs))
            ## a quick hack to prevent round off errors missing the
            ## first or last points of the splined domain
            if not i[0] and i[1]:
                i[0] = True
            if not i[-1] and i[-2]:
                i[-1] = True
            ## end of hack
            spline.set_x(self.x[i])
            spline.clear_format_input_functions()
            _cache['i'] = i
        i = _cache['i']
        ## add to y
        self.y[i] *= spline.y  

    def auto_scale_piecewise_sinusoid(
            self,
            xjoin=10, # specified interval join points or distance separating them
            xbeg=None,xend=None, # limit to this range
            vary=False,Avary=False, # vary phase, frequencey and/or amplitude
            Aspline= True,          # amplitude is splined, else pieces wise constant
    ):
        """Automatically find regions for use in
        scale_by_piecewise_sinusoid."""
        ## get join points between regions and begining and ending points
        if np.isscalar(xjoin):
            if xbeg is None:
                xbeg = self.x[0]
            if xend is None:
                xend = self.x[-1]
            i = slice(*np.searchsorted(self.x,[xbeg,xend]))
            xjoin = np.concatenate((arange(self.x[i][0],self.x[i][-1],xjoin),self.x[i][-1:]))
        else:
            if xbeg is None:
                xbeg = xjoin[0]
            else:
                xbeg = max(xbeg,xjoin[0])
            if xend is None:
                xend = xjoin[-1]
            else:
                xend = max(xend,xjoin[-1])
        ## loop over all regions, gettin dominant frequency and phase
        ## from the residual error power spectrum
        regions = []
        for xbegi,xendi in zip(xjoin[:-1],xjoin[1:]):
            i = slice(*np.searchsorted(self.x,[xbegi,xendi]))
            # i = (self.x>=xbegi)&(self.x<=xendi)
            residual = self.yexp[i] - self.y[i]
            FT = fft.fft(residual)
            imax = np.argmax(np.abs(FT)[1:])+1 # exclude 0
            phase = np.arctan(FT.imag[imax]/FT.real[imax])
            if FT.real[imax]<0:
                phase += π
            dx = (self.x[i][-1]-self.x[i][0])/(len(self.x[i])-1)
            frequency = 1/dx*imax/len(FT)
            amplitude = tools.rms(residual)/self.y[i].mean()
            # amplitude = tools.rms(residual)
            regions.append([
                xbegi,xendi,
                P(amplitude,Avary,1e-5),
                P(frequency,vary,frequency*1e-3),
                P(phase,vary,2*π*1e-3),])
        self.scale_by_piecewise_sinusoid(regions,Aspline=Aspline)
        return regions

    @optimise_method()
    def scale_piecewise_sinusoid(self,regions,_cache=None,_parameters=None):
        """Scale by a piecewise function 1+A*sin(2πf(x-xa)+φ) for a set
        regions = [(xa,xb,A,f,φ),...].  Probably should initialise
        with auto_scale_by_piecewise_sinusoid."""
        ## spline interpolated amplitudes
        if (
                self._clean_construct
                or np.any([t._last_modify_value_time > self._last_construct_time
                           for t in _parameters])
        ):
            i = (self.x >= regions[0][0]) & (self.x <= regions[-1][1]) # x-indices defined by regions
            sinusoid = tools.piecewise_sinusoid(self.x[i],regions)
            scale = 1 + sinusoid
            _cache['scale'] = scale
            _cache['i'] = i
        scale = _cache['scale']
        i = _cache['i']
        self.y[i] *= scale

    def auto_add_piecewise_sinusoid(self,xi=10,make_plot=False,vary=False,optimise=False):
        """Fit a spline interpolated sinusoid to current model residual, and
        add it to the model."""
        ## refit intensity_sinusoid
        regions = tools.fit_piecewise_sinusoid(
            self.x,
            self.get_residual(),
            xi=xi,
            make_plot=make_plot,
            make_optimisation=optimise,
        )
        region_parameters = [[xbeg,xend,P(amplitude,vary),P(frequency,vary),P(phase,vary,2*π*1e-5),]
                             for (xbeg,xend,amplitude,frequency,phase) in regions]
        self.add_piecewise_sinusoid(region_parameters)
        return region_parameters

    @optimise_method()
    def add_piecewise_sinusoid(self,regions,_cache=None,_parameters=None):
        """Scale by a piecewise function 1+A*sin(2πf(x-xa)+φ) for a set
        regions = [(xa,xb,A,f,φ),...].  Probably should initialise
        with auto_scale_by_piecewise_sinusoid."""
        if (
                self._clean_construct
                or np.any([t._last_modify_value_time > self._last_construct_time
                           for t in _parameters])
        ):
            ## get spline points and compute splien
            i = (self.x >= regions[0][0]) & (self.x <= regions[-1][1]) # x-indices defined by regions
            sinusoid = tools.piecewise_sinusoid(self.x[i],regions)
            _cache['sinusoid'] = sinusoid
            _cache['i'] = i
        sinusoid = _cache['sinusoid']
        i = _cache['i']
        self.y[i] += sinusoid

    @optimise_method()
    def shift_by_constant(self,shift):
        """Shift by a constant amount."""
        self.y += float(shift)

    @optimise_method()
    def convolve_with_gaussian(self,width=1,fwhms_to_include=20):
        """Convolve with gaussian."""
        ## get x-grid -- skip whole convolution if it does not exist
        if len(self.x) == 0:
            return
        x,y = self.x,self.y
        if width == 0:
            ## nothing to be done
            return
        abswidth = abs(width)
        max_width = 1
        if abswidth > 1e2:
            raise Exception(f'Gaussian width > 100')
        if self.verbose and abswidth < 3*np.diff(self.x).min(): 
            warnings.warn('Convolving gaussian with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
        ## get padded spectrum to minimise edge effects
        dx = (x[-1]-x[0])/(len(x)-1)
        padding = np.arange(dx,fwhms_to_include*abswidth+dx,dx)
        xpad = np.concatenate((x[0]-padding[-1::-1],x,x[-1]+padding))
        ypad = np.concatenate((y[0]*np.ones(padding.shape,dtype=float),y,y[-1]*np.ones(padding.shape,dtype=float)))
        ## generate gaussian to convolve with
        xconv_beg = -fwhms_to_include*abswidth
        xconv_end =  fwhms_to_include*abswidth
            # warnings.warn('Convolution domain length very long.')
        xconv = np.arange(xconv_beg,xconv_end,dx)
        if len(xconv)%2==0: xconv = xconv[0:-1] # easier if there is a zero point
        yconv = np.exp(-(xconv-xconv.mean())**2*4*np.log(2)/abswidth**2) # peak normalised gaussian
        yconv = yconv/yconv.sum() # sum normalised
        ## convolve and return, discarding padding
        self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        ## self.y = signal.fftconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]
        ## self.y = signal.convolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]

    @optimise_method()
    def convolve_with_gaussian_sum(self,widths_areas_offsets,fwhms_to_include=10):
        """Convolve with a sum of Gaussians of different widths and (positive)
        areas in a list [[width1,area1],...] but together normalised to 1."""
        ## maximum width
        width = sum([width for width,area,offset in widths_areas_offsets])
        ## get padded spectrum to minimise edge effects
        dx = (self.x[-1]-self.x[0])/(len(self.x)-1)
        padding = np.arange(dx,fwhms_to_include*width+dx,dx)
        xpad = np.concatenate((self.x[0]-padding[-1::-1],self.x,self.x[-1]+padding))
        ypad = np.concatenate((self.y[0]*np.ones(padding.shape,dtype=float),self.y,self.y[-1]*np.ones(padding.shape,dtype=float)))
        ## generate x grid to convolve with, ensure there is a centre
        ## zero
        xconv_beg = -fwhms_to_include*width
        xconv_end =  fwhms_to_include*width
        xconv = np.arange(xconv_beg,xconv_end,dx)
        if len(xconv)%2==0:
            xconv = xconv[0:-1]
        xconv = xconv-xconv.mean()
        ## normalised sum of gaussians
        yconv = np.full(xconv.shape,0.0)
        for width,area,offset in widths_areas_offsets:
            yconv += lineshapes.gaussian(xconv,float(offset),float(area),abs(width))
        yconv /= yconv.sum() 
        ## convolve padded y and substitute into self
        self.y = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(self.x)]

    def convolve_with_lorentzian(self,width,fwhms_to_include=100):
        """Convolve with lorentzian."""
        p = self.add_parameter_set('convolve_with_lorentzian',width=width,step_default={'width':0.01})
        self.add_format_input_function(lambda: f'{self.name}.convolve_with_lorentzian({p.format_input()})')
        ## check if there is a risk that subsampling will ruin the convolution
        def f():
            x,y = self.x,self.y
            width = np.abs(p['width'])
            if self.verbose and width < 3*np.diff(self.x).min(): warnings.warn('Convolving Lorentzian with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
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
        if self.verbose and p['width'] < 3*np.diff(self.x).min(): warnings.warn('Convolving sinc with width close to sampling frequency. Consider setting a higher interpolate_model_factor.')
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
                    y = signal.oaconvolve(y,lineshapes.sinc(x,Γ=abs(p['sinc_fwhm'])),'same')
                ## convolve with gaussian function
                if p['gaussian_fwhm']!=0:
                    y = signal.oaconvolve(y,lineshapes.gaussian(x,Γ=abs(p['gaussian_fwhm'])),'same')
                ## convolve with lorentzian function
                if p['lorentzian_fwhm']!=0:
                    y = signal.oaconvolve(y,lineshapes.lorentzian(x,Γ=abs(p['lorentzian_fwhm'])),'same')
                ## if necessary account for phase correction by convolving with a signum
                if p['signum_magnitude']!=0:
                    ty = 1/x*p['signum_magnitude'] # hyperbolically decaying signum on either side
                    ty[imidpoint] = 1 # the central part  -- a delta function
                    y = signal.oaconvolve(y,ty,'same')
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
        assert False,'review before use. Fold into set_instrument_function?'
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

    @optimise_method()
    def convolve_with_blackman_harris(
            self,
            terms=3,
            resolution=None,
            fwhms_to_include=10,
            _cache=None,
    ):
        """Convolve with the coefficients equivalent to a Blackman-Harris
        window. Coefficients taken from harris1978 p. 65.  There are
        multiple coefficents given for 3- and 5-Term windows. I use
        the left.  This is faster than apodisation in the
        length-domain.  The resolution is the FWHM of the unapodised
        (boxcar) spectrum."""
        # if self._clean_construct or self._xchanged:
        if self._clean_construct:
            dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
            if resolution is not None:
                pass
            elif self.experiment is not None and 'resolution' in self.experiment.experimental_parameters:
                resolution = self.experiment.experimental_parameters['resolution']
            else:
                raise Exception('Resolution not specified as argument or in experimental data')
            width = resolution/2*1.2 # distance between sinc peak and first zero
            xconv = np.arange(0,fwhms_to_include*width,dx)
            xconv = np.concatenate((-xconv[-1:0:-1],xconv))
            if terms == 3:
                yconv =  0.42323*np.sinc(xconv/width)
                yconv += 0.5*0.49755*np.sinc(xconv/width-1) 
                yconv += 0.5*0.49755*np.sinc(xconv/width+1)
                yconv += 0.5*0.07922*np.sinc(xconv/width-2)
                yconv += 0.5*0.07922*np.sinc(xconv/width+2)
            elif terms == 4:
                yconv =  0.35875*np.sinc(xconv/width)
                yconv += 0.5*0.48829*np.sinc(xconv/width-1) 
                yconv += 0.5*0.48829*np.sinc(xconv/width+1)
                yconv += 0.5*0.14128*np.sinc(xconv/width-2)
                yconv += 0.5*0.14128*np.sinc(xconv/width+2)
                yconv += 0.5*0.01168*np.sinc(xconv/width-3)
                yconv += 0.5*0.01168*np.sinc(xconv/width+3)
            else: 
                raise Exception("Only 3 and 4 terms version implemented.")
            yconv /= yconv.sum()                    # normalised
            _cache['xconv'] = xconv
            _cache['yconv'] = yconv
        else:
            xconv = _cache['xconv']
            yconv = _cache['yconv']
        ## convolve
        self.y = tools.convolve_with_padding(self.x,self.y,xconv,yconv)

    @optimise_method()
    def convolve_with_signum(
            self,
            amplitude,          # amplitude of signum
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
        yconv = np.empty(xconv.shape,dtype=float)
        inonzero = np.abs(xconv) != 0
        yconv[inonzero] = amplitude/xconv[inonzero]/len(xconv)               # signum function
        yconv[int((len(yconv)-1)/2)] = 1.       # add δ function at center
        yconv /= yconv.sum()                    # normalised
        self.y[i] = signal.oaconvolve(ypad,yconv,mode='same')[len(padding):len(padding)+len(x)]

    @optimise_method()
    def convolve_with_soleil_instrument_function(
            self,
            sinc_fwhm=None,
            gaussian_fwhm=0.1,
            lorentzian_fwhm=None,
            signum_magnitude=None,
            sinc_fwhms_to_include=200,
            gaussian_fwhms_to_include=10,
            lorentzian_fwhms_to_include=10,
            _cache=None,
    ):
        """Convolve with soleil instrument function."""
        ## first run only
        # if len(_cache) == 0:
        ## get automatically set values if not given explicitly
        if sinc_fwhm is None:
            sinc_fwhm = self.experiment.experimental_parameters['sinc_FWHM']
        if abs(self.experiment.experimental_parameters['sinc_FWHM']-sinc_fwhm)>(1e-5*sinc_fwhm):
            warnings.warn(f'sinc FWHM {float(sinc_fwhm)} does not match soleil data file header {float(self.experiment.experimental_parameters["sinc_FWHM"])}')
        # if gaussian_fwhm is None:
            # gaussian_fwhm = 0.1
        ## compute instrument function
        dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
        _cache['dx'] = dx
        ## get total extent of instrument function
        width = 0.0
        if sinc_fwhm is not None:
            width += abs(sinc_fwhm)*sinc_fwhms_to_include
        if gaussian_fwhm is not None:
            width += abs(gaussian_fwhm)*gaussian_fwhms_to_include
        if lorentzian_fwhm is not None:
            width += abs(lorentzian_fwhm)*lorentzian_fwhms_to_include
        _cache['width'] = width
        ## if necessary compute instrument function on a reduced
        ## subsampling to ensure accurate convolutions -- this wont
        ## help with an accurate convolution against the actual
        ## data!
        required_points_per_fwhm = 10
        subsample_factor = int(np.ceil(max(
            1,
            (required_points_per_fwhm*dx/sinc_fwhm if sinc_fwhm is not None else 1),
            (required_points_per_fwhm*dx/gaussian_fwhm if gaussian_fwhm is not None else 1),
            )))
        ## create the instrument function on a regular grid -- odd length with 0 in the middle
        x = np.arange(0,width+dx/subsample_factor*0.5,dx/subsample_factor,dtype=float)
        x = np.concatenate((-x[-1:0:-1],x))
        imidpoint = int((len(x)-1)/2)
        ## initial function is a delta function
        y = np.full(x.shape,0.)
        y[imidpoint] = 1.
        ## convolve with sinc function
        if sinc_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.sinc(x,Γ=abs(sinc_fwhm)),'same')
        ## convolve with gaussian function
        if gaussian_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.gaussian(x,Γ=abs(gaussian_fwhm)),'same')
        ## convolve with lorentzian function
        if lorentzian_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.lorentzian(x,Γ=abs(lorentzian_fwhm)),'same')
        ## if necessary account for phase correction by convolving with a signum
        if signum_magnitude is not None:
            x[imidpoint] = 1e-99  # avoid divide by zero warning
            ty = 1/x*signum_magnitude # hyperbolically decaying signum on either side
            ty[imidpoint] = 1 # the central part  -- a delta function
            y = signal.oaconvolve(y,ty,'same')
        ## convert back to data grid if it has been subsampled
        if subsample_factor!=1:
            a = y[imidpoint-subsample_factor:0:-subsample_factor][::-1]
            b = y[imidpoint::subsample_factor]
            b = b[:len(a)+1] 
            y = np.concatenate((a,b))
        ## normalise 
        y = y/y.sum()
        _cache['y'] = y
        ## convolve model with instrument function
        padding = np.arange(dx,_cache['width']+dx, dx)
        xpad = np.concatenate((self.x[0]-padding[-1::-1],self.x,self.x[-1]+padding))
        ypad = np.concatenate((np.full(padding.shape,self.y[0]),self.y,np.full(padding.shape,self.y[-1])))
        self.y = signal.oaconvolve(ypad,_cache['y'],mode='same')[len(padding):len(padding)+len(self.x)]

    @optimise_method()
    def convolve_with_instrument_function(
            self,
            sinc_fwhm=None,
            gaussian_fwhm=None,
            lorentzian_fwhm=None,
            signum_magnitude=None,
            width=None,
            _cache=None,
    ):
        """Convolve with soleil instrument function."""
        ## get automatically set values if not given explicitly
        if 'sinc_FWHM' in self.experiment.experimental_parameters:
            if sinc_fwhm is None:
                sinc_fwhm = self.experiment.experimental_parameters['sinc_FWHM']
            else:
                if abs(self.experiment.experimental_parameters['sinc_FWHM']-sinc_fwhm)>(1e-5*sinc_fwhm):
                    warnings.warn(f'sinc FWHM {float(sinc_fwhm)} does not match soleil data file header {float(self.experiment.experimental_parameters["sinc_FWHM"])}')
        ## compute instrument function
        dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
        ## instrument function grid
        if width is None:
            width = (self.x[-1]-self.x[0])/2
        x = np.arange(0,width+dx*0.5,dx,dtype=float)
        x = np.concatenate((-x[-1:0:-1],x))
        imidpoint = int((len(x)-1)/2)
        ## initial function is a delta function
        y = np.full(x.shape,0.)
        y[imidpoint] = 1.
        ## convolve with sinc function
        if sinc_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.sinc(x,Γ=abs(sinc_fwhm)),'same')
        ## convolve with gaussian function
        if gaussian_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.gaussian(x,Γ=abs(gaussian_fwhm)),'same')
        ## convolve with lorentzian function
        if lorentzian_fwhm is not None:
            y = signal.oaconvolve(y,lineshapes.lorentzian(x,Γ=abs(lorentzian_fwhm)),'same')
        ## if necessary account for phase correction by convolving with a signum
        if signum_magnitude is not None:
            x[imidpoint] = 1e-99  # avoid divide by zero warning
            ty = 1/x*signum_magnitude # hyperbolically decaying signum on either side
            ty[imidpoint] = 1 # the central part  -- a delta function
            y = signal.oaconvolve(y,ty,'same')
        ## normalise 
        y = y/y.sum()
        ## convolve model with instrument function
        self.y = tools.convolve_with_padding(self.x,self.y,x,y)

    @optimise_method(format_multi_line=99)
    def set_soleil_sidebands(self,yscale=0.1,shift=1000,signum_magnitude=None,_parameters=None,_cache=None):
        """Adds two copies of spectrum onto itself shifted rightwards and
        leftwards by shift (cm-1) and scaled by ±yscale."""
        ## load and cache spectrum
        if self._clean_construct:
            x,y,header = load_soleil_spectrum_from_file(self.experiment.experimental_parameters['filename'])
            _cache['x'],_cache['y'] = x,y
        x,y = _cache['x'],_cache['y']
        ## get signum convolution kernel if it is the first run or one
        ## of the input parameters has changed
        if (self._clean_construct
            or 'sidebands' not in _cache
            or np.any([_cache[id(p)]!=float(p) for p in _parameters])):
            for p in _parameters:
                _cache[id(p)] = float(p)
            sidebands = np.full(self.x.shape,0.0)
            ## shift from left
            i = ((x+shift) >= self.x[0]) & ((x+shift) <= self.x[-1])
            xi,yi = x[i]+shift,y[i]
            j = (self.x>=xi[0]) & (self.x<=xi[-1])
            sidebands[j] = yscale*tools.spline(xi,yi,self.x[j])
            ## shift from right
            i = ((x-shift) >= self.x[0]) & ((x-shift) <= self.x[-1])
            xi,yi = x[i]-shift,y[i]
            j = (self.x>=xi[0]) & (self.x<=xi[-1])
            sidebands[j] -= yscale*tools.spline(xi,yi,self.x[j])
            ## convolve with phase error
            if signum_magnitude is not None:
                dx = (x[-1]-x[0])/(len(x)-1)
                nconv = int(10/dx)
                xconv = np.linspace(-nconv,nconv,2*nconv+1)
                yconv = np.full(len(xconv),1.0)
                i = xconv!=0
                yconv[i] = 1/xconv[i]*signum_magnitude
                sidebands = signal.oaconvolve(sidebands,yconv,'same')
                _cache['sidebands'] = sidebands
        ## add sidebands
        sidebands = _cache['sidebands']
        self.y += sidebands
        
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
            label_match=None,   # label only matching things
            label_key=None,     # this key is used to label series
            label_zkeys=None,   # divide series by these keys
            minimum_τ_to_label=None, # for absorption lines
            minimum_I_to_label=None, # for emission lines
            minimum_Sij_to_label=None, # for emission lines
            plot_title=False,
            title=None,
            plot_legend= True,
            # contaminants_to_plot=('default',), # what contaminant to label
            plot_contaminants=False,
            contaminants=None,
            shift_residual=0.,
            xlabel=None,ylabel=None,
            invert_model=False,
            plot_kwargs=None,
            xticks=None,
            yticks=None,
            plot_text=True,
    ):
        """Plot experimental and model spectra."""
        if not plot_text:
            plot_labels=False
            plot_title=False
            plot_legend=False
            xlabel=''
            ylabel=''
            xticks=[]
            yticks=[]
        ## faster plotting
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
            xmin,xmax = min(xmin,self.x.min()),max(xmax,self.x.max())
            tkwargs = dict(color=plotting.newcolor(0), label=f'Experimental spectrum: {self.experiment.name}', **plot_kwargs)
            ax.plot(self.xexp,self.yexp,**tkwargs)
        if plot_model and self.y is not None:
            if invert_model:
                self.y *= -1
            ymin,ymax = min(ymin,self.y.min(),-0.1*self.y.max()),max(ymax,self.y.max()*1.1)
            xmin,xmax = min(xmin,self.x.min()),max(xmax,self.x.max())
            tkwargs = dict(color=plotting.newcolor(1), label=f'Model spectrum: {self.name}', **plot_kwargs)
            ax.plot(self.x,self.y,**tkwargs)
            if invert_model:
                self.y *= -1
        if plot_residual and self.y is not None and self.experiment is not None and self.experiment.y is not None:
            if self._compute_residual is not None:
                yres = self.yexp - self.y
                if self.residual_weighting is not None:
                    yres *= self.residual_weighting
                ymin,ymax = min(ymin,yres.min()+shift_residual),max(ymax,yres.max()+shift_residual)
                xmin,xmax = min(xmin,self.x.min()),max(xmax,self.x.max())
                tkwargs = dict(color=plotting.newcolor(2), label='Experiment-Model residual', **plot_kwargs)
                ax.plot(self.x,yres+shift_residual,zorder=-1,**tkwargs) # plot fit residual
        ## annotate rotational series
        if plot_labels:
            ystep = ymax/20.
            for line in self.suboptimisers:
                if not isinstance(line,lines.Generic):
                    continue
                ## limit to qn -- if fail to match to label_match then
                ## do not label at all
                if label_match is None:
                    label_match = {}
                try:
                    i = line.match(**label_match)
                except exceptions.InferException:
                    # warnings.warn(f'Not labelling because InferException on {label_match=}')
                    continue
                ## limit to ν-range and sufficiently strong line
                i &= (line['ν']>self.x[0])&(line['ν']<self.x[-1])
                if minimum_τ_to_label is not None:
                    i &= line['τ']>minimum_τ_to_label
                if minimum_I_to_label is not None:
                    i &= line['I']>minimum_I_to_label
                if minimum_Sij_to_label is not None:
                    i &= line['Sij']>minimum_Sij_to_label
                line = line[i]
                if len(line)==0:
                    continue
                ## get labelling keys
                zkeys = (label_zkeys if label_zkeys is not None else line.default_zkeys)
                ## plot annotations
                branch_annotations = plotting.annotate_spectrum_by_branch(
                    line,
                    ymax+ystep/2.,
                    ystep,
                    zkeys=zkeys,  
                    length=-0.02, # fraction of axes coords
                    # color_by=('branch' if 'branch' in zkeys else zkeys),
                    labelsize='xx-small',namesize='x-small', namepos='float',    
                    label_key=(label_key if label_key is not None else line.default_xkey),
                )
                ymax += ystep*(len(branch_annotations)+1)
        ## plot branch heads
        if plot_branch_heads:
            for transition in self.transitions:
                annotate_branch_heads(transition,qn_defining_branch,match_branch_re=label_match_name_re)
        ## plot contaminants
        if plot_contaminants:
            if contaminants is None:
                ## set default list of contaminants
                contaminants = (
                    ('¹H₂',{'v_l':0}),
                    ('Xe',{}),
                    ('Ar',{}),
                    ('Kr',{}),
                    # ('H',{}),
                    ('O',{}),
                    ('N',{}),
                )
            icontaminant = 0
            for tspecies,tmatch in contaminants:
                l = database.get_lines(tspecies)
                tmatch |= {'min_ν':self.x.min(),'max_ν':self.x.max()}
                l = l.matches(**tmatch)
                if len(l) > 0:
                    ax.plot(l['ν'],np.full(len(l),ax.get_ylim()[0]/2),
                            label=tspecies,color=plotting.newcolor(icontaminant+3),
                            ls='',marker='o',markersize=6,mfc='none')
                    icontaminant += 1
        ## finalise plot
        if plot_title:
            if title is None:
                title = self.name
            t = ax.set_title(title,fontsize='x-small')
            t.set_in_layout(False)
        if plot_legend:
            tools.legend_colored_text(loc='upper left')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.grid(True,color='gray')
        plotting.simple_tick_labels(ax=ax)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.xaxis.set_ticks(xticks)
        if yticks is not None:
            ax.yaxis.set_ticks(yticks)
        self._figure = fig
        return fig 

    def output_data_to_directory(
            self,
            directory,
            output_model_residual=False,
            output_transition_linelists=False,
            output_individual_optical_depths=False,
    ):
        """Save various files from this optimsiation to a directory."""
        tools.mkdir(directory)
        ## model data
        if self.x is not None and self.y is not None:
            t = Spectrum(ν=self.x,I=self.y,description=f'Model spectrum of {self.name}')
            t.save(f'{directory}/spectrum',filetype='directory')
        if self._figure is not None:
            ## svg / pdf are the fastest output formats. Significantly
            ## faster if there is not text on the figure
            # self._figure.savefig(directory+'/figure.png',dpi=300)
            self._figure.savefig(directory+'/figure.pdf')
        if output_transition_linelists:
            tools.mkdir_if_necessary(directory+'/transitions')
            for transition in self.absorption_transitions:
                transition.save_to_file(directory+'/transitions/'+transition.name+'.h5')

def load_soleil_spectrum_from_file(filename,remove_HeNe=False):
    """ Load soleil spectrum from file with given path."""
    ## resolve soleil filename
    if os.path.exists(tools.expand_path(filename)):
        ## filename is an actual path to a file
        filename = tools.expand_path(filename)
    elif os.path.exists(f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.hdf5'):
        ## filename is a scan base name in default data directory
        filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.hdf5'
    elif os.path.exists(f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.h5'):
        ## filename is a scan base name in default data directory
        filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.h5'
    else:
        ## else look for unique prefix in scan database
        # t = tools.sheet_to_dict('/home/heays/exp/SOLEIL/summary_of_scans.rs',comment='#')
        t = dataset.load('/home/heays/exp/SOLEIL/summary_of_scans.rs')
        i = tools.find_regexp(r'^'+re.escape(filename)+'.*',t['filename'])
        if len(i)==1:
            filename = t['filename'][int(i)]
            filename = f'/home/heays/exp/SOLEIL/scans/{filename}.wavenumbers.h5'
        else:
            raise Exception(f"Could not find SOLEIL spectrum: {repr(filename)}")
    extension = os.path.splitext(filename)[1]
    ## get header data if possible, not possible if an hdf5 file is used.
    header = dict(filename=filename,header=[])
    if extension in ('.TXT','.wavenumbers'): 
        with open(filename,'r',encoding='latin-1') as fid:
            header['header'] = []
            while True:
                line = fid.readline()[:-1]
                if re.match(r'^ *[0-9.eE+-]+[, ]+[0-9.eE+-]+ *$',line): break # end of header
                header['header'].append(line) # save all lines to 'header'
                ## post-processing zero-adding leads to an
                ## interpolation of data by this factor
                r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
                if r:
                    header['interpolation_factor'] = float(r.group(1))
                ## the resolution before any interpolation
                r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
                if r: header['ds'] = float(r.group(1))
                ## NMAX parameter indicates that the spectrometer is
                ## being run at maximum resolution. This is not an
                ## even power of two. Then the spectrum is zero padded
                ## to have 2**21 points. This means that there is an
                ## additional interpolation factor of 2**21/NMAX. This
                ## will likely be non-integer.
                r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
                if r:
                    header['interpolation_factor'] *= 2**21/float(r.group(1))
                ## extract pressure from header
                r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
                if r:
                    header['date_time'] = r.group(1)
                    header['pressure_pirani'] = float(r.group(2))
                    header['pressure_baratron'] = float(r.group(3))
            header['header'] = '\n'.join(header['header'])
            ## compute instrumental resolution, FWHM
    elif extension in ('.hdf5','.h5'): # expect header stored in 'README'
        data = tools.hdf5_to_dict(filename)
        header['header'] = data['README']
        for line in header['header'].split('\n'):
            ## post-processing zero-adding leads to an
            ## interpolation of data by this factor
            r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
            if r:
                header['interpolation_factor'] = float(r.group(1))
            ## the resolution before any interpolation
            r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
            if r: header['ds'] = float(r.group(1))
            ## NMAX parameter -- see above
            r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
            if r:
                header['interpolation_factor'] *= 2**21/float(r.group(1))
            ## extract pressure from header
            r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
            if r:
                header['date_time'] = r.group(1)
                header['pressure_pirani'] = float(r.group(2))
                header['pressure_baratron'] = float(r.group(3))
    else:
        raise Exception(f"bad extension: {repr(extension)}")
    ## compute instrumental resolution, FWHM
    header['sinc_FWHM'] = 1.2*header['interpolation_factor']*header['ds'] 
    ## get spectrum
    if extension=='.TXT':
        x,y = [],[]
        data_started = False
        for line in tools.file_to_lines(filename,encoding='latin-1'):
            r = re.match(r'^([0-9]+),([0-9.eE+-]+)$',line) # data point line
            if r:
                data_started = True # header is passed
                x.append(float(r.group(1))),y.append(float(r.group(2))) # data point
            else:
                if data_started: break            # end of data
                else: continue                    # skip header line
        x,y = np.array(x)*header['ds'],np.array(y)
    elif extension=='.wavenumbers':
        x,y = tools.file_to_array(filename,unpack=True,comments='#',encoding='latin-1')
    elif extension in ('.hdf5','.h5'):
        data = tools.hdf5_to_dict(filename)
        x,y = data['data'].transpose()
    else:
        raise Exception(f"bad extension: {repr(extension)}")
    ## process a bit. Sort and remove HeNe line profile and jitter
    ## estimate. This is done assumign the spectrum comes
    ## first. and finding the first index were the wavenumber
    ## scale takes a backward step
    if remove_HeNe:
        i = x>31600
        x,y = x[i],y[i]
    t = tools.find(np.diff(x)<0)
    if len(t)>0:
        i = t[0]+1 
        x,y = x[:i],y[:i]
    ## get x range
    header['xmin'],header['xmax'] = x.min(),x.max()
    header['xcentre'] = 0.5*(header['xmin']+header['xmax'])
    return (x,y,header)

class Spline(Optimiser):
    """A spline curve with optimisable knots.  Internally stores last
    calculated x and y."""

    def __init__(self,name='spline',knots=None,order=3):
        optimise.Optimiser.__init__(self,name)
        self.clear_format_input_functions()
        self.automatic_format_input_function()
        self.order = order
        ## spline knots
        self.xs = []
        self.ys = []
        self.set_knots(knots)
        ## spline domain and value
        self.x = None
        self.y = None

    @optimise_method()
    def add_knot(self,x,y):
        """Add one spline knot."""
        self.xs.append(x)
        self.ys.append(y)

    @optimise_method()
    def set_knots(self,knots):
        """Add multiple spline knots."""
        self.xs.clear()
        self.ys.clear()
        for x,y in knots:
            self.xs.append(x)
            self.ys.append(y)

    @optimise_method()
    def set_x(self,x):
        """Compute spline at x. Store internally."""
        xs = np.array([float(t) for t in self.xs])
        ys = np.array([float(t) for t in self.ys])
        self.x = x
        self.y = tools.spline(xs,ys,x,order=self.order)

    def __getitem__(self,x):
        self.set_x(x)
        return self.y

def load_spectrum(filename,**kwargs):
    """Use a heuristic method to load a directory output by
    Spectrum."""

    x = Spectrum()
    x.load_from_directory(filename,**kwargs)
    return(x)

class Spectrum(Dataset):

    default_xkey = 'x'
    default_zkeys = ()
    default_prototypes = {
        'x':{'description':'x-scale'          ,  'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'ν':{'description':'Wavenumber scale' , 'units':'cm-1' , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'λ':{'description':'Wavelength scale' , 'units':'nm'   , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'f':{'description':'Frequency scale'  , 'units':'MHz'  , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'σ':{'description':'Cross section'    , 'units':'cm2'  , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'I':{'description':'Intensity'        ,  'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
    }


class FitReferenceAbsorption():

    def __init__(
            self,
            name='fit_reference_absorption',
            p=None,
            verbose=False,
            make_plot=True,
            max_nfev=5,
            figure_number=1,
            ncpus=1,
            **parameters
    ):
        self.name = name
        ## provided parameters
        if p is None:
            p = {}
        self.p = p
        ## update with kwrags
        self.p |= parameters
        ## set defaults
        # for key,val in {
                # 'xbeg':600,
                # 'xend':6000,
                # 'interpolate_model':0.001,
                # # 'min_S296K' :1e-23,
                # # 'noise':{'xbeg':5951.26,'xend':5959.4},
        # }.items():
            # self.p.setdefault(key,val)
        # ## prepare other variables
        self.verbose = verbose
        self.make_plot =  make_plot
        self.figure_number = figure_number
        self.ncpus = ncpus
        self.experiment = None

    characteristic_bands = {
        'H2O':[[1400,1750],[3800,4000],],
        'CO2':[[2200,2400], [3550,3750], [4800,5150],] ,
        'CO':[[1980,2280], [4150,4350],],
        'C2H2':[[600,850], [3200,3400],],
        'HCN':[[600,850],[3200,3400],],
        'NH3':[[800,1200],],
        'SO2':[[1050,1400], [2450,2550],],
        'NO2':[[1540,1660],],
        'NO':[[1750,2000],],
        'N2O':[[2175,2270],],
        'CH4':[[1200,1400],[2800,3200],],
        'CS2':[[1500,1560],],
        'OCS':[[2000,2100],],
        'CH3Cl':[[2750,3500],],
        'CS':[[1200,1350],],
        'H2S':[[1000,1600],[3500,4100]],
        'C2H6':[[2850,3100]],
        'HCOOH':[[1000,1200],[1690,1850]],
    }

    characteristic_lines = {
        'H2O':[[1550,1580],],
        # 'H2O':[[1550,1580],[3880,3882]],
        'CO2':[[2310,2330],],
        'CO':[[2160,2180],],
        # 'CO':[[2160,2180],[4285,4292]],
        ## 'NH3':[[920,980],],
        'NH3':[[950,970],],
        'HCN':[[710,730],],
        ## 'SO2':[[1350,1370],[1100,1150],],
        'SO2':[[1350,1370],],
        'NO2':[[1590,1610],],
        'NO':[[1830,1850],],
        'N2O':[[2199,2210],],
        ##'CH4':[[3000,3030],],
        'CH4':[[3010,3020],],
        ## 'H2S':[[1250,1300],[3700,3750],],
        'H2S':[[1275,1300],],
        'CS2':[[1530,1550],],
        'C2H2':[[3255,3260],],
        'OCS':[[2070,2080],],
        'C2H6':[[2975,2995]],
        'HCOOH':[[1770,1780]],
    }

    def __str__(self):
        retval = tools.dict_expanded_repr(self.p,newline_depth=3)
        return retval

    def verbose_print(self,*args,**kwargs):
        if self.verbose:
            print(*args,**kwargs)

    def load_experiment(self):
        print('loading experiment')
        p = self.p
        ## load exp data
        self.experiment = Experiment('experiment')
        self.experiment.set_spectrum_from_opus_file(p['filename'])
        ## scale x coordinate
        if 'scalex' in p:
            p['scalex'].vary = False
            self.experiment.scalex(p['scalex'])
        ## fit noise level
        self.p.setdefault('noise',{})
        self.p['noise'].setdefault('xbeg',self.experiment.x[-1]-20)
        self.p['noise'].setdefault('xend',self.experiment.x[-1])
        self.p['noise'].setdefault('rms', self.experiment.fit_noise(xbeg=self.p['noise']['xbeg'],xend=self.p['noise']['xend'],n=3,make_plot=False))
        self.experiment.experimental_parameters['noise_rms'] = p['noise']['rms']
        self.experiment.residual_scale_factor = 1/p['noise']['rms']
        ## bin data for speed
        if 'bin_experiment' in p:
            self.experiment.bin_data(width=p['bin_experiment'],mean=True)

    def auto_intensity(self,xi=5):
        """Automatic background. Not optimised."""
        print('auto_intensity_spline')
        model = self.make_model(xbeg=self.p['xbeg'],xend=self.p['xend'])
        model.construct()
        self.p['intensity_spline'] = model.auto_multiply_spline(x=xi,construct=False,vary=False,)

    def auto_sinusoid(self,xi=10):
        """Automatic background. Not optimised."""
        print('auto_intensity_sinusoid')
        ## refit intensity_sinusoid
        if 'intensity_sinusoid' in self.p:
            self.p.pop('intensity_sinusoid')
        model = self.make_model(xbeg=self.p['xbeg'],xend=self.p['xend'])
        model.construct()
        self.p['intensity_sinusoid'] = model.auto_add_piecewise_sinusoid(xi=xi,make_plot=False,vary=False)

    def full_model(
            self,
            species_to_fit=None,
            xbeg=None,xend=None,
            max_nfev=20,
            **make_model_kwargs
    ):
        """Full model."""
        print('full_model')
        if xbeg is None:
            xbeg = self.p['xbeg']
        if xend is None:
            xend = self.p['xend']
        model = self.make_model(species_to_fit=species_to_fit,xbeg=xbeg,xend=xend,**make_model_kwargs)
        model.optimise(make_plot=self.make_plot,monitor_frequency='rms decrease',verbose=self.verbose,max_nfev=max_nfev)
        model_no_absorption = self.make_model(xbeg,xend,list(self.p['N']),neglect_species_to_fit=True)
        self.plot(model,model_no_absorption)
        return model,model_no_absorption

    def fit_region(self,xbeg,xend,max_nfev=5,**make_model_kwargs):
        """Full model."""
        print('fit_region',xbeg,xend)
        model = self.make_model(xbeg,xend,**make_model_kwargs)
        model.name = f'fit_region_{int(xbeg)}_{int(xend)}'
        model.optimise(make_plot=self.make_plot,verbose=self.verbose,max_nfev=max_nfev)
        self.plot(model)

    def fit_regions(self,width=100,overlap=0.9,max_nfev=5,**make_model_kwargs):
        """Full model."""
        print('fit_regions')
        p = self.p
        xbeg = p['xbeg']
        while xbeg < p['xend']:
            xend = min(xbeg+width,p['xend'])
            model = self.make_model(xbeg,xend,**make_model_kwargs)
            model.optimise(make_plot=self.make_plot,verbose=self.verbose,max_nfev=max_nfev)
            self.plot(model)
            xbeg += overlap*width

    def fit_species(
            self,
            species_to_fit=None,
            regions='lines',
            fit_species_individually=True,
            max_nfev=20,
            **make_model_kwargs,
    ):
        """Fit species_to_fit individually using their 'lines' or 'bands'
        preset regions, or the 'full' region."""
        print('fit_species',regions,end=" ")
        p = self.p
        ## get species list
        if len(species_to_fit) == 0:
            species_to_fit = list(p['N'])
        species_to_fit = tools.ensure_iterable(species_to_fit)
        ## fit species individually to all regions or fit all species at once
        first_plot = True
        if fit_species_individually:
            for species in species_to_fit:
                print(species,end=" ",flush=True)
                ## get region list
                if regions == 'lines':
                    regions_to_fit = self.characteristic_lines[species]
                elif regions == 'bands':
                    regions_to_fit = self.characteristic_bands[species]
                elif regions == 'full':
                    regions_to_fit = [[p['xbeg'],p['end']]]
                else:
                    assert False
                main = Optimiser(name=species)
                models = []
                for (xbeg,xend) in regions_to_fit:
                    ## get xbeg,xend in scan ragne
                    xbeg = max(xbeg,p['xbeg'])
                    xend = min(xend,p['xend'])
                    if xend - xbeg <= 0:
                        continue
                    ## make optimise model
                    model = self.make_model(xbeg,xend,[species],**make_model_kwargs)
                    main.add_suboptimiser(model)
                    model_no_absorption = self.make_model(xbeg,xend,[species],neglect_species_to_fit=True)
                    models.append((model,model_no_absorption))
                if len(models) == 0:
                    raise Exception(f'Cannot fit species {species}, no regions defined in range {xbeg} to {xend}')
                ## optimise plot indiviudal speciesmodels
                residual = main.optimise(make_plot=self.make_plot,max_nfev=max_nfev,verbose=self.verbose)
                for model,model_no_absorption in models:
                    self.plot(model,model_no_absorption,clf=first_plot)
                    first_plot = False
            print()
        else:
            ## get region list
            if regions == 'lines':
                regions_to_fit = []
                for species in species_to_fit:
                    regions_to_fit.extend(self.characteristic_lines[species])
            elif regions == 'bands':
                regions_to_fit = []
                for species in species_to_fit:
                    regions_to_fit.extend(self.characteristic_bands[species])
            elif regions == 'full':
                regions_to_fit = [[p['xbeg'],p['end']]]
            else:
                assert False
            self.verbose_print(f'fit_species: {species:10} ',end='')
            main = Optimiser(name='_'.join(species_to_fit))
            models = []
            for (xbeg,xend) in regions_to_fit:
                ## get xbeg,xend in scan ragne
                xbeg = max(xbeg,p['xbeg'])
                xend = min(xend,p['xend'])
                if xend - xbeg <= 0:
                    continue
                ## make optimise model
                model = self.make_model(xbeg,xend,species_to_fit,**make_model_kwargs)
                main.add_suboptimiser(model)
                model_no_absorption = self.make_model(xbeg,xend,species_to_fit,neglect_species_to_fit=True)
                models.append((model,model_no_absorption))
            ## optimise plot indiviudal species models
            residual = main.optimise(make_plot=self.make_plot,max_nfev=max_nfev,verbose=self.verbose)
            for model,model_no_absorption in models:
                self.plot(model,model_no_absorption,clf=first_plot)
                first_plot = False

    def auto_fit(
            self,
            species_to_fit=None,
            prefit=True,
            cycles=3,
            fit_intensity=True,
            fit_sinusoid=True,
            regions='bands',
            fit_full_model=False,
            reference_species='H2O',
            fit_instrument=True,
            fit_scalex=False,
            fit_temperature=False,
            max_nfev=5,
    ):
        """Fit spectrum in a standardised way."""
        print('auto_fit')
        time = timestamp()
        if prefit:
            ## first fit background and species roughly, replace
            ## existing fit data
            for key in ('pair','N','instrument_gaussian'):
                if key in self.p:
                    self.p.pop(key)
            if fit_sinusoid and 'intensity_sinusoid' in self.p:
                self.p.pop('intensity_sinusoid')
            if fit_intensity and 'intensity_spline' in self.p:
                self.p.pop('intensity_spline')
            if fit_intensity:
                self.auto_intensity()
            if fit_instrument or fit_temperature:
                self.fit_species(
                    reference_species,
                    regions='lines',
                    fit_N=True,
                    fit_pair=True,
                    fit_instrument=fit_instrument,
                    fit_intensity=fit_intensity,
                    fit_temperature=fit_temperature,
                    max_nfev=max_nfev)
            self.fit_species(
                species_to_fit,
                regions='lines',
                fit_N=True,
                fit_pair=True,
                fit_intensity=fit_intensity,
                max_nfev=max_nfev,)
        ## then cycle on careful fit
        for n in range(cycles):
            ## background
            if fit_sinusoid:
                self.auto_sinusoid()
            ## # if fit_intensity:
            ##     # self.auto_intensity()
            if fit_sinusoid or fit_intensity:
                self.fit_regions(
                    fit_intensity=fit_intensity,
                    fit_sinusoid=fit_sinusoid,
                    max_nfev=max_nfev,)
            ## instrument function etc
            if fit_instrument or fit_scalex or fit_temperature:
                self.fit_species(
                    reference_species,
                    regions=regions,
                    fit_N=True,
                    fit_pair=True,
                    fit_scalex=fit_scalex,
                    fit_instrument=fit_instrument,
                    fit_intensity=fit_intensity,
                    fit_sinusoid=fit_sinusoid,
                    fit_temperature=fit_temperature,
                    max_nfev=max_nfev,
                )
            ## all species
            self.fit_species(
                species_to_fit,
                regions=regions,
                fit_N=True,
                fit_pair=True,
                fit_intensity=fit_intensity,
                fit_sinusoid=fit_sinusoid,
                max_nfev=max_nfev,
            )
        ## make final model
        if fit_full_model:
            self.full_model(
                species_to_fit,
                fit_N=True,
                fit_pair=True,
                max_nfev=max_nfev,
            )
        print('Time elapsed:',format(timestamp()-time,'12.6f'))

    def plot(
            self,
            model,
            model_no_absorption=None,
            clf=True,
            **plot_kwargs
    ):
        """Plot the results of some models on individual subplots. Residuals
        from models_no_absorption will be underplotted."""
        if self.figure_number is None:
            return
        if clf:
            fig = plotting.qfig(self.figure_number,show=True)
        else:
            fig = plotting.plt.figure(self.figure_number)
        plot_kwargs = dict(
            plot_legend=False,
            plot_title= True,
            plot_text= True,
            ) | plot_kwargs
        ax = plotting.subplot(fig=fig)
        model.plot(ax=ax,**plot_kwargs)
        if model_no_absorption is not None:
            ax.plot(
                model_no_absorption.x,
                model_no_absorption.get_residual(),
                # model.y-model_no_absorption.y,
                color='orange',
                # label=f'model_no_absorption: full/noabs={tools.rms(model.get_residual()):0.3e}/{tools.rms(model_no_absorption.get_residual()):0.3e}',
                zorder=-2)
            # ax.legend()
        fig.suptitle(self.name)
        ax.axhline(0,color='gray',zorder=-5)
        plotting.qupdate(fig)

    def make_model(
            self,
            xbeg,xend,
            species_to_fit=None,
            species_to_model=None,
            fit_intensity=False,
            fit_N=False,
            fit_pair=False,
            fit_scalex=False,
            fit_sinusoid=False,
            fit_instrument=False,
            fit_temperature=False,
            neglect_species_to_fit=False,
            verbose=False,
    ):
        if species_to_fit is None:
            species_to_fit = []
        ## get parameters — protect from if necessary
        p = self.p
        if neglect_species_to_fit:
            p = deepcopy(p)
        ## ensure experiment loaded
        if self.experiment is None:
            self.load_experiment()
        ## fit experiment frequency scale
        if 'scalex' not in p:
            p['scalex'] = P(1,False,1e-9)
        p['scalex'].vary = fit_scalex
        ## start model
        model = Model(
            name='_'.join(['make_model',*species_to_fit]),
            experiment=self.experiment,
            xbeg=xbeg,xend=xend)
        model.permit_construct_on_add =  True
        ## set interpolated model grid
        self.p.setdefault('interpolate_model',0.001)
        if p['interpolate_model'] is not None:
            model.interpolate(p['interpolate_model'])
        ## add unit intensity
        model.add_intensity(1)
        ## add absorption lines
        p.setdefault('Teq',P(296,False,1,nan,(285,305)))
        p['Teq'].vary = fit_temperature
        p.setdefault('N',{})
        p.setdefault('pair',{})
        if species_to_model is None:
            species_to_model = set(p['N'])
        species_to_model = set(species_to_model) | set(species_to_fit)
        for species in species_to_model:
            if neglect_species_to_fit and species in species_to_fit:
                continue
            # self.verbose_printrint( f' {species}',end='')
            ## load column desnity and effective air-broadening
            ## pressure species-by-species and perhaps optimise them
            p['N'].setdefault(species,P(1e16, False,1e13 ,nan,(0,np.inf)))
            p['pair'].setdefault(species,P(500, False,1e0,nan,(1e-3,10000),))
            if species in species_to_fit:
                p['N'][species].vary = fit_N
                p['pair'][species].vary = fit_pair
            else:
                p['N'][species].vary =False
                p['pair'][species].vary =False
            ## load data from HITRAN linelists
            tline = hitran.get_lines(species)
            ## Trim lines that are too weak to matter
            if species in species_to_fit:
                ## Remove very weak lines that probably wont
                ## contribute at any reasonable column density
                tline.limit_to_match(S296K_min=1e-25)
            else:
                ## if species not fit then lines can be trimmed to a minimum τ
                τpeak_min = 1e-3    # approx minimum peak τ to include a line
                # τpeak_min = 1e-5    # approx minimum peak τ to include a line
                S296K_min = τpeak_min*1e-3/p['N'][species]    # resulting approx min S296K
                if species in species_to_fit:
                    ## if this species is to fitted make sure at least
                    ## some lines are included
                    S296K_min = min(S296K_min,np.max(tline['S296K'])/10) 
                tline = tline.matches(S296K_min=S296K_min)
            ## add lines
            model.add_line(
                tline,
                Teq=p['Teq'],
                Nself=p['N'][species],
                pair=p['pair'][species],
                ymin=None,
                ncpus=self.ncpus,
                nfwhmL=3000,
                lineshape='voigt',
                verbose=False,
            )
        ## uninterpolate model grid
        if p['interpolate_model'] is not None:
            model.uninterpolate(average= True)
        ## scale to correct background intensity — vary points in range and neighbouring
        ## fit background if needed
        if fit_intensity and 'intensity_spline' not in p:
            p['intensity_spline'] = self.auto_intensity()
        if 'intensity_spline' in p:
            for i,(xi,yi) in enumerate(p['intensity_spline']):
                yi.vary = False
                if fit_intensity:
                    if xi >= xbeg and  xi <= xend:
                        yi.vary = True
                    if i > 0:
                        xprev,yprev = p['intensity_spline'][i-1]
                        if xi >= xbeg and xprev < xbeg:
                            yprev.vary = True
                        if xi >= xend and xprev < xend:
                            yi.vary = True
            model.multiply_spline(p['intensity_spline'])
        ## scale by sinusoidal background, vary points completely within range
        if fit_sinusoid and 'intensity_sinusoid' not in p:
            p['intensity_sinusoid'] = self.auto_sinusoid()
        if 'intensity_sinusoid' in p:
            for i,(xbegi,xendi,freqi,phasei,amplitudei) in enumerate(p['intensity_sinusoid']):
                freqi.vary = phasei.vary = amplitudei.vary = False
                if fit_sinusoid and xbegi >= xbeg and xendi <= xend:
                    freqi.vary = phasei.vary = amplitudei.vary =  True
            model.add_piecewise_sinusoid(p['intensity_sinusoid'])
        ## # ## scale by sinusoidal background, vary points completely within range
        ## # if fit_second_sinusoid:
        ##     # p.setdefault('second_intensity_sinusoid',self.auto_sinusoid())
        ## # if 'second_intensity_sinusoid' in p:
        ##     # for i,(xbegi,xendi,freqi,phasei,amplitudei) in enumerate(p['second_intensity_sinusoid']):
        ##         # freqi.vary = phasei.vary = amplitudei.vary = False
        ##         # if fit_sinusoid and xbegi >= xbeg and xendi <= xend:
        ##             # freqi.vary = phasei.vary = amplitudei.vary =  True
        ##     # model.add_piecewise_sinusoid(p['second_intensity_sinusoid'])
        ## instrument broadening
        if 'instrument_gaussian' in p or fit_instrument:
            p.setdefault('instrument_gaussian',P(0.02,True,1e-5,nan,(0.01,0.1)))
            p['instrument_gaussian'].vary = fit_instrument
            model.convolve_with_gaussian(p['instrument_gaussian'])
        else:
            model.convolve_with_blackman_harris()
        ## build it now
        model.construct()
        return model
        
def _similar_within_fraction(x,y,maxfrac=1e14):
    """Test if nonzero values of x and y are similar within a maximum fraction abs(x/y)."""
    i = (x!=0)&(y!=0)
    x,y = x[i],y[i]
    assert ~np.any(x==0),'some x is 0 when y is not'
    assert ~np.any(y==0),'some y is 0 when x is not'
    frac = x/y
    fracmax,fracmin = np.max(frac),np.min(frac)
    if fracmax == 0 and fracmin == 0:
        total_fractional_range = 0
    else:
        total_fractional_range = abs((fracmax-fracmin)/((fracmax+fracmin)/2))
    return total_fractional_range < maxfrac

def auto_fit(
        filename,
        xbeg=600,
        xend=6000,
        species_to_fit=('H2O','CO','CO2','NH3','SO2','H2S','CH4','CS2','HCN',
                        'N2O','NO','NO2','OCS','C2H2','C2H6','HCOOH',),
        verbose=False,
        make_plot=True,
        prefit=True,
        cycles=3,
        fit_intensity=True,
        fit_sinusoid=False,
        fit_instrument=True,
        fit_scalex=False,
        fit_temperature=False,
        regions='bands',
        fit_full_model=False,
        reference_species='H2O',
        interpolate_model=0.001,
        max_nfev=5,
        
):

    

    o = FitReferenceAbsorption(
        filename=filename,
        xbeg=xbeg,
        xend=xend,
        verbose=verbose,
        make_plot=make_plot,
        interpolate_model=interpolate_model,
        )
    
    o.figure_number = plotting.gcf().number
    o.ncpus = 1
    o.auto_fit(
        species_to_fit=species_to_fit,
        prefit=prefit,
        cycles=cycles,
        fit_intensity=fit_intensity,
        fit_sinusoid=fit_sinusoid,
        fit_instrument=fit_instrument,
        fit_temperature=fit_temperature,
        fit_scalex=fit_scalex,
        regions=regions,
        fit_full_model=fit_full_model,
        reference_species=reference_species,
        max_nfev=max_nfev,
    )
    o.full_model()
    return o
