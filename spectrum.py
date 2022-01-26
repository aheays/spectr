
import inspect,re
from copy import copy,deepcopy
from pprint import pprint
import warnings

from matplotlib import pyplot as plt
from scipy import signal,constants,fft,interpolate
from scipy.constants import pi as π
import numpy as np
from numpy import arange
import h5py
# from immutabledict import immutabledict as idict


from . import optimise
from .optimise import *
from . import plotting
from . import tools
from .tools import timestamp
from . import hitran
from . import bruker
from . import lineshapes
from . import lines
from . import levels
from .exceptions import DatabaseException,InferException
from . import dataset
from . import database
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
            # store=None,
    ):
        ## initalise optimiser variables
        # optimise.Optimiser.__init__(self,name,store=store)
        optimise.Optimiser.__init__(self,name)
        self.pop_format_input_function() 
        # store = self.store
        self.automatic_format_input_function()
        ## initialise data arrays
        self.x = None
        self.y = None
        self.experimental_parameters = {} # a dictionary containing any additional known experimental parameters
        ## if filename give then load the data -- some limited attempt
        ## to interpret the filetype correctly
        if filename is not None:
            filetype = tools.infer_filetype(filename)
            if filetype == 'opus':
                self.set_spectrum_from_opus_file(filename,xbeg=xbeg,xend=xend)
            else:
                self.set_spectrum_from_file(filename,xbeg=xbeg,xend=xend)
            self.pop_format_input_function()
        ## spectrum given as arrays
        if x is not None and y is not None:
            self.set_spectrum(x,y,xbeg,xend)
        if noise_rms is not None:
            self.experimental_parameters['noise_rms'] = float(noise_rms)
        self.add_save_to_directory_function(self.output_data_to_directory)
        self._figure = None
        self.background = None  # on same grid as x and y, the "background" spectrum – whatever that means

    @optimise_method()
    def set_spectrum(
            self,
            x,y,
            xbeg=None,
            xend=None,
            xspline=None,
            _cache=None,
            **experimental_parameters
    ):
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
            if len(x) == 0:
                raise Exception('no points in spectrum')
            if len(x) == 1:
                raise Exception('only one point in spectrum')
            self.experimental_parameters['xbeg'] = xbeg 
            self.experimental_parameters['xend'] = xend
            ## spline to a new grid:
            if xspline == 'auto':
                ## respline the data to the smallest grid step in its range
                dx = np.min(np.diff(x))
                xnew = linspace(x[0],x[-1],int((x[-1]-x[0])/dx)+1)
                ynew = tools.spline(x,y,xnew)
                x,y = xnew,ynew
            elif tools.isnumeric(xspline):
                ## respline to given grid step
                dx = np.min(np.diff(x))
                xnew = arange(x[0],x[-1],xspline)
                ynew = tools.spline(x,y,xnew)
                x,y = xnew,ynew
            elif tools.isnumeric(xspline):
                ## respline to given grid
                x,y = xspline,tools.spline(x,y,xspline)
            ## check for regular x grid
            t0,t1 = np.diff(x).min(),np.diff(x).max()
            assert (t1-t0)/t1<1e-3, 'Experimental data must be on an uniform grid.' # within a factor of 1e3
            self.experimental_parameters.setdefault('data_source','unknown')
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

    @optimise_method()
    def set_spectrum_from_dataset(self,filename,xbeg=None,xend=None,xkey='x',ykey='y',_cache=None):
        """Load a spectrum to fit from an x,y file."""
        if len(_cache) == 0:
            data = dataset.load(filename)
            _cache['experimental_parameters'] = deepcopy(data.attributes)
            _cache['experimental_parameters']['filename'] = filename
            _cache['x'] = data[xkey]
            _cache['y'] = data[ykey]
        self.set_spectrum(x=_cache['x'],y=_cache['y'],xbeg=xbeg,xend=xend,**_cache['experimental_parameters'])
        self.pop_format_input_function()

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

    @format_input_method()
    def set_spectrum_from_opus_file(self,filename,xbeg=None,xend=None,_cache=None):
        """Load a spectrum in an Bruker opus binary file. If filename is a
        list of filenames then compute the average."""
        self.experimental_parameters['filename'] = filename
        ## divide first filename from the rest for averaging
        if isinstance(filename,str):
            other_filenames = []
        else:
            filename,other_filenames = filename[0],filename[1:]
        ## load first filename
        opusdata = bruker.OpusData(filename)
        x,y = opusdata.get_spectrum()
        d = opusdata.data
        self.experimental_parameters['data_source'] = 'opus'
        translate_apodisation_function = {'BX':'boxcar','B3':'Blackman-Harris 3-term',}
        if 'Fourier Transformation' in d:
            self.experimental_parameters['interpolation_factor'] = float(d['Fourier Transformation']['ZFF'])
            self.experimental_parameters['apodisation_function'] = translate_apodisation_function[d['Fourier Transformation']['APF']]
        if 'Acquisition' in d:
            ## opus resolution is zero-to-zero of central sinc function peak
            ##self.experimental_parameters['resolution'] = float(d['Acquisition']['RES'])
            self.experimental_parameters['resolution'] = opusdata.get_resolution(kind='resolution',return_none_on_error=True)
            self.experimental_parameters['sinc_fwhm'] = opusdata.get_resolution(kind='fwhm',return_none_on_error=True)
        ## if necessary load further filenames and average
        if len(other_filenames) > 0:
            for filename in other_filenames: 
                opusdata = bruker.OpusData(filename)
                xnew,ynew = opusdata.get_spectrum()
                if not np.all(xnew==x):
                    raise Exception(f'x-grid of {filename} does not match {self.experimental_parameters["filename"]}')
                y += ynew
                d = opusdata.data
                if 'interpolation_factor' in self.experimental_parameters:
                    if self.experimental_parameters['interpolation_factor'] != float(d['Fourier Transformation']['ZFF']):
                        raise Exception(f'Interpolation factor of {filename} does not match {self.experimental_parameters["filename"]}')
                if 'apodisation_function' in  self.experimental_parameters:
                    if self.experimental_parameters['apodisation_function'] != translate_apodisation_function[d['Fourier Transformation']['APF']]:
                        raise Exception(f'Apodisation function of {filename} does not match {self.experimental_parameters["filename"]}')
                if 'resolution' in  self.experimental_parameters:
                    if self.experimental_parameters['resolution'] != opusdata.get_resolution():
                        raise Exception(f'Sinc FWHM function of {filename} does not match {self.experimental_parameters["filename"]}')
            y /= len(other_filenames)+1
        self.set_spectrum(x,y,xbeg,xend)
        self.pop_format_input_function() 
    
    @optimise_method()
    def set_spectrum_from_soleil_file(self,filename,xbeg=None,xend=None,_cache=None):
        """ Load soleil spectrum from file with given path."""
        ## only runs once
        if 'has_run' in _cache:
            return
        _cache['has_run'] = True
        x,y,header = load_soleil_spectrum_from_file(filename)
        self.experimental_parameters['filename'] = filename
        self.experimental_parameters['data_source'] = 'DESIRS FTS'
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
    def fit_noise(
            self,
            xbeg=None,
            xend=None,
            xedge=None,
            n=1,
            make_plot=False,
            figure_number=None,
    ):
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

    def plot(
            self,
            ax=None,
            plot_background= True,
            fig=None,          # figure object or number
    ):                          
        """Plot spectrum."""
        self.construct()
        ##
        if self.x is None:
            raise Exception('No data. File loaded?')
        ## use current figure or start a new one wit number fig.
        if fig is None:
            fig = plotting.gcf()
        elif isinstance(fig,int):
            fig = plotting.qfig(n=fig)
        else:
            fig = plotting.gcf()
        ## reuse current axes if not specified
        if ax is None:
            ax = fig.gca()
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
        if plot_background and self.background is not None:
            ax.plot(self.x,self.background,label='background')
        plotting.legend(ax=ax,loc='upper left')
        if 'filename' in self.experimental_parameters:
            ax.set_title(str(self.experimental_parameters['filename']))
        self._figure = plotting.gcf()
        return ax

    @optimise_method()
    def fit_background_spline(self,xi=100,xbeg=None,xend=None,make_plot=False,**fit_kwargs):
        """Use fit_spline_to_extrema_or_median to fit the background in the presence of lines."""
        xs,ys,yf = tools.fit_spline_to_extrema_or_median(self.x,self.y,xi=xi,make_plot=make_plot,**fit_kwargs)
        self.background = yf

    @optimise_method()
    def normalise_background(self):
        """Use fit_spline_to_extrema_or_median to fit the background in the presence of lines."""
        if self.background is None:
            raise Exception(r'Background must be defined before calling normalise_background.')
        self.y /= self.background 
        self.background /= self.background

    def integrate_signal(self):
        """Trapezoidally integrated signal."""
        from scipy import integrate
        self.experimental_parameters['integrated_signal'] = integrate.trapz(self.y,self.x)
        return self.experimental_parameters['integrated_signal']

    def integrate_excess(self,method='trapz'):
        """Trapezoidally integrate difference between signal and background."""
        self.experimental_parameters['integrated_excess'] = tools.integrate(self.x,self.y-self.background,method=method)
        return self.experimental_parameters['integrated_excess']

@format_input_method()
class Model(Optimiser):

    def __init__(
            self,
            name=None,
            experiment=None,
            load_experiment_args=None,
            residual_weighting=None,
            verbose=None,
            xbeg=None,xend=None,
            x=None
    ):
        ## set experimental data if provided
        if load_experiment_args is not None:
            ## build an Experiment from method givens as dict of method:args
            experiment = Experiment('experiment')
            for method,args in load_experiment_args.items():
                getattr(experiment,method)(*args)
        if isinstance(experiment,str):
            ## load experiment='filename'
            experiment = Experiment(filename=experiment)
        self.experiment = experiment
        ## set na,e
        if name is None:
            if self.experiment is None:
                name = 'model'
            else:
                name = f'model_of_{experiment.name}'
        self.x = None
        self.y = None
        self._xin = x          
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
        self._initialise()
        self.add_post_construct_function(self.get_residual)
        self.add_save_to_directory_function(self.output_data_to_directory)
        self.add_plot_function(lambda: self.plot(plot_labels=False))
        self._figure = None

        # ## make a format_input_function
        # args = {}
        # for key in ('name','experiment',
                    # 'load_experiment_args',
                    # 'residual_weighting',
                    # 'verbose', 'xbeg','xend','x',):
            # val = locals()[key]
            # if val is not None:
                # args[key] = val
        # self.add_format_input_function(
            # lambda: (f'{self.name} = spectrum.Model('
                      # + ','.join([f'{key}={val!r}' for key,val in args.items()])
                      # + ')'))


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
                self.x = linspace(100,1000,1000,dtype=float)
                self._compute_residual =False
        ## new grid
        self.y = np.zeros(self.x.shape,dtype=float)

    def __len__(self):
        return len(self.x)
        
    def get_spectrum(self,x):
        """Construct a model spectrum at x."""
        self._xin = np.asarray(x,dtype=float) # needed in _initialise
        # self.timestamp = -1     # force reconstruction, but not suboptimisers
        self._clean_construct = True   # force reconstruction
        self.construct()
        self._xin = None        # might be needed in next _initialise
        return self.y

    def get_residual(self,return_x=False):
        """Compute residual error."""
        ## no experiment to compute with
        if self.experiment is None:
            return None
        ## no residual requested
        if not self._compute_residual:
            return None
        ## possible subsample removing _interpolate_factor so grid
        ## matches experiment
        if self._interpolate_factor is None:
            residual = self.yexp - self.y
        else:
            residual = self.yexp - self.y[::self._interpolate_factor]
        ## weight residual
        if self.residual_weighting is not None:
            residual *= self.residual_weighting
        ## return residual, possible with x
        if return_x:
            ## if return x return uninterpolated version
            if self._interpolate_factor is None:
                x = self.x
            else:
                x = self.x[::self._interpolate_factor]
            return residual,x
        else:
            return residual

    @optimise_method()
    def interpolate(self,dx,_cache=None):
        """When calculating model set to dx grid (or less to achieve overlap
        with experimental points. Always an odd number of intervals /
        even number of interstitial points. Call before anything else,
        self.y is deleted and replaced with zeros on the new grid."""
        if self._clean_construct:
            xstep = (self.x[-1]-self.x[0])/(len(self.x)-1)
            interpolate_factor = int(np.ceil(xstep/dx))
            if interpolate_factor%2 == 0:
                interpolate_factor += 1
            _cache['x'] = np.linspace(self.x[0],self.x[-1],1+(len(self.x)-1)*interpolate_factor)
            if not np.all(self.y==0):
                raise Exception('interpolate will erase nonzero, y, it is intended to be the first Model method called')
            _cache['y'] = np.zeros(_cache['x'].shape,dtype=float)
            _cache['interpolate_factor'] = interpolate_factor
        self._interpolate_factor = _cache['interpolate_factor']
        if self._interpolate_factor != 1:
            self.x = _cache['x']
            self.y = _cache['y'].copy() # delete current y!!

    @optimise_method()
    def uninterpolate(self,average=None):
        """If the model has been interpolated then restore it to
        original grid. If average=True then average the values in each
        interpolated interval."""
        from .fortran_tools import fortran_tools
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
                ## reduce to non-interpolated points -- using fortran code
                y = np.empty(self.x.shape,dtype=float)
                fortran_tools.uninterpolate_with_averaging(self.y,y,self._interpolate_factor)
                self.y = y
            else:
                ## reduce to non-interpolated points
                self.y = self.y[::self._interpolate_factor]
            self._interpolate_factor = None

    @format_input_method()
    def add_hitran_line(
            self,
            species,
            min_S296K=0,        # only lines with greater absorption line strength
            *args,
            **kwargs
    ):
        """Automatically load a HITRAN linelist for a species
        (isotopologue or natural abundance) and then call add_line."""
        line = hitran.get_line(
            species,
            # cache=False,
            # copy_cache=False,
            # force_download= True,
            # force_convert= True,
            match={
                'min_ν':self.x[0]-10,
                'max_ν':self.x[-1]+10,
                'min_S296K':min_S296K,
                },
        )

        line.include_in_output = False
        line.clear_format_input_functions()
        self.add_line(line,*args,**kwargs)
        self.pop_format_input_function()

    @optimise_method()
    def add_line(
            self,
            line,               # lines.Generic or a subclass
            kind='absorption',  # 'absorption', 'emission' or else any other key defined in line
            nfwhmL=inf,          # number of Lorentzian fwhms to include
            nfwhmG=10,          # number of Gaussian fwhms to include
            ymin=0,             # minimum value of ykey to include
            lineshape=None,     # as in lineshapes.py, or will be automatically selected
            ncpus=1,            # for multiprocessing
            verbose=None,       # print info
            force_full_recalc=None,
            match=None,         # only include lines matching keys:vals in this dictionary
            xedge=1, # include lines within this much of the domain edges
            _cache=None,
            **set_keys_vals
    ):
        if verbose:
            timer_start = timestamp()
        if len(self.x) == 0 or len(line) == 0:
            ## nothing to be done
            return
        if self._clean_construct:
            ## first run — initalise local copy of lines data, do not
            ## set keys if they are in set_keys_vals
            tmatch = ({} if match is None else copy(match))
            tmatch.setdefault('min_ν',(self.x[0] -xedge))
            tmatch.setdefault('max_ν',(self.x[-1]+xedge))
            imatch = line.match(**tmatch)
            nmatch = np.sum(imatch)
            keys = [key for key in line.explicitly_set_keys() if key not in set_keys_vals]
            line_copy = line.copy(index=imatch,keys=keys)
            if verbose is not None:
                line_copy.verbose = verbose
            if verbose:
                print(f'add_line: {line.name}: clean construct')
            ## set parameter/constant data. If a vector of data is
            ## given then its length matches the input dataset
            for key,val in set_keys_vals.items():
                if tools.isiterable(val):
                    line_copy[key] = val[imatch]
                else:
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
                    ymin=ymin, ncpus=ncpus, lineshape=lineshape,index=index,
                    xedge=xedge)
                if kind == 'absorption':
                    y = np.exp(-y)
                return y
            y = _calculate_spectrum(line_copy,None)
            ## cache
            _cache['line_copy_prev'] = line_copy.copy()
            _cache['y'] = y
            _cache['imatch'] = imatch
            _cache['nmatch'] = nmatch
            _cache['line_copy'] = line_copy
            _cache['ykey'] = ykey
            _cache['_calculate_spectrum'] = _calculate_spectrum
        else:
            ## subsequent runs -- maybe only recompute a few line
            ##
            ## load cache
            line_copy_prev = _cache['line_copy_prev']
            y = _cache['y']
            imatch = _cache['imatch']
            nmatch = _cache['nmatch']
            line_copy = _cache['line_copy']
            ykey = _cache['ykey']
            _calculate_spectrum = _cache['_calculate_spectrum']
            ## nothing to be done
            if nmatch == 0:
                return
            ## set modified data in set_keys_vals if they have changed
            ## from cached values.  Only update Parameters that have changed
            for key,val in set_keys_vals.items():
                if isinstance(val,Parameter):
                    if self._last_construct_time < val._last_modify_value_time:
                        line_copy[key] = val
                elif tools.isiterable(val):
                    line_copy.set(key,'value',val[imatch],set_changed_only=True)
                else:
                    line_copy.set(key,'value',val,set_changed_only=True)
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
                if force_full_recalc:
                    ##  full ## recalculation
                    if verbose:
                        print(f'add_line: {line.name}: {force_full_recalc=}, full recalculation')
                    y = _calculate_spectrum(line_copy,None)
                    ## new previous data
                    _cache['line_copy_prev'] = line_copy.copy()
                elif (
                        ## all lines have changed
                        (nchanged == len(ichanged))
                        ## ykey has changed
                        and (line_copy[ykey,'_modify_time'] > self._last_construct_time)
                        ## no key other than ykey has changed
                        and (np.all([line_copy[key,'_modify_time'] < self._last_construct_time for key in line_copy_prev if key != ykey]))
                        ## ykey has changed by a near-constant factor -- RISKY!!!!
                        and _similar_within_fraction(line_copy_prev[ykey],line_copy[ykey])
                        ## if ymin is set then scaling is dangerous -- lines can fail to appear when scaled up
                        and (ymin is None or ymin == 0)
                ):
                    ## constant factor ykey -- scale saved spectrum
                    if ymin is not None and ymin != 0:
                        warnings.warn(f'Scaling spectrum uniformly but ymin is set to a nonzero value, {repr(ymin)}.  This could lead to lines appearing in subsequent model constructions.')
                    if verbose:
                        print(f'add_line: {line.name}: constant factor scaling all lines')
                    ## ignore zero values
                    i = (line_copy[ykey]!=0)&(line_copy_prev[ykey]!=0)
                    scale = np.mean(line_copy[ykey,i]/line_copy_prev[ykey][i])
                    if kind == 'absorption':
                        y = y**scale
                    else:
                        y = y*scale
                    ## new previous data
                    _cache['line_copy_prev'] = line_copy.copy()
                elif nchanged/len(ichanged) > 0.5:
                    ## more than half lines have changed -- full
                    ## recalculation
                    if verbose:
                        print(f'add_line: {line.name}: more than half the lines ({nchanged}/{len(ichanged)}) have changed, full recalculation')
                    y = _calculate_spectrum(line_copy,None)
                    ## new previous data
                    _cache['line_copy_prev'] = line_copy.copy()
                elif nchanged > 0:
                    ## a few lines have changed, update these only
                    if verbose:
                        print(f'add_line: {line.name}: {nchanged} lines have changed, recalculate these')
                    ## temporary object to calculate old spectrum
                    yold = _calculate_spectrum(line_copy_prev,ichanged)
                    ## partial recalculation
                    ynew = _calculate_spectrum(line_copy,ichanged)
                    if kind == 'absorption':
                        y = y * ynew / yold
                    else:
                        y = y + ynew - yold
                    ## new previous data
                    _cache['line_copy_prev'] = line_copy.copy()
                else:
                    ## nothing changed keep old spectrum
                    pass
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
        _cache['set_keys_vals'] = {key:(val.value if isinstance(val,Parameter) else val) for key,val in set_keys_vals.items()}
        if verbose:
            print(f'add_line: {line.name}: time = {timestamp()-timer_start:0.3e}')

    @optimise_method()
    def add_spectrum(self,spectrum,kind='absorption',_cache=None,**set_keys_vals):
        """Add spectrum in a Spectrum object. kind='absorption' or
        'emission'. set_keys_vals set data according to Spectrum
        prototypes before adding to the model."""
        if self._first_construct:
            ## Only runs on the initial construction of the model.
            ## Make a copy of the input spectrum so it is not altered
            ## by this method.
            spectrum_copy = spectrum.copy(optimise=True)
            spectrum_copy.include_in_output = False
            ## Set set_keys_vals in the copied spectrum object.
            for key,val in set_keys_vals.items():
                spectrum_copy.set_value(key,val)
            ## Add a construct_function to the copied spectrum objec
            ## that computes the cross section splined to the
            ## experimental grid
            if kind == 'absorption':
                xkey = 'T'          # transmittance
            elif kind == 'emission':
                xkey = 'I'          # intensity
            else:
                raise Exception(f'Unknown {kind=}')
            ## limit spline to a meaningful frequency range
            index = tools.inrange(spectrum['ν'],self.x[0],self.x[-1],include_adjacent=True,return_as='slice')
            def compute_spline():
                _cache['spline'] = tools.spline(
                    spectrum_copy['ν',index],spectrum_copy[xkey,index],
                    self.x,out_of_bounds='zero')
            spectrum_copy.add_construct_function(compute_spline)
            ## add the copied spectrum to self to ensure it is up to
            ## date before adding the spline to the model
            self.add_suboptimiser(spectrum_copy)
            _cache['spectrum_copy'] = spectrum_copy
        spectrum_copy = _cache['spectrum_copy']
        ## modify self.y according to what kind of spectrum this is,
        ## scale by transmittance or add intensity
        if kind == 'absorption':
            self.y *= _cache['spline']
        elif kind == 'emission':
            self.y += _cache['spline']
        else:
            assert False

    @optimise_method()
    def add_absorption_cross_section_array(self,x,y,column_density=1,_cache=None):
        """Add absorption cross section in arrays x and y."""
        if self._clean_construct:
            _cache['ys'] = tools.spline(x,y,self.x,out_of_bounds='zero')
        ys = _cache['ys']
        self.y *= np.exp(-column_density*ys)

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
    def add_constant(self,constant=1):
        """Shift by a spline defined function."""
        self.y += float(constant)

    @optimise_method()
    def add_from_file(self,filename,yscale=1,_cache=None):
        """Spline contents of file to model grid and add to spectrum."""
        if self._clean_construct:
            if 'x' not in _cache:
                x,y = tools.file_to_array_unpack(filename)
                _cache['x'],_cache['y'] = x,y
            x,y = _cache['x'],_cache['y']
            ys = tools.spline(x,y,self.x)
            _cache['ys'] = ys
        ys = _cache['ys']
        self.y += ys * float(yscale)

    # @optimise_method()
    # def add_spectrum(self,spectrum,scale=1,_cache=None):
        # """Add x and y arrays to model, splining to model grid."""
        # if self._clean_construct:
            # ys = tools.spline(x,y,self.x)
            # _cache['ys'] = ys
        # ys = _cache['ys']
        # self.y += ys * float(scale)

    @optimise_method()
    def add_arrays(self,x,y,scale=1,_cache=None):
        """Add x and y arrays to model, splining to model grid."""
        if self._clean_construct:
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
    def multiply_spline(
            self,
            knots=None,
            order=3,
            _cache=None,
            autovary= True,
    ):
        """Multiple y by a spline function."""
        if self._clean_construct:
            spline = Spline(knots=knots,order=order)
            i = (self.x >= np.min(spline.xs)) & (self.x <= np.max(spline.xs))
            ## a quick hack to prevent round off errors missing the
            ## first or last points of the splined domain
            if not i[0] and i[1]:
                i[0] = True
            if not i[-1] and i[-2]:
                i[-1] = True
            ## to be implemented propertly
            i = np.full(len(self.x),True)
            _cache['i'] = i
            _cache['spline'] = spline
            ## vary only those knots in model domain
            _cache.setdefault('first_invocation',True)
            if autovary and _cache['first_invocation']:
                self.autovary_in_range(knots,self.x[i][0],self.x[i][-1])
                _cache['first_invocation'] = False
        i = _cache['i']
        spline = _cache['spline']
        if spline.last_change_time > self.last_construct_time:
            _cache['yspline'] = spline(self.x[i])
        ## add to y
        self.y[i] *= _cache['yspline']
    
    @optimise_method()
    def add_spline(self,knots=None,order=3,_cache=None,autovary=False):
        """Multiple y by a spline function."""
        if self._clean_construct:
            spline = Spline(knots=knots,order=order)
            i = (self.x >= np.min(spline.xs)) & (self.x <= np.max(spline.xs))
            ## a quick hack to prevent round off errors missing the
            ## first or last points of the splined domain
            if not i[0] and i[1]:
                i[0] = True
            if not i[-1] and i[-2]:
                i[-1] = True
            _cache['i'] = i
            _cache['spline'] = spline
            ## vary only those knots in model domain
            _cache.setdefault('first_invocation',True)
            if autovary and _cache['first_invocation']:
                self.autovary_in_range(knots,self.x[i][0],self.x[i][-1])
                _cache['first_invocation'] = False
        i = _cache['i']
        spline = _cache['spline']
        if spline.last_change_time > self.last_construct_time:
            _cache['yspline'] = spline(self.x[i])
        ## add to y
        self.y[i] += _cache['yspline']
    
    def autovary_in_range(
            self,               
            parameters,         # list of lists, [x,rest..]
            xbeg=None,xend=None, # range to set to vary, defaults to current range of Model
            vary=True,           # set to this vary if in range
            vary_outside_range=False, # set to this vary if outside range, None to do nothing to these Parameters
            include_adjacent=True, # include points immediately outside range
    ):
        """Set Parameters in list of lists to vary if in defined x range."""
        ## default to entire model
        if xbeg is None:
            xbeg = self.x[0]
        if xend is None:
            xend = self.x[-1]
        ## find points in range -- assumes xs sorted
        xs = np.array([t[0] for t in parameters])
        ps = np.array([t[1:] for t in parameters])
        i = tools.inrange(xs,xbeg,xend,include_adjacent)
        ## set in-range vary
        for p in ps[i]:
            for t in p:
                if isinstance(p,Parameter):
                    p.vary = vary
        ## set out-of-range vary
        if vary_outside_range is not None:
            for p in ps[~i]:
                for t in p:
                    if isinstance(p,Parameter):
                        p.vary = vary_outside_range
        
    @optimise_method()
    def scale_and_shift_from_unity_splines(
            self,
            total_knots=None,
            shift_knots=None,
            order=3,
            autovary=False,
            _cache=None,
    ):
        """Assuming a self.y is normalised in some sense, shift it by
        shift_knots and scale it so that a unity corresponds to
        total_knots."""
        if self._clean_construct:
            ## make splines objects
            total_spline = Spline(f'{self.name}_total_spline',total_knots,order)
            shift_spline = Spline(f'{self.name}_shift_spline',shift_knots,order)
            ## update indices on clean construct
            i = ((self.x >= max(np.min(total_spline.xs),np.min(shift_spline.xs)))
                 & (self.x <= min(np.max(total_spline.xs),np.max(shift_spline.xs))))
            ## a quick hack to prevent round off errors missing the
            ## first or last points of the splined domain
            if not i[0] and i[1]:
                i[0] = True
            if not i[-1] and i[-2]:
                i[-1] = True
            ## if autovary then set spline parameters to be varied if
            ## in the range of i or the immediate points outside range

            ## vary only those knots in model domain
            _cache.setdefault('first_invocation',True)
            if autovary and _cache['first_invocation']:
                txbeg,txend = self.x[i][0],self.x[i][-1]
                self.autovary_in_range(total_knots,txbeg,txend)
                self.autovary_in_range(shift_knots,txbeg,txend)
                _cache['first_invocation'] = False
            _cache['i'] = i
            _cache['total_spline'] = total_spline
            _cache['shift_spline'] = shift_spline
        ## compute
        i = _cache['i']
        total_spline = _cache['total_spline']
        shift_spline = _cache['shift_spline']
        ## recompute spline if necessary
        if total_spline.last_change_time > self.last_construct_time:
            _cache['ytotal'] = total_spline[self.x[i]]
        ytotal = _cache['ytotal']
        if shift_spline.last_change_time > self.last_construct_time:
            _cache['yshift'] = shift_spline[self.x[i]]
        yshift = _cache['yshift']
        ## compute scaled and shifted y
        self.y[i] = self.y[i]*(ytotal-yshift) + yshift

    # def auto_scale_piecewise_sinusoid(
            # self,
            # xjoin=10, # specified interval join points or distance separating them
            # xbeg=None,xend=None, # limit to this range
            # vary=False,Avary=False, # vary phase, frequencey and/or amplitude
    # ):
        # """Automatically find regions for use in
        # scale_by_piecewise_sinusoid."""
        # warnings.warn('deprecated in favour of add_sinusoid_spline, or modify to match')
        # ## get join points between regions and begining and ending points
        # if np.isscalar(xjoin):
            # if xbeg is None:
                # xbeg = self.x[0]
            # if xend is None:
                # xend = self.x[-1]
            # i = slice(*np.searchsorted(self.x,[xbeg,xend]))
            # xjoin = np.concatenate((arange(self.x[i][0],self.x[i][-1],xjoin),self.x[i][-1:]))
        # else:
            # if xbeg is None:
                # xbeg = xjoin[0]
            # else:
                # xbeg = max(xbeg,xjoin[0])
            # if xend is None:
                # xend = xjoin[-1]
            # else:
                # xend = max(xend,xjoin[-1])
        # ## loop over all regions, gettin dominant frequency and phase
        # ## from the residual error power spectrum
        # regions = []
        # x,y = self.x,self.y
        # if self._interpolate_factor is not None:
            # x,y = x[::self._interpolate_factor],y[::self._interpolate_factor]
        # for xbegi,xendi in zip(xjoin[:-1],xjoin[1:]):
            # i = slice(*np.searchsorted(x,[xbegi,xendi]))
            # residual = self.yexp[i] - y[i]
            # FT = fft.fft(residual)
            # imax = np.argmax(np.abs(FT)[1:])+1 # exclude 0
            # phase = np.arctan(FT.imag[imax]/FT.real[imax])
            # if FT.real[imax]<0:
                # phase += π
            # dx = (x[i][-1]-x[i][0])/(len(x[i])-1)
            # frequency = 1/dx*imax/len(FT)
            # amplitude = tools.rms(residual)/y[i].mean()
            # ## amplitude = tools.rms(residual)
            # regions.append([
                # xbegi,xendi,
                # P(amplitude,Avary,1e-5),
                # P(frequency,vary,frequency*1e-3),
                # P(phase,vary,2*π*1e-3),])
        # self.scale_piecewise_sinusoid(regions)
        # return regions

    def auto_scale_piecewise_sinusoid(
            self,
            xi=50,
            make_plot=False,
            vary=False,
            add_construct=True,
            # xbeg=None,xend=None,
    ):
        """Fit a spline interpolated sinusoid to current model residual, and
        add it to the model."""
        if self.experiment is None:
            raise Exception('auto_add_piecewise_sinusoid requires an experimental residual to fit')
        residual,x =self.get_residual(return_x=True)
        regions = tools.fit_piecewise_sinusoid(
            x,residual,
            # self.x,
            # self.get_residual(),
            xi=xi,
            plot=make_plot,
            optimise=False)
        ## scale amplitudes to signal
        for region in [list(t) for t in regions]:
            i,j = np.searchsorted(self.x,(region[0],region[1]))
            region[2] /= np.mean(self.y[i:j])
        ## make into Parmaters
        regions = [
            [xbeg,xend,P(amplitude,vary),P(frequency,vary),P(phase,vary,2*π*1e-5),]
                    for (xbeg,xend,amplitude,frequency,phase) in regions]
        if add_construct:
            self.scale_piecewise_sinusoid(regions)
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

    def auto_add_piecewise_sinusoid(
            self,
            xi=10,
            make_plot=False,
            vary=False,
            add_construct=True,
            # xbeg=None,xend=None,
    ):
        """Fit a spline interpolated sinusoid to current model residual, and
        add it to the model."""
        if self.experiment is None:
            raise Exception('auto_add_piecewise_sinusoid requires an experimental residual to fit')
        residual,x =self.get_residual(return_x=True)
        regions = tools.fit_piecewise_sinusoid(
            x,residual,
            # self.x,
            # self.get_residual(),
            xi=xi,
            plot=make_plot,
            optimise=False)
        regions = [
            [xbeg,xend,P(amplitude,vary),P(frequency,vary),P(phase,vary,2*π*1e-5),]
                    for (xbeg,xend,amplitude,frequency,phase) in regions]
        if add_construct:
            self.add_piecewise_sinusoid(regions)
        return regions

    @optimise_method()
    def add_piecewise_sinusoid(self,regions,_cache=None,_parameters=None):
        """Scale by a piecewise function 1+A*sin(2πf(x-xa)+φ) for a set
        regions = [(xa,xb,A,f,φ),...].  Probably should initialise
        with auto_scale_by_piecewise_sinusoid."""
        if (self._clean_construct or
            np.any([t._last_modify_value_time > self._last_construct_time for t in _parameters])):
            ## get spline points and compute splien
            i = (self.x >= regions[0][0]) & (self.x <= regions[-1][1]) # x-indices defined by regions
            sinusoid = tools.piecewise_sinusoid(self.x[i],regions)
            _cache['sinusoid'] = sinusoid
            _cache['i'] = i
        sinusoid = _cache['sinusoid']
        i = _cache['i']
        self.y[i] += sinusoid

    def auto_convolve_spline_signum(
            self,
            spline_step,
            amplitude=1e-3,
            vary=False,
            **convolve_spline_signum_kwargs
    ):
        """Convolve with a signum function with spline-varying amplitude."""
        amplitude_spline = []
        for x in linspace(self.x[0],self.x[-1],int((self.x[-1]-self.x[0])/spline_step)):
            y = P(amplitude,vary,1e-8)
            self.add_parameter(y)
            amplitude_spline.append([x,y])
        self.convolve_spline_signum(amplitude_spline,**convolve_spline_signum_kwargs)
        return amplitude_spline

    @optimise_method()
    def convolve_spline_signum(self,amplitude_spline,order=3,xmax=10,autovary=False,_cache=None):
        """Convolve with a signum function with spline-varying amplitude."""

        ## vary only those knots in model domain
        _cache.setdefault('first_invocation',True)
        if autovary and _cache['first_invocation']:
            _cache['first_invocation'] = False
            self.autovary_in_range(amplitude_spline)
        x,y = self.x,self.y
        dx = (x[-1]-x[0])/(len(x)-1) # grid step -- x must be regular
        ## get hyperbola to convolve -- Δx=0 is zero
        xconv = arange(dx,xmax,dx)
        yconv = 1/xconv
        xconv = np.concatenate((-xconv[::-1],[0],xconv))
        yconv = np.concatenate((-yconv[::-1],[0],yconv))
        ## scale y but signum magnitude
        yscaled = y*tools.spline_from_list(amplitude_spline,x,order=order)*dx
        ## get convolved asymmetric y to add to self 
        yadd = tools.convolve_with_padding(x,yscaled,xconv,yconv,)
        ## full signum added spectrum
        self.y += yadd

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

    @optimise_method()
    def convolve_with_sinc(self,width=None,fwhms_to_include=100,_cache=None):
        """Convolve with sinc function, width is FWHM."""
        ## check if there is a risk that subsampling will ruin the convolution
        if (self._clean_construct
            # or width is None
            or 'width' not in _cache
            or _cache['width'] != width):
            # if width is None:
                # if 'sinc_fwhm' in self.experiment.experimental_parameters:
                    # width = self.experiment.experimental_parameters['sinc_fwhm']
                # else:
                    # raise Exception("Width is None and could not be inferred from experimental_parameters")
            dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
            xconv = np.arange(0,fwhms_to_include*width,dx)
            xconv = np.concatenate((-xconv[-1:0:-1],xconv))
            yconv = lineshapes.sinc(xconv,Γ=width)
            yconv = yconv/yconv.sum() # sum normalised
            ## convolve and return, discarding padding
            _cache['xconv'] = xconv
            _cache['yconv'] = yconv
            _cache['width'] = float(width)
        self.y = tools.convolve_with_padding(self.x,self.y,_cache['xconv'],_cache['yconv'])

    @optimise_method()
    def convolve_with_blackman_harris(
            self,
            resolution,
            terms=3,
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
            # if resolution is not None:
                # pass
            # # elif self.experiment is not None and 'resolution' in self.experiment.experimental_parameters:
            # #     resolution = self.experiment.experimental_parameters['resolution']
            # elif self.experiment is not None and 'sinc_fwhm' in self.experiment.experimental_parameters:
                # resolution = self.experiment.experimental_parameters['sinc_fwhm']*1.2
            # else:
                # raise Exception('Resolution not specified as argument or in experimental data')
            width = resolution*0.6 # distance between sinc peak and first zero
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

    def auto_convolve_with_instrument_function(self,vary=None,**kwargs):
        """Convolve with instrument function."""
        data = self.experiment.experimental_parameters
        ## Bruker OPUS fts apodisation
        if data['data_source'] == 'opus':
            if data['apodisation_function'] == 'boxcar':
                sinc_fwhm = data['resolution']*1.2
                if vary is None:
                    kwargs['sinc_fwhm'] = float(sinc_fwhm)
                else:
                    kwargs['sinc_fwhm'] = P(sinc_fwhm,vary,1e-5,bounds=(0,inf))
            if data['apodisation_function'] == 'Blackman-Harris 3-term':
                kwargs['blackman_harris_order'] = 3
                if vary is None:
                    kwargs['blackman_harris_resolution'] = float(data['resolution'])
                else:
                    kwargs['blackman_harris_resolution'] = P(data['resolution'],vary,bounds=(0,inf))
        ## SOLEIL DESIRS FTS boxcar apodisation0
        elif data['data_source'] == 'DESIRS FTS':
            if vary is None:
                kwargs['sinc_fwhm'] = float(data['sinc_fwhm'])
            else:
                kwargs['sinc_fwhm'] = P(data['sinc_fwhm'],vary,bounds=(0,inf))
        else:
            raise Exception(f'Cannot auto_convolve_with_instrument_function')
        ## call function
        self.convolve_with_instrument_function(**kwargs)
        return kwargs

    @optimise_method()
    def convolve_with_instrument_function(
            self,
            sinc_fwhm=None,
            gaussian_fwhm=None,
            lorentzian_fwhm=None,
            signum_magnitude=None,
            blackman_harris_resolution=None,
            blackman_harris_order=None,
            width=None,
            _cache=None,
    ):
        """Convolve with soleil instrument function."""
        ## compute instrument function
        dx = (self.x[-1]-self.x[0])/(len(self.x)-1) # ASSUMES EVEN SPACED GRID
        ## instrument function grid
        if width is None:
            ## width = (self.x[-1]-self.x[0])/2
            width = 10
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
        ## convolve with sinc functions corresponding to
        ## Blackman-Harris apodisation
        if blackman_harris_resolution is not None:
            twidth = blackman_harris_resolution*0.6*1.2 # factor is confusing 
            if blackman_harris_order == 3:
                yconv =  0.42323*np.sinc(x/twidth)
                yconv += 0.5*0.49755*np.sinc(x/twidth-1) 
                yconv += 0.5*0.49755*np.sinc(x/twidth+1)
                yconv += 0.5*0.07922*np.sinc(x/twidth-2)
                yconv += 0.5*0.07922*np.sinc(x/twidth+2)
            elif blackman_harris_order == 4:
                yconv =  0.35875*np.sinc(x/twidth)
                yconv += 0.5*0.48829*np.sinc(x/twidth-1) 
                yconv += 0.5*0.48829*np.sinc(x/twidth+1)
                yconv += 0.5*0.14128*np.sinc(x/twidth-2)
                yconv += 0.5*0.14128*np.sinc(x/twidth+2)
                yconv += 0.5*0.01168*np.sinc(x/twidth-3)
                yconv += 0.5*0.01168*np.sinc(x/twidth+3)
            else: 
                raise Exception("Only 3 and 4 Blackman-Harris terms implemented.")
            y = signal.oaconvolve(y,yconv, 'same')
        ## if necessary account for phase correction by convolving with a signum
        if signum_magnitude is not None:
            # x[imidpoint] = 1e-99  # avoid divide by zero warning
            ## hyperbolically decaying signum on either side. Use this
            ## preallocation to avoid calculation at imidpoint which
            ## will be a divide-by-zero warning
            ty = np.full(x.shape,1.0)
            for i in (slice(0,imidpoint),slice(imidpoint+1,len(x))):
                ty[i] = signum_magnitude/x[i]*dx
            # ty = 1/x*signum_magnitude
            # ty[imidpoint] = 1 # the central part  -- a delta function
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
        
    @format_input_method()
    def plot(
            self,
            ## control axes to plot on
            ax=None,            # axes object to use
            fig=None,           # figure object to use
            ## plot curves
            plot_model= True,
            plot_experiment= True,
            plot_residual= True,
            invert_model=False,
            shift_residual=0,
            scale_residual=1,
            ## label transitions
            plot_labels=False,
            label_match=None,   # label only matching things
            label_key='default',     # this key is used to label series, None for none
            label_zkeys=None,   # divide series by these keys
            minimum_τ_to_label=None, # for absorption lines
            minimum_I_to_label=None, # for emission lines
            minimum_Sij_to_label=None, # for emission lines
            ## show reference line frequencies
            plot_contaminants=False,
            contaminants=None,
            ## various things added to plot
            plot_title=False,
            title=None,
            plot_legend=True,
            xlabel=None,
            ylabel=None,
            xticks=None,
            yticks=None,
            xlim=None,
            ylim=None,
            plot_kwargs=None,
            ## no text at all if False, much faster draw
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
                fig = plotting.qfig(n=fig)
            else:
                fig = plotting.gcf()
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
        ymin,ymax = np.inf,-np.inf
        xmin,xmax = np.inf,-np.inf
        ## if model has no data return immediately
        if len(self) == 0:
            return fig
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
        if (plot_residual
            and self.y is not None
            and self.experiment is not None
            and self.experiment.y is not None
            and self._compute_residual is not None):
                yres = self.get_residual()
                ymin,ymax = min(ymin,yres.min()+shift_residual),max(ymax,yres.max()+shift_residual)
                xmin,xmax = min(xmin,self.x.min()),max(xmax,self.x.max())
                tkwargs = dict(color=plotting.newcolor(2), label='Experiment-Model residual', **plot_kwargs)
                ax.plot(self.xexp,yres*scale_residual+shift_residual,zorder=-1,**tkwargs) # plot fit residual
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
                except InferException:
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
                    labelsize='small',namesize='medium', namepos='float',    
                    label_key=(line.default_xkey if label_key == 'default' else label_key),
                )
                ymax += ystep*(len(branch_annotations)+1)
        # ## plot branch heads
        # if plot_branch_heads:
            # for line in self.suboptimisers:
                # if not isinstance(line,lines.Generic):
                    # continue
                # annotate_branch_heads(line,qn_defining_branch,match_branch_re=label_match_name_re)
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
            plotting.legend_colored_text(loc='upper left')
        if xlim is None:
            xlim = (xmin,xmax)
        if ylim is None:
            ylim = (ymin,ymax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
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
    header['sinc_fwhm'] = 1.2*header['interpolation_factor']*header['ds'] 
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

    def __init__(
            self,
            name='spline',
            xs=None,
            ys=None,
            knots=None,         # [(xsi,ysi),..] pairs
            order=3,
    ):
        optimise.Optimiser.__init__(self,name)
        self.clear_format_input_functions()
        self.automatic_format_input_function()
        self.order = order
        ## spline domain and value
        self.x = None
        self.y = None
        self.xs = []
        self.ys = []
        ## set knots
        if xs is not None:
            self.set_knots(xs,ys)
        if knots is not None:
            self.set_knots([t[0] for t in knots],[t[1] for t in knots])
            
    @optimise_method()
    def add_knots(self,x,y):
        """Add one spline knot."""
        self.xs.append(x)
        self.ys.append(y)
        
    @optimise_method()
    def set_knots(self,xs,ys):
        """Set knots to multiple."""
        self.xs.clear()
        self.ys.clear()
        self.xs.extend(xs)
        self.ys.extend(ys)

    def __call__(self,x):
        """Compute value of spline at x. And store in self."""
        xs = np.array([float(t) for t in self.xs])
        ys = np.array([float(t) for t in self.ys])
        y = tools.spline(xs,ys,x,order=self.order)
        self.x = x
        self.y = y
        return y

def load_spectrum(filename,**kwargs):
    """Use a heuristic method to load a directory output by
    Spectrum."""
    x = Spectrum()
    x.load_from_directory(filename,**kwargs)
    return(x)

class Spectrum(Dataset):
    """A spectrum of some kind."""

    default_xkeys = 'x'
    default_ykeys = 'y'
    default_zkeys = ()
    default_prototypes = {
        'x':{'description':'x-scale'          , 'kind':'f' , 'fmt':'0.8f' , 'infer':[]} , 
        'y':{'description':'y-scale'          , 'kind':'f' , 'fmt':'0.8f' , 'infer':[]} , 
        'ν':{'description':'Wavenumber'       , 'units':'cm-1' , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'λ':{'description':'Wavelength scale' , 'units':'nm'   , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'f':{'description':'Frequency scale'  , 'units':'MHz'  , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'σ':{'description':'Cross section'    , 'units':'cm2'  , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'τ':{'description':'Optical depth'    , 'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
        'T':{'description':'Transmittance'    , 'kind':'a' , 'fmt':'0.8f' , 'infer':[
            (('σ','N'),lambda self,σ,N: np.exp(-N*σ))
        ]} , 
        'I':{'description':'Intensity'        ,  'kind':'a' , 'fmt':'0.8f' , 'infer':[]} , 
    }



class FitAbsorption():

    """Fit molecular absorption to an experimental spectrum. All data is
    stored in the "parameters" attribute which has valid values.  Most
    numerical values can be included replaced with an optimisation
    parameter, e.g., xscale=1 for a fixed xscaling or xscal=P(1,True)
    to fit this value.

    Values that may appear in parameters dict:

    (a/b is a subdict, e.g., parameters['a']['b'])

     - 'filename': path to experimental data (required)
     - 'xbeg': experimental from this value (defaults to beginning of file)
     - 'xend': experimental to this value (defaults to endof file)
     - 'scalex': scale x-axis of experimental data by this amount
     - 'noise'/'rms': noise rms, used to compute uncertainties,
                      default value is fitted to the last 20cm-1 of the experimental
                      spectrum (requiring no absorption in this region to be
                      accurate, or be settiung noise/xbeg noise/xend)
     - 'interpolate_model': Interpolate to at least this grid (cm-1)
     - 'Teq': Equiblibrium temperature
     - 'N'/species: Column density of species by name (cm-2)
     - 'pair'/species: Effect air pressure for broadeing of species by name (Pa)
     - 'intensity'/'spline': Spline points defining background intensity
                             [[x0,y0],[x1,y1],...]
     - 'intensity'/'spline_step': Separation of spline grid for fitting background 
                                  intensity if 'intensity'/'spline' is not present.
     - 'intensity'/'spline_order': Order of spline
     - 'sinusoid'/'spline': Spline points defining background sinusoid intensity
                            [[x0,amplitude0,frequency0,phase0],...]
     - 'sinusoid'/'spline_step': Separation of spline grid for fitting sinusoid 
                                 intensity if 'intensity'/'sinusoid' is not present.
     - 'instrument': Parameters controlling the instrumental lineshape.
                              These are the arguments of convolve_with_instrument_function
                              and will be deduced automatically (hopefully) if
                              not present.
     - 'FTS_H2O': Parameters controlling extra H2O absorption in from water in the evacuated FTS.
     - 'nfwhmL':  compute this many Lorentizan full-width khalf-maximums
     - 'min_S296K':  include lines from fitted species with this linestrength or more
    

"""

    def __init__(
            self,
            name='fit_absorption',
            parameters=None,    # pass in values to parameters
            verbose= True,      # print more information for debugging
            # make_plot=False,     # plot results of every fit
            ncpus=1,            # compute Voigt spectrum with 1 or more cpus
            default_species=None,
            **more_parameters   # some other values in parameters as kwargs
    ):
        self.name = name
        ## provided parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        ## update with kwargs
        self.parameters |= more_parameters
        ## initialise control variables
        self.verbose = verbose
        ## the Experimebnt object
        self.experiment = None
        ## fit_* methods save Model objects and their reference
        ## versions (without fit_species absorption)
        self.models = []
        ## if no species_to_fit species in fit_species then fit all
        ## these
        self.default_species = default_species
        if self.default_species is None:
            self.default_species = [
                'H2O', 'CO2', 'CO', 'NO2', 'NO', 'N2O', 'NH3', 'HCN',
                'CH4', 'C2H2', 'SO2', 'H2S', 'CS2', 'OCS',
            ]

    def __getitem__(self,key):
        """Access parameters directly."""
        return self.parameters[key]

    def __setitem__(self,key,value):
        """Access parameters directly. If multiple keys, e.g.,
        ('species','CO2','N') then add in the parameters dictionary
        hierarchy, making subdictionarys as needed."""
        if isinstance(key,str):
            self.parameters[key] = value
        elif isinstance(key,tuple) and len(key) > 0:
            d = self.parameters
            for tkey in key[:-1]:
                d.setdefault(tkey,{})
                d = d[tkey]
            d[key[-1]] = value
        else:
            raise KeyError(f'Key must be string or tuple of strings.')

    def keys(self):
        """Parameters keys as list."""
        return list(self.parameters.keys())

    def pop(self,key):
        """Remove key from parameters if it is set."""
        if key in self.parameters:
            self.parameters.pop(key)

    def __str__(self):
        """Summarise parameters."""
        ## print full parameters dict
        retval = f'== Summary of FitAbsorption object {self.name!r} =='
        retval += '\n\nAll parameters:\n'
        retval += tools.format_dict(self.parameters, newline_depth=3, max_line_length=1000,)
        retval += '\n\nFitted species:\n'
        retval += str(self.get_species_dataset())
        return retval

    def fit(
            self,
            species_to_fit=(), # 'default','existing', or a list of species names
            other_species='existing', # 'existing', or a list of species names
            region='full', # 'lines','bands','full', a region [xbeg,xend], or a list of regions [[xbeg0,xend0],[xbeg1,xend1],...]'
            subregion=None,     # only fit one or more subregions of region, e.g., [xbeg0,xend0], or [[xbeg0,xend0],[xbeg1,xend1],...], or an interval Δx dividing the region into equal sized pieces
            ## plot options
            fig=None,           # if not None then plot in this figure window
            plot_legend=False,
            ## optimisation keywords
            max_nfev=100,         
            ftol=1e-6,
            xtol=1e-8,
            calculate_uncertainty=True,
            ## make_model keywords controlling what to fit
            **make_model_kwargs,
    ):
        """Fit list of species individually from one another using their
        preset 'lines' or 'bands' regions, or the full region."""
        ## set all Parameters to vary=False
        p = self.parameters
        optimise.set_contained_parameters(p,vary=False)
        ## gather species_to_fit list
        if species_to_fit == 'default':
            species_to_fit = self.default_species
        if species_to_fit == 'existing':
            p.setdefault('species',{})
            species_to_fit = list(p['species'])
        species_to_fit = tools.ensure_iterable(species_to_fit)
        if other_species == 'existing':
            p.setdefault('species',{})
            other_species = list(p['species'])
        other_species = [t for t in
                         tools.ensure_iterable(other_species)
                         if t not in species_to_fit]
        ## gather regions list
        if region == 'lines':
            region = []
            for tspecies in species_to_fit:
                try:
                    region += database.get_species_property(tspecies,'characteristic_infrared_lines')
                except DatabaseException:
                    raise Exception(f'No characteristic lines found for {tspecies}. You could add a \'characteristic_infrared_lines property\' to a {database.normalise_species(tspecies)!r} section in spectr/data/species_data.py')
        elif region == 'bands':
            region = []
            for tspecies in species_to_fit:
                try:
                    region += database.get_species_property(tspecies,'characteristic_infrared_bands')
                except DatabaseException:
                    raise Exception(f'No characteristic bands found for {tspecies}. You could add a \'characteristic_infrared_bands property\' to a {database.normalise_species(tspecies)!r} section in spectr/data/species_data.py')
        elif region == 'full':
            self.make_model(0,0,verbose=False)
            region = ((int(np.floor(p['xbeg'])), int(np.ceil(p['xend']))),)
        elif isinstance(region[0],(float,int)):
            region = [region]
        main = Optimiser(name="_".join(list(species_to_fit)+["main"]))
        ## initialise optimiser
        optimise_kwargs = {
            'make_plot':False,
            'verbose':self.verbose,
            'max_nfev':max_nfev,
            'ftol':ftol,
            'xtol':xtol,
            'calculate_uncertainty':calculate_uncertainty,
        }
        self.models = []
        self.subregion_models = []
        ## things are more complicated if subregions are defined
        if subregion is None:
            ## compute and optimise a model and reference model for
            ## each region
            for xbeg,xend in region:
                model = self.make_model(
                    xbeg,xend,
                    species_to_fit=species_to_fit,
                    other_species=other_species,
                    **make_model_kwargs)
                if model is not None:
                    self.models.append(model)
                    main.add_suboptimiser(model)
            ## optimise
            residual = main.optimise(**optimise_kwargs)
        else:
            ## loop through defined regions and their subreiongs,
            ## optimising them individually
            for xbeg,xend in region:
                ## collect subregions: scalar is translated into a
                ## linspace interval, a pair is a single subregion, a
                ## list of pairs for multiple subregions
                if np.isscalar(subregion):
                    x = linspace(xbeg,xend,int(np.ceil((xend-xbeg)/(subregion/2)))+1)
                    x[:-1] = [int(np.floor(t)) for t in x[:-1]]
                    x[-1] = int(np.ceil(x[-1]))
                    subregions = [(x[i],x[i+2]) for i in range(len(x)-2)]
                elif np.isscalar(subregion[0]):
                    subregions = [subregion]
                else:
                    subregions = subregion
                ## optimise a model for each subregion individually
                for isubregion,(subregion_xbeg,subregion_xend) in enumerate(subregions):
                    # print(f'subregion {isubregion+1} of {len(subregions)}: {(subregion_xbeg,subregion_xend)}')
                    model = self.make_model(
                        subregion_xbeg,subregion_xend,
                        species_to_fit=species_to_fit,
                        other_species=other_species,
                        region=(xbeg,xend),
                        **make_model_kwargs)
                    model._is_subregion = True
                    self.subregion_models.append(model)
                    model.optimise(**optimise_kwargs)
        ## make reference models
        if len(species_to_fit) > 0:
            for model in self.models:
                model._reference_model = self.make_model(
                    model.xbeg,model.xend,
                    other_species=other_species,
                    verbose=False,)
                model._reference_model_neglected_species = species_to_fit
        ## plot after fitting
        if fig is not None:
            self.plot(fig,plot_legend=plot_legend)

    def plot(
            self,
            fig=None,
            scale_residual=1,
            plot_legend=False,
            **plot_kwargs,
    ):
        """Plot the results of some models on individual subplots. Residuals
        from models_no_absorption will be underplotted."""
        ## default plot style
        plot_kwargs = {
            'plot_legend': plot_legend,
            'plot_title': True,
            'plot_text': True,
            'plot_kwargs':{'linewidth': 1},
        } | plot_kwargs
        ## get figure
        if fig is None:
            ## current figure
            fig = plotting.gcf()
        elif isinstance(fig,int):
            ## integer -- make a new figure with this number
            fig = plotting.qfig(fig)
        else:
            ## a figure object already
            pass
        fig.clf()
        ## plot subregion models
        for model in self.subregion_models:
            ax = plotting.subplot(fig=fig)
            model.plot(ax=ax,scale_residual=scale_residual,**plot_kwargs)
        ## plot each model in self.models
        for model in self.models:
            ## new subplot
            ax = plotting.subplot(fig=fig)
            model.plot(ax=ax,scale_residual=scale_residual,**plot_kwargs)
            ## show the fitted intensity and shift if this is not a
            ## subregion model
            pregion = self.parameters['region'][model.xbeg,model.xend]
            x = model.x
            if model._intensity_type == 'spline':
                background_intensity = tools.spline(
                    [t[0] for t in pregion['intensity']['spline']],
                    [t[1] for t in pregion['intensity']['spline']],
                    x,
                    order=pregion['intensity']['spline_order'])
                if 'shift' in pregion:
                    background_shift = tools.spline(
                        [t[0] for t in pregion['shift']['spline']],
                        [t[1] for t in pregion['shift']['spline']],
                        x,
                        order=pregion['shift']['spline_order'])
                else:
                    background_shift = np.full(len(x),0.0)
                ax.plot(x,background_intensity+background_shift,color='black',zorder=-5,label='source intensity')
                ax.plot(x,background_shift,color=plotting.newcolor(7),zorder=-5,label='intensity shift')
            ## plot reference line without fit_species absorption
            if hasattr(model,'_reference_model'):
                ax.plot(
                    model._reference_model.x,
                    model._reference_model.get_residual()*scale_residual,
                    color='orange',zorder=-2,
                    label='residual neglecting fitted species',
                )
            ## finalise axes
            if plot_legend:
                plotting.legend(ax=ax)
        ## finalise plot
        fig.suptitle(self.name)
        return fig
        

    def make_model(
            self,
            xbeg, xend,   # range of model, may be a subset of region
            species_to_fit=(), # species for optimisation, None, a name, or a list of names
            other_species=(), # species to model but not fit, defaults to existing list, always includes 'species'
            region=None,       #  key to 'region' parameters if different from (xbeg,xend)
            fit_intensity=False,         # fit background intensity spline, True, False
            intensity_spline_step=100,    # separation of points in the background intensity spline 
            fit_shift=False,             # fit baseline shift
            fit_N=False,                 # fit species column densities
            fit_pair=False,              # fit species pressure broadening (air coefficients)
            fit_scalex=False,            # fit uniformly scaled frequency  
            fit_sinusoid=False,          # fit sinusoidally varying intensity
            sinusoid_spline_step=50,     # separation of points in the spline describing the sinusoidal variation
            fit_instrument=False,        # fit instrumental broadening
            fit_temperature=False,       # fit excitation/Doppler temperature
            fit_FTS_H2O=False,           # fit column density and air-broadening coefficient to H2O in the spectrometer
            verbose=None,                # print more info for debugging
    ):
        """Make a model for fitting to the experimental spectrum.  Probably
        not called directly, instead used by various fit* methods."""
        if verbose is None:
            verbose = self.verbose
        ## get parameters — protect from changes if necessary
        p = self.parameters
        ## load experiment
        if self.experiment is None:
            ## load exp data
            self.experiment = Experiment('experiment')
            ## use provided x range or entire spectrum
            p.setdefault('xbeg',None)
            p.setdefault('xend',None)
            self.experiment.set_spectrum_from_opus_file(p['filename'],xbeg=p['xbeg'],xend=p['xend'])
            p['xbeg'] = self.experiment.x[0]
            p['xend'] = self.experiment.x[-1]
            ## scale x coordinate
            p.setdefault('scalex',P(1,False,1e-9))
            self.experiment.scalex(p['scalex'])
            ## fit noise level
            p.setdefault('noise',{})
            p['noise'].setdefault('xbeg',self.experiment.x[-1]-20)
            p['noise'].setdefault('xend',self.experiment.x[-1])
            p['noise'].setdefault('rms',None)
            if p['noise']['rms'] is None:
                self.experiment.fit_noise(xbeg=p['noise']['xbeg'],xend=p['noise']['xend'],n=3,make_plot=False)
                p['noise']['rms'] = self.experiment.experimental_parameters['noise_rms']
            self.experiment.experimental_parameters['noise_rms'] = p['noise']['rms']
            self.experiment.residual_scale_factor = 1/p['noise']['rms']
        ## get region, xbeg, xend
        p.setdefault('region',{})
        if region is None:
            region = (xbeg,xend)
        p['region'].setdefault(region,{})
        pregion = p['region'][region]
        xbeg = max(xbeg,region[0])
        xend = min(xend,region[1])
        ## whether to adjust experiment frequency scale
        p['scalex'].vary = fit_scalex
        ## start model
        model = Model(
            name='_'.join(['model']+list(species_to_fit)+[str(t) for t in region]),
            experiment=self.experiment,xbeg=xbeg,xend=xend
        )
        if verbose:
            if (xbeg,xend) == region:
                print(f'make_model: {model.name}')
            else:
                print(f'make_model: {model.name} subregion from {xbeg} to {xend}')
        ## if no experimental data for this region then immediately
        if len(model.x) == 0:
            return None
        ## set interpolated model grid
        self.parameters.setdefault('interpolate_model',0.001)
        if p['interpolate_model'] is not None:
            model.interpolate(p['interpolate_model'])
        ## add reference background spectrum if provided or default to
        ## unity
        if 'background' in p and 'filename' in p['background']:
            model._intensity_type = 'file'
            p['background'].setdefault('yscale',P(1,False,1e-4))
            p['background']['yscale'].vary = fit_intensity
            model.add_from_file(p['background']['filename'], yscale=p['background']['yscale'])
        else:            
            model._intensity_type = 'spline'
            model.add_constant(1)
        ## add absorption lines
        p.setdefault('Teq',P(296,False,1,nan,(20,1000)))
        p['Teq'].vary = fit_temperature
        p.setdefault('nfwhmL',100) # compute this many Lorentizan full-width half-maximums
        p.setdefault('min_S296K',1e-25) # include lines from fitted species with this linestrength or more
        p.setdefault('species',{})
        all_species = set(species_to_fit) | set(other_species)
        for speciesi in all_species:
            p['species'].setdefault(speciesi,{})
            pspecies = p['species'][speciesi]
            ## load column desnity
            pspecies.setdefault('N',P(1e16, False,1e13 ,nan,(0,np.inf)))
            pspecies['N'].vary = (speciesi in species_to_fit and fit_N)
            if 'filename' in pspecies:
                ## add absorption from a cross section file
                hitran_cross_section = hitran.load_cross_section(pspecies['filename'])
                model.add_spectrum(
                    hitran_cross_section,
                    kind='absorption',
                    N = pspecies['N'])
            else:
                ## add absorption from a linelist.
                ##
                ## load effective air-broadening pressure
                pspecies.setdefault('pair',P(1e3, False,1e0,nan,(1e-3,1.2e5),))
                pspecies['pair'].vary = (speciesi in species_to_fit and fit_pair)
                ## add lines
                model.add_hitran_line(
                    speciesi,
                    Teq=p['Teq'],
                    Nchemical_species=pspecies['N'],
                    pair=pspecies['pair'],
                    min_S296K=p['min_S296K'],
                    ncpus=1,
                    nfwhmL=p['nfwhmL'],
                    lineshape='voigt',)
        ## fit column density and air-broadening coefficient to H2O in the spectrometer
        if fit_FTS_H2O:
            p.setdefault('FTS_H2O',{})
            p['FTS_H2O'].setdefault('N',P(1e16,False,1e15,bounds=(0,1e22)))
            p['FTS_H2O'].setdefault('pair',P(100,False,1,bounds=(1,1000)))
            p['FTS_H2O'].setdefault('Teq',296)
        if 'FTS_H2O' in p:
            p['FTS_H2O']['N'].vary = fit_FTS_H2O
            p['FTS_H2O']['pair'].vary = fit_FTS_H2O
            model.add_hitran_line(
                'H₂O',
                Teq=p['FTS_H2O']['Teq'],
                Nchemical_species=p['FTS_H2O']['N'],
                pair=p['FTS_H2O']['pair'],
                min_S296K=p['min_S296K'],
                ncpus=1,
                nfwhmL=p['nfwhmL'],
                lineshape='voigt',)
        ## uninterpolate model grid
        if p['interpolate_model'] is not None:
            model.uninterpolate(average=True)
        ## scale to correct background intensity — vary points in range and neighbouring
        ## fit background if needed
        if model._intensity_type == 'spline':
            pregion.setdefault('intensity',{})
            pregion['intensity'].setdefault('spline_step',intensity_spline_step)
            pregion['intensity'].setdefault('spline_order',3)
            if 'spline' not in pregion['intensity']:
                i = tools.inrange(self.experiment.x,*region)
                xspline,yspline,t = tools.fit_spline_to_extrema_or_median(
                    self.experiment.x[i],
                    self.experiment.y[i],
                    xi=pregion['intensity']['spline_step'])
                pregion['intensity']['spline'] = [
                    [tx,P(ty,False,1e-5)] for tx,ty in zip(xspline,yspline)]
            optimise.set_contained_parameters(pregion['intensity']['spline'],vary=False)
            pspline = pregion['intensity']['spline']
            if fit_intensity:
                xspline = [t[0] for t in pspline]
                for i in tools.find(
                        tools.inrange(
                            xspline,xbeg,xend,
                            include_adjacent=True)):
                    for parameter in pspline[i][1:]:
                        parameter.vary = fit_intensity
            model.multiply_spline(knots=pspline,order=pregion['intensity']['spline_order'])
        ## shift entires spectrum -- light leakage or inteferometry
        ## problem that adds signal and shifts zero baseline
        if fit_shift: 
            pregion.setdefault('shift',{})
        if 'shift' in pregion:
            pregion['shift'].setdefault('spline_step',500)
            pregion['shift'].setdefault('spline_order',3)
            if 'spline' not in pregion['shift']:
                xspline = linspace(
                    xbeg,xend,
                    int((xend-xbeg)/pregion['shift']['spline_step'])+2)
                pregion['shift']['spline'] = [[tx,P(0,False,1e-7)] for tx in xspline]
            pspline = pregion['shift']['spline']
            optimise.set_contained_parameters(pspline,vary=False)
            for i in tools.find(tools.inrange(
                    [t[0] for t in pspline],xbeg,xend,include_adjacent=True)):
                for parameter in pspline[i][1:]:
                    parameter.vary = fit_shift
            model.add_spline(knots=pspline,order=pregion['shift']['spline_order'])
        ## instrument function
        if 'instrument' not in p:
            p['instrument'] = model.auto_convolve_with_instrument_function(vary=fit_instrument)
        else:
            for key,val in p['instrument'].items():
                if isinstance(val,Parameter):
                    val.vary = fit_instrument
                    val.bounds = (1e-4,1)  
            model.convolve_with_instrument_function(**p['instrument'])
        ## Scale by sinusoidally-varying background with slowly
        ## changing parameters. Put this last because it requires a
        ## very good residual fit to the rest of the model.  
        if fit_sinusoid:
            pregion.setdefault('sinusoid',{})
        if 'sinusoid' in pregion:
            pregion['sinusoid'].setdefault('spline_step',sinusoid_spline_step)
            pregion['sinusoid'].setdefault(
                'spline',
                model.auto_scale_piecewise_sinusoid(
                    xi=pregion['sinusoid']['spline_step'],
                    vary=False,add_construct=False,))
            pspline = pregion['sinusoid']['spline']
            optimise.set_contained_parameters(pspline,vary=False)
            for i in tools.find(
                    (array([t[0] for t in pspline]) < xend)
                    & (array([t[1] for t in pspline]) > xbeg)):
                for parameter in pspline[i][2:]:
                    parameter.vary = fit_sinusoid
            model.scale_piecewise_sinusoid(regions=pspline)
        ## return Model object
        return model
        
    def get_species_dataset(self):
        """Get a dataset listing the parameters of all fitted
        species."""
        ## load parameteres into a dataset
        self.parameters.setdefault('species',{})
        dataset = Dataset(name=f'species_in_{self.name}')
        dataset.description = f'''Parameters defining species in the FitAbsorption object {self.name!r}.'''
        if 'filename' in self.parameters:
            dataset.attributes['filename'] = self.parameters['filename']
        dataset.set_prototype('species',description=f'Name of species',units='species_unicode',kind='U')
        dataset.set_prototype('N',description=f'Column density',units='cm-2',kind='f',fmt='0.3e')
        dataset.set_prototype('pair',description=f'Effective air-broadening pressure',units='Pa',kind='f',fmt='0.3e')
        dataset.load_from_parameters_dict(self.parameters['species'], index_key='species',)
        dataset.sort_keys(('species',))
        return dataset

    def save_species_dataset(self,filename):
        """Save a dataset containing species data to a file."""
        dataset = self.get_species_dataset()
        dataset.save(filename)
        
    def load_parameters(self,filename):
        """Load parameters from a file."""
        self.parameters = tools.load_dict(filename,'parameters')
        
    def save_parameters(self,filename):
        """Save parameters to a file."""
        tools.save_dict(filename,parameters=self.parameters)

    def save_models(self, directory,):
        """Save parameters and experimental and model spectrum to a
        directory."""
        tools.mkdir(directory)
        ## get uniquified model names
        model_names = tools.uniquify_strings(
            [model.name for model in self.models])
        for model,name in zip(self.models,model_names):
            d = Dataset(description=f'Output of model {model.name!r} of FitAbsorption {self.name!r}')
            d.set_new('x',model.x,kind='f',fmt='0.6f',units='cm-1',description='Wavenumber scale')
            d.set_new('ymod',model.y,kind='f',fmt='0.4e',description='Model intensity')
            d.set_new('yexp',model.yexp,kind='f',fmt='0.4e',description='Experimental intensity')
            d.set_new('yres',model.get_residual(),kind='f',fmt='+0.4e',description='Residual intensity error, model-experimental')
            if hasattr(model,'_reference_model'):
                d.set_new(
                    'yref',
                    model.yexp-model._reference_model.y,
                    kind='f',fmt='+0.4e',
                          description=f'Residual intensity error of a model neglecting fitted species: {", ".join(model._reference_model_neglected_species)}')
            d.save(f'{directory}/{name}',filetype='directory')

    def save(self,directory,trash_existing=True):
        """Save parameters and experimental and model spectrum to a
        directory."""
        tools.mkdir(directory,trash_existing=trash_existing)
        self.save_parameters(f'{directory}/parameters.py')
        self.save_species_dataset(f'{directory}/species.psv')
        self.save_models(f'{directory}/model')

def fit_species_absorption(
        filename,               # experimental spectrum
        *species,               # species, e.g., "HCN", or "H2O", or "¹³C¹⁶O₂"
        **fit_kwargs            # passed to Fit_Absorption.fit
):
    """Shortuct to fit one or more species."""
    ## assemble kwargs
    kwargs = {
        'species_to_fit': species,
        'region':'bands',
        'fig':1,
        'fit_N': True,
        'fit_pair': True,
        'fit_intensity': True
    }
    kwargs |= fit_kwargs
    ## load and fit spectrum
    o = FitAbsorption(
        filename=filename,
        min_S296K=0,
    )
    o.fit(**kwargs)
    ## print results
    d = o.get_species_dataset()
    print('\nResults:\n')
    print(d.format())
    ## raise plot
    plotting.show()
    ## return FitAbsorption object
    return o

def collect_fit_absorption_results(parameters):
    """Turn a dictionary of FitAbsorption parameters into a Dataset."""
    ## load parameteres into a dataset
    data = Dataset()
    data.set_prototype('filename',description='Data filename',kind='U')
    data.set_prototype('xbeg',description='Lowest frequency in fitted region',kind='f',units='cm-1',fmt='0.0f')
    data.set_prototype('xend',description='Highest frequency in fitted region',kind='f',units='cm-1',fmt='0.0f')
    data.set_prototype('scalex',description='Rescale all frequencies',kind='f',fmt='0.10f')
    data.set_prototype('noise_xbeg',description='Beginning of frequency range for assessing noise level',kind='f',units='cm-1',fmt='0.0f')
    data.set_prototype('noise_xend',description='End of frequency range for assessing noise level',kind='f',units='cm-1',fmt='0.0f')
    data.set_prototype('noise_rms',description='Estimated root-mean-square noise level',kind='f',fmt='0.3e')
    data.set_prototype('interpolate_model',description='Approximate model grid spacing',kind='f',fmt='0.2e')
    data.set_prototype('Teq',description='Equilbrium temperature',kind='f',units='K',fmt='0.2f')
    data.set_prototype('nfwhmL',description='How many Lorentzian full-width half-maximums are used when computing absorption lineshapes',kind='f',fmt='g')
    data.set_prototype('min_S296K',description='Minimum absorption linestrength for including lines in the model.',units='cm2',kind='f',fmt='0.1e')
    data.set_prototype('key',description='Experimental scan identifier.',kind='U')
    data.load_from_parameters_dict(parameters)
    ## change keys and set metadata
    for key in list(data):
        ## species column densities and broadening pressures
        if r:=re.match(r'^species_([^_]+)_N',key):
            data.modify_key(key, rename=f'N_{r.group(1)}',
                            description=f'Column density of {r.group(1)}',
                            units='cm-2', kind='f',fmt='0.4e')
        if r:=re.match(r'^species_([^_]+)_pair',key):
            data.modify_key(key, rename=f'pair_{r.group(1)}',
                            description=f'Effective pressure for air broadening of {r.group(1)}',
                            units='Pa', kind='f',fmt='0.3e')
        if key == 'instrument_sinc_fwhm':
            data.modify_key(key, rename=f'sinc_fwhm',
                            description=f'Instrument function sinc',
                            units='cm-1.FWHM', kind='f',fmt='0.6f')
    ## remove some unimportant parameters
    for key in ('interpolate_model', 'intensity_spline_step', 'intensity_spline_order',):
        if data.is_set(key):
            data.unset(key)
    ## sort keys
    data.sort_keys(('key','filename',))
    return data    

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

