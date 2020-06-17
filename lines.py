import itertools
from copy import copy
from pprint import pprint

import numpy as np

from spectr.dataset import Dataset
from spectr.levels import Levels
from spectr import lineshapes
from spectr import tools

def expand_level_keys(level_class):
    retval = {}
    for key,val in level_class._prototypes.items():
        retval[key+'p'] = copy(val)
        retval[key+'pp'] = copy(val)
    return(retval)


class Lines(Dataset):
    """For now rotational lines."""

    _prototypes = {
        'class':{'description':"What kind of data this is.",'kind':'str',},
        'levels_class':{'description':"What kind of level this is a transition between.",'kind':'object','infer':{():lambda: Levels,}},
        'description':{'kind':str,'description':"",},
        'notes':{'description':"Notes regarding this line.", 'kind':str, },
        'author':{'description':"Author of data or printed file", 'kind':str, },
        'reference':{'description':"", 'kind':str, },
        'date':{'description':"Date data collected or printed", 'kind':str, },
        'branch':dict(description="Rotational branch ΔJ.Fp.Fpp.efp.efpp", dtype='8U', cast=str, fmt='<10s'),
        'ν':dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>13.6f', infer={}),
        'Γ':dict(description="Natural linewidth of transition (cm-1 FWHM)",kind=float,fmt='<10.5g',infer={('Γp','Γpp'):lambda Γp,Γpp: Γp+Γpp},),
        'ΓD':dict(description="Gaussian Doppler width (cm-1 FWHM)",kind=float,fmt='<10.5g', infer={}),
        'f':dict(description="Line f-value (dimensionless)",kind=float,fmt='<10.5e',infer={}),

    }
    _prototypes.update(expand_level_keys(Levels))
    _prototypes['ν']['infer']['Ep','Epp'] = lambda Ep,Epp: Ep-Epp
    _prototypes['Epp']['infer']['Ep','ν'] = lambda Ep,ν: Ep-ν
    _prototypes['Ep']['infer']['Epp','ν'] = lambda Epp,ν: Epp+ν

    def __init__(
            self,
            name=None,
            **keys_vals,
    ):
        Dataset.__init__(self)
        self.permit_nonprototyped_data = False
        self['class'] = type(self).__name__
        self.name = (name if name is not None else self['class'])
        for key,val in keys_vals.items():
            self[key] = val

    def calculate_spectrum(
            self,
            x=None,        # frequency grid (must be regular, I think), if None then construct a reasonable grid
            xkey='ν',      # strength to use, i.e., "ν", or "λ"
            ykey='σ',      # strength to use, i.e., "σ", "τ", or "I"
            ΓG='ΓD', # a key to use for Gaussian widths (i.e., "ΓDoppler"), a constant numeric value, or None to neglect Gaussian entirely
            ΓL='Γ',        # a key or for Lorentzian widths (i.e., "Γ"), a constant numeric value, or None to neglect Lorentzian entirely
            nfwhmG=20,         # how many Gaussian FWHMs to include in convolution
            nfwhmL=100,         # how many Lorentzian FWHMs to compute
            nx=10000,     # number of grid points used if ν not give
            ymin=None,     # minimum value of ykey before a line is ignored, None for use all lines
            gaussian_method='fortran', #'fortran stepwise', 'fortran', 'python'
            voigt_method='wofz',   
            # temperaturepp=None,
            # column_densitypp=None,
            use_multiprocessing=False, # might see a speed up
            use_cache=False,    # is it actually any faster?!?
            # ycache = None,
    ):
        """Calculate a Voigt/Lorentzian/Gaussian spectrum from data in self. Returns (x,σ)."""
        ## save input arguments in their original from
        ΓGin,ΓLin = ΓG,ΓL
        ## all input args except use_cache
        all_args = dict(x=x, xkey=xkey, ykey=ykey, ΓG=ΓG, ΓL=ΓL,
                        nfwhmG=nfwhmG, nfwhmL=nfwhmL, nx=nx, ymin=ymin, voigt_method=voigt_method,
                        use_multiprocessing=use_multiprocessing,
                        # temperaturepp=temperaturepp,
                        # column_densitypp=column_densitypp,
        )        
        ## no lines to add to cross section -- return quickly
        if len(self)==0:
            if x is None:
                return(np.array([]),np.array([]))
            else:
                return(x,np.zeros(x.shape))
        ## check this early to avoid problems in cache
        self.assert_known(xkey,ykey)
        if ΓG is None:
            pass
        elif isinstance(ΓG,str):
            ## if a key is given load it, if it is not known then proceed anyway
            if self.is_known(ΓG):
                ΓG = self[ΓG]
            else:
                ΓG = None
        elif np.isscalar:
            ΓG = np.full(len(self),ΓG,dtype=float)
        else:
            pass                # assumes already an array of the correct length and type
        if ΓL is None:
            pass
        elif isinstance(ΓL,str):
            ## if a key is given load it, if it is not known then proceed anyway
            if self.is_known(ΓL):
                ΓL = self[ΓL]
            else:
                ΓL = None
        elif np.isscalar:
            ΓL = np.full(len(self),ΓL,dtype=float)
        else:
            pass                # assumes already an array of the correct length and type
        ## test for appropriate use of cache
        if use_cache:
            for test,msg in (
                    (voigt_method=='wofz',('Cache only implemented for wofz.')),
                    (ΓG is not None,'Cache only implemented for given ΓG not None.'),
                     (ΓL is not None,'Cache only implemented for given ΓL not None.'),
            ):
                if not test:
                    warnings.warn(f'{self.name}: {msg}')
                    use_cache = False
        ## establish which data should be stored in cache and load
        ## cache if it exists
        if use_cache:
            cache_keys = (xkey,ykey,ΓLin,ΓGin)
            if 'calculate_spectrum' in self._cache:
                cache = self._cache['calculate_spectrum']
            else:
                cache = None
        ## compute spectrum not using cache, either do do not use this is the first computation
        if  (not use_cache) or cache is None:
            ## get a default frequency scale if none provided
            if x is None:
                x = np.linspace(max(0,self[xkey].min()-10.),self[xkey].max()+10.,nx)
            elif np.isscalar(x):
                x = np.arange(max(0,self[xkey].min()-10.),self[xkey].max()+10.,x)
            ## get spectrum type according to width specified
            ##
            ## divide between neighbouring points
            if ΓL is None and ΓG is None:
                y = lineshapes.centroided_spectrum(x,self[xkey],self[ykey],Smin=ymin)
            ## Spectrum of pure Lorentzians
            elif ΓL is not None and ΓG is None:
                y = linekshapes.lorentzian_spectrum(x,self[xkey],self[ykey],ΓL,nfwhm=nfwhmL,Smin=ymin)
            ## Spectrum of pure Gaussians if ΓL is explicitly None, or
            ## the key it specifies is not known
            elif ΓG is not None and ΓL is None:
                y = lineshapes.gaussian_spectrum(x, self[xkey], self[ykey], ΓG, nfwhm=nfwhmG,Smin=ymin,method=gaussian_method)
            ## Spectrum of Voigts computed with widths for each line
            elif ΓL is not None and ΓG is not None:
                if voigt_method=='wofz':
                    y = lineshapes.voigt_spectrum(
                        x,self[xkey],self[ykey],ΓL,ΓG,
                        nfwhmL,nfwhmG,Smin=ymin, use_multiprocessing=use_multiprocessing)
                ## spectrum of Voigts with common mass/temperature lines
                ## computed in groups with fortran code
                elif voigt_method=='fortran Doppler' and ΓLin=='Γ' and ΓGin=='ΓDoppler':
                    ## Spectrum of Voigts.
                    if use_multiprocessing and len(self)>100: # multprocess if requested, and there are enough lines to make it worthwhile
                        import multiprocessing
                        p = multiprocessing.Pool()
                        y = []
                        def handle_result(result):
                            y.append(result)
                        number_of_processes_per_mass_temperature_combination = 6 # if there are multiple temperature/mass combinations there will more be more processes, with an associated memory danger
                        self.assert_known(xkey,ykey,'Γ')
                        for d in self.unique_dicts('masspp','TDopplerpp'):
                            m = self[(xkey,ykey,'Γ')][self.match(**d)] # get relevant data as a recarrat -- faster than using unique_dicts_matches
                            ibeg,istep = 0,int(len(m)/number_of_processes_per_mass_temperature_combination)
                            while ibeg<len(m): # loop through lines in chunks, starting subprocesses
                                iend = min(ibeg+istep,len(m))
                                t = p.apply_async(lineshapes.voigt_spectrum_with_gaussian_doppler_width,
                                                  args=(x,m[xkey][ibeg:iend],m[ykey][ibeg:iend],m['Γ'][ibeg:iend],
                                                        d['masspp'],d['TDopplerpp'], nfwhmL,nfwhmG,ymin),
                                                  callback=handle_result)
                                ibeg += istep
                        ## wait for all subprocesses with a tidy keyboard quit
                        try:
                            p.close()
                            p.join()
                        except KeyboardInterrupt as err:
                            p.terminate()
                            p.join()
                            raise err
                        y = np.sum(y,axis=0)
                    else:
                        ## no multiprocessing
                        y = np.zeros(x.shape)
                        self.assert_known(ykey,xkey,'Γ','masspp','TDopplerpp')
                        for d in self.unique_dicts('masspp','TDopplerpp'):
                            m = self[(xkey,ykey,'Γ')][self.match(**d)] # get relevant data as a recarrat -- faster than using unique_dicts_matches
                            y += lineshapes.voigt_spectrum_with_gaussian_doppler_width(
                                x,m[xkey],m[ykey],m['Γ'],d['masspp'],d['TDopplerpp'],
                                nfwhmL=nfwhmL,nfwhmG=nfwhmG,Smin=ymin)
                else:
                    raise Exception(f"voigt_method unknown: {repr(voigt_method)}")
            else:
                raise Exception("No method for calculating spectrum implemented.")
        else:
            ## Compute using existing cache. Determine which lines
            ## need to be updated and update them. Do this row-by-row
            ## using recarray equality. Need to remove references to
            ## keys containing NaNs
            i = np.any([self[key]!=cache[key] for key in cache_keys],0)
            ## if too many lines changed just recompute everything
            tkwargs = dict(nfwhmL=nfwhmL,nfwhmG=nfwhmG,Smin=ymin,use_multiprocessing=use_multiprocessing)
            if (sum(i)/len(i))>0.25:
                self._cache.pop('calculate_spectrum')
                x,y = self.calculate_spectrum(**all_args,use_cache=use_cache)
            elif sum(i)==0:
                ## no change at all
                y = cache['y']
            else:
                ## add subtract old lines from cached data, add new lines
                y = (cache['y']
                     - lineshapes.voigt_spectrum(x,cache[xkey][i],cache[ykey][i],cache[ΓLin][i],cache[ΓGin][i],**tkwargs)
                     + lineshapes.voigt_spectrum(x,self[xkey][i],self[ykey][i],self[ΓLin][i],self[ΓGin][i],**tkwargs))
        ## save cache if necessary
        if use_cache:
            self._cache['calculate_spectrum'] = {'x':x,'y':y}
            for key in cache_keys:
                self._cache['calculate_spectrum'][key] = copy(self[key])
        return(x,y)

    def plot_spectrum(
            self,
            x=None,
            xkey='ν',
            ykey='σ',
            zkeys=None,         # None or list of keys to plot as separate lines
            ax=None,
            xunits = 'cm-1', # alternative is 'nm'
            xlim=None,
            show=False,
            **plot_kwargs # can be calculate_spectrum or plot kwargs
    ):
        from matplotlib import pyplot as plt
        """Plot a nice cross section. If zkeys given then divide into multiple
        lines accordings."""
        if ax is None:
            ax = plt.gca()
        if x is not None and len(x)==0:
            return(ax)
        if zkeys==None:
            ## one line
            x,y = self.calculate_spectrum(x,ykey=ykey,xkey=xkey)
            if xunits=='cm-1':
                x = x
            # elif xunits=='nm':
                # x = my.k2nm(x)
            else:
                raise Exception(f"Unknown sunits: {repr(xunits)}")
            line = ax.plot(x,y,**plot_kwargs)[0]
            if xlim is not None:   ax.set_xlim(*xlim)
        else:
            ## multiple lines
            for iz,(qn,t) in enumerate(self.unique_dicts_matches(*zkeys)):
                t_plot_kwargs = copy(plot_kwargs)
                t_plot_kwargs.setdefault('color',my.newcolor(iz))
                t_plot_kwargs.setdefault('label',my.dict_to_kwargs(qn))
                t.plot_spectrum(
                    x=x,ykey=ykey,zkeys=None,ax=ax,xunits=xunits,xlim=xlim,
                    **calculate_spectrum_kwargs,**t_plot_kwargs)
        if show:
            plt.show()
        return(ax)

    def get_levels(
            self,
            upper_or_lower='upper',      # 'upper' or 'lower'
            # treat_common_data='weighted average', # 'average','unweighted average','weighted_average', or 'reduce'
            # **level_kwargs,             # added to constructor of returned Level object
    ):
        """Get a Level object containing all possible data about the 'upper'
        or 'lower' level. If treat_common_data is 'reduce' get level
        from first transition involving this level, if 'average' then
        take the mean of level data from all relevant transitinos and
        add their uncertainties as if independent data."""
        levels = self['levels_class']()
        assert upper_or_lower in ('upper','lower'),f'upper_or_lower must be "lower" or "upper", not {repr(upper_or_lower)}'
        key_suffix = 'p' if upper_or_lower=='upper' else'pp'
        for key in self.keys():
            if len(key)<len(key_suffix) or key[:-len(key_suffix)] not in levels._prototypes:
                continue
            levels.set(key[:-len(key_suffix)],self.get_value(key),self.get_uncertainty(key))
        return(levels)

