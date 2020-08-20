import itertools
from copy import copy
from pprint import pprint

import numpy as np

from spectr.dataset import DataSet
from spectr.levels import *
from spectr import lineshapes
from spectr import tools
from spectr.data_prototypes import prototypes

level_suffix = {'upper':'_u','lower':'_l'}

def _expand_level_keys(level_class):
    """Take level keys and make upper and lower level copys for a Lines
    object."""
    retval = {}
    for key,val in level_class._prototypes.items():
        retval[key+level_suffix['upper']] = copy(val)
        retval[key+level_suffix['lower']] = copy(val)
    return(retval)

def _get_key_without_level_suffix(upper_or_lower,key):
    suffix = level_suffix[upper_or_lower]
    if len(key) <= len(suffix):
        return None
    if key[-len(suffix):] != suffix:
        return None
    return key[:-len(suffix)]

class Lines(DataSet):
    """For now rotational lines."""

    _prototypes = {key:prototypes[key] for key in (
        'species',
        # 'class',
        'levels_class', 'description',
        'notes', 'author', 'reference', 'date',
        'branch',
        'ΔJ',
        'ν', 'Γ', 'ΓD', 'f','Ae',
        'γair','nair','δair','γself',
    )}

    _levels_class = Levels

    _prototypes.update(_expand_level_keys(Levels))
    _prototypes['ν']['infer']['E_u','E_l'] = lambda Eu,El: Eu-El
    _prototypes['E_l']['infer']['E_u','ν'] = lambda Eu,ν: Eu-ν
    _prototypes['E_u']['infer']['E_l','ν'] = lambda El,ν: El+ν
    _prototypes['Γ']['infer']['Γ_u','Γ_l'] = lambda Γu,Γl: Γu+Γl
    _prototypes['Γ_l']['infer']['Γ','Γ_u'] = lambda Γ,Γu: Γ-Γu
    _prototypes['Γ_u']['infer']['Γ','Γ_l'] = lambda Γ,Γl: Γ-Γl

    
    def __init__(
            self,
            name=None,
            **keys_vals,
    ):
        DataSet.__init__(self)
        self.permit_nonprototyped_data = False
        # self['class'] = type(self).__name__
        # self.name = (name if name is not None else self['class'])
        self.name = (name if name is not None else type(self).__name__)
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
            # temperaturel=None,
            # column_densityl=None,
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
                        for d in self.unique_dicts('mass_l','TDoppler_l'):
                            m = self[(xkey,ykey,'Γ')][self.match(**d)] # get relevant data as a recarrat -- faster than using unique_dicts_matches
                            ibeg,istep = 0,int(len(m)/number_of_processes_per_mass_temperature_combination)
                            while ibeg<len(m): # loop through lines in chunks, starting subprocesses
                                iend = min(ibeg+istep,len(m))
                                t = p.apply_async(lineshapes.voigt_spectrum_with_gaussian_doppler_width,
                                                  args=(x,m[xkey][ibeg:iend],m[ykey][ibeg:iend],m['Γ'][ibeg:iend],
                                                        d['mass_l'],d['TDoppler_l'], nfwhmL,nfwhmG,ymin),
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
                        self.assert_known(ykey,xkey,'Γ','mass_l','TDoppler_l')
                        for d in self.unique_dicts('mass_l','TDoppler_l'):
                            m = self[(xkey,ykey,'Γ')][self.match(**d)] # get relevant data as a recarrat -- faster than using unique_dicts_matches
                            y += lineshapes.voigt_spectrum_with_gaussian_doppler_width(
                                x,m[xkey],m[ykey],m['Γ'],d['mass_l'],d['TDoppler_l'],
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

    def get_levels(self,upper_or_lower):
        """Get all data corresponding to 'upper' or 'lower' level in
        self."""
        levels = self._levels_class()
        assert upper_or_lower in ('upper','lower'),f'upper_or_lower must be "upper" or "lower", not {repr(upper_or_lower)}'
        for key in self.keys():
            if (level_key:=_get_key_without_level_suffix(upper_or_lower,key)) is not None:
                levels.set(level_key,self.get_value(key),self.get_uncertainty(key))
        return(levels)

    upper_levels = property(lambda self: self.get_levels('upper'))
    lower_levels = property(lambda self: self.get_levels('lower'))

class HeteronuclearDiatomicLines(Lines):

    _prototypes = copy(Lines._prototypes)
    _prototypes.update(_expand_level_keys(HeteronuclearDiatomicLevels))

    def load_from_hitran(self,filename):
        """Load HITRAN .data."""
        from . import hitran
        data = hitran.load_lines(filename)
        species = np.unique(hitran.translate_codes_to_species(data['Mol']))
        assert len(species)==1,'Cannot handle mixed species HITRAN linelist.'
        species = species[0]
        ## interpret into transition quantities common to all transitions
        kw = {
            'ν':data['ν'],
            'Ae':data['A'],
            'E'+level_suffix['lower']:data['E_l'],
            'g'+level_suffix['upper']:data['g_u'],
            'g'+level_suffix['lower']:data['g_l'],
            'γair':data['γair']*2, # HITRAN uses HWHM, I'm going to go with FWHM
            'nair':data['nair'],
            'δair':data['δair'],
            'γself':data['γself']*2, # HITRAN uses HWHM, I'm going to go with FWHM
        }
        ## get species
        assert len(np.unique(data['Mol']))==1
        try:
            ## full isotopologue
            kw['species'] = hitran.translate_codes_to_species(data['Mol'],data['Iso'])
        except KeyError:
            assert len(np.unique(data['Iso']))==1,'Cannot identify isotopologues and multiple are present.'
            kw['species'] = hitran.translate_codes_to_species(data['Mol'])
        ## interpret quantum numbers and insert into some kind of transition, this logic is in its infancy
        ## standin for diatomics
        kw['v'+level_suffix['upper']] = data['V_u']
        kw['v'+level_suffix['lower']] = data['V_l']
        branches = {'P':-1,'Q':0,'R':+1}
        ΔJ,J_l = [],[]
        for Q_l in data['Q_l']:
            branchi,Jli = Q_l.split()
            ΔJ.append(branches[branchi])
            J_l.append(Jli)
        kw['ΔJ'] = np.array(ΔJ,dtype=int)
        kw['J'+level_suffix['lower']] = np.array(J_l,dtype=float)
        self.extend(**kw)
        
