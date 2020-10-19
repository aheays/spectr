import itertools
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from scipy import constants

# from . import *
# from .dataset import Dataset
from . import tools
from . import levels
from . import lineshapes
from . import tools
from . import hitran
from . import database
from . import plotting
from .conversions import convert
from .exceptions import InferException


level_suffix = {'upper':'_u','lower':'_l'}
upper = level_suffix['upper']
lower = level_suffix['lower']

################
## prototypes ##
################
prototypes = {}

## copy some direct from levels
for key in (
        'classname','description','notes','author','reference','date',
        'species','isotopologue',
        'mass','reduced_mass','partition_source','partition',
):
    prototypes[key] = copy(levels.prototypes[key])
## import all from levels with suffices added
for key,val in levels.prototypes.items():
    tval = deepcopy(val)
    tval['infer'] = {tuple(key+level_suffix['upper']
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+level_suffix['upper']] = tval
    tval['infer'] = {tuple(key+level_suffix['lower']
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+level_suffix['lower']] = tval

## add lines things
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='8U', cast=str, fmt='<10s')
prototypes['ν'] = dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>0.6f', infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g',infer={
    ('γself','Pself','γair','Pair'):lambda γself,Pself,γair,Pair: γself*convert(Pself,'Pa','atm')+γair*convert(Pair,'Pa','atm'), # LINEAR COMBINATION!
    ('γself','Pself'):lambda γself,Pself: γself*convert(Pself,'Pa','atm'),
    ('γair','Pair'):lambda γair,Pair: γair*convert(Pair,'Pa','atm'),})
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind=float,fmt='<10.5g', infer={('mass','Ttr','ν'): lambda mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,})
prototypes['f'] = dict(description="Line f-value (dimensionless)",kind=float,fmt='<10.5e', infer={
    ('Ae','ν','g_u','g_l'):lambda Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2, 
    ('σ','α_l'):lambda σ,α_l: σ*1.1296e12/α_l,})
prototypes['σ'] = dict(description="Spectrally-integrated photoabsorption cross section (cm2.cm-1).", kind=float, fmt='<10.5e',infer={
    ('τa','Nself_l'):lambda τ,column_densitypp: τ/column_densitypp, 
    ('f','α_l'):lambda f,α_l: f/1.1296e12*α_l,
    ('S','ν','Teq'):lambda S,ν,Teq,: S/(1-np.exp(-convert(constants.Boltzmann,'J','cm-1')*ν/Teq)),})
def _f0(S296K,species,partition,E_l,Tex,ν):
    """See Eq. 9 of simeckova2006"""
    partition_296K = hitran.get_partition_function(species,296)
    c = convert(constants.Boltzmann,'J','cm-1') # hc/kB
    return (S296K
            *((np.exp(-E_l/(c*Tex))/partition)*(1-np.exp(-c*ν/Tex)))
            /((np.exp(-E_l/(c*296))/partition_296K)*(1-np.exp(-c*ν/296))))
prototypes['S'] = dict(description="Spectral line intensity (cm or cm-1/(molecular.cm-2) ", kind=float, fmt='<10.5e', infer={
    ('S296K','species','partition','E_l','Tex','ν'):_f0,})
prototypes['S296K'] = dict(description="Spectral line intensity at 296K reference temperature ( cm-1/(molecular.cm-2) ). This is not quite the same as HITRAN which also weights line intensities by their natural isotopologue abundance.", kind=float, fmt='<10.5e', infer={})
## Preferentially compute τ from the spectral line intensity, S,
## rather than than the photoabsorption cross section, σ, because the
## former considers the effect of stimulated emission.
prototypes['τ'] = dict(description="Integrated optical depth including stimulated emission (cm-1)", kind=float, fmt='<10.5e', infer={
    ('S','Nself_l'):lambda S,Nself_l: S*Nself_l,
},)
prototypes['τa'] = dict(description="Integrated optical depth from absorption only (cm-1)", kind=float, fmt='<10.5e', infer={
    ('σ','Nself_l'):lambda σ,Nself_l: σ*Nself_l,
},)
prototypes['Ae'] = dict(description="Radiative decay rate (s-1)", kind=float, fmt='<10.5g', infer={
    ('f','ν','g_u','g_l'):lambda f,ν,g_u,g_l: f/(1.49951*g_u/g_l/ν**2),
    ('At','Ad'): lambda At,Ad: At-Ad,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={
    'Teq':lambda Tex:Teq,
})
prototypes['Ttr'] = dict(description="Translational temperature (K)", kind=float, fmt='0.2f', infer={
    'Tex':lambda Tex:Tex,})
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind=float, fmt='>+4g', infer={
    ('Jp','Jpp'):lambda Jp,Jpp: Jp-Jpp,},)
prototypes['L'] = dict(description="Optical path length (m)", kind=float, fmt='0.5f', infer={})
prototypes['γair'] = dict(description="Pressure broadening coefficient in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['δair'] = dict(description="Pressure shift coefficient in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['nair'] = dict(description="Pressure broadening temperature dependence in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['γself'] = dict(description="Pressure self-broadening coefficient (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['Pself'] = dict(description="Pressure of self (Pa)", kind=float, fmt='0.5f', infer={})
prototypes['Pair'] = dict(description="Pressure of air (Pa)", kind=float, fmt='0.5f', infer={})
prototypes['Nself'] = dict(description="Column density (cm-2)",kind=float,fmt='<11.3e', infer={
    ('Pself','L','Teq'): lambda Pself,L,Teq: (Pself*L)/(database.constants.Boltzmann*Teq)*1e-4,})

## add infer functions -- could add some of these to above
prototypes['ν']['infer']['E_u','E_l'] = lambda Eu,El: Eu-El
prototypes['E_l']['infer']['E_u','ν'] = lambda Eu,ν: Eu-ν
prototypes['E_u']['infer']['E_l','ν'] = lambda El,ν: El+ν
prototypes['Γ']['infer']['Γ_u','Γ_l'] = lambda Γu,Γl: Γu+Γl
prototypes['Γ_l']['infer']['Γ','Γ_u'] = lambda Γ,Γu: Γ-Γu
prototypes['Γ_u']['infer']['Γ','Γ_l'] = lambda Γ,Γl: Γ-Γl
prototypes['J_u']['infer']['J_l','ΔJ'] = lambda J_l,ΔJ: J_l+ΔJ
prototypes['Tex']['infer']['Teq'] = lambda Teq: Teq
prototypes['Teq_u']['infer']['Teq'] = lambda Teq: Teq
prototypes['Teq_l']['infer']['Teq'] = lambda Teq: Teq
prototypes['Nself_u']['infer']['Nself'] = lambda Nself: Nself
prototypes['Nself_l']['infer']['Nself'] = lambda Nself: Nself
prototypes['species_l']['infer']['species'] = lambda species: species
prototypes['species_u']['infer']['species'] = lambda species: species
prototypes['ΔJ']['infer']['J_u','J_l'] = lambda J_u,J_l: J_u-J_l
prototypes['partition']['infer']['partition_l'] = lambda partition_l:partition_l
prototypes['partition']['infer']['partition_u'] = lambda partition_u:partition_u
prototypes['partition_l']['infer']['partition'] = lambda partition:partition
prototypes['partition_u']['infer']['partition'] = lambda partition:partition

def _get_key_without_level_suffix(upper_or_lower,key):
    suffix = level_suffix[upper_or_lower]
    if len(key) <= len(suffix):
        return None
    if key[-len(suffix):] != suffix:
        return None
    return key[:-len(suffix)]

def _expand_level_keys_to_upper_lower(levels_class):
    retval = []
    for key,val in levels_class.prototypes.items():
        retval.append(key+level_suffix['lower'])
        retval.append(key+level_suffix['upper'])
    return retval


#############
## classes ##
#############

class Base(levels._BaseLinesLevels):
    """For now rotational lines."""


    prototypes = {key:copy(prototypes[key]) for key in (
        ['classname', 'description', 'notes', 'author', 'reference', 'date',]
        + _expand_level_keys_to_upper_lower(levels.Base)
        + ['species','isotopologue','mass', 'reduced_mass',
           'branch', 'ΔJ',
           'ν',
           'Γ', 'ΓD',
           'f', 'σ', 
           'S','S296K', 
           'τ', 'Ae','τa',
           'Teq', 'Tex', 'Ttr','partition',
           'partition_source',
           'partition',
           'L',
           'γair', 'δair', 'γself', 'nair',
           'Pself', 'Pair', 'Nself',])}

    def plot_spectrum(
            self,
            x=None,
            xkey='ν',
            ykey='σ',
            zkeys=None,         # None or list of keys to plot as separate lines
            ax=None,
            **plot_kwargs # can be calculate_spectrum or plot kwargs
    ):
        from matplotlib import pyplot as plt
        """Plot a nice cross section. If zkeys given then divide into multiple
        lines accordings."""
        if ax is None:
            ax = plt.gca()
        if zkeys==None:
            if ykey == 'transmission': # special case
                x,y = self.calculate_spectrum(x,ykey='τ',xkey=xkey)
                y = np.exp(-y)
            else:
                x,y = self.calculate_spectrum(x,ykey=ykey,xkey=xkey)
                line = ax.plot(x,y,**plot_kwargs)[0]
        else:
            for iz,(qn,t) in enumerate(self.unique_dicts_matches(*zkeys)):
                t_plot_kwargs = copy(plot_kwargs)
                t_plot_kwargs.setdefault('color',my.newcolor(iz))
                t_plot_kwargs.setdefault('label',my.dict_to_kwargs(qn))
                t.plot_spectrum(x=x,ykey=ykey,zkeys=None,ax=ax,**t_plot_kwargs)
        return(ax)

    def plot_stick_spectrum(
            self,
            xkey='ν',
            ykey='σ',
            zkeys=None,         # None or list of keys to plot as separate lines
            ax=None,
            **plot_kwargs # can be calculate_spectrum or plot kwargs
    ):
        from matplotlib import pyplot as plt
        """Plot a nice cross section. If zkeys given then divide into multiple
        lines accordings."""
        if ax is None:
            ax = plt.gca()
        if zkeys==None:
            plotting.plot_sticks(self[xkey],self[ykey],**plot_kwargs)
        else:
            for iz,(qn,t) in enumerate(self.unique_dicts_matches(*zkeys)):
                t_plot_kwargs = copy(plot_kwargs)
                t_plot_kwargs.setdefault('color',my.newcolor(iz))
                t_plot_kwargs.setdefault('label',my.dict_to_kwargs(qn))
                t.plot_stick_spectrum(ykey=ykey,zkeys=None,ax=ax,**t_plot_kwargs)
        return(ax)

    def calculate_spectrum(
            self,
            x=None,        # frequency grid (must be regular, I think), if None then construct a reasonable grid
            xkey='ν',      # strength to use, i.e., "ν", or "λ"
            ykey='σ',      # strength to use, i.e., "σ", "τ", or "I"
            ΓG='ΓD', # a key to use for Gaussian widths, a constant numeric value, or None to neglect Gaussian entirely
            ΓL='Γ',        # a key or for Lorentzian widths (i.e., "Γ"), a constant numeric value, or None to neglect Lorentzian entirely
            nfwhmG=20,         # how many Gaussian FWHMs to include in convolution
            nfwhmL=100,         # how many Lorentzian FWHMs to compute
            nx=10000,     # number of grid points used if x not give
            ymin=None,     # minimum value of ykey before a line is ignored, None for use all lines
            gaussian_method='fortran', #'fortran stepwise', 'fortran', 'python'
            voigt_method='wofz',   
            # temperaturepp=None,
            # column_densitypp=None,
            use_multiprocessing=False, # might see a speed up
            use_cache=False,    # is it actually any faster?!?
            # ycache = None,
            **set_keys_vals,    # set some data first, e..g, the tempertaure
    ):
        """Calculate a Voigt/Lorentzian/Gaussian spectrum from data in self. Returns (x,σ)."""
        for key,val in set_keys_vals.items():
            self[key] = val
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
        ## check frequencies, strengthgs, widths are as expected
        self.assert_known(xkey,ykey)
        assert np.all(~np.isnan(self[xkey])),f'NaN values in xkey: {repr(xkey)}'
        assert np.all(~np.isnan(self[ykey])),f'NaN values in ykey: {repr(ykey)}'
        if ΓG is None:
            pass
        elif isinstance(ΓG,str):
            self.assert_known(ΓG)
            assert np.all(~np.isnan(self[ΓG])),f'NaN values in ΓG key: {repr(xkey)}'
            ΓG = self.get_value(ΓG,ensure_vector=True)
        elif np.isscalar:
            ΓG = np.full(len(self),ΓG,dtype=float)
        else:
            ΓG = np.asarray(ΓG,dtype=float)
            assert np.all(~np.isnan(ΓG)),f'NaN values in provided ΓG array'
            assert len(ΓG)==len(self),'Provided ΓG array wrong length'
        if ΓL is None:
            pass
        elif isinstance(ΓL,str):
            self.assert_known(ΓL)
            assert np.all(~np.isnan(self[ΓL])),f'NaN values in ΓL key: {repr(xkey)}'
            ΓL = self.get_value(ΓL,ensure_vector=True)
        elif np.isscalar:
            ΓL = np.full(len(self),ΓL,dtype=float)
        else:
            ΓL = np.asarray(ΓL,dtype=float)
            assert np.all(~np.isnan(ΓL)),f'NaN values in provided ΓL array'
            assert len(ΓL)==len(self),'Provided ΓL array wrong length'
        ## test for appropriate use of cache
        if use_cache:
            for test,msg in (
                    (voigt_method=='wofz','Cache only implemented for wofz.'),
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
        ## test is cache exist and is usable
        if (not use_cache   # no cache requested
            or cache is None         # cache does not exists - probably first run
            or len(cache['y'])!=len(x)   # x domain is the same length as the last cached calculation 
            or not np.all(cache['x']==x)):   # x domain is the same as the last cached calculation -- TEST REQUIRES MUCH MEMORY?
            ## comput entire spectrum with out cache
            ##
            ## get a default frequency scale if none provided
            if x is None:
                x = np.linspace(max(0,self[xkey].min()-10.),self[xkey].max()+10.,nx)
            elif np.isscalar(x):
                x = np.arange(max(0,self[xkey].min()-10.),self[xkey].max()+10.,x)
            else:
                x = np.asarray(x)
            ## get spectrum type according to width specified
            ##
            if ΓL is None and ΓG is None:
                ## divide centroided triangles
                y = lineshapes.centroided_spectrum(x,self[xkey],self[ykey],Smin=ymin)
            elif ΓL is not None and ΓG is None:
                ## spectrum Lorentzians
                y = lineshapes.lorentzian_spectrum(x,self[xkey],self[ykey],ΓL,nfwhm=nfwhmL,Smin=ymin)
            elif ΓG is not None and ΓL is None:
                ## spectrum Gaussians
                y = lineshapes.gaussian_spectrum(x, self[xkey], self[ykey], ΓG, nfwhm=nfwhmG,Smin=ymin,method=gaussian_method)
            elif ΓL is not None and ΓG is not None:
                if voigt_method=='wofz':
                    ## spectrum of Voigts computed by wofz
                    y = lineshapes.voigt_spectrum(
                        x,self[xkey],self[ykey],ΓL,ΓG,
                        nfwhmL,nfwhmG,Smin=ymin, use_multiprocessing=use_multiprocessing)
                elif voigt_method=='fortran Doppler' and ΓLin=='Γ' and ΓGin=='ΓDoppler':
                    ## spectrum of Voigts with common mass/temperature lines
                    ## computed in groups with fortran code
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
                        ## for d,m in self.unique_dicts_matches('masspp','temperaturepp'):
                        ##     y += lineshapes.voigt_spectrum_with_gaussian_doppler_width(
                        ##         x,m[xkey],m[ykey], m['Γ'],d['masspp'],d['temperaturepp'],
                        ##         nfwhmL=nfwhmL,nfwhmG=nfwhmG,Smin=ymin)
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
            i = np.any([self[key]!=cache[key] for key in cache_keys],0) # changed lines
            if (ykey == 'τ'     # absorption
                and self.vector_data['τ'].inferred_from == {'σ','column_densitypp'} # τ is computed from column_densitypp and σ
                and sum(i) == len(i)            # all data has changed
                and len(np.unique(cache['column_densitypp']/self['column_densitypp'])) == 1 # all column_densitypp has changed by the same factor
                and np.all(self['σ']==cache['σ']) # no σ has not changed
                and np.all([self[key]==cache[key] for key in cache_keys if key != 'τ'])): # frequencies and widths have not changed
                ## all lines have changed column density but nothing
                ## else. Rescale spectrum by change common change in
                ## column density.
                if self.verbose or True:
                    print('calculate_spectrum: using absorption column_densitypp shortcut')
                y = cache['y']/cache['column_densitypp'][0]*self['column_densitypp'][0]
            elif (ykey == 'I'     # emission -- UNSTABLE definition
                  and self.vector_data['I'].inferred_from == {'Ae','column_densityp'}
                  and sum(i) == len(i)            # all data has changed
                  and len(np.unique(cache['column_densityp']/self['column_densityp'])) == 1 # all column_densityp has changed by the same factor
                  and np.all(cache['Ae']==self['Ae']) # Ae are the same
                  and np.all([self[key]==cache[key] for key in cache_keys if key != 'I'])): # frequencies and widths have not changed
                if self.verbose:
                    print('calculate_spectrum: using emission column_densitypp shortcut')
                y = cache['y']/cache['column_densityp'][0]*self['column_densityp'][0]
            elif (sum(i)/len(i))>0.25:
                ## many lines have changed, just recompute all
                self._cache.pop('calculate_spectrum')
                x,y = self.calculate_spectrum(**all_args,use_cache=use_cache)
            elif sum(i)==0:
                ## no change at all
                y = cache['y']
            else:
                ## a few changed line, subtract old lines from cached
                ## spectrum, add new lines
                tkwargs = dict(nfwhmL=nfwhmL,nfwhmG=nfwhmG,Smin=ymin,use_multiprocessing=use_multiprocessing)
                y = (cache['y']
                     - lineshapes.voigt_spectrum(x,cache[xkey][i],cache[ykey][i],cache[ΓLin][i],cache[ΓGin][i],**tkwargs)
                     + lineshapes.voigt_spectrum(x,self[xkey][i],self[ykey][i],self[ΓLin][i],self[ΓGin][i],**tkwargs))
        ## save cache
        if use_cache:
            cache = {'x':x,'y':y}
            for key in cache_keys:
                cache[key] = copy(self[key])
            ## save these for rescale column density shortcuts
            if (ykey == 'τ' and self.vector_data['τ'].inferred_from == {'σ','column_densitypp'}):
                cache['column_densitypp'] = copy(self['column_densitypp'])
                cache['σ'] = copy(self['σ'])
            if (ykey == 'I' and self.vector_data['I'].inferred_from == {'Ae','column_densityp'}):
                cache['column_densityp'] = copy(self['column_densityp'])
                cache['Ae'] = copy(self['Ae'])
        return(x,y)

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

    def load_from_hitran(self,filename):
        """Load HITRAN .data."""
        data = hitran.load(filename)
        ## interpret into transition quantities common to all transitions
        new = self.__class__(**{
            'ν':data['ν'],
            ## 'Ae':data['A'],  # Ae data is incomplete but S296K will be complete
            'S296K':data['S'],
            'E_l':data['E_l'],
            'g_u':data['g_u'],
            'g_l':data['g_l'],
            'γair':data['γair']*2, # HITRAN uses HWHM, I'm going to go with FWHM
            'nair':data['nair'],
            'δair':data['δair'],
            'γself':data['γself']*2, # HITRAN uses HWHM, I'm going to go with FWHM
        })
        ## get species
        species,isotopologue = hitran.translate_codes_to_species_isotopologue(data['Mol'],data['Iso'])
        new['species'],new['isotopologue'] = species,isotopologue
        ## remove natural abundance weighting
        for d,i in new.unique_dicts_match('isotopologue'):
            new['S296K'][i] /=  hitran.molparam.get_unique_value('natural_abundance',**d)
        ## ## interpret quantum numbers and insert into some kind of transition, this logic is in its infancy
        ## ## standin for diatomics
        ## kw['v'+level_suffix['upper']] = data['V_u']
        ## kw['v'+level_suffix['lower']] = data['V_l']
        ## branches = {'P':-1,'Q':0,'R':+1}
        ## ΔJ,J_l = [],[]
        ## for Q_l in data['Q_l']:
        ##     branchi,Jli = Q_l.split()
        ##     ΔJ.append(branches[branchi])
        ##     J_l.append(Jli)
        ## kw['ΔJ'] = np.array(ΔJ,dtype=int)
        ## kw['J'+level_suffix['lower']] = np.array(J_l,dtype=float)
        self.concatenate(new)

class DiatomicCinfv(Base):

    prototypes = {key:copy(prototypes[key]) for key in (
        list(Base.prototypes)
        + _expand_level_keys_to_upper_lower(levels.DiatomicCinfv))}


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
        

# class TriatomicDinfh(Base):

    # prototypes = {key:copy(prototypes[key]) for key in (
        # list(Base.prototypes)
        # + _expand_level_keys_to_upper_lower(levels.TriatomicDinfh))}
