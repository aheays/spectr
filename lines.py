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
# from .lines import prototypes
from . import levels
from .levels import _unique

prototypes = {}

## copy some direct from levels
for key in (
        'species',
        'mass','reduced_mass',
        'partition_source','partition',
        'ΓD',
):
    prototypes[key] = copy(levels.prototypes[key])

## import all from levels with suffices added
for key,val in levels.prototypes.items():
    tval = deepcopy(val)
    tval['infer'] = {tuple(key+'_u'
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+'_u'] = tval
    tval['infer'] = {tuple(key+'_l'
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+'_l'] = tval

## add lines things
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='U', cast=str, fmt='<10s')
prototypes['ν'] = dict(description="Transition wavenumber (cm-1)", kind='f', fmt='>0.6f', infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind='f', fmt='<10.5g',infer={
    ('γself','Pself','γair','Pair'):lambda self,γself,Pself,γair,Pair: γself*convert(Pself,'Pa','atm')+γair*convert(Pair,'Pa','atm'), # LINEAR COMBINATION!
    ('γself','Pself'):lambda self,γself,Pself: γself*convert(Pself,'Pa','atm'),
    ('γair','Pair'):lambda self,γair,Pair: γair*convert(Pair,'Pa','atm'),})
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind='f',fmt='<10.5g', infer={('mass','Ttr','ν'): lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,})
prototypes['f'] = dict(description="Line f-value (dimensionless)",kind='f',fmt='<10.5e', infer={
    ('Ae','ν','g_u','g_l'):lambda self,Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2,
    ('Sij','ν','J_l'): lambda self,Sij,ν,J_l: 3.038e-6*ν*Sij/(2*J_l+1), 
    ('σ','α_l'):lambda self,σ,α_l: σ*1.1296e12/α_l,})
prototypes['σ'] = dict(description="Spectrally-integrated photoabsorption cross section (cm2.cm-1).", kind='f', fmt='<10.5e',infer={
    ('τa','Nself_l'):lambda self,τ,column_densitypp: τ/column_densitypp, 
    ('f','α_l'):lambda self,f,α_l: f/1.1296e12*α_l,
    ('S','ν','Teq'):lambda self,S,ν,Teq,: S/(1-np.exp(-convert(constants.Boltzmann,'J','cm-1')*ν/Teq)),})
# prototypes['σ'] =dict(description="Integrated cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('τ','column_densitypp'):lambda self,τ,column_densitypp: τ/column_densitypp, ('f','populationpp'):lambda self,f,populationpp: f/1.1296e12*populationpp,})
def _f0(self,S296K,species,partition,E_l,Tex,ν):
    """See Eq. 9 of simeckova2006"""
    partition_296K = hitran.get_partition_function(species,296)
    c = convert(constants.Boltzmann,'J','cm-1') # hc/kB
    return (S296K
            *((np.exp(-E_l/(c*Tex))/partition)*(1-np.exp(-c*ν/Tex)))
            /((np.exp(-E_l/(c*296))/partition_296K)*(1-np.exp(-c*ν/296))))
prototypes['S'] = dict(description="Spectral line intensity (cm or cm-1/(molecular.cm-2) ", kind='f', fmt='<10.5e', infer={('S296K','species','partition','E_l','Tex_l','ν'):_f0,})
prototypes['S296K'] = dict(description="Spectral line intensity at 296K reference temperature ( cm-1/(molecular.cm-2) ). This is not quite the same as HITRAN which also weights line intensities by their natural isotopologue abundance.", kind='f', fmt='<10.5e', infer={})
## Preferentially compute τ from the spectral line intensity, S,
## rather than than the photoabsorption cross section, σ, because the
## former considers the effect of stimulated emission.
prototypes['τ'] = dict(description="Integrated optical depth including stimulated emission (cm-1)", kind='f', fmt='<10.5e', infer={('S','Nself_l'):lambda self,S,Nself_l: S*Nself_l,},)
prototypes['τa'] = dict(description="Integrated optical depth from absorption only (cm-1)", kind='f', fmt='<10.5e', infer={('σ','Nself_l'):lambda self,σ,Nself_l: σ*Nself_l,},)
prototypes['Ae'] = dict(description="Radiative decay rate (s-1)", kind='f', fmt='<10.5g', infer={
    ('f','ν','g_u','g_l'):lambda self,f,ν,g_u,g_l: f/(1.49951*g_u/g_l/ν**2),
    ('At','Ad'): lambda self,At,Ad: At-Ad,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind='f', fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind='f', fmt='0.2f', infer={'Teq':lambda self,Tex:Teq,})
prototypes['Ttr'] = dict(description="Translational temperature (K)", kind='f', fmt='0.2f', infer={'Teq':lambda self,Teq:Teq,})
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind='f', fmt='>+4g', infer={('Jp','Jpp'):lambda self,Jp,Jpp: Jp-Jpp,},)
prototypes['L'] = dict(description="Optical path length (m)", kind='f', fmt='0.5f', infer={})
prototypes['γair'] = dict(description="Pressure broadening coefficient in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda x:np.abs(np.asarray(x),dtype=float), fmt='<10.5g', infer={},)
prototypes['δair'] = dict(description="Pressure shift coefficient in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda x:np.abs(np.asarray(x),dtype=float), fmt='<10.5g', infer={},)
prototypes['nair'] = dict(description="Pressure broadening temperature dependence in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda x:np.abs(np.asarray(x),dtype=float), fmt='<10.5g', infer={},)
prototypes['γself'] = dict(description="Pressure self-broadening coefficient (cm-1.atm-1.FWHM)", kind='f', cast=lambda x:np.abs(np.asarray(x),dtype=float), fmt='<10.5g', infer={},)
prototypes['Pself'] = dict(description="Pressure of self (Pa)", kind='f', fmt='0.5f', infer={})
prototypes['Pair'] = dict(description="Pressure of air (Pa)", kind='f', fmt='0.5f', infer={})
prototypes['Nself'] = dict(description="Column density (cm-2)",kind='f',fmt='<11.3e', infer={('Pself','L','Teq'): lambda self,Pself,L,Teq: (Pself*L)/(database.constants.Boltzmann*Teq)*1e-4,})

## vibrational transition frequencies
prototypes['νv'] = dict(description="Electronic-vibrational transition wavenumber (cm-1)", kined=float, fmt='>11.4f', infer={('Tvp','Tvpp'): lambda self,Tvp,Tvpp: Tvp-Tvpp, ('λv',): lambda self,λv: convert_units(λv,'nm','cm-1'),})
prototypes['λv'] = dict(description="Electronic-vibrational transition wavelength (nm)", kind='f', fmt='>11.4f', infer={('νv',): lambda self,νv: convert_units(νv,'cm-1','nm'),},)

## transition strengths
prototypes['M']   = dict(description="Pointer to electronic transition moment (au).", kind=object, infer={})
prototypes['Mv']   = dict(description="Electronic transition moment for this vibronic level (au).", kind='f', fmt='<10.5e', infer={('μ','FCfactor'): lambda self,μ,FCfactor: μ/np.sqrt(FCfactor),})
prototypes['μv']  = dict(description="Electronic-vibrational transition moment (au)", kind='f',  fmt='<10.5e', infer={('M','χp','χpp','R'): lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),},) # could infer from S but then sign would be unknown
prototypes['μ']   = dict(description="Electronic-vibrational-rotational transition moment (au)", kind='f',  fmt='<10.5e', infer={('M','χp','χpp','R'): lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),},) # could infer from S but then sign would be unknown
def _f0(self,fv,ν,Λp,Λpp):
    """Convert a summed band fvalue into a band_strength."""
    Sv = fv/3.038e-6/ν
    Sv[(Λpp==0)&(Λp!=0)] /= 2 # divisor of (2-δ(0,Λ")δ(0,Λ'))/(2-δ(0,Λ')
    return(Sv)
def _f1(self,Aev,ν,Λp,Λpp):
    """Convert an average band emission rate a band_strength"""
    Sv = Aev/2.026e-6/v**3
    Sv[(Λp==0)&(Λpp!=0)] /= 2.
    return(Sv)
prototypes['Sv'] =dict(description="Band strength (⟨vp|Re|vpp⟩**2, au)", kind='f',  fmt='<10.5e', infer={('Sij','SJ'): lambda self,Sij,SJ: Sij/SJ, ('μ',):lambda self,μ:μ**2, ('fv','ν','Λp','Λpp')  : lambda self,fv,ν,Λp,Λpp  : band_fvalue_to_band_strength(fv,ν,Λp,Λpp), ('fv','νv','Λp','Λpp') : lambda self,fv,νv,Λp,Λpp : band_fvalue_to_band_strength(fv,νv,Λp,Λpp), ('Aev','ν','Λp','Λpp') : lambda self,Aev,ν,Λp,Λpp : band_emission_rate_to_band_strength(Aev,ν,Λp,Λpp ), ('Aev','νv','Λp','Λpp'): lambda self,Aev,νv,Λp,Λpp: band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),},)
def _f1(self,f,SJ,Jpp,Λp,Λpp):
    """Get band fvalues from line strength"""
    fv = f/SJ*(2.*Jpp+1.)       # correct? What about 2S+1?
    fv[(Λpp==0)&(Λp!=0)] *= 2
    return(fv)
prototypes['fv'] = dict(description="Band f-value (dimensionless)", kind='f',  fmt='<10.5e', infer={('Sv','ν','Λp','Λpp'):  lambda self,Sv,ν,Λp,Λpp :  band_strength_to_band_fvalue(Sv,ν, Λp,Λpp), ('Sv','νv','Λp','Λpp'): lambda self,Sv,νv,Λp,Λpp:  band_strength_to_band_fvalue(Sv,νv,Λp,Λpp), ('f','SJ','Jpp','Λp','Λpp'): _f1,})
prototypes['Aev'] =dict(description="Einstein A coefficient / emission rate averaged over a band (s-1).", kind='f',  fmt='<10.5e', infer={('Sv','ν' ,'Λp','Λpp'): lambda self,Sv,ν ,Λp,Λpp: band_strength_to_band_emission_rate(Sv,ν ,Λp,Λpp), ('Sv','νv','Λp','Λpp'): lambda self,Sv,νv,Λp,Λpp: band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp),},) 
prototypes['σv'] =dict(description="Integrated cross section of an entire band (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('fv',):lambda self,fv: band_fvalue_to_band_cross_section(fv),},)
prototypes['Sij'] =dict(description="Line strength (au)", kind='f',  fmt='<10.5e', infer={('Sv','SJ'): lambda self,Sv,SJ:  Sv*SJ, ('f','ν','Jpp'): lambda self,f,ν,Jpp: f/3.038e-6/ν*(2*Jpp+1), ('Ae','ν','Jp'): lambda self,Ae,ν,Jp: Ae/(2.026e-6*ν**3/(2*Jp+1)),})
prototypes['Ae'] =dict(description="Einstein A coefficient / emission rate (s-1).", kind='f',  fmt='<10.5e', infer={('f','ν','Jp','Jpp'): lambda self,f,ν,Jp,Jpp: f*0.666886/(2*Jp+1)*(2*Jpp+1)*ν**2, ('Sij','ν','Jp'): lambda self,Sij,ν,Jp: Sij*2.026e-6*ν**3/(2*Jp+1),},)
prototypes['FCfactor'] =dict(description="Franck-Condon factor (dimensionless)", kind='f',  fmt='<10.5e', infer={('χp','χpp','R'): lambda self,χp,χpp,R: np.array([integrate.trapz(χpi*χppi,R)**2 for (χpi,χppi) in zip(χp,χpp)]),},)
prototypes['Rcentroid'] =dict(description="R-centroid (Å)", kind='f',  fmt='<10.5e', infer={('χp','χpp','R','FCfactor'): lambda self,χp,χpp,R,FCfactor: np.array([integrate.trapz(χpi*R*χppi,R)/integrate.trapz(χpi*χppi,R) for (χpi,χppi) in zip(χp,χpp)]),},)
def _f0(self,Sp,Spp,Ωp,Ωpp,Jp,Jpp):
    if not (np.all(Sp==0) and np.all(Spp==0)): raise InferException('Honl-London factors only defined here for singlet states.')
    try:
        return(quantum_numbers.honl_london_factor(Ωp,Ωpp,Jp,Jpp))
    except ValueError as err:
        if str(err)=='Could not find correct Honl-London case.':
            raise InferException('Could not compute rotational line strength')
        else:
            raise(err)
# prototypes['SJ'] = dict(description="Rotational line strength (dimensionless)", kind='f',  fmt='<10.5e', infer= {('Sp','Spp','Ωp','Ωpp','Jp','Jpp'): _f0,})
# prototypes['τ'] = dict(description="Integrated optical depth (cm-1)", kind='f',  fmt='<10.5e', infer={('σ','column_densitypp'):lambda self,σ,column_densitypp: σ*column_densitypp,},)
# prototypes['I'] = dict(description="Integrated emission energy intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp','ν'):lambda self,Ae,populationp,column_densityp,ν: Ae*populationp*column_densityp*ν,},)
# prototypes['Ip'] = dict(description="Integrated emission photon intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp'):lambda self,Ae,populationp,column_densityp: Ae*populationp*column_densityp,},)
# prototypes['σd'] = dict(description="Integrated photodissociation cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('σ','ηdp'):lambda self,σ,ηdp: σ*ηdp,})
# prototypes['Sabs'] = dict(description="Absorption intensity (cm-1/(molecule.cm-1)).", kind='f',  fmt='<10.5e', infer={})

## vibrational interaction energies
prototypes['ηv'] = dict(description="Reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ξv'] = dict(description="Reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ηDv'] = dict(description="Higher-order reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ξDv'] = dict(description="Higher-roder reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})




## add infer functions -- could add some of these to above
prototypes['ν']['infer']['E_u','E_l'] = lambda self,Eu,El: Eu-El
prototypes['E_l']['infer']['E_u','ν'] = lambda self,Eu,ν: Eu-ν
prototypes['E_u']['infer']['E_l','ν'] = lambda self,El,ν: El+ν
prototypes['Γ']['infer']['Γ_u','Γ_l'] = lambda self,Γu,Γl: Γu+Γl
prototypes['Γ_l']['infer']['Γ','Γ_u'] = lambda self,Γ,Γu: Γ-Γu
prototypes['Γ_u']['infer']['Γ','Γ_l'] = lambda self,Γ,Γl: Γ-Γl
prototypes['J_u']['infer']['J_l','ΔJ'] = lambda self,J_l,ΔJ: J_l+ΔJ
# prototypes['Teq_u']['infer']['Teq'] = lambda self,Teq: Teq
# prototypes['Teq_l']['infer']['Teq'] = lambda self,Teq: Teq
prototypes['partition_source_l']['infer']['partition_source'] = lambda self,x: x
prototypes['partition_source_u']['infer']['partition_source'] = lambda self,x: x
prototypes['Tex_u']['infer']['Teq'] = lambda self,Teq: Teq
prototypes['Nself_u']['infer']['Nself'] = lambda self,Nself: Nself
prototypes['Nself_l']['infer']['Nself'] = lambda self,Nself: Nself
prototypes['species_l']['infer']['species'] = lambda self,species: species
prototypes['species_u']['infer']['species'] = lambda self,species: species
prototypes['ΔJ']['infer']['J_u','J_l'] = lambda self,J_u,J_l: J_u-J_l
prototypes['partition']['infer']['partition_l'] = lambda self,partition_l:partition_l
prototypes['partition']['infer']['partition_u'] = lambda self,partition_u:partition_u
prototypes['partition_l']['infer']['partition'] = lambda self,partition:partition
prototypes['partition_u']['infer']['partition'] = lambda self,partition:partition





level_suffix = {'upper':'_u','lower':'_l'}

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
        retval.append(key+'_l')
        retval.append(key+'_u')
    return retval


class GenericLine(levels.Base):
    _levels_class = levels.GenericLevel
    _init_keys = _unique(_expand_level_keys_to_upper_lower(_levels_class)
                         + ['species', 'mass',
                            'ν',#'λ',
                            'ΔJ',
                            'f','σ','S','S296K','τ',
                            'γair','δair','nair','γself',
                            'Pself', 'Pair', 'Nself',
                            'Teq','Ttr','partition','partition_source',
                            'Γ','ΓD',
                            ])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

    # def __init__(self,*args,**kwargs):
        # levels.Base.__init__(self,*args,**kwargs)

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
            gaussian_method='python', #'fortran stepwise', 'fortran', 'python'
            voigt_method='wofz',   
            use_multiprocessing=False, # might see a speed up
            use_cache=False,    # is it actually any faster?!?
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
            ΓG = self[ΓG]
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
            ΓL = self[ΓL]
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
            if 'calculate_spectrum' not in self._cache:
                self._cache['calculate_spectrum'] = {}
            cache = self._cache['calculate_spectrum']
            cache_keys = (xkey,ykey,ΓLin,ΓGin)
        ## test is cache exist and is usable
        if (not use_cache   # no cache requested
            or len(cache) == 0         #  first run
            or len(cache['y'])!=len(x)   # x domain is the same length as the last cached calculation 
            or not np.all(cache['x']==x)):   # x domain is the same as the last cached calculation -- TEST REQUIRES MUCH MEMORY?
            ## comput entire spectrum with out cache
            ##
            ## get a default frequency scale if none provided
            if x is None:
                x = np.linspace(max(0,self[xkey].min()-10.),self[xkey].max()+10.,nx)
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
                elif voigt_method=='fortran Doppler' and ΓLin=='Γ' and ΓGin=='ΓD':
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
            if self.verbose:
                print('calculate_spectrum: using cached spectrum')
            ## Compute using existing cache. Determine which lines
            ## need to be updated and update them. Do this row-by-row
            ## using recarray equality. Need to remove references to
            ## keys containing NaNs
            i = np.any([self[key]!=cache[key] for key in cache_keys],0) # changed lines
            if (ykey == 'τ'     # absorption
                and self.is_inferred_from('τ','S') and self.is_inferred_from('τ','Nself_l') # τ is computed from absorption strength 
                and len(np.unique(cache['Nself_l']/self['Nself_l'])) == 1 # all column_densitypp has changed by the same factor
                and np.all(self['S']==cache['S']) # no σ has not changed
                and np.all([self[key]==cache[key] for key in cache_keys if key != 'τ'])): # frequencies and widths have not changed
                ## all lines have changed column density but nothing
                ## else. Rescale spectrum by change common change in
                ## column density.
                if self.verbose:
                    print('calculate_spectrum: using cached transmission spectrum')
                y = cache['y']/cache['Nself_l']*self['Nself_l'][0]
            elif (False and ykey == 'I'     # emission -- UNSTABLE definition
                  and self.vector_data['I'].inferred_from == {'Ae','column_densityp'}
                  and np.sum(i) == len(i)            # all data has changed
                  and len(np.unique(cache['column_densityp']/self['column_densityp'])) == 1 # all column_densityp has changed by the same factor
                  and np.all(cache['Ae']==self['Ae']) # Ae are the same
                  and np.all([self[key]==cache[key] for key in cache_keys if key != 'I'])): # frequencies and widths have not changed
                if self.verbose:
                    print('calculate_spectrum: using emission column_densitypp shortcut')
                y = cache['y']/cache['column_densityp'][0]*self['column_densityp'][0]
            elif (np.sum(i)/len(i))>0.25:
                ## many lines have changed, just recompute all
                self._cache.pop('calculate_spectrum')
                x,y = self.calculate_spectrum(**all_args,use_cache=use_cache)
            elif np.sum(i)==0:
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
            if self.verbose:
                print('calculate_spectrum: saving cache')
            cache['x'],cache['y']  = x,y
            for key in cache_keys:
                cache[key] = copy(self[key])
            ## save these for rescale column density shortcuts
            if (ykey == 'τ'     # absorption
                and self.is_inferred_from('τ','S') and self.is_inferred_from('τ','Nself_l')): # τ is computed from absorption strength 
                cache['Nself_l'],cache['S'] = copy(self['Nself_l'][0]),copy(self['S'])
            if False and (ykey == 'I' and self.vector_data['I'].inferred_from == {'Ae','column_densityp'}):
                cache['column_densityp'] = copy(self['column_densityp'])
                cache['Ae'] = copy(self['Ae'])
        return x,y

    def get_levels(self,upper_or_lower):
        """Get all data corresponding to 'upper' or 'lower' level in
        self."""
        levels = self._levels_class()
        assert upper_or_lower in ('upper','lower'),f'upper_or_lower must be "upper" or "lower", not {repr(upper_or_lower)}'
        for key in self.keys():
            if (level_key:=_get_key_without_level_suffix(upper_or_lower,key)) is not None:
                levels[level_key] = self[key]
        return levels

    def get_upper_levels(self):
        return self.get_levels('upper')

    def get_lower_levels(self):
        return self.get_levels('lower')

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
        i = hitran.molparam.find(species_ID=data['Mol'],local_isotopologue_ID=data['Iso'])
        new['species'] =  hitran.molparam['isotopologue'][i]
        ## remove natural abundance weighting
        new['S296K'] /=  hitran.molparam['natural_abundance'][i]
        ## ## interpret quantum numbers and insert into some kind of transition, this logic is in its infancy
        ## ## standin for diatomics
        ## kw['v'+'_u'] = data['V_u']
        ## kw['v'+'_l'] = data['V_l']
        ## branches = {'P':-1,'Q':0,'R':+1}
        ## ΔJ,J_l = [],[]
        ## for Q_l in data['Q_l']:
        ##     branchi,Jli = Q_l.split()
        ##     ΔJ.append(branches[branchi])
        ##     J_l.append(Jli)
        ## kw['ΔJ'] = np.array(ΔJ,dtype=int)
        ## kw['J'+'_l'] = np.array(J_l,dtype=float)
        self.extend(**new)

    def generate_from_levels(self,lu,ll):
        for key in lu:
            if lu.is_scalar(key):
                self[key+'_u'] = lu[key]
            else:
                self[key+'_u'] = np.ravel(np.tile(lu.as_vector(key),len(ll)))
        for key in ll:
            if ll.is_scalar(key):
                self[key+'_l'] = ll[key]
            else:
                self[key+'_l'] = np.ravel(np.repeat(ll.as_vector(key),len(lu)))



class HeteronuclearDiatomicElectronicLine(GenericLine):
    _levels_class = levels.HeteronuclearDiatomicElectronicLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + GenericLine._init_keys
        + ['Teq', 'Tex', 'Ttr', 'partition_source', 'partition', 'L', 'γair', 'δair', 'γself', 'nair', 'Pself', 'Pair', 'Nself',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HeteronuclearDiatomicVibrationalLine(HeteronuclearDiatomicElectronicLine):
    _levels_class = levels.HeteronuclearDiatomicVibrationalLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + HeteronuclearDiatomicElectronicLine._init_keys
        + ['ν','νv','μv',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HeteronuclearDiatomicRotationalLine(HeteronuclearDiatomicVibrationalLine):
    _levels_class = levels.HeteronuclearDiatomicRotationalLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + HeteronuclearDiatomicVibrationalLine._init_keys
        + ['branch', 'ΔJ', 'Γ', 'ΓD', 'f','σ','S','S296K', 'τ', 'Ae','τa', 'Sij',])
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

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
            'E'+'_l':data['E_l'],
            'g'+'_u':data['g_u'],
            'g'+'_l':data['g_l'],
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
        kw['v'+'_u'] = data['V_u']
        kw['v'+'_l'] = data['V_l']
        branches = {'P':-1,'Q':0,'R':+1}
        ΔJ,J_l = [],[]
        for Q_l in data['Q_l']:
            branchi,Jli = Q_l.split()
            ΔJ.append(branches[branchi])
            J_l.append(Jli)
        kw['ΔJ'] = np.array(ΔJ,dtype=int)
        kw['J'+'_l'] = np.array(J_l,dtype=float)
        self.extend(**kw)

class HomonuclearDiatomicElectronicLine(HeteronuclearDiatomicElectronicLine):
    _levels_class = levels.HomonuclearDiatomicElectronicLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + HeteronuclearDiatomicElectronicLine._init_keys)
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HomonuclearDiatomicVibrationalLine(HeteronuclearDiatomicVibrationalLine):
    _levels_class = levels.HomonuclearDiatomicVibrationalLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + HeteronuclearDiatomicVibrationalLine._init_keys)
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

class HomonuclearDiatomicRotationalLine(HeteronuclearDiatomicRotationalLine):
    _levels_class = levels.HomonuclearDiatomicRotationalLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + HeteronuclearDiatomicRotationalLine._init_keys)
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}

# class TriatomicDinfh(Base):

    # prototypes = {key:copy(prototypes[key]) for key in (
        # list(Base.prototypes)
        # + _expand_level_keys_to_upper_lower(levels.TriatomicDinfh))}

class LinearTriatomicLine(GenericLine):
    _levels_class = levels.LinearTriatomicLevel
    _init_keys = _unique(
        _expand_level_keys_to_upper_lower(_levels_class)
        + GenericLine._init_keys)
    prototypes = {key:copy(prototypes[key]) for key in _init_keys}
    default_zkeys=(
        'species',
        'label_u','v1_u','v2_u','v3_u','l2_u',
        'label_l','v1_l','v2_l','v3_l','l2_l',
        )

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
        i = hitran.molparam.find(species_ID=data['Mol'],local_isotopologue_ID=data['Iso'])
        new['species'] =  hitran.molparam['isotopologue'][i]
        ## remove natural abundance weighting
        new['S296K'] /=  hitran.molparam['natural_abundance'][i]
        ## interpret quantum numbers -- see rothman2005
        new['v1_u'] = [t[7:9] for t in data['V_u']]
        new['v2_u'] = [t[9:11] for t in data['V_u']]
        new['l2_u'] = [t[11:13] for t in data['V_u']]
        new['v3_u'] = [t[13:15] for t in data['V_u']]
        new['v1_l'] = [t[7:9] for t in data['V_l']]
        new['v2_l'] = [t[9:11] for t in data['V_l']]
        new['l2_l'] = [t[11:13] for t in data['V_l']]
        new['v3_l'] = [t[13:15] for t in data['V_l']]
        branches = {'P':-1,'Q':0,'R':+1}
        ΔJ,J_l = [],[]
        for Q_l in data['Q_l']:
            branchi,Jli = Q_l[5],Q_l[6:] 
            ΔJ.append(branches[branchi])
            J_l.append(Jli)
        new['ΔJ'] = np.array(ΔJ,dtype=int)
        new['J'+'_l'] = np.array(J_l,dtype=float)
        # ## add data to self
        self.extend(**new)

# class TriatomicDinfh(Base):

    # prototypes = {key:copy(prototypes[key]) for key in (
        # list(Base.prototypes)
        # + _expand_level_keys_to_upper_lower(levels.TriatomicDinfh))}

