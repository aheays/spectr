import itertools
from copy import copy,deepcopy
from pprint import pprint

import numpy as np
from numpy import nan,array,linspace
from scipy import constants

# from . import *
# from .dataset import Dataset
from . import tools
from . import levels
from . import lineshapes
from . import tools
from .tools import file_to_dict,vectorise,cache
from . import hitran
from . import database
from . import plotting
from .conversions import convert
from .exceptions import InferException
# from .lines import prototypes
from . import levels
from .dataset import Dataset

prototypes = {}

## copy some direct from levels
for key in (
        'species','point_group',
        'mass','reduced_mass',
        'Z',
        'ΓD',
):
    prototypes[key] = copy(levels.prototypes[key])

## import all from levels with suffices added
for key,val in levels.prototypes.items():
    tval = deepcopy(val)
    tval['infer'] = [(tuple(key+'_u' for key in tools.ensure_iterable(dependencies)),function)
                     for dependencies,function in val['infer']]
    prototypes[key+'_u'] = tval
    tval['infer'] = [(tuple(key+'_l' for key in tools.ensure_iterable(dependencies)),function)
                     for dependencies,function in val['infer']]
    prototypes[key+'_l'] = tval

## add lines things
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='U', cast=str, fmt='<10s')
prototypes['f'] = dict(description="Line f-value (dimensionless)",kind='f',fmt='<10.5e', infer=[
    (('Ae','ν','g_u','g_l'),lambda self,Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2),
    (('Sij','ν','J_l'), lambda self,Sij,ν,J_l: 3.038e-6*ν*Sij/(2*J_l+1)), 
    (('σ','α_l'),lambda self,σ,α_l: σ*1.1296e12/α_l,)])
prototypes['σ'] = dict(description="Spectrally-integrated photoabsorption cross section (cm2.cm-1).", kind='f', fmt='<10.5e',infer=[
    (('τa','Nself_l'),lambda self,τ,column_densitypp: τ/column_densitypp), 
    (('f','α_l'),lambda self,f,α_l: f/1.1296e12*α_l),
    (('S','ν','Tex'),lambda self,S,ν,Tex,: S/(1-np.exp(-convert(constants.Boltzmann,'J','cm-1')*ν/Tex))),])
## prototypes['σ'] =dict(description="Integrated cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('τ','column_densitypp'):lambda self,τ,column_densitypp: τ/column_densitypp, ('f','populationpp'):lambda self,f,populationpp: f/1.1296e12*populationpp,})
def _f0(self,S296K,species,Z,E_l,Tex,ν):
    """See Eq. 9 of simeckova2006"""
    Z296K = hitran.get_partition_function(species,296)
    c = convert(constants.Boltzmann,'J','cm-1') # hc/kB
    return (S296K
            *((np.exp(-E_l/(c*Tex))/Z)*(1-np.exp(-c*ν/Tex)))
            /((np.exp(-E_l/(c*296))/Z296K)*(1-np.exp(-c*ν/296))))
prototypes['S'] = dict(description="Spectral line intensity (cm or cm-1/(molecular.cm-2) ", kind='f', fmt='<10.5e', infer=[(('S296K','species','Z','E_l','Tex_l','ν'),_f0,)])
prototypes['S296K'] = dict(description="Spectral line intensity at 296K reference temperature ( cm-1/(molecular.cm-2) ). This is not quite the same as HITRAN which also weights line intensities by their natural isotopologue abundance.", kind='f', fmt='<10.5e', infer=[],cast=tools.cast_abs_float_array)
## Preferentially compute τ from the spectral line intensity, S,
## rather than than the photoabsorption cross section, σ, because the
## former considers the effect of stimulated emission.
prototypes['τ'] = dict(description="Integrated optical depth including stimulated emission (cm-1)", kind='f', fmt='<10.5e',
                       infer=[
                           (('S','Nself_l'),lambda self,S,Nself_l: S*Nself_l,),
                           (('σ','Nself_l'),lambda self,σ,Nself_l: σ*Nself_l,),
                       ],)
prototypes['τa'] = dict(description="Integrated optical depth from absorption only (cm-1)", kind='f', fmt='<10.5e', infer=[(('σ','Nself_l'),lambda self,σ,Nself_l: σ*Nself_l,)],)
prototypes['Ae'] = dict(description="Radiative decay rate (s-1)", kind='f', fmt='<10.5g', infer=[(('f','ν','g_u','g_l'),lambda self,f,ν,g_u,g_l: f/(1.49951*g_u/g_l/ν**2)),(('At','Ad'), lambda self,At,Ad: At-Ad,)])
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind='f', fmt='0.2f', infer=[])
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq,)])
prototypes['Ttr'] = dict(description="Translational temperature (K)", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq,)])
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind='f', fmt='>+4g', infer=[(('Jp','Jpp'),lambda self,Jp,Jpp: Jp-Jpp,)],)


## column 
prototypes['L'] = dict(description="Optical path length (m)", kind='f', fmt='0.5f', infer=[])
prototypes['Nself'] = dict(description="Column density (cm-2)",kind='f',fmt='<11.3e', infer=[(('pself','L','Teq'), lambda self,pself,L,Teq: convert((pself*L)/(database.constants.Boltzmann*Teq),'m-2','cm-2'),)])

## pressure broadening and shift parameters
prototypes['pair'] = dict(description="Pressure of air (Pa)", kind='f', fmt='0.5f',units='Pa',infer=[])
prototypes['γ0air'] = dict(description="Pressure broadening coefficient in air (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0air'] = dict(description="Pressure broadening temperature dependence in air (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0air'] = dict(description="Pressure shift coefficient in air (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0air'] = dict(description="Pressure shift temperature dependence in air (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['Γair'] = dict(description="Pressure broadening due to air (cm-1 FWHM)" , kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array, infer=[(('γ0air','nγ0air','pair','Ttr'),lambda self,γ,n,P,T: (296/T)**n*2*γ*convert(P,'Pa','atm')),])
prototypes['Δνair'] = dict(description="Pressure shift due to air (cm-1)" , kind='f', fmt='<10.5g',infer=[(('δ0air','nδ0air','pair','Ttr'),lambda self,δ,n,P,T: (296/T)**n*δ*convert(P,'Pa','atm')),])

prototypes['νvc'] = dict(description="Frequency of velocity changing collisions (which profile?) (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)

prototypes['pself'] = dict(description="Pressure of self (Pa)", kind='f', fmt='0.5f',units='Pa',infer=[])
prototypes['γ0self'] = dict(description="Pressure broadening coefficient in self (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0self'] = dict(description="Pressure broadening temperature dependence in self (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0self'] = dict(description="Pressure shift coefficient in self (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0self'] = dict(description="Pressure shift temperature dependence in self (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['Γself'] = dict(description="Pressure broadening due to self (cm-1 FWHM)" , kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array,infer=[(('γ0self','nγ0self','pself','Ttr'),lambda self,γ0,n,P,T: (296/T)**n*2*γ0*convert(P,'Pa','atm')),])
prototypes['Δνself'] = dict(description="Pressure shift due to self (cm-1 HWHM)" , kind='f', fmt='<10.5g',infer=[(('δ0self','nδ0self','pself','Ttr'),lambda self,δ0,n,P,T: (296/T)**n*δ0*convert(P,'Pa','atm')),])

prototypes['pX'] = dict(description="Pressure of X (Pa)", kind='f', fmt='0.5f',units='Pa',infer=[])
prototypes['γ0X'] = dict(description="Pressure broadening coefficient in X (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0X'] = dict(description="Pressure broadening temperature dependence in X (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0X'] = dict(description="Pressure shift coefficient in X (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0X'] = dict(description="Pressure shift temperature dependence in X (cm-1.atm-1 HWHM)", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['ΓX'] = dict(description="Pressure broadening due to X (cm-1 FWHM)" , kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array,infer=[(('γ0X','nγ0X','pX','Ttr'),lambda self,γ0,n,P,T: (296/T)**n*2*γ0*convert(P,'Pa','atm')),])
prototypes['ΔνX'] = dict(description="Pressure shift due to species X (cm-1 HWHM)" , kind='f', fmt='<10.5g',infer=[(('δ0X','nδ0X','pX','Ttr'),lambda X,δ0,n,P,T: (296/T)**n*δ0*convert(P,'Pa','atm')),])

## HITRAN encoded pressure and temperature dependent Hartmann-Tran
## line broadening and shifting coefficients
prototypes['HT_HITRAN_X'] = dict(description='Broadening species for a HITRAN-encoded Hartmann-Tran profile',kind='U')
def _f0(x):
    """Limiting values!!! Otherwise lineshape is bad -- should investigate this."""
    x = np.asarray(x,dtype=float)
    x[x<1e-50] = 1e-50
    return x
prototypes['HT_HITRAN_p'] = dict(description='Pressure  HITRAN-encoded Hartmann-Tran profile (atm)',kind='f',units='atm',cast=_f0,infer=[('pX',lambda self,pX:convert(pX,'Pa','atm'))])
prototypes['HT_HITRAN_Tref'] = dict(description='Reference temperature for a HITRAN-encoded Hartmann-Tran profile ',units='K',kind='f')
prototypes['HT_HITRAN_γ0'] = dict(description='Speed-averaged halfwidth in temperature range around Tref due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=tools.cast_abs_float_array)
prototypes['HT_HITRAN_n'] = dict(description='Temperature dependence exponent of γ0 in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f')
def _f0(x):
    """Limiting values!!! Otherwise lineshape is bad -- should investigate this."""
    x = np.asarray(x,dtype=float)
    x[x<1e-10] = 1e-10
    # x[x>0.03] = 0.03
    return x
prototypes['HT_HITRAN_γ2'] = dict(description='Speed-dependence of the halfwidth in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=_f0)
prototypes['HT_HITRAN_δ0'] = dict(description='Speed-averaged line shift in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f')
prototypes['HT_HITRAN_δp'] = dict(description='Linear temperature dependence coefficient for δ0 in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm.K-1',kind='f')
prototypes['HT_HITRAN_δ2'] = dict(description='Speed-dependence of the line shift in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f')
prototypes['HT_HITRAN_νVC'] = dict(description='Frequency of velocity changing collisions in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=tools.cast_abs_float_array)
prototypes['HT_HITRAN_κ'] = dict(description='Temperature dependence of νVC in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f')
# def _f0(x):
    # """Limiting values!!! Otherwise lineshape is bad -- should investigate this."""
    # x = np.abs(np.asarray(x),dtype=float)
    # x[x>1] = 0.99999
    # return x
prototypes['HT_HITRAN_η'] = dict(description='Correlation parameter in HT in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f',cast=tools.cast_abs_float_array,default_step=1e-5)
prototypes['HT_HITRAN_Y'] = dict(description='First-order (Rosenkranz) line coupling coefficient in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile; air-(self-) broadened case',units='cm-1.atm-1',kind='f')

## coefficients of the Hartmann-Tran lineshape
prototypes['HT_Γ0'] = dict(description='Speed-averaged halfwidth for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HT_HITRAN_p','HT_HITRAN_γ0','HT_HITRAN_Tref','Ttr','HT_HITRAN_n'),lambda self,p,γ0,Tref,T,n: γ0*p*(Tref/T)**n)])
prototypes['HT_Γ2'] = dict(description='Speed-dependence for the halfwidth for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HT_HITRAN_p','HT_HITRAN_γ2',),lambda self,p,γ2: p*γ2),])
prototypes['HT_Δ0'] = dict(description='Speed-averaged line shift for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HT_HITRAN_p','HT_HITRAN_δ0','HT_HITRAN_δp','HT_HITRAN_Tref','Ttr'),lambda self,p,δ0,δp,Tref,T: p*(δ0+δp*(T-Tref))),])
prototypes['HT_Δ2'] = dict(description='Speed-dependence for the line shift for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HT_HITRAN_p','HT_HITRAN_δ2'),lambda self,p,δ2: p*δ2),])
prototypes['HT_νVC'] = dict(description='Frequency of velocity changing collisions for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HT_HITRAN_p','HT_HITRAN_νVC','HT_HITRAN_Tref','Ttr','HT_HITRAN_κ'),lambda self,p,νVC,Tref,T,κ: p*νVC*(Tref/T)**κ),])
prototypes['HT_η'] = dict(description='Correlation parameter for the Hartmann-Tran profile',units='dimensionless',kind='f',infer=[(('HT_HITRAN_η',),lambda self,η:η),])

## linewidths
prototypes['Γ'] = dict(description="Total Lorentzian linewidth of level or transition (cm-1 FWHM)" , kind='f', fmt='<10.5g',infer=[
    ## manually input all permutations of broadening affects --  could
    ## use 'self' in a function but then infer connections will not be
    ## made
    (('Γself','Γair','ΓX'),lambda self,Γ0,Γ1,Γ2: Γ0+Γ1+Γ2),
    (('Γself','Γair'),lambda self,Γ0,Γ1: Γ0+Γ1),
    (('Γself','ΓX'),lambda self,Γ0,Γ1: Γ0+Γ1),
    (('Γair','ΓX'),lambda self,Γ0,Γ1: Γ0+Γ1),
    ('Γself',lambda self,Γ0: Γ0),
    ('Γair' ,lambda self,Γ0: Γ0),
    ('ΓX',lambda self,Γ0: Γ0),
])
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind='f',fmt='<10.5g', infer=[(('mass','Ttr','ν'), lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,)])

## line frequencies
prototypes['ν0'] = dict(description="Transition wavenumber in a vacuum (cm-1)", kind='f', fmt='>0.6f', default_step=1e-3, infer=[('ν',_f0)])
prototypes['ν'] = dict(description="Transition wavenumber (cm-1)", kind='f', fmt='>0.6f', infer=[
    ## manually input all permutations of broadening affects -- could
    ## use 'self' in a function but then infer connections will not be
    ## made
    (('ν0','Δνself','Δνair','ΔνX'),lambda self,ν0,Δν0,Δν1,Δν2: ν0+Δν0+Δν1+Δν2),
    (('ν0','Δνself','Δνair'),lambda self,ν0,Δν0,Δν1: ν0+Δν0+Δν1),
    (('ν0','Δνself','ΔνX'),lambda self,ν0,Δν0,Δν1: ν0+Δν0+Δν1),
    (('ν0','Δνair','ΔνX'),lambda self,ν0,Δν0,Δν1: ν0+Δν0+Δν1),
    (('ν0','Δνself'),lambda self,ν0,Δν0: ν0+Δν0),
    (('ν0','Δνair' ),lambda self,ν0,Δν: ν0+Δν),
    (('ν0','ΔνX'),lambda self,ν0,Δν0: ν0+Δν0),
    (('ν0',),lambda self,ν0: ν0),
])

## partition functions
def _f3(self,species,Tex,E_u,E_l,g_u,g_l):
    """Compute partition function from data in self."""
    if self.Zsource != 'self':
        raise InferException(f'Zsource not "self".')
    Z = np.full(len(species),0.)
    ## calculate separately for all (species,Tex) combinations
    for (speciesi,Texi),i in tools.unique_combinations_mask(species,Tex):
        i = tools.find(i)
        kT = convert(constants.Boltzmann,'J','cm-1')*Texi
        ## sum for all unique upper levels
        k = []
        for qn,j in tools.unique_combinations_mask(
                *[self[key+'_u'][i] for key in self._levels_class.defining_qn]):
            k.append((i[j])[0])
        Z[i] += np.sum(g_u[k]*np.exp(-E_u[k]/kT))
        ## sum for all unique lower levels
        k = []
        for qn,j in tools.unique_combinations_mask(
                *[self[key+'_l'][i] for key in self._levels_class.defining_qn]):
            k.append((i[j])[0])
        Z[i] += np.sum(g_l[k]*np.exp(-E_l[k]/kT))
    return Z
@vectorise(cache=True,vargs=(1,2))
def _f5(self,species,Tex):
    if self.attributes['Zsource'] != 'HITRAN':
        raise InferException(f'Zsource not "HITRAN".')
    from . import hitran
    return hitran.get_partition_function(species,Tex)
prototypes['Z'] = dict(description="Partition function including both upper and lower levels.", kind='f', fmt='<11.3e', infer=[(('species','Tex'),_f5),( ('species','Tex','E_u','E_l','g_u','g_l'),_f3,)])

## vibrational transition frequencies
prototypes['νv'] = dict(description="Electronic-vibrational transition wavenumber (cm-1)", kind='f', fmt='>11.4f', infer=[(('Tvp','Tvpp'), lambda self,Tvp,Tvpp: Tvp-Tvpp),( ('λv',), lambda self,λv: convert_units(λv,'nm','cm-1'),)])
prototypes['λv'] = dict(description="Electronic-vibrational transition wavelength (nm)", kind='f', fmt='>11.4f', infer=[(('νv',), lambda self,νv: convert_units(νv,'cm-1','nm'),)],)

## transition strengths
prototypes['M']   = dict(description="Pointer to electronic transition moment (au).", kind='O', infer=[])
prototypes['Mv']   = dict(description="Electronic transition moment for this vibronic level (au).", kind='f', fmt='<10.5e', infer=[(('μ','FCfactor'), lambda self,μ,FCfactor: μ/np.sqrt(FCfactor),)])
prototypes['μv']  = dict(description="Electronic-vibrational transition moment (au)", kind='f',  fmt='<10.5e', infer=[(('M','χp','χpp','R'), lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),)],) # could infer from S but then sign would be unknown
prototypes['μ']   = dict(description="Electronic-vibrational-rotational transition moment (au)", kind='f',  fmt='<10.5e', infer=[(('M','χp','χpp','R'), lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),)],) # could infer from S but then sign would be unknown
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
prototypes['Sv'] =dict(description="Band strength (⟨vp|Re|vpp⟩**2, au)", kind='f',  fmt='<10.5e', infer=[
    (('Sij','SJ'), lambda self,Sij,SJ: Sij/SJ),
    ( ('μ',),lambda self,μ:μ**2),
    (('fv','ν','Λp','Λpp'),lambda self,fv,ν,Λp,Λpp: band_fvalue_to_band_strength(fv,ν,Λp,Λpp)),
    (('fv','νv','Λp','Λpp'),lambda self,fv,νv,Λp,Λpp : band_fvalue_to_band_strength(fv,νv,Λp,Λpp)),
    (('Aev','ν','Λp','Λpp'),lambda self,Aev,ν,Λp,Λpp : band_emission_rate_to_band_strength(Aev,ν,Λp,Λpp )),
    ( ('Aev','νv','Λp','Λpp'), lambda self,Aev,νv,Λp,Λpp: band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),)],)
def _f1(self,f,SJ,Jpp,Λp,Λpp):
    """Get band fvalues from line strength"""
    fv = f/SJ*(2.*Jpp+1.)       # correct? What about 2S+1?
    fv[(Λpp==0)&(Λp!=0)] *= 2
    return(fv)
prototypes['fv'] = dict(description="Band f-value (dimensionless)", kind='f',  fmt='<10.5e', infer=[
    (('Sv','ν','Λp','Λpp'),  lambda self,Sv,ν,Λp,Λpp :  band_strength_to_band_fvalue(Sv,ν, Λp,Λpp)),
    ( ('Sv','νv','Λp','Λpp'), lambda self,Sv,νv,Λp,Λpp:  band_strength_to_band_fvalue(Sv,νv,Λp,Λpp)),
    ( ('f','SJ','Jpp','Λp','Λpp'), _f1,)])
prototypes['Aev'] =dict(description="Einstein A coefficient / emission rate averaged over a band (s-1).", kind='f',  fmt='<10.5e', infer=[(('Sv','ν' ,'Λp','Λpp'), lambda self,Sv,ν ,Λp,Λpp: band_strength_to_band_emission_rate(Sv,ν ,Λp,Λpp)),( ('Sv','νv','Λp','Λpp'), lambda self,Sv,νv,Λp,Λpp: band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp),)],) 
prototypes['σv'] =dict(description="Integrated cross section of an entire band (cm2.cm-1).", kind='f',  fmt='<10.5e', infer=[(('fv',),lambda self,fv: band_fvalue_to_band_cross_section(fv),)],)
prototypes['Sij'] =dict(description=" strength (au)", kind='f',  fmt='<10.5e', infer=[
    (('μ',), lambda self,μ: μ**2),
    (('Sv','SJ'), lambda self,Sv,SJ:  Sv*SJ),
    ( ('f','ν','Jpp'), lambda self,f,ν,Jpp: f/3.038e-6/ν*(2*Jpp+1)),
    ( ('Ae','ν','Jp'), lambda self,Ae,ν,Jp: Ae/(2.026e-6*ν**3/(2*Jp+1)),)])
prototypes['Ae'] =dict(description="Einstein A coefficient / emission rate (s-1).", kind='f',  fmt='<10.5e', infer=[(('f','ν','Jp','Jpp'), lambda self,f,ν,Jp,Jpp: f*0.666886/(2*Jp+1)*(2*Jpp+1)*ν**2),( ('Sij','ν','Jp'), lambda self,Sij,ν,Jp: Sij*2.026e-6*ν**3/(2*Jp+1))],)
prototypes['FCfactor'] =dict(description="Franck-Condon factor (dimensionless)", kind='f',  fmt='<10.5e', infer=[(('χp','χpp','R'), lambda self,χp,χpp,R: np.array([integrate.trapz(χpi*χppi,R)**2 for (χpi,χppi) in zip(χp,χpp)])),],)
prototypes['Rcentroid'] =dict(description="R-centroid (Å)", kind='f',  fmt='<10.5e', infer=[(('χp','χpp','R','FCfactor'), lambda self,χp,χpp,R,FCfactor: np.array([integrate.trapz(χpi*R*χppi,R)/integrate.trapz(χpi*χppi,R) for (χpi,χppi) in zip(χp,χpp)])),])
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
# prototypes['Sabs'] = dict(description="Absorption intensity (cm-1/(molecule.cm-1)).", kind='f',  fmt='<10.5e', infer=[])

## vibrational interaction energies
prototypes['ηv'] = dict(description="Reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer=[])
prototypes['ξv'] = dict(description="Reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer=[])
prototypes['ηDv'] = dict(description="Higher-order reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer=[])
prototypes['ξDv'] = dict(description="Higher-roder reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer=[])




## add infer functions -- could add some of these to above
prototypes['ν']['infer'].append((('E_u','E_l'),lambda self,Eu,El: Eu-El))
prototypes['E_l']['infer'].append((('E_u','ν'),lambda self,Eu,ν: Eu-ν))
prototypes['E_u']['infer'].append((('E_l','ν'),lambda self,El,ν: El+ν))
prototypes['Γ']['infer'].append((('Γ_u','Γ_l'),lambda self,Γu,Γl: Γu+Γl))
prototypes['Γ_l']['infer'].append((('Γ','Γ_u'),lambda self,Γ,Γu: Γ-Γu))
prototypes['Γ_u']['infer'].append((('Γ','Γ_l'),lambda self,Γ,Γl: Γ-Γl))
prototypes['J_u']['infer'].append((('J_l','ΔJ'),lambda self,J_l,ΔJ: J_l+ΔJ))
prototypes['Teq_u']['infer'].append((('Teq'),lambda self,Teq: Teq))
prototypes['Teq_l']['infer'].append((('Teq'),lambda self,Teq: Teq))
prototypes['Tex_u']['infer'].append((('Tex'),lambda self,Tex: Tex))
prototypes['Tex_l']['infer'].append((('Tex'),lambda self,Tex: Tex))
prototypes['Tex']['infer'].append((('Teq'),lambda self,Teq: Teq))
prototypes['Nself_u']['infer'].append((('Nself'),lambda self,Nself: Nself))
prototypes['Nself_l']['infer'].append((('Nself'),lambda self,Nself: Nself))
prototypes['species_l']['infer'].append((('species'),lambda self,species: species))
prototypes['species_u']['infer'].append((('species'),lambda self,species: species))
prototypes['species']['infer'].append((('species_l'),lambda self,species_l: species_l))
prototypes['species']['infer'].append((('species_u'),lambda self,species_u: species_u))
prototypes['ΔJ']['infer'].append((('J_u','J_l'),lambda self,J_u,J_l: J_u-J_l))
prototypes['Z_l']['infer'].append((('Z'),lambda self,Z:Z))
prototypes['Z_u']['infer'].append((('Z'),lambda self,Z:Z))

def _collect_prototypes(*keys,levels_class=None):
    """Take a list and return unique element.s"""
    keys = list(keys)
    for key in levels_class._prototypes:
        keys.append(key+'_l')
        keys.append(key+'_u')
    retval_protoypes = {key:prototypes[key] for key in keys}
    defining_qn = tuple(
        [key+'_u' for key in levels_class.defining_qn]
        +[key+'_l' for key in levels_class.defining_qn]
        )
    return retval_protoypes,defining_qn

class Generic(levels.Base):
    _levels_class = levels.Generic
    _prototypes,defining_qn = _collect_prototypes(
        'species', 'point_group','mass',
        'ν','ν0',
        # 'λ',
        'ΔJ',
        'f','σ','S','S296K','τ','Ae',

        'pair','γ0air','nγ0air','δ0air','nδ0air','Γair','Δνair',
        'pself','γ0self','nγ0self','δ0self','nδ0self','Γself','Δνself',
        'pX','γ0X','nγ0X','δ0X','nδ0X','ΓX','ΔνX',

        # 'νvc',                  # test Rautian profile

        ## test HITRAN Hartmann-Tran
        'HT_HITRAN_X','HT_HITRAN_p', 'HT_HITRAN_Tref', 'HT_HITRAN_γ0', 'HT_HITRAN_n', 'HT_HITRAN_γ2', 'HT_HITRAN_δ0', 'HT_HITRAN_δp', 'HT_HITRAN_δ2', 'HT_HITRAN_νVC', 'HT_HITRAN_κ', 'HT_HITRAN_η', 'HT_HITRAN_Y',
        'HT_Γ0', 'HT_Γ2', 'HT_Δ0', 'HT_Δ2', 'HT_νVC', 'HT_η',

        'Nself',
        'Teq','Tex','Ttr','Z',
        'Γ','ΓD',
        levels_class = _levels_class
    )

    def plot_spectrum(
            self,
            x=None,
            xkey='ν',
            ykey='σ',
            zkeys=None,         # None or list of keys to plot as separate lines
            ΓG='ΓD',
            ΓL='Γ',
            dx=None,
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
                x,y = self.calculate_spectrum(x,ykey='τ',xkey=xkey,ΓG=ΓG,ΓL=ΓL,dx=dx)
                y = np.exp(-y)
            else:
                x,y = self.calculate_spectrum(x,ykey=ykey,xkey=xkey,ΓG=ΓG,ΓL=ΓL,dx=dx)
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
            lineshape=None, # None for auto selection, or else one of ['voigt','gaussian','lorentzian','hartmann-tran']
            nfwhmG=20, # how many Gaussian FWHMs to include in convolution
            nfwhmL=100,         # how many Lorentzian FWHMs to compute
            dx=None, # grid step to use if x grid computed automatically
            nx=10000, # number of grid points to use if x grid computed automatically
            ymin=None, # minimum value of ykey before a line is ignored, None for use all lines
            ncpus = 1, # 1 for single process, more to use up to this amount when computing spectrum
            **set_keys_vals, # set some data first, e..g, the tempertaure
    ):
        """Calculate a spectrum from the data in self. Returns (x,y)."""
        ## set some data
        for key,val in set_keys_vals.items():
            self[key] = val
        ## no lines to add to cross section -- return quickly
        if len(self)==0:
            if x is None:
                return(np.array([]),np.array([]))
            else:
                return(x,np.zeros(x.shape))
        ## get a default frequency scale if none provided
        if x is None:
            if dx is not None:
                x = np.arange(max(0,self[xkey].min()-10.),self[xkey].max()+10.,dx)
            else:
                x = np.linspace(max(0,self[xkey].min()-10.),self[xkey].max()+10.,nx)
        else:
            x = np.asarray(x)
        ## check frequencies, strengths, widths are as expected
        # self.assert_known(xkey,ykey)
        # assert np.all(~np.isnan(self[xkey])),f'NaN values in xkey: {repr(xkey)}'
        # assert np.all(~np.isnan(self[ykey])),f'NaN values in ykey: {repr(ykey)}'
        ## neglect lines out of x-range -- NO ACCOUNTING FOR EDGES!!!!
        i = (self[xkey]>x[0]) & (self[xkey]<x[-1])
        ## neglect lines out of y-range
        if ymin is not None:
            i &= self[ykey] > ymin
        ## get line function and arguments
        if lineshape is None:
            if self.is_known('Γ','ΓD'):
                lineshape = 'voigt'
            elif self.is_known('Γ'):
                lineshape = 'lorentzian'
            elif self.is_known('ΓD'):
                lineshape = 'gaussian'
            else:
                raise Exception("No lineshape has computable widths.") 
        if lineshape == 'voigt':
            line_function = lineshapes.voigt
            line_args = (self[xkey][i],self[ykey][i],self['Γ'][i],self['ΓD'][i])
            line_kwargs = dict(nfwhmL=nfwhmL,nfwhmG=nfwhmG)
        elif lineshape == 'gaussian':
            line_function = lineshapes.gaussian
            line_args = (self[xkey][i],self[ykey][i],self['ΓD'][i])
            line_kwargs = dict(nfwhm=nfwhmG)
        elif lineshape == 'lorentzian':
            line_function = lineshapes.lorentzian
            line_args = (self[xkey][i],self[ykey][i],self['Γ'][i])
            line_kwargs = dict(nfwhm=nfwhmL)
        elif lineshape == 'hartmann-tran':
            if xkey != 'ν':
                raise Exception('Only valid option is ν')
            line_function = lineshapes.hartmann_tran
            line_args = (
                self['ν'][i],
                self[ykey][i],
                self['mass'][i],
                self['Ttr'][i],
                self['HT_νVC'][i],
                self['HT_η'][i],
                self['HT_Γ0'][i],
                self['HT_Γ2'][i],
                self['HT_Δ0'][i],
                self['HT_Δ2'][i],
            )
            line_kwargs = dict(nfwhmL=nfwhmL,nfwhmG=nfwhmG)
        else:
            raise Exception(f'Lineshape {repr(lineshape)} not implemented.')
        ## compute spectrum
        y = lineshapes.calculate_spectrum(
            x,
            line_function,
            *line_args,
            **line_kwargs,
            ncpus=ncpus,
            multiprocess_divide='lines',
        )
        return x,y

    def _get_level(self, u_or_l, reduce_to=None,):
        """Get all data corresponding to 'upper' or 'lower' level in
        self."""
        assert u_or_l in ('u','l')
        levels = self._levels_class()
        for key in self.keys():
            if len(key)>2 and key[-2:]==('_'+u_or_l):
                levels[key[:-2]] = self[key]
        if reduce_to == None:
            pass
        else:
            keys = [key for key in levels.defining_qn if levels.is_known(key)]
            new_levels = self._levels_class()
            for values,i in tools.unique_combinations_mask(*[levels[key] for key in keys]):
                i = tools.find(i)
                if reduce_to == 'first':
                    new_levels.extend(**levels[i[0:1]])
                else:
                    raise ImplementationError()
            levels = new_levels
        return levels

    def get_upper_level(self,reduce_to=None):
        return self._get_level('u',reduce_to)

    def get_lower_level(self,reduce_to=None):
        return self._get_level('l',reduce_to)

    def load_from_hitran(self,filename):
        """Load HITRAN .data."""
        data = hitran.load(filename)
        ## interpret into transition quantities common to all transitions
        new = self.__class__(**{
            'ν0':data['ν'],
            # 'Ae':data['A'],  # Ae data is incomplete but S296K will be complete
            'S296K':data['S'],
            'E_l':data['E_l'],
            'g_u':data['g_u'],
            'g_l':data['g_l'],
            'γ0air':data['γair'], 
            'nγ0air':data['nair'],
            'δ0air':data['δair'],
            'γ0self':data['γself'],
        })
        ## get species
        i = hitran.molparam.find(species_ID=data['Mol'],local_isotopologue_ID=data['Iso'])
        new['species'] =  hitran.molparam['isotopologue'][i]
        ## remove natural abundance weighting
        new['S296K'] /=  hitran.molparam['natural_abundance'][i]
        self.extend(**new)
        return data             # return raw HITRAN data
        
    def generate_from_levels(self,level_upper,level_lower):
        """Combeiong all combination of upper and lower levels into a line list."""
        for key in level_upper:
            self[key+'_u'] = np.ravel(np.tile(level_upper[key],len(level_lower)))
        for key in level_lower:
            self[key+'_l'] = np.ravel(np.repeat(level_lower[key],len(level_upper)))




class LinearTriatomic(Generic):
    """E.g., CO2, CS2."""

    _levels_class = levels.LinearTriatomic
    _prototypes,defining_qn = _collect_prototypes(

        'species', 'point_group','mass',
        'ν','ν0',
        # 'λ',
        'ΔJ',
        'f','σ','S','S296K','τ','Ae',

        'pair','γ0air','nγ0air','δ0air','nδ0air','Γair','Δνair',
        'pself','γ0self','nγ0self','δ0self','nδ0self','Γself','Δνself',
        'pX','γ0X','nγ0X','δ0X','nδ0X','ΓX','ΔνX',
        # 'νvc',                  # test Rautian profile
        ## test HITRAN Hartmann-Tran
        'HT_HITRAN_X','HT_HITRAN_p', 'HT_HITRAN_Tref', 'HT_HITRAN_γ0', 'HT_HITRAN_n', 'HT_HITRAN_γ2', 'HT_HITRAN_δ0', 'HT_HITRAN_δp', 'HT_HITRAN_δ2', 'HT_HITRAN_νVC', 'HT_HITRAN_κ', 'HT_HITRAN_η', 'HT_HITRAN_Y',
        'HT_Γ0', 'HT_Γ2', 'HT_Δ0', 'HT_Δ2', 'HT_νVC', 'HT_η',

        'Nself',
        'Teq','Tex','Ttr','Z',
        'Γ','ΓD',
        levels_class = _levels_class
    )

    def load_from_hitran(self,filename):
        """Load HITRAN .data."""
        ## load generic things using the method in Generic
        data = Generic.load_from_hitran(self,filename)
        ## interpret specific quantum numbers
        quantum_numbers = dict(
            ΔJ=np.empty(len(self),dtype=int),
            J_l=np.empty(len(self),dtype=float),
            ν1_u=np.empty(len(self),dtype=int),
            ν2_u=np.empty(len(self),dtype=int),
            l2_u=np.empty(len(self),dtype=int),
            ν3_u=np.empty(len(self),dtype=int),
            ν1_l=np.empty(len(self),dtype=int),
            ν2_l=np.empty(len(self),dtype=int),
            l2_l=np.empty(len(self),dtype=int),
            ν3_l=np.empty(len(self),dtype=int),
        )
        ## loop over upper quantum setting in arrays
        for i,V in enumerate(data['V_u']):
            quantum_numbers['ν1_u'][i] = int(V[7:9])
            quantum_numbers['ν2_u'][i] = int(V[9:11])
            quantum_numbers['l2_u'][i] = int(V[11:13])
            quantum_numbers['ν3_u'][i] = int(V[13:15])
        ## loop over lower quantum setting in arrays
        for i,V in enumerate(data['V_l']):
            quantum_numbers['ν1_l'][i] = int(V[7:9])
            quantum_numbers['ν2_l'][i] = int(V[9:11])
            quantum_numbers['l2_l'][i] = int(V[11:13])
            quantum_numbers['ν3_l'][i] = int(V[13:15])
        ## Q_u is blank, Q_l is  [PQR][J_l]
        translatePQR = {'P':-1,'Q':0,'R':+1}
        for i,Q in enumerate(data['Q_l']):
            quantum_numbers['ΔJ'][i] = translatePQR[Q[5]]
            quantum_numbers['J_l'][i] = float(Q[6:])
        ## add all this data to self
        for key,val in quantum_numbers.items():
            self[key] = val

class Diatomic(Generic):

    _levels_class = levels.Diatomic
    _prototypes,defining_qn = _collect_prototypes(
        'species', 'point_group','mass',
        'ν','νv',
        'μv',
        # 'λ','λv',
        'ΔJ',
        'f','σ','S','S296K','τ','Ae',
        # 'γ0air','δ0air','nair','γself',
        # 'pself', 'pair', 'Nself',
        'Teq','Ttr','Z',
        'Γ','ΓD',
        'Teq', 'Tex', 'Ttr', 'L',
        # 'γ0air', 'δ0air', 'γself', 'nair', 'pself', 'pair', 'Nself',
        'branch', 'ΔJ', 'Γ', 'ΓD', 'f','σ','S','S296K', 'τ', 'Ae','τa', 'Sij','μ',
        levels_class = _levels_class
    )

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
            'γ0air':data['γair'], # HITRAN uses HWHM, I'm going to go with FWHM
            'nair':data['nair'],
            'δ0air':data['δair'],
            'γ0self':data['γself'], # HITRAN uses HWHM, I'm going to go with FWHM
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

    def load_from_duo(self,filename,intensity_type):
        """Load an output line list computed by DUO (yurchenko2016)."""
        data = file_to_dict(
            filename,
            labels=('index_u','index_l','intensity','ν'))
        if len(data)==0:
            print(f'warning: no data found in {repr(filename)}')
            return
        data.pop('index_u')
        data.pop('index_l')
        if intensity_type == 'absorption':
            data['f'] = data.pop('intensity')
        elif intensity_type == 'emission':
            data['Ae'] = data.pop('intensity')
        elif intensity_type == 'partition':
            data['α'] = data.pop('intensity')
        else:
            raise Exception(f'intensity_type must be "absorption" or "emission" or "partition"')
        self.extend(**data)

# class LinearTriatomic(Generic):
    # _levels_class = levels.LinearTriatomic
    # _prototypes = _collect_prototypes(
        # *_expand_level_keys_to_upper_lower(_levels_class),
        # *GenericLine._prototypes)
    # default_zkeys=(
        # 'species',
        # 'label_u','v1_u','v2_u','v3_u','l2_u',
        # 'label_l','v1_l','v2_l','v3_l','l2_l',
        # )

    # def load_from_hitran(self,filename):
        # """Load HITRAN .data."""
        # data = hitran.load(filename)
        # ## interpret into transition quantities common to all transitions
        # new = self.__class__(**{
            # 'ν':data['ν'],
            # ## 'Ae':data['A'],  # Ae data is incomplete but S296K will be complete
            # 'S296K':data['S'],
            # 'E_l':data['E_l'],
            # 'g_u':data['g_u'],
            # 'g_l':data['g_l'],
            # 'γair':data['γair']*2, # HITRAN uses HWHM, I'm going to go with FWHM
            # 'nair':data['nair'],
            # 'δair':data['δair'],
            # 'γself':data['γself']*2, # HITRAN uses HWHM, I'm going to go with FWHM
        # })
        # ## get species
        # i = hitran.molparam.find(species_ID=data['Mol'],local_isotopologue_ID=data['Iso'])
        # new['species'] =  hitran.molparam['isotopologue'][i]
        # ## remove natural abundance weighting
        # new['S296K'] /=  hitran.molparam['natural_abundance'][i]
        # ## interpret quantum numbers -- see rothman2005
        # new['v1_u'] = [t[7:9] for t in data['V_u']]
        # new['v2_u'] = [t[9:11] for t in data['V_u']]
        # new['l2_u'] = [t[11:13] for t in data['V_u']]
        # new['v3_u'] = [t[13:15] for t in data['V_u']]
        # new['v1_l'] = [t[7:9] for t in data['V_l']]
        # new['v2_l'] = [t[9:11] for t in data['V_l']]
        # new['l2_l'] = [t[11:13] for t in data['V_l']]
        # new['v3_l'] = [t[13:15] for t in data['V_l']]
        # branches = {'P':-1,'Q':0,'R':+1}
        # ΔJ,J_l = [],[]
        # for Q_l in data['Q_l']:
            # branchi,Jli = Q_l[5],Q_l[6:] 
            # ΔJ.append(branches[branchi])
            # J_l.append(Jli)
        # new['ΔJ'] = np.array(ΔJ,dtype=int)
        # new['J'+'_l'] = np.array(J_l,dtype=float)
        # # ## add data to self
        # self.extend(**new)

# # class TriatomicDinfh(Base):

    # # prototypes = {key:copy(prototypes[key]) for key in (
        # # list(Base.prototypes)
        # # + _expand_level_keys_to_upper_lower(levels.TriatomicDinfh))}

 
