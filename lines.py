import itertools
from copy import copy,deepcopy
from pprint import pprint
import re

import numpy as np
from numpy import nan,array,linspace
from scipy import constants
from immutabledict import immutabledict as idict

# from . import *
# from .dataset import Dataset
from . import tools
from .tools import cast_abs_float_array
from . import levels
from . import lineshapes
from . import tools
from .tools import file_to_dict,vectorise,cache
from . import hitran
from . import database
from . import plotting
from . import convert
from . import quantum_numbers
from .exceptions import InferException
# from .lines import prototypes
from . import levels
from . import dataset
from .dataset import Dataset
from .optimise import Parameter,P,optimise_method



prototypes = {}

## copy some direct from levels
for key in (
        'reference',
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
    prototypes[key+'_u'] = copy(tval)
    tval['infer'] = [(tuple(key+'_l' for key in tools.ensure_iterable(dependencies)),function)
                     for dependencies,function in val['infer']]
    prototypes[key+'_l'] = copy(tval)


## add lines things
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='U', cast=str, fmt='<10s')

def _f1(self,fv,SJ,J_l,Λ_u,Λ_l):
    """Get band fvalues from line strength"""
    f = fv*SJ/(2.*J_l+1.)       # correct? What about 2S+1?
    f[(Λ_l==0)&(Λ_u!=0)] /= 2
    return f
prototypes['f'] = dict(description="Line f-value",units="dimensionless",kind='f',fmt='<10.5e', cast=cast_abs_float_array,infer=[
    (('Ae','ν','g_u','g_l'),lambda self,Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2),
    (('Sij','ν','J_l'), lambda self,Sij,ν,J_l: 3.038e-6*ν*Sij/(2*J_l+1)), 
    (('σ','α_l'),lambda self,σ,α_l: σ*1.1296e12/α_l,),
    (('fv','SJ','J_l','Λ_u','Λ_l'),_f1),
])

prototypes['σ'] = dict(description="Spectrally-integrated photoabsorption cross section.",units="cm2.cm-1", kind='f', fmt='<10.5e',infer=[
    (('τa','Nself_l'),lambda self,τ,column_densitypp: τ/column_densitypp), 
    (('f','α_l'),lambda self,f,α_l: f/1.1296e12*α_l),
    (('S','ν','Tex'),lambda self,S,ν,Tex,: S/(1-np.exp(-convert.units(constants.Boltzmann,'J','cm-1')*ν/Tex))),])
## prototypes['σ'] =dict(description="Integrated cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('τ','column_densitypp'):lambda self,τ,column_densitypp: τ/column_densitypp, ('f','populationpp'):lambda self,f,populationpp: f/1.1296e12*populationpp,})
def _f0(self,S296K,species,Z,E_l,Tex,ν):
    """See Eq. 9 of simeckova2006"""
    Z296K = hitran.get_partition_function(species,296)
    c = convert.units(constants.Boltzmann,'J','cm-1') # hc/kB
    return (S296K
            *((np.exp(-E_l/(c*Tex))/Z)*(1-np.exp(-c*ν/Tex)))
            /((np.exp(-E_l/(c*296))/Z296K)*(1-np.exp(-c*ν/296))))
prototypes['S'] = dict(description="Spectral line intensity ",units="cm or cm-1/(molecular.cm-2", kind='f', fmt='<10.5e', infer=[(('S296K','species','Z','E_l','Tex_l','ν'),_f0,)])
prototypes['S296K'] = dict(description="Spectral line intensity at 296K reference temperature ). This is not quite the same as HITRAN which also weights line intensities by their natural isotopologue abundance.",units=" cm-1/(molecular.cm-2", kind='f', fmt='<10.5e', infer=[],cast=tools.cast_abs_float_array)
## Preferentially compute τ from the spectral line intensity, S,
## rather than than the photoabsorption cross section, σ, because the
## former considers the effect of stimulated emission.
prototypes['τ'] = dict(description="Integrated optical depth including stimulated emission",units="cm-1", kind='f', fmt='<10.5e',
                       infer=[
                           (('S','Nself_l'),lambda self,S,Nself_l: S*Nself_l,),
                           (('σ','Nself_l'),lambda self,σ,Nself_l: σ*Nself_l,),
                       ],)
prototypes['τa'] = dict(description="Integrated optical depth from absorption only",units="cm-1", kind='f', fmt='<10.5e', infer=[(('σ','Nself_l'),lambda self,σ,Nself_l: σ*Nself_l,)],)
prototypes['Ae'] = dict(description="Radiative decay rate",units="s-1", kind='f', fmt='<10.5g', infer=[(('f','ν','g_u','g_l'),lambda self,f,ν,g_u,g_l: f/(1.49951*g_u/g_l/ν**2)),(('At','Ad'), lambda self,At,Ad: At-Ad,)])
prototypes['Teq'] = dict(description="Equilibriated temperature",units="K", kind='f', fmt='0.2f', infer=[],default_step=0.1)
prototypes['Tex'] = dict(description="Excitation temperature",units="K", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq,)],default_step=0.1)
prototypes['Ttr'] = dict(description="Translational temperature",units="K", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq,),],default_step=0.1)
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind='f', fmt='>+4g', infer=[(('Jp','Jpp'),lambda self,Jp,Jpp: Jp-Jpp,)],)


## column 
prototypes['L'] = dict(description="Optical path length",units="m", kind='f', fmt='0.5f', infer=[])
prototypes['Nself'] = dict(description="Column density",units="cm-2",kind='f',fmt='<11.3e', cast=cast_abs_float_array,infer=[(('pself','L','Teq'), lambda self,pself,L,Teq: convert.units((pself*L)/(database.constants.Boltzmann*Teq),'m-2','cm-2'),)])

####################################
## pressure broadening and shifts ##
####################################
def _f0(self,J_l,ΔJ):
    mJ_l = np.full(J_l.shape,np.nan)
    i = ΔJ==-1
    mJ_l[i] = -J_l[i]
    i = ΔJ==+1
    mJ_l[i] = J_l[i]+1
    if np.any(np.isnan(mJ_l)):
        raise InferException('Could not compute mJ_l for all (J_l,ΔJ)')
    return mJ_l
prototypes['mJ_l'] = dict(description="Pressure broadening J-coordinate. m(P-branch) = -J_l, m(R-branch) = J_l+1", kind='f', fmt='>3g',infer=[(('J_l','ΔJ'),_f0)])
prototypes['pair'] = dict(description="Pressure of air",units="Pa", kind='f', fmt='0.5f',infer=[])
prototypes['γ0air'] = dict(description="Pressure broadening coefficient in air",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0air'] = dict(description="Pressure broadening temperature dependence in air",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0air'] = dict(description="Pressure shift coefficient in air",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0air'] = dict(description="Pressure shift temperature dependence in air",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['Γair'] = dict(description="Pressure broadening due to air",units="cm-1.FWHM", kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array, infer=[(('γ0air','nγ0air','pair','Ttr'),lambda self,γ,n,P,T: (296/T)**n*2*γ*convert.units(P,'Pa','atm')),])
prototypes['Δνair'] = dict(description="Pressure shift due to air",units="cm-1",kind='f', fmt='<10.5g',infer=[(('δ0air','nδ0air','pair','Ttr'),lambda self,δ,n,P,T: (296/T)**n*δ*convert.units(P,'Pa','atm')),])
prototypes['νvc'] = dict(description="Frequency of velocity changing collsions (which profile?)",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['pself'] = dict(description="Pressure of self",units="Pa", kind='f', fmt='0.5f',infer=[])
prototypes['γ0self'] = dict(description="Pressure broadening coefficient in self",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0self'] = dict(description="Pressure broadening temperature dependence in self",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0self'] = dict(description="Pressure shift coefficient in self",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0self'] = dict(description="Pressure shift temperature dependence in self",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['Γself'] = dict(description="Pressure broadening due to self",units="cm-1.FWHM",kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array,infer=[(('γ0self','nγ0self','pself','Ttr'),lambda self,γ0,n,P,T: (296/T)**n*2*γ0*convert.units(P,'Pa','atm')),])
prototypes['Δνself'] = dict(description="Pressure shift due to self",units="cm-1.HWHM",kind='f', fmt='<10.5g',infer=[(('δ0self','nδ0self','pself','Ttr'),lambda self,δ0,n,P,T: (296/T)**n*δ0*convert.units(P,'Pa','atm')),])
prototypes['pX'] = dict(description="Pressure of X",units="Pa", kind='f', fmt='0.5f',infer=[])
prototypes['γ0X'] = dict(description="Pressure broadening coefficient in X",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],cast=tools.cast_abs_float_array,default_step=1e-3)
prototypes['nγ0X'] = dict(description="Pressure broadening temperature dependence in X",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['δ0X'] = dict(description="Pressure shift coefficient in X",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[],default_step=1e-4)
prototypes['nδ0X'] = dict(description="Pressure shift temperature dependence in X",units="cm-1.atm-1.HWHM", kind='f',  fmt='<10.5g', infer=[((),lambda self: 0)],)
prototypes['ΓX'] = dict(description="Pressure broadening due to X",units="cm-1.FWHM",kind='f', fmt='<10.5g',cast=tools.cast_abs_float_array,infer=[(('γ0X','nγ0X','pX','Ttr'),lambda self,γ0,n,P,T: 2*(296/T)**n*2*γ0*convert.units(P,'Pa','atm')),])
prototypes['ΔνX'] = dict(description="Pressure shift due to species X",units="cm-1.HWHM",kind='f', fmt='<10.5g',infer=[(('δ0X','nδ0X','pX','Ttr'),lambda X,δ0,n,P,T: (296/T)**n*δ0*convert.units(P,'Pa','atm')),])
## HITRAN encoded pressure and temperature dependent Hartmann-Tran
## line broadening and shifting coefficients
prototypes['HITRAN_HT_X'] = dict(description='Broadening species for a HITRAN-encoded Hartmann-Tran profile',kind='U')
prototypes['HITRAN_HT_pX'] = dict(description='Pressure HITRAN-encoded Hartmann-Tran profile',kind='f',units='atm',cast=tools.cast_abs_float_array,infer=[('pX',lambda self,pX:convert.units(pX,'Pa','atm'))])
prototypes['HITRAN_HT_Tref'] = dict(description='Reference temperature for a HITRAN-encoded Hartmann-Tran profile ',units='K',kind='f')
prototypes['HITRAN_HT_γ0'] = dict(description='Speed-averaged halfwidth in temperature range around Tref due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=tools.cast_abs_float_array)
prototypes['HITRAN_HT_n'] = dict(description='Temperature dependence exponent of γ0 in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f')
prototypes['HITRAN_HT_γ2'] = dict(description='Speed-dependence of the halfwidth in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=tools.cast_abs_float_array)
prototypes['HITRAN_HT_δ0'] = dict(description='Speed-averaged line shift in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f')
prototypes['HITRAN_HT_δp'] = dict(description='Linear temperature dependence coefficient for δ0 in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm.K-1',kind='f')
prototypes['HITRAN_HT_δ2'] = dict(description='Speed-dependence of the line shift in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f')
prototypes['HITRAN_HT_νVC'] = dict(description='Frequency of velocity changing collisions in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='cm-1.atm-1',kind='f',cast=tools.cast_abs_float_array)
prototypes['HITRAN_HT_κ'] = dict(description='Temperature dependence of νVC in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f')
### def _f0(x):
##    # """Limiting values!!! Otherwise lineshape is bad -- should investigate this."""
##    # x = np.abs(np.asarray(x),dtype=float)
##    # x[x>1] = 0.99999
##    # return x
prototypes['HITRAN_HT_η'] = dict(description='Correlation parameter in HT in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile',units='dimensionless',kind='f',cast=tools.cast_abs_float_array,default_step=1e-5)
prototypes['HITRAN_HT_Y'] = dict(description='First-order (Rosenkranz) line coupling coefficient in the Tref temperature range due to perturber X for a HITRAN-encoded Hartmann-Tran profile; air-(self-) broadened case',units='cm-1.atm-1',kind='f')
## coefficients of the Hartmann-Tran lineshape
prototypes['HT_Γ0'] = dict(description='Speed-averaged halfwidth for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HITRAN_HT_pX','HITRAN_HT_γ0','HITRAN_HT_Tref','Ttr','HITRAN_HT_n'),lambda self,p,γ0,Tref,T,n: γ0*p*(Tref/T)**n)])
prototypes['HT_Γ2'] = dict(description='Speed-dependence for the halfwidth for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HITRAN_HT_pX','HITRAN_HT_γ2',),lambda self,p,γ2: p*γ2),])
prototypes['HT_Δ0'] = dict(description='Speed-averaged line shift for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HITRAN_HT_pX','HITRAN_HT_δ0','HITRAN_HT_δp','HITRAN_HT_Tref','Ttr'),lambda self,p,δ0,δp,Tref,T: p*(δ0+δp*(T-Tref))),])
prototypes['HT_Δ2'] = dict(description='Speed-dependence for the line shift for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HITRAN_HT_pX','HITRAN_HT_δ2'),lambda self,p,δ2: p*δ2),])
prototypes['HT_νVC'] = dict(description='Frequency of velocity changing collisions for the Hartmann-Tran profile',units='cm-1.atm-1',kind='f',infer=[(('HITRAN_HT_pX','HITRAN_HT_νVC','HITRAN_HT_Tref','Ttr','HITRAN_HT_κ'),lambda self,p,νVC,Tref,T,κ: p*νVC*(Tref/T)**κ),])
prototypes['HT_η'] = dict(description='Correlation parameter for the Hartmann-Tran profile',units='dimensionless',kind='f',infer=[(('HITRAN_HT_η',),lambda self,η:η),])



## linewidths
prototypes['Γ'] = dict(description="Total Lorentzian linewidth of level or transition" ,units="cm-1.FWHM", kind='f', fmt='<10.5g',infer=[
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
prototypes['ΓD'] = dict(description="Gaussian Doppler width",units="cm-1.FWHM",kind='f',fmt='<10.5g', infer=[(('mass','Ttr','ν'), lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,)])

## line frequencies
prototypes['ν0'] = dict(description="Transition wavenumber in a vacuum",units="cm-1", kind='f', fmt='>0.6f', default_step=1e-3, infer=[])
prototypes['ν'] = dict(description="Transition wavenumber",units="cm-1", kind='f', fmt='>0.6f', infer=[
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
def _f3(self,species,Tex,E_u,E_l,g_u,g_l,Σ_l,Σ_u,ef_l,ef_u):
    """Compute partition function from data in self."""
    if self['Zsource'] != 'self':
        raise InferException(f'Zsource not "self".')
    Z = np.full(len(species),0.)
    ## calculate separately for all (species,Tex) combinations
    for (speciesi,Texi),i in tools.unique_combinations_mask(species,Tex):
        i = tools.find(i)
        kT = convert.units(constants.Boltzmann,'J','cm-1')*Texi
        ## sum for all unique upper levels
        k = []
        for qn,j in tools.unique_combinations_mask(
                *[self[key+'_u'][i] for key in self._level_class.defining_qn]
        ):
            k.append((i[j])[0])
        Z[i] += np.sum(g_u[k]*np.exp(-E_u[k]/kT))
        ## sum for all unique lower levels
        k = []
        for qn,j in tools.unique_combinations_mask(
                *[self[key+'_l'][i] for key in self._level_class.defining_qn]):
            k.append((i[j])[0])
        Z[i] += np.sum(g_l[k]*np.exp(-E_l[k]/kT))
    return Z
@vectorise(cache=True,vargs=(1,2))
def _f5(self,species,Tex):
    if self.attributes['Zsource'] != 'HITRAN':
        raise InferException(f'Zsource not "HITRAN".')
    from . import hitran
    return hitran.get_partition_function(species,Tex)
prototypes['Z'] = dict(description="Partition function including both upper and lower levels.", kind='f', fmt='<11.3e', infer=[
    (('species','Tex'),_f5),
    (('species','Tex','E_u','E_l','g_u','g_l','Σ_l','Σ_u','ef_l','ef_u'),_f3,),
])

## vibrational transition frequencies
prototypes['νv'] = dict(description="Electronic-vibrational transition wavenumber",units="cm-1",kind='f', fmt='>11.4f', infer=[(('Tvp','Tvpp'), lambda self,Tvp,Tvpp: Tvp-Tvpp),( ('λv',), lambda self,λv: convert_units(λv,'nm','cm-1'),)])
prototypes['λv'] = dict(description="Electronic-vibrational transition wavelength",units="nm",kind='f', fmt='>11.4f', infer=[(('νv',), lambda self,νv: convert_units(νv,'cm-1','nm'),)],)

## transition strengths
prototypes['M']   = dict(description="Pointer to electronic transition moment",units="au", kind='O', infer=[])
prototypes['Mv']   = dict(description="Electronic transition moment for this vibronic level",units="au", kind='f', fmt='<10.5e', infer=[(('μ','FCfactor'), lambda self,μ,FCfactor: μ/np.sqrt(FCfactor),)])
prototypes['μv']  = dict(description="Electronic-vibrational transition moment",units="au", kind='f',  fmt='<10.5e', infer=[(('M','χp','χpp','R'), lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),)],) # could infer from S but then sign would be unknown
prototypes['μ']   = dict(description="Electronic-vibrational-rotational transition moment",units="au", kind='f',  fmt='<10.5e', infer=[(('M','χp','χpp','R'), lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),)],) # could infer from S but then sign would be unknown
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
prototypes['Sv'] =dict(description="Band strength, ⟨vp|Re|vpp⟩**2",units="au", kind='f',  fmt='<10.5e',cast=cast_abs_float_array, infer=[
    (('Sij','SJ'), lambda self,Sij,SJ: Sij/SJ),
    ( ('μ',),lambda self,μ:μ**2),
    (('fv','ν','Λp','Λpp'),lambda self,fv,ν,Λp,Λpp: band_fvalue_to_band_strength(fv,ν,Λp,Λpp)),
    (('fv','νv','Λp','Λpp'),lambda self,fv,νv,Λp,Λpp : band_fvalue_to_band_strength(fv,νv,Λp,Λpp)),
    (('Aev','ν','Λp','Λpp'),lambda self,Aev,ν,Λp,Λpp : band_emission_rate_to_band_strength(Aev,ν,Λp,Λpp )),
    ( ('Aev','νv','Λp','Λpp'), lambda self,Aev,νv,Λp,Λpp: band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),)],)
def _f1(self,f,SJ,J_l,Λ_u,Λ_l):
    """Get band fvalues from line strength"""
    fv = f/SJ*(2.*J_l+1.)       # correct? What about 2S+1?
    fv[(Λ_l==0)&(Λ_u!=0)] *= 2
    return fv
prototypes['fv'] = dict(description="Band f-value",units="dimensionless", kind='f',  fmt='<10.5e',cast=cast_abs_float_array, infer=[
    (('Sv','ν','Λ_u','Λ_l'),  lambda self,Sv,ν,Λ_u,Λ_l :  band_strength_to_band_fvalue(Sv,ν, Λ_u,Λ_l)),
    ( ('Sv','νv','Λ_u','Λ_l'), lambda self,Sv,νv,Λ_u,Λ_l:  band_strength_to_band_fvalue(Sv,νv,Λ_u,Λ_l)),
    ( ('f','SJ','J_l','Λ_u','Λ_l'), _f1,)])
prototypes['Aev'] =dict(description="Einstein A coefficient / emission rate averaged over a band.",units="s-1", kind='f',  fmt='<10.5e', infer=[(('Sv','ν' ,'Λp','Λpp'), lambda self,Sv,ν ,Λp,Λpp: band_strength_to_band_emission_rate(Sv,ν ,Λp,Λpp)),( ('Sv','νv','Λp','Λpp'), lambda self,Sv,νv,Λp,Λpp: band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp),)],) 
prototypes['σv'] =dict(description="Integrated cross section of an entire band.",units="cm2.cm-1", kind='f',  fmt='<10.5e', infer=[(('fv',),lambda self,fv: band_fvalue_to_band_cross_section(fv),)],)
prototypes['Sij'] =dict(description=" strength",units="au", kind='f',  fmt='<10.5e', infer=[
    (('μ',), lambda self,μ: μ**2),
    (('Sv','SJ'), lambda self,Sv,SJ:  Sv*SJ),
    ( ('f','ν','Jpp'), lambda self,f,ν,Jpp: f/3.038e-6/ν*(2*Jpp+1)),
    ( ('Ae','ν','Jp'), lambda self,Ae,ν,Jp: Ae/(2.026e-6*ν**3/(2*Jp+1)),)])
prototypes['Ae'] =dict(description="Einstein A coefficient / emission rate.",units="s-1", kind='f',  fmt='<10.5e', infer=[(('f','ν','Jp','Jpp'), lambda self,f,ν,Jp,Jpp: f*0.666886/(2*Jp+1)*(2*Jpp+1)*ν**2),( ('Sij','ν','Jp'), lambda self,Sij,ν,Jp: Sij*2.026e-6*ν**3/(2*Jp+1))],)
prototypes['FCfactor'] =dict(description="Franck-Condon factor",units="dimensionless", kind='f',  fmt='<10.5e', infer=[(('χp','χpp','R'), lambda self,χp,χpp,R: np.array([integrate.trapz(χpi*χppi,R)**2 for (χpi,χppi) in zip(χp,χpp)])),],)
prototypes['Rcentroid'] =dict(description="R-centroid",units="Å", kind='f',  fmt='<10.5e', infer=[(('χp','χpp','R','FCfactor'), lambda self,χp,χpp,R,FCfactor: np.array([integrate.trapz(χpi*R*χppi,R)/integrate.trapz(χpi*χppi,R) for (χpi,χppi) in zip(χp,χpp)])),])

def _f0(self,S_u,S_l,Ω_u,Ω_l,J_u,J_l):
    if not (np.all(S_u==0) and np.all(S_l==0)):
        raise InferException('Honl-London factors only defined here for singlet states.')
    SJ = quantum_numbers.honl_london_factor(Ω_u,Ω_l,J_u,J_l)
    return SJ
prototypes['SJ'] = dict(description="Rotational line strength",units="dimensionless", kind='f',  fmt='<10.5e', infer=[(('S_u','S_l','Ω_u','Ω_l','J_u','J_l'),_f0),])

# prototypes['τ'] = dict(description="Integrated optical depth",units="cm-1", kind='f',  fmt='<10.5e', infer={('σ','column_densitypp'):lambda self,σ,column_densitypp: σ*column_densitypp,},)
# prototypes['I'] = dict(description="Integrated emission energy intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp','ν'):lambda self,Ae,populationp,column_densityp,ν: Ae*populationp*column_densityp*ν,},)
# prototypes['Ip'] = dict(description="Integrated emission photon intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp'):lambda self,Ae,populationp,column_densityp: Ae*populationp*column_densityp,},)
# prototypes['σd'] = dict(description="Integrated photodissociation cross section.",units="cm2.cm-1", kind='f',  fmt='<10.5e', infer={('σ','ηdp'):lambda self,σ,ηdp: σ*ηdp,})
# prototypes['Sabs'] = dict(description="Absorption intensity).",units="cm-1/(molecule.cm-1", kind='f',  fmt='<10.5e', infer=[])

## vibrational interaction energies
prototypes['ηv'] = dict(description="Reduced spin-orbit interaction energy mixing two vibronic levels.",units="cm-1", kind='f',  fmt='<10.5e', infer=[])
prototypes['ξv'] = dict(description="Reduced rotational interaction energy mixing two vibronic levels.",units="cm-1", kind='f',  fmt='<10.5e', infer=[])
prototypes['ηDv'] = dict(description="Higher-order reduced spin-orbit interaction energy mixing two vibronic levels.",units="cm-1", kind='f',  fmt='<10.5e', infer=[])
prototypes['ξDv'] = dict(description="Higher-roder reduced rotational interaction energy mixing two vibronic levels.",units="cm-1", kind='f',  fmt='<10.5e', infer=[])




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
prototypes['J_u']['infer'].append((('J_l','ΔJ'),lambda self,J_l,ΔJ: J_l+ΔJ))
prototypes['J_l']['infer'].append((('J_u','ΔJ'),lambda self,J_u,ΔJ: J_u-ΔJ))
prototypes['Z_l']['infer'].append((('Z'),lambda self,Z:Z))
prototypes['Z_u']['infer'].append((('Z'),lambda self,Z:Z))

def _collect_level(level_class):
    """Get prototype keys and defining qn from a level class."""
    keys = []
    for key in level_class.default_prototypes:
        keys.append(key+'_l')
        keys.append(key+'_u')
    defining_qn = tuple([key+'_u' for key in level_class.defining_qn]
                        +[key+'_l' for key in level_class.defining_qn])
    return level_class,keys,defining_qn

class Generic(levels.Base):

    _level_class,_level_keys,defining_qn = _collect_level(levels.Generic)
    _line_keys = (
        'reference',
        'species', 'point_group','mass',
        'ν','ν0', # 'λ',
        'ΔJ', 'branch',
        'ΔJ',
        'f','σ','S','S296K', 'τ', 'Ae','τa', 'Sij','μ',
        'Nself',
        'Teq','Tex','Ttr','Z',
        'Γ','ΓD',
        ## pressure broadening stuff
        'mJ_l',
        'pair','γ0air','nγ0air','δ0air','nδ0air','Γair','Δνair',
        'pself','γ0self','nγ0self','δ0self','nδ0self','Γself','Δνself',
        'pX','γ0X','nγ0X','δ0X','nδ0X','ΓX','ΔνX',
        ## test HITRAN Hartmann-Tran
        'HITRAN_HT_X','HITRAN_HT_pX', 'HITRAN_HT_Tref', 'HITRAN_HT_γ0', 'HITRAN_HT_n', 'HITRAN_HT_γ2', 'HITRAN_HT_δ0', 'HITRAN_HT_δp', 'HITRAN_HT_δ2', 'HITRAN_HT_νVC', 'HITRAN_HT_κ', 'HITRAN_HT_η', 'HITRAN_HT_Y',
        'HT_Γ0', 'HT_Γ2', 'HT_Δ0', 'HT_Δ2', 'HT_νVC', 'HT_η',
    )
    default_prototypes = {key:prototypes[key] for key in {*_level_keys,*_line_keys}}
    default_xkey = 'J_l'
    default_zkeys = ['species_u','label_u','species_l','label_l','ΔJ']
    
    def plot_spectrum(
            self,
            *calculate_spectrum_args,
            ax=None,
            plot_kwargs=None,
            zkeys=None,
            **calculate_spectrum_kwargs
    ):
        """Plot a nice cross section. If zkeys given then divide into multiple
        lines accordings."""
        ## deal with default args
        if ax is None:
            ax = plotting.qax()
        if plot_kwargs is None:
            plot_kwargs = {}
        ## calculate spectrum
        spectrum = self.calculate_spectrum(*calculate_spectrum_args,zkeys=zkeys,**calculate_spectrum_kwargs)
        if zkeys is not None:
            ## multiple zkeys -- plot and legend
            for qn,x,y in spectrum:
                ax.plot(x,y,label=tools.dict_to_kwargs(qn),**plot_kwargs)
            plotting.legend(loc='upper left')
        else:
            ## single spectrum only
            x,y = spectrum
            ax.plot(x,y,**plot_kwargs)
        return ax

    def plot_transmission_spectrum(
            self,
            *args,
            ax=None,
            plot_kwargs=None,
            zkeys=None,
            **kwargs
    ):
        """Plot a nice cross section. If zkeys given then divide into multiple
        lines accordings."""
        ## deal with default args
        if ax is None:
            ax = plotting.qax()
        if plot_kwargs is None:
            plot_kwargs = {}
        ## calculate spectrum
        spectrum = self.calculate_transmission_spectrum(*args,zkeys=zkeys,**kwargs)
        if zkeys is not None:
            ## multiple zkeys -- plot and legend
            for qn,x,y in spectrum:
                ax.plot(x,y,label=tools.dict_to_kwargs(qn),**plot_kwargs)
            plotting.legend(loc='upper left')
        else:
            ## single spectrum only
            x,y = spectrum
            ax.plot(x,y,**plot_kwargs)
        return ax

    def plot_stick_spectrum(
            self,
            xkey='ν',
            ykey='σ',
            zkeys=None,         # None or list of keys to plot as separate lines
            ax=None,
            **plot_kwargs # can be calculate_spectrum or plot kwargs
    ):
        """No  lineshapes, just plot as sticks."""
        if ax is None:
            ax = plotting.plt.gca()
        if zkeys is None:
            zkeys = self.default_zkeys
        for iz,(qn,t) in enumerate(self.unique_dicts_matches(*tools.ensure_iterable(zkeys))):
            t_plot_kwargs = plot_kwargs | {
                'color':plotting.newcolor(iz),
                'label':repr(qn),
            }
            plotting.plot_sticks(t[xkey],t[ykey],**t_plot_kwargs)
        plotting.legend(ax=ax,fontsize='x-small')
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        return ax

    def calculate_spectrum(
            self,
            x=None,        # frequency grid (must be regular, I think), if None then construct a reasonable grid
            xkey='ν',      # strength to use, i.e., "ν", or "λ"
            ykey=None,      # strength to use, i.e., "σ", "τ", or "I"
            zkeys=None,    # if not None then calculate separate spectra for unique combinations of these keys
            lineshape=None, # None for auto selection, or else one of ['voigt','gaussian','lorentzian','hartmann-tran']
            nfwhmG=20, # how many Gaussian FWHMs to include in convolution
            nfwhmL=100,         # how many Lorentzian FWHMs to compute
            dx=None, # grid step to use if x grid computed automatically
            nx=10000, # number of grid points to use if x grid computed automatically
            ymin=None, # minimum value of ykey before a line is ignored, None for use all lines
            ncpus=1, # 1 for single process, more to use up to this amount when computing spectrum
            index=None,         # only calculate for these indices
            # **set_keys_vals, # set some data first, e..g, the tempertaure
    ):
        """Calculate a spectrum from the data in self. Returns (x,y)."""
        ## guess a default ykey
        if ykey is None:
            for ykey in ('τ','σ','f','Ae','Sij','μ'):
                if self.is_known(ykey):
                    break
            else:
                raise Exception("Could not find a default ykey")
        ## guess a default lineshape
        if lineshape is None:
            if self.is_known('Γ','ΓD'):
                lineshape = 'voigt'
            elif self.is_known('Γ'):
                lineshape = 'lorentzian'
            elif self.is_known('ΓD'):
                lineshape = 'gaussian'
            else:
                raise Exception(f"Cannot determine lineshape because both Γ and ΓD are unknown")
        ## set some data
        # for key,val in set_keys_vals.items():
            # self[key] = val
        ## no lines to add to cross section -- return quickly
        if len(self)==0:
            if x is None:
                return np.array([]),np.array([])
            else:
                return x,np.zeros(x.shape)
        ## get x and ykeys
        self.assert_known(xkey,ykey)
        ## get a default frequency scale if none provided
        if x is None:
            if dx is not None:
                x = np.arange(max(0,self[xkey].min()-10.),self[xkey].max()+10.,dx)
            else:
                x = np.linspace(max(0,self[xkey].min()-10.),self[xkey].max()+10.,nx)
        else:
            x = np.asarray(x)
        ## branch to calc separate spectrum for unique combinations of
        ## zkeys if these are given -- common x-grid
        if zkeys is not None:
            retval = []
            for qn,match in self.unique_dicts_matches(*zkeys):
                x,y = match.calculate_spectrum(
                    x=x,xkey=xkey,ykey=ykey,zkeys=None,
                    lineshape=lineshape,nfwhmG=nfwhmG,nfwhmL=nfwhmL,
                    dx=dx,nx=nx,ymin=ymin,ncpus=ncpus)
                retval.append((qn,x,y))
            return retval
        ## check frequencies, strengths, widths are as expected
        # self.assert_known(xkey,ykey)
        # assert np.all(~np.isnan(self[xkey])),f'NaN values in xkey: {repr(xkey)}'
        # assert np.all(~np.isnan(self[ykey])),f'NaN values in ykey: {repr(ykey)}'
        ## indices given
        if index is not None:
            i = np.full(len(self),False)
            i[index] = True

        else:
            i = np.full(len(self),True)
        ## neglect lines out of x-range -- NO ACCOUNTING FOR EDGES!!!!
        i &= (self[xkey]>x[0]) & (self[xkey]<x[-1])
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
                raise Exception('Only valid xkey is "ν"')
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

    def calculate_transmission_spectrum(
            self,
            *args_calculate_spectrum,
            zkeys=None,
            **kwargs_calculate_spectrum,
    ):
        retval = self.calculate_spectrum(
            *args_calculate_spectrum,
            ykey='τ',
            zkeys=zkeys,
            **kwargs_calculate_spectrum,)
        if zkeys is None:
            x,y = retval
            retval = x,np.exp(-y)
        else:
            retval = [(d,x,np.exp(-y)) for d,x,y in retval]
        return retval 

    def _get_level(self,u_or_l,reduce_to=None,):
        """Get all data corresponding to 'upper' or 'lower' level in
        self."""
        ## try get all deining qn
        for key in self.defining_qn:
            try: 
                self[key]
            except InferException:
                pass
        ## 
        if u_or_l not in ('u','l'):
            raise Exception("u_or_l must be 'u' or 'l'")
        levels = self._level_class()
        for key in self.keys():
            if len(key)>2 and key[-2:]==('_'+u_or_l):
                levels[key[:-2]] = self[key]
        if reduce_to == None:
            pass
        else:
            keys = [key for key in levels.defining_qn if levels.is_known(key)]
            if reduce_to == 'first':
                ## find first indices of unique key combinations and reduce
                ## to those
                t,i = tools.unique_combinations_first_index(*[levels[key] for key in keys])
                levels.index(i)
            else:
                raise ImplementationError()
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
        
    def load_from_nist(self,filename):
        """Load  NIST csv atomic transition data file."""
        ## load into dict
        # data = tools.file_to_dict(filename)
        data = dataset.load(filename)
        ## manipulate some data
        for key in ('J_i','J_k'):
            if data.get_kind(key) == 'U':
                for i,J in enumerate(data[key]):
                    if '/' in J:
                        num,den = J.split('/')
                        data[key][i] = int(num)/int(den)
                    else:
                        data[key][i] = float(J)
        re_compiled_1 = re.compile(r' *([0-9]+)([A-Z])(\*?) *')
        re_compiled_2 = re.compile(r' *([0-9]+)(\[[0-9/]+\])(\*?) *')
        L_dict = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5}
        for key,ul in (('term_i','_l'),('term_k','_u')):
            L,S,gu = [],[],[]
            for i,term in enumerate(data[key]):
                if term == 'nan':
                    L.append(-1)
                    S.append(nan)
                    gu.append(0)
                else:
                    if r:=re.match(re_compiled_1,term):
                        S.append((int(r.group(1))-1)/2)
                        L.append(L_dict[r.group(2)])
                        if r.group(3) == '*':
                            gu.append(-1)
                        else:
                            gu.append(+1)
                    elif r:=re.match(re_compiled_2,term):
                        S.append((int(r.group(1))-1)/2)
                        L.append(-1)
                        if r.group(3) == '*':
                            gu.append(-1)
                        else:
                            gu.append(+1)
                    else:
                        raise Exception(f"Decode term failed: {repr(term)}")
            data['L'+ul] = L
            data['S'+ul] = S
            data['gu'+ul] = gu
        ## add to self
        for key0,key1 in (
                ('wn(cm-1)','ν'),
                ('Aki(s^-1)','Ae',),
                ('Ei(cm-1)','E_l'),
                ('Ek(cm-1)','E_u'),
                ('conf_i','conf_l'),
                ('J_i','J_l'),
                ('conf_k','conf_u'),
                ('J_k','J_u'),
                ('L_u','L_u'),
                ('S_u','S_u'),
                ('gu_u','gu_u'),
                ('L_l','L_l'),
                ('S_l','S_l'),
                ('gu_l','gu_l'),
        ):
            self[key1] = data[key0]
        self['reference'] = 'NIST'
        
    def load_from_pgopher(
            self,
            filename, # a csv file from the pgopher linelist export menu
            search_re='.*', # Limit the reading of lines to those lines matching this regexp against the Label field. 
            **append_to_self_kwargs, # given to self.append
    ):
        """Load data from an export pgopher model linelist. The
        constants are also in this file. To get such a file out of
        pgopher: File -> Export -> Line list -> some csv file. The
        output file might need to be renamed to have a csv suffix."""
        ## load pgopher output file
        with open(tools.expand_path(filename),mode='r',errors='ignore') as fid:
            ## find beginning of linelist
            for line in fid:
                if line=='''Molecule,Upper Manifold,J',Sym',#',Lower Manifold,"J""","Sym""","#""",Position,Intensity,Eupper,Elower,Spol,A,Width,Branch,Label\n''':
                    break
            else:
                raise Exception("Could not find beginning of linelist in file: "+repr(filename))
            ## loop through each line in linelist -- saving to dict of lists
            data = {key:[] for key in (
                'ν','E_l','E_u','v_u', 'label_u', 'J_u', 'N_u', 'F_u', 'ef_u', 
                'v_l', 'label_l', 'J_l', 'N_l', 'F_l', 'ef_l', 
                'Sij',
                # 'Ae',
            )}
            for line in fid:
                ## split on tabs
                line = line.split(',')
                # if len(line)!=30: continue # a quick and cheap test for a valid linelist line
                if len(line)!=18: break # a quick and cheap test for a valid linelist line
                if not re.match(search_re,line[17]): continue # skip this line as requested
                upper_manifold = line[1]
                lower_manifold = line[5]
                data['J_u'].append(float(line[2]))
                data['ef_u'].append((1 if line[3].strip()=='e' else -1))
                data['J_l'].append(float(line[6]))
                data['ef_l'].append((1 if line[7].strip()=='e' else -1))
                data['ν'].append(float(line[9]))
                data['E_u'].append(float(line[11]))
                data['E_l'].append(float(line[12]))
                data['Sij'].append(float(line[13])*3) # 3x PGopher Spol (single polarisation only)
                ## data['σ'].append(float(line[10])) # actually a cross section?
                # data['Ae'].append(float(line[14]))
                ## Decode transition name. Split into upper and lower
                ## levels. Get J, F, ef from pgopher encoded parts of
                ## the string. Use decode_lower_name_re and
                ## decode_upper_name_re to get the label and v which
                ## are influenced by the user.
                def decode_pgo_level_name(name,manifold_name):
                    """Try various ways to decode a pgopher level name."""
                    r = re.match(r"(Excited|Ground) ([a-zA-Z0-9']+)v=([0-9]+) +([0-9.]+) +([0-9.]+) +F([0-9]+)([ef]) *$",name) # e.g., "Excited Av=1 0.5  1 F2f"
                    if r:
                        return(dict(
                            label = r.group(2),
                            v = int(r.group(3)),
                            N = float(r.group(5)),
                            F = float(r.group(6)),
                        ))
                    r = re.match(r'(.*) +([0-9.]+) +([0-9.]+) +F([0-9]+)([ef]) *$',name) # e.g., "Excited A(v=1) 0.5  1 F2f"
                    if r:
                        retval = {}
                        label_v = r.group(1)
                        retval['N'] = float(r.group(3))
                        retval['F'] = float(r.group(4))
                        decoded_level_name = quantum_numbers.decode_level(label_v.strip()[len(manifold_name):])
                        if 'label' in decoded_level_name: 
                            retval['label'] = decoded_level_name['label']
                        else:
                            retval['label'] = '?'
                        if 'v' in decoded_level_name:
                            retval['v'] = decoded_level_name['v']
                        else:
                            retval['v'] = -1
                        return retval
                    r = re.match(r'(.*) +([a-zA-Z]+)\(([0-9.]+)\) +([0-9.]+) +([ef]) *$',name) # e.g., "Excited C(0)  3 e"
                    if r:
                        return(dict(label = r.group(2), v = r.group(3),))
                    raise Exception('Could not decode pgo level name: '+repr(name))
                transition_name = line[17]
                upper_level_name,lower_level_name = transition_name.strip().split(' - ')
                tdata = decode_pgo_level_name(lower_level_name.strip(),lower_manifold)
                for key,val in tdata.items():
                    data[key+'_l'].append(val)
                tdata = decode_pgo_level_name(upper_level_name.strip(),upper_manifold)
                for key,val in tdata.items():
                    data[key+'_u'].append(val)
        ## submit
        for key in [t for t in data.keys()]:
            if len(data[key])==0: data.pop(key) # remove empty keys
            
        self.extend(**data)

    # @optimise_method()
    # def generate_from_levels(
    #         self,
    #         levelu,levell,    # upper and lower level objects
    #         matchu=idict(),    # only use matching upper levels
    #         matchl=idict(),    # only use matching lower levels
    #         match=idict(),      # only use matching lines
    #         _cache=()
    # ):
    #     """Combine all combination of upper and lower levels into a line list."""
    #     self.add_suboptimiser(levelu)
    #     self.add_suboptimiser(levell)
    #     if len(_cache) == 0:
    #         ## first run
    #         ibeg = len(self)    # initial length
    #         ## match upper and lowe levels
    #         iu,il = levelu.match(matchu),levell.match(matchl)
    #         nu,nl = np.sum(iu),np.sum(il)
    #         ## first run add all data to self
    #         for key in levelu.root_keys():
    #             self[key+'_u'] = np.ravel(np.tile(levelu[key][iu],nl))
    #         for key in levell.root_keys():
    #             self[key+'_l'] = np.ravel(np.repeat(levell[key][il],nu))
    #         ## limit to allowed transition -- record indices for later
    #         i = self.match(match)[ibeg:]
    #         self.index(i)
    #         _cache['i'],_cache['iu'],_cache['il'],_cache['nu'],_cache['nl'] = i,iu,il,nu,nl
    #     else:
    #         ## later runs - only update changed data
    #         ##
    #         i,iu,il,nu,nl = _cache['i'],_cache['iu'],_cache['il'],_cache['nu'],_cache['nl']
    #         ## collect data
    #         data = {}
    #         for key in levelu.root_keys():
    #             data[key+'_u'] = np.ravel(np.tile(levelu[key][iu],nl))
    #         for key in levell.root_keys():
    #             data[key+'_l'] = np.ravel(np.repeat(levell[key][il],nu))
    #         ## add to self if data has changed, assumes
    #         ## match_keys_vals has not changed
    #         for key,val in data.items():
    #             if np.any(val[i] != self[key]):
    #                 self[key] = val[i]

    @optimise_method()
    def generate_from_levels(
            self,
            levelu,levell,      # upper and lower level objects
            matchu=idict(),     # only use matching upper levels
            matchl=idict(),     # only use matching lower levels
            match=idict(),      # only keep matching lines
            _cache=()
    ):
        """Combine upper and lower levels into a line list. Extend these after any existing lines."""
        ## get indices and lengths of levels to combine
        if 'iu' not in _cache:
            _cache['iu'] = levelu.match(matchu)
            _cache['il'] = levell.match(matchl)
            _cache['nu'] = np.sum(_cache['iu'])
            _cache['nl'] = np.sum(_cache['il'])
        iu,il,nu,nl = _cache['iu'],_cache['il'],_cache['nu'],_cache['nl']
        ## collect level data
        data = {}
        for key in levelu.root_keys():
            data[key+'_u'] = np.ravel(np.tile(levelu[key][iu],nl))
        for key in levell.root_keys():
            data[key+'_l'] = np.ravel(np.repeat(levell[key][il],nu))
        ## get where to add data in self
        if 'ibeg' not in _cache:
            _cache['ibeg'] = len(self)    # initial length
        ibeg = _cache['ibeg']             # beginning index of new data
        ## get combined levels to add
        if 'imatch' not in _cache:
            ## first run -- add and then reduced to matches
            self.extend(data)
            ## limit to previous data and new data matching match
            imatch = self.match(match)
            imatch[:ibeg] = True     # keep all existing data
            self.index(imatch)
            ##
            imatch = imatch[ibeg:]              # index of new data only
            iend = len(self)    # end index of new data
            ilines = np.arange(ibeg,iend) # index array fo new data
            _cache['imatch'] = imatch
            _cache['ilines'] = ilines
        else:
            ## later run, add to self if data has changed
            ilines = _cache['ilines']
            imatch = _cache['imatch']
            for key,val in data.items():
                val = val[imatch]
                ichanged = np.any(val != self[key][ilines])
                self.set(key,val[ichanged],index=ilines[ichanged])

    def set_levels(self,match=None,**keys_vals):
        for key,val in keys_vals.items():
            suffix = key[-2:]
            assert suffix in ('_u','_l')
            qn_keys = [t+suffix for t in self._level_class.defining_qn]
            ## find match if requested
            if match is not None:
                imatch = self.match(**match)
            ## loop through all sets of common levels setting key=val
            for d,i in self.unique_dicts_match(*qn_keys):
                ## limit to match is requested
                if match is not None:
                    i &= imatch
                    if not np.any(i):
                        continue
                ## make a copy of value -- and a Parameter if
                ## necessary. Substitute current value into if is NaN
                ## is given
                if np.isscalar(val):
                    vali = val
                else:
                    vali = Parameter(*val)
                    if np.isnan(vali.value):
                        vali.value = self[key][i][0]
                self.set_parameter(key,vali,match=d)
    
    def vary_upper_level_energy(self,match=None,vary=False,step=None):
        if match is not None:
            raise ImplementationError()
        keys = [key+'_u' for key in self._level_class.defining_qn]
        for d,m in self.unique_dicts_match(*keys):
            i = tools.find(m)
            self.set_parameter('E_u',Parameter(self['E_u'][i[0]],vary,step),match=d)

class Atomic(Generic):
    _level_class,_level_keys,defining_qn = _collect_level(levels.Atomic)
    default_prototypes = {key:prototypes[key] for key in {
        *_level_keys,
        *Generic.default_prototypes,
    }}
    default_xkey = 'J_l'
    default_zkeys = ['species_u','conf_u','species_l','conf_l','ΔJ']

class Linear(Generic):
    _level_class,_level_keys,defining_qn = _collect_level(levels.Linear)
    default_prototypes = {key:prototypes[key] for key in {
        *_level_keys,
        *Generic.default_prototypes,
        'fv', 'νv', 'μv',
        'SJ',
    }}
    default_xkey = 'J_l'
    default_zkeys = ['species_u','label_u','species_l','label_l','ΔJ']

class LinearDiatomic(Linear):

    _level_class,_level_keys,defining_qn = _collect_level(levels.LinearDiatomic)
    default_prototypes = {key:prototypes[key] for key in
                          {*_level_keys, *Linear.default_prototypes,}}
    default_xkey = 'J_u'
    default_zkeys = ['species_u','label_u','v_u','species_l','label_l','v_l', 'ΔJ']

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


class LinearTriatomic(Linear):
    """E.g., CO2, CS2."""

    _level_class,_level_keys,defining_qn = _collect_level(levels.LinearTriatomic)
    default_prototypes = {key:prototypes[key] for key in {*_level_keys, *Linear.default_prototypes,}}
    default_xkey = 'J_l'
    default_zkeys = ['species_u', 'ν1_u', 'ν2_u', 'ν3_u', 'l2_u', 'species_l', 'ν1_l', 'ν2_l', 'ν3_l', 'l2_l', 'ΔJ']

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

