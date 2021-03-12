import itertools 
import functools
from copy import copy,deepcopy
import re
from pprint import pprint

from scipy import constants
import numpy as np
from numpy import nan,array

from .dataset import Dataset
from . import convert
from .tools import vectorise,cache,file_to_dict
from . import tools
from . import database
from . import kinetics
from .exceptions import InferException,MissingDataException
# from .levels_prototypes import prototypes


prototypes = {}

prototypes['notes'] = dict(description="Notes regarding this line" , kind='U' ,infer=[])
prototypes['author'] = dict(description="Author of data or printed file" ,kind='U' ,infer=[])
prototypes['reference'] = dict(description="Published reference" ,kind='U' ,infer=[])
prototypes['date'] = dict(description="Date data collected or printed" ,kind='U' ,infer=[])
prototypes['species'] = dict(description="Chemical species" ,kind='U' ,infer=[])

@vectorise(cache=True,vargs=(1,))
def _f0(self,species):
    return kinetics.get_species(species).point_group
prototypes['point_group']  = dict(description="Symmetry point group of species.", kind='U',fmt='s', infer=[(('species',),_f0)])

@vectorise(vargs=(1,),cache=True)
def _f0(self,species):
    return kinetics.get_species(species)['mass']
@vectorise(vargs=(1,),cache=True)
def _f1(self,species):
    try:
        return database.get_species_property(species,'mass')
    except MissingDataException as err:
        raise InferException(str(err))
prototypes['mass'] = dict(description="Mass (amu)",kind='f', fmt='<11.4f', infer=[(('species',), _f0),])
prototypes['reduced_mass'] = dict(description="Reduced mass (amu)", kind='f', fmt='<11.4f', infer=[(('species','database',), lambda self,species: _get_species_property(species,'reduced_mass'))])
prototypes['E'] = dict(description="Level energy (cm-1)" ,kind='f' ,fmt='<14.7f' ,infer=[],)
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind='f',infer=[])
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind='f',fmt='<10.5g', infer=[(('mass','Ttr','ν'), lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν)])


@vectorise(cache=True,vargs=(1,))
def _f0(self,point_group):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if point_group in ('K','C∞v'):
        return 1.
    else:
        raise InferException

@vectorise(cache=True,vargs=(1,2,3))
def _f1(self,point_group,Inuclear,sa):
    """Calculate homonuclear diatomic molecule level degeneracy."""
    if point_group in ('D∞h'):
        ## get total number of even or odd exchange combinations
        ntotal = (2*Inuclear+1)**2
        neven = 2*Inuclear+1 + (ntotal-(2*Inuclear+1))/2
        nodd = ntotal - neven
        if Inuclear%1==0:
            ## fermion
            if sa==+1:
                return neven
            else:
                return nodd
        else:
            ## boson
            if sa==+1:
                return nodd
            else:
                return neven
    else:
        raise InferException()

prototypes['gnuclear'] = dict(description="Nuclear spin level degeneracy (relative only)." , kind='i' , infer=[(('point_group',),_f0),( ('point_group','Inuclear','sa'),_f1),])
prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind='i' , infer=[(('J','gnuclear'),lambda self,J,gnuclear: (2*J+1)*gnuclear,)])
prototypes['pm'] = dict(description="Total inversion symmetry" ,kind='i' ,infer=[])
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind='f', fmt='<10.5g', infer=[(('A',),lambda self,τ: 5.309e-12*A,)])
prototypes['τ'] = dict(description="Total decay lifetime (s)", kind='f', infer=[(('A',), lambda self,A: 1/A,)])       
prototypes['A'] = dict(description="Total decay rate (s-1)", kind='f', infer=[(('Γ',),lambda self,Γ: Γ/5.309e-12,)])
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind='f',fmt='>0.1f',infer=[])
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind='f', infer=[(('J','SR'),lambda self,J,SR: J-SR,)])
prototypes['S'] = dict(description="Total electronic spin quantum number", kind='f',infer=[(('species','label'),lambda self,species,label: database.get_electronic_state_property(species,label,'S'),)])
prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind='f',infer=[((),lambda self,: 0.,)])
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind='f', fmt='0.2f', infer=[])
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq)])
# prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind='U', infer=[(('database',), lambda self,: 'database',)])

# @vectorise(cache=True,vargs=(1,2))
def _f5(self,species,Tex):
    if self.Zsource != 'HITRAN':
        raise InferException(f'Zsource not "HITRAN".')
    from . import hitran
    return hitran.get_partition_function(species,Tex)

def _f3(self,species,Tex,E,g):
    """Compute partition function from data in self."""
    if self.Zsource != 'self':
        raise InferException(f'Zsource not "self".')
    retval = np.full(species.shape,nan)
    for (speciesi,Texi),i in tools.unique_combinations_mask(species,Tex):
        kT = convert.units(constants.Boltzmann,'J','cm-1')*Texi
        retval[i] = np.sum(g[i]*np.exp(-E[i]/kT))
    return retval
prototypes['Z'] = dict(description="Partition function.", kind='f', fmt='<11.3e', infer=[(('species','Tex','E','g'),_f3), (('species','Tex'),_f5),])
prototypes['α'] = dict(description="State population", kind='f', fmt='<11.4e', infer=[(('Z','E','g','Tex'), lambda self,Z,E,g,Tex : g*np.exp(-E/(convert.units(constants.Boltzmann,'J','cm-1')*Tex))/Z,)])
prototypes['Nself'] = dict(description="Column density (cm2)",kind='f',fmt='<11.3e', infer=[])
prototypes['label'] = dict(description="Label of electronic state", kind='U',infer=[])
prototypes['v'] = dict(description="Vibrational quantum number", kind='i',infer=[])
prototypes['ν1'] = dict(description="Vibrational quantum number for mode 1", kind='i',infer=[])
prototypes['ν2'] = dict(description="Vibrational quantum number for mode 2", kind='i',infer=[])
prototypes['ν3'] = dict(description="Vibrational quantum number for mode 3", kind='i',infer=[])
prototypes['ν4'] = dict(description="Vibrational quantum number for mode 4", kind='i',infer=[])
prototypes['l2'] = dict(description="Vibrational angular momentum 2", kind='i',infer=[])
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind='i',infer=[])
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind='i',infer=[])
prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind='i',infer=[])
@vectorise(cache=True,vargs=(1,2))
def _f0(self,ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['i'] = dict(description="Total parity.", kind='i',infer=[])
prototypes['σv'] = dict(description="Symmetry with respect to σv reflection.", kind='i',infer=[(('ef','J'),_f0,)])
prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind='i',infer=[])
prototypes['sa'] = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind='i',infer=[(('σv','gu'),lambda self,σv,gu: σv*gu,)])
def _f0(self,S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef'] = dict(description="e/f symmetry", kind='i',infer=[(('S','Λ','s'),_f0,)],fmt='+1d')
prototypes['Fi'] = dict(description="Spin multiplet index", kind='i',infer=[])
prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind='f',fmt='g',infer=[(('Λ','Σ'),lambda self,Λ,S: Λ+S,)])
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind='f',infer=[],fmt='+g',)
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind='f',infer=[])
prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind='f',infer=[])


## Effective Hamiltonian parameters
prototypes['Tv']  = dict(description='Term origin (cm-1)' ,kind='f',fmt='0.6f',infer=[])
prototypes['dTv'] = dict(description='Uncertainty in Term origin (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Bv']  = dict(description='Rotational constant (cm-1)' ,kind='f',fmt='0.8f',infer=[])
prototypes['dBv'] = dict(description='Uncertainty in rotational constant (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Dv']  = dict(description='Centrifugal distortion (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dDv'] = dict(description='Uncertainty in centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['Hv']  = dict(description='Third order centrifugal distortion (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dHv'] = dict(description='Uncertainty in thrid order centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='3g',infer=[])
prototypes['Lv']  = dict(description='Fourth order centrifugal distortion (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dLv'] = dict(description='Uncertainty in fourth order centrifugal distortion (1σ, cm-1)' ,kind='f',fmt='3g',infer=[])
prototypes['Av']  = dict(description='Spin-orbit energy (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dAv'] = dict(description='Uncertainty in spin-orbit energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['ADv'] = dict(description='Spin-orbit centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dADv']= dict(description='Uncertainty in spin-orbit centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['AHv'] = dict(description='Higher-order spin-orbit centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dAHv']= dict(description='Uncertainty in higher-order spin-orbit centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['λv']  = dict(description='Spin-spin energy (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dλv'] = dict(description='Uncertainty in spin-spin energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['λDv'] = dict(description='Spin-spin centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dλDv']= dict(description='Uncertainty in spin-spin centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['λHv'] = dict(description='Higher-order spin-spin centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dλHv']= dict(description='Uncertainty in higher-order spin-spin centrifugal distortion (1σ, cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['γv']  = dict(description='Spin-rotation energy (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dγv'] = dict(description='Uncertainty in spin-rotation energy (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['γDv'] = dict(description='Spin-rotation centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dγDv']= dict(description='Uncertainty in spin-rotation centrifugal distortion (cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['γHv'] = dict(description='Higher-orders pin-rotation centrifugal distortion (cm-1)',kind='f',fmt='0.6g',infer=[])
prototypes['dγHv']= dict(description='Uncertainty in higher-order spin-rotation centrifugal distortion (cm-1)',kind='f',fmt='0.2g',infer=[])
prototypes['ov']  = dict(description='Λ-doubling constant o (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dov'] = dict(description='Uncertainty in Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oDv']  = dict(description='Higher-order Λ-doubling constant o (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['doDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oHv']  = dict(description='Higher-order Λ-doubling constant o (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['doHv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['oLv']  = dict(description='Ligher-order Λ-doubling constant o (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['doLv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['pv']  = dict(description='Λ-doubling constant p (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dpv'] = dict(description='Uncertainty in Λ-doubling constant p (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['pDv']  = dict(description='Higher-order Λ-doubling constant p (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dpDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant p (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['qv']  = dict(description='Λ-doubling constant q (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dqv'] = dict(description='Uncertainty in Λ-doubling constant q (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])
prototypes['qDv']  = dict(description='Higher-order Λ-doubling constant q (cm-1)' ,kind='f',fmt='0.6g',infer=[])
prototypes['dqDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant q (1σ, cm-1)' ,kind='f',fmt='0.2g',infer=[])

prototypes['Γv'] = dict(description="Total electronic-vibrational linewidth (cm-1 FWHM)", kind='f',  fmt='<10.5g', strictly_positive=True, infer=[(('τ',),lambda self,τ: 5.309e-12/τ,)]) # tau=1/2/pi/gamma/c
prototypes['dΓv'] = dict(description="Uncertainty in total electronic-vibrational linewidth (cm-1 FWHM 1σ)", kind='f',  fmt='<10.5g', infer=[(('Γ','τ','dτ'), lambda self,Γ,τ,dτ: dτ*Γ/τ,)])
prototypes['τv'] = dict(description="Total electronic-vibrational decay lifetime (s)", kind='f',  fmt='<10.5g', infer=[(('Γv',), lambda self,Γv: 5.309e-12/Γv),( ('Atv',), lambda self,Atv: 1/Atv,)]) 
prototypes['dτv'] = dict(description="Uncertainty in total electronic-vibrational decay lifetime (s 1σ)", kind='f',  fmt='<10.5g', infer=[(('Γ','dΓ','τ'), lambda self,Γ,dΓ,τ: dΓ/Γ*τ),(('At','dAt','τ'), lambda self,At,dAt,τ: dAt/At*τ,)])
prototypes['Atv'] = dict(description="Total electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('τv',), lambda self,τv: 1/τv), (('Adv','Ave'), lambda self,Adv,Aev: Adv+Aev,), (('Aev',), lambda self,Aev: Aev),( ('Adv',), lambda self,Adv: Adv,)])# Test for Ad and Ae, if failed then one or the other is undefined/zero
prototypes['dAtv']= dict(description="Uncertainty in total electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[(('τ','dτ','At'), lambda self,τ,dτ,At: dτ/τ*At,)])
prototypes['Adv'] = dict(description="Nonradiative electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('At','Ae'), lambda self,At,Ae: At-Ae,)])
prototypes['dAdv']= dict(description="Uncertainty in nonradiative electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['Aev'] = dict(description="Radiative electronic-vibrational decay rate (s-1)", kind='f',  fmt='<10.5g', infer=[(('At','Ad'), lambda self,At,Ad: At-Ad,)])
prototypes['dAev']= dict(description="Uncertainty in radiative electronic-vibrational decay rate (s-1 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['ηdv'] = dict(description="Fractional probability of electronic-vibrational level decaying nonradiatively (dimensionless)", kind='f',  fmt='<10.5g', infer=[(('At','Ad'),lambda self,At,Ad:Ad/A,)])
prototypes['dηdv']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying by dissociation by any channels (dimensionless, 1σ)", kind='f',  fmt='<10.5g', infer=[])
prototypes['ηev'] = dict(description="Fractional probability of electronic-vibrational level decaying radiatively (dimensionless)", kind='f',  fmt='<10.5g', infer=[(('At','Ae'),lambda self,At,Ae:Ae/A,)])
prototypes['dηev']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying radiatively (dimensionless, 1σ)", kind='f',  fmt='<10.5g', infer=[])

## v+1 reduced versions of these
def _vibrationally_reduce(
        self,
        reduced_quantum_number,
        reduced_polynomial_order,
        val,dval=None,
):
    """Reduce a variable."""
    reduced = np.full(val.shape,np.nan)
    ## loop through unique rotational series
    for keyvals in self.unique_dicts(*self.qn_defining_independent_vibrational_progressions):
        i = self.match(**keyvals)
        x = self[reduced_quantum_number][i]*(self[reduced_quantum_number][i]+1)
        y = val[i]
        ## do an unweighted mean if uncertainties are not set or all the same
        if dval is None:
            dy = None
        else:
            dy = dval[i]
        p = my.polyfit(x,y,dy,order=max(0,min(reduced_polynomial_order,sum(i)-2)),error_on_missing_dy=False)
        reduced[i] = p['residuals']
    return(reduced)

prototypes['Tvreduced'] = dict(description="Vibrational term value reduced by a polynomial in (v+1/2) (cm-1)", kind='f',  fmt='<11.4f',
    infer=[(('self','reduced_quantum_number','reduced_polynomial_order','Tv','dTv'), _vibrationally_reduce), # dTv is known -- use in a weighted mean
           (('self','reduced_quantum_number','reduced_polynomial_order','Tv'), _vibrationally_reduce,)]) # dTv is not known
prototypes['dTvreduced'] = dict(description="Uncertainty in vibrational term value reduced by a polynomial in (v+1/2) (cm-1 1σ)", kind='f',  fmt='<11.4f', infer=[(('dT',),lambda self,dT: dT,)])
prototypes['Tvreduced_common'] = dict(description="Term values reduced by a common polynomial in (v+1/2) (cm-1)", kind='f',  fmt='<11.4f', infer=[(('v','Tv','Tvreduced_common_polynomial'), lambda self,v,Tv,Tvreduced_common_polynomial: Tv-np.polyval(Tvreduced_common_polynomial,v+0.5)),(('v','Tv'), lambda self,v,Tv: Tv-np.polyval(np.polyfit(v+0.5,Tv,3),v+0.5),)])
prototypes['dTvreduced_common'] = dict(description="Uncertaintty in term values reduced by a common polynomial in (v+1/2) (cm-1 1σ)", kind='f',  fmt='<11.4e', infer=[(('dTv',), lambda self,dTv: dTv)])
prototypes['Tvreduced_common_polynomial'] = dict(description="Polynomial in terms of (v+1/2) to reduce all term values commonly (cm-1)", kind='o', infer=[])
prototypes['Bv_μscaled']  = dict(description='Rotational constant scaled by reduced mass to an isotopologue-independent value (cm-1)' , kind='f',fmt='0.8f', infer=[(('Bv','reduced_mass'),lambda self,Bv,reduced_mass: Bv*reduced_mass,)])
prototypes['dBv_μscaled'] = dict(description='Uncertainty in Bv_μscaled (1σ, cm-1)' ,kind='f',fmt='0.2g', infer=[(('Bv','dBv','Bv_μscaled'),lambda self,Bv,dBv,Bv_μscaled:dBv/Bv*Bv_μscaled,)])

def _collect_prototypes(*keys):
    retval = {key:prototypes[key] for key in keys}
    return retval

class Base(Dataset):
    """Common stuff for for lines and levels."""
    _prototypes = _collect_prototypes()
    default_attributes = (*Dataset.default_attributes,'Zsource')

    def __init__(self,name=None,**kwargs):
        """Default_name is decoded to give default values. Kwargs ca be
        scalar data, further default values of vector data, or if vetors
        themselves will populate data arrays."""
        Dataset.__init__(self,name=name,prototypes=self._prototypes,**kwargs)
        self._cache = {}
        # self.pop_format_input_function()
        # self.automatic_format_input_function(limit_to_args=('name',))
        # self.automatic_format_input_function(multiline=True )
        # self.pop_format_input_function()
        # self.automatic_format_input_function(limit_to_args=('name',))
        self.default_zkeys = self.defining_qn

class Generic(Base):
    """A generic level."""
    _prototypes = _collect_prototypes(
        'species',
        'point_group',
        'E','Eref',
        'Γ','ΓD',
        'J',
        'g','gnuclear',
        'Teq','Tex','Z','α',
        'Nself',
    )
    defining_qn = ('species',)

class LinearTriatomic(Generic):
    """A generic level."""
    _prototypes = _collect_prototypes(
        'species',
        'label',
        'ν1','ν2','ν3','l2',
        'point_group',
        'E','Eref',
        'Γ','ΓD',
        'J',
        'g','gnuclear',
        'Teq','Tex','Z','α',
        'Nself',
    )
    defining_qn = ('species','label','ν1','ν2','ν3','l2','J')
        

class Diatomic(Base):
    _prototypes = _collect_prototypes(
        'species',
        'point_group',
        'E','Eref',
        'Γ','ΓD',
        'Inuclear',
        'g','gnuclear',
        'Teq','Tex','Z','α',
        'Nself',
        'label', 'Λ','s','S','LSsign',
        'i','gu', 'σv','sa','ef', 
        'Fi','Ω','Σ','SR',
        'v',
        'Γv','τv','Atv','Adv','Aev',
        'ηdv','ηev',
        'Tv','Bv','Dv','Hv','Lv',
        'Av','ADv','AHv',
        'λv','λDv','λHv',
        'γv','γDv','γHv',
        'ov','oDv','oHv','oLv',
        'pv','qv',
        'pDv','qDv',
        'Tvreduced','Tvreduced_common',
        'Bv_μscaled',
        'J','pm','N','S',
        'Teq',
        'α',
    )
    defining_qn = ('species','label','v','Σ','ef','J')

    def load_from_duo(self,filename):
        """Load an output level list computed by DUO (yurchenko2016)."""
        data = file_to_dict(filename)
        if len(data) == 11:
            for icolumn,key in enumerate(('index','E','g','J','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
        elif len(data) == 12:
            for icolumn,key in enumerate(('index','E','g','J','E_residual','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
            data.pop('E_residual') # not currently used
        if len(data)==0:
            print(f'warning: no data found in {repr(filename)}')
            return
        ## get Λ/Σ/Ω sign conventions to match mine: Λ is +, Σ + or -,
        ## Ω is absolute
        i = data['Λ'] < 0
        data['Σ'][i] *= -1
        data['Λ'] = np.abs(data['Λ'])
        data['Ω'] = np.abs(data['Ω'])
        ## translate +/- to +1/-1
        i = data['s']=='-'
        data['s'] = np.full(len(data['s']),+1,dtype=int)
        data['s'][i] = -1
        ## translate e/f to +1/-1
        i = data['ef']=='f'
        data['ef'] = np.full(len(data['ef']),+1,dtype=int)
        data['ef'][i] = -1
        data.pop('index')
        self.extend(**data)

    def load_from_spectra(self,filename):
        """Old filetype. Incomplete"""
        data = file_to_dict(filename)
        ## keys to translate
        for key_old,key_new in (
                ('T','E'),
                ('Tref','Eref'),
                ):
            if key_old in data:
                assert key_new not in data
                data[key_new] = data.pop(key_old)
        ## data to modify
        if 'ef' in data:
            i = data['ef']=='f'
            data['ef'] = np.full(len(data['ef']),+1,dtype=int)
            data['ef'][i] = -1
        ## data to ignore
        for key in (
                'level_transition_type',
                'partition_source',
                ):
            data.pop(key)
        self.extend(**data)

# class LinearTriatomic(Generic):
    # """Rotational levels of a linear triatomic.  Constructed to load
    # linear triatomic data from HITRAN (e.g., Table 3 in rothman2005."""
    # _prototypes = _collect_prototypes(*Generic._prototypes,
        # 'label', 'S',
        # 'v1','v2','v3','l2',    # vibrational coordinates
        # 'J',
    # )
    # default_zkeys = ('label','v1','v2','v3','l2',)
