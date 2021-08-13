from scipy import constants
import numpy as np

from . import tools

"""Convert between units and rigorously related quantities."""


## relationship between units and a standard SI unit.  Elements can be
## conversion factors to SI, or a pair of conversion and inverse
## conversion functions
groups = {

    'length': {
        'm'           :1.                ,  # astronomical units       ,
        'pc'          :3.0857e16       ,    # parsecs
        'fm'          :1e-15               ,
        'pm'          :1e-12               ,
        'nm'          :1e-9               ,
        'μm'          :1e-6               ,
        'mm'          :1e-3               ,
        'cm'          :1e-2               ,
        'km'          :1e3              ,
        'Mm'          :1e6              ,
        'Gm'          :1e9              ,
        'solar_radius':6.955e8         ,
        'AU'          :1.496e11        ,  # astronomical units       ,
        'au'          :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'Bohr'        :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'a0'          :5.2917721092e-11, #  atomic units (Bohr radius, a0),
        'Å'           :1e-10             ,
    },

    'inverse_length': {
        'm-1' :1   ,
        'cm-1':1e2,
        'mm-1':1e3,
        'μm-1':1e6,
        'nm-1':1e9,
    },

    'area': {
        'm2'   :  1., 
        'cm2'  :  1e-4, 
        'Mb'   :  1e-20, 
    },

    'inverse_area': {
        'm-2' : 1., 
        'cm-2': 1e4,
    },

    'volume': {
        'm3'  : 1., 
        'cm3' : 1e-6,
        'l'   : 1e-3,
        'L'   : 1e-3,
    },

    'inverse_volume': {
        'm-3' : 1., 
        'cm-3': 1e6,
    },

    'time': {
        's':1.,
        'ms':1e-3,
        'μs':1e-6,
        'ns':1e-9,
        'ps':1e-12,
        'fs':1e-15,
        'minute':60,
        'hour':60*60,
        'day':60*60*24,
        'week':60*60*24*7,
        'year':60*60*24*7*365,
    },

    'frequency': {
        'Hz' :1  ,
        'kHz':1e3,
        'MHz':1e6,
        'GHz':1e9,
        'radians':2*constants.pi,
    },

    'energy': {
        'J'         :1.                    ,
        'K'         :1/constants.Boltzmann ,
        'cal'       :4.184               ,
        'eV'        :1.602176634e-19     ,
        'erg'       :1e-7                   ,
        'Hartree'   :4.35974434e-18      , # atomic units /hartree
        'au'        :4.35974434e-18      , # atomic units /hartree
        'kJ.mol-1'  :1e3/constants.Avogadro,
        'kcal.mol-1':6.9477e-21          ,
    },

    'mass': {
        'kg'        :1                      ,
        'g'         :1e-3                    ,
        'solar_mass':1.98855e30           ,
        'amu'       :constants.atomic_mass,
    },
    
    'velocity': {
        'm.s-1'     :1.         ,
        'km.s-1'    :1e3         ,
    },

    'dipole moment': {
        'Debye' : 1.,
        'au'    : 2.541765,
    },

    'pressure': {
        'Pa'      :  1.        ,
        'kPa'     :  1e3      ,
        'bar'     :  1e5      ,
        'mbar'    :  1e2      ,
        'mb'      :  1e2      ,
        'atm'     :  101325.,
        'Torr'    :  133.322 ,
        'dyn.cm-2':  1/(1e5*1e-4)  ,
    },

    'photon': {
        'Hz' : 1,
        'kHz': 1e3,
        'MHz': 1e6,
        'GHz': 1e9,
        'J':   1/constants.h,
        'eV':  constants.electron_volt/constants.h,
        'm':   (lambda m: constants.c/m,lambda Hz: constants.c/Hz),
        'μm':  (lambda μm: constants.c/(μm*1e-6),lambda Hz: 1e6*constants.c/Hz),
        'nm':  (lambda nm: constants.c/(nm*1e-9),lambda Hz: 1e9*constants.c/Hz),
        'Å':   (lambda Å: constants.c/(Å*1e-10),lambda Hz: 1e10*constants.c/Hz),
        'm-1': (lambda m: m*constants.c,lambda Hz: Hz/constants.c,),
        'cm-1':(lambda invcm: constants.c*(1e2*invcm),lambda Hz: 1e-2/(constants.c/Hz)),
    },

    # unit_conversions[('Debye','au')] = lambda x: x/2.541765
    # unit_conversions[('au','Debye')] = lambda x: x*2.541765
    # unit_conversions[('Debye','Cm')] = lambda x: x*3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('Cm','Debye')] = lambda x: x/3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('C','statC')] = lambda x: 3.33564e-10*x # 1 Couloub = sqrt(4πε0/1e9)×stat Coulomb -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('statC','C')] = lambda x: 2997924580*x  # 1 stat Couloub = sqrt(1e9/4πε0)×Coulomb -- WARNING: these are not dimensionally similar units!!!


}
def units(value,unit_in,unit_out,group=None):       
    """Convert units. Group might be needed for units with common
    names."""
    ## cast to array if necessary
    if not np.isscalar(value):
        value = np.asarray(value)
    ## trivial case
    if unit_in == unit_out:
        return value
    ## find group containing this conversion if not specified
    if group is None:
        for factors in groups.values():
                if unit_in in factors and unit_out in factors:
                    break
        else:
            raise Exception(f"Could not find conversion group for {unit_in=} to {unit_out=}.")
    else:
        try:
            factors = groups[group]
        except KeyError:
            raise Exception(f"No conversion for {unit_in=} to {unit_out=} in {group=}.")
    ## convert to SI from unit_in
    factor = factors[unit_in]
    if isinstance(factor,(float,int)):
        value = value*factor
    else:
        value = factor[0](value)
    ## convert to unit_out from SI
    factor = factors[unit_out]
    if isinstance(factor,(float,int)):
        value = value/factor
    else:
        value = factor[1](value)
    return value

def difference(difference,value,unit_in,unit_out,group=None):
    """Convert an absolute finite difference -- not a linear
    approximation."""
    return(np.abs(
        +units(value+difference/2.,unit_in,unit_out,group)
        -units(value-difference/2.,unit_in,unit_out,group)))

        
###################################
## quantum mechanical quantities ##
###################################

def lifetime_to_linewidth(lifetime):
    """Convert lifetime (s) of transition to linewidth (cm-1 FWHM). tau=1/2/pi/gamma/c"""
    return 5.309e-12/lifetime

def linewidth_to_lifetime(linewidth):
    """Convert linewidth (cm-1 FWHM) of transition to lifetime (s). tau=1/2/pi/gamma/c"""
    return 5.309e-12/linewidth

def linewidth_to_rate(linewidth):
    """Convert linewidth (cm-1 FWHM) of transition to lifetime (s). tau=1/2/pi/gamma/c"""
    return linewidth/5.309e-12

def rate_to_linewidth(rate):
    """Convert lifetime (s) of transition to linewidth (cm-1 FWHM)."""
    return 5.309e-12*rate

def transition_moment_to_band_strength(μv):
    """Convert electronic-vibrational transition moment (au) into a band
    summed line strength. This is not really that well defined."""
    return(μv**2)

def transition_moment_to_band_fvalue(μv,νv,Λp,Λpp):
    return(
        band_strength_to_band_fvalue(
            transition_moment_to_band_strength(μv),
            νv,Λp,Λpp))

def band_fvalue_to_transition_moment(fv,νv,Λp,Λpp):
    """Convert band f-value to the absolute value of the transition mometn
    |μ|= |sqrt(q(v',v'')*Re**2)|"""
    return(np.sqrt(band_fvalue_to_band_strength(fv,νv,Λp,Λpp)))

def band_strength_to_band_fvalue(Sv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-summed f-value."""
    return(Sv*3.038e-6*νv*(2-my.kronecker_delta(0,Λp+Λpp))/(2-my.kronecker_delta(0,Λpp)))

def band_fvalue_to_band_strength(fv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-summed f-value."""
    return fv/(3.038e-6*νv *(2-tools.kronecker_delta(0,Λp+Λpp)) /(2-tools.kronecker_delta(0,Λpp)))

def band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-averaged emission
    rate."""
    return(2.026e-6*νv**3*Sv *(2-my.kronecker_delta(0,Λp+Λpp)) /(2-my.kronecker_delta(0,Λp)))

def band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp):
    """Convert band summed linestrength to a band-averaged emission
    rate."""
    return(Aev/(2.026e-6*νv**3 *(2-my.kronecker_delta(0,Λp+Λpp)) /(2-my.kronecker_delta(0,Λp))))

def band_emission_rate_to_band_fvalue(Aev,νv,Λp,Λpp):
    return(band_strength_to_band_fvalue(
        band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),
        νv,Λp,Λpp))

def band_fvalue_to_band_emission_rate(fv,νv,Λp,Λpp):
    return(band_strength_to_band_emission_rate(
        band_fvalue_to_band_strength(fv,νv,Λp,Λpp),
        νv,Λp,Λpp))

def fvalue_to_line_strength(f,Jpp,ν):
    """Convert f-value (dimensionless) to line strength (au). From Eq. 8 larsson1983."""
    return(f/3.038e-6/ν*(2*Jpp+1))

def line_strength_to_fvalue(Sij,Jpp,ν):
    """Convert line strength (au) to f-value (dimensionless). From Eq. 8 larsson1983."""
    return(3.038e-6*ν*Sij/(2*Jpp+1))

def emission_rate_to_line_strength(Ae,Jp,ν):
    """Convert f-value (dimensionless) to line strength (au). From Eq. 8 larsson1983."""
    return(Ae/(2.026e-6*ν**3/(2*Jp+1)))

def line_strength_to_emission_rate(Sij,Jp,ν):
    """Convert line strength (au) to f-value (dimensionless). From Eq. 8 larsson1983."""
    return(Sij*2.026e-6*ν**3/(2*Jp+1))

def fvalue_to_emission_rate(f,ν,Jpp,Jp):
    """Convert f-value to emission rate where upper and lower level
    degeneracies are computed from J."""
    return(line_strength_to_emission_rate(fvalue_to_line_strength(f,Jpp,ν), Jp,ν))

def emission_rate_to_fvalue(Ae,ν,Jpp,Jp):
    """Convert emission rate to f-value where upper and lower level
    degeneracies are computed from J."""
    return(line_strength_to_fvalue(emission_rate_to_line_strength(Ae,Jp,ν), Jpp,ν))

# def fvalue_to_band_fvalue(line_fvalue,symmetryp,branch,Jpp,symmetrypp='1σ'):
    # """Convert line f-value to band f-value. NOTE THAT honl_london_factor IS AN IMPROVEMENT OVER honllondon_factor, COULD IMPLEMENT THAT HERE."""
    # return(line_fvalue/honllondon_factor(Jpp=Jpp,branch=branch,symmetryp=symmetryp,symmetrypp=symmetrypp)*degen(Jpp,symmetryp))

# def band_fvalue_to_fvalue(fv,,branch,Jpp,symmetrypp='1σ'):
    # """ Convert line band f-value to f-value. NOTE THAT honl_london_factor IS AN IMPROVEMENT OVER honllondon_factor, COULD IMPLEMENT THAT HERE."""
    # return(band_fvalue*honllondonfactor(Jpp=Jpp,branch=branch,symmetryp=symmetryp,symmetrypp=symmetrypp)/degen(Jpp,symmetryp))

def band_cross_section_to_band_fvalue(σv):
    """Convert band integrated cross section band-summed f-value. From
    units of cm2*cm-1 to dimensionless."""
    return(1.1296e12*σv)

def band_fvalue_to_band_cross_section(fv):
    """Convert band integrated cross section band-summed f-value. From
    units of cm2*cm-1 to dimensionless."""
    return(fv/1.1296e12)

def cross_section_to_fvalue(σv,temperaturepp,**qnpp):
    """Convert line strength to line f-value. From units of cm2*cm-1
    to dimensionless. If temperature is none then the boltzmann distribution is not used."""
    return(1.1296e12*σv/database.get_boltzmann_population(temperaturepp,**qnpp).squeeze())

def fvalue_to_cross_section(f,temperaturepp,**qnpp):
    """Convert line strength to line f-value. From units of cm2*cm-1
    to dimensionless. If temperature is none then the boltzmann distribution is not used."""
    return(f/(1.1296e12/database.get_boltzmann_population(temperaturepp,**qnpp).squeeze()))

def differential_oscillator_strength_to_cross_section(
        df, units_in='(cm-1)-1', units_out='cm2'):
    if units_in=='(cm-1)-1':
        pass
    elif units_in=='eV-1':
        df /= convert_units(1,'eV','cm-1')
    σ = df/1.1296e12            # from (cm-1)-1 to cm2
    return(convert_units(σ,'cm2',units_out))
     
def pressure_temperature_to_density(p,T,punits='Pa',nunits='m-3'):
    """p = nkT"""
    p = units(p,punits,'Pa')
    n = p/(constants.Boltzmann*T)
    n = units(n,'m-3',nunits)
    return n

def pressure_to_column_density_density(p,T,L,punits='Pa',Lunits='cm',Nunits='cm-2'):
    """p = NLkT"""
    return units(units(p,punits,'Pa') /(constants.Boltzmann*T*unit(L,Lunits,'m')), 'm-3',Nunits)

def doppler_width(
        temperature,            # K
        mass,                   # amu
        ν,                      # wavenumber in cm-1
        units='cm-1.FWHM',      # Units of output widths.
):
    """Calculate Doppler width given temperature and species mass."""
    dk = 2.*6.331e-8*np.sqrt(temperature*32./mass)*ν
    if units=='cm-1.FWHM':
        return dk
    elif units in ('Å.FHWM','A.FWHM','Angstrom.FWHM','Angstroms.FWHM',):
        return tools.dk2dA(dk,ν)
    elif units=='nm.FWHM':
        return tools.dk2dnm(dk,ν)
    elif units=='km.s-1 1σ':
        return tools.dk2b(dk,ν)
    elif units=='km.s-1.FWHM':
        return tools.dk2bFWHM(dk,ν)
    else:
        raise ValueError('units not recognised: '+repr(units))
