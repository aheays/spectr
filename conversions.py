# from . import *
from scipy import constants
import numpy as np

###########
## units ##
###########
setattr(constants,'a0',5.291772108e-11) # Bohr radius, not in scipy.constants
setattr(constants,'Eh',4.35974417e-18) # Hatree energy, not in scipy.constants

## relationship between units and canonical SI unit
canonical_factors = {
    ## energy
    'J'         :(1.                    ,'energy'),
    'K'         :(1/constants.Boltzmann ,'energy'),
    'cal'       :(1/4.184               ,'energy'),
    'eV'        :(1/1.602176634e-19     ,'energy'),
    'erg'       :(1e7                   ,'energy'),
    'Hartree'   :(1/4.35974434e-18      ,'energy'), # atomic units /hartree
    'au_energy' :(1/4.35974434e-18      ,'energy'), # atomic units /hartree
    'kJ.mol-1'  :(constants.Avogadro/1e3,'energy'),
    'kcal.mol-1':(1/6.9477e-21          ,'energy'),
    ## frequency
    'Hz' :(1  ,'frequency'),
    'kHz':(1e3,'frequency'),
    'MHz':(1e6,'frequency'),
    'GHz':(1e9,'frequency'),
    ## wavenumbers
    'm-1' :(1   ,'wavenumber'),
    'cm-1':(1e-2,'wavenumber'),
    ## lengths
    'm'           :(1.                ,'length'),  # astronomical units       ,
    'pc'          :(1/3.0857e16       ,'length'),    # parsecs
    'μm'          :(1e6               ,'length'),
    'mm'          :(1e3               ,'length'),
    'cm'          :(1e2               ,'length'),
    'km'          :(1e-3              ,'length'),
    'Mm'          :(1e-6              ,'length'),
    'Gm'          :(1e-9              ,'length'),
    'nm'          :(1e9               ,'length'),
    'solar_radius':(1/6.955e8         ,'length'),
    'AU'          :(1/1.496e11        ,'length'),  # astronomical units       ,
    'au'          :(1/5.2917721092e-11,'length'), #  atomic units (Bohr radius, a0),
    'Å'           :(1e+10             ,'length'),
    ## mass
    'kg'        :(1.,          'mass')         ,
    'g'         :(1e3,         'mass')         ,
    'solar_mass':(1/1.98855e30,'mass'),
    ## velocity
    'm.s-1'     :(1.,          'velocity')         ,
    'km.s-1'    :(1e-3,        'velocity')         ,
    ## dipole moment
    'Debye' : (1.,          'dipole moment'),
    'au'    : (1/2.541765,  'dipole moment'),
    ## pressure
    'Pa'      :  (1.        ,  'pressure'),
    'kPa'     :  (1e-3      ,  'pressure'),
    'bar'     :  (1e-5      ,  'pressure'),
    'atm'     :  (1./101325.,  'pressure'),
    'Torr'    :  (1/133.322 ,  'pressure'),
    'dyn.cm-2':  (1e5*1e-4  ,  'pressure'),
    # unit_conversions[('Debye','au')] = lambda x: x/2.541765
    # unit_conversions[('au','Debye')] = lambda x: x*2.541765
    # unit_conversions[('Debye','Cm')] = lambda x: x*3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('Cm','Debye')] = lambda x: x/3.33564e-30 # 1 Debye is 1e-18.statC.cm-1 -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('C','statC')] = lambda x: 3.33564e-10*x # 1 Couloub = sqrt(4πε0/1e9)×stat Coulomb -- WARNING: these are not dimensionally similar units!!!
    # unit_conversions[('statC','C')] = lambda x: 2997924580*x  # 1 stat Couloub = sqrt(1e9/4πε0)×Coulomb -- WARNING: these are not dimensionally similar units!!!
    ## area / cross section
    'cm2'   :  (1.        ,  'area'),
    'Mb'    :  (1e+18     ,  'area'),
}

## formula for converting between canonical units in SI units --
## pretty contextual
canonical_conversion_functions = {
    ## photon energy / frequency / wavelength / wavenumber
    ('frequency' ,'energy'    ): lambda x: x*constants.h            , 
    ('energy'    ,'frequency' ): lambda x: x/constants.h            ,
    ('wavenumber','energy'    ): lambda x: x*(constants.h*constants.c), 
    ('energy'    ,'wavenumber'): lambda x: x/(constants.h*constants.c), 
    ('length'    ,'energy'    ): lambda x: constants.h*constants.c/x, 
    ('energy'    ,'length'    ): lambda x: constants.h*constants.c/x, 
    ('wavenumber','frequency' ): lambda x: x*constants.speed_of_light,
    ('frequency' ,'wavenumber'): lambda x: x/constants.speed_of_light,
    ('length'    ,'frequency' ): lambda x: constants.speed_of_light/x,
    ('frequency' ,'wavelength'): lambda x: constants.speed_of_light/x,
    ## frequency / wavelength / wavenumber
    ('length'    ,'wavenumber'): lambda x: 1/x                       ,
    ('wavenumber','length'    ): lambda x: 1/x                       ,
}

## this is checked first, overrides all other conversion logic, some
## have extra arguments
special_case_conversion_functions = {
    ## convert between various Doppler widths
    # ('dcm-1'       ,'dnm'    ): lambda dν ,ν:  1e+7/ν**2*np.abs(dν),
    # ('dnm'         ,'dcm-1'  ): lambda dnm,nm: 1e+7/nm**2*np.abs(dnm),
    # ('dnm'         ,'dm.s-1 FWHM' ): lambda dnm,nm: constants.c*dnm/nm ,
    # ('dnm'         ,'dkm.s-1 FWHM'): lambda dnm,nm: constants.c*dnm/nm/1e3 ,
    # ('dm.s-1 FWHM' ,'dnm'    ): lambda b  ,nm: nm/constants.c*b , 
    # ('dkm.s-1 FWHM','dnm'    ): lambda b  ,nm: nm/constants.c*b*1e3 ,
    # ('dm.s-1 FHHM' ,'dcm-1'  ): lambda b  ,ν : b*ν/299792.458,
}
        
def convert(
        value,                  # value to convert
        unit_in,                # units from
        unit_out,               # units to
        *args,                  # additional other args
):       
    """Convert units."""
    ## trivial case
    if unit_in == unit_out:
        return(value)
    ## cast to array if necessary
    if not np.isscalar(value):
        value = np.asarray(value)
    ## check special_cases
    if (unit_in,unit_out) in special_case_conversion_functions:
        f = special_case_conversion_functions[unit_in,unit_out]
        return(f(value,*args))
    ## get canonical units
    assert unit_in  in canonical_factors,f'Unknown unit: {repr(unit_in)}'
    assert unit_out in canonical_factors,f'Unknown unit: {repr(unit_out)}'
    factor_in,canonical_unit_in = canonical_factors[unit_in]
    factor_out,canonical_unit_out = canonical_factors[unit_out]
    ## compute
    if canonical_unit_in == canonical_unit_out:
        return(factor_out*value/factor_in)
    else:
        f = canonical_conversion_functions[canonical_unit_in,canonical_unit_out]
        return(factor_out*f(value/factor_in))

convert_units = convert         # deprecated

def convert_difference(
        difference,             # difference to convert
        value,                  # value to convert
        unit_in,                # units from
        unit_out,               # units to
        # differentiation_step_size = 1e-10,
):
    """Convert a finite difference between units."""
    return(abs(
        +convert_units(value+difference/2.,unit_in,unit_out)
        -convert_units(value-difference/2.,unit_in,unit_out)))
        
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
    return(fv/(3.038e-6*νv *(2-my.kronecker_delta(0,Λp+Λpp)) /(2-my.kronecker_delta(0,Λpp))))

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

# def band_fvalue_to_fvalue(band_fvalue,symmetryp,branch,Jpp,symmetrypp='1σ'):
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
     
