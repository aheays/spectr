import functools
from functools import lru_cache
import warnings
from pprint import pprint


import numpy as np

from . import tools
from . import dataset
from . import kinetics
from . import convert
from .exceptions import DatabaseException,NonUniqueValueException

## module data and caches
from . import kinetics





data_directory = tools.expand_path('~/src/python/spectr/data/')

@tools.vectorise(cache=True,dtype='U30')
def normalise_species(species):
    """Try to normalise a species name."""
    retval = kinetics.get_species(species).name
    return retval

@tools.vectorise(cache=True)
def normalise_chemical_species(species):
    """Normalise species without masses."""
    retval = kinetics.get_species(species).chemical_name
    return retval
    
def normalise_electronic_state_label(label):
    """Try to normalise."""
    if len(label) == 2 and label[1] in ('p',"'"):
        label = label[0]+'′'
    if len(label) == 2 and label[1] in ('"',):
        label = label[0]+'″'
    if len(label) == 3 and label[1:2] in ('pp',"''"):
        label = label[0]+'″'
    return label

@tools.vectorise(cache=True)
def get_electronic_state_property(species,label,prop):
    """Get a quantum number defining a particular electronic state by label."""
    from .data.electronic_states import electronic_states
    ## normalise names
    species = normalise_chemical_species(species)
    label = normalise_electronic_state_label(label)
    ## look for data
    if (species,label) in electronic_states:
        data = electronic_states[(species,label)]
    else:
        raise DatabaseException(f'Cannot find electronic state: {repr((species,label))}')    
    if len(data) == 0:
        raise DatabaseException(f"Cannot find data for electronic state with {species=} {label=}")
    if prop not in data:
        raise DatabaseException(f"Cannot find property {prop=} for electronic state with {species=} {label=}")
    return data[prop]

_species_data_cache = None
@functools.lru_cache(maxsize=1024)
def get_species_data(species):
    """Get a dictionary of data for this species."""
    ## load data
    global _species_data_cache
    if _species_data_cache is None:
        _species_data_cache = dataset.load(data_directory+'/species.psv')
    data = _species_data_cache
    try:
        retval = data.unique_row(species=normalise_species(species))
    except NonUniqueValueException as err:
        raise DatabaseException(err)
    return retval

@tools.vectorise()
def get_species_property(species,prop):
    """Get a property fo this species using get_species_data. If an
    array of species then return an array of properties. Scalar output, cached."""
    data = get_species_data(species)
    if prop not in data:
        raise Exception(f'Property {repr(prop)} of species {repr(species)} not known to database.')
    retval = data[prop]
    ## test for missing data -- real value that is nan, or string that is 'nan'. Not very elegant.
    if ((np.isreal(retval) and np.isnan(retval))
        or retval=='nan'): 
        raise DatabaseException(f"Property is unknown: {repr(species)}, {repr(prop)}")
    return retval

@tools.cache
def get_level(species):
    """Load a Level object containing data about a species (all
    isotopologues)."""
    species = normalise_species(species)
    try:
        retval =  dataset.load(f'{data_directory}/levels/{species}.h5')
    except FileNotFoundError as err:
        raise DatabaseException(str(err))
    # if len(match) > 0:
        # retval = retval.matches(match)
    return retval

@tools.vectorise(cache=True,dtype=float)
def get_level_energy(species,Eref=0,**match_qn):
    """Get uniquely matching level energies."""
    species = normalise_species(species)
    level = get_level(species)
    i = tools.find(level.match(match_qn,species=species))
    if len(i) == 0:
        raise DatabaseException(f'No match found: species={repr(species)}, match_qn={repr(match_qn)}')
    if len(i) > 1:
        raise DatabaseException(f'Multiple matches found: species={repr(species)}, match_qn={repr(match_qn)}')
    if Eref == 'E0':
        E = level['E'][i][0] + level['Eref'][i][0] + level['E0'][i][0]
    else:
        E = level['E'][i][0] + level['Eref'][i][0] - Eref
    return E

    

@tools.vectorise(cache=True,dtype=float)
def get_partition_function(
        species,     # molecular species
        Tex,         # K
        Eref=0,      # Energy referenced to lowest energy level
):             
    """Get partition function."""
    import scipy
    level = get_level(species)
    kB = convert.units(scipy.constants.Boltzmann,'J','cm-1')
    Z = np.sum(level['g']*np.exp(-(level['E']+level['Eref']-Eref)/(kB*Tex)))
    return Z

@lru_cache
def get_isotopes(element_name):
    """Return an ordered list of mass numbers and fractional abundances of
    element given as a string name."""
    import  periodictable
    element = getattr(periodictable,element_name)
    isotopes = [(mass_number,element[mass_number].abundance/100.)
                for mass_number in element.isotopes]
    isotopes = tuple(sorted(isotopes,key=lambda x: -x[1]))
    return isotopes

@lru_cache
def get_most_abundant_isotope_mass_number(element_name):
    """Return mass number of the most abundance isotope of an element."""
    import periodictable
    element = getattr(periodictable,element_name)
    i = np.argmax([element[m].abundance
                   for m in element.isotopes])
    return element.isotopes[i]

@lru_cache
def get_atomic_mass(element_name,mass_number):
    """Return the atomic mass of a particular elemental isotope."""
    import periodictable
    if mass_number is None:
        ## average mass natural abundance
        return getattr(periodictable,element_name).mass
    else:
        ## mass of isotope
        return getattr(periodictable,element_name)[mass_number].mass

def get_lines(species):
    """Load spectral lines from reference data."""
    species = normalise_species(species)
    data = dataset.load(f'{data_directory}/lines/{species}.h5')
    data.name = tools.make_valid_python_symbol_name(f'lines_{species}')
    return data

@tools.cache
def get_hitran_lines(species,**match):
    """Load spectral lines from reference data."""
    from . import hitran
    return hitran.get_lines(species,**match)



