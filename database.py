## standard library
import functools
from functools import lru_cache
import warnings
from pprint import pprint
from copy import copy,deepcopy

## non-standard library
import numpy as np

## submodules of this module
from . import tools
from . import dataset
from . import kinetics
from . import convert
from .exceptions import DatabaseException,NonUniqueValueException,DecodeSpeciesException
from . import kinetics

## get a dynamic absolute path to the data directory.  Requires import
## of parent module, which is a bit ugly.
import spectr
data_directory = spectr.__path__[0]+'/data'

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

@functools.lru_cache(maxsize=1024)
def get_species_data(species):
    """Get a dictionary of data for this species. Data stored in data/species_data.py."""
    from .data.species_data import data as species_data
    retval = species_data[species]
    return retval

# @tools.vectorise()
# def get_species_property(species,prop):
#     """Get a database property of this species. Data stored in data/species_data.py."""
#     from .data.species_data import data as species_data
#     species = normalise_species(species)
#     if species not in species_data or prop not in species_data[species]:
#         raise DatabaseException(f"Species property is unknown: {species=}, {prop=}")
#     retval = species_data[species][prop]
#     return retval

@tools.vectorise()
def get_species_property(species,prop):
    """Get a database property of this species. Data stored in data/species_data.py."""
    from .data.species_data import data as species_data
    species = normalise_species(species)
    if species in species_data and prop in species_data[species]:
        retval = species_data[species][prop]
    else:
        try: 
            retval = kinetics.get_species_property(species,prop)
        except DecodeSpeciesException:
            raise DatabaseException(f"Species property is unknown: {species=}, {prop=}")
    return retval

@tools.cache
def _get_level_internal(species):
    """Cached load of level from data/levels"""
    retval =  dataset.load(f'{data_directory}/levels/{species}.h5')
    return retval

def get_level(species,source='auto'):
    """Load a Level object containing data about a species (all
    isotopologues)."""
    species = normalise_species(species)
    if source == 'auto':
        ## try multiple sources
        for source in ('levels','hitran'):
            try:
                retval = get_level(species,source)
                break
            except DatabaseException:
                pass
        else:
            raise DatabaseException(f'Could not find a source for levels of {species!r}')
    if source == 'levels':
        ## my collected data
        try:
            retval =  _get_level_internal(species).copy()
        except FileNotFoundError as err:
            raise DatabaseException(str(err))
    elif source == 'hitran':
        ## hitran
        from . import hitran
        retval =  hitran.get_level(species)
    else:
        raise Exception(f'Unknown source: {source!r}')
    return retval

@tools.vectorise(cache=True,dtype=float)
def get_level_energy(species,Eref=0,**match_qn):
    """Get uniquely matching level energies."""
    ## get level
    species = normalise_species(species)
    level = get_level(species)
    try:
        i = level.find_unique(match_qn,species=species)
    except NonUniqueValueException as err:
        i = level.find(match_qn,species=species)
        raise DatabaseException(f'Non-unique level energy, found {len(i)}')
    ## get energy with correct reference
    if Eref == 'E0':
        E = level['E'][i] + level['Eref'][i] + level['E0'][i]
    else:
        E = level['E'][i] + level['Eref'][i] - Eref
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
    data.name = tools.regularise_symbol(f'lines_{species}')
    return data

@tools.cache
def get_hitran_lines(species,**match):
    """Load spectral lines from reference data."""
    from . import hitran
    return hitran.get_lines(species,**match)
