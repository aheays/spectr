import functools
from functools import lru_cache
import warnings
from pprint import pprint


import numpy as np
from scipy import constants
import periodictable
from tinydb import TinyDB, Query

from . import tools
from .tools import cache,vectorise
from . import dataset
from . import kinetics
from . import convert
from .exceptions import DatabaseException

## module data and caches
from . import kinetics





data_directory = tools.expand_path('~/src/python/spectr/data/')


# global _boltzmann_population_cache
# global _boltzmann_partition_function_cache
# _electronic_state_propety_data_cache = None   # lots of data about individual electronic states
# _level_data = dict()                           # level objects indexed by tuple (species,state)
# _boltzmann_population_cache = dict()
# _boltzmann_partition_function_cache = dict()

# electronic_states = TinyDB(f'{data_directory}/electronic_states.json')



# # @cachetools.cached(cache=cachetools.LRUCache(1e3))
# @functools.lru_cache(maxsize=1024)
# def _get_electronic_state_property_scalar(species,label,prop):
    # species = get_species_property(species,'iso_indep') # currently the data should be isotopologue independent
    # ## Load data from cache if possible or from disk if necessary.
    # global _electronic_state_propety_data_cache
    # if _electronic_state_propety_data_cache is None:
        # _electronic_state_propety_data_cache = tools.file_to_recarray(data_directory+'/electronic_states.csv',table_name='linear molecules')
    # ## find result
    # retval = _electronic_state_propety_data_cache[prop][
        # (_electronic_state_propety_data_cache['species']==species)
        # &(_electronic_state_propety_data_cache['label']==label)]
    # if len(retval)==1:
        # retval = retval[0]
        # ## test for missing data -- real value that is nan, or string that is 'nan'. Not very elegant.
        # if ((np.isreal(retval) and np.isnan(retval))
            # or retval=='nan'): 
            # raise DatabaseException(f"Property is unknown: {repr(species)}, {repr(label)}, {repr(prop)}")
        # return(retval)
    # elif len(retval)==0:
        # raise DatabaseException('No match found for species and label: '+repr(species)+' '+repr(label))
    # elif len(retval)>1:
        # raise Exception('Non-unique matches found for species and label: '+repr(species)+' '+repr(label))


@tools.vectorise(cache=True)
def normalise_species(species):
    """Try to normalise a species name."""
    return kinetics.get_species(species).name

@tools.vectorise(cache=True)
def normalise_chemical_species(species):
    """Normalise species without masses."""
    return kinetics.get_species(species).chemical_name
    
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

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@functools.lru_cache(maxsize=1024)
def get_species_data(species):
    """Get a dictionary of data for this species."""
    species = normalise_species(species)
    data = tools.file_to_recarray(data_directory+'/species.csv',table_name='data')
    if species not in data['species']:
        raise DatabaseException(f"Species unknown: {repr(species)}")
    return data[data['species']==species] 

# species_property_dtypes = {'species':'U50','iso_indep':float,'mass':float,
                           # 'reduced_mass':float,'group':'U5','Ihomo':float,'latex':'U50',
                           # 'T0-Te':float,}

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@tools.vectorise()
def get_species_property(species,prop):
    """Get a property fo this species using get_species_data. If an
    array of species then return an array of properties. Scalar output, cached."""
    d = get_species_data(species)
    assert prop in d.dtype.names,f'Property {repr(prop)} of species {repr(species)} not known to database.'
    retval = d[prop][0]
    ## test for missing data -- real value that is nan, or string that is 'nan'. Not very elegant.
    if ((np.isreal(retval) and np.isnan(retval))
        or retval=='nan'): 
        raise DatabaseException(f"Property is unknown: {repr(species)}, {repr(prop)}")
    return retval

@cache
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

@tools.vectorise(cache=True)
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

    

@tools.vectorise(cache=True)
def get_partition_function(
        species,     # molecular species
        Tex,         # K
        Eref=0,      # Energy referenced to lowest energy level
):             
    """Get partition function."""
    level = get_level(species)
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    Z = np.sum(level['g']*np.exp(-(level['E']+level['Eref']-Eref)/(kB*Tex)))
    return Z


@lru_cache
def get_isotopes(element_name):
    """Return an ordered list of mass numbers and fractional abundances of
    element given as a string name."""
    element = getattr(periodictable,element_name)
    isotopes = [(mass_number,element[mass_number].abundance/100.)
                for mass_number in element.isotopes]
    isotopes = tuple(sorted(isotopes,key=lambda x: -x[1]))
    return isotopes

@lru_cache
def get_most_abundant_isotope_mass_number(element_name):
    """Return mass number of the most abundance isotope of an element."""
    element = getattr(periodictable,element_name)
    i = np.argmax([element[m].abundance
                   for m in element.isotopes])
    return element.isotopes[i]

@lru_cache
def get_atomic_mass(element_name,mass_number):
    """Return the atomic mass of a particular elemental isotope."""
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

electronic_states={
    ('C₂','X')  :{'Λ':0,'S':0,'s'  :1,'gu'    :1},
    ('C₂','a')  :{'Λ':1,'S':1,'s'  :0,'gu'    :-1},
    ('C₂','b')  :{'Λ':0,'S':1,'s'  :1,'gu'    :1},
    ('C₂','A')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1},
    ('C₂','c')  :{'Λ':0,'S':1,'s'  :0,'gu'    :-1},
    ('C₂','B')  :{'Λ':2,'S':0,'s'  :0,'gu'    :1},
    ('C₂','d')  :{'Λ':1,'S':1,'s'  :0,'gu'    :1},
    ('C₂','C')  :{'Λ':1,'S':0,'s'  :0,'gu'    :1},
    ('C₂','e')  :{'Λ':1,'S':1,'s'  :0,'gu'    :1},
    ('C₂','D')  :{'Λ':0,'S':0,'s'  :0,'gu'    :-1},
    ('C₂','E')  :{'Λ':0,'S':0,'s'  :0,'gu'    :1},
    ('C₂','f')  :{'Λ':0,'S':1,'s'  :1,'gu'    :1},
    ('C₂','g')  :{'Λ':2,'S':1,'s'  :0,'gu'    :1},
    ('C₂','F')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1},
    ('C₂','A′') :{'Λ':0,'S':1,'s'  :1,'gu'    :1},
    ('C₂','B′') :{'Λ':0,'S':0,'s'  :0,'gu'    :1},
    ('C₂','C′') :{'Λ':1,'S':0,'s'  :0,'gu'    :1},
    ('CN','X')  :{'Λ':0,'S':1,'s'  :0,'LSsign':1},
    ('CN','A')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('CN','B')  :{'Λ':0,'S':1,'s'  :0,'LSsign':1},
    ('CN','D')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('CN','E')  :{'Λ':0,'S':1,'s'  :0,'LSsign':1},
    ('CO','A')  :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('CO','B')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','C')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','D')  :{'Λ':2,'S':0,'s'  :0,'LSsign':1},
    ('CO','E')  :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('CO','F')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','I')  :{'Λ':0,'S':0,'s'  :1,'LSsign':1},
    ('CO','J')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','K')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','L')  :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('CO','L′') :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('CO','W')  :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('CO','X')  :{'Λ':0,'S':0,'s'  :0,'LSsign':1},
    ('CO','a′') :{'Λ':0,'S':1,'s'  :0,'LSsign':-1},
    ('CO','a')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('CO','d')  :{'Λ':2,'S':1,'s'  :0,'LSsign':-1},
    ('CO','e')  :{'Λ':0,'S':1,'s'  :1,'LSsign':-1},
    ('CO','k')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('H₂','X')  : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : 1,'LSsign'  : 1},
    ('H₂','GK') : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : 1,'LSsign'  : 1},
    ('H₂','H')  : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : 1,'LSsign'  : 1},
    ('H₂','B')  : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : -1,'LSsign' : 1},
    ('H₂','C')  : {'Λ' : 1,'S' : 0,'s' : 0,'gu' : -1,'LSsign' : 1},
    ('H₂','B′') : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : -1,'LSsign' : 1},
    ('H₂','D')  : {'Λ' : 1,'S' : 0,'s' : 0,'gu' : -1,'LSsign' : 1},
    ('H₂','I')  : {'Λ' : 1,'S' : 0,'s' : 0,'gu' : +1,'LSsign' : 1},
    ('H₂','B″') : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : -1,'LSsign' : 1},
    ('H₂','J')  : {'Λ' : 2,'S' : 0,'s' : 0,'gu' : +1,'LSsign' : 1},
    ('H₂','b')  : {'Λ' : 2,'S' : 1,'s' : 0,'gu' : +1,'LSsign' : 1},
    ('H₂','c')  : {'Λ' : 1,'S' : 1,'s' : 0,'gu' : +1,'LSsign' : 1},
    ('H₂','EF') : {'Λ' : 0,'S' : 0,'s' : 0,'gu' : +1,'LSsign' : 1},
    ('OH','X')  :{'Λ':1,'S':0.5,'s':0,'LSsign':-1},
    ('OH','A')  :{'Λ':0,'S':0.5,'s':0,'LSsign':1},
    ('OH','B')  :{'Λ':0,'S':0.5,'s':0,'LSsign':1},
    ('OH','C')  :{'Λ':0,'S':0.5,'s':0,'LSsign':1},
    ('OH','D')  :{'Λ':0,'S':0.5,'s':1,'LSsign':1},
    ('NO','X')  :{'Λ':1,'S':0.5,'s':0,'LSsign':1},
    ('NO','A')  :{'Λ':0,'S':0.5,'s':0,'LSsign':1},
    ('NO','B')  :{'Λ':1,'S':0.5,'s':0,'LSsign':1},
    ('NO','C')  :{'Λ':1,'S':0.5,'s':0,'LSsign':1},
    ('NO','D')  :{'Λ':0,'S':0.5,'s':0,'LSsign':1},
    ('NO','B′') :{'Λ':2,'S':0.5,'s':0,'LSsign':1},
    ('NO','F')  :{'Λ':2,'S':0.5,'s':0,'LSsign':1},
    ('N₂','X')  :{'Λ':0,'S':0,'s'  :0,'gu'    :1,'LSsign' :1},
    ('N₂','A')  :{'Λ':0,'S':1,'s'  :0,'gu'    :-1,'LSsign':-1},
    ('N₂','B')  :{'Λ':1,'S':1,'s'  :0,'gu'    :1,'LSsign' :1},
    ('N₂','W')  :{'Λ':2,'S':1,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','B′') :{'Λ':1,'S':1,'s'  :1,'gu'    :-1,'LSsign':1},
    ('N₂','b')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c3') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c4') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c5') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c6') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e3') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e4') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e5') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e6') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','o')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','o3') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','o4') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','o5') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','o6') :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','b′') :{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′') :{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′4'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′5'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′6'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′7'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′8'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','c′9'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′') :{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′4'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′5'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′6'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′7'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′8'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','e′9'):{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','C')  :{'Λ':1,'S':1,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','F')  :{'Λ':1,'S':1,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','G')  :{'Λ':1,'S':1,'s'  :0,'gu'    :-1,'LSsign':-1},
    ('N₂','C′') :{'Λ':1,'S':1,'s'  :0,'gu'    :-1,'LSsign':1},
    ('N₂','D')  :{'Λ':0,'S':1,'s'  :0,'gu'    :-1,'LSsign':-1,'comment':'correctLSsign?'},
    ('SO','X')  :{'Λ':0,'S':1,'s'  :1,'LSsign':1},
    ('SO','A')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('SO','B')  :{'Λ':0,'S':1,'s'  :1,'LSsign':1},
    ('SO','C')  :{'Λ':1,'S':1,'s'  :0,'LSsign':-1},
    ('SO','d')  :{'Λ':1,'S':0,'s'  :0,'LSsign':1},
    ('SO','A″'):{'Λ':0,'S':1,'s'  :1,'LSsign':-1},
    ('SO','a')  :{'Λ':2,'S':0,'s'  :0,'LSsign':1},
    ('SO','f')  :{'Λ':2,'S':0,'s'  :0,'LSsign':1},
    ('S₂','X')  :{'Λ':0,'S':1,'s'  :1,'gu'    :1,'LSsign' :1},
    ('S₂','a')  :{'Λ':2,'S':0,'s'  :0,'gu'    :1,'LSsign' :1},
    ('S₂','b')  :{'Λ':0,'S':0,'s'  :0,'gu'    :1,'LSsign' :1},
    ('S₂','B')  :{'Λ':0,'S':1,'s'  :1,'gu'    :-1,'LSsign':1},
    ('S₂','B″'):{'Λ':1,'S':1,'s'  :0,'gu'    :-1,'LSsign':1},
    ('S₂','f')  :{'Λ':2,'S':0,'s'  :0,'gu'    :-1,'LSsign':1}
}


