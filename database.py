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
    # if species in species_synonyms:
        # species = species_synonyms[species]
    # return species
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
        return dataset.load(f'{data_directory}/levels/{species}.h5')
    except FileNotFoundError as err:
        raise DatabaseException(str(err))

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

# def get_term_values(
        # species,
        # label,                  # compulsory for caching
        # Tref=None,              # return values relative to this amount relative to equilbrium energy of, or "T0-Te", if None use whatever is in database
        # **quantum_numbers,      # limit to these
# ):
    # """Load term value data. Returns as array. If a quantum number is
    # iterable then return as array of values. Always returns a 1D
    # array."""
    # ## check if their is an interable quantum number (one only)
    # iterable_quantum_numbers = {}
    # scalar_quantum_numbers = {}
    # for key,val in quantum_numbers.items():
        # if tools.is_iterable_not_string(val):
            # iterable_quantum_numbers[key] = val
        # else:
            # scalar_quantum_numbers[key] = val
    # assert len(iterable_quantum_numbers)<=1,"Only one iterable key permitted."
    # ## load data and reduce by scalar quantum numbers
    # level = get_level(species,label)
    # # if len(scalar_quantum_numbers)>0:
    # #     # level = level.matches(**scalar_quantum_numbers)
    # #     iscalar = level.match(**scalar_quantum_numbers)
    # # T = level['T'][i]
    # # ## if there is an iterable quantum number reduce to matching
    # # ## values -- in this order -- must be unique
    # # if len(iterable_quantum_numbers)>0:
    # #     key = [t for t in iterable_quantum_numbers.keys()][0]
    # #     val = [t for t in iterable_quantum_numbers.values()][0]
    # # try:
    # #     T = T[tools.findin(val,level[key])]
    # # # except Exception as err:
    # #     # raise DatabaseException(str(err))
    # # # ## set requested Tref
    # # # if Tref is None:
    # #     # pass
    # # # elif Tref=='T0-Te':
    # #     # T += level['Tref'] - get_species_property(species,'T0-Te')
    # # # else:
    # #     # T += level['Tref']-Tref
    # # # return(T)
    # i = level.match(**quantum_numbers)
    # T = level['T'][i]
    # # ## if there is an iterable quantum number reduce to matching
    # # ## values -- in this order -- must be unique
    # # if len(iterable_quantum_numbers)>0:
        # # key = [t for t in iterable_quantum_numbers.keys()][0]
        # # val = [t for t in iterable_quantum_numbers.values()][0]
    # # try:
        # # T = T[tools.findin(val,level[key])]
    # # # except Exception as err:
        # # # raise DatabaseException(str(err))
    # # # ## set requested Tref
    # if Tref is None:
        # pass
    # elif Tref=='T0-Te':
        # T += level['Tref'] - get_species_property(species,'T0-Te')
    # else:
        # T += level['Tref']-Tref
    # return(T)
    # return(1.0)


# def get_term_value(*args,**quantum_numbers):
    # """Load a strictly scalar term value."""
    # retval = get_term_values(*args,**quantum_numbers)
    # if len(retval)==0:
        # raise Exception(f"No term value found for: {repr(args)}, {repr(quantum_numbers)}")
    # elif len(retval)>1:
        # raise Exception(f"Multiple ({len(retval)}) term values found for: {repr(args)}, {repr(quantum_numbers)}")
    # return(float(retval))


# # @cachetools.cached(cache=cachetools.LRUCache(1e3))
# def get_boltzmann_population(temperature,species,J,**quantum_numbers):
    # """Load term value data. All quantum numbers must be scalar. If J
    # is vector then return array, else return a scalar float."""
    # partition_function = get_partition_function(species,temperature)
    # level = get_level(species,J=J,**quantum_numbers)
    # population = level['g']*np.exp(-1.4387751297850826/temperature*level['T'])/partition_function
        # # raise DatabaseException(str(err))
    # # if np.isscalar(J):  return(float(population[i][j]))
    # # else:               return(population[i][j])
    # return(population)



# def decode_species(species,raise_error_if_unrecognised=False):
    # """Some species have abbreviations, they are corrected here to a
    # standard form."""
    # if   species in ['14',14,'14N2','N2']:   return '14N2'
    # elif species in ['1415',1415,'14N15N']:  return '14N15N'
    # elif species in ['15',15,'15N2']:        return '15N2'
    # elif species in ['12C16O',1216,'1216']:  return '12C16O'
    # elif species in ['12C17O',1217,'1217']:  return '12C17O'
    # elif species in ['12C18O',1218,'1218']:  return '12C18O'
    # elif species in ['13C16O',1316,'1316']:  return '13C16O'
    # elif species in ['13C17O',1317,'1317']:  return '13C17O'
    # elif species in ['13C18O',1318,'1318']:  return '13C18O'
    # elif species in ['14C16O',1416,'1416']:  return '14C16O'
    # elif species in ['14C17O',1417,'1417']:  return '14C17O'
    # elif species in ['14C18O',1418,'1418']:  return '14C18O'
    # elif species in ['NI','N I','N']:  return 'N'
    # elif species in ['Ar I','Ar','Argon','argon']:  return 'Ar'
    # elif species in ['Xe I','Xe','Xenon','xenon']:  return 'Xe'
    # elif species in ['H2',]:     return 'H2'
    # elif species in ['D2',]:     return 'D2'
    # elif species in ['HD',]:     return 'HD'
    # elif species in ['NO','nitric oxide','nitrogen monoxide']:     return 'NO'
    # elif species in ['OH','hydroxyl radical']:     return 'OH'
    # elif species in ['OD','isotopic hydroxyl radical']:     return 'OD'
    # else:
        # if raise_error_if_unrecognised:
            # raise Exception("Unrecognised species: "+str(species))
        # else:
            # return(species)       # if cannot decode return as is

# # @cachetools.cached(cache=cachetools.LRUCache(1e2))
# @functools.lru_cache
# def get_spectral_contaminant_linelist(*species,νbeg=None,νend=None):
    # """Load contaminant data and return.  Keys reference to data files in
    # data_dir/spectral_contaminants. Default species is a mixture. Could add more H2 and D2 from
    # ~/data/species/H2/lines_levels/meudon_observatory/all_emission_lines
    # ~/data/species/H2/lines_levels/meudon_D2/all_lines """
    # ## default mix
    # if len(species)==0:
        # species = ('default_mix',)
    # ## load all files
    # linelist = Dynamic_Recarray(
        # load_from_filename=[
            # f'{data_directory}/spectral_contaminants/{speciesi}'
            # for speciesi in species])
    # ## filter
    # if νbeg is not None:
        # linelist.remove(linelist['ν']<νbeg)
    # if νend is not None:
        # linelist.remove(linelist['ν']>νend)
    # return(linelist)

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
def get_atomic_mass(element_name,mass_number):
    """Return the atomic mass of a particular elemental isotope."""
    if mass_number is None:
        ## average mass natural abundance
        return getattr(periodictable,element_name).mass
    else:
        ## mass of isotope
        return getattr(periodictable,element_name)[mass_number].mass

def load_lines(species):
    """Load spectral lines from reference data."""
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
    ('CO','e')  :{'Λ':0,'S':1,'s'  :1,'LSsign':1},
    ('CO','k')  :{'Λ':1,'S':1,'s'  :0,'LSsign':1},
    ('H₂','X')  :{'Λ':0,'S':0,'s'  :0,'gu'    :1,'LSsign' :1},
    ('H₂','B')  :{'Λ':0,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
    ('H₂','C')  :{'Λ':1,'S':0,'s'  :0,'gu'    :-1,'LSsign':1},
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


species_synonyms = {
    'HH':'H2',
    '[1H][1H]':'H2',
    }
