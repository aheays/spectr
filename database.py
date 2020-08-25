import functools
import numpy as np
from scipy import constants

from . import tools
from .exceptions import MissingDataException

## module data and caches
from .species import get_species

data_directory = '~/data/reference_data/' # root directory where data is stored in various ways
global _level_data
global _boltzmann_population_cache
global _boltzmann_partition_function_cache
_electronic_state_propety_data_cache = None   # lots of data about individual electronic states
_level_data = dict()                           # level objects indexed by tuple (species,state)
_boltzmann_population_cache = dict()
_boltzmann_partition_function_cache = dict()

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@functools.lru_cache(maxsize=1024)
def _get_electronic_state_property_scalar(species,label,prop):
    species = get_species_property(species,'iso_indep') # currently the data should be isotopologue independent
    ## Load data from cache if possible or from disk if necessary.
    global _electronic_state_propety_data_cache
    if _electronic_state_propety_data_cache is None:
        _electronic_state_propety_data_cache = tools.file_to_recarray(data_directory+'/electronic_states.csv',table_name='linear molecules')
    ## find result
    retval = _electronic_state_propety_data_cache[prop][
        (_electronic_state_propety_data_cache['species']==species)
        &(_electronic_state_propety_data_cache['label']==label)]
    if len(retval)==1:
        retval = retval[0]
        ## test for missing data -- real value that is nan, or string that is 'nan'. Not very elegant.
        if ((np.isreal(retval) and np.isnan(retval))
            or retval=='nan'): 
            raise MissingDataException(f"Property is unknown: {repr(species)}, {repr(label)}, {repr(prop)}")
        return(retval)
    elif len(retval)==0:
        raise MissingDataException('No match found for species and label: '+repr(species)+' '+repr(label))
    elif len(retval)>1:
        raise Exception('Non-unique matches found for species and label: '+repr(species)+' '+repr(label))

def get_electronic_state_property(species,label,prop):
    """Get a quantum number defining a particular electronic state by label."""
    ## one value required
    if np.isscalar(species) and np.isscalar(label):
        return(_get_electronic_state_property_scalar(species,label,prop))
    ## species or lable are a list, or both are lists of the same length
    else:
        assert len(species)==len(label),'one scalar one not not implemented'
        species,label = np.array(species),np.array(label)
        retval = None
        for speciesi in np.unique(species): 
            i = tools.find(species==speciesi)
            for labeli in np.unique(label[i]):
                j = tools.find(label[i]==labeli)
                t = get_electronic_state_property(speciesi,labeli,prop)
                if retval is None: retval = np.empty(species.shape,dtype=type(t)) # create return array
                retval[i[j]] = t # add to return array
        return(retval)
get_electronic_state_quantum_number = get_electronic_state_property # deprecated name

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@functools.lru_cache(maxsize=1024)
def get_species_data(species):
    """Get a dictionary of data for this species."""
    data = tools.file_to_recarray(data_directory+'/species.csv',table_name='data')
    if species not in data['species']:
        raise MissingDataException(f"Species unknown: {repr(species)}")
    return(data[data['species']==species])

species_property_dtypes = {'species':'U50','iso_indep':float,'mass':float,
                           'reduced_mass':float,'group':'U5','Ihomo':float,'latex':'U50',
                           'T0-Te':float,}

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@functools.lru_cache(maxsize=1024)
def _get_species_property_scalar(species,prop):
    """Get a property fo this species using get_species_data. If an
    array of species then return an array of properties. Scalar output, cached."""
    translate_prop = {'μ':'reduced_mass'}
    if prop in translate_prop:
        prop = translate_prop[prop]
    d = get_species_data(species)
    assert prop in d.dtype.names,f'Property {repr(prop)} of species {repr(species)} not known to database.'
    retval = d[prop][0]
    ## test for missing data -- real value that is nan, or string that is 'nan'. Not very elegant.
    if ((np.isreal(retval) and np.isnan(retval))
        or retval=='nan'): 
        raise MissingDataException(f"Property is unknown: {repr(species)}, {repr(prop)}")
    return(retval)

def get_species_property(species,prop):
    """Get a property fo this species using get_species_data. If an array
    of species then return an array of properties. Can be vector. """
    if np.isscalar(species):
        return(_get_species_property_scalar(species,prop))
    else:
        retval = np.ones(species.shape,dtype=species_property_dtypes[prop])
        for speciesi in np.unique(species):
            retval[species==speciesi] = _get_species_property_scalar(speciesi,prop)
        return(retval)

## deprecate this in favour of get_species_property?

# @functools.lru_cache
@tools.vectorise_in_chunks
def get_mass(species):
    t = get_species(species)
    return(t.mass)

## deprecate this in favour of get_species_property?
def get_reduced_mass(species):
    raise Exception('use get_species_property(species,prop):')
    return(float(get_species_data(species)['reduced_mass']))

## deprecate this in favour of get_species_property?
def get_isotopologue_independent_species(species):
    """Get the isotopologue-free name for a species, e.g, CO from
    13C17O."""
    raise Exception('use get_species_property(species,prop):')
    return(get_species_data(species)['iso_indep'][0])

def get_level(species,label,Tref=0,**other_quantum_numbers):
    """Load a Level object containing data about an electronic
    level. This will be filtered by other_quantum_numbers if
    provided. E.g., species="12C16O", label='X'. NOTE THAT THIS IS
    CACHED AND MUTABLE. IF YOU CHANGE IT WITHOUT A DEEPCOPY, THINGS
    WILL BREAK."""
    import spectra
    ## get data from cache if already there
    global _level_data
    if (species,label) not in _level_data:
        filename = data_directory+'/levels/'+species+'_'+label
        try:
            level = spectra.load_level(filename)
        except FileNotFoundError:
            raise MissingDataException("Could not find data for "+repr(species)+" "+repr(label)+" in file: "+repr(filename))
        _level_data[(species,label)] = level
    else:
        level = _level_data[(species,label)]
    ## filter further if requested
    if len(other_quantum_numbers)!=0:
        level = level.matches(**other_quantum_numbers)
    ## set Tref
    if Tref!=level['Tref']:
        level.set_Tref(Tref)
    return(level)

def get_term_values(
        species,
        label,                  # compulsory for caching
        Tref=None,              # return values relative to this amount relative to equilbrium energy of, or "T0-Te", if None use whatever is in database
        **quantum_numbers,      # limit to these
):
    """Load term value data. Returns as array. If a quantum number is
    iterable then return as array of values. Always returns a 1D
    array."""
    ## check if their is an interable quantum number (one only)
    iterable_quantum_numbers = {}
    scalar_quantum_numbers = {}
    for key,val in quantum_numbers.items():
        if tools.is_iterable_not_string(val):
            iterable_quantum_numbers[key] = val
        else:
            scalar_quantum_numbers[key] = val
    assert len(iterable_quantum_numbers)<=1,"Only one iterable key permitted."
    ## load data and reduce by scalar quantum numbers
    level = get_level(species,label)
    # if len(scalar_quantum_numbers)>0:
    #     # level = level.matches(**scalar_quantum_numbers)
    #     iscalar = level.match(**scalar_quantum_numbers)
    # T = level['T'][i]
    # ## if there is an iterable quantum number reduce to matching
    # ## values -- in this order -- must be unique
    # if len(iterable_quantum_numbers)>0:
    #     key = [t for t in iterable_quantum_numbers.keys()][0]
    #     val = [t for t in iterable_quantum_numbers.values()][0]
    # try:
    #     T = T[tools.findin(val,level[key])]
    # # except Exception as err:
    #     # raise MissingDataException(str(err))
    # # ## set requested Tref
    # # if Tref is None:
    #     # pass
    # # elif Tref=='T0-Te':
    #     # T += level['Tref'] - get_species_property(species,'T0-Te')
    # # else:
    #     # T += level['Tref']-Tref
    # # return(T)
    i = level.match(**quantum_numbers)
    T = level['T'][i]
    # ## if there is an iterable quantum number reduce to matching
    # ## values -- in this order -- must be unique
    # if len(iterable_quantum_numbers)>0:
        # key = [t for t in iterable_quantum_numbers.keys()][0]
        # val = [t for t in iterable_quantum_numbers.values()][0]
    # try:
        # T = T[tools.findin(val,level[key])]
    # # except Exception as err:
        # # raise MissingDataException(str(err))
    # # ## set requested Tref
    if Tref is None:
        pass
    elif Tref=='T0-Te':
        T += level['Tref'] - get_species_property(species,'T0-Te')
    else:
        T += level['Tref']-Tref
    return(T)
    return(1.0)


def get_term_value(*args,**quantum_numbers):
    """Load a strictly scalar term value."""
    retval = get_term_values(*args,**quantum_numbers)
    if len(retval)==0:
        raise Exception(f"No term value found for: {repr(args)}, {repr(quantum_numbers)}")
    elif len(retval)>1:
        raise Exception(f"Multiple ({len(retval)}) term values found for: {repr(args)}, {repr(quantum_numbers)}")
    return(float(retval))

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
@functools.lru_cache(maxsize=1024)
def get_partition_function(
        species,                # molecular species
        temperature,            # K
        Tref=None,              # Return a partition function with this energy reference, itself referenced to the equilibrium geometry energy, Te. If None then use whatever is in the databse
):              # Energy reference relative to equilibrium energy
    """Get partition function."""
    level = get_level(species,'X')
    ## no Tref -- no problems, cowboy away
    if Tref is None:
        Tref = database_Tref = 0.
    else:
        database_Tref = level['Tref']
    partition_function = np.sum(level['g']*np.exp(-(level['T']+database_Tref-Tref)/(tools.J2k(constants.Boltzmann)*temperature)))
    return(partition_function)

# @cachetools.cached(cache=cachetools.LRUCache(1e3))
def get_boltzmann_population(temperature,species,J,**quantum_numbers):
    """Load term value data. All quantum numbers must be scalar. If J
    is vector then return array, else return a scalar float."""
    partition_function = get_partition_function(species,temperature)
    level = get_level(species,J=J,**quantum_numbers)
    population = level['g']*np.exp(-1.4387751297850826/temperature*level['T'])/partition_function
        # raise MissingDataException(str(err))
    # if np.isscalar(J):  return(float(population[i][j]))
    # else:               return(population[i][j])
    return(population)


def get_doppler_width(
        temperature,            # temperature in K
        mass_or_species,        # name of a species or its mass in amu
        ν,             # wavenumber in cm-1
        units='cm-1.FWHM',  # Units of output widths.
):
    """Calculate Doppler width given temperature and species mass."""
    if isinstance(mass_or_species,str):
        mass_or_species = get_species_property(mass_or_species,'mass')
    dk = 2.*6.331e-8*np.sqrt(temperature*32./mass_or_species)*ν
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

def decode_species(species,raise_error_if_unrecognised=False):
    """Some species have abbreviations, they are corrected here to a
    standard form."""
    if   species in ['14',14,'14N2','N2']:   return '14N2'
    elif species in ['1415',1415,'14N15N']:  return '14N15N'
    elif species in ['15',15,'15N2']:        return '15N2'
    elif species in ['12C16O',1216,'1216']:  return '12C16O'
    elif species in ['12C17O',1217,'1217']:  return '12C17O'
    elif species in ['12C18O',1218,'1218']:  return '12C18O'
    elif species in ['13C16O',1316,'1316']:  return '13C16O'
    elif species in ['13C17O',1317,'1317']:  return '13C17O'
    elif species in ['13C18O',1318,'1318']:  return '13C18O'
    elif species in ['14C16O',1416,'1416']:  return '14C16O'
    elif species in ['14C17O',1417,'1417']:  return '14C17O'
    elif species in ['14C18O',1418,'1418']:  return '14C18O'
    elif species in ['NI','N I','N']:  return 'N'
    elif species in ['Ar I','Ar','Argon','argon']:  return 'Ar'
    elif species in ['Xe I','Xe','Xenon','xenon']:  return 'Xe'
    elif species in ['H2',]:     return 'H2'
    elif species in ['D2',]:     return 'D2'
    elif species in ['HD',]:     return 'HD'
    elif species in ['NO','nitric oxide','nitrogen monoxide']:     return 'NO'
    elif species in ['OH','hydroxyl radical']:     return 'OH'
    elif species in ['OD','isotopic hydroxyl radical']:     return 'OD'
    else:
        if raise_error_if_unrecognised:
            raise Exception("Unrecognised species: "+str(species))
        else:
            return(species)       # if cannot decode return as is

# @cachetools.cached(cache=cachetools.LRUCache(1e2))
@functools.lru_cache
def get_spectral_contaminant_linelist(*species,νbeg=None,νend=None):
    """Load contaminant data and return.  Keys reference to data files in
    data_dir/spectral_contaminants. Default species is a mixture. Could add more H2 and D2 from
    ~/data/species/H2/lines_levels/meudon_observatory/all_emission_lines
    ~/data/species/H2/lines_levels/meudon_D2/all_lines """
    ## default mix
    if len(species)==0:
        species = ('default_mix',)
    ## load all files
    linelist = Dynamic_Recarray(
        load_from_filename=[
            f'{data_directory}/spectral_contaminants/{speciesi}'
            for speciesi in species])
    ## filter
    if νbeg is not None:
        linelist.remove(linelist['ν']<νbeg)
    if νend is not None:
        linelist.remove(linelist['ν']>νend)
    return(linelist)




