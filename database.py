##standardlibrary
import functools
from functools import lru_cache
import warnings
from pprint import pprint
from copy import copy,deepcopy
import os
import re

## non-standard library
import numpy as np

## submodules of this module
from . import tools
from .tools import vectorise
from . import dataset
from . import kinetics
from . import convert
from .exceptions import DatabaseException,NonUniqueValueException,DecodeSpeciesException
from . import kinetics

## get a dynamic absolute path to the data directory.  Requires import
## of parent module, which is a bit ugly.
import spectr

## most data stored here
data_directory = spectr.__path__[0]+'/data'

## SOLEIL spectra stored here
soleil_data_directory = '/home/heays/exp/SOLEIL' 

@tools.vectorise(cache=True,dtype='U30')
def normalise_species(species):
    """Try to normalise a species name."""
    retval = convert.species(species,'ascii_or_unicode','unicode')
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
def get_electronic_state_property(species,label,prop,encoding='ascii_or_unicode'):
    """Get a quantum number defining a particular electronic state by label."""
    from .data.electronic_states import electronic_states
    ## normalise names
    chemical_formula = get_species_property(species,'chemical_formula',encoding)
    label = normalise_electronic_state_label(label)
    ## look for data
    if (chemical_formula,label) in electronic_states:
        data = electronic_states[(chemical_formula,label)]
    else:
        raise DatabaseException(f'Cannot find electronic state: {repr((species,label))}')    
    if len(data) == 0:
        raise DatabaseException(f"Cannot find data for electronic state with {species=} {label=}")
    if prop not in data:
        raise DatabaseException(f"Cannot find property {prop=} for electronic state with {species=} {label=}")
    return data[prop]

_translate_deprecated_species_propery = {
    'chemical_species':'chemical_formula',
}

@vectorise(cache=True)
def get_species_property(species,prop,encoding='ascii_or_unicode'):
    """Get property of a chemical or isotopologue species. See data.species_data.py for property information."""
    ## translate some deprecated species properties -- delete some data (2022-01-31)
    if prop in _translate_deprecated_species_propery:
        prop = _translate_deprecated_species_propery[prop]
    ## try load property from stored database
    species_unicode = convert.species(species,encoding,'unicode')
    from .data.species_data import data as _species_data
    if (species_unicode in _species_data
        and prop in _species_data[species_unicode]):
        retval = _species_data[species_unicode][prop]
    ## otherwise try to compute
    else:
        species_tuple = convert.species(species,encoding,'tuple')
        prefix,nuclei,charge = species_tuple[0],species_tuple[1:-1],species_tuple[-1]
        if prop == 'prefix':
            retval =  prefix
        elif prop == 'nuclei':
            retval =  nuclei
        elif prop == 'charge':
            retval =  charge
        elif prop == 'formula':
            retval =  species_unicode
        elif prop == 'mass':
            retval =  np.sum([mult*get_atomic_mass(elem,mass) for mass,elem,mult in nuclei])
        elif prop == 'is_isotopologue':
            retval = False
            for mass,elem,mult in nuclei:
                if mass is not None:
                    retval = True
                    break
            retval = retval
        elif prop == 'isotopes':
            retval = []
            for mass,elem,mult in nuclei:
                if mass is None:
                    mass = get_most_abundant_isotope_mass_number(elem)
                for i in range(mult):
                    retval.append((mass,elem))
            retval = tuple(retval)
        elif prop == 'elements':
            retval = []
            for mass,elem,mult in nuclei:
                for i in range(mult):
                    retval.extend(elem)
            retval = tuple(retval)
        elif prop == 'nnuclei':
            retval = len(get_species_property(species,'elements',encoding))
        elif prop == 'nelectrons':
            retval = sum([get_atomic_number(elem)*mult for mass,elem,mult in nuclei]) - charge
        elif prop == 'reduced_mass':
            isotopes = get_species_property(species,'isotopes',encoding)
            if len(isotopes) != 2:
                raise DatabaseException(f'Can only compute reduced mass for nnuclei==2, not for {species!r}')
            m0 = get_atomic_mass(isotopes[0][1],isotopes[0][0])
            m1 = get_atomic_mass(isotopes[1][1],isotopes[1][0])
            retval = m0*m1/(m0+m1)
        elif prop == 'point_group':
            isotopes = get_species_property(species,'isotopes',encoding)
            if len(isotopes) == 1:
                ## nuclei
                retval = "K"
            elif len(isotopes) == 2:
                ## Homonumclear or heteronuclear diatomic
                if isotopes[0] == isotopes[1]:
                    retval = 'D∞h'
                else:
                    retval = 'C∞v'
            else:
                raise InferException("Can only compute reduced mass for nuclei and diatomic species.")
        elif prop == 'chemical_formula':
            ## if adjacent nuclei have different mass number how to know
            ## if they should be joined into a single tuple term? Raise an
            ## errork.
            chemical_nuclei = list(nuclei)
            for i in range(len(chemical_nuclei)):
                if (i>0
                    and nuclei[i][1] == nuclei[i-1][1]
                    and nuclei[i][0] != nuclei[i-1][0]):
                    raise Exception(f'Ambiguous isotopologue to chemical species conversion: {species!r}')
                if chemical_nuclei[i][0] is not None:
                    chemical_nuclei[i] = (None, chemical_nuclei[i][1], chemical_nuclei[i][2])
            retval = convert.species(
                (prefix,*chemical_nuclei,charge),'tuple','unicode')
        elif prop == 'isotopologue_formula':
            retval = convert.species(
                (prefix,
                 *[((get_most_abundant_isotope_mass_number(elem) if mass is None else mass),
                    elem,mul) for mass,elem,mul in nuclei] ,charge) ,'tuple','unicode')
        else:
            raise DatabaseException(f'Unknown (species,property) combination: ({species!r},{prop!r})')
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
    elif source == 'levels':
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
def get_atomic_mass(element_name,mass_number=None):
    """Return the atomic mass of a particular elemental isotope."""
    import periodictable
    if mass_number is None:
        ## average mass natural abundance
        return getattr(periodictable,element_name).mass
    else:
        ## mass of isotope
        return getattr(periodictable,element_name)[mass_number].mass

@lru_cache
def get_atomic_number(element):
    """Return the atomic mass of a particular elemental isotope."""
    import periodictable
    return getattr(periodictable,element).number

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

def load_soleil_spectrum_from_file(filename,remove_HeNe=False):
    """ Load soleil spectrum from file with given path."""
    ## load from given location, or search for unique match as the
    ## beginning of a file in the SOLEIL scan data directory
    filename = tools.expand_path(filename)
    if not os.path.exists(filename):
        filename = tools.glob_unique(
            f'{soleil_data_directory}/scans/{filename}*.h5')
    ## get header data if possible, not possible if an hdf5 file is used.
    extension = os.path.splitext(filename)[1]
    header = dict(filename=filename,header=[])
    if extension in ('.TXT','.wavenumbers'): 
        with open(filename,'r',encoding='latin-1') as fid:
            header['header'] = []
            while True:
                line = fid.readline()[:-1]
                if re.match(r'^ *[0-9.eE+-]+[, ]+[0-9.eE+-]+ *$',line): break # end of header
                header['header'].append(line) # save all lines to 'header'
                ## post-processing zero-adding leads to an
                ## interpolation of data by this factor
                r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
                if r:
                    header['interpolation_factor'] = float(r.group(1))
                ## the resolution before any interpolation
                r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
                if r: header['ds'] = float(r.group(1))
                ## NMAX parameter indicates that the spectrometer is
                ## being run at maximum resolution. This is not an
                ## even power of two. Then the spectrum is zero padded
                ## to have 2**21 points. This means that there is an
                ## additional interpolation factor of 2**21/NMAX. This
                ## will likely be non-integer.
                r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
                if r:
                    header['interpolation_factor'] *= 2**21/float(r.group(1))
                ## extract pressure from header
                r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
                if r:
                    header['date_time'] = r.group(1)
                    header['pressure_pirani'] = float(r.group(2))
                    header['pressure_baratron'] = float(r.group(3))
            header['header'] = '\n'.join(header['header'])
            ## compute instrumental resolution, FWHM
    elif extension == '.h5':
        ## newer binary format 2022-04-04
        data = tools.hdf5_to_dict(filename)
        header = data['attributes']
    elif extension == '.hdf5':
        ## older binary format 2022-04-04
        data = tools.hdf5_to_dict(filename)
        header['header'] = data['README']
        for line in header['header'].split('\n'):
            ## post-processing zero-adding leads to an
            ## interpolation of data by this factor
            r = re.match(r'^.*Interpol[^0-9]+([0-9]+).*',line)
            if r:
                header['interpolation_factor'] = float(r.group(1))
            ## the resolution before any interpolation
            r = re.match(r'^[ "#]*([0-9.]+), , ds\(cm-1\)',line)
            if r: header['ds'] = float(r.group(1))
            ## NMAX parameter -- see above
            r = re.match(r'^[ #"]*Nmax=([0-9]+)[" ]*$',line)
            if r:
                header['interpolation_factor'] *= 2**21/float(r.group(1))
            ## extract pressure from header
            r = re.match(r".*date/time 1rst scan: (.*)  Av\(Pirani\): (.*) mbar  Av\(Baratron\): (.*) mbar.*",line)
            if r:
                header['date_time'] = r.group(1)
                header['pressure_pirani'] = float(r.group(2))
                header['pressure_baratron'] = float(r.group(3))
    else:
        raise Exception(f"bad extension: {repr(extension)}")
    ## compute instrumental resolution, FWHM
    header['sinc_fwhm'] = 1.2*header['interpolation_factor']*header['ds'] 
    ## get spectrum
    if extension=='.TXT':
        x,y = [],[]
        data_started = False
        for line in tools.file_to_lines(filename,encoding='latin-1'):
            r = re.match(r'^([0-9]+),([0-9.eE+-]+)$',line) # data point line
            if r:
                data_started = True # header is passed
                x.append(float(r.group(1))),y.append(float(r.group(2))) # data point
            else:
                if data_started: break            # end of data
                else: continue                    # skip header line
        x,y = np.array(x)*header['ds'],np.array(y)
    elif extension=='.wavenumbers':
        x,y = tools.file_to_array(filename,unpack=True,comments='#',encoding='latin-1')
    elif extension == '.h5':
        ## newer binary format 2022-04-04
        # data = tools.hdf5_to_dict(filename)
        x = data['data']['ν']['value']
        y = data['data']['I']['value']
    elif extension == '.hdf5':
        ## older binary format 2022-04-04
        # data = tools.hdf5_to_dict(filename)
        x,y = data['data'].transpose()
    else:
        raise Exception(f"bad extension: {repr(extension)}")
    ## process a bit. Sort and remove HeNe line profile and jitter
    ## estimate. This is done assumign the spectrum comes
    ## first. and finding the first index were the wavenumber
    ## scale takes a backward step
    if remove_HeNe:
        i = x>31600
        x,y = x[i],y[i]
    t = tools.find(np.diff(x)<0)
    if len(t)>0:
        i = t[0]+1 
        x,y = x[:i],y[:i]
    ## get x range
    header['xmin'],header['xmax'] = x.min(),x.max()
    header['xcentre'] = 0.5*(header['xmin']+header['xmax'])
    return (x,y,header)

