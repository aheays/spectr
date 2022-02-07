## standard library
import functools
from copy import deepcopy
import os,shutil

## nonstandard library
import numpy as np

## internal imports
from . import lines
from . import plotting
from . import tools
from .tools import *
from . import kinetics
from . import dataset
from . import database
from .database import get_species_property
from . import quantum_numbers
from .dataset import Dataset
from. exceptions import DatabaseException

def _import_hapi(silence=True):
    """Import without accrediation message -- bad idea?"""
    from hapi import hapi
    return hapi

@tools.vectorise(cache=True,dtype=float)
def get_partition_function(species,temperature):
    """Use hapi to get a partition function.  Uses main isotopologue if
    not given."""
    hapi = _import_hapi()
    Mol,Iso = translate_species_to_codes(species)
    return hapi.partitionSum(Mol,Iso,temperature)

@tools.cache
def get_natural_abundance(species):
    """Get isotopologue natural abundance"""
    species = database.normalise_species(species)
    data = get_molparam()
    row = data.unique_row(species=species)
    return row['natural_abundance']

@tools.cache
def is_known_species(species):
    """Get isotopologue natural abundance"""
    species = database.normalise_species(species)
    data = get_molparam()
    return species in data['species']

@tools.cache
def is_known_chemical_species(chemical_species):
    """Get isotopologue natural abundance"""
    chemical_species = database.normalise_species(chemical_species)
    data = get_molparam()
    return chemical_species in data['chemical_species']

@tools.cache
def get_isotopologues(chemical_species):
    """Get isotopologue natural abundance"""
    chemical_species = database.normalise_species(chemical_species)
    if not is_known_chemical_species(chemical_species):
        raise Exception(f'chemical_species unknown: {chemical_species!r}')
    data = get_molparam()
    return data.get('species','value',chemical_species=chemical_species)

_molparam = None
def get_molparam(**match_keys_vals):
    """Get molparam Dataset."""
    global _molparam
    ## load data if needed
    if _molparam is None:
        _molparam = dataset.load(f'{database.data_directory}/hitran/molparam.psv')
    retval = _molparam.matches(**match_keys_vals)
    assert len(retval)>0,f'No molparams: {match_keys_vals=}'
    return retval

def get_molparam_row(species_ID,local_isotopologue_ID):
    """Get Molparam data for one species/isotopologue"""
    retval = get_molparam().unique_row(
        species_ID=species_ID,
        local_isotopologue_ID=local_isotopologue_ID)
    return retval

# @tools.vectorise(cache=True)
# def get_molparam_for_species(species,key=None):
    # """Get HITEAN params data form a species in standard encoding. If
    # species given returns main isotopologue."""
    # data = get_molparam().unique_row(species=database.normalise_species(species))
    # if key is None:
        # return data
    # else:
        # return data[key]

# def translate_codes_to_species(
        # species_ID,
        # local_isotopologue_ID,
# ):
    # """Turn HITRAN species and local isotoplogue codes into a standard
    # species string. If no isotopologue_ID assumes the main one."""
    # i = molparam.find(species_ID=species_ID,
                      # local_isotopologue_ID=local_isotopologue_ID)
    # return molparam['chemical_species'][i],molparam['species'][i]

def translate_species_to_codes(species):
    """Get hitran species and isotopologue codes from species name.
    Assumes primary isotopologue if not indicated."""
    molparam = get_molparam()
    if len(i:=find(molparam.match(species=species))) > 0:
        assert len(i) == 1
        i = i[0]
    elif len(i:=find(molparam.match(chemical_species=species))) > 0:
        i = i[np.argmax(molparam['natural_abundance'][i])]
    else:
        raise DatabaseException(f"Cannot find {species=}")
    return (int(molparam['species_ID'][i]),
            int(molparam['local_isotopologue_ID'][i]))

def download_linelist(
        species, 
        νbeg,νend,
        data_directory='td',
        table_name=None,        # defaults to species
):
    """Doiwnload linelist for a species"""
    hapi = _import_hapi()
    if table_name is None:
        table_name = species
    MOL,ISO = translate_species_to_codes(species)
    mkdir(data_directory)
    hapi.db_begin(data_directory)
    hapi.fetch(table_name,int(MOL),int(ISO),νbeg,νend)

def calculate_spectrum_with_hapi(
        species,
        data_directory,
        T,p,
        νbeg,νend,νstep,
        table_name=None,
        make_plot= True,
):
    """Plot data. Must be already downloaded with download_linelist."""
    hapi = _import_hapi()
    if table_name is None:
        table_name = species
    MOL,ISO = translate_species_to_codes(species)
    hapi.db_begin(data_directory)
    ## calculate spectrum
    ν,coef = hapi.absorptionCoefficient_Lorentz(
        SourceTables=table_name,
        WavenumberRange=[νbeg,νend],
        WavenumberStep=νstep,
        Environment={"T":T,"p":p},)
    if make_plot:
        ax = plotting.gca()
        ax.plot(ν,coef,label=species)
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber (cm-1)')
        ax.set_ylabel('Absorption coefficient')
    return ν,coef

def calculate_spectrum(
        species,
        Teq=296,
        pair=1e5,
        N=1e16,
        xkey='ν',
        ykey='τ',
        **line_calculate_spectrum_kwargs
):
    """Calculate cross section."""
    line = get_line(species)
    line['Teq'] = Teq
    line['pair'] = pair
    line['Nchemical_species'] = N
    x,y = line.calculate_spectrum(
        xkey=xkey,
        ykey=ykey,
        **line_calculate_spectrum_kwargs)
    return x,y

def plot_spectrum(species,**calculate_spectrum_kwargs):
    ax = plotting.gca()
    for speciesi in tools.ensure_iterable(species):
        x,y = calculate_spectrum(
            species=speciesi,
            **calculate_spectrum_kwargs)
        ax.plot(x,y,label=speciesi)
    plotting.legend(ax=ax)

def load_linelist(filename,modify=True):
    """Load HITRAN .data file into a dictionary. If modify=True then
    return as a Dataset with various changes in line with the
    subclasses in lines.py."""
    data = np.genfromtxt(
        expand_path(filename),
        dtype=[
            ('Mol',int),    # molecule code number
            ('Iso','U1'),    # isotopologue code number
            ('ν0',float),    # wavenumber
            ('S296K',float), # spectral line intensity at 296K cm-1(molecular.cm-2), the integrated cross section at 296K
            ('A',float), # Einstein A-coefficient
            ('γ0air',float),  # air broadening coefficient
            ('γ0self',float), # self broadening coefficient
            ('E_l',float), # lower level energy
            ('nγ0air',float), # broadening temperature dependence
            ('δ0air',float), # pressure shift in ari (cm-1/atm)
            ('V_u','U15'),       # Upper-state “global” quanta 
            ('V_l','U15'),      # Lower-state “global” quanta 
            ('Q_u','U15'),       # Upper-state “local” quanta  
            ('Q_l','U15'),      # Lower-state “local” quanta  
            ('Ierr',int),     # Uncertainty indices         
            ('Iref',int),     # Reference indices           
            ('asterisk','U1'), # Flag                           
            ('g_u',float),    # lower level degeneracy
            ('g_l',float),    # upper level degeneracy
        ],
        delimiter=(2,1,12,10,10,5,5,10,4,8,15,15,15,15,6,12,1,7,7), # column widths
    )
    retval = {key:data[key] for key in data.dtype.names}
    iso_translate = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7,
                     '8':8, '9':9, '0':10, 'A':11, 'B':12, 'C':13, 'D':14, 'E':15,
                     'F':16, 'G':17, 'H':18, 'I':19, 'J':20,}
    retval['Iso'] = [iso_translate[t] for t in retval['Iso']]
    ## if modify then compute species and chemical_species and remove
    ## weighting by isotopologue natural abundance
    if modify:
        retval = Dataset(**retval)
        retval['species'] = ''
        retval['chemical_species'] = ''
        for (Mol,Iso),i in tools.unique_combinations_masks(
                retval['Mol'],retval['Iso']):
            row = get_molparam_row(Mol,Iso)
            retval['chemical_species',i] = row['chemical_species']
            retval['species',i] = row['species']
            retval['S296K',i] /= row['natural_abundance']
        ## not used
        retval.unset('Mol')
        retval.unset('Iso')
        retval.unset('A')
        retval.unset('Ierr')
        retval.unset('Iref')
        retval.unset('asterisk')
    return retval

def load_cross_section(filename,name='hitran_cross_section'):
    """Load cross section in HITRAN .xsc format, and save into a Spectrum
    dataset."""
    from .spectrum import Spectrum
    retval = Spectrum(name=name)
    retval.attributes['filename'] = filename
    retval.description = f'Data in HITRAN cross section file {filename!r}'
    with open(tools.expand_path(filename),'r') as fid:
        ## read header
        line = fid.readline()
        strip = lambda x: x.strip()
        def broadener(x):
            x = x.strip()
            if len(x) == 0:
                x = 'self'
            return x
        for key,length,cast in (
                ('species',20,strip),
                ('νbeg',10,float),
                ('νend',10,float),
                ('npoints',7,int),
                ('temperature',7,float),
                ('pressure',6,float),
                ('σmax',10,float),
                ('resolution',5,float),
                ('name',15,strip),
                ('unused',4,strip),
                ('broadener',3,broadener),
                ('hitran_reference_code',3,int),
        ):
            retval.attributes[key],line = cast(line[:length]),line[length:]
        ## read spectrum
        data = fid.readlines()
        retval['σ'] = np.concatenate([np.array(t.split(),dtype=float) for t in data])[:retval.attributes['npoints']]
        retval['ν'] = np.linspace(retval.attributes['νbeg'], retval.attributes['νend'], retval.attributes['npoints'],)
    ## accurate format input function
    retval.pop_format_input_function()
    retval.add_format_input_function(lambda: f'{retval.name} = hitran.load_cross_section({filename=},{name=})')
    return retval

# @tools.cache
# def _load_and_cache(filename):
    # line = dataset.load(filename)
    # return line

_get_line_cache = {}

def get_line(
        species,
        name=None,
        match=None,
        force_download=False,   # download HITRAN data even if it is already present
        force_convert=False,    # recompute line object from HITRAN data even if it is alreay present
        use_cache=True,             # cache loaded line for later faster loading
        copy_cache=True,        # copy cached line to prevent side effects if loaded twice
        **match_kwargs):
    """Hitran linelist.  If species not an isotopologue then load a list
    of the natural abundance mixture.  Adds some extra quantum numbers
    from additional_electronic_quantum_numbers_to_add."""
    species = database.normalise_species(species)
    chemical_species = get_species_property(species,'chemical_formula')
    directory = f'{database.data_directory}/hitran/cache/{species}'
    ## delete data directory to force download if requested
    hitran_filename = f'{directory}/hitran_linelist.data'
    line_filename = f'{directory}/line'
    if not force_download and not force_convert and os.path.exists(line_filename):
        ## get data cached on disk
        if use_cache:
            ## get data cached in memory
            if line_filename in _get_line_cache:
                line = _get_line_cache[line_filename]
            else:
                line = dataset.load(line_filename)
                _get_line_cache[line_filename] = line
            if copy_cache:
                line = line.copy()
        else:
            line = dataset.load(line_filename)
    else:
        ## get new data onto disk
        if is_known_species(species):
            ## an isotopologue download HITRAN linelist
            if force_download or not os.path.exists(hitran_filename):
                ## download new data
                print(f'Downloading hitran data for {species!r}')
                download_linelist(species,νbeg=0,νend=999999,data_directory=directory,table_name='hitran_linelist')
                ## delete any existing line list
                if os.path.exists(line_filename):
                    shutil.rmtree(line_filename)
            if force_convert or not os.path.exists(line_filename):
                ## make new line object from HITRAN data
                print(f'Making linelist for {species!r}')
                try:
                    classname = get_species_property(chemical_species,'classname')
                except DatabaseException:
                    classname = 'Generic'
                line = dataset.make(
                    classname=f'lines.{classname}',
                    description=f'HITRAN linelist for {species}, downloaded {date_string()}',
                    Zsource='HITRAN',)
                line.load_from_hitran(hitran_filename)
                line['mass']                # compute now for speed later
                line.save(line_filename,filetype='directory')
        elif is_known_chemical_species(species):
            ## a chemical species -- get natural abundance mixture
            if force_download or force_convert or not os.path.exists(line_filename):
                ## need to make a linelist
                for j,tspecies in enumerate(get_isotopologues(species)):
                    ## concatenate all isotopologues
                    tline = get_line(tspecies,force_download=force_download,force_convert=force_convert)
                    tline['isotopologue_ratio'] = get_natural_abundance(tspecies)
                    if j==0:
                        line = tline.copy()
                    else:
                        line.concatenate(tline)
                line['chemical_species'] = species
                line.save(line_filename,filetype='directory')
        else:    
            raise DatabaseException(f'Species or chemical_species unknown to hitran.py: {species!r}')
        ## save to cache
        if use_cache:
            _get_line_cache[line_filename] = line
    ## filter data
    if name is None:
        line.name =f'hitran_line_{species}'
    else:
        line.name = name
    ## limit data
    line.limit_to_match(match,**match_kwargs)
    line.unset_inferred()
    ## replace format input function with a reference to this function
    line.clear_format_input_functions()
    def f():
        retval = f'{line.name} = hitran.get_line({repr(species)}'
        if line.name is not None:
            retval += f',name={repr(line.name)}'
        if match is not None:
            retval += f',match={repr(match)}'
        retval += ')'
        return retval
    line.add_format_input_function(f)
    return line

_get_level_cache = {}

def get_level(
        species,
        name=None,
        match=None,
        force_download=False,
        force_convert=False,
        **match_kwargs):
    """Get upper level from HITRAN data."""
    species = database.normalise_species(species)
    filename = f'{database.data_directory}/hitran/cache/{species}/level'
    if not force_download and not force_convert and os.path.exists(filename):
        ## load existing data
        if cache:
            if filename in _get_level_cache:
                level = _get_level_cache[filename].copy()
            else:
                level = dataset.load(filename)
                _get_level_cache[filename] = level.copy()
        else:
            level = dataset.load(filename)
    else:
        ## compute from line
        line = get_line(
            species,
            force_download=force_download,
            force_convert=force_convert,)
        level = line.get_level(reduce_to='first')
        level.sort('E')
        level.name =f'hitran_level_{species}'
        ## save to disk cache
        level.save(filename,filetype='directory')
    ## match etc
    if name is not None:
        level.name = name
    level.limit_to_match(match,**match_kwargs)
    level.unset_inferred()
    return level





