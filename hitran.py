import functools
from copy import deepcopy

import numpy as np

from . import lines
from . import plotting
from . import tools
from .tools import *
from . import kinetics
from . import dataset
from . import database
from . import quantum_numbers
from .dataset import Dataset
from. exceptions import DatabaseException


@tools.vectorise(cache=True,dtype=float)
def get_partition_function(species,temperature):
    """Use hapi to get a partition function.  Uses main isotopologue if
    not given."""
    from hapi import hapi
    Mol,Iso = translate_species_to_codes(species)
    return hapi.partitionSum(Mol,Iso,temperature)

_molparam = None
def get_molparam(**match_keys_vals):
    global _molparam
    ## load data if needed
    if _molparam is None:
        _molparam = dataset.load(f'{database.data_directory}/hitran/molparam.psv')
    retval = _molparam.matches(**match_keys_vals)
    assert len(retval)>0,f'No molparams: {match_keys_vals=}'
    return retval

@tools.vectorise(cache=True)
def get_molparam_for_species(species,key=None):
    """Get HITEAN params data form a species in standard encoding. If
    species given returns main isotopologue."""
    data = get_molparam().unique_row(species=database.normalise_species(species))
    if key is None:
        return data
    else:
        return data[key]

def translate_codes_to_species(
        species_ID,
        local_isotopologue_ID,
):
    """Turn HITRAN species and local isotoplogue codes into a standard
    species string. If no isotopologue_ID assumes the main one."""
    i = molparam.find(species_ID=species_ID,
                      local_isotopologue_ID=local_isotopologue_ID)
    return molparam['chemical_species'][i],molparam['species'][i]

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
    from hapi import hapi
    if table_name is None:
        table_name = species
    MOL,ISO = translate_species_to_codes(species)
    mkdir(data_directory)
    hapi.db_begin(data_directory)
    hapi.fetch(table_name,int(MOL),int(ISO),νbeg,νend)

def calc_spectrum(
        species,
        data_directory,
        T,p,
        νbeg,νend,νstep,
        table_name=None,
        make_plot= True,
):
    """Plot data. Must be already downloaded with download_linelist."""
    from hapi import hapi
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

def load(filename):
    """Load HITRAN .data file into a Dataset."""
    data = np.genfromtxt(
        expand_path(filename),
        dtype=[
            ('Mol',int),    # molecule code number
            ('Iso','U1'),    # isotopologue code number
            ('ν',float),    # wavenumber
            ('S',float), # spectral line intensity at 296K cm-1(molecular.cm-2), the integrated cross section at 296K
            ('A',float), # Einstein A-coefficient
            ('γair',float),  # air broadening coefficient
            ('γself',float), # self broadening coefficient
            ('E_l',float), # lower level energy
            ('nair',float), # broadening temperature dependence
            ('δair',float), # pressure shift in ari (cm-1/atm)
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
    retval = Dataset(**retval)
    return retval

@functools.lru_cache
def _get_lines_internal(filename):
    line = dataset.load(filename)
    line['Zsource'] = 'HITRAN'
    line['mass']                # compute now for speed later
    return line

## these are added to HITRAN data to fill in missing quantum numbesr --very hacky
_electronic_quantum_numbers_to_add_to_hitran = {
    'CS₂': {'label_u':'X_u','Λ_u':0,'S_u':0,'s_u':0,'ef_u':1,
            'label_l':'X_l','Λ_l':0,'S_l':0,'s_l':0,'ef_l':1,}, # verify symmetry!
    }

def get_line(species,name=None,match=None,force_download=False,**match_kwargs):
    """Hitran linelist.  If species not an isotopologue then load a list
    of the natural abundance mixture."""
    species = database.normalise_species(species)
    chemical_species = database.normalise_chemical_species(species)
    directory = f'{database.data_directory}/hitran/cache/{species}'
    ## delete data directory to force download if requested
    if force_download and os.path.exists(directory):
        import shutil
        shutil.rmtree(directory)
    hitran_filename = f'{directory}/hitran_linelist.data'
    line_filename = f'{directory}/lines'
    if os.path.exists(line_filename) and not force_download:
        ## load existing data
        line = deepcopy(_get_lines_internal(line_filename))
    else:
        data = get_molparam()
        if np.any(i:=data.match(chemical_species=species)):
            ## a chemical species -- get natural abundance mixture
            for j,row in enumerate(data[i].rows()):
                new_line = get_line(row['species'],force_download=force_download)
                new_line['isotopologue_ratio'] = row['natural_abundance']
                if j == 0:
                    line = new_line
                else:
                    line.concatenate(new_line)
            line['chemical_species'] = species
        elif np.any(i:=data.match(species=species)):
            ## an isotopologue download HITRAN linelist if necessary
            ## and convert lines
            if not os.path.exists(hitran_filename) or force_download:
                print(f'Downloading hitran data for {species!r}')
                download_linelist(species,νbeg=0,νend=999999,data_directory=directory,table_name='hitran_linelist')
            ## make line object
            print(f'Making linelist for {species!r}')
            classname = get_molparam(species=species)['dataset_classname'][0]
            line = dataset.make(classname,description=f'HITRAN linelist for {species}, probably downloaded {date_string()}')
            line.load_from_hitran(hitran_filename)
            line['species'] = species
        else:    
            raise Exception(f'Species or chemical_species unknown to hitran.py: {species!r}')
        ## quantum number hacks
        for key_to_try in (chemical_species,species):
            if key_to_try in _electronic_quantum_numbers_to_add_to_hitran:
                for key,val in _electronic_quantum_numbers_to_add_to_hitran[key_to_try].items():
                    line[key] = val
        ## save data
        line.save(line_filename,filetype='directory')
    ## filter data
    if name is None:
        line.name =f'hitran_lines_{species}'
    else:
        line.name = name
    ## limit data
    if match is not None or len(match_kwargs) > 0:
        line.limit_to_match(match,**match_kwargs)
    ## replace format input function with a reference to this function
    line.clear_format_input_functions()
    def f():
        retval = f'{line.name} = hitran.get_lines({repr(species)}'
        if line.name is not None:
            retval += f',name={repr(name)}'
        if match is not None:
            retval += f',match={repr(match)}'
        retval += ')'
        return retval
    line.add_format_input_function(f)
    return line

def get_level(species,*get_line_args,**get_line_kwargs):
    """Get upper level from HITRAN data."""
    ## load HITRAN line data
    line = get_line(species,*get_line_args,**get_line_kwargs)
    ## combine upper and lower levels into a single Dataset and remove duplicates
    required_keys=('E',)
    level = line.get_lower_level(reduce_to='first',required_keys=required_keys)
    level.concatenate(line.get_upper_level(reduce_to='first',required_keys=required_keys))
    qnhash,i = np.unique(level['qnhash'],return_index=True)
    level.index(i)
    level.sort('E')
    return level


