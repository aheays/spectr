import numpy as np

import brukeropusreader

def load(filename):
    """Load a binary Bruker Opus file into a dictionary."""
    return brukeropusreader.read_file(expand_path(filename))

def load_spectrum(filename,datablock='ScSm'):
    """Load a binary Bruker Opus file, returning a specific datablock as well as all data in a
    dictionary. Useful datablocks:
    IgSm:  the single-channel sample interferogram
    ScSm:  the single-channel sample spectrum
    IgRf:  the reference (background) interferogram
    ScRf:  the reference (background) spectrum
    """
    d = load_bruker_opus(filename)
    if datablock not in d:
        raise Exception(f'Cannot find datablock: {repr(datablock)}.  Existing datablocks are: {repr(list(d))}')
    parameters = d[f'{datablock} Data Parameter']
    x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
    y = d[datablock][:parameters['NPT']] # for some reason the data in the datablock can be too long
    return x,y


