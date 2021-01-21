import numpy as np

import brukeropusreader
from .tools import *
from .plotting import *

class OpusData:

    def __init__(self,filename):
        """Load a binary Bruker Opus file into a dictionary."""
        self.filename = filename
        self.data = brukeropusreader.read_file(expand_path(filename))

    def has_spectrum(self):
        return 'ScSm' in self.data

    def has_background(self):
        return 'ScRf' in self.data

    def get_spectrum(self):
        assert self.has_spectrum(),'No spectrum (ScSm) found.'
        parameters = self.data[f'ScSm Data Parameter']
        x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        y = self.data['ScSm'][:parameters['NPT']] # for some reason the data in the datablock can be too long
        return x,y

    def get_background(self):
        assert self.has_background(),'No background (ScRf) found.'
        parameters = self.data[f'ScRf Data Parameter']
        x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        y = self.data['ScRf'][:parameters['NPT']] # for some reason the data in the datablock can be too long
        return x,y
    
    def plot(
            self,
            plot_spectrum=True,
            plot_background=True,
            ax=None, 
            **plot_kwargs):
        """Plot data."""
        if ax is None:
            ax = plt.gca()
        if plot_spectrum:
            if self.has_background():
                kwargs = copy(plot_kwargs)
                kwargs.setdefault('color',newcolor(1))
                kwargs.setdefault('label','background')
                ax.plot(*self.get_spectrum(),**kwargs)
            else:
                print(f'No background in file: {self.filename}')    
            if self.has_spectrum():
                kwargs = copy(plot_kwargs)
                kwargs.setdefault('color',newcolor(0))
                kwargs.setdefault('label','spectrum')
                ax.plot(*self.get_spectrum(),**kwargs)
            else:
                print(f'No spectrum in file: {self.filename}')    
            
        
        
    # def load_spectrum(filename,datablock='ScSm'):
        # """Load a binary Bruker Opus file, returning a specific datablock as well as all data in a
        # dictionary. Useful datablocks:
        # IgSm:  the single-channel sample interferogram
        # ScSm:  the single-channel sample spectrum
        # IgRf:  the reference (background) interferogram
        # ScRf:  the reference (background) spectrum
        # """
        # d = load(filename)
        # if datablock not in d:
            # raise Exception(f'Cannot find datablock: {repr(datablock)}.  Existing datablocks are: {repr(list(d))}')
        # parameters = d[f'{datablock} Data Parameter']
        # x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        # y = d[datablock][:parameters['NPT']] # for some reason the data in the datablock can be too long
        # return x,y

    # def load_background(filename):
        # """Load a binary Bruker Opus file background,"""
        # return load_spectrum(filename,'ScRf')


