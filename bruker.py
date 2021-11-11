import numpy as np

import brukeropusreader
from .tools import *
from .plotting import *

class OpusData:

    def __init__(self,filename):
        """Load a binary Bruker Opus file into a dictionary."""
        self.filename = filename
        self.data = brukeropusreader.read_file(expand_path(filename))
        ##pprint(self.data)       #  DEBUG

    def has_spectrum(self):
        return 'ScSm' in self.data

    def has_interferogram(self):
        return 'IgSm' in self.data

    def has_background(self):
        return 'ScRf' in self.data

    def get_spectrum(self):
        if not self.has_spectrum():
            print('No spectrum (ScSm) found, looking for background.')
            return self.get_background()
        parameters = self.data[f'ScSm Data Parameter']
        x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        y = self.data['ScSm'][:parameters['NPT']] # for some reason the data in the datablock can be too long
        return x,y

    def get_interferogram(self):
        if not self.has_interferogram():
            raise Exception('No interferogram (IgSm) found.')
            return self.get_background()
        parameters = self.data[f'IgSm Data Parameter']
        y = self.data['IgSm'][:parameters['NPT']] 
        return y

    def set_spectrum(self,y):
        assert self.has_spectrum(),'No spectrum (ScSm) found.'
        parameters = self.data[f'ScSm Data Parameter']
        x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        self.data['ScSm'][:parameters['NPT']] = y # for some reason the data in the datablock can be too long

    def get_background(self):
        assert self.has_background(),'No background (ScRf) found.'
        parameters = self.data[f'ScRf Data Parameter']
        x = np.linspace(parameters['FXV'], parameters['LXV'], parameters['NPT'])
        y = self.data['ScRf'][:parameters['NPT']] # for some reason the data in the datablock can be too long
        return x,y
    
    def get_interpolation_factor(self):
        return float(self.data['Fourier Transformation']['ZFF'])

    def get_apodisation_function(self):
        if self.data['Fourier Transformation']['APF'] == 'B3':
            return 'Blackman-Harris 3-term'
        else:
            raise Exception(f"Unknown opus apodisation function: {repr(d['Fourier Transformation']['APF'])}")

    def get_resolution(self,kind='resolution'):
        """Return spectral resolution."""
        ## get spectrum grid step (cm-1)
        retval = ((self.data[f'ScSm Data Parameter']['LXV']
               -self.data[f'ScSm Data Parameter']['FXV'])
              /(self.data[f'ScSm Data Parameter']['NPT']-1))
        ## account for interpolation
        retval = retval*int(self.data['Fourier Transformation']['ZFF'])
        ## fixed to powers of 2 of 1e-3cm-1
        # retval = 2**round(np.log2(retval*1000))/1000
        ## convert to full-width half-maximum if requested
        # if kind == 'zero-to-zero':
            # ## distance between central zeros of sinc
            # pass
        if kind == 'resolution':
            ## distance between peak and first zero of sinc -- this is
            ## the definition of resolution in Opus
            pass
        elif kind == 'fwhm':
            ## convert to fwhm of sinc central peak 
           # retval /= 1.2
            retval *= 1.2
            pass                # confuse
        return retval

    def plot_interferogram(self,ax=None,**plot_kwargs):
        """Plot interferogram."""
        assert self.has_interferogram()
        if ax is None:
            ax = plt.gca()
        ax.plot(*self.get_interferogram(),**plot_kwargs)

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
        
def load_spectrum(filename):
    """Load ScSm from a file"""
    o = OpusData(filename)
    return o.get_spectrum()

def load_data(filename):
    """Load ScSm from a file"""
    o = OpusData(filename)
    return o.data
