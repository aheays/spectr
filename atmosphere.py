import functools
from copy import copy,deepcopy

from scipy import constants
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import kinetics
from . import tools
from . import database
from .exceptions import InferException
from . import plotting as plt


class Atmosphere(Dataset):
    """1D model atmosphere"""

    prototypes = {}
    prototypes['description'] = dict( description="",kind=str ,infer={})
    prototypes['notes'] = dict(description="Notes regarding this line" , kind=str ,infer={})
    prototypes['author'] = dict(description="Author of data or printed file" ,kind=str ,infer={})
    prototypes['reference'] = dict(description="Published reference" ,kind=str ,infer={})
    prototypes['date'] = dict(description="Date data collected or printed" ,kind=str ,infer={})
    prototypes['z'] = dict(description="Height above surface (cm)" ,kind=float ,infer={})
    prototypes['T'] = dict(description="Temperature (K)" ,kind=float ,infer={})
    prototypes['nt'] = dict(description="Total number density (cm-3)" ,kind=float ,infer={})
    prototypes['p'] = dict(description="Pressure (bar)" ,kind=float ,infer={})
    prototypes['Kzz'] = dict(description="Turbulent diffusion constant (cm2.s-1)" ,kind=float ,infer={})
    prototypes['Hz'] = dict(description="Local scale height (cm1)" ,kind=float ,infer={})
    
    
    def load_argo_depth(self,filename):
        """Load an argo depth.dat file."""
        data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        ## load physical parameters
        for key_from,key_to in (
                ('p(bar)','p'),
                ('T(K)','T'),
                ('NH(cm-3)','nt'),
                ('Kzz(cm2s-1)','Kzz'),
                ('Hz(cm)','z'),
                ('zeta(s-1)','zeta(s-1)'),
                ('h','h'),
                ('f+','f+')
        ):
            if key_to in self:
                assert np.all(self[key_to] == data.pop(key_from))
            else:
                self[key_to] = data.pop(key_from)
        ## load abundances
        for key in data:
            standard_key = kinetics.translate_species(key,'ARGO','standard')
            self['n('+standard_key+')'] = data[key]
    
    def load_argo_lifetime(self,filename):
        """Load an argo lifetime.dat file."""
        data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        ## load physical parameters
        for key_from,key_to in (
                ('p(bar)','p'), ('T(K)','T'), ('NH(cm-3)','nt'),
                ('Kzz(cm2s-1)','Kzz'), ('Hz(cm)','z'),
                ('zeta(s-1)','zeta(s-1)'), ('h','h'), ('f+','f+')
        ):
            if key_to in self:
                assert np.all(self[key_to] == data.pop(key_from))
            else:
                self[key_to] = data.pop(key_from)
        ## load abundances
        for key in data:
            standard_key = kinetics.translate_species(key,'ARGO','standard')
            self['τ('+standard_key+')'] = data[key]
    
    def plot_vertical(self,ykey,*xkeys,ax=None):
        if ax is None:
            ax = plt.gca()
        for xkey in xkeys:
            ax.plot(self[xkey],self[ykey],label=xkey)
        ax.set_xscale('log')
        ax.set_ylim(self[ykey].min(),self[ykey].max())
        plt.legend(ax=ax)

