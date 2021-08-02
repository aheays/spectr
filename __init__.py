## standard library
from copy import copy,deepcopy
from pprint import pprint,pformat
import shutil
import warnings

## nonstandard library
import numpy as np
np.set_printoptions(linewidth=np.nan) 
from numpy import array,nan,arange,linspace
from scipy import integrate
from scipy.constants import pi as π
from scipy.constants import Boltzmann as kB

## import subpackages of this library
try:
    from .fortran_tools import fortran_tools
except ModuleNotFoundError:
    warnings.warn("Could not import fortran_tools.  Is it compiled?")
from . import tools
from . import plotting
from . import optimise
from . import dataset
from .dataset import Dataset
from . import levels
from . import lines 
from . import database
from . import spectrum
from . import kinetics
from . import thermochemistry
from . import atmosphere
from . import hitran
from . import bruker
from . import electronic_states
from . import viblevel
from . import cross_section
from . import lineshapes

## import more explicitly for interactive use
from .dataset import Dataset
from . import convert
from .convert import units
from matplotlib.pyplot import * # before plotting
from .plotting import *
from .tools import *
from .optimise import Optimiser,Parameter,P,Fixed

