## standard library
from copy import copy,deepcopy
from pprint import pprint,pformat
import shutil
import warnings

## nonstandard library
import numpy as np
np.set_printoptions(linewidth=np.nan) 
from numpy import array,nan,arange,linspace,logspace,isnan,inf
from scipy import integrate
from scipy.constants import pi as π
from scipy.constants import Boltzmann as kB

## import subpackages of this library
try:
    from .fortran_tools import fortran_tools
except ModuleNotFoundError:
    fortran_tools = None
    warnings.warn("Could not import fortran_tools.  Is it compiled?")

from . import tools
from . import quantum_numbers
from . import dataset
from . import optimise
from . import plotting
from . import database
from . import levels
from . import lines
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
from . import convert

from .tools import *
from .dataset import Dataset
from .convert import units
from matplotlib.pyplot import * # before plotting
from .optimise import Optimiser,Parameter,P,Fixed
from .plotting import *
from . import plotting
