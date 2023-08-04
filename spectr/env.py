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
from matplotlib.pyplot import * 

## import top level
import spectr

## try to import fortran stuff
try:
    from .fortran_tools import fortran_tools
except ModuleNotFoundError:
    fortran_tools = None
    warnings.warn("Could not import fortran_tools.  Is it compiled?")


## import submodules referenced to top level
import spectr.tools
import spectr.quantum_numbers
import spectr.dataset
import spectr.optimise
import spectr.plotting
import spectr.database
import spectr.levels
import spectr.lines
import spectr.spectrum
import spectr.kinetics
import spectr.thermochemistry
import spectr.atmosphere
import spectr.hitran
import spectr.bruker
import spectr.electronic_states
import spectr.viblevel
import spectr.lineshapes
import spectr.convert

## import submodules referenced to nothing
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
from . import lineshapes
from . import convert

## import contents of submodules
from .tools import *
from .database import get_species_property
from .dataset import Dataset
from .convert import units
from .optimise import Optimiser,Parameter,P,Fixed
from .plotting import *

