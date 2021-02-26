## standard library
from copy import copy,deepcopy
from pprint import pprint
import shutil

## nonstandard library
import numpy as np
from numpy import array,nan,arange,linspace
from scipy import integrate
from scipy.constants import pi as π

## import subpackages of this library
from .fortran_tools import fortran_tools
from . import tools
from . import plotting
from . import optimise
from . import dataset
from . import levels
from . import lines 
from . import database
from . import spectrum
from . import kinetics
from . import atmosphere
from . import hitran
from . import bruker
from . import electronic_states
from . import viblevel
from . import cross_section
from . import lineshapes

## import more explicitly for interactive use
from .dataset import Dataset
from .conversions import convert
from matplotlib.pyplot import * # before plotting
from .plotting import *
from .tools import *
from .optimise import Optimiser,Parameter,P,Fixed

