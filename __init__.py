## to speed up import I shifted all this to env.py so can import everything with "from spectr.env import *"

## standard library
# from copy import copy,deepcopy
# from pprint import pprint,pformat
# import shutil
# import warnings

# # ## nonstandard library
# # import numpy as np
# # np.set_printoptions(linewidth=np.nan) 
# # from numpy import array,nan,arange,linspace,logspace,isnan,inf
# # from scipy import integrate
# # from scipy.constants import pi as π
# # from scipy.constants import Boltzmann as kB

# ## import subpackages of this library
# try:
    # from .fortran_tools import fortran_tools
# except ModuleNotFoundError:
    # fortran_tools = None
    # warnings.warn("Could not import fortran_tools.  Is it compiled?")
# # from . import tools
# # from . import dataset
# # from . import optimise
# # from . import plotting
# # from . import database
# # from . import spectrum
# # from . import kinetics
# # from . import thermochemistry
# # from . import atmosphere
# # from . import hitran
# # from . import bruker
# # from . import electronic_states
# # from . import viblevel
# # from . import cross_section
# # from . import lineshapes

# # ## import more explicitly for interactive use
# # from .dataset import Dataset
# # from . import convert
# # from .convert import units
# # from matplotlib.pyplot import * # before plotting
# # from .plotting import *
# # from .tools import *
# # from .optimise import Optimiser,Parameter,P,Fixed

