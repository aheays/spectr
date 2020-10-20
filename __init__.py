## standard library
from copy import copy,deepcopy
from pprint import pprint

## nonstandard library
import numpy as np

## import subpackages of this library
from . import tools
from . import plotting
from . import optimise
from . import dataset
from . import levels
from . import lines
from . import hitran
from . import database
from . import spectrum
from . import kinetics
from . import bruker

## import more explicitly for interactive use
from .dataset import Dataset
from .conversions import convert
from .plotting import *
from .tools import *

