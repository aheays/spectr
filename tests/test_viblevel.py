from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import viblevel
# from spectr import spectrum
# from spectr import tools
# from spectr import lines
# from spectr import levels
# from spectr import plotting

make_plot = False 

def test_init():
    viblevel.VibLevel('test','N2')
