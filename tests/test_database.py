import pytest
from pytest import raises,approx
import numpy as np

from spectr import database
# from spectr.optimise import P
# from spectr.exceptions import InferException


def test_get_electronic_state_property():
    assert database.get_electronic_state_property('N2','X','Λ') == 0
    assert database.get_electronic_state_property('N2','X','S') == 0
    assert database.get_electronic_state_property('N2','X','s') == 0
    assert database.get_electronic_state_property('N2','X','gu') == 1
    assert database.get_electronic_state_property('C2','X','S') == 0
    assert list(database.get_electronic_state_property('C2',['X','a','A'],'S')) == [0,1,0]
    assert list(database.get_electronic_state_property(['C2','C2','C2'],['X','a','A'],'S')) == [0,1,0]
    assert list(database.get_electronic_state_property('C2','X',['S','gu'])) == [0,1]
