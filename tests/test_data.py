import pytest
import numpy as np
from spectr.data import Data

def test_construct():
    t = Data(value=[1,2])
    assert list(t.value) == [1,2]
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    assert list(t.value) == [1.5,2.5]
    assert list(t.uncertainty) == [0.3,0.2]
    assert len(t) == 2
    t = Data(value=[1.5,2.5],uncertainty=1)
    assert list(t.value) == [1.5,2.5]
    assert list(t.uncertainty) == [1,1]
    t = Data(value=[1,2],uncertainty=1)
    assert list(t.value) == [1,2]
    assert list(t.uncertainty) == [1,1]
    with pytest.raises(ValueError):
        t = Data(value=[1,2],uncertainty=[1,2,3])

def test_has_uncertainty():
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    assert t.has_uncertainty()
    t = Data(value=[1.5,2.5])
    assert not t.has_uncertainty()

def test_index():
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    t.index([True,False])
    assert list(t.value) == [1.5]
    assert list(t.uncertainty) == [0.3]
    assert len(t) == 1

def test_append():
    t = Data(value=[2,34])
    t.append(5)
    assert list(t.value) == [2,34,5]
    t = Data(value=[2.,34.],uncertainty=[0.2,0.3])
    t.append(5,0.5)
    assert list(t.value) == [2.,34.,5.]
    assert list(t.uncertainty) == [0.2,0.3,0.5]

def test_extend():
    t = Data(value=['a','b'])
    t.extend(['c','d'])
    assert list(t) == ['a','b','c','d']
    t = Data(value=[1.,2.],uncertainty=[0.1,0.1])
    t.extend([3,4],[1,2])
    assert list(t.value) == [1.,2.,3.,4.,]
    assert list(t.uncertainty) == [0.1,0.1,1.,2.]

def test_object():
    t = Data(value=[[1,2],None],kind=object)
    assert list(t.value) == [[1,2],None]
    t = Data(value=[{},np.array([1,2,3]),None],kind=object)
    assert t.value[0] == {}

# def test_deepcopy():
    # x = Data(value=[1,2,3])
    # y = x.copy()
    # assert list(y.get_value()) == [1,2,3]
    # y = x.copy([0,1])
    # assert list(y.get_value()) == [1,2]

def test_str():
    t = Data(value=[1,2,3])
    assert str(t)=='1\n2\n3'
    t = Data(value=[1,2,3],uncertainty=[0.1,5,6])
    assert str(t)=='+1.00000000e+00 ± 0.1\n+2.00000000e+00 ± 5\n+3.00000000e+00 ± 6'
    t = Data(value=['a','b','c'])
    assert str(t)=='a\nb\nc'

