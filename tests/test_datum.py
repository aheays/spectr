import pytest
import numpy as np

from spectr.datum import Datum

def test_construct():
    t = Datum(value=1)
    assert t._value == 1
    assert t.value == 1
    assert t.kind == 'i'
    assert not t.has_uncertainty()
    t = Datum(value='dd')
    assert t.kind == 'U'
    assert t.value == 'dd'
    t = Datum(value=1.5,uncertainty=0.3)
    assert t.has_uncertainty()
    assert t.kind == 'f'
    assert t.value == 1.5
    assert t.uncertainty == 0.3
    t = Datum(value=1,uncertainty=0.1)
    assert t.kind == 'f'
    assert t.has_uncertainty()
    t = Datum(value=-51,kind=float,cast=lambda x:float(abs(x)))
    assert t.value == 51
    t = Datum(value=[1,2,3])
    assert t.value == [1,2,3]
    assert t.kind == 'O'
    assert not t.has_uncertainty()
    with pytest.raises(ValueError):
        t = Datum(value='a',kind=float)
    with pytest.raises(AssertionError):
        t = Datum(value='a',uncertainty=1.)

def test_has_uncertainty():
    t = Datum(value=1,uncertainty=1)
    assert t.has_uncertainty()
    t = Datum(value=1)
    assert not t.has_uncertainty()

def test_str():
    t = Datum(value=1,uncertainty=0.1)
    assert str(t)=='+1.00000000e+00 ± 0.1'
    t = Datum(value=1,uncertainty=0.1,fmt='0.5f')
    assert str(t)=='1.00000 ± 0.1'
    t = Datum(value='a')
    assert str(t)=='a'
    t = Datum(value=1)
    assert str(t)=='1'

def test_math_builtins():
    t = Datum(value=5)
    assert -t == -5
    assert float(t) == 5.0
    assert +t == 5
    assert abs(t) == 5
    assert t + 2 == 7
    assert 2 + t == 7
    assert t - 2 == 3
    assert 2 - t == -3
    assert t / 2 == 2.5
    assert 10 / t == 2
    assert t * 2 == 10
    assert 10 * t == 50
    assert t**2 == 25
    assert 2**t == 32
    
