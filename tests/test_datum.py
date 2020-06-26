from spectr.datum import Datum
import numpy as np

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
