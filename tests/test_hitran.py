import pytest
from spectr import hitran,lines

def test_load_lines():
    data = hitran.load_lines('data/hitran_data/CO/hitran_linelist.data')
    print( set(data.dtype.names))
    assert len(data) == 849
    assert set(data.dtype.names) == {'asterisk', 'Ierr', 'δair', 'g_u', 'Iref', 'γair', 'γself', 'A', 'g_l', 'nair', 'S', 'Q_l', 'V_u', 'Q_u', 'Iso', 'Mol', 'V_l', 'E_l', 'ν'}

def test_load_into_HeteronuclearDiatomicLines():
    t = lines.HeteronuclearDiatomicLines()
    t.load_from_hitran('data/hitran_data/CO/hitran_linelist.data')
    assert len(t) == 849
    assert set(t.keys()) == {'γair', 'v_u', 'Ae', 'g_u', 'v_l', 'ΔJ', 'γself', 'J_l', 'species', 'ν', 'nair', 'δair', 'E_l', 'g_l'}
