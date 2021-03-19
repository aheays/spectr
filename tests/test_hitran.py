import pytest
from spectr import hitran,lines

def test_load_lines():
    data = hitran.load('data/hitran_data/CO/[12C][16O]/hitran_linelist.data')
    assert len(data) == 19
    assert len(data['ν']) == 1344
    assert set(data.keys()) == {'asterisk', 'Ierr', 'δair', 'g_u', 'Iref', 'γair', 'γself', 'A', 'g_l', 'nair', 'S', 'Q_l', 'V_u', 'Q_u', 'Iso', 'Mol', 'V_l', 'E_l', 'ν'}

# def test_load_into_HeteronuclearDiatomic():
    # t = lines.DiatomicCinfv()
    # t.load_from_hitran('data/hitran_data/CO/hitran_linelist.data')
    # assert len(t) == 849
    # assert set(t.keys()) == {'γair', 'v_u', 'Ae', 'g_u', 'v_l', 'ΔJ', 'γself', 'J_l', 'species', 'ν', 'nair', 'δair', 'E_l', 'g_l'}

