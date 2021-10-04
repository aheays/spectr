from pytest import approx
import numpy as np

from spectr import lines
from spectr import levels
from spectr import plotting
from spectr import hitran
from spectr import convert
from spectr import tools

show_plots = True

def test_construct():
    t = lines.Generic()
    # assert t.name == 'generic_line'

def test_assignment():
    t = lines.Generic(name='ddd')
    assert t.name == 'ddd'
    t.description = 'fff'
    t['ν'] = [1,2]
    assert all(t['ν'] == [1,2])
    t['ν'] = 100.
    t['E_l'] = 150.

def test_infer_with_level_keys():
    t = lines.Generic(E_l=[100],E_u=[150])
    assert t['E_l'] == [100.]
    assert t['E_u'] == [150.]
    assert t['ν'] == [50.]
    t = lines.Generic(data={'ν':[100],'E_u':[150]})
    assert t['E_l'] == [50.]
    t = lines.Generic(data={'E_l':[100],'E_l:unc':[0.5],'E_u':[150],'E_u:unc':[0.2]})
    t.set('E_l','unc',0.5)
    t.set('E_u','unc',0.2)
    assert t['ν'] == [50.]
    assert t['ν','unc'] == [approx(np.sqrt(0.5**2+0.2**2))]

def test_load_lines():
    t = lines.Generic()
    t.load('data/test_lines')
    assert abs(t['ν'][0]-38358.664)<1e-2
    assert len(t)==32

def test_calculate_plot_spectrum():
    t = lines.Generic()
    t.load('data/test_lines')
    x,y = t.calculate_spectrum(xkey='ν',ykey='f')
    assert len(x)==10000
    assert sum(y) == approx(0.00903753325337632)
    t.plot_spectrum(xkey='ν',ykey='f')
    if show_plots:
        plotting.plt.title('test_calculate_plot_spectrum')
        plotting.show()

def test_get_upper_lower_levels():
    t = lines.Generic()
    t.load('data/test_lines')
    u = t.get_upper_level()
    l = t.get_lower_level()
    assert len(u) == 32
    assert len(l) == 32

def test_spectrum_calc_against_hapi():
    νbeg,νend = 2127,2141
    Teq = 296
    ## calc with hapi
    xhapi,yhapi = hitran.calc_spectrum(
        species='CO',
        data_directory='data/hitran_data/CO/[12C][16O]/',
        T=Teq,p=1,
        νbeg=νbeg,νend=νend,νstep=0.001,
        table_name='hitran_linelist',
        make_plot=False,)
    ## calc with spectr
    l = hitran.get_lines('¹²C¹⁶O')
    l['Teq'] = Teq
    l['pair'] = convert.units(1,'atm','Pa')
    l.limit_to_match(min_ν=νbeg,max_ν=νend)
    xspectr,yspectr = l.calculate_spectrum(np.arange(νbeg,νend,0.001),ykey='S')
    from scipy import integrate
    inthapi = tools.integrate(xhapi,yhapi)
    intspectr = tools.integrate(xspectr,yspectr)
    if show_plots:
        ax = plotting.qax()
        ax.plot(xhapi,yhapi,label=f'hapi: {inthapi:0.3e}')
        ax.plot(xspectr,yspectr,label=f'spectr:  {intspectr:0.3e}')
        ax.plot(xspectr,np.abs(yhapi[:-1]-yspectr),label=f'hapi-spectr: {(intspectr-inthapi)/inthapi:0.3e}')
        ax.legend()
        plotting.plt.title('test_spectrum_calc_against_hapi')
        plotting.show()
    ## not a very stringent threshold!
    assert np.all(np.abs(yhapi[:-1]-yspectr) < 3e-20)




