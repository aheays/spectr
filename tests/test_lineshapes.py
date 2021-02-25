from pytest import raises,approx,fixture

from spectr import *

show_plot = False 

@fixture
def linelist():
    ## line data including Voigt and Hartmann-Tran pressure broadening
    linelist = lines.Generic()
    linelist['ν']  =  [2001,2001.2]
    linelist['σ']  =  [1,1]
    linelist['Γ']    =  [0.01,0.02]
    linelist['species']     =  'H2O'
    linelist['Teq']     =  296
    linelist['HT_HITRAN_γ0'] = linelist['Γ']/2*10
    linelist['HT_HITRAN_Tref']  =  296
    linelist['HT_HITRAN_p']     =  0.1
    linelist['HT_HITRAN_X']     =  'H2'
    linelist['HT_HITRAN_n']     =  0
    linelist['HT_HITRAN_γ2']    =  0
    linelist['HT_HITRAN_δ0']    =  0
    linelist['HT_HITRAN_δp']    =  0
    linelist['HT_HITRAN_δ2']    =  0
    linelist['HT_HITRAN_νVC']   =  0.05
    linelist['HT_HITRAN_κ']     =  0
    linelist['HT_HITRAN_η']     =  0
    return linelist
    
def test_plot_lineshapes(linelist):
    if show_plot:
        ax = plotting.qax(1)
        for lineshape in (
                'gaussian',
                'lorentzian',
                'voigt',
                'hartmann-tran',
                ):
            x,y = linelist.calculate_spectrum(
                np.arange(linelist['ν'].min()-0.2,linelist['ν'].max()+0.2,1e-3),
                lineshape=lineshape, nfwhmG=None, nfwhmL=None,)
            ax.plot(x,y,label=f'{lineshape:20} {integrate.trapz(y,x):10.5e}')
        plotting.legend()
        plotting.show()

def test_plot_HT_limit_equivalences(linelist):
    x = np.arange(linelist['ν'].min()-0.2,linelist['ν'].max()+0.2,1e-4)

    linelist['HT_HITRAN_νVC']   =  0.
    x0,y0 = linelist.calculate_spectrum(x,lineshape='hartmann-tran')
    x1,y1 = linelist.calculate_spectrum(x,lineshape='voigt')
    assert np.max(np.abs((y0-y1)/y0)) < 2e-5

    linelist['HT_HITRAN_νVC']   =  0.
    linelist['HT_HITRAN_γ0']   =  0
    x0,y0 = linelist.calculate_spectrum(x,lineshape='hartmann-tran')
    x1,y1 = linelist.calculate_spectrum(x,lineshape='gaussian')
    i = y0>1e-4
    assert np.max(np.abs(y0[i]-y1[i])/y0[i]) < 2e-3
