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
    linelist['HITRAN_HT_pX']     =  700
    linelist['HITRAN_HT_γ0'] = linelist['Γ']/2*10
    linelist['HITRAN_HT_Tref']  =  296
    linelist['HITRAN_HT_p']     =  0.1
    linelist['HITRAN_HT_X']     =  'H2'
    linelist['HITRAN_HT_n']     =  0
    linelist['HITRAN_HT_γ2']    =  0
    linelist['HITRAN_HT_δ0']    =  0
    linelist['HITRAN_HT_δp']    =  0
    linelist['HITRAN_HT_δ2']    =  0
    linelist['HITRAN_HT_νVC']   =  0.05
    linelist['HITRAN_HT_κ']     =  0
    linelist['HITRAN_HT_η']     =  0
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
    linelist['HITRAN_HT_νVC']   =  0.
    linelist['HT_Γ0'] = linelist['Γ']/2
    x0,y0 = linelist.calculate_spectrum(x,lineshape='hartmann-tran')
    x1,y1 = linelist.calculate_spectrum(x,lineshape='voigt')
    assert np.max(np.abs((y0-y1)/y0)) < 2e-3
    linelist['HT_Γ0'] = 0.
    linelist['HITRAN_HT_νVC']   =  0.
    linelist['HITRAN_HT_γ0']   =  0
    x0,y0 = linelist.calculate_spectrum(x,lineshape='hartmann-tran')
    x1,y1 = linelist.calculate_spectrum(x,lineshape='gaussian')
    i = y0>1e-4
    assert np.max(np.abs(y0[i]-y1[i])/y0[i]) < 2e-3

def test_compare_HT_implementations():
    ## compute spectrum using multiple methods and compare methods
    if show_plot:
        qfig(1)
    for iline,l in enumerate((
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':1e-10, 'Δ0':0, 'Δ2':0,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0001, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'H2', 'T':296, 'Γ0':0.002, 'Γ2':0.0001, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'SO2', 'T':296, 'Γ0':0.002, 'Γ2':0.0001, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':100, 'Γ0':0.002, 'Γ2':0.0001, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':1000, 'Γ0':0.002, 'Γ2':0.0001, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0002, 'Δ0':0.01, 'Δ2':0.002,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0002, 'Δ0':0.01, 'Δ2':0.004,'νVC':0, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0002, 'Δ0':0.01, 'Δ2':0.004,'νVC':0.1, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0002, 'Δ0':0.01, 'Δ2':0.004,'νVC':1, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':0.0002, 'Δ0':0.01, 'Δ2':0.004,'νVC':10, 'η':0,},
            {'species':'H2O', 'T':296, 'Γ0':0.002,   'Γ2':1e-10, 'Δ0':0, 'Δ2':0,'νVC':0, 'η':0},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':1e-10, 'Δ0':0, 'Δ2':0,'νVC':1e-2, 'η':0},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':1e-10, 'Δ0':0, 'Δ2':0,'νVC':1e-2, 'η':0.5},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':1e-3, 'Δ0':0.01, 'Δ2':0.01,'νVC':1, 'η':0},
            {'species':'H2O', 'T':296, 'Γ0':0.002, 'Γ2':1e-3, 'Δ0':0.01, 'Δ2':0.01,'νVC':1, 'η':0.5},
            )):
        x0 = 1000 + 0.1*iline
        S= 1
        m = database.get_species_property(l['species'],'mass')
        x = arange(x0-0.05,x0+0.05,1e-4)
        y0 = lineshapes.hartmann_tran(x,x0=x0,S=S,m=m,T=l['T'],νVC=l['νVC'],η=l['η'],Γ0=l['Γ0'],Γ2=l['Γ2'],Δ0=l['Δ0'],Δ2=l['Δ2'],method='python')
        y1 = lineshapes.hartmann_tran(x,x0=x0,S=S,m=m,T=l['T'],νVC=l['νVC'],η=l['η'],Γ0=l['Γ0'],Γ2=l['Γ2'],Δ0=l['Δ0'],Δ2=l['Δ2'],method='tran2014')
        assert np.all(np.abs(y0-y1) < 5e-3)
        if show_plot:
            line = plot(x,y0,color=newcolor(0))
            annotate_line(pformat(l),line=line,fontsize='x-small')
            plot(x,y1,color=newcolor(1),)
            plot(x,(y0-y1)*1000,color=newcolor(2))
    if show_plot:
        title('compare python HT profile with reference code of tran2014')
        legend({'color':newcolor(0),'label':'python'},
               {'color':newcolor(1),'label':'tran2014'},
               {'color':newcolor(2),'label':'(python-tran2014)*1000'},)
        ylim(-10,300)
        show()

