## 13C18O a'(5)-A(0) SOLEIL absorption spectrum to a band model

from spectr import *

lower = viblevel.VibLevel(name='lower',species='[13C][18O]')
lower.add_level(name='X.1Σ+(v=0)',Tv=1031.055619,Bv=1.746408199,Dv=5.0488146e-6,Hv=4.35471e-12,)
 
upper = viblevel.VibLevel(name='upper',species='[13C][18O]')
upper.add_level(name='A.1Π(v=0)',Tv=64762.750+1031.055619,Bv=1.4574165,Dv=6.0830e-6)
upper.add_level(name='ap.3Σ+(v=5)',
    Tv=P(61981.6505242, True,0.001,0.003),
    Bv=P(1.13803172462, True,1e-05,9e-06),
    Dv=P(4.00075877615e-06, True,1e-09,1.3e-07),
    λv=P(-1.22033884134, True,0.001,0.0012),
    γv=P(0.0013189545394, True,1e-05,0.00011),
)
upper.add_LS_coupling('A.1Π(v=0)','ap.3Σ+(v=5)',ηv=1)


transition = viblevel.VibLine('transition',upper,lower,J_l=range(31),)
transition.add_transition_moment('A.1Π(v=0)','X.1Σ+(v=0)',μv=1)
transition.construct()

transition.line['Zsource'] = 'self'
transition.line.set_parameters(
    Teq=P(235.791023479, True,1,2.3),
    Nself=P(1.43053973766e+20, True,1e+17,1e+18),)

experiment = spectrum.Experiment('experiment',)
experiment.set_spectrum_from_soleil_file(
    filename='data/170216-Jmoy13-538A-13C18O-73mb-coRa-RP.TXT.wavenumbers.h5',
    xbeg=60700, xend=61000,)
model = spectrum.Model('model',experiment)
model.add_intensity_spline(knots=[(60600, P(0.0772604178868, True,0.0001,6.1e-05)), (61100, P(0.115131675836, True,0.0001,6.1e-05))],)
model.add_absorption_lines(lines=transition.line,lineshape='gaussian')
model.convolve_with_soleil_instrument_function()

model.optimise(method='lm')
model.save_input(filename='t.py')
model.save_to_directory('td')

# qfig(1)
# model.plot(plot_labels=False)
# show()

