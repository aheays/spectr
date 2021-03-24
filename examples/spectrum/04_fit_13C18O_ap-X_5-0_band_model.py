## fit experimental CS2 absorption by modifying a HITRAN linelist
from spectr import *

lower = viblevel.VibLevel(name='lower',species='[13C][18O]')
lower.add_level(name='X.1Σ+(v=0)',Tv=1031.055619,Bv=1.746408199,Dv=5.0488146e-6,Hv=4.35471e-12,)
 
upper = viblevel.VibLevel(name='lower',species='[13C][18O]')
upper.add_level(name='A.1Π(v=0)',Tv=64762.750+1031.055619,Bv=1.4574165,Dv=6.0830e-6)
upper.add_level(name='ap.3Σ+(v=5)',
    Tv=P(61981.6505242,False,0.001,0.003),
    Bv=P(1.13803172462,False,1e-05,9e-06),
    Dv=P(4.00075877615e-06,False,1e-09,1.3e-07),
    λv=P(-1.22033884134,False,0.001,0.0012),
    γv=P(0.0013189545394,False,1e-05,0.00011),
)
upper.add_LS_coupling('A.1Π(v=0)','ap.3Σ+(v=5)',ηv=1)


transition = viblevel.VibLine('transition',upper,lower,J_l=range(31),)
transition.add_transition_moment('A.1Π(v=0)','X.1Σ+(v=0)',μv=1)
transition.construct()

linelist = transition.rotational_line
linelist['Zsource'] = 'self'
linelist.set_parameters(
    Teq=P(235.791023479,False,1,2.3),
    Nself=P(1.43053973766e+20,False,1e+17,1e+18),
)

experiment = spectrum.Experiment(
    'experiment',
)
experiment.set_spectrum_from_soleil_file(
    filename='data/170216-Jmoy13-538A-13C18O-73mb-coRa-RP.TXT.wavenumbers.h5',
    xbeg=60700, xend=61000,)
model = spectrum.Model('model',experiment)
# model.interpolate(1e-3)
model.add_intensity_spline(knots=[(60600, P(0.0772604178868,False,0.0001,6.1e-05)), (61100, P(0.115131675836,False,0.0001,6.1e-05))],)
# model.add_intensity(0.1)
model.add_absorption_lines(lines=linelist,lineshape='gaussian')
model.convolve_with_soleil_instrument_function()

# ## optimise the model
model.optimise(method='lm')

## makes a new input file
model.save_input(filename='t.py')
# model.save_to_directory('td')

# # # ## print the linelist with fitted broadening coefficients
# # # print(linelist.format())

## plot the fit
qfig(1)
model.plot(plot_labels=False)
show()

# print( transition.rotational_line[:10])
