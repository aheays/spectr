## fit experimental CS2 absorption by modifying a HITRAN linelist
from spectr import *

## frequency range
νbeg,νend = 2194.15,2200

## HITRAN linelist spectrum
S296K_frac_max = 2e-4
linelist = lines.LinearTriatom(
    name='linelist',
    load_from_file='data/CS2_HITRAN_2021.h5',
    Zsource='HITRAN',
    Teq=295,
    pself=34,
)
linelist.limit_to_match(ν_min=νbeg,ν_max=νend,S296K_min=np.max(linelist['S296K'])*S296K_frac_max)

## optimise column density -- common to all lines
linelist.set_parameter('Nself',P(2.43466013568e+19/2, True,1e+18,7.7e+17)) 

## optimise self-broadening coefficient of each line initially set to
## 0.01 cm-1.atm-1.HWHM
linelist.set('γ0self',0.01,vary= True,step=1e-4)

## load experimental data
experiment = spectrum.Experiment()
experiment.set_spectrum_from_opus_file('data/CS2_experimental_spectrum.opus', xbeg=νbeg,xend=νend)

## define model that fits the spectrum to exp
model = spectrum.Model('model',experiment)
model.interpolate(1e-3)
model.add_intensity_spline(knots=[(1900, P(0.238010785774,False,0.0001,nan)), (2500, P(0.213998002,False,0.0001,nan))],)
model.add_absorption_lines(lines=linelist)
model.apodise()

## optimise the model
model.optimise(normalise_suboptimiser_residuals= True)

## makes a new input file
model.save_input(filename='t.py')

## print the linelist with fitted broadening coefficients
print(linelist.format())

## plot the fit
qfig(1)
model.plot()

## plot fitted broadening coefficients
qfig(2)
linelist.plot('J_l','γ0self',)
ylim(0,1)

show()

