## fit experimental CS2 absorption by modifying a HITRAN linelist
from spectr import *

## frequency range
νbeg,νend = 2194.15,2200

## HITRAN linelist spectrum
S296K_frac_max = 2e-4
lines_CS2 = lines.Generic(load_from_filename='data/HITRAN_linelist_CS2_2021-01-21.h5')
lines_CS2.Zsource = 'HITRAN'                       
lines_CS2.limit_to_match(ν_min=νbeg,ν_max=νend,S296K_min=np.max(lines_CS2['S296K'])*S296K_frac_max)
lines_CS2['γself'] = 0.1
lines_CS2.set('γself',0.1,vary= True,step=1e-2)

## load experimental data
exp = spectrum.Experiment()
exp.set_spectrum_from_opus_file('data/CS2_experimental_spectrum.opus', xbeg=νbeg,xend=νend)

## define model that fits the spectrum to exp
mod = spectrum.Model('mod',exp)
mod.interpolate(1e-3)
mod.add_intensity_spline(knots=[(1900, P(0.238010785774,False,0.0001,nan)), (2500, P(0.213998002,False,0.0001,nan))],)
mod.add_absorption_lines(
    lines=lines_CS2,τmin=1e-5,
    Teq=295,
    Nself_l=P(1.72501277476e+19,False,1e+14,nan),
    Pself=convert(0.258,'Torr','Pa'),
    )
mod.apodise()

## optimise the model
mod.optimise(normalise_suboptimiser_residuals= True,)

## makes a new input file
mod.save_input(filename='data/t0.py')

## print the linelist with fitted broadening coefficients
print(lines_CS2.format())

## plot the fit
qfig(1)
mod.plot()
