############################################
## simulate H2O spectrum from HITRAN data ##
############################################

from spectr import *

## load HIRAN linelist
νbeg,νend = 1472,1485           # freq range to use
S296K_frac_max = 1e-2           # ignore lines this far below the strongest
linelist = lines.Generic(load_from_file='data/H2O_HITRAN_2021.h5')
linelist.limit_to_match(ν0_min=νbeg,ν0_max=νend)

linelist['Zsource']='HITRAN' # source of partition function (hapi)
linelist['Teq'] = 300         # equilibrium temperature
linelist['L'] = 10            # optical path length (m)
linelist['pself'] = convert.units(0.01,'Torr','Pa') # pressure
linelist['pair'] = convert.units(100,'Torr','Pa')   # pressure

## define model spectrum
mod = spectrum.Model()
mod.add_intensity(intensity=1)
mod.add_absorption_lines(lines=linelist)
mod.convolve_with_gaussian(0.1)
mod.add_noise(rms=0.002)

## plot model spectrum
qfig(1)
mod.get_spectrum(x=arange(νbeg,νend,1e-4))
mod.plot()
show()



