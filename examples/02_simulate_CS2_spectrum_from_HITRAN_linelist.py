## simulate CS2 spectrum from HITRAN data
from spectr import *

## sample cell
length = 10                     # optical path length (m)
T = 300                         # K
P_CS2 = convert(0.001,'Torr','Pa') # Pa
P_air = convert(100,'Torr','Pa')  # Pa
N_CS2 = convert(P_CS2*length/(constants.Boltzmann*T), 'm-2','cm-2') # N = Pl/kT (cm-2)

## spectrometer
instrument_width = 0
noise_level = 0

## load HIRAN linelist
νbeg,νend = 1472,1485           # freq range to use
S296K_frac_max = 1e-2           # ignore lines this far below the strongest
lines_CS2 = lines.Generic(load_from_filename='~/data/species/CS2/infrared/linelists/HITRAN_2020-12-08/lines.h5')
lines_CS2.Zsource = 'HITRAN'                       
lines_CS2.limit_to_match(ν_min=νbeg, ν_max=νend, S296K_min=np.max(lines_CS2['S296K'])*S296K_frac_max)

## define model spectrum
mod = spectrum.Model('mod')
mod.add_intensity(intensity=1)
mod.add_absorption_lines(
    lines=lines_CS2,
    τmin=1e-5,
    use_cache= True,
    Teq=T,
    Nself_l=N_CS2,
    Pself=P_CS2,
    Pair=P_air,
)
mod.convolve_with_gaussian(width=instrument_width)
mod.add_noise(rms=noise_level)

## plot model spectrum
qfig(1)
mod.get_spectrum(x=np.arange(νbeg,νend,1e-4))
mod.plot()
show()



