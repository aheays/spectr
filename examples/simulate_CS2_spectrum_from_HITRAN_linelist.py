## simulate CS2 spectrum from HITRAN data

from spectr import *

## load HITRAN linelist
lexp = lines.Generic(load_from_filename='data/HITRAN_linelist_CS2_2021-01-21.h5')
lexp.Zsource = 'HITRAN'         # use HITRAN partition function -- downloaded iwth hapy
lexp.limit_to_match(
    ν_min=1472, ν_max=1485,     # limit to lines between 1472 and 1485 cm-1
    S296K_min=1e-20,            # limit to liens with S296K > 1e-20 cm2.cm-1
)

## define model of the spectrum
model = spectrum.Model('model')
model.add_intensity(intensity=1) # add background intensity
model.add_absorption_lines(      # add absorption 
    lines=lexp,                  # the HITRAN linelist
    τmin=1e-5,                   # only include lines with integrated optical depth exceeding 1e-5 cm
    Teq=295,                     # tempeature 295K
    Nself_l=1e16,                # column density 1e16 cm-2
    Pself=convert(100,'Torr','Pa'), # self-broadening 100 Torr
)
model.convolve_with_gaussian(width=0.05) # model a 0.05cm-1 FWHM Gaussian instrument broadening
model.add_noise(rms=0.005)               # simulate noise

## calculate a model spectrum
y = model.get_spectrum(
    x=np.arange(lexp['ν'].min(),lexp['ν'].max(),0.002) # the wavenumber grid to calculate on
)

## plot the spectrum
qfig(1)
model.plot()
show()

