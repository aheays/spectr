## automatically fit one spectrum

## import spectr environment
from spectr.env import *

## make FitAbsorption object for this spectrum
o = spectrum.FitAbsorption(filename='../scans/miniPALS/2021_11_25_HCN_723Torr_mix+N2.0')

## fit HCN
o.fit('HCN', 'bands',fit_N= True, fit_pair= True,fit_intensity= True)
o.plot(fig=1)

## fit background to entire scan
o.fit()
o.plot(fig=2)

## print fitted results
print(o)

## save everything to disk
o.save('output_fit_one_spectrum')



