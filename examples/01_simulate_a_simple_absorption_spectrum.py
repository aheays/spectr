## simulate a simple absorption spectrum

from spectr import *

## make a linelist
linelist = lines.Generic('linelist')
linelist.load_from_string('''

species = 'H2O'
Teq = 300

ν  |τ  |Γ
100|0.1|1
110|0.5|1
115|2  |3

''')

print( linelist)

## get model spectrum
mod = spectrum.Model('mod')
mod.add_intensity(intensity=1)
mod.add_absorption_lines(lines=linelist)
mod.get_spectrum(x=np.arange(90,130,1e-2))

## plot
qfig(1)
mod.plot()
show()
