from spectr.env import *


experiment = spectrum.Experiment('experiment',filename='../scans/2021_11_30_benzene_10.4Torr.0')
background = spectrum.Experiment('background',filename='../scans/2021_11_30_bcgr.0')
# cross_section = hitran.load_spectrum('../hitran/C6H6_296.2K-5.2Torr_650.0-1540.0_110.xsc')
cross_section = hitran.load_spectrum('../hitran/C6H6_298.0_760.0_600.0-6500.0_09.xsc')

model = spectrum.Model('model',experiment=experiment,
                       # xbeg=1350,xend=1600,
                       )

model.add_arrays(
    background.x,
    background.y,
    scale      = P(1.02618005365,False,1e-05,3.4e-05),
)

model.add_absorption_cross_section(
    x=cross_section['x'],
    y=cross_section['y'],
    N          = P(8.55274662962e+18,False,1e+15,3.8e+15)
)

model.optimise()


qfig(2)
model.plot()
model.save_input('t0.py')
