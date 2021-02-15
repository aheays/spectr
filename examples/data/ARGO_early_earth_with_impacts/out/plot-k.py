import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

h, k_up1, k_up2 = np.loadtxt('./Reactions/Kup.dat',skiprows=3,usecols=(1,5799+1,5800+1),unpack=True)

h *= 1e-5

k_up = k_up1 + k_up2

plt.xscale('log')

plt.plot(k_up,h,'k-')

plt.savefig('./N2O-rates.pdf',bbox_inches='tight')
