import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt

#Plot the file in terms of height if True, pressure if False
height = True

#If false, plots all above the threshold mixing ratio
OnlySelect = True
threshold = 1e-6

#The mixing ratio bounds, if negative, mixing ratio bounds are automatically determined
xmin=-1.0
xmax = -1.0

select = np.array(['X'])

#Files are saved with filename in the folder ./out/Figures/
#For Earth Chemistry
select = np.array(['CH4','O3','H2O','N2','O2','CO2','N2O','NO','NO2'])
filename = 'test-earth'

#For Venus Chemistry:
#select = np.array(['CO','O2','H2S','SO','SO2','H2O','OCS'])
#filename = 'test-venus'
#select = np.array(['S','S2','S3','S4','S5','S6','S7','S8'])
#filename = 'test-venus-allo'


def plotnow(threshold,pmax,pmin,hmax,height,xmin=-1.,xmax=-1.,filename='test'):
	filename = filename + '.pdf'
	plt.xscale('log')
	if height == False: plt.yscale('log')
	if xmin <= 0. or xmax <= 0.:
		plt.xlim([threshold*1e-9,1.0])
	if xmin > 0. and xmax > 0.:
		plt.xlim([xmin,xmax])
	if height == False: plt.ylim([pmax,pmin])
	if height == True: plt.ylim([0.0,hmax*1e-5])
	plt.xlabel('Mixing Ratio')
	if height == True: plt.ylabel('Height (km)')
	if height == False: plt.ylabel('Pressure (bar)')
	plt.legend(loc='upper left', bbox_to_anchor=(1.0,1.0))
	plt.savefig('./out/Figures/'+filename,bbox_inches='tight')


#Choose which file to plot. depth.dat has the final output profiles.

fname = './out/depth.dat'
#fname = './out/old-depth.dat'
#fname = './out/depth-up.dat'
#fname = './out/depth-down.dat'

Bulk = np.genfromtxt(fname,dtype=str,skip_header=2)

Bulk = np.transpose(Bulk)

p = Bulk[0,1:].astype(float)
pmax = np.max(p)
pmin = np.min(p)

h = Bulk[4,1:].astype(float)
hmax = np.max(h)

cstyle = ['r','b','g','k','c']
lstyle = ['-','--',':']

c1 = 0
c2 = 0
for i in range(len(Bulk)):
	species = Bulk[i,0]
	mixrat = Bulk[i,1:] .astype(float)
	mixmax = np.max(mixrat)
	if not OnlySelect:
		if i > 5 and 'B' not in species and 'J' not in species and (mixmax > threshold or species in select):
			if height == False: plt.plot(mixrat,p,cstyle[c1]+lstyle[c2],label=species)
			if height == True: plt.plot(mixrat,h*1e-5,cstyle[c1]+lstyle[c2],label=species)
			c1 += 1
			if c1 > 4:
				c1 = 0
				c2 += 1
			if c2 == 2:
				c1 = 0
				c2 = 0 
				plotnow(threshold,pmax,pmin,hmax,height,xmin=xmin,xmax=xmax,filename=filename)
	if OnlySelect:
		if i > 5 and species in select:
			if height == False: plt.plot(mixrat,p,cstyle[c1]+lstyle[c2],label=species)
			if height == True: plt.plot(mixrat,h*1e-5,cstyle[c1]+lstyle[c2],label=species)
			c1 += 1
			if c1 > 4:
				c1 = 0
				c2 += 1
			if c2 == 2:
				c1 = 0
				c2 = 0 
				plotnow(threshold,pmax,pmin,hmax,height,xmin=xmin,xmax=xmax,filename=filename)

plotnow(threshold,pmax,pmin,hmax,height,xmin=xmin,xmax=xmax,filename=filename)
		 
