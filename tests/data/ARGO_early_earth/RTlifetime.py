import numpy as np
import copy
import sys

def life (gcount):

	select = np.loadtxt('./outgas.dat',usecols=(1,),unpack=True,dtype=str)

	tf = np.zeros(len(select))

	Bulk = np.genfromtxt('./out/lifetime.dat',skip_header=2,dtype=str)
	
	Bulk = np.transpose(Bulk)

	p = Bulk[0,1:].astype(float)
	n = Bulk[2,1:].astype(float)
	Kzz = Bulk[3,1:].astype(float)
	h = Bulk[4,1:].astype(float)

	dh = np.array([])
	for i in range(len(h)-1):
		dh = np.append(dh,(h[i+1] - h[i]))
	dh = np.append(dh,dh[len(dh)-1])

	fout = open('./out/surfacelife.dat','w')
	fout.write('Species    t[s]')
 	for i in range(6,len(Bulk)):
		species = Bulk[i,0]
		nuchem = Bulk[i,1:].astype(float)
		nuk = 0.0
		desnuk = 0.0
		for k in range(len(nuchem)):
			tchem = 1e20
			sign = 'none'
			if nuchem[k] > 1e-20: sign = 'plus'
			if nuchem[k] < -1e-20: 
				sign = 'minus'
			if nuchem[k] > 1e-20 and sign == 'plus': tchem = 1.0/nuchem[k]
			if nuchem[k] < -1e-20 and sign == 'minus': tchem = -1.0/nuchem[k]
			if tchem > 1e20 or sign == 'none': tchem = 1e20
			tsum = dh[:k]/Kzz[:k]
			tdyn = 0.25*h[k]*np.sum(tsum)
			if sign == 'plus': 
				nuk += ((tchem + tdyn)*n[0]/n[k])**-1.0
				desnuk += 0.0
			if sign == 'minus': 
				nuk -= ((tchem + tdyn)*n[0]/n[k])**-1.0
				desnuk += ((tchem + tdyn)*n[0]/n[k])**-1.0
		ltime = 1e20
		if nuk > 1e-20 or nuk < -1e-20: ltime = 1.0/nuk
		if desnuk > 1e-20: desltime = 1.0/desnuk
		if desnuk <= 1e-20: desltime = 1e20
		fout.write('\n')
		fout.write(species + '   ' + ('%.3e' % ltime))
		for j in range(len(select)):
			if species == select[j]: tf[j] = desltime
			
	fout.close()
	if gcount > 1:
		fin = open('resource.dat','r')
		fout = open('reservoir.dat','w')

		for line in fin:
			if 'XXXXXXXXXX' in line:
				for i in range(len(select)):
					if 'G'+select[i] in line.split():
						stf = '%.4e' % tf[i]
						stf = stf.replace('e','D')
						line = line.replace('XXXXXXXXXX',stf)
			fout.write(line)

		fin.close()
		fout.close()

def start():
	fin = open('resource.dat','r')
	fout = open('reservoir.dat','w')

	for line in fin:
		line = line.replace('XXXXXXXXXX','0.0000D+00')
		fout.write(line)

	fin.close()
	fout.close()
