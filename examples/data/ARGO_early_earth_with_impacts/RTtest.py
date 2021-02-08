#This is a python file.

import sys
import numpy as np

def ResetCondInitial(file1,file2):
	fout = open('./transition.dat','w')

	fin = open(file2,'r')

	count = 0

	for line in fin:
		if count < 10 and count != 1: fout.write(line)
		if count == 1: fout.write('c transition.dat     \n')
		count += 1

	fin.close()

	Bulk1 = np.genfromtxt(file1,skip_header=10,dtype=str)
	Bulk2 = np.genfromtxt(file2,skip_header=10,dtype=str)

	for i in range(len(Bulk1)):
		if i > 0: fout.write('\n')
		if len(Bulk1[i,0]) == 1: fout.write('   ' + Bulk1[i,0])
		if len(Bulk1[i,0]) == 2: fout.write('  ' + Bulk1[i,0])
		if len(Bulk1[i,0]) == 3: fout.write(' ' + Bulk1[i,0])
		fout.write(' ' + Bulk1[i,1])
		if Bulk1[i,2] == '1' or Bulk1[i,2] == '0':
			for j in range(11-len(Bulk1[i,1])): fout.write(' ')
		if Bulk1[i,2] == '-1':
			for j in range(10-len(Bulk1[i,1])): fout.write(' ')
		fout.write(Bulk1[i,2])
		for j in range(3,18):
			fout.write('  '+Bulk1[i,j])
		fout.write('  ')
		if Bulk1[i,18] != '0.0000D+00': fout.write(Bulk1[i,18])
		if Bulk1[i,18] == '0.0000D+00': fout.write(Bulk2[i,18])
		for j in range(19,len(Bulk1[i])):
			if len(Bulk1[i,j]) == 11: fout.write(' ' + Bulk1[i,j])
			if len(Bulk1[i,j]) < 11: fout.write('  ' + Bulk1[i,j])

	fout.close()

	

def DoesItConverge(ConvergenceCriterion,ConvergenceTimescale,VZ_Order,muav,LOGG):

	count = 0

	f1 = './out/old-depth.dat'
	f2 = './out/depth.dat'
	
	array1 = np.genfromtxt(f1,skip_header=2,dtype=str)
	array2 = np.genfromtxt(f2,skip_header=2,dtype=str)
	
	array1 = np.transpose(array1)
	array2 = np.transpose(array2)
	
	z = array1[4,1:].astype(float)
	n = array1[2,1:].astype(float)
	n0 = np.max(n)
	T = array1[1,1:].astype(float)
	if VZ_Order == 1: vz = array1[3,1:].astype(float)
	if VZ_Order == 2: Kzz = array1[3,1:].astype(float)
	g = 10.0**LOGG

	epsilon = ConvergenceCriterion

	H = 8.254e7*T/(muav*g)

	dz = np.diff(z)

	dz = np.append(dz,dz[len(dz)-1])
	
	if VZ_Order == 1: k_tau = dz*n0/(vz*n)
	if VZ_Order == 2: k_tau = H*dz*n0/(Kzz*n)

	tau = np.full((len(array1),len(z)),1.0e20)

	for i in range(6,len(array1)):
		mix1 = array1[i,1:].astype(float)
		mix2 = array2[i,1:].astype(float)
		dmix = np.abs(mix1-mix2)
		species = array1[i,0]
		for j in range(len(dmix)):
			if mix1[j] > 0.0 and dmix[j] > 0.0:
				tau[i,j] = np.max([k_tau[j]*mix1[j]/(epsilon*dmix[j]),k_tau[j]*1e-9/dmix[j]])
			if mix1[j] == 0.0 and dmix[j] > 0.0:
				tau[i,j] = np.max([k_tau[j]*mix2[j]/(epsilon*dmix[j]),k_tau[j]*1e-9/dmix[j]])
			if dmix[j] == 0.0:
				tau[i,j] = 1.0e20
			#if species == 'SO2': print tau[i,j]
	taumin = 1.0e20
        zmin = 0.0
        specmin = 'no species'
	for i in range(len(tau)):
		species = array1[i,0]
		for j in range(len(tau[i])):
			if tau[i,j] < taumin: 
				taumin = tau[i,j]
				zmin = z[j]
				specmin = species

	print '\nThe convergence timescale is = ' + ('%.3e' % (taumin*3.17e-8)) + ' years. The offending species is ' + specmin + ' at height ' + ('%.3e' % (zmin*1e-5)) + ' km.\n'
		
	if taumin > ConvergenceTimescale: return True
	if taumin <= ConvergenceTimescale: return False

#print DoesItConverge(1e-2,1e10,2,2.33,3.33)
