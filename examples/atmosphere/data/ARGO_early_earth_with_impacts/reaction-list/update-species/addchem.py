import numpy as np
import re

def GetThermo(species):

	if species == 'Cl2S': species = 'SCL2'
	if species == 'Cl2S2': species = 'S2CL2'
	if species == 'ClCO': species = 'COCL'
	if species == 'ClS': species = 'SCL'
	if species == 'ClS2': species = 'S2CL'
	if species == 'HS': species = 'SH'
	if species == 'HOCl': species = 'HOCL'
	if species == 'OCS': species = 'COS'
	if species == 'SO2Cl2': species = 'SO2Cl2'
        if species == 'H2CS': species = 'CH2=S'

	fdata = open('BURCAT.THR','r')
	
	count = 0
	a = np.full(14,' 0.0000D+00')
	switch = False
	for line in fdata:
		if switch:
			newline = re.sub(r'(\d)-(\d)',r'\1 -\2',line)
			element = newline.split()
			end = element[-1]
			if end == '2': 
				a[0] = '%.4e' % (float(element[0]))
				a[1] = '%.4e' % (float(element[1]))
				a[2] = '%.4e' % (float(element[2]))
				a[3] = '%.4e' % (float(element[3]))
				a[4] = '%.4e' % (float(element[4]))
			if end == '3':
				a[5] = '%.4e' % (float(element[0]))
				a[6] = '%.4e' % (float(element[1]))
				a[7] = '%.4e' % (float(element[2]))
				a[8] = '%.4e' % (float(element[3]))
				a[9] = '%.4e' % (float(element[4]))
			if end == '4':
				a[10] ='%.4e' % (float(element[0]))
				a[11] ='%.4e' % (float(element[1]))
				a[12] ='%.4e' % (float(element[2]))
				a[13] ='%.4e' % (float(element[3]))
				switch = False
		if count > 150:
			element = line.split()
			try:
				end = element[-1]
			except IndexError:
				end = 'zero'
			if end == '1':
				scheck = element[0]
				if element[0] == species:
					switch = True
	
		count += 1

	fdata.close()

	for i in range(len(a)):
		element = list(a[i])
		start = element[0]
		if start != '-' and 'D' not in a[i]: a[i] = ' ' + a[i]
		a[i] = a[i].replace('e','D')

	return a

fin = open('cond_source.dat','r')
fout = open('cond_initial.dat','w')

count = 1
for line in fin:
	element = line.split()
	if element[-1] == 'NOINFO':
		species = element[1]
		a = GetThermo(species)
		print a
		line = line.replace('0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  0.0000D+00  NOINFO','0.0000D+00 ' + a[0] + ' ' + a[1] + ' ' + a[2] + ' ' + a[3] + ' ' + a[4] + ' ' + a[5] + ' ' + a[6] + ' ' + a[7] + ' ' + a[8] + ' ' + a[9] + ' ' + a[10] + ' ' + a[11] + ' ' + a[12] + ' ' + a[13] + '  STINFO')
	if 'XXX' in line:
		print line
		if count < 10: line = line.replace('XXX','  '+str(count))	
		if count >= 10 and count < 99: line = line.replace('XXX',' '+str(count))	
		if count > 99: line = line.replace('XXX',str(count))	
		count += 1
	fout.write(line)

fin.close()
fout.close()
