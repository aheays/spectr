import re

def SetOutput(InitialConditionFileName,lineset,NELEM):
	fin2 = open(InitialConditionFileName,'r')

	linenum = 0
	lineset += 3

	X = []
	Xadd = []

	threshold = 0

	fin1 = open('./out/depth-up.dat','r')
	for line in fin1:
		if linenum == lineset:
			element = line.split()
			numden = float(element[2])
			numdenabove = float(element[2])
		if linenum == lineset+1:
			element = line.split()
			numdenabove = float(element[2])
		linenum += 1
	fin1.close()

	linenum = 0

	fin1 = open('./out/depth-up.dat','r')
	for line in fin1:
	        if linenum == 2:
	        	element = line.split()
			for i in range(len(element)):
	        		if 'B' in element[i] and threshold == 0: 
	        			threshold = i-6
	        	outputnum = -10
	        	fin3 = open('./output.dat','r')
	        	for line2 in fin3:
	        		#if outputnum >= 0 and outputnum < threshold: Xadd.append(0.0)
	        		#if outputnum >= threshold:
				if outputnum >= 0:
	        			element2 = line2.split()
	        			sdummy = element2[NELEM+2].replace('D','e')
	        			try:
						dummy = float(sdummy)
					except ValueError:
						dummy = 0.0
					except IndexError:
				 		dummy = 0.0
					Xadd.append(dummy)
				outputnum += 1
			fin3.close()
		if linenum == lineset:
			element = line.split()
			frac = numdenabove/numden
			for i in range(len(element)):
				if i > 5 and i <= threshold + 5:
					dummy = float(element[i])
					Xav = Xadd[i-6]*frac + dummy*(1.0-frac)
					if Xav >= 1e-50: X.append(Xav)
					if Xav < 1e-50: X.append(0.0)
				if i > threshold + 5:
					X.append(0.0)
		linenum += 1

	fout = open('./output.dat','w')
        elenum = -10
	for line in fin2:
		if elenum < 0: fout.write(line)
		if elenum >= 0: 
			sdummy = '%.4e' % X[elenum]
			sdummy = sdummy.replace('e','D')
			fout.write(re.sub('\d\.\d{4}D[-\+]\d{2}',sdummy,line,count=1))
		elenum += 1

	fin1.close()
	fin2.close()
	fout.close()

#SetOutput('cond_initial/cond_initial_Case1.dat',75,16)
