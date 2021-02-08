import numpy as np
import sys
import os

def combine(ptfilename):
	BulkUp = np.genfromtxt('./out/depth-up.dat',skip_header=2,dtype=str)
	BulkDownRev = np.genfromtxt('./out/depth-down.dat',skip_header=2,dtype=str)
	
	BulkUp = np.transpose(BulkUp)
	BulkDownRev = np.transpose(BulkDownRev)
	
	BulkDown = []
	
	for i in range(len(BulkDownRev)):
		BulkDummy = BulkDownRev[i,:0:-1]
		BulkDummy = np.append(BulkDownRev[i,0],BulkDummy)
		BulkDown.append(BulkDummy)
	
	BulkDown = np.array(BulkDown)
	
	fout = open('./out/depth.dat','w')
	
	fout.write('ARGO Output for the pressure-temperature profile of ' + ptfilename + ' using the STAND 2019 Network\n')
	
	switch = False
	
	for j in range(len(BulkUp[0])):
		fout.write('\n')
		for i in range(len(BulkUp)):
			if j == 0:
				if i < 6 or 'B' not in BulkUp[i,j]:
					fout.write(BulkUp[i,j] + '   ')
			if j > 0:
				if i < 6 or 'B' not in BulkUp[i,0]:
					xUp = float(BulkUp[i,j])
					xDown = float(BulkDown[i,j])
					fout.write('%.3e' % xDown)
					fout.write('  ')
	fout.close()

def combineLife(ptfilename):
        BulkUp = np.genfromtxt('./out/lifetime-up.dat',skip_header=2,dtype=str)
        BulkDownRev = np.genfromtxt('./out/lifetime-down.dat',skip_header=2,dtype=str)

        BulkUp = np.transpose(BulkUp)
        BulkDownRev = np.transpose(BulkDownRev)

        BulkDown = []

        for i in range(len(BulkDownRev)):
                BulkDummy = BulkDownRev[i,:0:-1]
                BulkDummy = np.append(BulkDownRev[i,0],BulkDummy)
                BulkDown.append(BulkDummy)

        BulkDown = np.array(BulkDown)

        fout = open('./out/lifetime.dat','w')

	fout.write('ARGO Chemical Lifetimes for the pressure-temperature profile of '+ptfilename+' using the STAND 2019 Network\n\n')

        switch = False

        for j in range(len(BulkUp[0])):
                fout.write('\n')
                for i in range(len(BulkUp)):
                        if j == 0:
                                if i < 6 or 'B' not in BulkUp[i,j]:
                                        fout.write(BulkUp[i,j] + '   ')
                        if j > 0:
                                if i < 6 or 'B' not in BulkUp[i,0]:
                                        xUp = float(BulkUp[i,j])
                                        xDown = float(BulkDown[i,j])
					xAv = 0.5*(xUp + xDown)
                                        fout.write('%.3e' % xAv)
                                        fout.write('  ')
        fout.close()

#def combine():
#	BulkUp = np.genfromtxt('./out/depth-up.dat',skip_header=2,dtype=str)
#	BulkDownRev = np.genfromtxt('./out/depth-down.dat',skip_header=2,dtype=str)
#	
#	BulkUp = np.transpose(BulkUp)
#	BulkDownRev = np.transpose(BulkDownRev)
#	
#	BulkDown = []
#	
#	for i in range(len(BulkDownRev)):
#		BulkDummy = BulkDownRev[i,:0:-1]
#		BulkDummy = np.append(BulkDownRev[i,0],BulkDummy)
#		BulkDown.append(BulkDummy)
#	
#	BulkDown = np.array(BulkDown)
#	
#	fout = open('./out/depth.dat','w')
#	
#	fout.write('ARGO Output for the pressure-temperature profile of ./profiles/Tian-profile.dat using the STAND 2019 Network\n')
#	
#	switch = False
#	
#	for j in range(len(BulkUp[0])):
#		fout.write('\n')
#		for i in range(len(BulkUp)):
#			if j == 0:
#				if i < 6 or 'B' not in BulkUp[i,j]:
#					fout.write(BulkUp[i,j] + '   ')
#			if j > 0:
#				if i < 6 or 'B' not in BulkUp[i,0]:
#					xUp = float(BulkUp[i,j])
#					xDown = float(BulkDown[i,j])
#					if xUp >= xDown: fout.write('%.3e' % xUp)
#					if xUp < xDown: fout.write('%.3e' % xDown)
#					fout.write('  ')
#	fout.close()

#combineLife('Nothing')
