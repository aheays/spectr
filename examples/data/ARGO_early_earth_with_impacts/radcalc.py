import math
import re
import sys
import numpy as np

def f7(seq):
	seen = set()
	seen_add = seen.add
	return [ x for x in seq if not (x in seen or seen_add(x))]

def calcbigtau (gcount):
	fsigma = open('photo-sigma.dat','r')

	#if gcount < 4: exclude = np.array(['SO2'])
	#if gcount >= 4: exclude = np.array(['None'])

	exclude = np.array(['None'])

	Lambda = []
	react = []
	species = []
	colden = []
	sigma = []
	tau = []

	Lambda0_H2 = 4300.0
	Lambda0_N2 = 5322.43
	Lambda0_CO = 5322.43
	Lambda0_CO2 = 5322.43
	Lambda0_CH4 = 5322.43
	Lambda0_N2O = 5322.43
	Lambda0_H2O = 6000.00
	Lambda0_O2 = 5145.00
	sigmaR0_H2 = 2.3e-27
	sigmaR0_N2 = 5.1e-27
	sigmaR0_CO = 6.19e-27
	sigmaR0_CO2 = 1.24e-26
	sigmaR0_CH4 = 1.247e-26
	sigmaR0_N2O = 1.59e-26
	sigmaR0_H2O = 2.6e-27
	sigmaR0_O2 = 4.88e-27
	
	if 'H2' in exclude: sigmaR0_H2 = 0.0
	if 'N2' in exclude: sigmaR0_N2 = 0.0
	if 'CO' in exclude: sigmaR0_CO = 0.0
	if 'CO2' in exclude: sigmaR0_CO2 = 0.0
	if 'CH4' in exclude: sigmaR0_CH4 = 0.0
	if 'N2O' in exclude: sigmaR0_N2O = 0.0
	if 'H2O' in exclude: sigmaR0_H2O = 0.0
	if 'O2' in exclude: sigmaR0_O2 = 0.0

	if 'NoR' in exclude:
		sigmaR0_H2 = 0.0
        	sigmaR0_N2 = 0.0
        	sigmaR0_CO = 0.0
        	sigmaR0_CO2 = 0.0
        	sigmaR0_CH4 = 0.0
        	sigmaR0_N2O = 0.0
        	sigmaR0_H2O = 0.0
        	sigmaR0_O2 = 0.0

	i = 0

	i = 0
	OldElement = ''
	for line in fsigma:
		element = line.split()
		if i == 0:
			for j in range(len(element)):
				reactprod = element[j].split('->')
				stripreact = re.sub('\[\d\]','',reactprod[0])
				stripreact = re.sub('\[\d\d\]','',stripreact)
				stripreact = re.sub('\[\d\d\d\]','',stripreact)
				react.append(stripreact)
		if i > 0:
		  	Lambda.append(element[0])
		  	subsigma = []
		  	for j in range(1,len(element)):
		  		if react[j] != OldElement:
			  		subsigma.append(float(element[j]))
		  		if react[j] == OldElement:
			  		subsigma[len(subsigma)-1] += float(element[j])
		  		OldElement = react[j]
		  	sigma.append(subsigma)
		  	
		i += 1
	fsigma.close()
	redreact = react

	for j in range(len(redreact)):
		if redreact[j] == 'C(3P)': redreact[j] = 'C'
		if redreact[j] == 'C(1D)': redreact[j] = 'C_1D'
		if redreact[j] == 'C(1S)': redreact[j] = 'C_1S'
		if redreact[j] == 'CH3CHO': redreact[j] = 'C2H4O'
		if redreact[j] == 'H2CO': redreact[j] = 'CH2O'
		if redreact[j] == 'sCO': redreact[j] = 'CO'
		if redreact[j] == 'NH2': redreact[j] = 'H2N'
		if redreact[j] == 'NH3': redreact[j] = 'H3N'
		if redreact[j] == 'O(1D)': redreact[j] = 'O_1D'
		if redreact[j] == 'O(1S)': redreact[j] = 'O_1S'
		if redreact[j] == 'O(3P)': redreact[j] = 'O'
		if redreact[j] == 'OHe': redreact[j] = 'HO'

	i = 0
	
	fdata = open('./out/old-depth.dat','r')
	for line in fdata:
		element = line.split()
		if i == 2:
			for j in range(6,len(element)):
				species.append(element[j])
		i += 1
	
	fdata.close()

	AllData = np.loadtxt('./out/old-depth.dat',skiprows=3,unpack=True)
	
#        AllData = AllData[6:]
	KeepSpecies = []
	KeepData = []

        for i in range(6):
            KeepData.append(AllData[i])

#        for i in range(len(AllData)-6):
#                if all(AllData[i+6]) >= 1e-6 and species[i] in redreact: 
#                    KeepSpecies.append(species[i])
#                    KeepData.append(AllData[i+6])
        for i in range(len(AllData)-6):
                if species[i] in redreact: 
                    KeepSpecies.append(species[i])
                    KeepData.append(AllData[i+6])

        KeepData = np.array(KeepData)

        Sigma = np.loadtxt('photo-sigma.dat',skiprows=1,unpack=True)
#        h = KeepData[4]
        n = KeepData[2]

        Kappa = []
        h = []

        for i in range(len(KeepData[0])-1,-1,-1):
            switch = False
            for j in range(len(KeepSpecies)):
                for k in range(len(redreact)):
                    if KeepSpecies[j] == redreact[k] and switch == False:
                        h.append(KeepData[4,i])
                        if KeepSpecies[j] not in exclude: Kappa.append(Sigma[k]*KeepData[j+6,i]*n[i])
                        if KeepSpecies[j] in exclude: Kappa.append(0.0)
                        switch = True
                    if KeepSpecies[j] == redreact[k] and switch == True and KeepSpecies[j] not in exclude:
                        place = len(KeepData[0])-1-i
                        Kappa[place] += Sigma[k]*KeepData[j+6,i]*n[i]
	        
        Kappa = np.array(Kappa)
	
	KappaR = []

        Lambda = Sigma[0]

        for i in range(len(KeepData[0])-1,-1,-1):
            switch = False
            for j in range(len(KeepSpecies)):
		if KeepSpecies[j] == 'CO':
		    sigmaR = sigmaR0_CO*(Lambda/Lambda0_CO)**-4.0
		    KappaR.append(n[i]*sigmaR*KeepData[j+6,i])
		    place = len(KeepData[0])-1-i
	        if KeepSpecies[j] == 'H2':
		    sigmaR = sigmaR0_H2*(Lambda/Lambda0_H2)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'N2':
		    sigmaR = sigmaR0_N2*(Lambda/Lambda0_N2)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'CO2':
		    sigmaR = sigmaR0_CO2*(Lambda/Lambda0_CO2)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'CH4':
		    sigmaR = sigmaR0_CH4*(Lambda/Lambda0_CH4)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'N2O':
		    sigmaR = sigmaR0_N2O*(Lambda/Lambda0_N2O)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'H2O':
		    sigmaR = sigmaR0_H2O*(Lambda/Lambda0_H2O)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]
		if KeepSpecies[j] == 'O2':
		    sigmaR = sigmaR0_O2*(Lambda/Lambda0_O2)**-4.0
		    KappaR[place] += n[i]*sigmaR*KeepData[j+6,i]

        
        KappaR = np.array(KappaR)

        tau = []

        for i in range(len(Kappa)):
            if i < len(Kappa)-1: dh = h[i] - h[i+1]
            tau.append((Kappa[i]+KappaR[i])*dh)

        tau = np.array(tau)

	return tau

def findradfield (tau,mu,pnow,ptotal,radfilename):
        
        taunow = []

        for i in range(pnow,ptotal):
            if i == pnow: 
                taunow = np.array(tau[i])
            if i > pnow:
                taunow += tau[i]

        Lambda, Flux = np.loadtxt(radfilename,usecols=(0,1),unpack=True,skiprows=1)

        Flux *= np.exp(-taunow/mu)

        fout = open('blocked.dat','w')
        fout.write('lambda(angstrom)  Bnu(cm-2 s-1 Angstrom-1)\n')
        for i in range(len(Lambda)):
            if Flux[i] < 1.0e-90: Flux[i] = 0.0
            fout.write('   ')
            fout.write('%.3e' % Lambda[i])
            fout.write('   ')
            fout.write('%.3e' % Flux[i])
            fout.write('\n')
	fout.close()

def venusabsorber (h):

	Lambda, Flux = np.loadtxt('blocked.dat',skiprows=1,usecols=(0,1),unpack=True)

	if h > 6.7e6: tau = 3.0*0.056*np.exp(-(h-6.7e6)/3e5)*np.exp(-0.001*(Lambda-3600.0))
	if h <= 6.7e6 and h >= 5.8e6: tau = (70.0 - 1e-5*h)*0.056*np.exp(-0.001*(Lambda-3600.0))
	if h < 5.8e6: tau = 12.0*0.056*np.exp(-0.001*(Lambda-3600.0))
	
        fout = open('blocked.dat','w')
        fout.write('lambda(angstrom)  Bnu(cm-2 s-1 Angstrom-1)\n')
        for i in range(len(Lambda)):
            Flux[i] *= np.exp(-tau[i])
            if Flux[i] < 1.0e-90: Flux[i] = 0.0
            fout.write('   ')
            fout.write('%.3e' % Lambda[i])
            fout.write('   ')
            fout.write('%.3e' % Flux[i])
            fout.write('\n')
	fout.close()

def findoptdepth (tau,mu,radfilename):
        
	taunow = np.zeros(len(tau[0]),dtype=int)

	for j in range(len(tau[0])):
		switch = False
		for i in range(len(tau)):
			if tau[i,j] > 1.0 and not switch:
				taunow[j] = i
				switch = True
			if i == len(tau)-1 and not switch:
				taunow[j] = i

	p,h = np.loadtxt('./out/old-depth.dat',skiprows=3,unpack=1,usecols=(0,4))
        Lambda, Flux = np.loadtxt(radfilename,usecols=(0,1),unpack=True,skiprows=1)

	fout = open('./out/tauone.dat','w')

	fout.write('Wavelength(A)   p,tau=1(bar)   h,tau=1(cm)')

	for i in range(len(taunow)):
		fout.write('\n')
		fout.write('%.3e' % Lambda[i])
		fout.write('   ')
		fout.write('%.3e' % p[len(p)-taunow[i]-1])
		fout.write('   ')
		fout.write('%.3e' % h[len(h)-taunow[i]-1])

	fout.close()

def zeta (pnow,ptotal,z,mu):
	
	fdata = open('./out/old-depth.dat','r')
	
        mp = 1.673e-24			#Proton mass in grams

	i = 0

	colden = 0.0
	colden_h = 0.0
	
	for line in reversed(fdata.readlines()):
		element = line.split()

                if i >= 1 and i < ptotal:                                                                                   # Patrick: changed 0 to 1
                        if i == ptotal-1: ngas = float(element[2])
                        try:
                                colden   += float(element[2])*(old_height-float(element[4]))    # Should be multiplied by distance increment! - that was your comment, Paul

                                                               # Patrick: changed it to be multiplied with increment of height
                                # Patrick: hydrogen nuclei column density: n_H (i=11) + 2*n_H2 (29) + n_H+      (293)
                                colden_h += float(element[2])*(float(element[11])+2*float(element[29])+float(element[293]))*(old_height-float(element[4]))

			except ValueError:
				print 'Not the right value at i = '+str(i)+'!'
			except IndexError:
				print 'Index out of range at i = '+str(i)+'!'
                if i < ptotal:
                        old_height = float(element[4])                                                                                  # Patrick: save height of layer above
		i += 1
	fdata.close()
	
	NH     = colden
	NH_nuc = colden_h
	X      = NH*mp*mu
 
        NH = colden
        X = NH*mp*mu

#SEP for HD189733b
#        if NH_nuc < 1e19: NH_nuc = 1e19
#        a_coeff = -0.61
#        b_coeff = -2.61
#        zeta_L  = 1.06 * 10**(-7.05) # s^-1
#        zeta_H  = 8.34 * 10**(-2.05) # s^-1
#        ne      = 2.5e25 # column density threshold [cm^-2]
#
#        zeta_0 = ( 1/(zeta_L * (NH_nuc/1e20)**a_coeff) + 1/(zeta_H * (NH_nuc/1e20)**b_coeff))**(-1)
#
#        if (NH_nuc>ne): zeta = zeta_0 * np.exp(-1*((NH_nuc/ne)-1.0))
#        else: zeta = zeta_0
#
#        # Prevent exponent having 3 digits after sign:
#        if zeta < 1e-90: zeta = 0



#Nichols suggestion:
#        if X <= 1.67: zeta = 1.565e-7*(X/0.001)**(0.2)
#        if X > 1.67: zeta = 2.6249e-4*(X/0.001)**(-0.8)*math.exp(-X/36.0)

#Rimmer & Vorgul calculation
#        zeta = 1.165e-6*(X/0.001)**(-0.6)*math.exp(-X/1000.0)
#        if zeta > 1e-8: zeta = 1e-8

#        ------------------------        

#Rimmer & Rugheimer calculation
#        N = 1.452e4
#        k = 1.944
#        l = 29.386
#        Q0 = 3.139
#        Q = N*k/l*(X/l)**(k)*math.exp(-X/l) + Q0
#        zeta = Q/ngas*100.0

#Updated Rimmer & Rugheimer Calculation
#	z = z*1e-5
#	if z <= 38.0: zeta = 1e-18
#	if z > 38.0: zeta = 2.9e-30*np.exp(0.7*z)

#Chuanfei calculation
#	z = z*1e-5
#
#	if z >= 115.7: zeta = 2.0e-7
#	if z < 115.7 and z >= 9.58: zeta = 1.0*math.exp((z - 230.5)/7.44)
#	if z < 9.58: zeta = 1.0*math.exp((z - 143.2)/4.5)

#Updated Chuanfei calculation
#	z = z*1e-5
#
#	if z >= 100.0: zeta = 7.689e-9
#	if z < 100.0 and z >= 80.0: zeta = 1.0*math.exp(-0.01*z - 17.704)
#	if z < 80.0 and z >= 23.0: zeta = 1.0*math.exp(0.06*z - 23.304)
#	if z < 23.0 and z >= 7.8: zeta = 1.0*math.exp(0.18*z - 26.064)
#	if z < 7.8: zeta = 1.0*math.exp(0.3*z - 27.0)
#
#Updated Chuanfei calculation (10 GeV case)
	z = z*1e-5

	if z >= 110.0: zeta = 1.63e-10
	if z < 110.0 and z >= 95.0: zeta = 1.0*math.exp(-0.0093*z - 21.52)
	if z < 95.0 and z >= 85.7: zeta = 1.0*math.exp(-0.022*z - 20.31)
	if z < 85.7 and z >= 83.1: zeta = 2.28e-10
	if z < 83.1 and z >= 70.0: zeta = 1.0*math.exp(0.111*z - 31.43)
	if z < 70.0 and z >= 40.0: zeta = 1.0*math.exp(0.107*z - 31.15)
	if z < 40.0 and z >= 5.0: zeta = 1.0*math.exp(0.276*z**0.836 - 32.9)
	if z < 5.0 and z >= 1.0: zeta = 1.0*math.exp(0.275*z - 33.275)
	if z < 1.0: zeta = 4.66e-15

#        ------------------------        
#        ------------------------        

#        print 'ngas = ' + str(ngas) + ' Q = ' + str(Q) + ' zeta = ' + str(zeta)

#  	AV = colden/2.3e21
#  	NH = colden
#  	K = 5.978e26
#	zetae = 5.0*5.5e2/ngas*(NH/K)**1*(1-(NH/K))**4
#	if NH > 6e26: zetae = 0.0
#	zetap = 3.05e-16*AV**-0.6 + 1.0e-17
#	if AV < 0.02:
#		zetap = 3e-15
#	if AV > 217:
#		zetap = 1e-17*math.exp(-3.92e-5*AV)
#
#	zeta = zetap + zetae
#
	return zeta



#First entry in findradfield is tau (a huge matrix), second is always zero, the third is the distance in numbers of vertical height segments from the top of the atmosphere, and four is the datafile being read.
#tau = calcbigtau(2)
#findoptdepth(tau,0.707,'modern_solar_venus.dat')
#zeta(0,10,12.0,0.5)
#findradfield(tau,0.707,0,83,'modern-solar.dat')
