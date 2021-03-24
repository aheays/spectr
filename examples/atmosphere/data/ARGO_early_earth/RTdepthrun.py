import math
import numpy as np
import os
import copy
import sys
import radcalc
import time
import RTheader
import RTlifetime
import RTsetdown
import RTmakedepth
import RTrenormalize
#import RTplot
import RTtest
import RTsaverates

NELEM = 16

kB = 1.3806e-16					#Boltzmann constant in erg/K
mp = 1.6726e-24					#Proton mass in g

threshhold = 1.e-10

P = 0.0
NH = 0.0
T = 0.0
Te = 0.0
VZ = 0.0
mu = RTheader.givemu()

globalmax = RTheader.givegmax()
ConvergenceCriterion = RTheader.givecc()
ConvergenceTimescale = RTheader.givect()
lmax = 0
gcount = 0
#Talfven = 2300
Talfven = RTheader.giveTalfven()

CosmicRaysOn = RTheader.giveCosmicRayYN()
AlfvenIonizationOn = RTheader.giveAlfvenYN()

IsVenus = False					#This includes the sulfur and chlorine chemistry appropriate
						#for Venus-like exoplanets
IsIsoprene = False				#This includes the isoprene chemistry
IsFixed = True					#This fixes the surface mixing ratios, typically = True
RunFromPrevious = False                         #If this is true, then the last old-depth.dat file 
                                                #is used as step 1, and radiation is immediately 
                                                #activated.

CondensationOn = RTheader.giveCondYN()
ColdTrapOn = RTheader.giveColdTrapYN()

ptfilename = RTheader.giveptfilename()
radfilename = RTheader.giveradfilename()
InitialConditionFileName = RTheader.giveInitialConditionFileName()

NH_Location = RTheader.giveNH_Location()
NH_Conversion = RTheader.giveNH_Conversion()
P_Location = RTheader.giveP_Location()
P_Conversion = RTheader.giveP_Conversion()
T_Location = RTheader.giveT_Location()
T_MultConversion = RTheader.giveT_MultConversion()
T_AddConversion = RTheader.giveT_AddConversion()
	              
VZ_Location = RTheader.giveVZ_Location()
VZ_Order = RTheader.giveVZ_Order()
VZ_Conversion = RTheader.giveVZ_Conversion()
HZ_Location = RTheader.giveHZ_Location()
HZ_Conversion = RTheader.giveHZ_Conversion()

muav = RTheader.givemuav()
LOGG = RTheader.givelogg()
sigmaav = RTheader.givesigmaav()

print 'Reading values from the temperature profile into lists...\n\n'

fprofile = open(ptfilename,'r')

ProfileList = []

for line in reversed(fprofile.readlines()):
	element = line.split()
	try:
		P = float(element[P_Location])*P_Conversion
		NH = float(element[NH_Location])*NH_Conversion
		T = float(element[T_Location])*T_MultConversion + T_AddConversion
		Te = T
		VZ = float(element[VZ_Location])*VZ_Conversion
		HZ = float(element[HZ_Location])*HZ_Conversion
		ProfileList.append([P,NH,T,Te,VZ,HZ])
	except ValueError:
	 	print 'Out of range!'
	except IndexError:
		print 'Out of range!'

fprofile.close()

print 'Beginning global radiative transfer ARGO + STAND model for pT profile = '+ptfilename+' and radiation data file = '+radfilename+' and initial conditions file = '+InitialConditionFileName

while gcount < globalmax:
        if RunFromPrevious == True and gcount == 0:
            gcount += 1
            globalmax += 1
            with open('./out/old-depth.dat','r') as fcount:
                for line in fcount:
                    lmax += 1
            lmax -= 3

	print '*******************************************'
	if gcount == 0 and globalmax > 1: print 'First Global Run! Radiation off!'
	if gcount == 0: os.system('cp -f ' + radfilename + ' blocked.dat') 
	if gcount == 0 and globalmax == 1: print 'Radiationless Model Run!'
        if RunFromPrevious == True: print 'WARNING! Running from a previous old-depth.out file!'
	if gcount == 1: print 'Global run 2/'+str(globalmax)+'! Radiation just turned on!'
	if gcount > 1: print 'Global run '+str(gcount+1)+'/'+str(globalmax)+'!\n'
	print '*******************************************\n\n'
        if gcount == 0: RTlifetime.start()
        if gcount == 0 and RunFromPrevious == False:
	    print 'REMOVING old depth, old old-depth, and old lifetime, old old-lifetime and old tauone files!'
            os.system('rm -f ./out/depth.dat ./out/depth-up.dat ./out/depth-down.dat')
            os.system('rm -f ./out/old-depth*')
            os.system('rm -f ./out/lifetime.dat ./out/old-lifetime.dat')
	    os.system('rm -f ./out/lifetime-up.dat ./out/lifetime-down.dat')
	    os.system('rm -f ./out/old-lifetime-up.dat ./out/old-lifetime-down.dat')

        if gcount == 1 and RunFromPrevious == True:
	    print 'REMOVING old depth and old lifetime files!'
            os.system('rm -f ./out/depth.dat ./out/depth-up.dat ./out/depth-down.dat')
	    os.system('rm -f ./out/lifetime.dat ./out/lifetime-up.dat ./out/lifetime-down.dat')
        if (gcount > 0 and RunFromPrevious == False) or (gcount > 1 and RunFromPrevious == True):
	    print 'MOVING old depth files to old-depth files, and lifetime files to old-lifetime files!'
	    os.system('mv -f ./out/depth.dat ./out/old-depth.dat')
	    os.system('cp -f ./out/old-depth.dat ./out/old-depth-'+str(gcount)+'.dat')
	    os.system('mv -f ./out/depth-up.dat ./out/old-depth-up.dat')
	    os.system('mv -f ./out/depth-down.dat ./out/old-depth-down.dat')
            os.system('mv -f ./out/lifetime-up.dat ./out/old-lifetime-up.dat')
            os.system('mv -f ./out/lifetime-down.dat ./out/old-lifetime-down.dat')
            os.system('mv -f ./out/lifetime.dat ./out/old-lifetime.dat')
	print 'Removing all the \'verif.dat\' files from the Reactions folder!'
	os.system('rm -f ./out/Reactions/*')
	print 'Removing all the \'plot.dat\' files from the Plot folder!'
	os.system('rm -f ./out/Plot/*')
	print 'Removing all the figure files from the Figures folder!'
	os.system('rm -f ./out/Figures/*')

	os.system('cp -f '+InitialConditionFileName+' input.dat')

	if gcount > 0 and radfilename != 'none.dat':
		print 'Calculating depth-dependent Radition Field...'
		sys.stdout.flush()
		tau = radcalc.calcbigtau(gcount)
		print 'Calculation complete\n\n'
		sys.stdout.flush()

	lcount = 0

	for i in range(len(ProfileList)):
		P = ProfileList[i][0]
		NH = ProfileList[i][1]
		try:
			NHabove = ProfileList[i+1][1]
		except IndexError:
			NHabove = NH
		try:
			NHbelow = ProfileList[i-1][1]
		except IndexError:
			NHbelow = NH
		T = ProfileList[i][2]
		try:
			Tabove = ProfileList[i+1][2]
		except IndexError:
			Tabove = T
		try:
			Tbelow = ProfileList[i-1][2]
		except IndexError:
			Tbelow = T
		Te = ProfileList[i][3]
		VZ = ProfileList[i][4]
		try:
			dHzabove = ProfileList[i+1][5]-ProfileList[i][5]
		except IndexError:
			dHzabove = ProfileList[i][5]-ProfileList[i-1][5]
		Hz = ProfileList[i][5]
		try:
			dHzbelow = ProfileList[i][5]-ProfileList[i-1][5]
		except IndexError:
			dHzbelow = ProfileList[i+1][5]-ProfileList[i][5]
		if i == 0: dHzbelow = ProfileList[i+1][5]-ProfileList[i][5]
		ScaleHeight = kB*T/(muav*mp*10.0**LOGG)
		if VZ_Order == 0: 
			VZ = VZ*ScaleHeight
			KZZ = VZ*ScaleHeight*ScaleHeight
			print 'Warning: Vertical mixing may not be properly scaled!'
                if VZ_Order == 1: 
			KZZ = copy.copy(VZ)
			KZZ *= ScaleHeight
		if VZ_Order == 2: 
			KZZ = copy.copy(VZ)
			VZ = VZ/ScaleHeight
		try:
			VZabove = ProfileList[i+1][4]
			if VZ_Order == 0: VZabove = VZabove*ScaleHeight
			if VZ_Order == 2: VZabove = VZabove/ScaleHeight
		except IndexError:
			VZabove = VZ
		try:
			VZbelow = ProfileList[i-1][4]
			if VZ_Order != 1: ScaleHeight = kB*T/(muav*mp*10.0**LOGG)
			if VZ_Order == 0: VZbelow = VZbelow*ScaleHeight
			if VZ_Order == 2: VZbelow = VZbelow/ScaleHeight
		except IndexError:
			VZbelow = VZ
		if i == 0:
			NHbelow = NH
			Tbelow = T
			VZbelow = VZ
		zeta = 0.0e+0
		if dHzabove < 0.0: dHzabove = -dHzabove		
		if dHzbelow < 0.0: dHzbelow = -dHzbelow		

		if gcount > 0 and lmax-lcount > 0: zeta = radcalc.zeta(0,lmax-lcount,Hz,mu)
                if CosmicRaysOn == False: zeta = 0.0e+0
                if AlfvenIonizationOn == True:
                    if gcount > 0 and lmax-lcount > 0 and T > Talfven: zeta = 1.0e-5
		fparamin = open('source.dat','r')
		fparamout = open('input_parameter.dat','w')
		for line in fparamin:
			sNH = '%.3e' % NH
			sNHabove = '%.3e' % NHabove
			sNHbelow = '%.3e' % NHbelow
			sT = '%.3e' % T
			sTabove = '%.3e' % Tabove
			sTbelow = '%.3e' % Tbelow
			sTe = '%.3e' % Te
			sVZ = '%.3e' % VZ
			sVZabove = '%.3e' % VZabove
			sVZbelow = '%.3e' % VZbelow
			sHz = '%.3e' % Hz
			sHzabove = '%.3e' % dHzabove
			sHzbelow = '%.3e' % dHzbelow
			szeta = '%.3e' % zeta
			smuav = '%.3e' % muav
			ssigmaav = '%.3e' % sigmaav
			slogg = '%.3e' % LOGG
			sNH = sNH.replace('e','D')
			sNHabove = sNHabove.replace('e','D')
			sNHbelow = sNHbelow.replace('e','D')
			sT = sT.replace('e','D')
			sTabove = sTabove.replace('e','D')
			sTbelow = sTbelow.replace('e','D')
			sTe = sTe.replace('e','D')
			sVZ = sVZ.replace('e','D')
			sVZabove = sVZabove.replace('e','D')
			sVZbelow = sVZbelow.replace('e','D')
			sHz = sHz.replace('e','D')
			sHzabove = sHzabove.replace('e','D')
			sHzbelow = sHzbelow.replace('e','D')
			szeta = szeta.replace('e','D')
			smuav = smuav.replace('e','D')
			ssigmaav = ssigmaav.replace('e','D')
			slogg = slogg.replace('e','D')
			if gcount > 0: line1 = line.replace('YN1','Y')
			if gcount == 0: line1 = line.replace('YN1','N')
			line2 = line1.replace('SIDEWAYS','UP')
			line3 = line2.replace('XXXXXXXXX',sNH)
			line4 = line3.replace('XXXXABOVE',sNHabove)
			line5 = line4.replace('XXXXBELOW',sNHbelow)
			line6 = line5.replace('YYYYYYYYY',sT)
			line7 = line6.replace('YYYYABOVE',sTabove)
			line8 = line7.replace('YYYYBELOW',sTbelow)
			line9 = line8.replace('EEEEEEEEE',sTe)
			line10 = line9.replace('ZZZZZZZZZ',sVZ)
			line11 = line10.replace('ZZZZABOVE',sVZabove)
			line12 = line11.replace('ZZZZBELOW',sVZbelow)
			line13 = line12.replace('TTTTTTTTT',sHz)
			line14 = line13.replace('TTTTABOVE',sHzabove)
			line15 = line14.replace('TTTTBELOW',sHzbelow)
			line16 = line15.replace('VVVVVVVVV',szeta)
			line17 = line16.replace('WWWWWWWWW',smuav)
			line18 = line17.replace('SIGMAXXXX',ssigmaav)
			line19 = line18.replace('LOGGXXXXX',slogg)
			if CondensationOn: line20 = line19.replace('YN2','Y')
			if not CondensationOn: line20 = line19.replace('YN2','N')
			if ColdTrapOn: line21 = line20.replace('YN3','Y')
			if not ColdTrapOn: line21 = line20.replace('YN3','N')
			if IsVenus: line22 = line21.replace('YN4','Y')
			if not IsVenus: line22 = line21.replace('YN4','N')
			if IsIsoprene: line23 = line22.replace('YN5','Y')
			if not IsIsoprene: line23 = line22.replace('YN5','N')
			fparamout.write(line23)
		fparamin.close()
		fparamout.close()
		if gcount > 0 and radfilename != 'none.dat': radcalc.findradfield(tau,mu,0,lmax-lcount,radfilename)
		if gcount > 0 and radfilename != 'none.dat' and IsVenus: radcalc.venusabsorber(Hz)
		print 'Running ARGO with STAND going upwards for h = '+('%.1f' % (1e-5*Hz)) + ' km, p = '+('%.2e' % P)+' bar and T = '+('%.2f' % T)+' K'
		sys.stdout.flush()
		#The big kahoona!
		os.system('./argo')
		os.system('cp -f ./verif.dat ./out/Reactions/up-P='+('%.2e' % P)+'_H='+('%.1f' % (Hz*1e-5))+'_verif.dat')
		os.system('cp -f ./plot.dat ./out/Plot/up-P='+('%.2e' % P)+'_H='+('%.1f' % (Hz*1e-5))+'_plot.dat')
		RTsaverates.save('Up',i,P,Hz)
		print 'Done with upwards run '+str(i+1)+'/'+str(len(ProfileList))+', Global '+str(gcount+1)+'/'+str(globalmax)+'! Generating output to out/depth-up.dat\n\n\n'
                #if Hz >= 6.4e6: sys.exit()
		species = []
		vf = []
		switch2 = 0
		os.system('cp -f output.dat input.dat')
		foneplace = open('output.dat','r')
		fdepth = open('./out/depth-up.dat','a')
		for line in foneplace:
			element = line.split()
			try: 
				if element[1] =='+-': time = element[0]
			except IndexError: continue
			if element[0] == '1': switch2 = 1
			if switch2 == 1:
				species.append(element[1])
				vf.append(element[NELEM+2])
		if i == 0:
			fdepth.write('ARGO Output for the pressure-temperature profile of '+ptfilename+' using the STAND 2019 Network\n\n')
			fdepth.write('p(bar)   T(K)   NH(cm-3)   Kzz(cm2s-1)   Hz(cm)   zeta(s-1)')
			for j in range(len(species)):
				fdepth.write('   '+species[j])
			fdepth.write('\n')
		fdepth.write('%.3e' % P)
		fdepth.write('   ')
		fdepth.write('%.1f' % T)
		fdepth.write('   ')
		fdepth.write('%.3e' % NH)
		fdepth.write('   ')
		fdepth.write('%.3e' % (KZZ))
		fdepth.write('   ')
		fdepth.write('%.3e' % Hz)
		fdepth.write('   ')
		fdepth.write('%.3e' % zeta)
		for j in range(len(vf)):
			fdepth.write('   ')
			vf[j] = vf[j].replace('D','e')
			try:
				fdepth.write('%.3e' % float(vf[j]))
			except ValueError:
				fdepth.write('0.000e+00')
		fdepth.write('\n')
	
		foneplace.close()
		fdepth.close()
		species, invtime = np.loadtxt('Tout.dat',usecols=(0,1),dtype=str,unpack=True)
		invtime = invtime.astype(float)
		flife = open('./out/lifetime-up.dat','a')
		if i == 0:
			flife.write('ARGO Chemical Lifetimes for the pressure-temperature profile of '+ptfilename+' using the STAND 2019 Network\n\n')
			flife.write('p(bar)   T(K)   NH(cm-3)   Kzz(cm2s-1)   Hz(cm)   zeta(s-1)')
			for j in range(len(species)):
				flife.write('   '+species[j])
			flife.write('\n')
		flife.write('%.3e' % P)
		flife.write('   ')
		flife.write('%.1f' % T)
		flife.write('   ')
		flife.write('%.3e' % NH)
		flife.write('   ')
		flife.write('%.3e' % (KZZ))
		flife.write('   ')
		flife.write('%.3e' % Hz)
		flife.write('   ')
		flife.write('%.3e' % zeta)
		for j in range(len(invtime)):
			flife.write('   ')
			flife.write('%.3e' % invtime[j])
		flife.write('\n')
	
		flife.close()
		lcount += 1

	if gcount == 0: lmax = lcount

	#Stop it after it does the upwards run here.
	#sys.exit()

	pmin = '%.3e' % P

	for i in range(len(ProfileList)-1,-1,-1):
		lcount -= 1
		P = ProfileList[i][0]
		NH = ProfileList[i][1]
		try:
			NHabove = ProfileList[i+1][1]
		except IndexError:
			NHabove = NH
		try:
			NHbelow = ProfileList[i-1][1]
		except IndexError:
			NHbelow = NH
		T = ProfileList[i][2]
		try:
			Tabove = ProfileList[i+1][2]
		except IndexError:
			Tabove = T
		try:
			Tbelow = ProfileList[i-1][2]
		except IndexError:
			Tbelow = T
		Te = ProfileList[i][3]
		VZ = ProfileList[i][4]
		Hz1 = ProfileList[i][5]
		ScaleHeight = kB*T/(muav*mp*10.0**LOGG)
		if VZ_Order == 0: 
			VZ = VZ*ScaleHeight
                        KZZ = VZ*ScaleHeight*ScaleHeight
                        print 'Warning: Vertical mixing may not be properly scaled!'
                if VZ_Order == 1:
                        KZZ = copy.copy(VZ)
                        KZZ *= ScaleHeight
                if VZ_Order == 2:
                        KZZ = copy.copy(VZ)
                        VZ = VZ/ScaleHeight
		try:
			VZabove = ProfileList[i+1][4]
			if VZ_Order != 1: ScaleHeight = kB*T/(muav*mp*10.0**LOGG)
			if VZ_Order == 0: VZabove = VZabove*ScaleHeight
			if VZ_Order == 2: VZabove = VZabove/ScaleHeight
		except IndexError:
			VZabove = VZ
		try:
			VZbelow = ProfileList[i-1][4]
			if VZ_Order != 1: ScaleHeight = kB*T/(muav*mp*10.0**LOGG)
			if VZ_Order == 0: VZbelow = VZbelow*ScaleHeight
			if VZ_Order == 2: VZbelow = VZbelow/ScaleHeight
		except IndexError:
			VZbelow = VZ
		try:
			dHzabove = ProfileList[i+1][5]-ProfileList[i][5]
		except IndexError:
			dHzabove = ProfileList[i][5]-ProfileList[i-1][5]
		Hz = ProfileList[i][5]
		try:
			dHzbelow = ProfileList[i][5]-ProfileList[i-1][5]
		except IndexError:
			dHzbelow = ProfileList[i+1][5]-ProfileList[i][5]
		if i == 0: dHzbelow = ProfileList[i+1][5]-ProfileList[i][5]
		if i == 0:
			NHbelow = NH
			Tbelow = T
			VZbelow = VZ
		zeta = 0.0e+0

		if dHzabove < 0.0: dHzabove = -dHzabove		
		if dHzbelow < 0.0: dHzbelow = -dHzbelow		

		if gcount > 0 and lmax-lcount > 0: zeta = radcalc.zeta(0,lmax-lcount,Hz,mu)
                if CosmicRaysOn == False: zeta = 0.0e+0
                if AlfvenIonizationOn == True:
                    if gcount > 0 and lmax-lcount > 0 and T > Talfven: zeta = 1.0e-5
		fparamin = open('source.dat','r')
		fparamout = open('input_parameter.dat','w')
		for line in fparamin:
			sNH = '%.3e' % NH
			sNHabove = '%.3e' % NHabove
			sNHbelow = '%.3e' % NHbelow
			sT = '%.3e' % T
			sTabove = '%.3e' % Tabove
			sTbelow = '%.3e' % Tbelow
			sTe = '%.3e' % Te
			sVZ = '%.3e' % VZ
			sVZabove = '%.3e' % VZabove
			sVZbelow = '%.3e' % VZbelow
			sHz = '%.3e' % Hz
			sHzabove = '%.3e' % dHzabove
			sHzbelow = '%.3e' % dHzbelow
			szeta = '%.3e' % zeta
			smuav = '%.3e' % muav
			ssigmaav = '%.3e' % sigmaav
			slogg = '%.3e' % LOGG
			sNH = sNH.replace('e','D')
			sNHabove = sNHabove.replace('e','D')
			sNHbelow = sNHbelow.replace('e','D')
			sT = sT.replace('e','D')
			sTabove = sTabove.replace('e','D')
			sTbelow = sTbelow.replace('e','D')
			sTe = sTe.replace('e','D')
			sVZ = sVZ.replace('e','D')
			sVZabove = sVZabove.replace('e','D')
			sVZbelow = sVZbelow.replace('e','D')
			sHz = sHz.replace('e','D')
			sHzabove = sHzabove.replace('e','D')
			sHzbelow = sHzbelow.replace('e','D')
			szeta = szeta.replace('e','D')
			smuav = smuav.replace('e','D')
			ssigmaav = ssigmaav.replace('e','D')
			slogg = slogg.replace('e','D')
			if gcount > 0: line1 = line.replace('YN1','Y')
			if gcount == 0: line1 = line.replace('YN1','N')
			line2 = line1.replace('SIDEWAYS','DN')
			line3 = line2.replace('XXXXXXXXX',sNH)
			line4 = line3.replace('XXXXABOVE',sNHabove)
			line5 = line4.replace('XXXXBELOW',sNHbelow)
			line6 = line5.replace('YYYYYYYYY',sT)
			line7 = line6.replace('YYYYABOVE',sTabove)
			line8 = line7.replace('YYYYBELOW',sTbelow)
			line9 = line8.replace('EEEEEEEEE',sTe)
                        line10 = line9.replace('ZZZZZZZZZ',sVZ)
                        line11 = line10.replace('ZZZZABOVE',sVZabove)
                        line12 = line11.replace('ZZZZBELOW',sVZbelow)
                        line13 = line12.replace('TTTTTTTTT',sHz)
                        line14 = line13.replace('TTTTABOVE',sHzabove)
                        line15 = line14.replace('TTTTBELOW',sHzbelow)
                        line16 = line15.replace('VVVVVVVVV',szeta)
                        line17 = line16.replace('WWWWWWWWW',smuav)
                        line18 = line17.replace('SIGMAXXXX',ssigmaav)
                        line19 = line18.replace('LOGGXXXXX',slogg)
			if CondensationOn: line20 = line19.replace('YN2','Y')
			if not CondensationOn: line20 = line19.replace('YN2','N')
			if ColdTrapOn: line21 = line20.replace('YN3','Y')
			if not ColdTrapOn: line21 = line20.replace('YN3','N')
			if IsVenus: line22 = line21.replace('YN4','Y')
			if not IsVenus: line22 = line21.replace('YN4','N')
			if IsIsoprene: line23 = line22.replace('YN5','Y')
			if not IsIsoprene: line23 = line22.replace('YN5','N')
			fparamout.write(line23)
		fparamin.close()
		fparamout.close()
		if gcount > 0 and radfilename != 'none.dat': radcalc.findradfield(tau,mu,0,lmax-lcount,radfilename)
		if gcount > 0 and radfilename != 'none.dat' and IsVenus: radcalc.venusabsorber(Hz)
		RTsetdown.SetOutput(InitialConditionFileName,i,NELEM)
		os.system('cp -f output.dat input.dat')
		print 'Running ARGO with STAND going downwards for h = '+('%.1f' % (1e-5*Hz)) + ' km, p = '+('%.2e' % P)+' bar and T = '+('%.2f' % T)+' K'
		sys.stdout.flush()
		#The big kahoona!
		os.system('./argo')
		os.system('cp -f ./verif.dat ./out/Reactions/down-P='+('%.2e' % P)+'_H='+('%.1f' % (Hz*1e-5))+'_verif.dat')
		os.system('cp -f ./plot.dat ./out/Plot/down-P='+('%.2e' % P)+'_H='+('%.1f' % (Hz*1e-5))+'_plot.dat')
                RTsaverates.save('Down',len(ProfileList)-1-i,P,Hz)
		print 'Done with downwards run '+str(len(ProfileList)-i)+'/'+str(len(ProfileList))+', Global '+str(gcount+1)+'/'+str(globalmax)+'! Generating output to out/depth-down.dat\n\n\n'
		species = []
		vf = []
		switch2 = 0
		foneplace = open('output.dat','r')
		fdepth = open('./out/depth-down.dat','a')
		for line in foneplace:
			element = line.split()
			try: 
				if element[1] =='+-': time = element[0]
			except IndexError: continue
			if element[0] == '1': switch2 = 1
			if switch2 == 1:
				species.append(element[1])
				vf.append(element[NELEM+2])
		if i == len(ProfileList)-1:
			fdepth.write('ARGO Output for the pressure-temperature profile of '+ptfilename+' using the STAND 2019 Network\n\n')
			fdepth.write('p(bar)   T(K)   NH(cm-3)   Kzz(cm2s-1)   Hz(cm)   zeta(s-1)')
			for j in range(len(species)):
				fdepth.write('   '+species[j])
			fdepth.write('\n')
		fdepth.write('%.3e' % P)
		fdepth.write('   ')
		fdepth.write('%.1f' % T)
		fdepth.write('   ')
		fdepth.write('%.3e' % NH)
		fdepth.write('   ')
		fdepth.write('%.3e' % (KZZ))
		fdepth.write('   ')
		fdepth.write('%.3e' % Hz)
		fdepth.write('   ')
		fdepth.write('%.3e' % zeta)
		for j in range(len(vf)):
			fdepth.write('   ')
			vf[j] = vf[j].replace('D','e')
			try:
				fdepth.write('%.3e' % float(vf[j]))
			except ValueError:
				fdepth.write('0.000e+00')
		fdepth.write('\n')
	
		foneplace.close()
		fdepth.close()
		species, invtime = np.loadtxt('Tout.dat',usecols=(0,1),dtype=str,unpack=True)
		invtime = invtime.astype(float)
		flife = open('./out/lifetime-down.dat','a')
		if i == len(ProfileList)-1:
			flife.write('ARGO Chemical Lifetimes for the pressure-temperature profile of '+ptfilename+' using the STAND 2019 Network\n\n')
			flife.write('p(bar)   T(K)   NH(cm-3)   Kzz(cm2s-1)   Hz(cm)   zeta(s-1)')
			for j in range(len(species)):
				flife.write('   '+species[j])
			flife.write('\n')
		flife.write('%.3e' % P)
		flife.write('   ')
		flife.write('%.1f' % T)
		flife.write('   ')
		flife.write('%.3e' % NH)
		flife.write('   ')
		flife.write('%.3e' % (KZZ))
		flife.write('   ')
		flife.write('%.3e' % Hz)
		flife.write('   ')
		flife.write('%.3e' % zeta)
		for j in range(len(invtime)):
			flife.write('   ')
			flife.write('%.3e' % invtime[j])
		flife.write('\n')

		flife.close()
	

	pmax = '%.3e' % P

	RTmakedepth.combine(ptfilename)
	RTmakedepth.combineLife(ptfilename)
	RTlifetime.life(gcount)

	if not IsFixed:
		RTtest.ResetCondInitial(InitialConditionFileName,'./output.dat')

	print '\n\n ****SUCCESSFUL END OF SINGLE GLOBAL RUN!!!****\n\n'

	gcount += 1

	DidItConverge = False

	if gcount > 1:
		DidItConverge = RTtest.DoesItConverge(ConvergenceCriterion,ConvergenceTimescale,VZ_Order,muav,LOGG)
	
	if DidItConverge == True: 
		gcount = globalmax
		print '\n\n ****CONVERGENCE ACHIEVED WITHIN ' + str(ConvergenceCriterion*100.0) + '\%!****'

RTrenormalize.Renormalize()

if globalmax > 1:
	print '\n\n Creating a file for tau = 1 as a function of height: tauone.dat'
	radcalc.findoptdepth(tau,mu,radfilename)

#fin = open('./out/depth.dat','r')
#count = 0
#data = []
#species = []

#for line in fin:
#    if count == 2: names = line.split()[6:]
#    if count > 2: data.append(line.split()[6:])
#    count += 1
#
#fin.close()

#for i in range(len(data)):
#    switch = 0
#    for j in range(len(data[i])):
#        if float(data[i][j]) > threshhold: 
#            switch = 1
#        if switch == 1: 
#            species.append(names[j])
#            switch = 0

#seen = set()
#seen_add = seen.add
#species = [ x for x in species if not (x in seen or seen_add(x))]

#subspecies = []

#for i in range(len(species)):
#    subspecies.append(species[i])
#    if i % 5 == 0 and i != 0:
#        RTplot.PlotData('generic',subspecies,threshhold,dataloc='./out/depth.dat',outloc='./out/Figures/')
#        subspecies = []
#        print 'Finished figure '+str(i/5)+'/'+str((len(species)-1)/5)+'...'

print '\n\n'
print '*******************************'
print 'Model Complete! Successful Run!'
print '*******************************'
