ptfilename = 'pT-Case1-EarlySun.dat'

radfilename = 'early-solar.dat'                       #File where the UV field lives

InitialConditionFileName = 'cond_initial_NOx.dat'

InitialConditionFileName = 'cond_initial/'+InitialConditionFileName
ptfilename = 'profiles/'+ptfilename

muav = 31.2                                     #Mean molecular mass, necessary no matter what!
                                                #mu = 2.33 for Jupiter, mu = 28.97 for Earth, 
                                                #mu = 2.35 for 51 Eri b. mu = 29.6 for Earth NoH2
                                                #mu = 29.34 for Earth 001H2, mu = 27. for 01H2
                                                #mu = 24.4 for Earth 02H2, mu = 29.3 for 001CH4NoH2
                                                #mu = 24.1 for 001CH402H2, 43.34 for Mars, 36 for
						#the 0.2 bar Earth, mu = 28.4 for the Yang Earth.
						#mu = 43.45 for Venus.
LOGG = 2.99                                     #Necessary no matter what! 3.39 for Jupiter,
                                                #2.99 for Earth, 3.06 for HAT-P-11b, 3.50 for 51 Erib
						#2.57 for Mars, 3.3 for KELT-9b
                                                #3.93 for HR 8799b, 4.0 for the L-T profile, 
						#2.94 for Venus
sigmaav = 10.4                                   #average cross section of the gas, 
                                                #look at the second entry in the network
        					#entries for banked reactions! 8.5 for Earth, 
						#5.0 for He/H rich environment, 11.5 for Mars
						#10.4 for the Early Earth or Venus

globalmax = 10					#Maximum number of global runs allowed
cc = 1.0e-3					#Convergence criterion (% change)
ct = 3.154e10					#Convergence timescale (over what time; 1 year = 3.154e7 s; 1000 years = 3.154e10 s)

Talfven = 2000.0				#Alfven Temperature

mu = 0.54				#Average cos inclination angle, 0.391 for Azores, 0.707 for 45 deg, 
					#0.54 for average.

CosmicRayYN = False				#Run with cosmic rays?
AlfvenYN = False				#Run with Alfven ionization?
CondYN = True					#Run with condensation?
ColdTrapYN = True 				#Run with a cold trap?

#These are the column locations in the pT file. If your pT file does not have one of the
#requested values, please enter the value '-1'. Check the units in your pT file to the standard
#units for this code, and enter the appropriate conversion under the _Conversion variables. The
#standard units are given in comments next to the _Conversion variables. Example: If your pT
#is in units of Pascals and deg C, set T_AddConversion to 273.15 and P_Conversion to 1e-5.

NH_Location = 3                                  #pT-Earth profile
NH_Conversion = 1.0                              #NH standard units: cm-3.
P_Location = 4
P_Conversion = 1.0                               #P standard units: bar
T_Location = 1
T_MultConversion = 1.0                          #T standard units: Kelvin; 0.556 for deg. F
T_AddConversion = 0.0                           #This should be 0.0 for K, 273.15 for deg. C,
                                               #255.37 for deg. F.
VZ_Location = 2
VZ_Order = 2                                    #Order is 0 [s-1],1 [cm/s] or 2 [cm2/s]
VZ_Conversion = 1.0                             #Standard units as listed above
HZ_Location = 0                                 #Stepsize height standard units: cm
HZ_Conversion = 1.0e5                           #Hz standard units: cm

#NH_Location = 3                                  #Venus profile
#NH_Conversion = 1.0                              #NH standard units: cm-3.
#P_Location = 4
#P_Conversion = 1.0                               #P standard units: bar
#T_Location = 1
#T_MultConversion = 1.0                          #T standard units: Kelvin; 0.556 for deg. F
#T_AddConversion = 0.0                           #This should be 0.0 for K, 273.15 for deg. C,
#                                                #255.37 for deg. F.
#VZ_Location = 2
#VZ_Order = 2                                    #Order is 0 [s-1],1 [cm/s] or 2 [cm2/s]
#VZ_Conversion = 1.0                             #Standard units as listed above
#HZ_Location = 0                                 #Stepsize height standard units: cm
#HZ_Conversion = 1.0e5                             #Hz standard units: cm






def giveptfilename (): return ptfilename

def giveradfilename (): return radfilename

def giveInitialConditionFileName (): return InitialConditionFileName

def givemu (): return mu
def givemuav (): return muav

def givelogg (): return LOGG
def givesigmaav (): return sigmaav

def giveNH_Location (): return NH_Location
def giveNH_Conversion (): return NH_Conversion

def giveP_Location (): return P_Location
def giveP_Conversion (): return P_Conversion

def giveT_Location (): return T_Location
def giveT_MultConversion (): return T_MultConversion
def giveT_AddConversion (): return T_AddConversion

def giveVZ_Location (): return VZ_Location
def giveVZ_Order (): return VZ_Order
def giveVZ_Conversion (): return VZ_Conversion

def giveHZ_Location (): return HZ_Location
def giveHZ_Conversion (): return HZ_Conversion

def givegmax (): return globalmax
def givecc (): return cc
def givect (): return ct
def giveTalfven (): return Talfven
def giveCosmicRayYN (): return CosmicRayYN
def giveAlfvenYN (): return AlfvenYN

def giveCondYN (): return CondYN
def giveColdTrapYN (): return ColdTrapYN
