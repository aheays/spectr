import numpy as np
import matplotlib.pyplot as plt
from array import *

old_venus = '/Users/terezaconstantinou/ARGO/out/depth.dat'
p_old,T_old,NH_old,Kzz_old,Hz_old,zeta_old,C_old,Cg_old,Cd_old,C_1D_old,C_1S_old,H_old,K_old,N_old,O_old,O_1D_old,O_1S_old,Ar_old,Cl_old,Fe_old,He_old,Mg_old,Na_old,Si_old,Ti_old,C2_old,CH_old,CN_old,CO_old,H2_old,HN_old,HNX_old,HO_old,KH_old,N2_old,NO_old,O2_old,O2_D_old,O3_old,C2H_old,C4H_old,C2N_old,C2O_old,CH2_old,CH2X_old,CH3_old,CH4_old,JCH4_old,CHO_old,CN2_old,CNC_old,CNN_old,CNO_old,CO2_old,H2N_old,H2O_old,JH2O_old,H3N_old,JH3N_old,HCN_old,JHCN_old,HCl_old,HN2_old,HNC_old,HNO_old,HO2_old,HO3_old,KOH_old,KCl_old,N2O_old,JN2O_old,NCN_old,NCO_old,NO2_old,NO3_old,FeH_old,FeO_old,MgH_old,MgO_old,NaH_old,SiH_old,SiO_old,TiO_old,C2H2_old,JC2H2_old,C2H3_old,C2H4_old,JC2H4_old,C2H5_old,C2H6_old,JC2H6_old,C3H3_old,C3H4_old,JC3H4_old,C3H5_old,C3H6_old,C4H2_old,C4H3_old,C4H4_old,C4H6_old,JC4H6_old,CH2O_old,CH3N_old,CH3O_old,CH5N_old,JCH5N_old,CHN2_old,COOH_old,H2CN_old,H2N2_old,H2O2_old,H4N2_old,H4O2_old,HC3N_old,JHC3N_old,HCCN_old,HCCO_old,HCNO_old,HNCO_old,HNNH_old,HNO2_old,HNO3_old,HOCH_old,HOCN_old,HONC_old,N2O3_old,NCCN_old,JNCCN_old,NH2O_old,OCCN_old,NaOH_old,NaCl_old,SiH2_old,SiH3_old,SiH4_old,FeO2_old,MgHO_old,MgO2_old,aNO2_old,C2H2O_old,C2H3N_old,JC2H3N_old,C2H3O_old,C2H4O_old,C2H6N_old,C3H2O_old,C3H3N_old,C3H3O_old,C4H10_old,JC4H10_old,CH2CN_old,CH2N2_old,CH2O2_old,CH2OH_old,CH3CN_old,CH3NO_old,CH3O2_old,CH3OH_old,CH4O2_old,HCOCN_old,HCOOH_old,JHCOOH_old,NH2NH_old,NH2OH_old,Si2H6_old,C2H2O2_old,C2H3NO_old,C2H4O2_old,C2H4O3_old,C2H5NO_old,C2H5OH_old,C2H5OO_old,C2H6O2_old,CH2CHO_old,CH3N2H_old,CH3NO2_old,CH3NO3_old,CH3OCO_old,CH3ONO_old,H2NNO2_old,MgO2H2_old,aC2H3O_old,aC2H4O_old,aCH32O_old,aCH3N2_old,aCH3O2_old,C2H3NO2_old,C2H5NO2_old,CH2NCH3_old,CH3CH2O_old,CH3CHOH_old,CH3COOH_old,CH3OCHO_old,CH3OCH2_old,H2C2HOH_old,aCH2OH2_old,aCH32CO_old,aCN2H6O_old,aCO2H5N_old,bCO2H5N_old,Cl2_old,Cl2S_old,Cl2S2_old,ClCO_old,ClCO3_old,ClO_old,ClS_old,ClS2_old,ClSO2_old,COCl2_old,H2S_old,HS_old,HSCl_old,HSO3_old,H2SO3_old,JH2SO3_old,H2SO4_old,JH2SO4_old,HOCl_old,HOCl2_old,OCS_old,OSCl_old,S_old,S2_old,JS2_old,S2O_old,S3_old,JS3_old,S4_old,JS4_old,S5_old,JS5_old,S6_old,JS6_old,S7_old,JS7_old,S8_old,JS8_old,SNO_old,SO_old,SO2_old,WSO2_old,SO2Cl2_old,SO3_old,SOSO_old,HSO_old,HOSO_old,HSOO_old,HSO2_old,HSNO_old,H2S2_old,CS_old,CS2_old,HCS_old,H2CS_old,CS2OH_old,CH3S_old,CH3SH_old,CH3SO_old,CH3SNO_old,H2C3S_old,PH3_old,PH2_old,C5H8_old,ISOPOH_old,ISOPO2_old,MACR_old,MACROH_old,MVK_old,MVKOH_old,MACRO2_old,ACETOL_old,MGLYOX_old,CH3CO3_old,HMVKO2_old,GLYALDE_old,HOCH2CO = np.loadtxt(old_venus, unpack=True, skiprows=3, usecols=range(0, 277))


filename = '/Users/terezaconstantinou/ARGO/outReverseAllotropes/depth.dat'
p,T,NH,Kzz,Hz,zeta,C,Cg,Cd,C_1D,C_1S,H,K,N,O,O_1D,O_1S,Ar,Cl,Fe,He,Mg,Na,Si,Ti,C2,CH,CN,CO,H2,HN,HNX,HO,KH,N2,NO,O2,O2_D,O3,C2H,C4H,C2N,C2O,CH2,CH2X,CH3,CH4,JCH4,CHO,CN2,CNC,CNN,CNO,CO2,H2N,H2O,JH2O,H3N,JH3N,HCN,JHCN,HCl,HN2,HNC,HNO,HO2,HO3,KOH,KCl,N2O,JN2O,NCN,NCO,NO2,NO3,FeH,FeO,MgH,MgO,NaH,SiH,SiO,TiO,C2H2,JC2H2,C2H3,C2H4,JC2H4,C2H5,C2H6,JC2H6,C3H3,C3H4,JC3H4,C3H5,C3H6,C4H2,C4H3,C4H4,C4H6,JC4H6,CH2O,CH3N,CH3O,CH5N,JCH5N,CHN2,COOH,H2CN,H2N2,H2O2,H4N2,H4O2,HC3N,JHC3N,HCCN,HCCO,HCNO,HNCO,HNNH,HNO2,HNO3,HOCH,HOCN,HONC,N2O3,NCCN,JNCCN,NH2O,OCCN,NaOH,NaCl,SiH2,SiH3,SiH4,FeO2,MgHO,MgO2,aNO2,C2H2O,C2H3N,JC2H3N,C2H3O,C2H4O,C2H6N,C3H2O,C3H3N,C3H3O,C4H10,JC4H10,CH2CN,CH2N2,CH2O2,CH2OH,CH3CN,CH3NO,CH3O2,CH3OH,CH4O2,HCOCN,HCOOH,JHCOOH,NH2NH,NH2OH,Si2H6,C2H2O2,C2H3NO,C2H4O2,C2H4O3,C2H5NO,C2H5OH,C2H5OO,C2H6O2,CH2CHO,CH3N2H,CH3NO2,CH3NO3,CH3OCO,CH3ONO,H2NNO2,MgO2H2,aC2H3O,aC2H4O,aCH32O,aCH3N2,aCH3O2,C2H3NO2,C2H5NO2,CH2NCH3,CH3CH2O,CH3CHOH,CH3COOH,CH3OCHO,CH3OCH2,H2C2HOH,aCH2OH2,aCH32CO,aCN2H6O,aCO2H5N,bCO2H5N,Cl2,Cl2S,Cl2S2,ClCO,ClCO3,ClO,ClS,ClS2,ClSO2,COCl2,H2S,HS,HSCl,HSO3,H2SO3,JH2SO3,H2SO4,JH2SO4,HOCl,HOCl2,OCS,OSCl,S,S2,JS2,S2O,S3,JS3,S4,JS4,S5,JS5,S6,JS6,S7,JS7,S8,JS8,SNO,SO,SO2,WSO2,SO2Cl2,SO3,SOSO,HSO,HOSO,HSOO,HSO2,HSNO,H2S2,CS,CS2,HCS,H2CS,CS2OH,CH3S,CH3SH,CH3SO,CH3SNO,H2C3S,PH3,PH2,C5H8,ISOPOH,ISOPO2,MACR,MACROH,MVK,MVKOH,MACRO2,ACETOL,MGLYOX,CH3CO3,HMVKO2,GLYALDE,HOCH2CO = np.loadtxt(filename, unpack=True, skiprows=3, usecols=range(0, 277))


h_km = array('f')

for i in range(0,len(Hz)):
    h_km.append( Hz[i]*1e-5)

# Plot

s="SO"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 1)
plt.plot(SO_old,h_km, label='SO')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.ylabel('Height / km')
plt.legend()

s="SO2"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 2)
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.plot(SO2_old,h_km, label='SO2')
plt.xscale('log')
plt.legend()

s="H2SO4"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x= np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 3)
plt.plot(H2SO4_old,h_km, label='H2SO4')
plt.errorbar(x,y,fmt="+",color="black")
plt.xscale('log')
plt.legend()

plt.subplot(4, 3, 6)
plt.plot(S2_old,h_km, label='S2')
plt.xscale('log')
plt.legend()

s="S3"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 5)
plt.plot(S3_old,h_km, label='S3')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.legend()

s="S4"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 12)
plt.plot(S4_old,h_km, label='S4')
xerr_low[0]= np.log(xerr_low[0])
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.xlabel('Mixing Ratio')
plt.legend()

s="HCl"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 9)
plt.plot(HCl_old,h_km, label='HCl')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low], fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.xlim(10**-7, 10**-6)
plt.legend()

s="OCS"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 8)
plt.plot(OCS_old,h_km, label='OCS')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.legend()

s="CO"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 7)
plt.plot(CO_old,h_km, label='CO')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.ylabel('Height / km')
plt.legend()

s="H2O"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x,xerr_high,xerr_low = np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 10)
plt.plot(H2O_old,h_km, label='H2O')
plt.errorbar(x,y,xerr=[xerr_high, xerr_low],fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.xlabel('Mixing Ratio')
plt.ylabel('Height / km')
plt.legend()

plt.subplot(4, 3, 4)
plt.plot(H2SO3_old,h_km, label='H2SO3')
plt.xscale('log')
plt.ylabel('Height / km')
plt.legend()

s="H2S"
fobs = "/Users/terezaconstantinou/ARGO/venus_data/%s.dat"%s
y,x= np.loadtxt(fobs,unpack=True,skiprows=1)

plt.subplot(4, 3, 11)
plt.plot(H2S_old,h_km, label='H2S')
plt.errorbar(x,y,fmt="+",color="black",capsize=2)
plt.xscale('log')
plt.xlabel('Mixing Ratio')
plt.legend()
plt.show()

plt.savefig()
