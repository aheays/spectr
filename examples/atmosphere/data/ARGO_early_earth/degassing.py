import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22})#increased font size

depth = 'depth.dat'
P,T,NH,Kzz,Hz,zeta,C,Cg,Cd,C_1D,C_1S,H,K,N,O,O_1D,O_1S,Ar,Cl,Fe,He,Mg,Na,Si,Ti,C2,CH,CN,CO,H2,HN,HNX,HO,KH,N2,NO,O2,O2_D,O3,C2H,C4H,C2N,C2O,CH2,CH2X,CH3,CH4,JCH4,CHO,CN2,CNC,CNN,CNO,CO2,H2N,H2O,JH2O,H3N,JH3N,HCN,JHCN,HCl,HN2,HNC,HNO,HO2,HO3,KOH,KCl,N2O,JN2O,NCN,NCO,NO2,NO3,FeH,FeO,MgH,MgO,NaH,SiH,SiO,TiO,C2H2,JC2H2,C2H3,C2H4,JC2H4,C2H5,C2H6,JC2H6,C3H3,C3H4,JC3H4,C3H5,C3H6,C4H2,C4H3,C4H4,C4H6,JC4H6,CH2O,CH3N,CH3O,CH5N,JCH5N,CHN2,COOH,H2CN,H2N2,H2O2,H4N2,H4O2,HC3N,JHC3N,HCCN,HCCO,HCNO,HNCO,HNNH,HNO2,HNO3,HOCH,HOCN,HONC,N2O3,NCCN,JNCCN,NH2O,OCCN,NaOH,NaCl,SiH2,SiH3,SiH4,FeO2,MgHO,MgO2,aNO2,C2H2O,C2H3N,JC2H3N,C2H3O,C2H4O,C2H6N,C3H2O,C3H3N,C3H3O,C4H10,JC4H10,CH2CN,CH2N2,CH2O2,CH2OH,CH3CN,CH3NO,CH3O2,CH3OH,CH4O2,HCOCN,HCOOH,JHCOOH,NH2NH,NH2OH,Si2H6,C2H2O2,C2H3NO,C2H4O2,C2H4O3,C2H5NO,C2H5OH,C2H5OO,C2H6O2,CH2CHO,CH3N2H,CH3NO2,CH3NO3,CH3OCO,CH3ONO,H2NNO2,MgO2H2,aC2H3O,aC2H4O,aCH32O,aCH3N2,aCH3O2,C2H3NO2,C2H5NO2,CH2NCH3,CH3CH2O,CH3CHOH,CH3COOH,CH3OCHO,CH3OCH2,H2C2HOH,aCH2OH2,aCH32CO,aCN2H6O,aCO2H5N,bCO2H5N,Cl2,Cl2S,Cl2S2,ClCO,ClCO3,ClO,ClS,ClS2,ClSO2,COCl2,H2S,HS,HSCl,HSO3,H2SO3,JH2SO3,H2SO4,JH2SO4,HOCl,HOCl2,OCS,OSCl,S,S2,JS2,S2O,S3,JS3,S4,JS4,S5,JS5,S6,JS6,S7,JS7,S8,JS8,SNO,SO,SO2,WSO2,SO2Cl2,SO3,SOSO,HSO,HOSO,HSOO,HSO2,HSNO,H2S2,CS,CS2,HCS,H2CS,CS2OH,CH3S,CH3SH,CH3SO,CH3SNO,H2C3S,PH3,PH2,C5H8,ISOPOH,ISOPO2,MACR,MACROH,MVK,MVKOH,MACRO2,ACETOL,MGLYOX,CH3CO3,HMVKO2,GLYALDE,HOCH2CO = np.loadtxt(depth, unpack=True, skiprows=3, usecols=range(0, 277))

lifetime = 'lifetime.dat'
p_t, T_t, NH_t, Kzz_t, Hz_t, zeta_t,  C_t, Cg_t, Cd_t, C_1D_t, C_1S_t, H_t, K_t, N_t, O_t, O_1D_t, O_1S_t, Ar_t, Cl_t, Fe_t, He_t, Mg_t, Na_t, Si_t, Ti_t, C2_t, CH_t, CN_t, CO_t, H2_t, HN_t, HNX_t, HO_t, KH_t, N2_t, NO_t, O2_t, O2_D_t, O3_t, C2H_t, C4H_t, C2N_t, C2O_t, CH2_t, CH2X_t, CH3_t, CH4_t, JCH4_t, CHO_t, CN2_t, CNC_t, CNN_t, CNO_t, CO2_t, H2N_t, H2O_t, JH2O_t, H3N_t, JH3N_t, HCN_t, JHCN_t, HCl_t, HN2_t, HNC_t, HNO_t, HO2_t, HO3_t, KOH_t, KCl_t, N2O_t, JN2O_t, NCN_t, NCO_t, NO2_t, NO3_t, FeH_t, FeO_t, MgH_t, MgO_t, NaH_t, SiH_t, SiO_t, TiO_t, C2H2_t, JC2H2_t, C2H3_t, C2H4_t, JC2H4_t, C2H5_t, C2H6_t, JC2H6_t, C3H3_t, C3H4_t, JC3H4_t, C3H5_t, C3H6_t, C4H2_t, C4H3_t, C4H4_t, C4H6_t, JC4H6_t, CH2O_t, CH3N_t, CH3O_t, CH5N_t, JCH5N_t, CHN2_t, COOH_t, H2CN_t, H2N2_t, H2O2_t, H4N2_t, H4O2_t, HC3N_t, JHC3N_t, HCCN_t, HCCO_t, HCNO_t, HNCO_t, HNNH_t, HNO2_t, HNO3_t, HOCH_t, HOCN_t, HONC_t, N2O3_t, NCCN_t, JNCCN_t, NH2O_t, OCCN_t, NaOH_t, NaCl_t, SiH2_t, SiH3_t, SiH4_t, FeO2_t, MgHO_t, MgO2_t, aNO2_t, C2H2O_t, C2H3N_t, JC2H3N_t, C2H3O_t, C2H4O_t, C2H6N_t, C3H2O_t, C3H3N_t, C3H3O_t, C4H10_t, JC4H10_t, CH2CN_t, CH2N2_t, CH2O2_t, CH2OH_t, CH3CN_t, CH3NO_t, CH3O2_t, CH3OH_t, CH4O2_t, HCOCN_t, HCOOH_t, JHCOOH_t, NH2NH_t, NH2OH_t, Si2H6_t, C2H2O2_t, C2H3NO_t, C2H4O2_t, C2H4O3_t, C2H5NO_t, C2H5OH_t, C2H5OO_t, C2H6O2_t, CH2CHO_t, CH3N2H_t, CH3NO2_t, CH3NO3_t, CH3OCO_t, CH3ONO_t, H2NNO2_t, MgO2H2_t, aC2H3O_t, aC2H4O_t, aCH32O_t, aCH3N2_t, aCH3O2_t, C2H3NO2_t, C2H5NO2_t, CH2NCH3_t, CH3CH2O_t, CH3CHOH_t, CH3COOH_t, CH3OCHO_t, CH3OCH2_t, H2C2HOH_t, aCH2OH2_t, aCH32CO_t, aCN2H6O_t, aCO2H5N_t, bCO2H5N_t, Cl2_t, Cl2S_t, Cl2S2_t, ClCO_t, ClCO3_t, ClO_t, ClS_t, ClS2_t, ClSO2_t, COCl2_t, H2S_t, HS_t, HSCl_t, HSO3_t, H2SO3_t, JH2SO3_t, H2SO4_t, JH2SO4_t, HOCl_t, HOCl2_t, OCS_t, OSCl_t, S_t, S2_t, JS2_t, S2O_t, S3_t, JS3_t, S4_t, JS4_t, S5_t, JS5_t, S6_t, JS6_t, S7_t, JS7_t, S8_t, JS8_t, SNO_t, SO_t, SO2_t, WSO2_t, SO2Cl2_t, SO3_t, SOSO_t, HSO_t, HOSO_t, HSOO_t, HSO2_t, HSNO_t, H2S2_t, CS_t, CS2_t, HCS_t, H2CS_t, CS2OH_t, CH3S_t, CH3SH_t, CH3SO_t, CH3SNO_t, H2C3S_t, PH3_t, PH2_t, C5H8_t, ISOPOH_t, ISOPO2_t, MACR_t, MACROH_t, MVK_t, MVKOH_t, MACRO2_t, ACETOL_t, MGLYOX_t, CH3CO3_t, HMVKO2_t, GLYALDE_t, HOCH2CO_t, HOCH2CO3_t = np.loadtxt(lifetime, unpack=True, skiprows=4, usecols=range(0, 278))


Hscale = 15.9 * 10**5 #cm
mu = 43.45
#mp = 1.66* 10**(-27)

def degassing(X, X_t, t_dep):
    #deposition = (92 *10**3 * X[0])/(mu * mp * 887 * t_dep)
    photochem = sum(( x_t * x * 10**5 * p*10**3 )/(temp * 1.38* 10**(-19)) for x, p, x_t, temp in zip(X, P, X_t, T))
    outgassing = - photochem #deposition - photochem

    return outgassing

def colAbundance(X):
    average = sum((x * p*10**3 )/( temp * 1.38* 10**(-19)) for x, p, temp in zip(X, P, T))
    return average


X = H2S
X_t = H2S_t
t_dep = 30 * 10**6 * 365 * 24 * 3600

outgas = degassing(X, X_t, t_dep)
print("outgassing = " + str(outgas) + " cm-2 s-1")


tau = []
for x in X_t:
    t = 1/abs(x)
    tau.append(t)

h = []
for x in Hz:
    height = x/100000
    h.append(height)

s = "X"
plt.plot(tau, h, linewidth=3)
plt.ylabel('Height /km')
plt.xlabel('%s Inverse Photochemical Lifetime /s$^{-1}$')%s
plt.xscale('log')
plt.show()
