from copy import copy,deepcopy

from . import tools
# from . import levels
from . import levels_prototypes

prototypes = {}

## copy some direct from levels
for key in (
        'classname','description','notes','author','reference','date',
        'species', 'mass','reduced_mass','partition_source','partition',
        'ΓD',
):
    prototypes[key] = copy(levels_prototypes.prototypes[key])

## import all from levels with suffices added
for key,val in levels_prototypes.prototypes.items():
    tval = deepcopy(val)
    tval['infer'] = {tuple(key+'_u'
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+'_u'] = tval
    tval['infer'] = {tuple(key+'_l'
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+'_l'] = tval

## add lines things
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='U', cast=str, fmt='<10s')
prototypes['ν'] = dict(description="Transition wavenumber (cm-1)", kind='f', fmt='>0.6f', infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind='f', fmt='<10.5g',infer={
    ('γself','Pself','γair','Pair'):lambda self,γself,Pself,γair,Pair: γself*convert(Pself,'Pa','atm')+γair*convert(Pair,'Pa','atm'), # LINEAR COMBINATION!
    ('γself','Pself'):lambda self,γself,Pself: γself*convert(Pself,'Pa','atm'),
    ('γair','Pair'):lambda self,γair,Pair: γair*convert(Pair,'Pa','atm'),})
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind='f',fmt='<10.5g', infer={('mass','Ttr','ν'): lambda self,mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,})
prototypes['f'] = dict(description="Line f-value (dimensionless)",kind='f',fmt='<10.5e', infer={
    ('Ae','ν','g_u','g_l'):lambda self,Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2,
    ('Sij','ν','J_l'): lambda self,Sij,ν,J_l: 3.038e-6*ν*Sij/(2*J_l+1), 
    ('σ','α_l'):lambda self,σ,α_l: σ*1.1296e12/α_l,})
prototypes['σ'] = dict(description="Spectrally-integrated photoabsorption cross section (cm2.cm-1).", kind='f', fmt='<10.5e',infer={
    ('τa','Nself_l'):lambda self,τ,column_densitypp: τ/column_densitypp, 
    ('f','α_l'):lambda self,f,α_l: f/1.1296e12*α_l,
    ('S','ν','Teq'):lambda self,S,ν,Teq,: S/(1-np.exp(-convert(constants.Boltzmann,'J','cm-1')*ν/Teq)),})
# prototypes['σ'] =dict(description="Integrated cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('τ','column_densitypp'):lambda self,τ,column_densitypp: τ/column_densitypp, ('f','populationpp'):lambda self,f,populationpp: f/1.1296e12*populationpp,})


def _f0(self,S296K,species,partition,E_l,Tex,ν):
    """See Eq. 9 of simeckova2006"""
    partition_296K = hitran.get_partition_function(species,296)
    c = convert(constants.Boltzmann,'J','cm-1') # hc/kB
    return (S296K
            *((np.exp(-E_l/(c*Tex))/partition)*(1-np.exp(-c*ν/Tex)))
            /((np.exp(-E_l/(c*296))/partition_296K)*(1-np.exp(-c*ν/296))))
prototypes['S'] = dict(description="Spectral line intensity (cm or cm-1/(molecular.cm-2) ", kind='f', fmt='<10.5e', infer={
    ('S296K','species','partition','E_l','Tex','ν'):_f0,})
prototypes['S296K'] = dict(description="Spectral line intensity at 296K reference temperature ( cm-1/(molecular.cm-2) ). This is not quite the same as HITRAN which also weights line intensities by their natural isotopologue abundance.", kind='f', fmt='<10.5e', infer={})
## Preferentially compute τ from the spectral line intensity, S,
## rather than than the photoabsorption cross section, σ, because the
## former considers the effect of stimulated emission.
prototypes['τ'] = dict(description="Integrated optical depth including stimulated emission (cm-1)", kind='f', fmt='<10.5e', infer={
    ('S','Nself_l'):lambda self,S,Nself_l: S*Nself_l,
},)
prototypes['τa'] = dict(description="Integrated optical depth from absorption only (cm-1)", kind='f', fmt='<10.5e', infer={
    ('σ','Nself_l'):lambda self,σ,Nself_l: σ*Nself_l,
},)
prototypes['Ae'] = dict(description="Radiative decay rate (s-1)", kind='f', fmt='<10.5g', infer={
    ('f','ν','g_u','g_l'):lambda self,f,ν,g_u,g_l: f/(1.49951*g_u/g_l/ν**2),
    ('At','Ad'): lambda self,At,Ad: At-Ad,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind='f', fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind='f', fmt='0.2f', infer={
    'Teq':lambda self,Tex:Teq,
})
prototypes['Ttr'] = dict(description="Translational temperature (K)", kind='f', fmt='0.2f', infer={
    'Tex':lambda self,Tex:Tex,})
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind='f', fmt='>+4g', infer={
    ('Jp','Jpp'):lambda self,Jp,Jpp: Jp-Jpp,},)
prototypes['L'] = dict(description="Optical path length (m)", kind='f', fmt='0.5f', infer={})
prototypes['γair'] = dict(description="Pressure broadening coefficient in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda self,x:abs(x), fmt='<10.5g', infer={},)
prototypes['δair'] = dict(description="Pressure shift coefficient in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda self,x:abs(x), fmt='<10.5g', infer={},)
prototypes['nair'] = dict(description="Pressure broadening temperature dependence in air (cm-1.atm-1.FWHM)", kind='f', cast=lambda self,x:abs(x), fmt='<10.5g', infer={},)
prototypes['γself'] = dict(description="Pressure self-broadening coefficient (cm-1.atm-1.FWHM)", kind='f', cast=lambda self,x:abs(x), fmt='<10.5g', infer={},)
prototypes['Pself'] = dict(description="Pressure of self (Pa)", kind='f', fmt='0.5f', infer={})
prototypes['Pair'] = dict(description="Pressure of air (Pa)", kind='f', fmt='0.5f', infer={})
prototypes['Nself'] = dict(description="Column density (cm-2)",kind='f',fmt='<11.3e', infer={
    ('Pself','L','Teq'): lambda self,Pself,L,Teq: (Pself*L)/(database.constants.Boltzmann*Teq)*1e-4,})


## vibratiobanl transition frequencies
prototypes['νv'] = dict(description="Electronic-vibrational transition wavenumber (cm-1)", kined=float, fmt='>11.4f', infer={('Tvp','Tvpp'): lambda self,Tvp,Tvpp: Tvp-Tvpp, ('λv',): lambda self,λv: convert_units(λv,'nm','cm-1'),})
prototypes['λv'] = dict(description="Electronic-vibrational transition wavelength (nm)", kind='f', fmt='>11.4f', infer={('νv',): lambda self,νv: convert_units(νv,'cm-1','nm'),},)

## transition strengths
prototypes['M']   = dict(description="Pointer to electronic transition moment (au).", kind=object, infer={})
prototypes['Mv']   = dict(description="Electronic transition moment for this vibronic level (au).", kind='f', fmt='<10.5e', infer={('μ','FCfactor'): lambda self,μ,FCfactor: μ/np.sqrt(FCfactor),})
prototypes['μv']  = dict(description="Electronic-vibrational transition moment (au)", kind='f',  fmt='<10.5e', infer={('M','χp','χpp','R'): lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),},) # could infer from S but then sign would be unknown
prototypes['μ']   = dict(description="Electronic-vibrational-rotational transition moment (au)", kind='f',  fmt='<10.5e', infer={('M','χp','χpp','R'): lambda self,M,χp,χpp,R: np.array([integrate.trapz(χpi*np.conj(χppi)*Mi,R) for (χpi,χppi,Mi) in zip(χp,χpp,M)]),},) # could infer from S but then sign would be unknown
def _f0(self,fv,ν,Λp,Λpp):
    """Convert a summed band fvalue into a band_strength."""
    Sv = fv/3.038e-6/ν
    Sv[(Λpp==0)&(Λp!=0)] /= 2 # divisor of (2-δ(0,Λ")δ(0,Λ'))/(2-δ(0,Λ')
    return(Sv)
def _f1(self,Aev,ν,Λp,Λpp):
    """Convert an average band emission rate a band_strength"""
    Sv = Aev/2.026e-6/v**3
    Sv[(Λp==0)&(Λpp!=0)] /= 2.
    return(Sv)
prototypes['Sv'] =dict(description="Band strength (⟨vp|Re|vpp⟩**2, au)", kind='f',  fmt='<10.5e', infer={('Sij','SJ'): lambda self,Sij,SJ: Sij/SJ, ('μ',):lambda self,μ:μ**2, ('fv','ν','Λp','Λpp')  : lambda self,fv,ν,Λp,Λpp  : band_fvalue_to_band_strength(fv,ν,Λp,Λpp), ('fv','νv','Λp','Λpp') : lambda self,fv,νv,Λp,Λpp : band_fvalue_to_band_strength(fv,νv,Λp,Λpp), ('Aev','ν','Λp','Λpp') : lambda self,Aev,ν,Λp,Λpp : band_emission_rate_to_band_strength(Aev,ν,Λp,Λpp ), ('Aev','νv','Λp','Λpp'): lambda self,Aev,νv,Λp,Λpp: band_emission_rate_to_band_strength(Aev,νv,Λp,Λpp),},)
def _f1(self,f,SJ,Jpp,Λp,Λpp):
    """Get band fvalues from line strength"""
    fv = f/SJ*(2.*Jpp+1.)       # correct? What about 2S+1?
    fv[(Λpp==0)&(Λp!=0)] *= 2
    return(fv)
prototypes['fv'] = dict(description="Band f-value (dimensionless)", kind='f',  fmt='<10.5e', infer={('Sv','ν','Λp','Λpp'):  lambda self,Sv,ν,Λp,Λpp :  band_strength_to_band_fvalue(Sv,ν, Λp,Λpp), ('Sv','νv','Λp','Λpp'): lambda self,Sv,νv,Λp,Λpp:  band_strength_to_band_fvalue(Sv,νv,Λp,Λpp), ('f','SJ','Jpp','Λp','Λpp'): _f1,})
prototypes['Aev'] =dict(description="Einstein A coefficient / emission rate averaged over a band (s-1).", kind='f',  fmt='<10.5e', infer={('Sv','ν' ,'Λp','Λpp'): lambda self,Sv,ν ,Λp,Λpp: band_strength_to_band_emission_rate(Sv,ν ,Λp,Λpp), ('Sv','νv','Λp','Λpp'): lambda self,Sv,νv,Λp,Λpp: band_strength_to_band_emission_rate(Sv,νv,Λp,Λpp),},) 
prototypes['σv'] =dict(description="Integrated cross section of an entire band (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('fv',):lambda self,fv: band_fvalue_to_band_cross_section(fv),},)
prototypes['Sij'] =dict(description="Line strength (au)", kind='f',  fmt='<10.5e', infer={('Sv','SJ'): lambda self,Sv,SJ:  Sv*SJ, ('f','ν','Jpp'): lambda self,f,ν,Jpp: f/3.038e-6/ν*(2*Jpp+1), ('Ae','ν','Jp'): lambda self,Ae,ν,Jp: Ae/(2.026e-6*ν**3/(2*Jp+1)),})
prototypes['Ae'] =dict(description="Einstein A coefficient / emission rate (s-1).", kind='f',  fmt='<10.5e', infer={('f','ν','Jp','Jpp'): lambda self,f,ν,Jp,Jpp: f*0.666886/(2*Jp+1)*(2*Jpp+1)*ν**2, ('Sij','ν','Jp'): lambda self,Sij,ν,Jp: Sij*2.026e-6*ν**3/(2*Jp+1),},)
prototypes['FCfactor'] =dict(description="Franck-Condon factor (dimensionless)", kind='f',  fmt='<10.5e', infer={('χp','χpp','R'): lambda self,χp,χpp,R: np.array([integrate.trapz(χpi*χppi,R)**2 for (χpi,χppi) in zip(χp,χpp)]),},)
prototypes['Rcentroid'] =dict(description="R-centroid (Å)", kind='f',  fmt='<10.5e', infer={('χp','χpp','R','FCfactor'): lambda self,χp,χpp,R,FCfactor: np.array([integrate.trapz(χpi*R*χppi,R)/integrate.trapz(χpi*χppi,R) for (χpi,χppi) in zip(χp,χpp)]),},)
def _f0(self,Sp,Spp,Ωp,Ωpp,Jp,Jpp):
    if not (np.all(Sp==0) and np.all(Spp==0)): raise InferException('Honl-London factors only defined here for singlet states.')
    try:
        return(quantum_numbers.honl_london_factor(Ωp,Ωpp,Jp,Jpp))
    except ValueError as err:
        if str(err)=='Could not find correct Honl-London case.':
            raise InferException('Could not compute rotational line strength')
        else:
            raise(err)
# prototypes['SJ'] = dict(description="Rotational line strength (dimensionless)", kind='f',  fmt='<10.5e', infer= {('Sp','Spp','Ωp','Ωpp','Jp','Jpp'): _f0,})
# prototypes['τ'] = dict(description="Integrated optical depth (cm-1)", kind='f',  fmt='<10.5e', infer={('σ','column_densitypp'):lambda self,σ,column_densitypp: σ*column_densitypp,},)
# prototypes['I'] = dict(description="Integrated emission energy intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp','ν'):lambda self,Ae,populationp,column_densityp,ν: Ae*populationp*column_densityp*ν,},)
# prototypes['Ip'] = dict(description="Integrated emission photon intensity -- ABSOLUTE SCALE NOT PROPERLY DEFINED", kind='f',  fmt='<10.5e', infer={('Ae','populationp','column_densityp'):lambda self,Ae,populationp,column_densityp: Ae*populationp*column_densityp,},)
# prototypes['σd'] = dict(description="Integrated photodissociation cross section (cm2.cm-1).", kind='f',  fmt='<10.5e', infer={('σ','ηdp'):lambda self,σ,ηdp: σ*ηdp,})
# prototypes['Sabs'] = dict(description="Absorption intensity (cm-1/(molecule.cm-1)).", kind='f',  fmt='<10.5e', infer={})

## vibrational interaction energies
prototypes['ηv'] = dict(description="Reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ξv'] = dict(description="Reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ηDv'] = dict(description="Higher-order reduced spin-orbit interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})
prototypes['ξDv'] = dict(description="Higher-roder reduced rotational interaction energy mixing two vibronic levels (cm-1).", kind='f',  fmt='<10.5e', infer={})




## add infer functions -- could add some of these to above
prototypes['ν']['infer']['E_u','E_l'] = lambda self,Eu,El: Eu-El
prototypes['E_l']['infer']['E_u','ν'] = lambda self,Eu,ν: Eu-ν
prototypes['E_u']['infer']['E_l','ν'] = lambda self,El,ν: El+ν
prototypes['Γ']['infer']['Γ_u','Γ_l'] = lambda self,Γu,Γl: Γu+Γl
prototypes['Γ_l']['infer']['Γ','Γ_u'] = lambda self,Γ,Γu: Γ-Γu
prototypes['Γ_u']['infer']['Γ','Γ_l'] = lambda self,Γ,Γl: Γ-Γl
prototypes['J_u']['infer']['J_l','ΔJ'] = lambda self,J_l,ΔJ: J_l+ΔJ
prototypes['Tex']['infer']['Teq'] = lambda self,Teq: Teq
prototypes['Teq_u']['infer']['Teq'] = lambda self,Teq: Teq
prototypes['Teq_l']['infer']['Teq'] = lambda self,Teq: Teq
prototypes['Nself_u']['infer']['Nself'] = lambda self,Nself: Nself
prototypes['Nself_l']['infer']['Nself'] = lambda self,Nself: Nself
prototypes['species_l']['infer']['species'] = lambda self,species: species
prototypes['species_u']['infer']['species'] = lambda self,species: species
prototypes['ΔJ']['infer']['J_u','J_l'] = lambda self,J_u,J_l: J_u-J_l
prototypes['partition']['infer']['partition_l'] = lambda self,partition_l:partition_l
prototypes['partition']['infer']['partition_u'] = lambda self,partition_u:partition_u
prototypes['partition_l']['infer']['partition'] = lambda self,partition:partition
prototypes['partition_u']['infer']['partition'] = lambda self,partition:partition


