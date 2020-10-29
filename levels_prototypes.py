prototypes = {}

prototypes['classname'] = dict( description="Type of levels of lines object.",kind=str ,infer={})
prototypes['description'] = dict( description="",kind=str ,infer={})
prototypes['notes'] = dict(description="Notes regarding this line" , kind=str ,infer={})
prototypes['author'] = dict(description="Author of data or printed file" ,kind=str ,infer={})
prototypes['reference'] = dict(description="Published reference" ,kind=str ,infer={})
prototypes['date'] = dict(description="Date data collected or printed" ,kind=str ,infer={})
prototypes['species'] = dict(description="Chemical species" ,kind=str ,infer={})
prototypes['mass'] = dict(description="Mass (amu)",kind=float, fmt='<11.4f', infer={('species',): lambda species: database.get_mass(species),})
prototypes['reduced_mass'] = dict(description="Reduced mass (amu)", kind=float, fmt='<11.4f', infer={('species','database',): lambda species: _get_species_property(species,'reduced_mass')})
prototypes['E'] = dict(description="Level energy (cm-1)" ,kind=float ,fmt='<14.7f' ,infer={})
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind=float,infer={})


def _f0(classname,J):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if classname == 'HomonuclearDiatomic':
        return 2*J+1
    else:
        raise InferException('Only valid of HomonuclearDiatomic')
@tools.vectorise_function
@functools.lru_cache
def _f1(classname,J,Inuclear,sa):
    """Calculate homonuclear diatomic molecule level degeneracy."""
    ## get total number of even or odd exchange combinations
    ntotal = (2*Inuclear+1)**2
    neven = 2*Inuclear+1 + (ntotal-(2*Inuclear+1))/2
    nodd = ntotal - neven
    if Inuclear%1==0:
        ## fermion
        if sa==+1:
            return (2*J+1)*neven
        else:
            return (2*J+1)*nodd
    else:
        ## boson
        if sa==+1:
            return (2*J+1)*nodd
        else:
            return (2*J+1)*neven
prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind=int , infer={('classname','J'):_f0, ('classname','J','Inuclear','sa'):_f1,})
prototypes['pm'] = dict(description="Total inversion symmetry" ,kind=int ,infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g', infer={('A',):lambda τ: 5.309e-12*A,})
prototypes['τ'] = dict(description="Total decay lifetime (s)", kind=float, infer={ ('A',): lambda A: 1/A,})       
prototypes['A'] = dict(description="Total decay rate (s-1)", kind=float, infer={('Γ',): lambda Γ: Γ/5.309e-12,})
prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin", kind=float,fmt='>0.1f',infer={})
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind=float, infer={('J','SR') : lambda J,SR: J-SR,})
prototypes['S'] = dict(description="Total electronic spin quantum number", kind=float,infer={})
prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum (cm-1).", kind=float,infer={() :lambda : 0.,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={'Teq':lambda Teq:Teq})
prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={('database',): lambda : 'database',})
@tools.vectorise_function_in_chunks()
def _f5(partition_source,species,Tex):
    from . import hitran
    if partition_source!='HITRAN':
        raise InferException(f'Partition source not "HITRAN".')
    return hitran.get_partition_function(species,Tex)
prototypes['partition'] = dict(description="Partition function.", kind=float, fmt='<11.3e', infer={('partition_source','species','Tex'):_f5,})
prototypes['α'] = dict(description="State population", kind=float, fmt='<11.4e', infer={('partition','E','g','Tex'): lambda partition,E,g,Tex : g*np.exp(-E/(convert(constants.Boltzmann,'J','cm-1')*Tex))/partition,})
prototypes['Nself'] = dict(description="Column density (cm2)",kind=float,fmt='<11.3e', infer={})
prototypes['label'] = dict(description="Label of electronic state", kind=str,infer={})
prototypes['v'] = dict(description="Vibrational quantum number", kind=int,infer={})
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={})
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={})
prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={})
@tools.vectorise_arguments
def _f0(ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['σv'] = dict(description="Symmetry with respect to σv reflection.", kind=int,infer={('ef','J'):_f0,})
prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind=int,infer={})
prototypes['sa'] = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={('σv','gu'):lambda σv,gu: σv*gu,})
@tools.vectorise_arguments
def _f0(S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef'] = dict(description="e/f symmetry", kind=int,infer={('S','Λ','s'):_f0,})
prototypes['Fi'] = dict(description="Spin multiplet index", kind=int,infer={})
prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={})
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={})
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={})
prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind=float,infer={})


## Effective Hamiltonian parameters
prototypes['Tv']  = dict(description='Term origin (cm-1)' ,kind =float,fmt='0.6f',infer={})
prototypes['dTv'] = dict(description='Uncertainty in Term origin (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['Bv']  = dict(description='Rotational constant (cm-1)' ,kind =float,fmt='0.8f',infer={})
prototypes['dBv'] = dict(description='Uncertainty in rotational constant (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['Dv']  = dict(description='Centrifugal distortion (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dDv'] = dict(description='Uncertainty in centrifugal distortion (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['Hv']  = dict(description='Third order centrifugal distortion (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dHv'] = dict(description='Uncertainty in thrid order centrifugal distortion (1σ, cm-1)' ,kind=float,fmt='3g',infer  ={})
prototypes['Lv']  = dict(description='Fourth order centrifugal distortion (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dLv'] = dict(description='Uncertainty in fourth order centrifugal distortion (1σ, cm-1)' ,kind=float,fmt='3g',infer  ={})
prototypes['Av']  = dict(description='Spin-orbit energy (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dAv'] = dict(description='Uncertainty in spin-orbit energy (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['ADv'] = dict(description='Spin-orbit centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dADv']= dict(description='Uncertainty in spin-orbit centrifugal distortion (1σ, cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['AHv'] = dict(description='Higher-order spin-orbit centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dAHv']= dict(description='Uncertainty in higher-order spin-orbit centrifugal distortion (1σ, cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['λv']  = dict(description='Spin-spin energy (cm-1)',kind=float,fmt='0.6g',infer={})
prototypes['dλv'] = dict(description='Uncertainty in spin-spin energy (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['λDv'] = dict(description='Spin-spin centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dλDv']= dict(description='Uncertainty in spin-spin centrifugal distortion (1σ, cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['λHv'] = dict(description='Higher-order spin-spin centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dλHv']= dict(description='Uncertainty in higher-order spin-spin centrifugal distortion (1σ, cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['γv']  = dict(description='Spin-rotation energy (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dγv'] = dict(description='Uncertainty in spin-rotation energy (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['γDv'] = dict(description='Spin-rotation centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dγDv']= dict(description='Uncertainty in spin-rotation centrifugal distortion (cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['γHv'] = dict(description='Higher-orders pin-rotation centrifugal distortion (cm-1)',kind =float,fmt='0.6g',infer={})
prototypes['dγHv']= dict(description='Uncertainty in higher-order spin-rotation centrifugal distortion (cm-1)',kind=float,fmt='0.2g',infer={})
prototypes['ov']  = dict(description='Λ-doubling constant o (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dov'] = dict(description='Uncertainty in Λ-doubling constant o (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['oDv']  = dict(description='Higher-order Λ-doubling constant o (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['doDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['oHv']  = dict(description='Higher-order Λ-doubling constant o (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['doHv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['oLv']  = dict(description='Ligher-order Λ-doubling constant o (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['doLv'] = dict(description='Uncertainty in higher-order Λ-doubling constant o (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['pv']  = dict(description='Λ-doubling constant p (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dpv'] = dict(description='Uncertainty in Λ-doubling constant p (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['pDv']  = dict(description='Higher-order Λ-doubling constant p (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dpDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant p (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['qv']  = dict(description='Λ-doubling constant q (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dqv'] = dict(description='Uncertainty in Λ-doubling constant q (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})
prototypes['qDv']  = dict(description='Higher-order Λ-doubling constant q (cm-1)' ,kind =float,fmt='0.6g',infer={})
prototypes['dqDv'] = dict(description='Uncertainty in higher-order Λ-doubling constant q (1σ, cm-1)' ,kind=float,fmt='0.2g',infer={})

prototypes['Γv'] = dict(description="Total electronic-vibrational linewidth (cm-1 FWHM)", kind =float,  fmt='<10.5g', strictly_positive=True, infer={('τ',):lambda τ: 5.309e-12/τ,}) # tau=1/2/pi/gamma/c
prototypes['dΓv'] = dict(description="Uncertainty in total electronic-vibrational linewidth (cm-1 FWHM 1σ)", kind =float,  fmt='<10.5g', infer ={('Γ','τ','dτ'): lambda Γ,τ,dτ: dτ*Γ/τ,})
prototypes['τv'] = dict(description="Total electronic-vibrational decay lifetime (s)", kind=float,  fmt='<10.5g', infer ={('Γv',): lambda Γv: 5.309e-12/Γv, ('Atv',): lambda Atv: 1/Atv,}) 
prototypes['dτv'] = dict(description="Uncertainty in total electronic-vibrational decay lifetime (s 1σ)", kind =float,  fmt='<10.5g', infer ={('Γ','dΓ','τ'): lambda Γ,dΓ,τ: dΓ/Γ*τ, ('At','dAt','τ'): lambda At,dAt,τ: dAt/At*τ,})
prototypes['Atv'] = dict(description="Total electronic-vibrational decay rate (s-1)", kind =float,  fmt='<10.5g', infer ={('τv',): lambda τv: 1/τv, ('Adv','Ave'): lambda Adv,Aev: Adv+Aev, ('Aev',): lambda Aev: Aev, ('Adv',): lambda Adv: Adv,})# Test for Ad and Ae, if failed then one or the other is undefined/zero
prototypes['dAtv']= dict(description="Uncertainty in total electronic-vibrational decay rate (s-1 1σ)", kind =float,  fmt='<10.5g', infer ={('τ','dτ','At'): lambda τ,dτ,At: dτ/τ*At,})
prototypes['Adv'] = dict(description="Nonradiative electronic-vibrational decay rate (s-1)", kind =float,  fmt='<10.5g', infer ={('At','Ae'): lambda At,Ae: At-Ae,})
prototypes['dAdv']= dict(description="Uncertainty in nonradiative electronic-vibrational decay rate (s-1 1σ)", kind =float,  fmt='<10.5g', infer ={})
prototypes['Aev'] = dict(description="Radiative electronic-vibrational decay rate (s-1)", kind =float,  fmt='<10.5g', infer ={('At','Ad'): lambda At,Ad: At-Ad,})
prototypes['dAev']= dict(description="Uncertainty in radiative electronic-vibrational decay rate (s-1 1σ)", kind =float,  fmt='<10.5g', infer ={})
prototypes['ηdv'] = dict(description="Fractional probability of electronic-vibrational level decaying nonradiatively (dimensionless)", kind =float,  fmt='<10.5g', infer ={('At','Ad'):lambda At,Ad:Ad/A,})
prototypes['dηdv']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying by dissociation by any channels (dimensionless, 1σ)", kind=float,  fmt='<10.5g', infer ={})
prototypes['ηev'] = dict(description="Fractional probability of electronic-vibrational level decaying radiatively (dimensionless)", kind =float,  fmt='<10.5g', infer ={('At','Ae'):lambda At,Ae:Ae/A,})
prototypes['dηev']= dict(description="Uncertainty in the fractional probability of electronic-vibrational level decaying radiatively (dimensionless, 1σ)", kind=float,  fmt='<10.5g', infer ={})

## v+1 reduced versions of these
def _vibrationally_reduce(
        self,
        reduced_quantum_number,
        reduced_polynomial_order,
        val,dval=None,
):
    """Reduce a variable."""
    reduced = np.full(val.shape,np.nan)
    ## loop through unique rotational series
    for keyvals in self.unique_dicts(*self.qn_defining_independent_vibrational_progressions):
        i = self.match(**keyvals)
        x = self[reduced_quantum_number][i]*(self[reduced_quantum_number][i]+1)
        y = val[i]
        ## do an unweighted mean if uncertainties are not set or all the same
        if dval is None:
            dy = None
        else:
            dy = dval[i]
        p = my.polyfit(x,y,dy,order=max(0,min(reduced_polynomial_order,sum(i)-2)),error_on_missing_dy=False)
        reduced[i] = p['residuals']
    return(reduced)

prototypes['Tvreduced'] = dict(description="Vibrational term value reduced by a polynomial in (v+1/2) (cm-1)", kind=float,  fmt='<11.4f',
    infer={('self','reduced_quantum_number','reduced_polynomial_order','Tv','dTv'): _vibrationally_reduce, # dTv is known -- use in a weighted mean
           ('self','reduced_quantum_number','reduced_polynomial_order','Tv'): _vibrationally_reduce,}) # dTv is not known
prototypes['dTvreduced'] = dict(description="Uncertainty in vibrational term value reduced by a polynomial in (v+1/2) (cm-1 1σ)", kind=float,  fmt='<11.4f', infer={('dT',):lambda dT: dT,})
prototypes['Tvreduced_common'] = dict(description="Term values reduced by a common polynomial in (v+1/2) (cm-1)", kind=float,  fmt='<11.4f', infer={('v','Tv','Tvreduced_common_polynomial'): lambda v,Tv,Tvreduced_common_polynomial: Tv-np.polyval(Tvreduced_common_polynomial,v+0.5), ('v','Tv'): lambda v,Tv: Tv-np.polyval(np.polyfit(v+0.5,Tv,3),v+0.5),})
prototypes['dTvreduced_common'] = dict(description="Uncertaintty in term values reduced by a common polynomial in (v+1/2) (cm-1 1σ)", kind=float,  fmt='<11.4e', infer={('dTv',): lambda dTv: dTv})
prototypes['Tvreduced_common_polynomial'] = dict(description="Polynomial in terms of (v+1/2) to reduce all term values commonly (cm-1)", kind=object, infer={})
prototypes['Bv_μscaled']  = dict(description='Rotational constant scaled by reduced mass to an isotopologue-independent value (cm-1)' , kind =float,fmt='0.8f', infer={('Bv','reduced_mass'):lambda Bv,reduced_mass: Bv*reduced_mass,})
prototypes['dBv_μscaled'] = dict(description='Uncertainty in Bv_μscaled (1σ, cm-1)' ,kind=float,fmt='0.2g', infer={('Bv','dBv','Bv_μscaled'):lambda Bv,dBv,Bv_μscaled:dBv/Bv*Bv_μscaled,})
