from copy import deepcopy

## nonstandard library
import numpy as np
from numpy import nan

from .conversions import *
from . import tools

prototypes = {}


##########################
## first are level keys ##
##########################

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

prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind=int , infer={})
def _f(classname,J):
    """Calculate heteronuclear diatomic molecule level degeneracy."""
    if classname == 'DiatomicCinfv':
        return 2*J+1
    else:
        raise InferException('Only valid of DiatomicCinfv')
prototypes['g']['infer']['classname','J'] = _f

@tools.vectorise_function
def _f(classname,J,Inuclear,sa):
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
prototypes['g']['infer']['classname','J','Inuclear','sa'] = _f

prototypes['pm'] = dict(description="Total inversion symmetry" ,kind=int ,infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g',
                       infer={
                           # ('τ',):lambda τ: 5.309e-12/τ,
                       })
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
prototypes['partition'] = dict(description="Partition function.", kind=float, fmt='<11.3e', infer={
    ('partition_source','species','Tex'):_f5,
})

@tools.vectorise_function_in_chunks()
def _f(partition_source,species):
    from . import hitran
    if partition_source!='HITRAN':
        raise InferException(f'Partition source not "HITRAN".')
    return hitran.get_partition_function(species,296)
prototypes['partition296K'] = dict(description="The partition function at 296K", kind=float, fmt='<10.5e', infer={('partition_source','species'):_f,})


prototypes['α'] = dict(description="State population", kind=float, fmt='<11.4e', infer={('partition','E','g','Tex'): lambda partition,E,g,Tex : g*np.exp(-E/(convert(constants.Boltzmann,'J','cm-1')*Tex))/partition,})
prototypes['Nself'] = dict(description="Column density (cm2)",kind=float,fmt='<11.3e', infer={})
prototypes['label'] = dict(description="Label of electronic state", kind=str,infer={})
prototypes['v'] = dict(description="Vibrational quantum number", kind=int,infer={})
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind=int,infer={})
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy.", kind=int,infer={})
prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind=int,infer={})

prototypes['σv'] = dict(description="Symmetry with respect to σv reflection.", kind=int,infer={})
@tools.vectorise_arguments
def _f(ef,J):
    """Calculate σv symmetry"""
    exponent = np.zeros(ef.shape,dtype=int)
    exponent[ef==-1] += 1
    exponent[J%2==1] += 1
    σv = np.full(ef.shape,+1,dtype=int)
    σv[exponent%2==1] = -1
    return σv
prototypes['σv']['infer']['ef','J'] = _f

prototypes['gu'] = dict(description="Symmetry with respect to reflection through a plane perpendicular to the internuclear axis.", kind=int,infer={})

prototypes['sa'] = dict(description="Symmetry with respect to nuclear exchange, s=symmetric, a=antisymmetric.", kind=int,infer={})
@tools.vectorise_arguments
def _f(σv,gu):
    return σv*gu
prototypes['sa']['infer']['σv','gu'] = _f

prototypes['ef'] = dict(description="e/f symmetry", kind=int,infer={})
@tools.vectorise_arguments
def _f(S,Λ,s):
    """Calculate gu symmetry for 1Σ- and 1Σ+ states only."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(len(S),+1)
    ef[s==1] = -1
    return ef
prototypes['ef']['infer']['S','Λ','s'] = _f

prototypes['Fi'] = dict(description="Spin multiplet index", kind=int,infer={})
prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind=float,infer={})
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind=float,infer={})
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z (rotation) axis.", kind=float,infer={})
prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei.", kind=float,infer={})


########################################################################
## Take level keys (all prototypes defined up to this point) and make ##
## upper and lower level copys for a Lines object.                    ##
########################################################################

level_suffix = {'upper':'_u','lower':'_l'}

for key,val in list(prototypes.items()):
    tval = deepcopy(val)
    tval['infer'] = {tuple(key+level_suffix['upper']
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+level_suffix['upper']] = tval
    tval['infer'] = {tuple(key+level_suffix['lower']
                           for key in tools.ensure_iterable(dependencies)):function
                     for dependencies,function in val['infer'].items()}
    prototypes[key+level_suffix['lower']] = tval



#########################
## add more lines keys ##
#########################
    
prototypes['branch'] = dict(description="Rotational branch ΔJ.Fu.Fl.efu.efl", kind='8U', cast=str, fmt='<10s')
prototypes['ν'] = dict(description="Transition wavenumber (cm-1)", kind=float, fmt='>0.6f', infer={})
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition (cm-1 FWHM)" , kind=float, fmt='<10.5g',infer={
    ('γself','Pself','γair','Pair'):lambda γself,Pself,γair,Pair: γself*convert(Pself,'Pa','atm')+γair*convert(Pair,'Pa','atm'), # LINEAR COMBINATION!
    ('γself','Pself'):lambda γself,Pself: γself*convert(Pself,'Pa','atm'),
    ('γair','Pair'):lambda γair,Pair: γair*convert(Pair,'Pa','atm'),})
prototypes['ΓD'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind=float,fmt='<10.5g', infer={})
prototypes['f'] = dict(description="Line f-value (dimensionless)",kind=float,fmt='<10.5e', infer={('Ae','ν','g_u','g_l'):lambda Ae,ν,g_u,g_l: Ae*1.49951*g_u/g_l/ν**2,})
prototypes['σ'] = dict(description="Integrated cross section (cm2.cm-1).", kind=float, fmt='<10.5e',infer={
    ('τ','Nself_l'):lambda τ,column_densitypp: τ/column_densitypp,
    ('f','α_l'):lambda f,α_l: f/1.1296e12*α_l,
    ('S296K','Tex','partition','partition296K'):lambda S296K,Tex,pa: S296K*partition296K/partition*np.exp(-E/convert(constants.Boltzmann,'J','cm-1')(1/Tex-1/296)), # E.g.. Eq.9 of simeckova2006 -- BUT LEAVING OUT THE FINAL TERM
         })
prototypes['S296K'] = dict(description="Spectral line intensity at 296K cm-1(molecular.cm-2) as used in HITRAN -- the integrated cross section at 296K", kind=float, fmt='<10.5e', infer={})
prototypes['τ'] = dict(description="Integrated optical depth (cm-1)", kind=float, fmt='<10.5e', infer={('σ','Nself_l'):lambda σ,Nself_l: σ*Nself_l,},)
prototypes['Ae'] = dict(description="Radiative decay rate (s-1)", kind=float, fmt='<10.5g', infer={('At','Ad'): lambda At,Ad: At-Ad,})
prototypes['Teq'] = dict(description="Equilibriated temperature (K)", kind=float, fmt='0.2f', infer={})
prototypes['Tex'] = dict(description="Excitation temperature (K)", kind=float, fmt='0.2f', infer={'Teq':lambda Tex:Teq})
prototypes['Ttr'] = dict(description="Translational temperature (K)", kind=float, fmt='0.2f', infer={'Tex':lambda Tex:Tex})
prototypes['ΔJ'] = dict(description="Jp-Jpp", kind=float, fmt='>+4g', infer={('Jp','Jpp'):lambda Jp,Jpp: Jp-Jpp,},)
prototypes['partition_source'] = dict(description="Data source for computing partition function, 'self' or 'database' (default).", kind=str, infer={(): lambda : 'database',})
prototypes['partition'] = dict(description="Partition function.", kind=float, fmt='<11.3e', infer={})
# prototypes['ΓDoppler'] = dict(description="Gaussian Doppler width (cm-1 FWHM)",kind=float,fmt='<10.5g', infer={('mass','Ttr','ν'): lambda mass,Ttr,ν:2.*6.331e-8*np.sqrt(Ttr*32./mass)*ν,})
prototypes['L'] = dict(description="Optical path length (m)", kind=float, fmt='0.5f', infer={})
prototypes['γair'] = dict(description="Pressure broadening coefficient in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['δair'] = dict(description="Pressure shift coefficient in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['nair'] = dict(description="Pressure broadening temperature dependence in air (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['γself'] = dict(description="Pressure self-broadening coefficient (cm-1.atm-1.FWHM)", kind=float, cast=lambda x:abs(x), fmt='<10.5g', infer={},)
prototypes['Pself'] = dict(description="Pressure of self (Pa)", kind=float, fmt='0.5f', infer={})
prototypes['Pair'] = dict(description="Pressure of air (Pa)", kind=float, fmt='0.5f', infer={})
prototypes['Nself'] = dict(description="Column density (cm-2)",kind=float,fmt='<11.3e', infer={('Pself','L','Teq'): lambda Pself,L,Teq: (Pself*L)/(database.constants.Boltzmann*Teq)*1e-4,})



##############################
## Add more infer functions ##
##############################


prototypes['ν']['infer']['E_u','E_l'] = lambda Eu,El: Eu-El
prototypes['E_l']['infer']['E_u','ν'] = lambda Eu,ν: Eu-ν
prototypes['E_u']['infer']['E_l','ν'] = lambda El,ν: El+ν
prototypes['Γ']['infer']['Γ_u','Γ_l'] = lambda Γu,Γl: Γu+Γl
prototypes['Γ_l']['infer']['Γ','Γ_u'] = lambda Γ,Γu: Γ-Γu
prototypes['Γ_u']['infer']['Γ','Γ_l'] = lambda Γ,Γl: Γ-Γl
prototypes['J_u']['infer']['J_l','ΔJ'] = lambda J_l,ΔJ: J_l+ΔJ
prototypes['Tex']['infer']['Teq'] = lambda Teq: Teq
prototypes['Teq_u']['infer']['Teq'] = lambda Teq: Teq
prototypes['Teq_l']['infer']['Teq'] = lambda Teq: Teq
prototypes['Nself_u']['infer']['Nself'] = lambda Nself: Nself
prototypes['Nself_l']['infer']['Nself'] = lambda Nself: Nself
prototypes['species_l']['infer']['species'] = lambda species: species
prototypes['species_u']['infer']['species'] = lambda species: species
prototypes['ΔJ']['infer']['J_u','J_l'] = lambda J_u,J_l: J_u-J_l
prototypes['partition']['infer']['partition_l'] = lambda partition_l:partition_l
prototypes['partition']['infer']['partition_u'] = lambda partition_u:partition_u
prototypes['partition_l']['infer']['partition'] = lambda partition:partition
prototypes['partition_u']['infer']['partition'] = lambda partition:partition
