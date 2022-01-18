## standard libraries
import itertools 
import functools
from copy import copy,deepcopy
import re
from pprint import pprint
import hashlib
import pickle
import warnings

## nonstandard libraries
from scipy import constants
import numpy as np
from numpy import nan,array

## this module
from .dataset import Dataset
from . import convert
from .tools import vectorise,cache,file_to_dict,cast_abs_float_array
from . import tools
from . import database
from . import kinetics
from . import quantum_numbers
from .exceptions import InferException,DatabaseException
from .optimise import optimise_method,Parameter

prototypes = {}

prototypes['notes'] = dict(description="Notes regarding this line" , kind='U' ,infer=[])
prototypes['author'] = dict(description="Author of data or printed file" ,kind='U' ,infer=[])
prototypes['reference'] = dict(description="Reference",kind='U',infer=[])
prototypes['date'] = dict(description="Date data collected or printed" ,kind='U' ,infer=[])


def _f0(species):
    """Cast species using normalise species if the data is not too
    much (it is slow)."""
    if np.isscalar(species):
        ## a scalara species -- normalise it 
        return kinetics.get_species(species).name
    elif len(species) < 10000:
        ## a short list of species -- normalise them
        return database.normalise_species(species)
    else:
        ## too long -- no normalisation -- WITHOUT WARNING
        return np.asarray(species,dtype=str)
prototypes['species'] = dict(description="Chemical species with isotope specification" ,kind='U',infer=[],
                             # cast=database.normalise_species,
                             cast=_f0,)
prototypes['_species_hash'] = dict(description="Hash of species", kind='i',infer=[('species',lambda self,species:[hash(t) for t in species]),])

@vectorise(cache=True,vargs=(1,))
def _f0(self,species):
    species_object = kinetics.get_species(species)
    return species_object['chemical_name']
prototypes['chemical_species'] = dict(description="Chemical species without isotope specification" ,kind='U' ,infer=[('species',_f0)])
prototypes['point_group']  = dict(description="Symmetry point group of species", kind='U',fmt='s', infer=[(('species',),lambda self,species:database.get_species_property(species,'point_group'))])

@vectorise(vargs=(1,),dtype=float)
def _f0(self,species):
    return kinetics.get_species(species)['mass']
def _f1(self,species,_species_hash):
    mass = np.empty(len(species),dtype=float)
    for t,i in zip(*np.unique(_species_hash,return_index=True)):
        j = _species_hash == t
        mass[j] = kinetics.get_species(species[i])['mass']
    return mass
prototypes['mass'] = dict(description="Mass",units="amu",kind='f', fmt='<11.4f', infer=[(('species','_species_hash',), _f1), (('species',), _f0),])
prototypes['reduced_mass'] = dict(description="Reduced mass",units="amu", kind='f', fmt='<11.4f', infer=[(('species','database',), lambda self,species: _get_species_property(species,'reduced_mass'))])

## level energies
prototypes['E'] = dict(description="Level energy referenced to Eref",units='cm-1',kind='f' ,fmt='<14.7f',default_step=1e-3 ,infer=[(('Ee','E0','Eref'),lambda self,Ee,E0,Eref: Ee-E0-Eref), (('species','_qnhash','Eref'),lambda self,species,_qnhash,Eref: database.get_level_energy(species,Eref,_qnhash=_qnhash)),]) 
prototypes['Ee'] = dict(description="Level energy relative to equilibrium geometry at J=0 and neglecting spin" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','E0'),lambda self,E,E0: E+E0),],default_step=1e-3)
prototypes['Eref'] = dict(description="Reference energy referenced to the lowest physical energy level" ,units='cm-1',kind='f' ,fmt='<14.7f',default=0,infer=[])
prototypes['Eexp'] = dict(description="Experimental level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','Eres'),lambda self,E,Eres: E+Eres)])
prototypes['Eres'] = dict(description="Residual difference between level energy and experimental level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('E','Eexp'),lambda self,E,Eexp: Eexp-E)])
prototypes['E0'] = dict(description="Energy of the lowest physical energy level relative to Ee" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[('species',lambda self,species: database.get_species_property(species,'E0')),],default_step=1e-3)


prototypes['term'] = dict(description="Spectroscopic term symbol",kind='U',cast=lambda term: np.array(quantum_numbers.normalise_term_symbol(term),dtype=str),infer=[])
prototypes['lande_g'] = dict(description="Lande g factor",units='dimensionless',kind='f' ,fmt='6.5f',infer=[]) 

def _f0(self,species,label,v,Σ,ef,J,E):
    """Compute separate best-fit reduced energy levels for each
    sublevel rotational series."""
    if hasattr(self,'reduced_order'):
        order = self.reduced_order
    else:
        order = 3
    Ereduced = np.full(E.shape,0.0)
    for di,i in tools.unique_combinations_masks(species,label,v,Σ,ef):
        speciesi,labeli,vi,Σi,efi = di
        Ji,Ei = [],[]
        for Jj,j in tools.unique_combinations_masks(J[i]):
            Ji.append(Jj[0])
            Ei.append(E[i][j][0])
        Ji,Ei = array(Ji),array(Ei)
        pi = np.polyfit(Ji*(Ji+1),Ei,min(order,len(Ei)-1))
        if self.verbose:
            print(f'{species=} {label=} {v=} {Σ=} {ef=} {pi=}')
        Ereduced[i] = E[i] - np.polyval(pi,J[i]*(J[i]+1))
    return Ereduced
def _df0(self,Ereduced,species,dspecies,label,dlabel,v,dv,Σ,dΣ,ef,ddef,J,dJ,E,dE):
    """Uncertainty calculation to go with _f0."""
    if dE is None:
        raise InferException()
    dEreduced = dE
    return dEreduced
prototypes['Ereduced'] = dict(description="Reduced level energy" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('species','label','v','Σ','ef','J','E'),(_f0,_df0)),],)

def _f0(self,JJ,E):
    """Compute separate best-fit reduced energy levels for each
    sublevel rotational series."""
    Ereduced_common = E - np.polyval(np.polyfit(JJ,E,1), JJ)
    return Ereduced_common
def _df0(self,Ereduced_common,JJ,dJJ,E,dE):
    """Uncertainty calculation to go with _f0."""
    if dE is None:
        raise InferException()
    dEreduced_common = dE
    return dEreduced_common
prototypes['Ereduced_common'] = dict(description="Reduced level energy common to all bands" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[(('JJ','E'),(_f0,_df0)),],)


## infer function for Ereduced_by_JJ etc is set at import time by
## _collect_prototypes
prototypes['Ereduced_JJ'] = dict(description="Level energy reduced by a best-fit polynomial in terms of J(J+1)" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[],)
prototypes['Ereduced_vv'] = dict(description="Level energy reduced by a best-fit polynomial in terms of (v+1/2)" ,units='cm-1',kind='f' ,fmt='<14.7f' ,infer=[],)


@vectorise(cache=True,vargs=(1,))
def _f0(self,point_group):
    """Calculate heteronuclear diatomic molecule level degeneracy"""
    if point_group in ('K','C∞v'):
        return 1.
    else:
        raise InferException(f'Trivial gnuclear only possible from point_group in (K,C∞v)')
@vectorise(cache=True,vargs=(1,2,3))
def _f1(self,point_group,Inuclear,sa):
    """Calculate homonuclear diatomic molecule level degeneracy"""
    if point_group in ('D∞h'):
        ## get total number of even or odd exchange combinations
        ntotal = (2*Inuclear+1)**2
        neven = 2*Inuclear+1 + (ntotal-(2*Inuclear+1))/2
        nodd = ntotal - neven
        if Inuclear%1==0:
            ## fermion
            if sa==+1:
                return neven
            else:
                return nodd
        else:
            ## boson
            if sa==+1:
                return nodd
            else:
                return neven
    else:
        raise InferException()
prototypes['gnuclear'] = dict(description="Nuclear spin level degeneracy (relative only)" , kind='i' , infer=[(('point_group',),_f0),( ('point_group','Inuclear','sa'),_f1),])
prototypes['Inuclear'] = dict(description="Nuclear spin of individual nuclei", kind='f',infer=[(('species',), lambda self,species: database.get_species_property(species,'Inuclear'))])
prototypes['g'] = dict(description="Level degeneracy including nuclear spin statistics" , kind='i' , infer=[(('J','gnuclear'),lambda self,J,gnuclear: (2*J+1)*gnuclear,)])
# prototypes['pm'] = dict(description="Total inversion symmetry" ,kind='i' ,infer=[])
prototypes['Γ'] = dict(description="Total natural linewidth of level or transition" ,units="cm-1 FWHM",kind='f',cast=cast_abs_float_array,fmt='<10.5g', infer=[('At',lambda self,At: 5.309e-12*At,)])
prototypes['Γexp'] = dict(description="Reference level natural linewidth" ,units='cm-1.FWHM',kind='f' ,fmt='<14.7f' ,infer=[])
prototypes['Γres'] = dict(description="Residual error of level natural linewidth" ,units='cm-1.FWHM',kind='f' ,fmt='<14.7f' ,infer=[(('Γ','Γexp'),lambda self,Γ,Γexp: Γ-Γexp)])
prototypes['Γd'] = dict(description="Dissociation width of level" ,units="cm-1 FWHM",kind='f',default=0.0,cast=cast_abs_float_array,fmt='<10.5g', infer=[('Ad',lambda self,Ad: 5.309e-12*Ad)])
prototypes['Γe'] = dict(description="Emission width of level" ,units="cm-1 FWHM",kind='f',default=0.0,cast=cast_abs_float_array,fmt='<10.5g', infer=[('Ae',lambda self,Ae: 5.309e-12*Ae)])
prototypes['τ'] = dict(description="Total decay lifetime",units="s", kind='f', cast=cast_abs_float_array,infer=[(('A',), lambda self,A: 1/A,)])       
prototypes['At'] = dict(description="Total decay rate",units="s-1", kind='f', fmt='0.5e',cast=cast_abs_float_array,infer=[
    (('Γ',),lambda self,Γ: Γ/5.309e-12), 
    (('Ae','Ad'),lambda self,Ae,Ad: Ae+Ad),
],)
prototypes['Ae'] = dict(description="Total emissive decay rate",units="s-1", kind='f', fmt='0.5e',cast=cast_abs_float_array,infer=[('Γe',lambda self,Γ: Γ/5.309e-12),(('At','Ad'),lambda self,At,Ad: At-Ad)])
prototypes['Ad'] = dict(description="Total dissociative decay rate",units="s-1", kind='f', fmt='0.5e',cast=cast_abs_float_array,infer=[('Γd',lambda self,Γ: Γ/5.309e-12),(('At','Ae'),lambda self,At,Ae: At-Ae)])
prototypes['ηd'] = dict(description="Fractional probability dissociative decay",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ad'),lambda self,At,Ad:Ad/At,)])
prototypes['ηe'] = dict(description="Fractional probability emissive decay",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ae'),lambda self,At,Ae:Ae/At,)])

prototypes['J'] = dict(description="Total angular momentum quantum number excluding nuclear spin" , kind='f',fmt='g',infer=[])
prototypes['JJ'] = dict(description="J(J+1)" , kind='f',fmt='g',infer=[('J',lambda self,J:J*(J+1))])
prototypes['N'] = dict(description="Angular momentum excluding nuclear and electronic spin", kind='f', infer=[(('J','SR'),lambda self,J,SR: J-SR,)])
prototypes['S'] = dict(description="Total electronic spin quantum number", kind='f',fmt='g',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'S'),)])
# prototypes['Eref'] = dict(description="Reference point of energy scale relative to potential-energy minimum",units='cm-1', kind='f',infer=[((),lambda self,: 0.,)])
prototypes['Teq'] = dict(description="Equilibriated temperature",units="K", kind='f', fmt='0.2f', infer=[],cast=cast_abs_float_array,default_step=0.1)
prototypes['Tex'] = dict(description="Excitation temperature",units="K", kind='f', fmt='0.2f', infer=[('Teq',lambda self,Teq:Teq)],cast=cast_abs_float_array,default_step=0.1)
prototypes['Tvib'] = dict(description="Vibrational excitation temperature",units="K", kind='f', fmt='0.2f', infer=[],cast=cast_abs_float_array,default_step=0.1)
prototypes['Trot'] = dict(description="Rotational excitation temperature",units="K", kind='f', fmt='0.2f', infer=[],cast=cast_abs_float_array,default_step=0.1)
prototypes['conf'] = dict(description="Electronic configuration", kind='U', fmt='10s', infer=[])

# @vectorise(cache=True,vargs=(1,2))

## partition function
_valid_Zsource = (
    "self",                     # calculate from levels in self
    "HITRAN",                   # access HITRAN through hapy
    "database",                 # use the internal database of levels
    'unity',                    # set to 1
)
_Zsource_char_length = max(*[len(t) for t in _valid_Zsource])
def _f2(Zsource):
    retval = np.asarray(Zsource,dtype=f'U{_Zsource_char_length}')
    for Zsourcei in np.unique(Zsource):
        if Zsourcei not in _valid_Zsource:
            raise Exception(f'Invalid Zsource: {Zsourcei!r}. Valid: {_valid_Zsource!r}')
    return retval
prototypes['Zsource'] = dict(description=f'Source of partition function (valid: {_valid_Zsource})', cast=_f2,kind='U', fmt='8s',default='self',infer=[])

def _f6(self,Tex,Zsource):
    """Get partition function set to unity."""
    if np.any(Zsource != 'unity'):
        raise InferException(f'Zsource not all "unity"')
    Z = np.full(len(self),1.0)
    return Z
def _f5(self,species,Tex,Eref,Zsource):
    """Get HITRAN partition function."""
    if np.any(Zsource != 'HITRAN'):
        raise InferException(f'Zsource not all "HITRAN"')
    if np.any(Eref != 0):
        raise InferException(f'Cannot use "HITRAN" Zsource when Eref is not 0.')
    from . import hitran
    Z = hitran.get_partition_function(species,Tex)
    return Z
def _f4(self,species,Tex,Eref,Zsource):
    """Get partition function from internal database."""
    if np.any(Zsource != 'database'):
        raise InferException(f'Zsource not all "database"')
    Z = database.get_partition_function(species,Tex,Eref)
    return Z
def _f3(self,species,Tex,E,Eref,g,_qnhash,Zsource):
    """Compute partition function from data in self. For unique
    combinations of T/species sum over unique level energies. Always
    referenced to E0."""
    if np.any(Zsource != 'self'):
        raise InferException(f'Zsource not all "self"')
    if len(np.unique(Tex)) > 1:
        raise InferException("Non-unique Tex")
    retval = np.full(species.shape,nan)
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    for speciesi,i in tools.unique_combinations_masks(species):
        t,j = np.unique(_qnhash[i],return_index=True)
        retval[i] = np.sum(g[i][j]*np.exp(-(E[i][j]-Eref[i][j])/(kB*Tex[0])))
    return retval
prototypes['Z'] = dict(description="Partition function", kind='f', fmt='<11.3e', infer=[
    (('species','Tex','E','Eref','g','_qnhash','Zsource'),_f3),
    (('species','Tex','Eref','Zsource'),_f5),
    (('species','Tex','Eref','Zsource'),_f4),
    (('Tex','Zsource'),_f6),
])

def _f6(self,species,Tvib,Eref,Tv,v,_qnhash,Zsource):
    """Compute vibrational partition function from data in self with
     separate vibrational temperature. Compute separately for
     different species and sum over unique levels only."""
    if np.any(Zsource != 'self'):
        raise InferException(f'Zsource not all "self"')
    Zvib = np.full(species.shape,nan)
    if len(np.unique(Tvib)) > 1:
        raise InferException("Non-unique Tvib")
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    for speciesi,i in tools.unique_combinations_masks(species):
        t,j = np.unique(_qnhash[i],return_index=True)
        t,k = np.unique(v[i][j],return_index=True)
        Zvib[i] = np.sum(np.exp(-(Tv[i][j][k]-Eref[i][j][k])/(kB*Tvib[0])))
    return Zvib
def _f7(self,Tvib,Zsource):
    """Get vibrational partition function set to unity."""
    if np.any(Zsource != 'unity'):
        raise InferException(f'Zsource not all "unity"')
    Zvib = np.full(len(self),1.0)
    return Zvib
prototypes['Zvib'] = dict(description="Vibrational partition function", kind='f', fmt='<11.3e', infer=[
    (('species','Tvib','Eref','Tv','v','_qnhash','Zsource'),_f6),
    (('Tvib','Zsource'),_f7),
])

def _f6(self,species,Trot,E,Tv,g,v,_qnhash,Zsource):
    """Compute partition function from data in self with separate
     vibrational temperature. Compute separately for
    different species and sum over unique levels only."""
    if np.any(Zsource != 'self'):
        raise InferException(f'Zsource not all "self"')
    if len(np.unique(Trot)) > 1:
        raise InferException("Non-unique Trot")
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    Zrot = np.full(species.shape,nan)
    for (speciesi,vi),i in tools.unique_combinations_masks(species,v):
        t,j = np.unique(_qnhash[i],return_index=True)
        Zrot[i] = np.sum(np.exp(-(E[i][j]-Tv[i][j])/(kB*Trot[0])))
    return Zrot
def _f7(self,Trot,Zsource):
    """Get vibrational partition function set to unity."""
    if np.any(Zsource != 'unity'):
        raise InferException(f'Zsource not all "unity"')
    Zrot = np.full(len(self),1.0)
    return Zrot
prototypes['Zrot'] = dict(description="Vibrational partition function", kind='f', fmt='<11.3e', infer=[
    (('species','Trot','E','Tv','g','v','_qnhash','Zsource'),_f6),
    (('Trot','Zsource'),_f7),
])

## level populations
def _f0(self,Z,E,Eref,g,Tex):
    """Compute level population from equilibrium excitation temperature."""
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    α = g*np.exp(-(E-Eref)/(kB*Tex))/Z
    return α
prototypes['α'] = dict(description="State population", kind='f', fmt='<11.4e', infer=[
    (('Z','E','Eref','g','Tex',), _f0), 
    (('αvib','αrot'),lambda self,αvib,αrot: αvib*αrot),])
def _f0(self,Zvib,Tv,Eref,Tvib):
    """Compute vibrational level population from vibrational
    temperature."""
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    αvib = np.exp(-(Tv-Eref)/(kB*Tvib))/Zvib
    return αvib
prototypes['αvib'] = dict(description="Vibrational state population", kind='f', fmt='<11.4e', infer=[(('Zvib','Tv','Eref','Tvib',), _f0),])
def _f0(self,Zrot,Tv,E,g,Trot):
    """Compute rotational level population from rotational
    temperature."""
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    αrot = g*np.exp(-(E-Tv)/(kB*Trot))/Zrot
    return αrot
prototypes['αrot'] = dict(description="Rotational state population", kind='f', fmt='<11.4e', infer=[(('Zrot','Tv','E','g','Trot',), _f0),])
def _f0(self,species,E,Eref,g,Zsource):
    """Get level populations at 296K assuming a single excitation
    temperature. CURRENTLY ONLY IMPLEMENTED FOR ZSOURCE=HITRAN."""
    if np.any(Zsource != 'HITRAN'):
        raise InferException(f'Can only compute α296K if Zsource all "HITRAN"')
    Tex = 296
    from . import hitran
    Z = hitran.get_partition_function(species,Tex)
    kB = convert.units(constants.Boltzmann,'J','cm-1')
    α = g*np.exp(-(E-Eref)/(kB*Tex))/Z
    return α
prototypes['α296K'] = dict(description="Equilibrium level population at 296K",units="dimensionless", kind='f', fmt='<10.5e',cast=tools.cast_abs_float_array,infer=[(('species','E','Eref','g','Zsource'),_f0),])

## should these columns even be in levels?
prototypes['Nchemical_species'] = dict(description="Combined column density of all isotopolouges of this chemical species",units="cm-2",kind='a',fmt='<11.3e', infer=[])
prototypes['Nspecies'] = dict(description="Column density of this species",units="cm-2",kind='a',fmt='<11.3e', infer=[(('Nchemical_species','isotopologue_ratio'),lambda self,Nchemical_species,isotopologue_ratio: Nchemical_species*isotopologue_ratio),])
prototypes['isotopologue_ratio'] = dict(description="Ratio of this isotopologue to the all isotopologues combined",units="cm-2",kind='a',fmt='<11.3e', infer=[
    # (('species'),lambda self,species: database.get_species_property(species,'isotopologue_ratio')),
    (('Nspecies','Nchemical_species'),lambda self,Nspecies,Nchemical_species: Nspecies/Nchemical_species),
])
prototypes['label'] = dict(description="Label of electronic state", kind='U',infer=[])
prototypes['v'] = dict(description="Vibrational quantum number", kind='i',infer=[])
prototypes['vv'] = dict(description="(v+1/2)", kind='i',infer=[('v',lambda self,v:v+1/2)])
prototypes['ν1'] = dict(description="Vibrational quantum number for mode 1", kind='i',infer=[])
prototypes['ν2'] = dict(description="Vibrational quantum number for mode 2", kind='i',infer=[])
prototypes['ν3'] = dict(description="Vibrational quantum number for mode 3", kind='i',infer=[])
prototypes['ν4'] = dict(description="Vibrational quantum number for mode 4", kind='i',infer=[])
prototypes['l2'] = dict(description="Vibrational angular momentum 2", kind='i',infer=[])
prototypes['Λ'] = dict(description="Total orbital angular momentum aligned with internuclear axis", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'Λ'))]) # 
prototypes['L'] = dict(description="Total orbital angular momentum", kind='i',infer=[])

def _f0(self,Av):
    """Determine LSsign from sign of spin-orbit constant. Always right?"""
    if np.any(np.isnan(Av)|(Av==0)):
        raise InferException("Cannot determine LSsign from NaN or zero Av.")
    LSsign = np.sign(Av)
    return LSsign
def _f1(self,λv,Bv):
    """Determine LSsign from sign of spin-orbit constant. Always right?"""
    if np.any(np.isnan(λv)|(λv==0)):
        raise InferException("Cannot determine LSsign from NaN or zero λv.")
    LSsign = np.sign(λv-Bv)
    return LSsign
prototypes['LSsign'] = dict(description="For Λ>0 states this is the sign of the spin-orbit interacting energy. For Λ=0 states this is the sign of λ-B. In either case it controls whether the lowest Σ level is at the highest or lower energy", kind='i',
                            infer=[(('species','label'),lambda self,species,label: database.get_electronic_state_property(species,label,'LSsign')),
                                   ('Av',_f0),
                                   (('λv','Bv',),_f1),])

prototypes['s'] = dict(description="s=1 for Σ- states and 0 for all other states", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'s'))])

## inversion symmetry
# @vectorise(cache=True,vargs=(1,2))
def _f0(self,ef,J):
    i = np.full(ef.shape,+1,dtype=int)
    i[ef==-1] *= -1
    i[J%2==1] *= -1
    return i
prototypes['i'] = dict(description="Sign change on inversion, total parity", kind='i',infer=[(('ef','J'),_f0,)])
prototypes['σv'] = dict(description="Linear molecule sign change on σv reflection", kind='i',infer=[('i',lambda self,i:i,)])
prototypes['gu'] = dict(description="Sign change on inversion through the centre of symmetry", kind='i',infer=[(('chemical_species','label'),lambda self,chemical_species,label: database.get_electronic_state_property(chemical_species,label,'gu'))], cast=quantum_numbers.decode_gu,)
prototypes['sa'] = dict(description="Linear molecule sign change on nuclear exchange)", kind='i',infer=[(('i','gu'),lambda self,i,gu: i*gu)])
def _f0(self,S,Λ,s):
    """Calculate ef symmetry for non-degenerate 1Σ- and 1Σ+ states."""
    if np.any(S!=0) or np.any(Λ!=0):
        raise InferException('ef for Sp!=0 and Λ!=0 not implemented')
    ef = np.full(S.shape,+1,dtype=int)
    ef[s==1] = -1
    return ef
def _f1(self,i,J):
    ef = np.full(i.shape,+1,dtype=int)
    ef[i==-1] *= -1
    ef[J%2==1] *= -1
    return ef
prototypes['ef'] = dict(description="Sign change on inversion without rotational factor (-1)**J ", kind='i',infer=[(('i','J'),_f1),(('S','Λ','s'),_f0,)],fmt='+1d')

# def _f0(self,S,SR,Λ,s,ef):
    # Fi = np.full(S.shape,np.nan)
    # ## Π-state and above order Fi according to SR
    # i = Λ>0
    # Fi[i] = S[i]-SR[i]+1.
    # # ## special case Σ± states -- odd-man-out parity state in the middle
    # ## special case even-electron Σ states -- odd-man-out parity state in the middle
    # i = (Λ==0)&(S%1==0)
    # ef0 = 
    # i = ~i
    # Fi[i] = S[i]+SR[i]+1.
    # print('DEBUG:', S)
    # print('DEBUG:', SR)
    # print('DEBUG:', ef)
    # print('DEBUG:', Fi)
    # import pdb; pdb.set_trace(); # DEBUG
    # if np.any(np.isnan(Fi)):
        # raise InferException('Failed to computed Fi')
    # return Fi
prototypes['Fi'] = dict(description="Spin multiplet index", kind='i',infer=[
    ('sublevel',lambda self,sublevel: [int(t[:-1]) for t in sublevel]),
    (('S','SR'),lambda self,S,SR: S-SR+1,)
    # (('S','SR','Λ','s','ef'),_f0),
])

prototypes['Ω'] = dict(description="Absolute value of total angular momentum aligned with internuclear axis", kind='f',fmt='g',infer=[(('Λ','Σ'),lambda self,Λ,S: Λ+S,)])

def _f3(self,S):
    Σ = np.full(S.shape,np.nan)
    Σ[S==0] = 0             # singlet state
    if np.any(np.isnan(Σ)): raise InferException('Could not determine Σ solely from S.')
    return(Σ)
def _f6(self,S,Λ):
    Σ = np.full(S.shape,np.nan)
    Σ[S==0] = 0             # singlet state
    Σ[(S==0.5)&(Λ==0)] = 0.5         # 2Σ± state
    if np.any(np.isnan(Σ)): raise InferException('Could not determine Σ solely from S.')
    return(Σ)
def _f4(self,Λ,S,SR,LSsign):
    if np.any((Λ==0)&(S>1)): raise Exception("Not implemented")
    Σ = np.full(Λ.shape,np.nan)
    Σ[S==0] = 0.
    ## special case Σ states since sign of Σ is not well defined
    Σ[(Λ==0)&(S==1/2)] = 1/2
    Σ[(Λ==0)&(S==1)&(SR==0)] = 1
    Σ[(Λ==0)&(S==1)&(SR==+1)&(LSsign==+1)] = 0 # regular 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==+1)&(LSsign==-1)] = 1 # regular 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==-1)&(LSsign==+1)] = 1 # inverted 3Σ states
    Σ[(Λ==0)&(S==1)&(SR==-1)&(LSsign==-1)] = 0 # inverted 3Σ states
    ## general case -- frequently wrong
    i = np.isnan(Σ)
    Σ[i] = -SR[i]*LSsign[i]
    # Σ[i] = SR[i]*LSsign[i]
    return(Σ)
def _f5(self,Λ,Ω,S):
    Σ = np.full(Λ.shape,np.nan)
    i = Λ==0
    Σ[i] = np.abs(Ω[i])
    i = S<=Λ
    Σ[i] = Ω[i]-Λ[i]
    if np.any(np.isnan(Σ)): raise InferException()
    return(Σ)
prototypes['Σ'] = dict(description="Signed spin projection in the direction of Λ, or strictly positive if Λ=0", kind='f',
                       infer=[
                           (('S',), _f3),                  # trivial S==0
                           (('S','Λ'), _f6),               # trivial S==0 or 2Σ
                           (('Λ','Ω','S'), _f5),           # mostly works
                           (('Λ','S','SR','LSsign'), _f4), # general case
                       ],fmt='+g',)


def _f6(self,Λ,S,Σ,s,ef,LSsign):
    if np.any((Λ==0)&(S>1)): raise Exception("Not implemented")
    SR = np.full(Λ.shape,np.nan)
    SR[S==0] = 0                # trivial case
    ## special cases Σ states since sign of Σ is not well defined
    SR[(Λ==0)&(S==1/2)&(s==0)&(ef==+1)] = +1/2
    SR[(Λ==0)&(S==1/2)&(s==1)&(ef==-1)] = +1/2
    SR[(Λ==0)&(S==1/2)&(s==0)&(ef==-1)] = -1/2
    SR[(Λ==0)&(S==1/2)&(s==1)&(ef==+1)] = -1/2
    i = (Λ==0)&(S==1)&(s==0)&(Σ==0)&(ef==-1); SR[i] = +1*LSsign[i] # 3Σ+(Σ=0,f)
    i = (Λ==0)&(S==1)&(s==0)&(Σ==1)&(ef==+1); SR[i] = 0 # 3Σ+(Σ=1,e)
    i = (Λ==0)&(S==1)&(s==0)&(Σ==1)&(ef==-1); SR[i] = -1*LSsign[i] # 3Σ+(Σ=1,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==0)&(ef==+1); SR[i] = +1*LSsign[i] # 3Σ+(Σ=0,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==1)&(ef==-1); SR[i] = 0 # 3Σ-(Σ=1,f)
    i = (Λ==0)&(S==1)&(s==1)&(Σ==1)&(ef==+1); SR[i] = -1*LSsign[i] # 3Σ+(Σ=1,f)
    ## general case
    i = np.isnan(SR)
    SR[i] = -Σ[i]*LSsign[i]
    return SR
def _f5(self,S):
    if not np.all(S==0): raise InferException()
    return(np.zeros(S.shape))
prototypes['SR'] = dict(description="Signed projection of spin angular momentum onto the molecule-fixed z axis", kind='f',infer=[
    (('S',), _f5), # trivial case, S=0
    (('J','N'),lambda self,J,N: J-N),
    (('S','Fi'),lambda self,S,Fi: S-Fi+1),                   # Fi ordering follows decreasing SR
    (('Λ','S','Σ','s','ef','LSsign'), _f6), # most general case
])

## derived from defining quantum numbers
prototypes['_qnhash'] = dict(description="Hash of defining quantum numbers", kind='i',infer=[])
prototypes['qn_encoded'] = dict(description="String-encoded defining quantum numbers", kind='U',infer=[])

## Effective Hamiltonian parameters
prototypes['Tv']  = dict(description='Term origin' ,units='cm-1',kind='f',fmt='0.6f',default=0,infer=[])
prototypes['Tv']  = dict(description='Electronic-vibrational energy.' ,units='cm-1',kind='f',fmt='0.6f',default=0,infer=[])
prototypes['Bv']  = dict(description='Rotational constant' ,units='cm-1',kind='f',fmt='0.8f',default=0,infer=[])
prototypes['Dv']  = dict(description='Centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Hv']  = dict(description='Third order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Lv']  = dict(description='Fourth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Mv']  = dict(description='Fifth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Nv']  = dict(description='Sixth order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Ov']  = dict(description='Seventh order centrifugal distortion' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['Av']  = dict(description='Spin-orbit energy' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['ADv'] = dict(description='Spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['AHv'] = dict(description='Higher-order spin-orbit centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['λv']  = dict(description='Spin-spin energy',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['λDv'] = dict(description='Spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['λHv'] = dict(description='Higher-order spin-spin centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['γv']  = dict(description='Spin-rotation energy' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['γDv'] = dict(description='Spin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['γHv'] = dict(description='Higher-orders pin-rotation centrifugal distortion',units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['ov']  = dict(description='Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['oDv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['oHv']  = dict(description='Higher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['oLv']  = dict(description='Ligher-order Λ-doubling constant o' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['pv']  = dict(description='Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['pDv']  = dict(description='Higher-order Λ-doubling constant p' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['qv']  = dict(description='Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])
prototypes['qDv']  = dict(description='Higher-order Λ-doubling constant q' ,units='cm-1',kind='f',fmt='0.6g',default=0,infer=[])

prototypes['Γv'] = dict(description="Total electronic-vibrational linewidth",units="cm-1 FWHM", kind='f',  fmt='<10.5g', strictly_positive=True, infer=[(('τ',),lambda self,τ: 5.309e-12/τ,)]) # tau=1/2/pi/gamma/c
prototypes['τv'] = dict(description="Total electronic-vibrational decay lifetime",units="s", kind='f',  fmt='<10.5g', infer=[(('Γv',), lambda self,Γv: 5.309e-12/Γv),( ('Atv',), lambda self,Atv: 1/Atv,)]) 
prototypes['Atv'] = dict(description="Total electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('τv',), lambda self,τv: 1/τv), (('Adv','Ave'), lambda self,Adv,Aev: Adv+Aev,), (('Aev',), lambda self,Aev: Aev),( ('Adv',), lambda self,Adv: Adv,)])# Test for Ad and Ae, if failed then one or the other is undefined/zero
prototypes['Adv'] = dict(description="Nonradiative electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('At','Ae'), lambda self,At,Ae: At-Ae,)])
prototypes['Aev'] = dict(description="Radiative electronic-vibrational decay rate",units="s-1", kind='f',  fmt='<10.5g', infer=[(('At','Ad'), lambda self,At,Ad: At-Ad,)])
prototypes['ηdv'] = dict(description="Fractional probability of electronic-vibrational level decaying nonradiatively",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ad'),lambda self,At,Ad:Ad/A,)])
prototypes['ηev'] = dict(description="Fractional probability of electronic-vibrational level decaying radiatively",units=None, kind='f',  fmt='<10.5g', infer=[(('At','Ae'),lambda self,At,Ae:Ae/A,)])

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
prototypes['Tvreduced'] = dict(description="Vibrational term value reduced by a polynomial in v+1/2",units='cm-1', kind='f',  fmt='<11.4f',
    infer=[(('self','reduced_quantum_number','reduced_polynomial_order','Tv','dTv'), _vibrationally_reduce), # dTv is known -- use in a weighted mean
           (('self','reduced_quantum_number','reduced_polynomial_order','Tv'), _vibrationally_reduce,)]) # dTv is not known
prototypes['Tvreduced_common'] = dict(description="Term values reduced by a common polynomial in v+1/2",units='cm-1', kind='f',  fmt='<11.4f', infer=[(('v','Tv','Tvreduced_common_polynomial'), lambda self,v,Tv,Tvreduced_common_polynomial: Tv-np.polyval(Tvreduced_common_polynomial,v+0.5)),(('v','Tv'), lambda self,v,Tv: Tv-np.polyval(np.polyfit(v+0.5,Tv,3),v+0.5),)])
prototypes['Tvreduced_common_polynomial'] = dict(description="Polynomial in terms of v+1/2 to reduce all term values commonly",units='cm-1', kind='o', infer=[])
prototypes['Bv_μscaled']  = dict(description='Rotational constant scaled by reduced mass to an isotopologue-independent value' ,units='cm-1', kind='f',fmt='0.8f', infer=[(('Bv','reduced_mass'),lambda self,Bv,reduced_mass: Bv*reduced_mass,)])

def _get_key_from_qn(self,qn,key):
    try:
        return [self.decode_qn(t)[key] for t in qn]
    except KeyError as err:
        raise InferException('Could not determine from qn: {str(err)}')

def _qn_hash(self,*qn):
    """Compute qn hash."""
    _qnhash = [hash(qni) for qni in zip(*qn)]
    # _qnhash = np.empty(len(qn[0]),dtype=int)
    # for i,qni in enumerate(zip(*qn)):
        # qni = tuple([t.strip() if isinstance(t,str) else t for t in qni])
        # _qnhash[i] = hash(qni)
    return _qnhash

def _calc_reduced(self,x,y,*z):
    """Compute reduced y in terms of y. Separate reductions for unique
    combinations of z arrays.  Note that if multiple (x,y,z) points
    exist only the first is used to compute a polynomial fit.."""
    self.global_attributes.setdefault('reduced_order',3)
    order = self.global_attributes['reduced_order']
    yreduced = np.full(y.shape,0.0)
    ## loop through unique z combinations, fitting (x,y) polynomial to
    ## each subset
    for zi,i in tools.unique_combinations_masks(*z):
        ## find unique x and corresponding y (takes the first if
        ## multiple similar x)
        xi,yi = [],[]
        for xj,j in zip(*np.unique(x[i],return_index=True)):
            xi.append(xj)
            yi.append(y[i][j])
        pi = np.polyfit(xi,yi,min(order,len(yi)-1))
        yreduced[i] = y[i] - np.polyval(pi,x[i])
    return yreduced

def _calc_reduced_uncertainty(self,yreduced,x,dx,y,dy,*z_dz):
    """Uncertainty of reduced y is still dy."""
    if dy is None:
        raise InferException()
    dyreduced = dy
    return dyreduced

def _collect_prototypes(*keys,defining_qn=()):
    ## collect from module prototypes list
    default_prototypes = {key:deepcopy(prototypes[key]) for key in keys}
    ## add infer functions for between '_qnhash', 'encoded_qn', and
    ## defining_qn
    if '_qnhash' in default_prototypes:
        default_prototypes['_qnhash']['infer'].append((defining_qn,_qn_hash),)
    if 'qn_encoded' in default_prototypes:
        default_prototypes['qn_encoded']['infer'].append(
            (defining_qn, lambda self,*qn:
                     [self.encode_qn({key:qni[j] for (key,qni) in zip(defining_qn,qn)}) for j in range(len(self))]))
    for key in defining_qn:
        default_prototypes[key]['infer'].append(
            ('qn', lambda self,qn,key=key: _get_key_from_qn(self,qn,key)))
    ## species hash, can't remember what problem this solves!
    if 'species' in default_prototypes and '_species_hash' not in default_prototypes:
        default_prototypes['_species_hash'] = deepcopy(prototypes['_species_hash'])
    ## add Ereduced
    if 'Ereduced_JJ' in keys:
        z = [key for key in defining_qn if key!='J']
        default_prototypes['Ereduced_JJ']['infer'].append((('JJ','E',*z),(_calc_reduced,_calc_reduced_uncertainty)))
    if 'Ereduced_vv' in keys:
        z = [key for key in defining_qn if key!='v']
        default_prototypes['Ereduced_vv']['infer'].append((('vv','E',*z),(_calc_reduced,_calc_reduced_uncertainty)))
    return default_prototypes

class Base(Dataset):
    """Common stuff for for lines and levels."""
    defining_qn = ()
    default_xkey = None
    default_zkeys = ()
    default_prototypes = {}

    def __init__(
            self,
            *args,
            encoded_qn=None,
            defining_qn=None,   # overwrite built in defining_qn
            **kwargs,
    ):
        kwargs.setdefault('permit_nonprototyped_data',False)
        ## decode encoded_qn
        if encoded_qn is not None:
            if isinstance(encoded_qn,str):
                kwargs = self.decode_qn(encoded_qn) | kwargs
            else:
                t = {}
                for i,encoded_qni in enumerate(encoded_qn):
                    qni = self.decode_qn(encoded_qni)
                    if i == 0:
                        for key in qni:
                            t[key] = [qni[key]]
                    else:
                        if length(qni) != length(t):
                            raise Exception
                        for key in qni:
                            t[key].append(qni[key])
        if defining_qn is not None:
            self.defining_qn = defining_qn
        Dataset.__init__(self,*args,**kwargs)

    def decode_qn(self,encoded_qn):
        """Decode string into quantum numbers"""
        raise Exception("not implemented")

    def encode_qn(self,qn):
        """Encode dictionary of quantum numbers into a string"""
        return repr(qn)

    def default_zlabel_format_function(self,*args,**kwargs):
        return self.encode_qn(*args,**kwargs)

    def assert_unique_qn(self,verbose=False):
        """Assert no two levels/lines are the same"""
        t,i,c = np.unique(self['_qnhash'],return_index=True,return_counts=True)
        if len(i) < len(self):
            j = array([ti for ti,tc in zip(i,c) if tc > 1])
            if verbose or self.verbose:
                print('\nNon-unique levels:\n')
                print()
            raise Exception(f"There are {len(j)} sets of quantum numbers that are repeated (set verbose=True to print).")

    def sort(self,*sort_keys,**dataset_sort_kwargs):
        """Overload sort to include automatic keys."""
        if len(sort_keys) == 0:
            sort_keys = [key for key in self.defining_qn if self.is_known(key)]
        Dataset.sort(self,*sort_keys,**dataset_sort_kwargs)

    def match(self,keys_vals=None,**kwargs):
        """Overload Dataset.match to handle 'encoded_qn' which is decoded
before matching."""
        if keys_vals is None:
            keys_vals = {}
        keys_vals = keys_vals | kwargs 
        if 'encoded_qn' in keys_vals:
            self.decode_qn(keys_vals['encoded_qn'])
            for key,val in self.decode_qn(keys_vals.pop('encoded_qn')).items():
                keys_vals.setdefault(key,val)
        return Dataset.match(self,keys_vals)

    def set(self,key,subkey,value,index=None,match=None,set_changed_only=False,**match_kwargs):
        """Overload Dataset.set to handle 'encoded_qn' which is decoded before
        setting individual qn.."""
        if key == 'encoded_qn':
            assert subkey == 'vector'
            for tkey,tvalue in self.decode_qn(value).items():
                Dataset.set(self,tkey,subkey,tvalue,index,match,set_changed_only,**match_kwargs)
        else:
            Dataset.set(self,key,subkey,value,index,match,set_changed_only,**match_kwargs)

    def append(self,*args,**kwargs):
        """Overload Dataset.append to handle 'encoded_qn' which is decoded
        before appending individual qn.."""
        if 'encoded_qn' in kwargs:
            qn = self.decode_qn(kwargs.pop('encoded_qn'))
            kwargs = qn | kwargs
        Dataset.append(self,*args,**kwargs)

    def find_common(x,y,keys=None,verbose=False):
        """Default keys to defining_qn."""
        if keys is None:
            keys = x.defining_qn
        return Dataset.find_common(self,x,y,keys=keys,verbose=verbose)
        
    def plot(self,*args,reduced_order=None,**kwargs):
        """Wrapper for Datset.plot"""
        if reduced_order is not None:
            self.reduced_order = reduced_order
        return Dataset.plot(self,*args,**kwargs)

    def normalise_species(self):
        """Normalise encoded species names."""
        for key in ('species','species_u','species_l'):
            if self.is_set(key):
                self[key] = database.normalise_species(self[key])

    def load(self,*args,normalise_species=False,**kwargs):
        """Overload Dataset load to optionally normalise loaded species names."""
        Dataset.load(self,*args,**kwargs)
        if normalise_species:
            self.normalise_species()


class Generic(Base):
    """A generic level."""
    defining_qn = ('species','label','ef','J')
    default_xkey = 'J'
    default_zkeys = ('species','label','ef')
    default_prototypes = _collect_prototypes(
        'species','label','ef','J',
        'JJ','Ereduced_JJ',
        '_species_hash',
        'chemical_species','isotopologue_ratio',
        'reference','_qnhash',
        'point_group',
        'mass','reduced_mass',
        'E','Ee','E0','Ereduced','Ereduced_common','Eref','Eres','Eexp',
        'Γ','Γexp','Γres',
        'N','S',
        'g','gnuclear','Inuclear',
        'Teq','Tex',
        'Zsource','Z','α','α296K',
        'Nchemical_species','Nspecies',
        'L',
        'At','Ae','Ad','ηd','ηe','Γe','Γd',         # destruction rates and branching
        defining_qn=defining_qn)
    Ereduced_common_polynomial = (1,0)

    def encode_qn(self,qn):
        """Encode qn into a string"""
        return quantum_numbers.encode_linear_level(qn)

    def decode_qn(self,encoded_qn):
        """Decode string into quantum numbers"""
        return quantum_numbers.decode_linear_level(encoded_qn)
    
class Atom(Generic):
    defining_qn = ('species','conf','J','S',)
    default_zkeys = ('species',)
    default_prototypes = _collect_prototypes(
        *Generic.default_prototypes,
        'conf','L','gu','lande_g','term',
        defining_qn=defining_qn)

    def load_from_nist(self,filename):
        """Load NIST tab-separated atomic levels data file."""
        ## load into dict
        data_string = tools.file_to_string(filename)
        data_string = data_string.replace('\t','|')
        data_string = data_string.replace('"','')
        data_string = [t for i,t in enumerate(data_string.split('\n')) if i==0 or len(t)<3 or t[:3]!='obs']
        data_string = '\n'.join(data_string)
        data = Dataset()
        data.load_from_string(data_string,delimiter='|')
        ## manipulate some data
        for key in ('Level (cm-1)',):
            if data.get_kind(key) == 'U':
                tre = re.compile(r'\[(.*)\]')
                for i,t in enumerate(data[key]):
                    if re.match(tre,t):
                        data[key][i] = t[1:-1]
        ## remove term symbols I do not know how to decode
        for regexp in (
                r'^nan$',
                r'^\*$',
                ):
            i = data.match_re(Term=regexp)
            if np.any(i):
                warnings.warn(f'Removing {sum(i)} terms I do not understand matching regexp {repr(regexp)}')
                data.index(~i)
        ## add to self
        for key0,key1 in (
                ('Configuration','conf'),
                ('Term',None),
                ('J','J'),
                ('Level (cm-1)','E'),
                ('Uncertainty (cm-1)',None), # could add
                ('Lande','lande_g'),
                ('Reference',None)
        ):
            if key1 is not None and key0 in data:
                self[key1] = data[key0]
        self['reference'] = 'NIST'
        ## decode NIST terms -- incomplete
        S = []
        warnings_issued = []
        for i,t in enumerate(data['Term']):
            if r:=re.match(r'^([0-9]+)([SPDFGH])(\*?)$',t): 
                ## e.g., 1S
                S.append((float(r.group(1))-1)/2)
            elif r:=re.match(r'^([0-9]+)\[([0-9/]+)\](\*?)$',t): 
                ## e.g., 2[3/2]*
                S.append((float(r.group(1))-1)/2)
            elif r:=re.match(r'^\(([0-9]+/[0-9]+),([0-9]+/[0-9]+)\)([*]?)$',t):
                ## e.g., (1/2,1/2)*. What does this mean? Strong
                ## LS-coupling so instead JJ levels are shown?, I will
                ## incorrectly set S to zero
                warning_text = f'Atomic term {t!r} not understood and setting S=0'
                if warning_text not in warnings_issued:
                    warnings.warn(warning_text)
                    warnings_issued.append(warning_text)
                S.append(0)     
            else:
                raise Exception(f"Could not decode NIST atomic term: {repr(t)}")
        self['S'] = S

class Linear(Generic):
    defining_qn = ('species','label','ef','J')
    default_zkeys = ('species','label','ef','Σ')
    default_prototypes = _collect_prototypes(
        *Generic.default_prototypes,
        'Λ','s','Σ','SR','Ω','Fi','LSsign',
        'i','gu','σv','sa','ef',
        defining_qn=defining_qn)

    def encode_qn(self,qn_dict):
        """Encode qn into a string"""
        return quantum_numbers.encode_linear_level(qn_dict)

    def decode_qn(self,encoded_qn):
        """Decode string into quantum numbers"""
        return quantum_numbers.decode_linear_level(encoded_qn)

    def load_pgopher_constants(self, filename,decode_name_function=None):
        """Load constants from a Pgopher .pgo xml file."""
        ## load xml as dictionary
        import xmltodict
        data = xmltodict.parse(tools.file_to_string(filename))
        ## extract different parts of the dictionatry
        mixture  = data['Mixture']
        species = mixture['Species']
        molecule = species['LinearMolecule']
        manifolds = molecule['LinearManifold']
        ## for each LinearManifold extract quantum numbers and
        ## molecular constants, add to self
        # self.set_prototype('pgopher_name','U',description='PGopher manifold name')
        for manifold in manifolds:
            manifold_name =  manifold['@Name']
            if isinstance(manifold['Linear'],dict):
                manifold['Linear'] = [manifold['Linear']]
            for linear in manifold['Linear']:
                ## decode name
                linear_name = linear.pop('@Name')
                # linear['pgopher_name'] = linear_name
                if decode_name_function is None:
                    linear['label'] = linear_name
                else:
                    linear.update(decode_name_function(linear_name))
                ## get molecular constants
                for p in linear['Parameter']:
                    linear[p['@Name']] = p['@Value']
                linear.pop('Parameter')
                ## translate key names, key1=None to delete,
                ## default!=None to add a default
                for key0,key1,cast,default in (
                        ('@Comment' , None  , None                                   , None)     , 
                        ('@Colour'  , None  , None                                   , None)     , 
                        ('@Lambda'  , 'Λ'   , str                                    , 'Sigma+') , 
                        ('@S'       , 'S'   , lambda S:float(S)/2                    , 0)        , 
                        ('@gerade'  , 'gu'  , lambda gu: (1 if gu == 'True' else -1) , None)     , 
                        ('A'        , 'Av'  , float                                   , None)     , 
                        ('AD'       , 'ADv' , float                                   , None)     , 
                        ('B'        , 'Bv'  , float                                   , None)     , 
                        ('D'        , 'Dv'  , float                                   , None)     , 
                        ('H'        , 'Hv'  , float                                   , None)     , 
                        ('LambdaSS' , 'λv'  , float                                   , None)     , 
                        ('Origin'   , 'Tv'  , float                                   , None)     , 
                        ('gamma'    , 'γv'  , float                                   , None)     , 
                        ('o'        , 'ov'  , float                                   , None)     , 
                        ('p'        , 'pv'  , float                                   , None)     , 
                        ('q'        , 'qv'  , float                                   , None)     , 
                        ('L'        , 'Lv'  , float                                   , None)     , 
                        ('M'        , 'Mv'  , float                                   , None)     , 
                        ('PP'       , 'Nv'  , float                                   , None)     , 
                ):
                    if key0 in linear:
                        if key1 is None:
                            linear.pop(key0)
                        elif cast is None:
                            linear[key1] = linear.pop(key0)
                        else:
                            linear[key1] = cast(linear.pop(key0))
                    elif default is not None:
                        linear[key1] = default
                ## decode gu
                if 'gu' in linear:
                    linear['gu'] = (1 if linear['gu'] == 'True' else -1)
                ## decode Λ
                if linear['Λ'] == 'Sigma+':
                    linear['Λ'],linear['s'] = 0,0
                elif linear['Λ'] == 'Sigma-':
                    linear['Λ'],linear['s'] = 0,1
                elif linear['Λ'] == 'Pi':
                    linear['Λ'],linear['s'] = 1,0
                elif linear['Λ'] == 'Delta':
                    linear['Λ'],linear['s'] = 2,0
                elif linear['Λ'] == 'Phi':
                    linear['Λ'],linear['s'] = 3,0
                elif linear['Λ'] == 'Gamma':
                    linear['Λ'],linear['s'] = 4,0
                else:
                    raise Exception(f'Lambda code not implemented: {repr(linear["Λ"])}')
                ## add to data
                self.append(linear)

class LinearTriatom(Linear):
    """A generic level."""
    defining_qn = ('species','label','ef','ν1','ν2','ν3','l2','J')
    defining_zkeys = ('species','label','ef','ν1','ν2','ν3','l2')
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'ν1','ν2','ν3','l2',
        defining_qn=defining_qn)
    ## correctly label vibrationnal modes
    default_prototypes['ν1']['description'] = "Vibrational quantum number for symmetric stretching"
    default_prototypes['ν2']['description'] = "Vibrational quantum number for bending"
    default_prototypes['l2']['description'] = "Vibrational angular momentum of the bending mode"
    default_prototypes['ν2']['description'] = "Vibrational quantum number for asymmetric stretching"

class Diatom(Linear):
    defining_qn = ('species','label','v','Σ','ef','J')
    default_zkeys = ('species','label','v','Σ','ef')
    default_prototypes = _collect_prototypes(
        *Linear.default_prototypes,
        'v','vv','Ereduced_vv',
        'Γv','τv','Atv','Adv','Aev',
        'ηdv','ηev',
        'Tvib','Trot','αvib','αrot','Zvib','Zrot',
        'Tv','Bv','Dv','Hv','Lv','Mv','Nv','Ov',
        'Av','ADv','AHv',
        'λv','λDv','λHv',
        'γv','γDv','γHv',
        'ov','oDv','oHv','oLv',
        'pv','qv',
        'pDv','qDv',
        'Tvreduced','Tvreduced_common',
        'Bv_μscaled',
        defining_qn=defining_qn)

    def load_from_duo(self,filename):
        """Load an output level list computed by DUO (yurchenko2016)."""
        data = file_to_dict(filename)
        if len(data) == 11:
            for icolumn,key in enumerate(('index','E','g','J','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
        elif len(data) == 12:
            for icolumn,key in enumerate(('index','E','g','J','E_residual','s','ef','label','v','Λ','Σ','Ω',)):
                data[key] = data.pop(f'column{icolumn}')
            data.pop('E_residual') # not currently used
        if len(data)==0:
            print(f'warning: no data found in {repr(filename)}')
            return
        ## get Λ/Σ/Ω sign conventions to match mine: Λ is +, Σ + or -,
        ## Ω is absolute
        i = data['Λ'] < 0
        data['Σ'][i] *= -1
        data['Λ'] = np.abs(data['Λ'])
        data['Ω'] = np.abs(data['Ω'])
        ## translate +/- to +1/-1
        i = data['s']=='-'
        data['s'] = np.full(len(data['s']),+1,dtype=int)
        data['s'][i] = -1
        ## translate e/f to +1/-1
        i = data['ef']=='f'
        data['ef'] = np.full(len(data['ef']),+1,dtype=int)
        data['ef'][i] = -1
        data.pop('index')
        self.extend(**data)

    def load_from_spectra(self,filename):
        """Old filetype. Incomplete"""
        data = file_to_dict(filename)
        ## keys to translate
        for key_old,key_new in (
                ('T','E'),
                ('Tref','E0'),
                ):
            if key_old in data:
                assert key_new not in data
                data[key_new] = data.pop(key_old)
        ## data to modify
        if 'ef' in data:
            i = data['ef']=='f'
            data['ef'] = np.full(len(data['ef']),+1,dtype=int)
            data['ef'][i] = -1
        ## data to ignore
        for key in (
                'level_transition_type',
                'partition_source',
                ):
            if key in data:
                data.pop(key)
        self.extend(**data)

