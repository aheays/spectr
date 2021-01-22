import re
import warnings
from copy import copy
from functools import lru_cache

import numpy as np
from bidict import bidict
import periodictable

from .dataset import Dataset
from . import tools
from .tools import cache
from . import database
from .exceptions import MissingDataException


#############
## species ##
#############

@tools.vectorise()
def decode_species(name,encoding):
    """Decode into standard name format from a foreign format."""
    if encoding == 'standard':
        return name
    if (encoding in _species_name_translation_dict
        and name in _species_name_translation_dict[encoding].inverse):
        ## try _species_name_translation_functions
        return _species_name_translation_dict[encoding].inverse[name]
    elif (encoding,'standard') in _species_name_translation_functions:
        ## try _species_name_translation_functions
        return _species_name_translation_functions[(encoding,'standard')](name)
    raise Exception(f"Could not decode {name=} from {encoding=}")

@tools.vectorise()
def encode_species(name,encoding):
    """Encode from standard name into a foreign format."""
    if encoding=='standard':
        return name         
    ## try _species_name_translation_functions
    if (encoding in _species_name_translation_dict
        and name in _species_name_translation_dict[encoding]):
        return _species_name_translation_dict[encoding][name]
    ## try _species_name_translation_functions
    if ('standard',encoding) in _species_name_translation_functions:
        return _species_name_translation_functions['standard',encoding](name)
    raise Exception(f"Could not encode {name=} into {encoding=}")

def translate_species(name,input_encoding,output_encoding):
    """Translate species name between different formats."""
    return encode_species(decode_species(name,input_encoding),output_encoding)
    
## dictionary for converting species name standard:foreign
_species_name_translation_dict = {}

## functions for converting a species name -- used after _species_name_translation_dict
_species_name_translation_functions = {}

## matplotlib
_species_name_translation_dict['matplotlib'] = bidict({
    '14N2':'${}^{14}$N$_2$',
    '12C18O':r'${}^{12}$C${}^{18}$O',
    '32S16O':r'${}^{32}$S${}^{16}$O',
    '33S16O':r'${}^{33}$S${}^{16}$O',
    '34S16O':r'${}^{34}$S${}^{16}$O',
    '36S16O':r'${}^{36}$S${}^{16}$O',
    '32S18O':r'${}^{32}$S${}^{18}$O',
    '33S18O':r'${}^{33}$S${}^{18}$O',
    '34S18O':r'${}^{34}$S${}^{18}$O',
    '36S18O':r'${}^{36}$S${}^{18}$O',
})

def _f(name):
    """Translate form my normal species names into something that
    looks nice in matplotlib."""
    name = re.sub(r'([0-9]+)',r'$_{\1}$',name) # subscript multiplicity 
    name = re.sub(r'([+-])',r'$^{\1}$',name) # superscript charge
    return(name)
_species_name_translation_functions[('standard','matplotlib')] = _f

## leiden
_species_name_translation_dict['leiden'] = bidict({
    'Ca':'ca', 'He':'he',
    'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn',
    'Na':'na', 'Ni':'ni', 'Rb':'rb', 'Ti':'ti',
    'Zn':'zn', 'Si':'si', 'Li':'li', 'Fe':'fe',
    'HCl':'hcl', 'Al':'al', 'AlH':'alh',
    'LiH':'lih', 'MgH':'mgh', 'NaCl':'nacl',
    'NaH':'nah', 'SiH':'sih', 'Co':'cob'
})

def _f(leiden_name):
    """Translate from Leidne data base to standard."""
    ## default to uper casing
    name = leiden_name.upper()
    name = name.replace('C-','c-')
    name = name.replace('L-','l-')
    ## look for two-letter element names
    name = name.replace('CL','Cl')
    name = name.replace('SI','Si')
    name = name.replace('CA','Ca')
    ## look for isotopologues
    name = name.replace('C*','13C')
    name = name.replace('O*','18O')
    name = name.replace('N*','15N')
    ## assume final p implies +
    if name[-1]=='P' and name!='P':
        name = name[:-1]+'+'
    return name
_species_name_translation_functions[('leiden','standard')] = _f

def _f(standard_name):
    """Translate form my normal species names into the Leiden database
    equivalent."""
    standard_name  = standard_name.replace('+','p')
    return standard_name.lower()
_species_name_translation_functions[('standard','leiden')] = _f

## meudon_pdr
_species_name_translation_dict['meudon_pdr'] = bidict({
    'Ca':'ca', 'Ca+':'ca+', 'He':'he', 'He+':'he+',
    'Cl':'cl', 'Cr':'cr', 'Mg':'mg', 'Mn':'mn', 'Na':'na', 'Ni':'ni',
    'Rb':'rb', 'Ti':'ti', 'Zn':'zn', 'Si':'si', 'Si+':'si+',
    'Li':'li', 'Fe':'fe', 'Fe+':'fe+', 'HCl':'hcl', 'HCl+':'hcl+',
    'Al':'al', 'AlH':'alh', 'h3+':'H3+', 'l-C3H2':'h2c3' ,
    'l-C3H':'c3h' , 'l-C4':'c4' , 'l-C4H':'c4h' , 'CH3CN':'c2h3n',
    'CH3CHO':'c2h4o', 'CH3OCH3':'c2h7o', 'C2H5OH':'c2h6o',
    'CH2CO':'c2h2o', 'HC3N':'c3hn', 'e-':'electr', # not sure
    ## non chemical processes
        'phosec':'phosec', 'phot':'phot', 'photon':'photon', 'grain':'grain',
    # '?':'c3h4',                 # one of either H2CCCH2 or H3CCCH
    # '?':'c3o',                  # not sure
    # '?':'ch2o2',                  # not sure
    })

def _f(name):
    """Standard to Meudon PDR with old isotope labellgin."""
    name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
    for t in (('[18O]','O*'),('[13C]','C*'),('[15N]','N*'),): # isotopes
        name = name.replace(*t)
    return(name.lower())
_species_name_translation_functions[('standard','meudon old isotope labelling')] = _f

def _f(name):
    """Standard to Meudon PDR."""
    name = re.sub(r'\([0-9][SPDF][0-9]?\)','',name) # remove atomic terms e.g., O(3P1) ⟶ O
    for t in (('[18O]','_18O'),('[13C]','_13C'),('[15N]','_15N'),): # isotopes
        name = name.replace(*t)
    name = re.sub(r'^_', r'' ,name)
    name = re.sub(r' _', r' ',name)
    name = re.sub(r'_ ', r' ',name)
    name = re.sub(r'_$', r'' ,name)
    name = re.sub(r'_\+',r'+',name)
    return(name.lower())
_species_name_translation_functions[('standard','meudon')] = _f

## STAND reaction network used in ARGO model
_species_name_translation_dict['STAND'] = bidict({
    'NH3':'H3N',
    'O2+_X2Πg':'O2+_P',
    'O2_a1Δg' :'O2_D',
    'O+_3P':'O2P',       # error in STAND O^+^(^3^P) should be O^+^(^2^P)?
    'O+_2D':'O2D',
    'O_1S':'O1S',
    'O_1D':'O1D',
    'C_1S':'C1S',
    'C_1D':'C1D',
    'NH3':'H3N',
    'OH':'HO',
})
def _f(name):
    return(name)
_species_name_translation_functions[('STAND','standard')] = _f

## kida
def _f(name):
    return(name)
_species_name_translation_functions[('kida','standard')] = _f

## latex
def _f(name):
    """Makes a nice latex version of species. THIS COULD BE EXTENDED"""
    try:
        return(database.get_species_property(name,'latex'))
    except:
        return(r'\ce{'+name.strip()+r'}')
_species_name_translation_functions[('standard','latex')] = _f


@cache
def get_species(name):
    return Species(name)


class Species:
    """Info about a species. Currently assumed to be immutable data only."""

    def __init__(self,name):
        # ## determine if name is for a species or an isotopologue
        # print('DEBUG:', name)
        # if '[' in name or ']' in name:
            # self.isotopologue = name
            # try:
                # self.species = database.get_species_property(self.isotopologue,'iso_indep')
            # except MissingDataException:
                # self.species = re.sub(r'\[[0-9]*([A-Z-az])\]',r'\1',name)
        # else:
            # self.species = name
            # self.isotopologue = database.get_species_property(self.species,'iso_main')
        self._name = name
        self._charge = None     # net charge
        self._nelectrons = None # number of electrnos

    def _get_charge(self):
        if self._charge is None:
            if r:=re.match('^(.*[^+-])([+]+|-+)$',self.species): # ddd+ / ddd++
                self._charge = r.group(2).count('+') - r.group(2).count('-')
            else:
                self._charge = 0
        return self._charge

    def _get_elements(self):
        if self._elements is not None:
            pass
        else:
            self._elements = []
            for part in re.split(r'([A-Z][a-z]?[0-9]*)',self.species):
                if r:= re.match('^([A-Z][a-z]?)([0-9]*)',part):
                    element = r.group(1)
                    multiplicity = (1 if r.group(2)=='' else int(r.group(2)))
                    for i in range(multiplicity):
                        self._elements.append(element)
            self._elements.sort()
        return self._elements


    def _get_nelectrons(self):
        if self._nelectrons is None:
            if self.species in ('e-','photon'):
                raise NotImplementedError()
            self._nelectrons = 0
            for element in self['elements']:
                self._nelectrons += getattr(periodictable,element).number # add electrons attributable to each nucleus
            self._nelectrons -= self['charge'] # account for ionisation
        return self._nelectrons

    def __str__(self):
        if self.isotopologue is None:
            return self.species
        else:
            return self.isotopologue

    ## for sorting a list of Species objects
    def __lt__(self,other):
        return self.name < other

    def __gt__(self,other):
        return self.name > other

    @lru_cache
    def __getitem__(self,key):
        """Access these properties by index rather than attributes in order
        for simple caching.  -- move other get_ methods to here someday"""
        if key == 'name':
            return self._name
        elif key == 'charge':
            return self._get_charge()
        elif key == 'elements':
            elements = tuple(t[0] for t in self['isotopes'])
            return elements
        elif key == 'isotopes':
            ## match elemental or isotopolgue components with multiplicity,
            ## e.g., H2 → H2, OH → O + H, [12C][16O]2 → [12C] + [16O]2
            isotopes = []
            for part in re.split(r'([A-Z][a-z]?[0-9]*|\[[0-9]+[A-Z][a-z]?\][0-9]*)',self['name']):
                if part=='':
                    continue
                if r:= re.match(r'^([A-Z][a-z]?)([0-9]*)',part):
                    ## an element
                    element = r.group(1)
                    multiplicity = (1 if r.group(2)=='' else int(r.group(2)))
                    mass_number = database.get_isotopes(element)[0][0]
                elif r:= re.match(r'^\[([0-9]+)([A-Z][a-z]?)\]([0-9]*)',part):
                    ## an isotope
                    mass_number,element,multiplicity = r.groups()
                else:
                    raise Exception(f'Could not determine isotope from part {repr(part)} of name {repr(self.name)}')
                for i in range(1 if multiplicity=='' else int(multiplicity)):
                    isotopes.append((element,int(mass_number)))
            isotopes = tuple(sorted(isotopes))
            return isotopes
        elif key == 'nelectrons':
            return self._get_nelectrons()
        elif key == 'mass':
            return sum([database.get_atomic_mass(element,mass_number)
                        for element,mass_number in self['isotopes']])
        elif key == 'reduced_mass':
            if len(self['isotopes']) != 2:
                raise Exception("Can only compute reduced mass for diatomic species.")
            m1 = database.get_atomic_mass(*self['isotopes'][0])
            m2 = database.get_atomic_mass(*self['isotopes'][1])
            return m1*m2/(m1+m2)
        elif key == 'chemical_name':
            return ''.join(self['elements'])
        elif key == 'isotopologue':
            return ''.join([f'[{t[1]}{t[0]}]' for t in self['isotopes']])
        elif key == 'point_group':
            if len(self.isotopes) != 2:
                raise Exception("Can only compute reduced mass for diatomic species.")
            ## Homonumclear or heteronuclear diatomic
            if self.isotopes[0] == self.isotopes[1]:
                return 'D∞h'
            else:
                return 'C∞v'
        else:
                raise Exception(f"Unknown speices property: {key}")
    
    name = property(lambda self: self['name'])
    elements = property(lambda self: self['elements'])
    isotopes = property(lambda self: self['isotopes'])
    chemical_name = property(lambda self: self['chemical_name'])
    isotopologue = property(lambda self: self['isotopologue'])
    species = property(lambda self: self.name)
    mass = property(lambda self: self['mass'])
    reduced_mass = property(lambda self: self['reduced_mass'])
    point_group = property(lambda self: self['point_group'])
    
    
########################
## Chemical reactions ##
########################



def decode_reaction(reaction,encoding='standard'):
    """Decode a reaction into a list of reactants and products, and other
    information. Encoding is for species names. Encoding of reaction
    string formatting not implemented."""
    ## split parts
    reactants_string,products_string = reaction.split('→')
    reactants,products = [],[]
    for r_or_p_string,r_or_p_list in ((reactants_string,reactants),
                                      (products_string,products)):
        ## decode reactants or products string into a list
        for species in r_or_p_string.split(' + '):
            species = species.strip()
            if r:=re.match(r'^([0-9]+)(.*)',species):
                multiplicity = int(r.group(1))
                species = species.group(1)
            else:
                multiplicity = 1
            species = decode_species(species,encoding)
            for i in range(multiplicity):
                r_or_p_list.append(species)
        r_or_p_list.sort()
    return reactants,products

def encode_reaction(reactants,products,encoding='standard'):
    """Only just started"""
    if encoding!='standard':
        raise ImplementationError()
    return ' + '.join(reactants)+' → '+' + '.join(products)

## formulae for computing rate coefficients from reaction constants c
## and state variables p
_reaction_coefficient_formulae = {
    'constant'               :lambda c,p: c['k'],
    'arrhenius'              :lambda c,p: c['A']*(p['Ttr']/300.)**c['B'],
    'KIDA modified arrhenius':lambda c,p: c['A']*(p['Ttr']/300.)**c['B']*np.exp(-c['C']*p['Ttr']),
    'NIST'                   :lambda c,p: c['A']*(p['Ttr']/298.)**c['n']*np.exp(-c['Ea']/8.314472e-3/p['Ttr']),
    'NIST_3rd_body_hack'     :lambda c,p: 1e19*c['A']*(p['Ttr']/298.)**c['n']*np.exp(-c['Ea']/8.314472e-3/p['Ttr']),
    'photoreaction'          :lambda c,p: scipy.integrate.trapz(c['σ'](p['Ttr'])*p['I'],c['λ']),
    'kooij'                  :lambda c,p: c['α']*(p['Ttr']/300.)**c['β']*np.exp(-c['γ']/p['Ttr']), # α[cm-3], T[K], β[], γ[K]
    'impact_test_2020-12-07' :lambda c,p: p['Ttr']*0 ,
    # 'kooij'                  :lambda c,p: np.full(p['T'].shape,c['α'])
} 

def _f(c,p):
    """STAND 3-body reaction scheme. Eqs. 9,10,11 in rimmer2016."""
    k0 = c['α0']*(p['Ttr']/300)**c['β0']*np.exp(-c['γ0']/p['Ttr']) 
    kinf = c['αinf']*(p['Ttr']/300.)**c['βinf']*np.exp(-c['γinf']/p['Ttr'])
    pr = k0*p['nt']/kinf             # p['nt'] = total density = M 3-body density
    k2 = (kinf*pr)/(1+pr)
    return k2
_reaction_coefficient_formulae['STAND 3-body'] = _f

class Reaction:
    """A class for manipulating a chemical reaction."""

    def __init__(
            self,
            name=None,
            reactants=None,
            products=None,
            formula='constant', # type of reaction, defined in get_rate_coefficient
            coefficients=None,     # used for computing rate coefficient according to formula
            encoding='standard', # of reaction name or species in products/reactants
    ):

        ## get reactants and products from name or provided lists
        if name is None and reactants is not None and products is not None:
            self.reactants = [decode_species(t,encoding) for t in reactants]
            self.products = [decode_species(t,encoding) for t in products]
        elif name is not None and reactants is None and products is None:
            self.reactants,self.products = decode_reaction(name,encoding)
        else:
            raise Exception('Must be name is not None, or reactants/products are not None, but not both.')
        self.reactants,self.products = tuple(self.reactants),tuple(self.products) # make immutable
        self._hash = hash((self.reactants,self.products))
        ## tidy name
        self.name = encode_reaction(self.reactants,self.products)
        self.formula = formula  # name of formula
        self._formula = _reaction_coefficient_formulae[formula] # function
        self.coefficients = ({} if coefficients is None else coefficients)
        self.rate_coefficient = None     
        self.rate = None     


    def __getitem__(self,key):
        return(self.coefficients[key])

    def __setitem__(self,key,val):
        self.coefficients[key] = val

    def get_rate_coefficient(self,state=None):
        """Calculate rate coefficient from rate constants and state
        variables."""
        self.rate_coefficient = self._formula(self.coefficients, ({} if state is None else state))
        return self.rate_coefficient

    def format_aligned_reaction_string(
            number_of_reactants=3, # how many columns of reactants to assume
            number_of_products=3, # products
            species_name_length=12, # character length of names
            ):
        ## get reactants and products padded with blanks if required
        reactants = copy(self.reactants)+['' for i in range(number_of_reactants-len(reactants))]
        products = copy(self.products)+['' for i in range(number_of_products-len(products))]
        return((' + '.join([format(t,f'<{species_name_length}s') for t in reactants])
                +' ⟶ '+' + '.join([format(t,f'<{species_name_length}s') for t in reactants])))

    def __repr__(self):
        return ', '.join([f'name=" {self.name:60}"']
                         +[format(f'formula="{self.formula}"',"20")]
                         +[format(f'{key}={repr(val)}','20') for key,val in self.coefficients.items()])


    def __str__(self):
        return self.name

class ReactionNetwork:

    def __init__(self):
        self.density = Dataset()       # name:density
        self.state = Dataset()         # name:value
        self.reactions = []     # [Reactions]
        self.verbose = False

    def __getitem__(self,key):
        if key in self.state:
            return self.state[key]
        elif key in self.density:
            return self.density[key]
        else:
            raise KeyError

    def calc_rate_coefficients(self):
        self.rate_coefficients = [reaction.get_rate_coefficient(self.state)
                                  for reaction in self.reactions]

    def calc_rates(self):
        # self.calc_rate_coefficients()
        self.rates = []
        for r in self.reactions:
            k = copy(r.rate_coefficient)
            for s in r.reactants:
                if s not in ['γ','e-']:
                    k *= self.density[s]
            r.rate = k

    def plot_species(self, *species, ykey=None, ax=None,):
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        if ykey is not None:
            for s in species:
                ax.plot(self[s],self[ykey],label=s)
                ax.set_ylabel(ykey)
                ax.set_xscale('log')
                ax.set_xlabel('Number density (cm$^{-3}$)')
        else:
            for s in species:
                ax.plot(self[s],label=s)
                ax.set_yscale('log')
                ax.set_ylabel('Number density (cm$^{-3}$)')
        plotting.legend()
        return ax

    def append(self,*args,**kwargs):
        """Append a Reaction, or arguments to define one."""
        if len(args)==1 and len(kwargs)==0 and  isinstance(args[0],Reaction):
            r = args[0]
        else:
            r = Reaction(*args,**kwargs)
        self.reactions.append(r)

    def remove_unnecessary_reactions(self):
        """Remove all reactions containing species that have no
        density."""
        for r in copy(self.reactions):
            if any([s not in ('γ','e-') and s not in self.density
                     for s in list(r.reactants)+list(r.products)]):
                if self.verbose:
                    print(f'Removing reaction containing species not in model: {str(r)}')
                self.reactions.remove(r)

    def __iter__(self):
        for t in self.reactions: yield(t)

    def __len__(self):
        return(len(self.reactions))

    def get_reactions(
            self,
            name=None,            # exact match of name
            reactants=None, # must be exact reactant list 
            products=None,  # must be exact product list  
            with_reactants=None,       # must be in reactant list
            with_products=None,        # must be in product list  
            without_reactants=None,   # must not be in reactant list 
            without_products=None,    # must not be in product list  
            coefficients=None,    # dictionary of matching coefficients
            formula=None,         # formula type
    ):
        """Return a list of reactions with this reactant."""
        retval = []
        if reactants is not None:
            reactants = sorted(reactants)
        if products is not None:
            products = sorted(products)
        for reaction in self.reactions:
            if name is not None and reaction.name != name:
                continue    
            if (with_reactants is not None
                and any([t not in reaction.reactants
                         for t in tools.ensure_iterable(with_reactants)])):
                continue
            if (with_products is not None
                and any([t not in reaction.products
                         for t in tools.ensure_iterable(with_products)])):
                continue
            if (without_reactants is not None
                and any([t in reaction.reactants
                     for t in tools.ensure_iterable(without_reactants)])):
                continue
            if (without_products is not None
                and any([t in reaction.products
                     for t in tools.ensure_iterable(without_products)])):
                continue
            if reactants is not None and sorted(reaction.reactants) != reactants:
                continue
            if products is not None and sorted(reaction.products) != products:
                continue
            if (coefficients is not None
                and any([(key not in reaction.coefficients
                         or reaction.coefficients[key] != coefficients[key])
                        for key in coefficients])):
                continue
            if formula is not None and reaction.formula != formula:
                continue
            retval.append(reaction)
        return retval

    def get_reaction(self, **kwargs_get_reactions,):
        """Return a uniquely matching reaction."""
        retval = self.get_reactions(**kwargs_get_reactions)
        if len(retval) == 0:
            raise Exception('No matching reaction found')
        elif len(retval) == 1:
            return retval[0]
        else:
            raise Exception('Multiple matching reaction found')

    def __str__(self):
        retval = []
        for t in self.reactions:
            retval.append(str(t))
        return('\n'.join(retval))

    def save(self,filename):
        """Save encoded reactions and coefficients to a file."""
        tools.string_to_file(filename,str(self))

    def load(self,filename):
        """Load encoded reactions and coefficients from a file."""
        with open(tools.expand_path(filename),'r') as fid:
            for line in fid:
                self.append(**eval(f'dict({line[:-1]})'),encoding='standard')

    def check_reactions(self):
        """Sanity check on reaction list."""
        ## warn on repeated reactions
        hasharray = np.array([r._hash for r in self.reactions])
        if len(np.unique(hasharray)) != len(hasharray):
            for h in hasharray:
                i = tools.find(hasharray==h)
                if len(i)>1:
                    warnings.warn(f'Reaction appears {len(i)} times'+'\n    '+'\n    '.join([
                    repr(self.reactions[j]) for j in i]))

    def load_stand(self,filename):
        """Load encoded reactions and coefficients from a file."""
        def get_line():
            line = fid.readline()
            if line.strip() == '':
                ## blank line
                return None
            ## decode
            retval = {'reactants':[],'products':[]}
            if (t:=line[0:8].strip()) != '':
                retval['reactants'].append(decode_species(t,'STAND'))
            if (t:=line[8:16].strip()) != '':
                retval['reactants'].append(decode_species(t,'STAND'))
            if (t:=line[16:24].strip()) != '':
                retval['reactants'].append(decode_species(t,'STAND'))
            if (t:=line[24:32].strip()) != '':
                retval['products'].append(decode_species(t,'STAND'))
            if (t:=line[32:40].strip()) != '':
                retval['products'].append(decode_species(t,'STAND'))
            if (t:=line[40:48].strip()) != '':
                retval['products'].append(decode_species(t,'STAND'))
            if (t:=line[48:54].strip()) != '':
                retval['products'].append(decode_species(t,'STAND'))
            retval['α'] = float(line[64:73])
            retval['β'] = float(line[73:82])
            retval['γ'] = float(line[82:91])
            retval['type'] = int(line[91:93])
            retval['reaction_number'] = int(line[108:112])
            return retval
        ## 
        reaction_types = {      # STAND2020 2020-10-23
            0 :'B = Neutral Termolecular and Thermal Decomposition Reactions',
            1 :'Z = Cosmic Ray Reactions (All of these are set equal zero for the Venus model ',
            2 :'D = Ion-Neutral Bimolecular Reactions',
            3 :'L = Reverse Ion-Neutral Bimolecular Reactions',
            5 :'T = Three-Body Recombination Reactions',
            6 :'I = Thermal Ionization Reactions: ',
            7 :'A = Neutral Bimolecular Reactions',
            8 :'RA = Radiative Association Reactions',
            10:'U = Dissociative Recombination Reactions',
            11:'Q = Ion-Neutral Termolecular and Thermal Decomposition Reaction',
            13:'V = Photochemical Reactions',
            14:'F = Reverse Neutral Bimolecular Reactions',
            15:'G = Reverse Neutral Termolecular and Thermal Decomposition Reactions',
            16:'R = Reverse Ion-Neutral Termolecular and Thermal Decomposition Reactions',
            17:'EV = Condensation/Evaporation Reactions',
            18:'DP = Unknown, not in Reaction-Guide.txt',
            44:'ID = Alan Heays test',
            66:'BA = Unknown, not in Reaction-Guide.txt',
            67:'GO = Unknown, not in Reaction-Guide.txt',
            88:'TR = Unknown, not in Reaction-Guide.txt',
            95:'BI = Neutral Termolecular and Thermal Decomposition Reactions (added to model Isoprene',
            96:'AS (or AI?) = Neutral Bimolecular Reactions (added to model Isoprene)',
            97:'BS = Neutral Termolecular and Thermal Decomposition Reactions (added to model Venus)',
            98:'AS = Neutral Bimolecular Reactions (added to model Venus)',
            99:'GL = Unknown, not in Reaction-Guide.txt',
        }
        already_warned_for_types = []
        ## loop through network
        with open(tools.expand_path(filename),'r') as fid:
            while True:
                line = get_line()
                if line is None:
                    ## end of file
                    break
                ## three body reactions
                elif line['type'] in (0,5,11):
                    line0 = line    # low-density limit
                    lineinf = get_line()    # high-density limit
                    self.append(Reaction(
                        formula='STAND 3-body',
                        reactants=line['reactants'],products=line['products'],
                        coefficients=dict(
                            α0=line0['α'],β0=line0['β'],γ0=line0['γ'],
                            αinf=lineinf['α'],βinf=lineinf['β'],γinf=lineinf['γ'],
                            reaction_number=line['reaction_number'],type=line['type'],)))
                ## bimolecular reactions and thermal ionisation
                elif line['type'] in (2,7,10):
                    self.append(Reaction(
                        formula='kooij',
                        reactants=line['reactants'],products=line['products'],
                        coefficients=dict(α=line['α'],β=line['β'],γ=line['γ'],
                                          reaction_number=line['reaction_number'],type=line['type'],)))
                ## bimolecular reactions producing photon
                elif line['type'] in (8,):
                    self.append(Reaction(
                        formula='kooij',
                        reactants=line['reactants'],products=line['products']+['γ'],
                        coefficients=dict(
                            α=line['α'],β=line['β'],γ=line['γ'],
                            reaction_number=line['reaction_number'],type=line['type'],)))
                ## unknown reaction type -- termolecular -- consume two lines
                elif line['type'] in ( 15, 16, 95, 97,):
                    if line['type'] not in already_warned_for_types:
                        warnings.warn(f'reaction type not implemented, added with zero coefficient: {line["type"]}: {reaction_types[line["type"]]}')
                        already_warned_for_types.append(line['type'])
                    line0 = line    # low-density limit
                    lineinf = get_line()    # high-density limit
                    self.append(Reaction(
                        formula='constant',
                        reactants=line['reactants'],products=line['products'],
                        coefficients={
                            'k':0,
                            'type':line0['type'],
                            'reaction_number':line0['reaction_number'],
                        }))
                ## unknown non-termolecular reaction type -- consume one line
                elif line['type'] in ( 1, 3, 6, 14, 17, 18, 66, 67, 88, 96, 98, 99):
                    if line['type'] not in already_warned_for_types:
                        warnings.warn(f'reaction type not implemented, added with zero coefficient: {line["type"]}: {reaction_types[line["type"]]}')
                        already_warned_for_types.append(line['type'])
                    self.append(Reaction(
                        formula='constant',
                        reactants=line['reactants'],products=line['products'],
                        coefficients={
                            'k':0,
                            'type':line['type'],
                            'reaction_number':line['reaction_number'],
                        }))
                ## photo reactions -- add γ reactant
                elif line['type'] in (  13,):
                    if line['type'] not in already_warned_for_types:
                        warnings.warn(f'reaction type not implemented, added with zero coefficient: {line["type"]}: {reaction_types[line["type"]]}')
                        already_warned_for_types.append(line['type'])
                    self.append(Reaction(
                        formula='constant',
                        reactants=line['reactants']+['γ'],products=line['products'],
                        coefficients={
                            'k':0,
                            'type':line['type'],
                            'reaction_number':line['reaction_number'],
                        }))
                ## skip 
                elif line['type'] in (44,):
                    self.append(Reaction(
                        formula='impact_test_2020-12-07',
                        reactants=line['reactants'],products=line['products'],
                        coefficients={
                            'surface_flux':line['α'],
                            'total_column':line['β'],
                            'reaction_number':line['reaction_number'],
                            'reaction_number':line['reaction_number'],'type':line['type'],
                        }))
                else:
                    raise Exception( 'unspecified line type',line['type'])
            ## verify
            self.check_reactions()


