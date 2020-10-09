from . import *

## non-standard library
from bidict import bidict


################################
## species name normalisation ##
################################

@tools.vectorise_function
def decode_species(name,encoding):
    """Decode into standard name format from a foreign format."""
    if encoding == 'standard':
        return name
    ## try _species_name_translation_functions
    if (encoding in _species_name_translation_dict
        and name in n_species_name_translation_dict[encoding].inverse):
        return _species_name_translation_dict[encoding].inverse[name]
    ## try _species_name_translation_functions
    if (encoding,'standard') in _species_name_translation_functions:
        return _species_name_translation_functions[(encoding,'standard')](name)
    raise Exception(f"Could not decode {name=} from {encoding=}")

@tools.vectorise_function
def encode_species(name,encoding):
    """Encode from standard name into a foreign format."""
    ## try _species_name_translation_functions
    if (encoding in _species_name_translation_dict
        and name in n_species_name_translation_dict[encoding]):
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

## argo
_species_name_translation_dict['argo'] = bidict({
    'NH3':'H3N',
})

def _f(name):
    return(name)
_species_name_translation_functions[('argo','standard')] = _f

## hitran numeric codes - https://hitran.org/docs/iso-meta/
_species_name_translation_dict['hitran_codes'] = bidict({
    'H2O':1,'CO2': 2,'O3': 3,'N2O': 4,
    'CO':5,'CH4': 6,'O2': 7,'NO': 8,'SO2': 9,'NO2': 10,'NH3': 11,
    'HNO3':12,'OH': 13,'HF': 14,'HCl': 15,'HBr': 16,'HI': 17,'ClO':18,
    'OCS':19,'H2CO': 20,'HOCl': 21,'N2': 22,'HCN': 23,
    'CH3Cl':24,'H2O2': 25,'C2H2': 26,'C2H6': 27,'PH3': 28,'COF2': 29,
    'SF6':30,'H2S': 31,'HCOOH': 32,'HO2': 33,'O': 34,'ClONO2': 35,
    'HOBr':37,'C2H4': 38,'CH3OH': 39,'CH3Br': 40,'CH3CN': 41,
    'CF4':42,'C4H2': 43,'HC3N': 44,'H2': 45,'CS': 46,'SO3': 47,
    'C2N2':48,'COCl2': 49,'[12C][16O]2': (2,1),'[13C][16O]2': (2,2),
    # '[16O][12C][18O]':(2,3),'[16O][12C][17O]': (2,4),'[16O][13C][18O]': (2,5),'[16O][13C][17O]': (2,6),
    # '[12C][18O]2':(2,7),'[17O][12C][18O]': (2,8),'[12C][17O]2': (2,9),
    # '[13C][18O]2':(2,10),'[18O][13C][17O]': (2,11),'[13C][17O]2': (2,12),
    # '12C16O':(5,1),'13C16O':(5,2),'12C18O':(5,3),'12C17O':(5,4),'13C18O':(5,5),'13C17O':(5,6),
    # '[12C][16O]':(5,1),
    # '[13C][16O]':(5,2),'[12C][18O]':(5,3),
    # '[12C][17O]':(5,4),'[13C][18O]':(5,5),'[13C][17O]':(5,6),
    # '14N16O':(8,1),'15N16O': (8,2),'14N18O': (8,3), 
    # '[14N][16O]':(8,1),'[15N][16O]': (8,2),'[14N][18O]': (8,3),
    # '[16O][12C][32S]':(19,1),'[16O][12C][34S]': (19,2),'[16O][13C][32S]': (19,3),'[16O][12C][33S]': (19,4),'[18O][12C][32S]': (19,5), # OCS
    # '[12C][1H]3[35Cl]':(24,1),'[12C][1H]3[37Cl]': (24,2), # CH3Cl
    # '[12C][1H]4':(6,1),'[13C][1H]4': (6,2),'[12C][1H]3[2H]': (6,3),'[13C][1H]3[2H]': (6,4),
    # '[1H][35Cl]':(15,1),'[1H][37Cl]': (15,2),'[2H][35Cl]': (15,3),'[2H][37Cl]': (15,4),
})

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



##################################
## Complex object for a species ##
##################################

_get_species_cache = {}
def get_species(name):
    """Get species from name, assume immutable and potentially
    cached."""
    if name not in _get_species_cache:
        _get_species_cache[name] = Species(name=name)
    return(_get_species_cache[name])

class Species:
    """Info about a species. Currently assumed to be immutable data only."""

    def __init__(self,name):
        # ## look for isomer, e..,g c-C3 vs l-C3
        # if r:=re.match('^(.+)-(.*)$',name):
            # self.isomer,name = r.groups()
        # else:
            # self.isomer = None
        # ## decode name
        # ## translate diatomic special cases -- remove one day, e.g., 32S16O to [32]S[16]O
        # name = re.sub(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)$',r'[\1\2][\3\4]',name)
        # name = re.sub(r'^([0-9]+)([A-Z][a-z]?)2$',r'[\1\2]2',name)
        self.name = name # must be unique in standard form -- use a hash instead?
        self._reduced_elements = None            # keys are names of elements, values are multiplicity in ONE molecule
        self._elements = None            # list of elements, possibly repeated
        self._element_species = None            # list of elements as Species objects
        self._charge = None
        self._nelectrons = None
        self._mass = None            # of one molecule
        self._reduced_mass = None   # reduced mass (amu), actually a property
        self._nonisotopic_name = None # with mass symbols removed, e.g., '[32S][16O]' to 'SO'
        self.density = None     # cm-3

    def _get_nonisotopic_name(self):
        """NOT WELL TESTED, ADD SPECIAL CASE FOR DEUTERIUM"""
        if self._nonisotopic_name is None:
            if r:=re.match(r'^[0-9]+([A-Z].*)',self.name):
                ## simple atom ,e.g., 13C
                self._nonisotopic_name = r.group(1)
            else:
                self._nonisotopic_name = r.sub(r'\[[0-9]*([A-Za-z]+)\]',r'\1',self.name)
        return self._nonisotopic_name

    nonisotopic_name = property(_get_nonisotopic_name)

    def _get_reduced_mass(self): 
        if self._reduced_mass is None:
            self._reduced_mass = database.get_species_property(self.name,'reduced_mass')
        return(self._reduced_mass)

    def _set_reduced_mass(self,reduced_mass):
        self._reduced_mass = reduced_mass

    reduced_mass = property(_get_reduced_mass,_set_reduced_mass)

    def _get_mass(self):
        if self._mass is None:
            self._mass = database.get_species_property(self.name,'mass')
        return(self._mass)

    mass = property(_get_mass)

    def _get_charge(self):
        if self._charge is not None:
            pass
        elif self.name=='e-':
            self._charge = -1
        elif self.name=='photon':
            self.charge = 0
        elif r:=re.match('^(.*)[^]([-+][0-9]+)$',self.name): # ddd^+3 / ddd^-3 / ddd^3
            self._charge = int(r.group(2))
        elif r:=re.match('^(.*)[^]([0-9]+)([+-])$',self.name): # ddd^3+ / ddd^3-
            self._charge = int(r.group(3)+r.group(2))
        elif r:=re.match('^(.*[^+-])([+]+|-+)$',self.name): # ddd+ / ddd++
            self._charge = r.group(2).count('+') - r.group(2).count('-')
        else:
            self._charge = 0
        return self._charge

    charge = property(_get_charge)

    def _get_elements(self):
        if self._elements is not None:
            pass
        else:
            self._elements = []
            self._reduced_elements = {}
            for part in re.split(r'(\[[0-9]*[A-Z][a-z]?\][0-9]*|[A-Z][a-z]?[0-9]*)',self.name):
                if len(part)==0: continue
                r = re.match('^(.+?)([0-9]*)$',part)
                if r.group(2)=='':
                    multiplicity = 1
                else:
                    multiplicity = int(r.group(2))
                element = r.group(1)
                element = element.replace(']','').replace('[','')
                for i in range(multiplicity):
                    self._elements.append(element)
                if element in self.reduced_elements:
                    self.reduced_elements[element] += multiplicity
                else:
                    self.reduced_elements[element] = multiplicity
        return self._elements

    elements = property(_get_elements)

    def _get_element_species(self):
        if self._element_species is None:
            self._element_species = [Species(element) for element in self.elements]
        return(self._element_species)

    element_species = property(_get_element_species)

    def _get_reduced_elements(self):
        if self._reduced_elements is None:
            self._get_elements()
        return self._reduced_elements

    reduced_elements = property(_get_reduced_elements)

    def _get_nelectrons(self):
        if self._nelectrons is not None:
            pass
        elif self.name == 'e-':
            self.charge = -1
            self._nelectrons = 1
        elif self.name == 'photon':
            self.charge = 0
            self.nelectrons = 0
        else:
            self._nelectrons = 0
            for element,multiplicity in self.reduced_elements.items():
                element = re.sub(r'^\[?[0-9]*([A-Za-z]+)\]?',r'\1',element)
                self._nelectrons += multiplicity*getattr(periodictable,element).number # add electrons attributable to each nucleus
            self._nelectrons -= self.charge # account for ionisation
        return self._nelectrons

    nelectrons = property(_get_nelectrons)

    def encode_elements(self,elements,charge=None,isomer=None):
        ## convert list of elements into elemets with degeneracies
        element_degeneracy = []
        for element in elements:
            if len(element_degeneracy)==0 or element_degeneracy[-1][0]!=element: # first or new
                element_degeneracy.append([element,1])
            elif element_degeneracy[-1][0]==element: # continuation
                element_degeneracy[-1][1] += 1
        ## concatenate
        name = ''.join(
            [(element if element[0] not in '0123456789' else '['+element+']') # bracket isotopes
             +('' if degeneracy==1 else str(degeneracy)) # multiplicity
             for element,degeneracy in element_degeneracy])
        if charge is not None:
            if charge>0:
                for i in range(charge): name += '+'
            if charge<0:
                for i in range(-charge): name += '-'
        if isomer is not None and isomer!='':
            name = isomer+'-'+name
        self.decode_name(name)

    def get_isotopologues(self,element_from,element_to):
        """Find disctinct single-substitutions of one element."""
        isotopologues = []      # list of Species, one for each distinct isotopologue
        for i,element in enumerate(self.elements):
            if i<(len(self.elements)-1) and element==self.elements[i+1]: continue # skip identical subsitutiotns, i.,e., keep rightmost, CO[18O] and not C[18O]O for CO2 substitution
            if element==element_from:
                t = copy(self.elements)
                t[i] = element_to
                isotopologues.append(Species(elements=t,charge=self.charge,isomer=self.isomer))
        return(isotopologues)         

    def __str__(self): return(self.name)

    ## for sorting a list of Species objects
    def __lt__(self,other): return(self.name<other.name)
    def __gt__(self,other): return(self.name>other.name)


class Mixture:
    """A mixture of species.  PERHAPS USABLE BY Reaction."""

    def __init__(self,name='mixture'):
        self.name = name
        self.species = {}       # dictionary list {Species:amount}
        self.other_data = {}

    def __getitem__(self,key):
        if key in self.other_data:
            return(self.other_data[key])
        elif key in self.species:
            return(self.species[key])
        elif (t:=get_species(key)) in self.species:
            return(self.species[t])

    def __setitem__(self,key,val):
        if key in self.species:
            self.species[key] = val
        elif key in self.other_data:
            self.other_data[key] = val
        else:
            self.add_species(key,val)

    def add_data(self,key,val):
        """Add some kind of data."""
        self.other_data[key] = val

    def add_species(self,name_or_Species,amount):
        """Add a species to the mixture."""
        if isinstance(name_or_Species,str):
            name_or_Species = get_species(name_or_Species)
        name,species = name_or_Species.name,name_or_Species
        ## if already in mixture add amount, else add new element to
        ## self.species
        for tspecies in self.species:
            if tspecies.name==name:
                break
        else:
            self.species[species] = amount
        return(species)

    def get_elements(self):
        elements = {}
        for species,amount in self.species.items():
            for element,multiplicity in species.reduced_elements.items():
                if element in elements:
                    elements[element] += multiplicity*amount
                else:
                    elements[element] = multiplicity*amount
        return(elements)

    def __str__(self):
        return('\n'.join(
            [f'{str(species):>10s} = {amount}'
            for species,amount in self.species.items()]))

    def get_atom_number(self):
        elements = self.get_elements()
        total_atom_number = sum(elements.values())
        return(total_atom_number)


########################
## Chemical reactions ##
########################

def decode_reaction(reaction,encoding='standard'):
    """Decode a reaction into a list of reactants and products, and other
    information. In a dictionary."""
    retval = {}
    retval['reactants'],retval['products'],retval['kind'] = _split_reaction_functions[encoding](reaction)
    retval['reactants'] = [decode_species(species,encoding) for species in retval['reactants']]
    retval['products'] = [decode_species(species,encoding) for species in retval['products']]
    return retval

_split_reaction_functions = {}

def _f(reaction):
    if '→' in reaction:
        kind = 'forward'
        reactants,products = reaction.split('→')
    elif '←' in reaction:
        kind = 'backward'
        reactants,products = reaction.split('←')
    elif '⇌' in reaction:
        kind = 'equilibrium'
        reactants,products = reaction.split('⇌')
    else:
        raise Exception(f'Cannot split {reaction=} with {encoding=}')
    reactants = [t.strip() for t in reactants.split(' + ')]
    products = [t.strip() for t in products.split(' + ')]
    return (reactants,products,kind)
_split_reaction_functions['standard'] = _f

def _f(reaction):
    if '->' in reaction:
        kind = 'forward'
        reactants,products = reaction.split('->')
    elif '<-' in reaction:
        kind = 'backward'
        reactants,products = reaction.split('<-')
    elif '<=>' in reaction:
        kind = 'equilibrium'
        reactants,products = reaction.split('<=>')
    else:
        raise Exception(f'Cannot split {reaction=} with {encoding=}')
    reactants = [t.strip() for t in reactants.split(' + ')]
    products = [t.strip() for t in products.split(' + ')]
    return (reactants,products,kind)
_split_reaction_functions['ascii'] = _split_reaction_functions['argo'] = _f

## formulae for computing rate coefficients from reaction constants c
## and state variables p
_reaction_coefficient_formulae = {
    'constant'               :lambda c,p: c['k'],
    'arrhenius'              :lambda c,p: c['A']*(p['T']/300.)**c['B'],
    'KIDA modified arrhenius':lambda c,p: c['A']*(p['T']/300.)**c['B']*np.exp(-c['C']*p['T']),
    'NIST'                   :lambda c,p: c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/8.314472e-3/p['T']),
    'NIST_3rd_body_hack'     :lambda c,p: 1e19*c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/8.314472e-3/p['T']),
    'photoreaction'          :lambda c,p: scipy.integrate.trapz(c['σ'](p['T'])*p['I'],c['λ']),
    'kooij'                  :lambda c,p: c['α']*(p['T']/300.)**c['β']*np.exp(-c['γ']/p['T']),
} 
    
class Reaction:
    """A class for manipulating a chemical reaction."""

    def __init__(
            self,
            name,
            encoding='standard',
            formula='constant', # type of reaction, defined in get_rate_coefficient
            **coefficients,     # used for computing rate coefficient according to formula
    ):
        t = decode_reaction(name,encoding)
        self.reactants = t['reactants']
        self.products = t['products']
        self.formula = _reaction_coefficient_formulae[formula]
        self.coefficients = coefficients 
        self.rate_coefficient = None     

    def get_hash(self):
        """A convenient way to summarise a reaction by its name. I don't use
        __hash__ because I worry that the reactants/products might mutate."""
        return(hash(' + '.join(sorted([t.name for t in self.reactants]))+' ⟶ '+' + '.join(sorted([t.name for t in self.products]))))

    def __getitem__(self,key):
        return(self.coefficients[key])

    def __setitem__(self,key,val):
        self.coefficients[key] = val

    def get_rate_coefficient(self,**state_variables):
        """Calculate rate coefficient from rate constants and state
        variables."""
        self.rate_coefficient = self.formula(self.coefficients,state_variables)
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

    def __str__(self):
        return ', '.join([f'name=" {self.name:60}"']
                         +[format(f'formula="{self.formula.strip()}"',"20")]
                         +[format(f'{key}={repr(val)}','20') for key,val in self.coefficients.items()])


class Reaction_Network:

    def __init__(self,kida_filename=None):
        self.reactions = []
        self.T = 300            # constant temperature or a function of time
        self.species = {}       # indexd by species name
        self.n = dict()         # indexd by species name
        self.rate = dict()      # indexd by species name
        if kida_filename is not None:
            self.load_from_kida(kida_filename)

    def print_rate_coefficients(self):
        for reaction in self.reactions:
            print( format(reaction.name,'40'),reaction.get_rate_coefficient(T=self.T))

    def get_rates(self,time,density):
        """Calculate all rate coefficients in the network at a given time and
        with given densities."""
        rates = np.zeros(len(density),dtype=float) # rates for all species
        self.reaction_rates = np.zeros(len(self.reactions),dtype=float)
        self.rate_coefficients = np.zeros(len(self.reactions),dtype=float)
        ## T is a constant or function fo time
        if np.isscalar(self.T):
            T = self.T
        else:
            T = self.T(time)
        for ireaction,reaction in enumerate(self.reactions):
            ## compute rates of reaction
            rate_coefficient = reaction.get_rate_coefficient(T=T)
            reaction_rate = (rate_coefficient*np.prod([
                                density[self._species_index[reactant.name]]
                                 for reactant in reaction.reactants]))
            self.reaction_rates[ireaction] = reaction_rate
            self.rate_coefficients[ireaction] = rate_coefficient
            ## add contribution to product/reactant
            ## formation/destruction rates
            for reactant in reaction.reactants:
                rates[self._species_index[reactant.name]] -= reaction_rate
            for product in reaction.products:
                rates[self._species_index[product.name]] += reaction_rate
        self.rates = rates
        return(rates)

    def integrate(self,time,nsave_points=10,**initial_densities):
        ## collect all species names
        species = set()
        for reaction in self.reactions:
            for reactant in reaction.reactants:
                species.add(reactant.name)
            for product in reaction.products:
                species.add(product.name)
        for name in initial_densities:
            species.add(name)
        ## species_name:index_of_species_in_internal_arrays
        self._species_index = {name:i for (i,name) in enumerate(species)} 
        ## time steps
        time = np.array(time,dtype=float)
        if time[0]!=0:
            time = np.concatenate(([0],time))
        density = np.full((len(species),len(time)),0.)
        ## set initial conditions
        for key,val in initial_densities.items():
            density[self._species_index[key],0] = val
        ## initialise integrator
        r = integrate.ode(self.get_rates)
        r.set_integrator('lsoda', with_jacobian=True)
        r.set_initial_value(density[:,0])
        ## run saving data at requested number of times
        for itime,timei in enumerate(time[1:]):
            r.integrate(timei)
            density[:,itime+1] = r.y
        ##
        retval = Dynamic_Recarray(
            time=time,
            T=self.T(time),
        )
        for speciesi,densityi in zip(species,density):
            retval[speciesi] = densityi
        self.species = species
        self.density = density
        self.time = time
        self.rates = self.get_rates(self.time[-1],self.density[:,-1])
        return(retval)

    def print_rates(self):
        for reaction,rate,rate_coefficient in zip(self.reactions,self.reaction_rates,self.rate_coefficients):
            print( format(reaction.name,'35'),format(rate_coefficient,'<+10.2e'),format(rate,'<+10.2e'))

    def append(self,*args,**kwargs):
        """Append a Reaction, or arguments to define one."""
        if len(args)==1 and len(kwargs)==0 and  isinstance(args[0],Reaction):
            self.reactions.append(args[0])
        else:
            self.reactions.append(Reaction(*args,**kwargs))


    def extend(self,reactions):
        self.reactions.extend(reactions)

    ## a list of unique reactants, useful where these branch
    unique_reactants = property(lambda self:set([t.reactants for t in self.reactions]))

    def get_species(self):
        """Return a list of all species in this network."""
        return(list(np.unique(np.concatenate([list(t.products)+list(t.reactants) for t in self.reactions]))))

    def get_product_branches(self,reactants,with_reactants=[],without_reactants=[],with_products=[],without_products=[]):
        """Get a list of reactions with different products and the
        same reactants. Restricut to some products"""
        return([t for t in self.reactions
                if t.reactants==reactants
                and np.all([t1 in t.products  for t1 in with_products])
                and not np.any([t1 in t.products for t1 in without_products])
                and np.all([t1 in t.reactants for t1 in with_reactants])
                and not np.any([t1 not in t.reactants for t1 in without_reactants])])

    def __iter__(self):
        for t in self.reactions: yield(t)

    def __len__(self): return(len(self.reactions))

    def get_matching_reactions(
            self,
            reactants=None,products=None,
            not_reactants=None,not_products=None,
    ):
        """Return a list of reactions with this reactant."""
        retval = []
        for r in self.reactions:
            if reactants is not None:
                if  np.isscalar(reactants):
                    if reactants not in r.reactants:
                        continue
                else:
                    if not all(reactants in r.reactants):
                        continue
            retval.append(r)
            # if reactants is not None:
                # if  np.isscalar(reactants):
                    # if reactants not in t.reactants:
                        # continue
                # else:
                    # if not all(reactants in t.reactants):
                        # continue

        # return([t for t in self.reactions if (
            # t.reaction_type in reaction_type
            # and all(
                    # [t0 in t.reactants for t0 in reactants]
                    # +[t0 in t.products for t0 in products]
                    # +[t0 not in t.reactants for t0 in not_reactants]
                    # +[t0 not in t.products for t0 in not_products]
                # ))])
        return retval

    def get_reaction(self,name):
        for t in self.reactions:
            if t.name==name:
                return(t)
        else:
            raise IndexError('Could not find reaction: '+repr(name))    

    def __str__(self):
        retval = []
        for name,species in self.species.items():
            retval.append(f'{name:20} {species.density}')
        for t in self.reactions:
            retval.append(str(t))
        return('\n'.join(retval))

    # def get_rates(self):
        # self.rates = dict()
        # for reaction in self.reactions:
            # self.rates[reaction] = reaction.get_rate(T=self.T,n=self.n)
            # for multiplicity,species in reaction.reactants:
                # self.rate[species] -= multiplicity*self.rates[reaction]
            # for multiplicity,species in reaction.products:
                # self.rate[species] += multiplicity*self.rates[reaction]

    # def load(self,filename):


    def save(self,filename):
        """Save encoded reactions and coefficients to a file."""
        my.string_to_file(filename,str(self))

    def load(self,filename):
        """Load encoded reactions and coefficients from a file."""
        with open(my.expand_path(filename),'r') as fid:
            for line in fid:
                self.append(**eval(f'dict({line[:-1]})'))


    def get_recarray(self,keys=None):
        data = collections.OrderedDict()
        ## list reactants up to max number of reactants
        for i in range(np.max([len(t.reactants) for t in self.reactions])):
            data['reactant'+str(i)] = [t.reactants[i] if len(t.reactants)>i else ''for t in self.reactions]
        ## list products up to max number of products
        for i in range(np.max([len(t.products) for t in self.reactions])):
            data['product'+str(i)] = [t.products[i] if len(t.products)>i else '' for t in self.reactions]
        ## all coefficients
        if keys is None:
            keys = np.unique(np.concatenate(
                [list(t.coefficients.keys()) for t in self.reactions]))
        for key in keys:
            data[key] = [t[key] if key in t.coefficients else np.nan for t in self.reactions]
        return(my.dict_to_recarray(data))



