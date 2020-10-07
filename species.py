from . import database

import re
import periodictable


_get_species_cache = {}
def get_species(name):
    """Get species from name, assume immutable and potentially
    cached."""
    if name not in _get_species_cache:
        _get_species_cache[name] = Species(name=name)
    return(_get_species_cache[name])
    

class Species:
    """Info about a species. Currently assumed to be immutable data only."""

    def __init__(
            self,
            name,
            elements=None,
            charge=None,
            isomer=None,
            encoding=None,
    ):

        ## look for isomer, e..,g c-C3 vs l-C3
        if r:=re.match('^(.+)-(.*)$',name):
            self.isomer,name = r.groups()
        else:
            self.isomer = None
        ## decode name
        ## translate diatomic special cases -- remove one day, e.g., 32S16O to [32]S[16]O
        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)$',r'[\1\2][\3\4]',name)
        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)2$',r'[\1\2]2',name)
        if encoding is not None:
            name = translate_species_to_standard(encoding,name)
        self.name = name # must be unique -- use a hash instead?
        self._reduced_elements = None            # keys are names of elements, values are multiplicity in ONE molecule
        self._elements = None            # list of elements, possibly repeated
        self._element_species = None            # list of elements as Species objects
        self._charge = None
        self._nelectrons = None
        self._mass = None            # of one molecule
        self._reduced_mass = None   # reduced mass (amu), actually a property
        self._nonisotopic_name = None # with mass symbols removed, e.g., '[32S][16O]' to 'SO'
        self._point_group = None
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

    def _get_point_group(self):
        if self._point_group is None:
            self._point_group = database.get_species_property(self.name,'point_group')
        return(self._point_group)

    point_group = property(_get_point_group)

    def _get_reduced_mass(self):
        if self._reduced_mass is None:
            self._reduced_mass = database.get_species_property(self.name,'reduced_mass')
        return(self._reduced_mass)

    def _set_reduced_mass(self,reduced_mass):
        self._reduced_mass = reduced_mass

    reduced_mass = property(_get_reduced_mass,_set_reduced_mass)
    
    def _get_mass(self):
        self._mass = 0.
        for element in self.elements:
            mass,telement = re.match('([0-9]+)([A-Z][a-z]?)',element).groups()
            self._mass += getattr(periodictable,telement)[int(mass)].mass
        return(self._mass)

    mass = property(_get_mass)
    
    def _get_charge(self):
        if self._charge is not None:
            pass
        elif self.name=='e-':
            self._charge = -1
        elif self.name=='photon':
            self.charge = 0
        elif r:=re.match('^(.*)\^([-+][0-9]+)$',self.name): # ddd^+3 / ddd^-3 / ddd^3
            self._charge = int(r.group(2))
        elif r:=re.match('^(.*)\^([0-9]+)([+-])$',self.name): # ddd^3+ / ddd^3-
            self._charge = int(r.group(3)+r.group(2))
        elif r:=re.match('^(.*[^+-])(\++|-+)$',self.name): # ddd+ / ddd++
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
            
##    def decode_name(self,name,encoding=None):
##        """Decode a name and set all internal variables."""
##        ## translate diatomic special cases -- remove one day, e.g., 32S16O to [32]S[16]O
##        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)([0-9]+)([A-Z][a-z]?)$',r'[\1\2][\3\4]',name)
##        name = re.sub(r'^([0-9]+)([A-Z][a-z]?)2$',r'[\1\2]2',name)
##        ## translate from some other encoding if requested
##        if encoding is not None:
##            name = translate_species_to_standard(encoding,name)
##        self.name = name
##        self.reduced_elements={}            # keys are names of elements, values are multiplicity in ONE molecule
##        self.elements=[]            # keys are names of elements, values are multiplicity in ONE molecule
##        self.charge=0
##        self.isomer=''
##        self.nelectrons=0
##        self.mass=np.nan            # of one molecule
##        ## short cut for electrons and photons
##        if name=='e-':
##            self.charge = -1
##            self.nelectrons = 1
##            self.mass = constants.m_e
##        elif name=='photon':
##            self.charge = 0
##            self.nelectrons = 0
##            self.mass = 0
##        else:
##            ## get charge
##            for t in [0]:
##                r = re.match('^(.*)\^([-+][0-9]+)$',name) # ddd^+3 / ddd^-3 / ddd^3
##                if r:
##                    self.charge = int(r.group(2))
##                    name = r.group(1)
##                    break
##                r = re.match('^(.*)\^([0-9]+)([+-])$',name) # ddd^3+ / ddd^3-
##                if r:
##                    self.charge = int(r.group(3)+r.group(2))
##                    name = r.group(1)
##                    break
##                r = re.match('^(.*[^+-])(\++|-+)$',name) # ddd+ / ddd++
##                if r:
##                    self.charge = r.group(2).count('+') - r.group(2).count('-')
##                    name = r.group(1)
##                    break
##            ## get isomer
##            r = re.match('^(.+)-(.*)$',name)
##            if r is not None:
##                self.isomer,name = r.groups()
##            ## get elements and their multiplicity
##            for part in re.split(r'(\[[0-9]*[A-Z][a-z]?\][0-9]*|[A-Z][a-z]?[0-9]*)',name):
##                if len(part)==0: continue
##                r = re.match('^(.+?)([0-9]*)$',part)
##                if r.group(2)=='':
##                    multiplicity=1
##                else:
##                    multiplicity = int(r.group(2))
##                element = r.group(1)
##                # element = element.replace(']','').replace('[','')
##                for i in range(multiplicity):
##                    self.elements.append(element)
##                if element in self.reduced_elements:
##                    self.reduced_elements[element] += multiplicity
##                else:
##                    self.reduced_elements[element] = multiplicity
##            ## try determine mass in amu and number of electrons --
##            ## MOVE THIS TO A PROPERTY?
##            self.mass = 0.
##            self.nelectrons = 0
##            for element,multiplicity in self.reduced_elements.items():
##                r = re.match(r'([0-9]+)([A-Za-z])+',element)
##                if r:
##                    isotope,element = r.groups()
##                    self.mass += multiplicity*getattr(periodictable,element)[int(isotope)].mass # isotopic mass
##                else:
##                    self.mass += multiplicity*getattr(periodictable,element).mass # natural mixture mass
##                self.nelectrons += multiplicity*getattr(periodictable,element).number # add electrons attributable to each nucleus
##            self.nelectrons -= self.charge # account for ionisation
        
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

    # def __str__(self):
    #     return('\n'.join([
    #         f'{"name":<10s} = {repr(self.name)}',
    #         f'{"elements":<10s} = {repr(self.elements)}',
    #         f'{"reduced_elements":<10s} = {repr(self.reduced_elements)}',
    #         f'{"charge":<10s} = {repr(self.charge)}',
    #         f'{"isomer":<10s} = {repr(self.isomer)}',
    #         f'{"nelectrons":<10s} = {repr(self.nelectrons)}',
    #         f'{"mass":<10s} = {repr(self.mass)}',
    #         ]))

    def __str__(self): return(self.name)

    ## for sorting a list of Species objects
    def __lt__(self,other): return(self.name<other.name)
    def __gt__(self,other): return(self.name>other.name)
    
