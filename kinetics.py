import re
import warnings
from copy import copy
from functools import lru_cache
from pprint import pprint

import numpy as np
from numpy import array,arange
from bidict import bidict
import scipy

from . import tools
from . import convert
from .tools import cache
from .dataset import Dataset
from .exceptions import DecodeSpeciesException,InferException
from . import database
from . import plotting


#############
## species ##
#############


_chemical_species_hacks = {
    'HCO₂H':'HCOOH',
    'SCS': 'CS₂',
}

@cache
def get_inchikey(name):
    return Species(name)['inchikey']

@cache
def get_species(name):
    return Species(name=name)

@cache
def get_species_property(name,prop):
    species = Species(name=name)
    retval = species[prop]
    return retval

@cache
def get_chemical_species(name):
    species = get_species(name)
    return species['chemical_species']
    
class Species:
    """Info about a species. Currently assumed to be immutable data only."""

    all_properties = ('name', 'charge', 'elements', 'species',
                      'isotopes', 'natoms', 'nelectrons', 'mass', 'reduced_mass',)

    def __init__(
            self,
            name,
            name_encoding='ascii_or_unicode',
            encoding='unicode',
    ):

        if name_encoding == 'ascii_or_unicode':
            if re.match(r'.*[0-9+-].*',name):
                name_encoding = 'ascii'
            else:
                name_encoding = 'unicode'
        self._name = convert.species(name,name_encoding,encoding)
        self.encoding = encoding
        self._decode_name()
                
    def _decode_name(self):
        """Turn standard name string into ordered isotope list and charge.  If
        any isotopic masses are given then they will be added to all
        elements."""
        name = convert.species(self.name,self.encoding,'unicode') 
        ## e.g., ¹²C¹⁶O₂²⁺
        r = re.match(r'^((?:[⁰¹²³⁴⁵⁶⁷⁸⁹]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻]?)$',name)
        if not r:
            raise Exception(f'Could not decode unicode encoded species name: {name!r}')
        name_no_charge = r.group(1)
        if r.group(2) == '':
            charge = 0
        elif r.group(2) == '⁺':
            charge = +1
        elif r.group(2) == '⁻':
            charge = -1
        elif '⁺' in r.group(2):
            charge = int(tools.regularise_unicode(r.group(2)[:-1]))
        else:
            charge = -int(tools.regularise_unicode(r.group(2)[:-1]))
        elements_isotopes = []                   # (element,mass_number,multiplicity)
        isotope_found = False
        element_found = False
        for part in re.split(r'([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)',name_no_charge):
            if part=='':
                continue
            elif r:= re.match(r'([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ]*)([A-Z][a-z]?)([₀₁₂₃₄₅₆₇₈₉]*)',part):
                mass_number = ( int(tools.regularise_unicode(r.group(1))) if r.group(1) != '' else None )
                element = r.group(2)
                multiplicity = int(tools.regularise_unicode(r.group(3)) if r.group(3) != '' else 1)
                if mass_number is None:
                    element_found = True
                    elements_isotopes.append([element,multiplicity])
                else:
                    isotope_found = True 
                    elements_isotopes.append([mass_number,element,multiplicity])
            else:
                raise Exception(f'Could not decode element name {repr(part)} in  {repr(name)}')
        ## if any masses given, then make sure all are specified
        if isotope_found:
            if element_found:
                for i,t in enumerate(elements_isotopes):
                    if len(t) == 2:
                        elements_isotopes[i] = (database.get_most_abundant_isotope_mass_number(t[0]),t[0],t[1])
            isotopes = elements_isotopes
            elements = [t[1:] for t in isotopes]
            ## combine similar elements
            i = 0
            while i < (len(elements)-1):
                if elements[i][0] == elements[i+1][0]:
                    elements[i][1] += elements[i+1][1]
                    elements.pop(i+1)
                else:
                    i += 1
        elif element_found:
            isotopes = None
            elements = elements_isotopes
        else:
            raise Exception(f'Could not decode element name: {repr(name)}')
        self._elements = elements
        self._isotopes = isotopes
        self._charge = charge

    def __str__(self):
        retval = '\n'.join([
            f'{key:16} = {getattr(self,key)!r}' 
            for key in self.all_properties])
        return retval

    ## for sorting a list of Species objects
    def __lt__(self,other):
        return self.name < other

    def __gt__(self,other):
        return self.name > other

    def __getitem__(self,key):
        """Access these properties by index rather than attributes in order
        for simple caching.  -- move other get_ methods to here someday"""
        if key in self.all_properties:
            return getattr(self,key)
        elif key == 'chemical_species':
            return self.chemical_species
        elif key == 'point_group':
            ## deduce point group
            if len(self.elements) == 1:
                ## atoms
                return "K"
            elif len(self.elements) == 2:
                ## Homonumclear or heteronuclear diatomic
                if self.elements[0] == self.elements[1]:
                    return 'D∞h'
                else:
                    return 'C∞v'
            else:
                raise ImplementationError("Can only compute reduced mass for atoms and diatomic species.")
        elif key == 'matplotlib_name':
            return self.translate_name('matplotlib')
        else:
            raise DecodeSpeciesException(f"Unknown species property: {key}")

    def translate_name(self,encoding):
        
        if encoding == 'matplotlib':
            name = self['name']
            while r:=re.match(r'(^[^⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁺⁻]*)([⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ⁺⁻]+)(.*)$',name):
                name = r.group(1)+r'$^{'+tools.regularise_unicode(r.group(2))+r'}$'+r.group(3)
            while r:=re.match(r'^([^₀₁₂₃₄₅₆₇₈₉]*)([₀₁₂₃₄₅₆₇₈₉]+)(.*)$',name):
                name = r.group(1)+r'$_{'+tools.regularise_unicode(r.group(2))+'}$'+r.group(3)
        else:
            raise Exception
        return name

    def is_isotopologue(self):
        """Is this species a particular isotopologue."""
        if self['chemical_species'] == self['name']:
            return False
        else:
            return True

    def _get_unicode_encoded_charge(self):
        if self.charge == 0:
            retval = ''
        elif self.charge < -1:
            retval = str(-self.charge)+'-'
        elif self.charge < 0:
            retval = '-'
        elif self.charge > 1:
            retval = str(self.charge)+'+'
        else:
            retval = '+'
        retval = tools.superscript_numerals(retval)
        return retval
    
    def _get_nelectrons(self):
        if not hasattr(self,'_nelectrons'):
            import periodictable
            if self.chemical_species in ('e-','photon'):
                raise NotImplementedError()
            self._nelectrons = 0
            ## add electrons attributable to each nucleus
            for element,multiplicity in self['elements']:
                self._nelectrons += multiplicity*getattr(periodictable,element).number
            ## account for ionisation
            self._nelectrons -= self['charge']
        return self._nelectrons

    def _get_species(self):
        if not hasattr(self,'_species'):
            if self._isotopes is None:
                self._species = None
            else:
                parts = []
                for mass,element,mult in self.isotopes:
                    parts.append(tools.superscript_numerals(str(mass)))
                    parts.append(element)
                    if mult > 1:
                        parts.append(tools.superscript_numerals(tools.subscript_numerals(str(mult))))
                parts.append(self._get_unicode_encoded_charge())
                self._species = ''.join(parts)
        return self._species
    
    def _get_chemical_species(self):
        if not hasattr(self,'_chemical_species'):
            parts = []
            for element,mult in self.elements:
                parts.append(element)
                if mult > 1:
                    parts.append(tools.superscript_numerals(tools.subscript_numerals(str(mult))))
            parts.append(self._get_unicode_encoded_charge())
            self._chemical_species = ''.join(parts)
        return self._chemical_species
                                    
    def _get_elements(self):
        return self._elements
        
    def _get_mass(self):
        if not hasattr(self,'_mass'):
            if self.isotopes is None:
                self._mass = sum([
                    database.get_atomic_mass(element)*multiplicity
                    for element,multiplicity in self.elements])
            else:
                self._mass =  sum([
                    database.get_atomic_mass(element,mass_number)*multiplicity
                    for mass_number,element,multiplicity in self.isotopes])
        return self._mass

    def _get_natoms(self):
        if not hasattr(self,'_natoms'):
            self._natoms = sum([mult for element,mult in self.elements])
        return self._natoms
            
    def _get_reduced_mass(self):
        if not hasattr(self,'_reduced_mass'):
            if self.natoms != 2:
                self._reduced_mass = None
            if self.isotopes is not None:
                m1 = database.get_atomic_mass(self.isotopes[0][1],self.isotopes[0][0])
                m2 = database.get_atomic_mass(self.isotopes[1][1],self.isotopes[1][0])
            else:
                m1 = database.get_atomic_mass(self.elements[0][0])
                m2 = database.get_atomic_mass(self.elements[1][0])
            self._reduced_mass = m1*m2/(m1+m2)
        return self._reduced_mass

    name = property(lambda self: self._name)
    elements = property(lambda self: self._elements)
    isotopes = property(lambda self: self._isotopes)
    charge = property(lambda self: self._charge)
    chemical_species = property(_get_chemical_species)
    species = property(_get_species)
    nelectrons = property(_get_nelectrons)
    natoms = property(_get_natoms)
    mass = property(_get_mass)
    reduced_mass = property(_get_reduced_mass)
    point_group = property(lambda self: self['point_group'])


class Mixture():
    """A zero-dimesional time-dependent chemical mixture."""

    def __init__(self):
        self.reaction_network = None
        self.density = Dataset()
        self.state = Dataset()

    def __getitem__(self,key):
        if (self.state.is_known(key)
            or key in self.state.prototypes):
            ## return state variable
            return self.state[key]
        elif key in self.density:
            ## return density of species
            return self.density[key]
        elif r:=re.match(r'x\((.+)\)',key):
            ## match mixing ratio of species
            return self.get_mixing_ratio(r.group(1))
        elif r:=re.match(r'n\((.+)\)',key):
            ## density of species
            return self.density[r.group(1)]
        else:
            raise Exception(f'Unknown {key=}')

    def get_species(self):
        """Get a list of all chemical species."""
        return list(self.density.keys())

    def get_mixing_ratio(self,species):
        return self.density[species]/self.state['nt']

    def __len__(self):
        return len(self.state)

    def get_rates(
            self,
            sort_method='max anywhere', # maximum somewhere
            nsort=3,           # return this many rates everywhere
            **kwargs_get_reactions 
    ):
        """Return larges-to-smallest reaction rates matching
        kwargs_get_reactions. """
        reaction_names = []
        rates = []
        for reaction in self.reaction_network.get_reactions(**kwargs_get_reactions):
            if reaction.rate is None:
                raise Exception(f'Reaction rate not set: {reaction}')
            rate = copy(reaction.rate)
            reaction_names.append(reaction.name)
            rates.append(rate)
        ## add total
        total = np.nansum([t for t in rates],axis=0)
        ## sort
        if sort_method == 'maximum':
            ## list all reactions by their reverse maximum rate 
            i = np.argsort([-np.max(t) for t in rates])
        elif sort_method == 'max anywhere':
            ## return all reactions that are in the top 5 ranking anywhere in their domain
            i = np.argsort(-np.row_stack(rates),0)
            i = np.unique(i[:(nsort),:].flatten()) 
        else:
            raise Exception(f'Unknown {sort_method=}')
        ## return as dict
        retval = {'total':total}
        retval.update({reaction_names[ii]:rates[ii] for ii in i})
        return retval

    def plot_density(
            self,
            species=5,
            xkey='t',
            ax=None,
            labelstyle='legend',
    ):
        """Plot density of species. If xkeys is an integer then plot that many
        most abundant anywhere species. Or else give a list of species
        names."""
        if isinstance(species,int):
            ## get most abundance species anywhere
            all_keys = np.array(self.density.keys())
            nsort = species
            species = []
            for i in range(len(self.density)):
                j = np.argsort([-self.density[t][i] for t in self.density])
                species.extend(all_keys[j[:nsort]])
            species = tools.unique(species)
            i = np.argsort([-np.max(self.density[tspecies]) for tspecies in species])
            species = np.array(species)[i]
        if ax is None:
            ax = plotting.gca()
        # ## plot total density
        # ax.plot(self['nt'],self[ykey],label='nt',color='black',alpha=0.3,linewidth=6)
        ## plot individual species
        for ykey in species:
            ax.plot(self[xkey],self.density[ykey],label=ykey)
        ax.set_yscale('log')
        # ax.set_ylim(self[ykey].min(),self[ykey].max())
        ax.set_xlabel(ykey)
        ax.set_ylabel('Density (cm-3)')
        if labelstyle == 'legend':
            plotting.legend(ax=ax)
        elif labelstyle == 'annotate':
            plotting.annotate_line(ax=ax)
        else:
            raise Exception(f'Unknown labelstyle: {repr(labelstyle)}')
                

    def plot_rates(
            self,
            xkey='t',
            ax=None,
            plot_total=True,    # plot sum of all rates
            plot_kwargs=None,
            normalise_to_species=None, # normalise rates divided by the density of this species
            **kwargs_get_rates         # for selecting rates to plot
    ):
        """Plot rates matching kwargs_get_rates."""
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        if plot_kwargs is None:
            plot_kwargs = {}
        x = self[xkey]
        rates = self.get_rates(**kwargs_get_rates)
        ylabel = 'Rate (cm$^{-3}$ s$^{-1}$)'
        names = list(rates)
        if normalise_to_species:
            for key in rates:
                rates[key] /= self.density[normalise_to_species]
        ## plot
        for i,name in enumerate(names):
            if name == 'total' and not plot_total:
                continue
            t_plot_kwargs = copy(plot_kwargs)
            t_plot_kwargs['label'] = f'{name}'
            if name == 'total':
                t_plot_kwargs.update(color='black',alpha=0.3,linewidth=6)
            ax.plot(x,rates[name],**t_plot_kwargs)
        ax.set_xlabel(xkey)
        ax.set_xlabel(ylabel)
        ax.set_yscale('log')
        ax.set_xlim(x.min(),x.max())
        plotting.legend()
        return ax,rates

    def plot_production_destruction(
            self,
            species,
            xkey='t',
            ax=None,
            normalise=False,    # divide rates by species abundance
            nsort=3,            # include this many ranked rates at each altitude
            plot_production=True,
            plot_destruction=True,
            plot_total=True,
            plot_difference=True,
            separate_axes=False
    ):
        """Plot destruction and production rates of on especies."""
        if separate_axes:
            fig = plotting.gcf()
            fig.clf()
            ax = plotting.subplot()
            self.plot_production_destruction(species,xkey,ax,normalise,nsort,True,False,plot_total,plot_difference,False)
            ax = plotting.subplot()
            self.plot_production_destruction(species,xkey,ax,normalise,nsort,False,True,plot_total,plot_difference,False)
            return
        ## start plot
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        ## production rates
        if plot_production:
            tax,production_rates = self.plot_rates(
                xkey=xkey,ax=ax,plot_total=plot_total,
                with_products=species,
                plot_kwargs={'linestyle':'-'},
                normalise_to_species=(species if normalise else None),
                nsort=nsort,
            )
        ## destruction
        if plot_destruction:
            tax,destruction_rates = self.plot_rates(
                xkey=xkey,ax=ax,plot_total=plot_total,
                with_reactants=species,
                plot_kwargs={'linestyle':':',},
                normalise_to_species=(species if normalise else None),
                nsort=nsort
            )
        ## plot the difference between production and
        ## destruction
        if plot_difference and plot_production and plot_destruction and plot_total:
            x = self[xkey]
            y = production_rates['total']-destruction_rates['total']
            i = y>0
            kwargs = dict(alpha=0.3,markersize=10,marker='o',linestyle='',zorder=-10)
            ax.plot(x[i],y[i],label='total production > destruction',color='blue',**kwargs)
            ax.plot(x[~i],np.abs(y[~i]),label='total production < destruction',color='red',**kwargs)
        ## finish plot
        if plot_production and plot_destruction:
            title = f'Production and destruction rates of {species}'
        elif plot_production:
            title = f'Production rates of {species}'
        elif plot_destruction:
            title = f'Destruction rates of {species}'
        else:
            title = ''    
        plotting.legend(show_style=True,title=title)
        return ax

    # def get_most_abundant(self,n=5):
        # """Return names of the n most abundant species."""
        # keys = [key for key in self if len(key)>2 and key[:2] == 'n_']
        # i = np.argsort([self[key].max() for key in keys])
        # retval = [keys[ii][2:] for ii in i[-1:-n:-1]]
        # return retval

    def load_cantera(
            self,
            gas,
            states=None,
            forward_rates=None,
            reverse_rates=None,
    ):
        if states is not None:
            ## load state
            self.state.extend(t=states.t,T=states.T,p=states.P,keys='new')
            ## load densities -- empty if states is None
            for species in gas.kinetics_species_names:
                self.density[species] = states.X[:, gas.species_index(species)]
        ## load reaction netwrok
        self.reaction_network = ReactionNetwork()
        self.reaction_network.load_cantera(gas,forward_rates,reverse_rates)
        

########################
## Chemical reactions ##
########################

def decode_reaction(reaction,encoding='ascii'):
    """Decode a reaction into a list of reactants and products, and other
    information. Encoding is for species names. Encoding of reaction
    string formatting not implemented."""
    if encoding == 'ascii':
        ## split parts
        reactants_string,products_string = reaction.split('→')
        reactants,products = [],[]
        for r_or_p_string,r_or_p_list in ((reactants_string,reactants), (products_string,products)):
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
    elif encoding == 'cantera':
        reaction = reaction.replace('(+M)',' + M')
        reaction = reaction.replace('(+M)',' + M')
        ## split parts
        reactants_string,products_string = re.split(r'\s[<>=]+\s',reaction)
        ## delete reference to 3rd bodies
        reactants,products = [],[]
        for r_or_p_string,r_or_p_list in ((reactants_string,reactants), (products_string,products)):
            ## decode reactants or products string into a list
            for species in r_or_p_string.split(' + '):
                species = species.strip()
                # if species == 'M':
                    # ## neglect 3rd bodies
                    # continue
                if r:=re.match(r'^([0-9]+) *(.*)',species):
                    multiplicity = int(r.group(1))
                    species = r.group(2)
                else:
                    multiplicity = 1
                species = convert.species(species,encoding,'unicode')
                for i in range(multiplicity):
                    r_or_p_list.append(species)
            r_or_p_list.sort()
    else:
        raise Exception(f"Unknown reaction encoding: {repr(encoding)}")
    return reactants,products

def encode_reaction(reactants,products,encoding='ascii'):
    """Only just started"""
    if encoding == 'ascii':
        retval = ' + '.join(reactants)+' -> '+' + '.join(products)
    elif encoding == 'unicode':
        retval = ' + '.join(reactants)+' → '+' + '.join(products)
    else:
        raise NotImplementedError(f'Unimplemented reation encoding: {encoding!r}')
    return retval

############################################################################
## formulae for computing rate coefficients from reaction constants c and ##
## state variables p                                                      ##
############################################################################

def _rimmer2016_3_body(c,p):
    """Eqs. 9-11 rimmer2016"""
    k0 = c['α0']*(p['T']/300)**c['β0']*np.exp(-c['γ0']/p['T'])
    kinf = c['αinf']*(p['T']/300)**c['βinf']*np.exp(-c['γinf']/p['T'])
    pr = k0*p['nt']/kinf
    k2 = (kinf*pr)/(1+pr)
    return k2

def _test(c,p):
    nt = p['nt']
    T = p['T']
    ## forward reaction N + N → N2
    k0 = c['α0']*(T/300)**c['β0']*np.exp(-c['γ0']/T)
    kinf = c['αinf']*(T/300)**c['βinf']*np.exp(-c['γinf']/T)
    pr = k0*nt/kinf
    k2 = (kinf*pr)/(1+pr)
    ## reverse it
    a = {
        'N':[    2.4159e+00,1.7489e-04,-1.1902e-07,3.0226e-11,-2.0361e-15,5.6134e+04,4.6496e+00],
        'N2':[   2.9526e+00,1.3969e-03,-4.9263e-07,7.8601e-11,-4.6076e-15,-9.2395e+02,5.8719e+00],
    }
    reactants = ['N2']
    products = ['N','N']
    Greactants = 0.
    for s in reactants:
        a1,a2,a3,a4,a5,a6,a7 = a[s]
        Greactants += a1*np.log(T-1) + a2*T/2 + a3*T**2/6 + a4*T**3/12 + a5*T**4/20 + a6/T + a7
    Gproducts = 0.
    for s in products:
        a1,a2,a3,a4,a5,a6,a7 = a[s]
        Gproducts += a1*np.log(T-1) + a2*T/2 + a3*T**2/6 + a4*T**3/12 + a5*T**4/20 + a6/T + a7
    # Kc = (constants.R*T)**(len(products)-len(reactants))*np.exp(Gproducts-Greactants)
    factor = np.exp(Greactants-Gproducts)
    k2reversed =  k2/factor
    # k2rev = 4e-32
    # k2rev
    # print('DEBUG:',nt,T,Gproducts,Greactants,k2,k2reversed)
    print('DEBUG:',
          format(Greactants,'12.3e'),
          format(Gproducts,'12.3e'),
          format(T,'12.3e'),
          format(nt,'12.3e'),
          format(factor,'12.3e'),
          format(k2,'12.3e'),
          format(k2reversed,'12.3e'),
          )
    return k2reversed
    

_reaction_coefficient_formulae = {
    'constant'               :lambda c,p: c['k'],
    'arrhenius'              :lambda c,p: c['A']*(p['T']/300.)**c['B'],
    'KIDA modified arrhenius':lambda c,p: c['A']*(p['T']/300.)**c['B']*np.exp(-c['C']*p['T']),
    'NIST'                   :lambda c,p: c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/(scipy.constants.R*p['T'])),
    'NIST_3rd_body_hack'     :lambda c,p: 1e19*c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/(scipy.constants.R*p['T'])),
    'photoreaction'          :lambda c,p: scipy.integrate.trapz(c['σ'](p['T'])*p['I'],c['λ']),
    'kooij'                  :lambda c,p: c['α']*(p['T']/300.)**c['β']*np.exp(-c['γ']/p['T']), # α[(cm3)^(n-1).s-1], T[K], β[], γ[K]
    'impact_test_2020-12-07' :lambda c,p: p['T']*0 ,
    'rimmer2016_3_body'      :_rimmer2016_3_body,
    'test'      :_test,
} 

def _f(c,p):
    """STAND 3-body reaction scheme. Eqs. 9,10,11 in rimmer2016."""
    k0 = c['α0']*(p['T']/300)**c['β0']*np.exp(-c['γ0']/p['T']) 
    kinf = c['αinf']*(p['T']/300.)**c['βinf']*np.exp(-c['γinf']/p['T'])
    pr = k0*p['nt']/kinf             # p['nt'] = total density = M 3-body density
    k2 = (kinf*pr)/(1+pr)
    return k2
_reaction_coefficient_formulae['stand 3-body'] = _f

class Reaction:
    """A class for manipulating a chemical reaction."""

    def __init__(
            self,
            name=None,
            reactants=None,
            products=None,
            formula='constant', # type of reaction, defined in get_rate_coefficient
            coefficients=None,     # used for computing rate coefficient according to formula
            encoding='unicode', # of reaction name or species in products/reactants
    ):

        ## get reactants and products from name or provided lists
        self.encoding = encoding
        if name is None and reactants is not None and products is not None:
            self.reactants = [convert.species(t,encoding,self.encoding) for t in reactants]
            self.products = [convert.species(t,encoding,self.encoding) for t in products]
        elif name is not None and reactants is None and products is None:
            self.reactants,self.products = decode_reaction(name,encoding)
        else:
            raise Exception('Must be name is not None, or reactants/products are not None, but not both.')
        self.reactants,self.products = tuple(self.reactants),tuple(self.products) # make immutable
        self._hash = hash((self.reactants,self.products))
        ## tidy name
        self.name = encode_reaction(self.reactants,self.products,encoding=self.encoding)
        self.formula = formula  # name of formula
        self._formula = _reaction_coefficient_formulae[formula] # function
        self.coefficients = ({} if coefficients is None else coefficients)
        self.rate_coefficient = None     
        self.rate = None     

    def set_encoding(self,encoding):
        if encoding == self.encoding:
            return
        raise NotImplementedError()

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

    def __init__(
            self,
            encoding='unicode',
    ):
        self.reactions = []
        self.species = set()
        self.verbose = False
        self.density = Dataset()
        self.state = Dataset()
        self.encoding = encoding # encoding of species and reactions
        
    def __getitem__(self,key):
        if key in self.state:
            return self.state[key]
        elif key in self.density:
            return self.density[key]
        else:
            raise KeyError

    def calc_rate_coefficients(self):
        """Compute rate coefficients in current state."""
        self.rate_coefficients = [reaction.get_rate_coefficient(self.state)
                                  for reaction in self.reactions]

    def calc_rates(self):
        """"Calculate reaction rates in current state and density."""
        for r in self.reactions:
            k = copy(r.rate_coefficient)
            for s in r.reactants:
                if s not in ['γ','e-']:
                    k *= self.density[s]
            r.rate = k

    # def integrate(
            # self,
            # time,
            # initial_density,
            # state=None,
            # nsave_points=10,
    # ):
        # """Integrate density with time."""
        # ## collect all species names
        # for name in initial_density:
            # self.species.add(name)
        # ## assign each species an index
        # _species_index = {name:i for (i,name) in enumerate(self.species)} 
        # ## time steps
        # time = np.array(time,dtype=float)
        # if time[0]!=0:
            # time = np.concatenate(([0],time))
        # ## set initial conditions
        # density = np.zeros((len(self.species),len(time)),dtype=float)
        # for key,val in initial_density.items():
            # density[_species_index[key],0] = val
        # ## set state
        # if state is not None:
            # self.state.clear()
            # self.state.update(state)
        # ## faster rate calculation function
        # def _get_rates(time,density):
            # rates = np.zeros(len(self.species),float)
            # ## update rate coefficients
            # self.calc_rate_coefficients()
            # ## loop through reactions
            # for ir,r in enumerate(self.reactions):
                # ## compute reaction rate
                # k = copy(r.rate_coefficient)
                # for s in r.reactants:
                    # if s not in ['γ','e-']:
                        # k *= density[_species_index[s]]
                # ## add to product/destruction rates
                # for s in r.reactants:
                    # rates[_species_index[s]] -= k
                # for s in r.products:
                    # rates[_species_index[s]] += k
            # return rates
        # ## initialise integrator
        # r = integrate.ode(_get_rates)
        # r.set_integrator('lsoda', with_jacobian=True)
        # r.set_initial_value(density[:,0])
        # ## run saving data at requested number of times
        # for itime,timei in enumerate(time[1:]):
            # r.integrate(timei)
            # density[:,itime+1] = r.y
        # retval = Dataset(t=time,**{t0:t1 for t0,t1 in zip(self.species,density)})
        # return retval

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

    def append(self,reaction=None,**new_reaction_kwargs):
        """Append a Reaction, or arguments to define one."""
        if reaction is None:
            reaction = Reaction(*args,**kwargs)
        reaction.set_encoding = self.encoding
        self.reactions.append(reaction)
        for species in reaction.reactants + reaction.products:
            self.species.add(species)
        return reaction

    def remove_unnecessary_reactions(self):
        """Remove all reactions containing species that have no
        density."""
         ## species of interest
        valid_species = list(self.density) + ['γ','e⁻']
        for r in copy(self.reactions):
            for species in list(r.reactants)+list(r.products):
                if species not in valid_species:
                    if self.verbose:
                        print(f'Removing reaction {str(r)} because {species!r} is not in the model species')
                    self.reactions.remove(r)
                    break

    def __iter__(self):
        for t in self.reactions: 
            yield(t)

    def __len__(self):
        return len(self.reactions)

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
            reactants = sorted(tools.ensure_iterable(reactants))
        if products is not None:
            products = sorted(tools.ensure_iterable(products))
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
            # retval.append(str(t))
            retval.append(repr(t))
        return('\n'.join(retval))

    def save(self,filename):
        """Save encoded reactions and coefficients to a file."""
        tools.string_to_file(filename,str(self))

    def load(self,filename):
        """Load encoded reactions and coefficients from a file."""
        with open(tools.expand_path(filename),'r') as fid:
            for line in fid:
                self.append(**eval(f'dict({line[:-1]})'),encoding=self.encoding)

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
                retval['reactants'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[8:16].strip()) != '':
                retval['reactants'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[16:24].strip()) != '':
                retval['reactants'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[24:32].strip()) != '':
                retval['products'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[32:40].strip()) != '':
                retval['products'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[40:48].strip()) != '':
                retval['products'].append(convert.species(t,'stand',self.encoding))
            if (t:=line[48:54].strip()) != '':
                retval['products'].append(convert.species(t,'stand',self.encoding))
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
                        formula='stand 3-body',
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

    def load_cantera(
            self,
            gas,
            forward_rates=None,
            reverse_rates=None,
    ):
        """Load reaction network and rates of progress.  You might like to
        provide separate time-dependnent rates of progress instead."""
        ## get rate constants and rates
        if forward_rates is None:
            forward_rates = gas.forward_rates_of_progress
        if reverse_rates is None:
            reverse_rates = gas.reverse_rates_of_progress
        ## add reactions
        for i,name in enumerate(gas.reaction_equations()):
            reactants,products = decode_reaction(name,'cantera')
            ## forward reaction
            r = self.append(reactants=reactants,products=products)
            r.rate = forward_rates[i]
            ## reverse reaction
            r = self.append(reactants=products,products=reactants)
            r.rate = reverse_rates[i]

def integrate_network(reaction_network,initial_density,state,time):
    """Integrate density with time."""
    ## get full list of species
    species = copy(reaction_network.species)
    for name in initial_density:
        species.add(name)
    ## assign each species an index
    _species_index = {name:i for (i,name) in enumerate(species)} 
    ## time steps
    time = np.array(time,dtype=float)
    if time[0]!=0:
        time = np.concatenate(([0],time))
    ## set initial conditions
    density = np.zeros((len(species),len(time)),dtype=float)
    for key,val in initial_density.items():
        density[_species_index[key],0] = val
    current_state = {}
    def _get_current_state(time,density):
        for key,val in state.items():
            if callable(val):
                current_state[key] = val(time)
            else:
                current_state[key] = val
            current_state['nt'] = np.sum(density)
        return current_state
    ## faster rate calculation function
    def _get_rates(time,density):
        rates = np.zeros(len(species),float)
        ## rate coefficients
        _get_current_state(time,density)
        rate_coefficients = np.array(
            [reaction.get_rate_coefficient(current_state)
             for reaction in reaction_network],float)
        ## loop through reactions
        for ireaction,(reaction,rate_coefficient) in enumerate(zip(reaction_network,rate_coefficients)):
            ## compute reaction rate
            k = rate_coefficient
            for s in reaction.reactants:
                if s not in ['γ','e-']:
                    k *= density[_species_index[s]]
            ## add to product/destruction rates
            for s in reaction.reactants:
                rates[_species_index[s]] -= k
            for s in reaction.products:
                rates[_species_index[s]] += k
        return rates
    ## initialise integrator
    r = scipy.integrate.ode(_get_rates)
    r.set_integrator('lsoda', with_jacobian=True)
    r.set_initial_value(density[:,0])
    ## run saving data at requested number of times
    save_state = Dataset()
    save_state.append(**_get_current_state(time=0,density=density[:,0]))
    for itime,timei in enumerate(time[1:]):
        r.integrate(timei)
        density[:,itime] = r.y
        save_state.append(**current_state)
    retval = Mixture(
        t=time,
        **{t0:t1 for t0,t1 in zip(species,density)},
        **save_state)
    return retval
