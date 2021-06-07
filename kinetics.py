import re
import warnings
from copy import copy
from functools import lru_cache

import numpy as np
from bidict import bidict
import periodictable
from scipy import integrate,constants

from .dataset import Dataset
from . import tools
from .tools import cache
from . import database
from . import plotting
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

## cantera
def _f(name):
    """From cantera species name to standard. No translation actually"""
    return name
_species_name_translation_functions[('cantera','standard')] = _f

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

@cache
def get_chemical_name(name):
    species = get_species(name)
    return species['chemical_name']
    
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
        
        self.decode_name(name)
        self.encode_name(self._isotopes,self._charge)

    # def _get_charge(self):
        # if self._charge is None:
            # if r:=re.match('^(.*[^+-])([+]+|-+)$',self.species): # ddd+ / ddd++
                # self._charge = r.group(2).count('+') - r.group(2).count('-')
            # else:
                # self._charge = 0
        # return self._charge

    def decode_name(self,name):
        """Turn standard name string into ordered isotope list and charge."""
        isotopes = []                   # (element,mass_number,multiplicity)
        ## e.g., CO2
        if r:=re.match(r'^((?:[A-Z][a-z]?[0-9]*)+)([-+]*)$',name):
            name_no_charge = r.group(1)
            charge = r.group(2).count('+') - r.group(2).count('-')
            for part in re.split(r'([A-Z][a-z]?[0-9]*)',name_no_charge):
                if part=='':
                    continue
                elif r:= re.match(r'^([A-Z][a-z]?)([0-9]*)',part):
                    isotopes.append((
                        r.group(1),
                        database.get_isotopes(r.group(1))[0][0],
                        int(r.group(2) if r.group(2) != '' else 1)))
                else:
                    raise Exception(f'Could not decode element name {repr(part)} in  {repr(name)}')
        ## e.g., [12C][16O]2
        elif r:=re.match(r'^((?:\[[0-9]+[A-Z][a-z]?\][0-9]*)+)([-+]*)$',name):
            name_no_charge = r.group(1)
            charge = r.group(2).count('+') - r.group(2).count('-')
            for part in re.split(r'(\[[0-9]+[A-Z][a-z]?\][0-9]*)',name_no_charge):
                if part=='':
                    continue
                elif r:= re.match(r'\[([0-9]+)([A-Z][a-z]?)\]([0-9]*)',part):
                    isotopes.append((
                        r.group(2),
                        int(r.group(1)),
                        int(r.group(3) if r.group(3) != '' else 1)))
                else:
                    raise Exception(f'Could not decode element name {repr(part)} in  {repr(name)}')
        ## e.g., ¹²C¹⁶O₂²⁺
        elif r:=re.match(r'^((?:[⁰¹²³⁴⁵⁶⁷⁸⁹]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻]?)$',name):
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
            for part in re.split(r'([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]*[A-Z][a-z]?[₀₁₂₃₄₅₆₇₈₉]*)',name_no_charge):
                if part=='':
                    continue
                elif r:= re.match(r'([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]*)([A-Z][a-z]?)([₀₁₂₃₄₅₆₇₈₉]*)',part):
                    isotopes.append((
                        r.group(2),
                        (database.get_isotopes(r.group(2))[0][0]
                                 if r.group(1) == '' else int(tools.regularise_unicode(r.group(1)))),
                        int(tools.regularise_unicode(r.group(3)) if r.group(3) != '' else 1)))
                else:
                    raise Exception(f'Could not decode element name {repr(part)} in  {repr(name)}')
        else:
            raise Exception(f'Could not decode species named: {repr(name)}')
        self._isotopes = tuple(isotopes)
        self._charge = charge

    def encode_name(self,isotopes,charge):
        """Turn ordered isotope list and charge into a name string."""
        retval = []
        for element,mass_number,multiplicity in isotopes:
            retval.append(tools.superscript_numerals(str(int(mass_number))))
            retval.append(element)
            if multiplicity > 1:
                retval.append(tools.subscript_numerals(str(int(multiplicity))))
        if charge == 1:
            retval.append('⁺')
        elif charge > 1:
            retval.append(tools.superscript_numerals(str(int(charge)))+'⁺')
        elif charge == -1:
            retval.append('⁻')
        elif charge < -1:
            retval.append(tools.superscript_numerals(str(int(-charge))+'⁻'))
        retval = ''.join(retval)
        self._name = retval

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
        if self.species in ('e-','photon'):
            raise NotImplementedError()
        self._nelectrons = 0
        ## add electrons attributable to each nucleus
        for element in self['elements']:
            self._nelectrons += getattr(periodictable,element).number
        ## account for ionisation
        self._nelectrons -= self['charge']
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

    def __getitem__(self,key):
        """Access these properties by index rather than attributes in order
        for simple caching.  -- move other get_ methods to here someday"""
        if key == 'name':
            return self._name
        elif key == 'charge':
            return self._charge
        elif key == 'elements':
            if not hasattr(self,'_elements'):
                t = []
                for element,mass_number,multiplicity in self._isotopes:
                    for i in range(multiplicity):
                        t.append(element)
                self._elements = tuple(sorted(t))
            return self._elements
        elif key == 'isotopes':
            return self._isotopes
        elif key == 'nelectrons':
            if not hasattr(self,'_nelectrons'):
                self._get_nelectrons()
            return self._nelectrons
        elif key == 'mass':
            if not hasattr(self,'_mass'):
                self._mass =  sum([database.get_atomic_mass(element,mass_number)*multiplicity
                                   for element,mass_number,multiplicity in self['isotopes']])
            return self._mass
        elif key == 'reduced_mass':
            if not hasattr(self,'_reduced_mass'):
                if len(self['isotopes']) != 2:
                    raise Exception("Can only compute reduced mass for diatomic species.")
                m1 = database.get_atomic_mass(*self['isotopes'][0])
                m2 = database.get_atomic_mass(*self['isotopes'][1])
                self._reduced_mass = m1*m2/(m1+m2)
            return self._reduced_mass
        elif key == 'chemical_name':
            if not hasattr(self,'_chemical_name'):
                t = []
                for element,mass_number,multiplicity in self._isotopes:
                    t.append(element)
                    t.append(tools.subscript_numerals(multiplicity))
                self._chemical_name = ''.join(t)
            return self._chemical_name
        # elif key == 'isotopologue':
            # return ''.join([f'[{t[1]}{t[0]}]' for t in self['isotopes']])
        elif key == 'point_group':
            if len(self.isotopes) != 2:
                raise Exception("Can only compute reduced mass for diatomic species.")
            ## Homonumclear or heteronuclear diatomic
            if self.isotopes[0] == self.isotopes[1]:
                return 'D∞h'
            else:
                return 'C∞v'
        else:
                raise Exception(f"Unknown species property: {key}")
    
    name = property(lambda self: self['name'])
    elements = property(lambda self: self['elements'])
    isotopes = property(lambda self: self['isotopes'])
    chemical_name = property(lambda self: self['chemical_name'])
    isotopologue = property(lambda self: self['isotopologue'])
    species = property(lambda self: self.name)
    mass = property(lambda self: self['mass'])
    reduced_mass = property(lambda self: self['reduced_mass'])
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



def decode_reaction(reaction,encoding='standard'):
    """Decode a reaction into a list of reactants and products, and other
    information. Encoding is for species names. Encoding of reaction
    string formatting not implemented."""
    if encoding == 'standard':
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
                species = decode_species(species,encoding)
                for i in range(multiplicity):
                    r_or_p_list.append(species)
            r_or_p_list.sort()
        
    else:
        raise Exception(f"Unknown reaction encoding: {repr(encoding)}")
    
    return reactants,products

def encode_reaction(reactants,products,encoding='standard'):
    """Only just started"""
    if encoding!='standard':
        raise ImplementationError()
    return ' + '.join(reactants)+' → '+' + '.join(products)

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
    'NIST'                   :lambda c,p: c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/(constants.R*p['T'])),
    'NIST_3rd_body_hack'     :lambda c,p: 1e19*c['A']*(p['T']/298.)**c['n']*np.exp(-c['Ea']/(constants.R*p['T'])),
    'photoreaction'          :lambda c,p: scipy.integrate.trapz(c['σ'](p['T'])*p['I'],c['λ']),
    'kooij'                  :lambda c,p: c['α']*(p['T']/300.)**c['β']*np.exp(-c['γ']/p['T']), # α[cm-3], T[K], β[], γ[K]
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
        self.reactions = []
        self.species = set()
        self.verbose = False
        self.density = Dataset()
        self.state = Dataset()

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

    def append(self,*args,**kwargs):
        """Append a Reaction, or arguments to define one."""
        if len(args)==1 and len(kwargs)==0 and  isinstance(args[0],Reaction):
            r = args[0]
        else:
            r = Reaction(*args,**kwargs)
        self.reactions.append(r)
        for species in r.reactants + r.products:
            self.species.add(species)
        return r

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
    r = integrate.ode(_get_rates)
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
