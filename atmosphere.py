import functools
from copy import copy,deepcopy
import re

from scipy import constants,integrate
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import kinetics
from . import tools
from . import database
from .exceptions import InferException
from . import plotting


class Atmosphere(Dataset):
    """1D model atmosphere"""

    prototypes = {}
    prototypes['description'] = dict( description="",kind=str ,infer={})
    prototypes['notes'] = dict(description="Notes regarding this line" , kind=str ,infer={})
    prototypes['author'] = dict(description="Author of data or printed file" ,kind=str ,infer={})
    prototypes['reference'] = dict(description="Published reference" ,kind=str ,infer={})
    prototypes['date'] = dict(description="Date data collected or printed" ,kind=str ,infer={})
    prototypes['z'] = dict(description="Height above surface (cm)" ,kind=float ,infer={})
    prototypes['z(km)'] = dict(description="Height above surface (km)" ,kind=float ,infer={'z':lambda z: z*1e-5,})
    prototypes['T'] = dict(description="Temperature (K)" ,kind=float ,infer={})
    prototypes['nt'] = dict(description="Total number density (cm-3)" ,kind=float ,infer={})
    prototypes['p'] = dict(description="Pressure (bar)" ,kind=float ,infer={})
    prototypes['Kzz'] = dict(description="Turbulent diffusion constant (cm2.s-1)" ,kind=float ,infer={})
    prototypes['Hz'] = dict(description="Local scale height (cm1)" ,kind=float ,infer={})

    def __init__(self,*args,**kwargs):
        Dataset.__init__(self,*args,**kwargs)
        self.permit_reference_breaking = False
        self.reaction_network = kinetics.ReactionNetwork()
        self.reaction_network.state = self
    
    def load_ARGO_depth(self,filename):
        """Load an ARGO depth.dat file."""
        data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        nt = data['NH(cm-3)']
        ## load physical parameters
        for key_from,key_to in (
                ('p(bar)','p'),
                ('T(K)','T'),
                ('NH(cm-3)','nt'),
                ('Kzz(cm2s-1)','Kzz'),
                ('Hz(cm)','z'),
                ('zeta(s-1)','zeta(s-1)'),
                ('h','h'),
                ('f+','f+')
        ):
            if key_to in self:
                assert np.all(self[key_to] == data.pop(key_from))
            else:
                self[key_to] = data.pop(key_from)
        ## load volume density
        for key in data:
            self.set_abundance(
                kinetics.translate_species(key,'ARGO','standard'),
                data[key]*nt)

    def load_ARGO_reaction_rate_coefficients(self,filename):
        """Load the rate coefficients from an ARGO Reactions/Kup.dat or
        Kdown.dat."""
        ## load data -- check consistency with self
        data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        assert np.all(data.pop('p[bar]') == self['p'])
        assert np.all(data.pop('h[cm]') == self['z'])
        reaction_numbers = np.array([t.coefficients['reaction_number'] for t in self.reaction_network.reactions])
        assert len(np.unique(reaction_numbers)) == len(reaction_numbers)
        rates_reaction_numbers = np.array([int(key[1:]) for key in data])
        if reaction_numbers.max() != rates_reaction_numbers.max():
            print(f'warning: reaction numbers do not align')
            # raise Exception
        ## load rate coefficients into self
        for r in self.reaction_network.reactions:
            r.rate_coefficient = data['R'+str(r.coefficients['reaction_number'])]

    def set_abundance(self,species,abundance):
        self['n_'+species] = abundance
        self.reaction_network.set_abundance(species,self['n_'+species])

    def load_ARGO_lifetime(self,filename):
        """Load an ARGO lifetime.dat file."""
        data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        ## load physical parameters
        for key_from,key_to in (
                ('p(bar)','p'), ('T(K)','T'), ('NH(cm-3)','nt'),
                ('Kzz(cm2s-1)','Kzz'), ('Hz(cm)','z'),
                ('zeta(s-1)','zeta(s-1)'), ('h','h'), ('f+','f+')
        ):
            if key_to in self:
                assert np.all(self[key_to] == data.pop(key_from))
            else:
                self[key_to] = data.pop(key_from)
        ## load abundances
        for key in data:
            standard_key = kinetics.translate_species(key,'ARGO','standard')
            self['τ_'+standard_key] = data[key]
    
    def plot_vertical(self,ykey,*xkeys,ax=None):
        if ax is None:
            ax = plotting.gca()
        for xkey in xkeys:
            ax.plot(self[xkey],self[ykey],label=xkey)
        ax.set_xscale('log')
        ax.set_ylim(self[ykey].min(),self[ykey].max())
        ax.set_ylabel(ykey)
        ax.set_xlabel('abundance?')
        plotting.legend(ax=ax)

    def calc_rates(self):
        self.reaction_network.calc_rates()


    def get_rates(
            self,
            normalise_to_species=None,
            sort_method='max anywhere', 
            nsort=3,          
            **kwargs_get_matching_reactions
    ):
        """Return larges-to-smallest reaction rates matching
        kwargs_get_matching_reactions.  Optionally divide by the
        density of normalise_to_species."""
        reaction_names = []
        rates = []
        for reaction in self.reaction_network.get_matching_reactions(**kwargs_get_matching_reactions):
            rate = reaction.rate
            if normalise_to_species is not None:
                rate /= self.reaction_network.species[normalise_to_species]
            reaction_names.append(reaction.name)
            rates.append(rate)
        ## sort
        if sort_method == 'maximum':
            ## list all reactions by their reverse maximum rate 
            i = np.argsort([-np.max(t) for t in rates])
        elif sort_method == 'max anywhere':
            ## return all reactions that are in the top 5 ranking anywhere in their domain
            i = np.argsort(-np.row_stack(rates),0)
            i = np.unique(i[:nsort,:].flatten())
        else:
            raise Exception(f'Unknown {sort_method=}')
        return {reaction_names[ii]:rates[ii] for ii in i}

    def plot_rates(
            self,
            ykey=None,
            ax=None,
            nplot=None,
            plot_total=True,
            plot_kwargs=None,
            normalise_to_species=None,
            **kwargs_get_rates
    ):
        """Plot rates matching kwargs_get_matching_reactions."""
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        if plot_kwargs is None:
            plot_kwargs = {}
        y = self[ykey]
        rates = self.get_rates(normalise_to_species=normalise_to_species,**kwargs_get_rates)
        if plot_total:
            total = np.sum(list(rates.values()),0)
            integrated = integrate.trapz(total,self['z'])
            ax.plot(total,y,color='black',alpha=0.3,linewidth=6,
                    label=f'{integrated:0.2e} total',
                    **plot_kwargs)
        for i,(name,rate) in enumerate(rates.items()):
            if nplot is not None and i > nplot:
                break
            integrated = integrate.trapz(rate,self['z'])
            ax.plot(rate,y, label=f'{integrated:0.2e} {name:40}', **plot_kwargs)
        ax.set_ylabel(ykey)
        if normalise_to_species is not None:
            ax.set_xlabel(f'Rate normalised to the density of {normalise_to_species} (s$^{{-1}}$)')
        else:
            ax.set_xlabel('Rate (cm$^{-3}$ s$^{-1}$)')
        ax.set_xscale('log')
        ax.set_ylim(y.min(),y.max())
        plotting.legend()
        return ax

    def plot_production_destruction(self,species,ykey='z',ax=None,normalise=False,nsort=3):
        """Plot rates matching kwargs_get_matching_reactions."""
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        y = self[ykey]
        ## production rates
        self.plot_rates(
            ykey=ykey, ax=ax, plot_total= True,
            products=species,
            plot_kwargs={'linestyle':'-',},
            normalise_to_species=(species if normalise else None),
            nsort=nsort,
        )
        ## destruction
        self.plot_rates(
            ykey=ykey, ax=ax, plot_total= True,
            reactants=species,
            plot_kwargs={'linestyle':':',},
            normalise_to_species=(species if normalise else None),
            nsort=nsort
        )
        plotting.legend(show_style=True)
        ax.set_title(f'Production and destruction rates of {species}')
        return ax
