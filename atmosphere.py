import functools
from copy import copy,deepcopy
import re
import time

from scipy import constants,integrate
import numpy as np

from .dataset import Dataset
from .conversions import convert
from . import kinetics
from . import tools
from . import database
from .exceptions import InferException
from . import plotting



class OneDimensionalAtmosphere(Dataset):

    prototypes = {}
    prototypes['description'] = dict( description="",kind=str ,infer={})
    prototypes['notes'] = dict(description="Notes regarding this line" , kind=str ,infer={})
    prototypes['author'] = dict(description="Author of data or printed file" ,kind=str ,infer={})
    prototypes['reference'] = dict(description="Published reference" ,kind=str ,infer={})
    prototypes['date'] = dict(description="Date data collected or printed" ,kind=str ,infer={})
    prototypes['z'] = dict(description="Height above surface (cm)" ,kind=float ,infer={})
    prototypes['z(km)'] = dict(description="Height above surface (km)" ,kind=float ,infer={'z':lambda z: z*1e-5,})
    prototypes['Ttr'] = dict(description="Translational temperature (K)" ,kind=float ,infer={})
    prototypes['nt'] = dict(description="Total number density (cm-3)" ,kind=float ,infer={})
    prototypes['p'] = dict(description="Pressure (bar)" ,kind=float ,infer={})
    prototypes['Kzz'] = dict(description="Turbulent diffusion constant (cm2.s-1)" ,kind=float ,infer={})
    prototypes['Hz'] = dict(description="Local scale height (cm1)" ,kind=float ,infer={})
    prototypes['zeta(s-1)'] = dict(description="not implemented" ,kind=float ,infer={})
    prototypes['h'] = dict(description="not implemented" ,kind=float ,infer={})
    prototypes['f+'] = dict(description="not implemented" ,kind=float ,infer={})

    def __init__(self):
        Dataset.__init__(self)
        self.permit_nonprototyped_data = False


class AtmosphericChemistry():
    """1D model atmosphere"""

    def __init__(self,*args,**kwargs):
        self.reaction_network = kinetics.ReactionNetwork()
        self.density = self.reaction_network.density = Dataset()
        self.state = self.reaction_network.state = OneDimensionalAtmosphere()
        self.verbose = self.reaction_network.verbose = False
    
    def load_ARGO(
            self,
            model_output_directory,
            reaction_network_filename=None,
            rate_coefficient_filename=None,
            load_rates_file=False,
           ):

        ## load depth.dat physical parameters and species volume density
        data = tools.file_to_dict(
            f'{model_output_directory}/depth.dat',
            skiprows=2,labels_commented=False)
        for key_from,key_to in (
                ('p(bar)','p'),
                ('T(K)','Ttr'),
                ('NH(cm-3)','nt'),
                ('Kzz(cm2s-1)','Kzz'),
                ('Hz(cm)','z'),
                ('zeta(s-1)','zeta(s-1)'),
                ('h','h'),
                ('f+','f+')
        ):
            self.state[key_to] = data.pop(key_from)
        for key in data:
            self.set_density(
                kinetics.translate_species(key,'STAND','standard'),
                data[key]*self.state['nt'])

        if reaction_network_filename is not None:
            ## load STAND reaction network
            self.reaction_network.load_STAND(reaction_network_filename)
            self.reaction_network.remove_unnecessary_reactions()
    
        if rate_coefficient_filename is not None:
            ## Load the rate coefficients from an ARGO
            ## Reactions/Kup.dat or
            data = tools.file_to_dict(rate_coefficient_filename,skiprows=2,labels_commented=False)
            if 'down' in rate_coefficient_filename:
                ## reverse z grid 
                for key in data:
                    data[key] = data[key][::-1]
            assert np.all(data.pop('p[bar]') == self['p']),'Mismatch between model pressure grid and reaction rate coefficient grid.'
            assert np.all(data.pop('h[cm]') == self['z']),'Mismatch between model z grid and reaction rate coefficient grid.'
            for r in self.reaction_network.reactions:
                r.rate_coefficient = data['R'+str(r.coefficients['reaction_number'])]

        if load_rates_file:
            ## load rates from the last time step in verif files
            ## loop over height files
            data = Dataset()
            for filename in tools.glob(f'{model_output_directory}/Reactions/down*verif.dat',):
                ## get data from filename
                r = re.match(r'.*(?:up|down)-P=(.*)_H=(.*)_verif.dat',filename,)
                assert r
                p,z = float(r.group(1)),float(r.group(2))*1e5
                ## load all data in file
                lines = tools.file_to_lines(filename)
                ## collect all rates in last timestep
                i = tools.find(['T(SEC)' in l for l in lines])
                t0,t1 = [],[]
                for line in lines[i[-1]+1:]:
                    tokens = line.split()
                    if len(tokens) == 2:
                        t0.append(int(tokens[0]))
                        t1.append(abs(float(tokens[1])))
                r = np.unique(tools.make_recarray(t0=t0,t1=t1)) # get rid of dupiliate in a recarray
                data.extend(p=p,z=z,reaction_number=r['t0'],rate=r['t1'])
            ## put rates into reaction_network
            assert np.all((np.sort(data.unique('z')))==np.sort(self['z'])) # check z coordinate matches self
            assert np.max(np.abs(np.sort(data.unique('p'))-np.sort(self['p']))/np.sort(self['p'])) # check p coordinate matches self
            data.sort('z')
            for r in self.reaction_network.reactions:
                r.rate = np.full(len(self),np.nan)
                i = tools.find(data.match(reaction_number=r.coefficients['reaction_number']))
                if len(i)==0: continue
                # j = findin_numeric(data['z'][i],self['z'])`
                j = tools.findin(data['z'][i],self['z'])
                r.rate[j] = data['rate'][i]
                
    # def load_ARGO_lifetime(self,filename):
        # """Load an ARGO lifetime.dat file."""
        # data = tools.file_to_dict(filename,skiprows=2,labels_commented=False)
        # ## load physical parameters
        # for key_from,key_to in (
                # ('p(bar)','p'), ('T(K)','T'), ('NH(cm-3)','nt'),
                # ('Kzz(cm2s-1)','Kzz'), ('Hz(cm)','z'),
                # ('zeta(s-1)','zeta(s-1)'), ('h','h'), ('f+','f+')
        # ):
            # if key_to in self:
                # assert np.all(self[key_to] == data.pop(key_from))
            # else:
                # self[key_to] = data.pop(key_from)
        # ## load densitys
        # for key in data:
            # standard_key = kinetics.translate_species(key,'ARGO','standard')
            # self['τ_'+standard_key] = data[key]

    def __getitem__(self,key):
        if self.state.is_known(key):
            return self.state[key]
        elif key in self.density:
            return self.density[key]
        else:
            raise Exception(f'Unknown {key=}')

    def __len__(self):
        return len(self.state)

    def set_density(self,species,density):
        self.density[species] = density

    def plot_vertical(self,ykey,*xkeys,ax=None):
        if ax is None:
            ax = plotting.gca()
        for xkey in xkeys:
            ax.plot(self[xkey],self[ykey],label=xkey)
        ax.set_xscale('log')
        ax.set_ylim(self[ykey].min(),self[ykey].max())
        ax.set_ylabel(ykey)
        ax.set_xlabel('density?')
        plotting.legend(ax=ax)

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
                rate /= self.density[normalise_to_species]
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
            ax.plot(total,y,color='black',alpha=0.3,linewidth=6, label=f'{integrated:0.2e} total', **plot_kwargs)
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

    def plot_production_destruction(
            self,
            species,
            ykey='z',
            ax=None,
            normalise=False,
            nsort=3,
    ):
        """Plot rates matching kwargs_get_matching_reactions."""
        if ax is None:
            ax = plotting.plt.gca()
            ax.cla()
        y = self[ykey]
        ## production rates
        self.plot_rates(
            ykey=ykey, ax=ax, plot_total= True,
            products=species,
            plot_kwargs={'linestyle':'-'},
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
