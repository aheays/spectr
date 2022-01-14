import functools
from copy import copy,deepcopy
import re
import time

from scipy import constants,integrate
import numpy as np

from .dataset import Dataset
from . import convert
from . import kinetics
from . import tools
from . import database
from .exceptions import InferException
from . import plotting
from .plotting import *


class OneDimensionalAtmosphere(Dataset):

    default_prototypes = {}
    default_prototypes['z'] = dict(description="Height above surface",units='cm',fmt='0.5e',kind='f' ,infer=[])
    default_prototypes['z(km)'] = dict(description="Height above surface",units='km',kind='f' ,infer=[('z',lambda self,z: z*1e-5,),])
    default_prototypes['Ttr'] = dict(description="Translational temperature",units='K',kind='f' ,infer=[('T',lambda self,T:T),])
    default_prototypes['T'] = dict(description="Temperature" ,units='K',kind='f' ,fmt='0.6g',infer=[])
    default_prototypes['nt'] = dict(description="Total number density",units='cm-3',kind='f' ,fmt='0.6e',infer=[])
    default_prototypes['Nt'] = dict(description="Total number column density (cm-2)" ,kind='f' ,infer=[(('z','nt'),lambda self,z,nt: tools.cumtrapz(nt,z,reverse=True),),])
    default_prototypes['p'] = dict(description="Pressure",units='bar',kind='f' ,fmt='0.6g',infer=[])
    default_prototypes['Kzz'] = dict(description="Turbulent diffusion coefficient",units='cm2.s-1' ,kind='f' ,infer=[])
    default_prototypes['Hz'] = dict(description="Local scale height (cm1)" ,kind='f' ,infer=[])
    default_prototypes['zeta(s-1)'] = dict(description="not implemented" ,kind='f' ,infer=[])
    default_prototypes['h'] = dict(description="not implemented" ,kind='f' ,infer=[])
    default_prototypes['f+'] = dict(description="not implemented" ,kind='f' ,infer=[])
    def _f(self,z):
        """Get height intervals. CURRENTLY NAIVELY COMPUTED!!!!"""
        dz = np.full(z.shape,0.)
        dz[1:-1] = (z[2:]-z[1:-1])/2+(z[1:-1]-z[:-2])/2
        dz[0] = (z[1]-z[0])/2
        dz[-1] = (z[-1]-z[-2])/2
        return dz
    default_prototypes['dz'] = dict(description="Depth of grid cell (cm)." ,kind='f' ,infer=[('z',_f),])
    default_permit_nonprototyped_data = False

class AtmosphericChemistry():
    """1D model atmosphere"""

    def __init__(self,*args,**kwargs):
        self.reaction_network = kinetics.ReactionNetwork()
        self.density = self.reaction_network.density = Dataset()
        self.state = self.reaction_network.state = OneDimensionalAtmosphere()
        self.verbose = self.reaction_network.verbose = False

    def calc_rates(self):
        return self.reaction_network.calc_rates()

    def calc_rate_coefficients(self):
        return self.reaction_network.calc_rate_coefficients()

    def load_argo(
            self,
            model_directory,
            load_reaction_network= True, # filename to load filename, True to guess filename
            load_rate_coefficients= True,
            load_rates=False,
            iteration=None,     # # to load an old-depth-#.dat file -- None for the final result
           ):

        ## load depth.dat physical parameters and species volume density
        if iteration is None:
            depth_filename = f'{model_directory}/out/depth.dat'
        else:
            depth_filename = f'{model_directory}/out/old-depth-{iteration:d}.dat'
        data = tools.file_to_dict(depth_filename,skiprows=2,labels_commented=False)
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
            self.state[key_to] = data.pop(key_from)
        for key in data:
            self.set_density(
                # kinetics.translate_species(key,'stand','ascii'),
                convert.species(key,'stand','ascii'),
                data[key]*self.state['nt'])

        if load_reaction_network is not False:
            if load_reaction_network is True:
                if self.verbose:
                    print('Searching for reaction network file: Stand*')
                load_reaction_network = tools.glob_unique(f'{model_directory}/Stand*')
                if self.verbose:
                    print(f'found: {load_reaction_network}')
            ## load STAND reaction network
            self.reaction_network.load_stand(load_reaction_network)
            self.reaction_network.remove_unnecessary_reactions()
    
        if load_rate_coefficients:
            ## Load the rate coefficients from an ARGO
            ## Reactions/Kup.dat or
            rate_coefficient_filename = f'{model_directory}/out/Reactions/Kup.dat'
            data = tools.file_to_dict(rate_coefficient_filename, skiprows=2,labels_commented=False)
            if 'down' in rate_coefficient_filename:
                ## reverse z grid 
                for key in data:
                    data[key] = data[key][::-1]
            assert np.all(data.pop('p[bar]') == self['p']),'Mismatch between model pressure grid and reaction rate coefficient grid.'
            assert np.all(data.pop('h[cm]') == self['z']),'Mismatch between model z grid and reaction rate coefficient grid.'
            for r in self.reaction_network.reactions:
                r.rate_coefficient = data['R'+str(r.coefficients['reaction_number'])]

        if load_rates:
            ## load rates from the last time step in verif files
            ## loop over height files
            data = Dataset()
            for filename in tools.glob(f'{model_directory}/out/Reactions/down*verif.dat',):
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
            assert len(np.sort(data.unique('z')))==len(np.sort(self['z'])),'depth.dat and verif.dat height scale lengths differ'
            data.sort('z')
            for r in self.reaction_network.reactions:
                r.rate = np.full(len(self),np.nan)
                i = tools.find(data.match(reaction_number=r.coefficients['reaction_number']))
                if len(i)==0: continue
                # j = findin_numeric(data['z'][i],self['z'])`
                j = tools.find_nearest(data['z'][i],self['z'])
                r.rate[j] = data['rate'][i]
                
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
        elif r:=re.match(r'N\((.+)\)',key):
            ## cumulative column density of species
            return self.get_column_density(r.group(1))
        else:
            raise Exception(f'Unknown {key=}')

    def get_species(self):
        """Get a list of all chemical species."""
        return list(self.density.keys())

    def get_mixing_ratio(self,species):
        return self.density[species]/self.state['nt']

    def get_column_density(self,species):
        return tools.cumtrapz(self.density[species], self['z'],reverse=True)

    def __len__(self):
        return len(self.state)

    def set_density(self,species,density):
        key = species
        self.density.set_prototype(
            key,'f',description=f'Number density of {species}',
            units='cm-3',fmt='10.5e',)
        self.density[key] = density

    def get_density(self,species):
        return self.density[species]

    def get_surface_density(self,*speciess):
        """Return dictionary of surface density ordered from most abundant
        species. THIS IS READ FROM THE LOWEST GRID STEP FO THE OUTPUT
        FILE, AND IS THEN SLIGHTLY DIFFERENT FROM THE INPUT DATA IN
        COND_INITIAL."""
        if len(speciess) == 0:
            speciess = self.density.keys()
        surface_density = np.array([self.get_density(species)[0] for species in speciess])
        retval = {speciess[i]:surface_density[i] for i in np.argsort(-surface_density)}
        return retval

    def get_surface_mixing_ratio(self,*speciess):
        """Return dictionary of mixing ratios ordered from most abundant
        species. THIS IS READ FROM THE LOWEST GRID STEP FO THE OUTPUT
        FILE, AND IS THEN SLIGHTLY DIFFERENT FROM THE INPUT DATA IN
        COND_INITIAL."""
        if len(speciess) == 0:
            speciess = self.density.keys()
        surface_density = np.array([self.get_mixing_ratio(species)[0] for species in speciess])
        retval = {speciess[i]:surface_density[i] for i in np.argsort(-surface_density)}
        return retval

    def plot_vertical(
            self,
            ykey,
            *xkeys,
            ax=None,
            plot_legend=True,
            **plot_kwargs,
    ):
        """Plot profiles of xkeys. ykey='z' might be a good choice."""
        if ax is None:
            ax = plt.gca()
        for ixkey,xkey in enumerate(xkeys):
            kw = {'color':newcolor(ixkey),}
            kw.update(plot_kwargs)
            ax.plot(self[xkey],self[ykey],label=xkey,**kw)
        if 'T' in xkeys:
            ax.set_xscale('linear')
        else:
            ax.set_xscale('log')
        ax.set_ylim(self[ykey].min(),self[ykey].max())
        ax.set_ylabel(ykey)
        if plot_legend:
            legend(ax=ax,allow_multiple_axes=True)

    def plot_density(self,xkeys=5,ykey='z(km)',ax=None):
        """Plot density of species. If xkeys is an integer then plot that many
        most abundant anywhere species. Or else xkeys must be a list
        of species names."""
        if isinstance(xkeys,int):
            ## get most abundance species anywhere
            all_keys = np.array(self.density.keys())
            xkeys = []
            for i in range(len(self.density)):
                j = np.argsort([-self.density[t][i] for t in self.density])
                xkeys.extend(all_keys[j[:5]])
            xkeys = tools.unique(xkeys)
        if ax is None:
            ax = gca()
        ## plot total density
        ax.plot(self['nt'],self[ykey],label='nt',color='black',alpha=0.3,linewidth=6)
        ## plot individual species
        for xkey in xkeys:
            ax.plot(self.density[xkey],self[ykey],label=xkey)
        ax.set_xscale('log')
        ax.set_ylim(self[ykey].min(),self[ykey].max())
        ax.set_ylabel(ykey)
        ax.set_xlabel('Density (cm-3)')
        legend(ax=ax)
        
    def plot(self):
        """A default plot of htis atmosphere."""
        fig = gcf()
        fig.clf()
        subplot()
        self.plot_vertical('z','p')
        subplot()
        self.plot_vertical('z','T')
        subplot()
        self.plot_density()
        
    def get_rates(
            self,
            sort_method='max anywhere', # maximum at some altitude
            nsort=3,            # return this many rates 
            **kwargs_get_reactions 
    ):
        """Return larges-to-smallest reaction rates matching
        kwargs_get_reactions. """
        reaction_names = []
        rates = []
        for reaction in self.reaction_network.get_reactions(**kwargs_get_reactions):
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

    def plot_rates(
            self,
            ykey='z',
            ax=None,
            plot_total=True,    # plot sum of all rates
            plot_kwargs=None,
            normalise_to_species=None, # normalise rates divided by the density of this species
            **kwargs_get_rates         # for selecting rates to plot
    ):
        """Plot rates matching kwargs_get_rates."""
        if ax is None:
            ax = plt.gca()
            ax.cla()
        if plot_kwargs is None:
            plot_kwargs = {}
        y = self[ykey]
        rates = self.get_rates(**kwargs_get_rates)
        ## normalise perhaps
        if normalise_to_species:
            for key in rates:
                rates[key] /= self.density[normalise_to_species]
            weight = self.density[normalise_to_species]*self['dz']
            summary_value = {key:np.sum(val*weight)/np.sum(weight) for key,val in rates.items()} # mean value weighted by normalise_to_species density 
            xlabel = f'Rate normalised to the density of {normalise_to_species} (s$^{{-1}}$)'
        else:
            summary_value = {key:tools.integrate(val,self['z']) for key,val in rates.items()} # integrated value
            xlabel = 'Rate (cm$^{-3}$ s$^{-1}$)'
        ## sort species to plot by their descending summary value
        names = list(rates)
        j = np.argsort(list(summary_value.values()))[::-1]
        names = [names[i] for i in j]
        ## plot
        for i,name in enumerate(names):
            t_plot_kwargs = copy(plot_kwargs)
            t_plot_kwargs['label'] = f'{summary_value[name]:0.2e} {name}'
            if name == 'total':
                if not plot_total:
                    continue
                t_plot_kwargs.update(color='black',alpha=0.3,linewidth=6)
            ax.plot(rates[name],y,**t_plot_kwargs)
        ax.set_ylabel(ykey)
        ax.set_xlabel(xlabel)
        ax.set_xscale('log')
        ax.set_ylim(y.min(),y.max())
        legend()
        return ax

    def summarise_species(self,species,doprint=True):
        """MAY NOT COUNT MULTIPLE PRODUCTS OF THE SAME SPECIES CORRECTLY"""
        column_mixing_ratio = tools.integrate(self.get_density(species),self['z']) / tools.integrate(self['nt'],self['z'])
        ## production summary
        production_reactions = self.reaction_network.get_reactions(with_products=(species,)) 
        production_rate = np.sum([t.rate for t in production_reactions],0)
        production_column_rate = tools.integrate(production_rate,self['z'])
        ## destruction summary
        destruction_reactions = self.reaction_network.get_reactions(with_reactants=(species,)) 
        destruction_rate = np.sum([t.rate for t in destruction_reactions],0)
        destruction_column_rate = tools.integrate(destruction_rate,self['z'])
        destruction_mean_loss_rate = destruction_column_rate/tools.integrate(self.get_density(species),self['z'])
        destruction_mean_lifetime = 1/destruction_mean_loss_rate
        lines = [
            f'species                                 : {species:>10s}',
            f'column mixing ratio             (number): {column_mixing_ratio:10.3e}',
            f'total column production rate  (s-1.cm-2): {production_column_rate:10.3e}',
            f'total column destruction rate (s-1.cm-2): {destruction_column_rate:10.3e}',
            f'mean destruction rate              (s-1): {destruction_mean_loss_rate:10.3e}',
            f'mean destruction lifetime         (days): {convert.units(destruction_mean_lifetime,"s","day"):10.3e}',
            ]
        print('\n'.join(lines))

    def plot_production_destruction(
            self,
            species,
            ykey='z',
            ax=None,
            normalise=False,    # divide rates by species abundance
            nsort=3,            # include this many ranked rates at each altitude
    ):
        """Plot destruction and production rates of on especies."""
        if ax is None:
            ax = plt.gca()
            ax.cla()
        y = self[ykey]
        ## production rates
        self.plot_rates(
            ykey=ykey, ax=ax, plot_total= True,
            with_products=species,
            plot_kwargs={'linestyle':'-'},
            normalise_to_species=(species if normalise else None),
            nsort=nsort,
        )
        ## destruction
        self.plot_rates(
            ykey=ykey, ax=ax, plot_total= True,
            with_reactants=species,
            plot_kwargs={'linestyle':':',},
            normalise_to_species=(species if normalise else None),
            nsort=nsort
        )
        legend(show_style=True,title=f'Production and destruction rates of {species}')
        # ax.set_title(f'Production and destruction rates of {species}')
        return ax

        
