from copy import copy
from pprint import pprint
from functools import lru_cache
import itertools

import numpy as np
from numpy import nan
import sympy
from scipy import linalg

from . import levels,lines
from . import quantum_numbers
from . import tools
from .dataset import Dataset
from .tools import find,cache
from .optimise import Optimiser,P,auto_construct_method
from .kinetics import get_species,Species



class VibLevel(Optimiser):
    """A vibronic interaction matrix."""

    def __init__(
            self,
            name='viblevel',
            species='14N2',
            # Rbeg=None, Rend=None, Rstep=None, # for solving Schrödinger equation if potential energy curves are used
            J=None,                       # compute on this, or default added
            ef=None,                      # compute for these e/f parities , default ('e','f')
            # Emin=-1e50,Emax=1e50,         # limit potential solution finding to this range
            experimental_level=None,      # a Level object for optimising relative to 
            eigenvalue_ordering='maximise coefficients', # for deciding on the quantum number assignments of mixed level, options are 'minimise residual', 'maximise coefficients', 'preserve energy ordering' or None
            # fitΓ=None,       # optimise for linewidths or not
            Eref=0.,       # energy reference relative to equilibrium energy, defaults to 0. -- not well defined
            # Tbeg=None,Tend=None,
            # input_vibrational_level=None, # contains a list of electronic-vibrational level parameters, automatically added to the interacting states
            **qn,               # applies to all levels
    ):
        self.name = name          # a nice name
        self.species = get_species(species)
        tkwargs = {
            'Eref':Eref,
            'permit_auto_defaults':True, 
            'permit_nonprototyped_data':False,
        }
        self.rotational_level = levels.Diatomic(**tkwargs)
        self.vibrational_level = levels.Diatomic(**tkwargs)
        self.vibrational_spin_level = levels.Diatomic(**tkwargs) 

        # self.states = []          # Vibronic_State objects
        # self.interactions = []            # Vibronic_Interaction objects
        # self._vibronic_spin_manifolds = [] #  for the benefit of get_vibrational_level
        # self._vibronic_interactions = [] #  for the benefit of get_vibrational_interactions
        # self.ef = (ef if ef is not None else ('e','f'))    # set this to restrict parity to ('e',) or ('f',)
        # self.Eref = Eref                                   # energy refernce relative to Te
        # self.Tbeg,self.Tend = Tbeg,Tend
        ## get a good default J range
        species_object = get_species(species)
        J_is_half_integer = species_object['nelectrons']%2==1
        if J is None:
            if J_is_half_integer:
                self.J = np.arange(0.5,30.5,1)
            else:
                self.J = np.arange(31)
        else:
            self.J = np.array(J)
            if J_is_half_integer:
                assert np.all(np.mod(self.J,1)==0.5),f'Half-integer J required for {repr(species)}'
            else:
                assert np.all(np.mod(self.J,1)==0),f'Integer J required for {repr(species)}'
        ## internuclear distance regular grid for solving Schrödinger
        ## equation if potential energy curves are used
        # self.Rbeg = (Rbeg if Rbeg is not None else 0.01)
        # self.Rend = (Rend if Rend is not None else 10.)
        # self.Rstep = (Rstep if Rstep is not None else 0.001)
        # self.R = np.arange(self.Rbeg,self.Rend,self.Rstep)
        self.verbose = False
        # for key in qn:
            # if key not in Rotational_Level._class_key_data['vector']:
                # raise Exception(f'Unknown quantum number: {repr(key)}')
        self.qn = qn            # quantum numbers describing this model, the only really applicable element in species
        ## various options for how to assign the perturbed levels: None
        ## 'maximise coefficients' 'preserve energy ordering'
        ## 'minimise residual' 'coefficient continuity'
        # self.eigenvalue_ordering = (eigenvalue_ordering if eigenvalue_ordering is not None else 'maximise coefficients')
        self.eigenvalue_ordering = eigenvalue_ordering
        ## outputs of diagonalisation
        self.H = None
        self.eigvals = None
        self.eigvects = None
        self.cache = {'J':None}
        ## a Level object containing data, better access through level property
        self._exp = None # an array of experimental data matching the model data in shape
        ## the optimiser
        Optimiser.__init__(self,name=self.name)
        self.format_input_functions = [] # no formatted input
        def f(directory):
            self.rotational_level.save_to_file(directory+'/Rotational_Level.h5')
            self.vibrational_level.save_to_file(directory+'/Vibrational_Level') # 
            t = self.get_vibronic_interactions()
            t.save_to_file(directory+'/vibronic_interactions') # 
        self.add_save_to_directory_function(f)
        self.add_construct_function(self._initialise_construct)
        self.add_post_construct_function(self._finalise_construct)
        ## if experimental data is present this might be an adjustable
        ## thing to so add as a suboptimiser
        # if experimental_level is None:
            # self.set_experimental_level()
        # else:
            # self.set_experimental_level(experimental_level)
        # self.add_monitor(self.get_level) # ensure level constructed at end
        def f():
            # retval = '\nfrom anh import *\n'
            # retval += f"{self.name} = spectra.VibLevel(name={repr(self.name)},species={repr(self.species)},"
            retval = f"{self.name} = spectra.VibLevel(name={repr(self.name)},species={repr(self.species)},"
            if Rbeg is not None: 
                retval += f'Rbeg={repr(Rbeg)},'
            if Rend is not None:
                retval += f'Rend={repr(Rend)},'
            if Rstep is not None:
                retval += f'Rstep={repr(Rstep)},'
            if J is not None:
                retval += f'J={repr(J)},'
            if ef is not None:
                retval += f'ef={repr(ef)},'
            if experimental_level is not None:
                retval += f'experimental_level={experimental_level.name},'
            if eigenvalue_ordering is not None:
                retval += f'eigenvalue_ordering={repr(eigenvalue_ordering)},'
            if fitΓ is not None:
                retval += f'fitΓ={repr(fitΓ)},'
            if Eref is not None:
                retval += f'Eref={repr(Eref)},'
            if len(self.qn)>0:
                retval += f"{my.dict_to_kwargs(self.qn)})"
            retval += ')'
            return retval
        self.format_input_functions.append(f)
        # ## add spin manifolds form input Vibrational_Level
        # if input_vibrational_level is not None:
            # self.add_spin_manifolds_from_dynamic_array(input_vibrational_level)

    def _initialise_construct(self):
        ## cache this stuff
        self.H = np.full((len(self.J),
                          len(self.vibrational_spin_level),
                          len(self.vibrational_spin_level)),0.)

    def _finalise_construct(self):
        """Diagonlise J-blocks, add to rotational_level"""
        ## cache this
        self.rotational_level.clear()
        # self.iallowed = {}
        # for J in self.J:
        #     self.iallowed[J] = self.vibrational_spin_level['Ω']<=J
        #     self.rotational_level.extend(
        #         J=J,E=np.nan,
        #         **self.vibrational_spin_level[self.iallowed[J]])
        self.rotational_level['J'] = np.repeat(self.J,len(self.vibrational_spin_level))
        for key in self.vibrational_spin_level:
            self.rotational_level[key] = np.tile(self.vibrational_spin_level[key], len(self.J))
        self.rotational_level['E'] = nan
        self.c = {}
        self.E = {}
        ibeg = 0
        for iJ,J in enumerate(self.J):
            H = self.H[iJ,:,:]
            iallowed = self.vibrational_spin_level['Ω']<=J
            H[~iallowed,:] = H[:,~iallowed] = 0. # because sometimes NaN, probably should be zero in the first place
            eigvals,eigvects = linalg.eig(H)
            ## maximise coefficients
            c = eigvects.real**2 # fractional character
            for i in np.argsort(np.max(c,axis=1)): # loop through columns, beginnign with column containing the largest c
                j = np.argmax(c[i,:])              # index of largest c in this column
                ii = list(range(len(c)))           # index of columns
                ii[i],ii[j] = j,i                  # swap largest c into diagonal position 
                c = c[:,ii]                        # swap for all columns
                eigvals,eigvects = eigvals[ii],eigvects[:,ii] # and eigvalues
            ## save data
            self.E[J] = eigvals.real
            self.c[J] = eigvects
            ## add allowed level to rotational_level
            # iend = ibeg+np.sum(iallowed)
            # ibeg = ien
        self.rotational_level['E'] = np.concatenate(list(self.E.values()))
        
    @auto_construct_method('add_level')
    def add_level(self,name='level',**kwargs):
        """Add a new electronic vibrational level. kwargs contains fitting
        parameters and optionally extra quantum numbers."""
        ## get all quantum numbers
        for key,val in quantum_numbers.decode_level(name).items():
            kwargs.setdefault(key,val)
        kwargs['species'] = self.species.isotopologue
        self.vibrational_level.append(**kwargs)
        ## Checks that integer/half-integer nature of J corresponds to
        ## quantum number S
        if kwargs['S']%1!=self.J[0]%1:
            raise Exception(f'Integer/half-integer nature of S and J do not match: {S%1} and {self.J[0]%1}')
        ## get Hamiltonian and insert adjustable parameters into
        ## functions
        ef,Σ,H,fH = _get_linear_H(kwargs['S'],kwargs['Λ'],kwargs['s'])
        fH = [[lambda J,f=fH[i,j]: f(J,**kwargs)
                    for i in range(fH.shape[0])]
                   for j in range(fH.shape[1])]
        ## add to Hamiltonian
        ibeg = len(self.vibrational_spin_level)
        self.vibrational_spin_level.extend(ef=ef,Σ=Σ,**kwargs)
        def construct_function():
            for i in range(len(fH)):
                for j in range(i,len(fH)): 
                    self.H[:,ibeg+i,ibeg+j] = fH[i][j](self.J)
                    if i!=j:
                        self.H[:,ibeg+j,ibeg+i] = self.H[:,ibeg+i,ibeg+j]
        return construct_function

    # def set_experimental_level(self,*rotational_levels):
        # """Set a Rotational_Level as the experimental data to optimise. If
        # called repeatedly then concatenate the experimental data."""
        # if len(rotational_levels) == 0:
            # self.experimental_level = None
        # else:
            # self.experimental_level = Rotational_Level(Name=f'experimental_rotational_level_for_{self.name}')
            # self.experimental_level.extend(*rotational_levels)

    # def add_spin_manifolds_from_vibrational_level(self,vibrational_level):
        # """Add all rows in an input Vibrational_Level as a separate spin
        # manifold."""
        # ## determine which keys to add, inclues quantum numbers and
        # ## molecular parameters
        # keys = [key for key in vibrational_level.vector_data if vibrational_level.is_set(key)]
        # ## add each row
        # for level in vibrational_level:
            # kwargs = {}
            # for key in keys:
                # val = level[key]
                # ## leave out NaN values
                # if my.isnumeric(val) and np.isnan(val):
                    # continue
                # ## remove 'v' suffix from keys -- BETTER TO CHANGE
                # ## KEYS IN ONE OR THE OTHER I THINK
                # if len(key)>1 and key[-1]=='v':
                    # key = key[:-1]
                # kwargs[key] = val
            # ## add spin manifold
            # self.add_spin_manifold(**kwargs)

    # def add_interactions_from_vibrational_transition(self,vibrational_transition):
        # """Add all rows in an input Vibrational_Transition object as
        # interactions between existing spin manifolds."""
        # ## ensure names set, currently quantum numbetrs not used to identify levels
        # vibrational_transition['namep'],vibrational_transition['namepp']
        # ## useful data
        # for transition in vibrational_transition:
            # if vibrational_transition.is_set('ξv') and ~np.isnan(transition['ξv']) and transition['ξv']!=0.:
                # print( transition['namep'],transition['namepp'],transition['ξv'],)
                # self.add_L_uncoupling(transition['namep'],transition['namepp'],p=transition['ξv'])
            # if vibrational_transition.is_set('ηv') and ~np.isnan(transition['ηv']) and transition['ηv']!=0.:
                # print( transition['namep'],transition['namepp'],transition['ηv'],)
                # self.add_LS_coupling(transition['namep'],transition['namepp'],p=transition['ηv'])


    # def add_linear_manifold(
            # self,
            # name,
            # T=None,B=None,D=None,H=None,L=None,
            # A=None,AD=None,
            # λ=None,λD=None,λH=None,γ=None,γD=None,
            # o=None,p=None,pD=None,q=None,qD=None,
            # # print_latex_Hamiltonian=False,
            # # return_symbolic_Hamiltonian=False,
            # **qn,
    # ):
        # '''An effective Hamiltonian for diatomic molecules taken from
        # brown1979. Λ and S are deduced form by decoding the
        # name. There is no Λ-doubling in this formulation, so identical
        # f-ef and e-ef levels are created. Centrifugal parameters are
        # currently defined up to D or H. I am not wise enough to know
        # whether adding further N**2 terms will be insignificant
        # compared to other approximations made by brown1979 while
        # refining their effective Hamiltonian.  Not all parameters are
        # relevant to all kinds of states.'''
        # # ## do nothing if outisde Tbeg/Tend range
        # # Ttest = my.ensure_iterable(T)[0]
        # # if (   (self.Tbeg is not None and Ttest<self.Tbeg)
            # # or (self.Tend is not None and Ttest>self.Tend)):
            # # print(f"warning: Spin manifold {repr(name)} outside of range: Tbeg={self.Tbeg}, Tend={self.Tend}")
            # # return
        # ## get all quantum numbers possible from the name
        # # for key in qn:
            # # assert key in Rotational_Level._class_key_data['vector'],f'Unknown quantum number: {repr(key)}'
        # # qn_input = copy(qn)     # preserve those that were input only
        # qn = copy(qn)
        # for key,val in quantum_numbers.decode_level(name).items():
            # qn.setdefault(key,val)
        # # qn = self._combine_quantum_numbers(name,**qn)
        # # qn.setdefault('species',self.species)
        # ## parameters for optimisation
        # # parameters = self.add_parameter_set(
            # # note=f'add_spin_manifold {encode_level(**qn)}',
            # # step_default = {'T':1e-3,'B':1e-5,'D':1e-7,'H':1e-11,'L':1e-13,'A':1e-3,'AD':1e-5,
                            # # 'λ':1e-3,'λD':1e-5,'γ':1e-3,'γD':1e-5,'o':1e-3,'p':1e-3,'pD':1e-5,'q':1e-5,'qD':1e-7,},
            # # T=T,B=B,D=D,H=H,L=L,A=A,AD=AD,λ=λ,λD=λD,λH=λH,γ=γ,γD=γD,o=o,p=p,pD=pD,q=q,qD=qD,)
        # ## ## in order to compute SR, F and N correctly the sign of
        # ## ## spin-orbit interaction must be known,
        # ## ## here they are set depending on the sign of A (before
        # ## ## optimisation) and assuming negative spin-orbit interaction, or λ-B for Σ states
        # ## if qn['Λ']>0:                    qn.setdefault('LSsign',(-1 if parameters['A']<0 else +1))
        # ## # elif qn['Λ']==0 and qn['S']>0:   qn.setdefault('LSsign',(+1 if parameters['λ']>parameters['B'] else -1))
        # ## else:                            qn.setdefault('LSsign',1)
        # ## not all parameters make physical sense for all kinds of
        # ## states, raise an error if they are given in the input
        # ## arguments or varied
        # invalid_parameters = set()
        # for (test,parameters) in (
                # (qn['S']==0,('A','AD','λ','λD','λH','γ','γD',)),
                # (qn['Λ']==0,('A','AD')),
                # (qn['Λ']==0 and qn['S']==0,('o','p','pD','q','qD')),
                # (qn['Λ']==1 and qn['S']==0.5,('o',)),
        # ):
            # if test:
                # invalid_parameters.union(parameters)
        # parameters = {}
        # for key,val in (
            # ('T',T),('B',B),('D',D),('H',H),
            # ('A',A),('AD',AD),('λ',λ),('λD',λD),('λH',λH),('γ',γ),('γD',γD),
            # ('o',o),('p',p),('pD',pD),('q',q),('qD',qD),
        # ):
            # if key in invalid_parameters:
                # if val is not None:
                    # raise Exception(f'Invalid parameter set: {key}')
            # else:
                # parameters[key] = val if val is not None else 0.
        # # for key,val in parameters.items():
            # # if key not in valid_parameters and (val.p!=0 or val.vary==True):
                # # raise Exception("Invalid parameter for this kind of spin-manifold: "+repr(key))
        # # ## recreate input
        # # self.format_input_functions.append(lambda qn=qn: f"{self.name}.add_spin_manifold({repr(name)},\n{parameters.format_multiline(neglect_fixed_zeros=True)},\n   {my.dict_to_kwargs(qn_input)})")
        # # ## store vibronic_spin_manifold data for use in printing the output etc
        # # self._vibronic_spin_manifolds.append(dict(qn=qn,parameter_set=parameters))
        # ## some numerical quantum numbers used below
        # s,S,Λ = qn['s'],qn['S'],qn['Λ']
        # ## Checks that integer/half-integer nature of J corresponds to quantum number S
        # assert S%1==self.J[0]%1,f'Integer/half-integer nature of S and J do not match: {S%1} and {self.J[0]%1}'
        # ## symbolic variables, Note that expected value of ef is +1 or -1 for 'e' and 'f'
        # Z = sympy.Symbol
        # J = Z('J')
        # case_a = quantum_numbers.get_case_a_basis(Λ,s,S,print_output=False)
        # efs = case_a['qnef']['ef']
        # Σs = case_a['qnef']['Σ']
        # NN = case_a['NNef']
        # NS = case_a['NSef']
        # ## construct some convenient matrices
        # def anticommutate(X,Y): return(X*Y+Y*X) # convenience function
        # ## Equation 18 of brown1979
        # I  = sympy.eye(case_a['n']) # unit matrix
        # HH = Z('T')*I + Z('B')*NN - Z('D')*NN**2 + Z('H')*NN**3       # T + BN**2 - DN**4 + HN**6
        # if S>0:
            # if Λ>0: HH += anticommutate(Z('A')*I+Z('AD')*NN,sympy.diag(*[float(Λ*Σ) for Σ in Σs]))/2 # 1/2[A+AN2,LzSz]+
            # HH += (Z('γ')*I+Z('γD')*NN)*NS # (γ+γD.N**2)N.S
            # HH += anticommutate(Z('λ')*I+Z('λD')*NN+Z('λH')*NN**2,sympy.diag(*[float(Σ**2) for Σ in Σs])-I/3*S*(S+1)) # [λ+λD.N**2),Sz**2-1/3*S**2]+
        # ## add Λ-doubling terms here, element-by-element.
        # for i,(Σi,efi) in enumerate(zip(Σs,efs)):
            # for j,(Σj,efj) in enumerate(zip(Σs,efs)):
                # if efi!=efj: continue
                # ef = (1 if efi=='e' else -1) # Here ef is +1 for 'e' and -1 for 'f'
                # ## 1Π state
                # if Λ>0 and S==0: 
                    # if efi=='f' and efj=='f': HH[i,j] += Z('q')*J*(J+1)   # coefficient
                # # ## 1Π states -- f levels are shifted by q
                # # elif Λ==1 and S==0:
                    # # if efi=='f' and efj=='f':
                        # # HH[i,j] += q
                # ## is ef=1 for e, ef=-1 for f 2Π states, amiot1981
                # ## table II, there are more distortion constants,
                # ## which I have not included here, but could be
                # ## done. Perhaps these could be included with N-matrix
                # ## multiplication?
                # elif Λ==1 and S==0.5:
                    # for i,(Σi,efi) in enumerate(zip(Σs,efs)):
                        # efi = (1 if efi=='e' else -1)
                        # for j,(Σj,efj) in enumerate(zip(Σs,efs)):
                            # efj = (1 if efj=='e' else -1)
                            # if efi!=efj: continue
                            # ## diagonal element for level 2 in amiot1981
                            # if   Σi==-0.5 and Σj==-0.5:
                                # HH[i,j] += efi*( -0.5*(J+0.5)*Z('p')
                                                 # -(J+0.5)*Z('q')
                                                 # -0.5*(J+0.5)*((J-0.5)*(J+0.5)+2)*Z('pD')
                                                 # -0.5*(3*(J-0.5)*(J+0.5)+4)*(J+0.5)*Z('qD')    )
                            # ## diagonal element for level 1 in amiot1981
                            # elif Σi==+0.5 and Σj==+0.5:
                                # HH[i,j] += efi*( -0.5*(J-0.5)*(J+1.5)*(J+0.5)*Z('q')
                                                 # -0.5*(J-0.5)*(J+0.5)*(J+0.5)*Z('qD')  )
                            # ## off-diagonal element
                            # elif Σi==-0.5 and Σj==+0.5:
                                # HH[i,j] += efi*(  0.5*((J-0.5)*(J+1.5))**0.5*(J+0.5)*Z('q')
                                                  # + 0.25*((J-0.5)*(J+0.5))**0.5*(J+0.5)*Z('pD')
                                                  # +0.5*((J-0.5)*(J+0.5))**(0.5)*((J-0.5)*(J+0.5)+2)*(J+0.5)*Z('qD') )
                # ## 3Π states, from Table I brown_merer1979            
                # elif Λ==1 and S==1:
                    # ## diagonal elements
                    # if   Σi==-1 and Σj==-1: HH[i,j] += -ef*(Z('o')+Z('p')+Z('q'))
                    # elif Σi== 0 and Σj== 0: HH[i,j] += -ef*Z('q')*J*(J+1)/2
                    # elif Σi==+1 and Σj==+1: HH[i,j] += 0
                    # ## off-diagonal elements
                    # elif Σi==-1 and Σj==+0:
                        # HH[i,j] += -sympy.sqrt(2*J*(J+1))*-1/2*(Z('p')+2*Z('q'))*ef
                        # HH[j,i] += -sympy.sqrt(2*J*(J+1))*-1/2*(Z('p')+2*Z('q'))*ef
                    # elif Σi==-1 and Σj==+1:
                        # HH[i,j] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*Z('q')*ef
                        # HH[j,i] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*Z('q')*ef
                    # elif Σi==+0 and Σj==+1: pass # zero
                # else:
                    # if o is not None or p is not None or q is not None:
                        # raise Exception("Cannot use o, p, or q because Λ-doubling not specified for state with Λ="+repr(Λ)+" and S="+repr(S))
        # ## simplify and print Hamiltonian if requested
        # # if self.verbose:
            # # HH.simplify()
            # # print('\nH for '+name)
            # # pprint(HH)
        # # if print_latex_Hamiltonian:
            # # raise Exception("print_latex_Hamiltonian is apparently unreliable")
            # # for i in range(HH.shape[0]):
                # # for j in range(0,i+1):
                    # # print(r'\bigl<{}\bigl|H\bigr|{}\bigr> &= {} \\'.format(
                        # # encode_latex_term_symbol(**qn,Ω=qn['Λ']+Σs[i],ef=efs[i]),
                        # # encode_latex_term_symbol(**qn,Ω=qn['Λ']+Σs[j],ef=efs[j]),
                        # # VibLevel._latexify_term(str(HH[i,j]))))
                        # # _tools.latexify_sympy_expression_string(str(HH[i,j]))))
        # if self.verbose:
            # print( )
            # print(self.format_input_functions[-1]())
            # print_matrix_elements(case_a['qnef'],HH,'H')
        # ## loop through all pairs of levels and makes Vibronic_State
        # ## and Vibronic_Interaction for them. This is complicatd by
        # ## the fact that both e and f parities are stored in the
        # ## obejcts, but the matrix of case_a states includes both
        # ## components in one
        # for i in range(len(Σs)):
            # for j in range(i,-1,-1): # lower triangular, begin with diagonal elements
                # Σi,efi,Σj,efj = Σs[i],efs[i],Σs[j],efs[j]
                # if efi!=efj: continue # nothing to do here
                # ## turn each symbolic function into a numerical
                # ## function using sympy printing and eval, this is a
                # ## slow step during the model initialisation
                # ## t = sympy.lambdify(['J','T','B','D','H','A','AD','λ','λD','λH','γ','γD','o','p','q'],HH[i,j]) # make symbolic matrix element into a numerical function
                # ## fH = lambda J,t=t: t(J=J,**{key:val for key,val in parameters.items() if key[0]!='Γ'})
                # print('DEBUG:', HH[i,j])
                # fH = tools.lambdify_sympy_expression(
                    # HH[i,j],
                    # ('J',),
                    # parameters.keys(),
                # )
                # if i==j:        # diagonal element, make a new Vibronic_State, or add parity level to existing one
                    # self.states.append(Vibronic_State(name=name+'_Σ='+format(Σi,'g'),Σ=Σi,Ω=np.abs(Σi+qn['Λ']),ef=efi, fH=fH, **qn))
                # else:        # off-diagonal element, make as Vibronic_Interaction
                    # statei = self.get_state(Σ=Σi,ef=efi,raise_exception_on_not_found=False,**qn)
                    # statej = self.get_state(Σ=Σj,ef=efj,raise_exception_on_not_found=False,**qn)
                    # if statei is None or statej is None: continue # one or both states does not exist
                    # self.interactions.append(Vibronic_Interaction(statei,statej,fH=fH))
        # ##
        # if return_symbolic_Hamiltonian:
            # ## print(case_a['qnef'])
            # return(HH)

    # def add_electronic_spin_manifold(
            # self,
            # electronic_spin_manifold,          # a Electronic_Spin_Manifold object
            # v=None,            # which vibrational levels to include, None for all in Emin/Emax range
    # ):
        # """Add vibrational levels computed from an Electronic_Spin_Manifold object."""
        # ## align R grid
        # electronic_spin_manifold.set_R(self.Rbeg,self.Rend,self.Rstep)
        # v = my.ensure_iterable(v)
        # self.suboptimisers.append(electronic_spin_manifold) # add subpoptimiser, to make electronic_spin_manifolds
        # ## make sure the necessary levels are computed by the electronic_spin_manifolds
        # electronic_spin_manifold.construct_functions.append(lambda: electronic_spin_manifold.find_bound_level(v,species=self.species,J=self.J))
        # ## new input function
        # self.format_input_functions.append(lambda:f'{self.name}.add_electronic_spin_manifold({electronic_spin_manifold.name},v={repr(v)})')
        # ## loop through all spin-vibrational levels adding a Vibronic_State to self
        # for vi in v:
            # states = []
            # for istate,case_a_qnef in enumerate(electronic_spin_manifold.case_a['qnef']):
                # ## get all relevant quantum numbers
                # qn = copy(electronic_spin_manifold.qn)
                # qn['v'],qn['species'],qn['Σ'],qn['ef'] = vi,self.species,case_a_qnef['Σ'],case_a_qnef['ef']
                # ## get functions returning energy and wavefunctions
                # def fH(J,qn=qn): # J is vector
                    # i = electronic_spin_manifold.level.match(J=J,**qn)
                    # Ji,Ti = electronic_spin_manifold.level['J'][i],electronic_spin_manifold.level['T'][i]
                    # ## If there are missing levels then replace with np.nan
                    # if len(Ji)==0:
                        # retval = np.full(len(J),np.nan)
                    # elif len(Ji)<len(J):
                        # warnings.warn(f"Missing rotational levels for {electronic_spin_manifold.name} v={v}")
                        # retval = np.full(len(J),np.nan)
                        # i,j = my.common(J,Ji)
                        # retval[i] = Ti[j]
                    # else:
                        # retval = Ti
                    # return(retval)
                    # # return(electronic_spin_manifold.level['T'][i])
                # def fχ(J,qn=qn): # J is vector
                    # return(list( # some manipulation required to get a good shape for the object array data
                        # electronic_spin_manifold.level['χ'][
                            # electronic_spin_manifold.level.match(J=J,**qn)]))
                # ## add state
                # state = Vibronic_State(name=encode_level(**qn),fH=fH,fχ=fχ,R=self.R,**qn)
                # self.states.append(state); states.append(state)
                # ## add interactions
                # for statep in states[:istate]:
                    # qnp = statep.qn
                    # if qn['ef']!=qnp['ef']: continue # no interaction allowed
                    # def fH(J,qn=qn,qnp=qnp):
                        # return(electronic_spin_manifold.get_V(species=self.species,J=J,Σi=qn['Σ'],Σj=qnp['Σ'],ef=qn['ef']))
                    # self.interactions.append(Vibronic_Interaction(state,statep, fH=fH))

    # def _combine_quantum_numbers(self,name=None,**qn):
        # """Combine a set of quantum numbers from a given name an set of # 
        # quantum numbers, adding the model species."""
        # retval = {}
        # if name is not None:
            # retval.update(decode_level(name)) # from the name
        # retval.update(self.qn)                                 # from the VibLevel object
        # retval.update(qn)                                      # from provide qn
        # return(retval)

    # # def _prepare_experimental_level(self):
    # #     """Add experimental in Level object, and prepare for use in
    # #     optimsiation. An internal function because it should only
    # #     called after all spin manifolds are defined."""
    # #     assert isinstance(self.experimental_level,Rotational_Level),'Must be a Level object'
    # #     ## some error checking
    # #     if self.experimental_level.is_known('T'):
    # #         assert not np.any(np.isnan(self.experimental_level['T'])), 'NaN in experimental level data: T'
    # #         assert not np.any(np.isnan(self.experimental_level['T'])), 'NaN in experimental level data: dT'
    # #     if self.experimental_level.is_known('dT'):
    # #         assert np.all(self.experimental_level['dT']>=0), 'negative value in experimental level data: dT'
    # #     # if self.experimental_level.is_known('Γ'):
    # #         # assert not np.any(np.isnan(self.experimental_level['Γ'])), 'NaN in experimental level data: Γ'
    # #         # assert np.all(self.experimental_level['Γ']>=0), 'negative value in experimental level data: Γ'
    # #     if self.experimental_level.is_known('dΓ'):
    # #         assert not np.any(np.isnan(self.experimental_level['dΓ'])&(~np.isnan(self.experimental_level['Γ']))), 'NaN in experimental level data: dΓ'
    # #         assert not np.all((self.experimental_level['dΓ']<=0)&(~np.isnan(self.experimental_level['Γ']))), 'Zero or negative value in experimental level data: dΓ'
    # #     if self.J is None: self.J = self.experimental_level.unique('J') # default J
    # #     ## divide experimental into J blocks, in the same order as states in model
    # #     self._exp = dict( T = np.full((len(self.J),len(self.states)),np.nan),
    # #                      dT = np.full((len(self.J),len(self.states)),1.), # defaults to 1
    # #                       Γ = np.full((len(self.J),len(self.states)),np.nan),
    # #                      dΓ = np.full((len(self.J),len(self.states)),1.),) # defaults to 1
    # #     for iJ,J in enumerate(self.J):
    # #         for istate,state in enumerate(self.states):
    # #             ## Get matching levels. This currently requires a
    # #             ## massive hack remove LSsign from state.qn.  That is
    # #             ## because this is not really a quantum number, but is
    # #             ## needed internally to compute Fi from Ω.  Could add
    # #             ## a new state property, state.parameters or somethign
    # #             ## and put LSsign in their.
    # #             tqn = copy(state.qn)
    # #             if 'LSsign' in tqn: tqn.pop('LSsign')
    # #             t = self.experimental_level.matches(J=J,**tqn)
    # #             # ## not hacked version -- LSsign must be set and matched!
    # #             # t = self.experimental_level.matches(J=J,**state.qn)
    # #             if len(t)==1:
    # #                 for key in ('T','dT','Γ','dΓ'):
    # #                     if not self.experimental_level.is_known(key): continue
    # #                     self._exp[key][iJ,istate] = float(t[key])
    # #             elif len(t)>1:
    # #                 raise Exception(f'Multiple experimental data found for J={repr(J)}  {repr(state.qn)}')
    # #     assert not np.any(self._exp['dT']==0), 'Zero term value errors in experimental level.'
    # #     assert not np.any(self._exp['dΓ']==0), 'Zero width errors in experimental level.'

    # def _prepare_experimental_level(self):
        # """Add experimental in Level object, and prepare for use in
        # optimsiation. An internal function because it should only
        # called after all spin manifolds are defined."""
        # ## divide experimental into J blocks, in the same order as states in model
        # self._exp = dict( T = np.full((len(self.J),len(self.states)),np.nan),
                         # dT = np.full((len(self.J),len(self.states)),1.), # defaults to 1
                          # Γ = np.full((len(self.J),len(self.states)),np.nan),
                         # dΓ = np.full((len(self.J),len(self.states)),1.),) # defaults to 1
        # ## find experimental data for each state
        # for istate,state in enumerate(self.states):
            # t = self.experimental_level.matches(**state.qn)
            # ## get required J
            # i,j = my.common(self.J,t['J'])
            # if len(j)==0: continue # no experimental data for this (state,J)
            # ## add to experimental data
            # for key in ('T','dT','Γ','dΓ'):
                # if not self.experimental_level.is_known(key): continue
                # self._exp[key][i,istate] = t[key][j]
        # assert not np.any(self._exp['dT']==0), 'Zero term value errors in experimental level.'
        # assert not np.any(self._exp['dΓ']==0), 'Zero width errors in experimental level.'

    # def construct_level(self):
        # """Generate and perturb energy levels."""
        # self._rotational_level.clear()
        # ## process experimental data if not already done once
        # if self.experimental_level is not None and self._exp is None:
            # self._prepare_experimental_level() 
        # # del self.residual_level          # this will be reconstructed if necessary
        # J = self.J
        # if self.verbose:
            # print( )
            # print('constructing states and interactions')
        # for t in self.states:
            # t.construct(self.J) # compute all energy levels
        # for t in self.interactions:
            # t.construct(self.J) # compute all off-diagonal matrix elements / interaction energies
        # ## make the zeroth-order matrix to digaonlise
        # if self.verbose:
            # print('diagonalising')
        # self.H = np.zeros((len(self.J),len(self.states),len(self.states),),dtype=complex)
        # for i,state in enumerate(self.states):
            # self.H[:,i,i] = state.H # diagonal elements
        # for interaction in self.interactions: # off diagonal elements
            # i,j = self.states.index(interaction.statei),self.states.index(interaction.statej)
            # self.H[:,i,j] += interaction.H
            # self.H[:,j,i] += interaction.H
        # self.eigvals = np.full((len(self.J),len(self.states)),np.nan,dtype=complex) # prepare an erray to hold eigenvalues
        # self.eigvects = np.zeros(self.H.shape) # and eigenvectors
        # ## diagonalise each (ef,J) matrix
        # for iJ,J in enumerate(self.J):
            # for ef in ('e','f'):
                # H = self.H[iJ,:,:]
                # ief = np.array([t.qn['ef']==ef for t in self.states])
                # ivalid = my.find((~np.isnan(H.diagonal()))&ief) # nans indicate state that have no level for this (ef,J)
                # H = H[ivalid,:][:,ivalid] # reduced H without such states
                # if len(H)==0: continue
                # eigvals,eigvects = linalg.eig(H) # diagonalise
                # ## optionally reorder eigenvalues in some way -- pretty slow
                # if self.eigenvalue_ordering is None:
                    # pass
                # ## another possible ordering to optionally use
                # elif self.eigenvalue_ordering=='preserve energy ordering': # adiabatic
                    # i = np.argsort(eigvals.real)[np.argsort(np.argsort(np.diag(H)))]
                    # eigvals,eigvects = eigvals[i],eigvects[:,i]
                # ## reorder to maximise coefficients on diagonal, either because asked or because a 'minimise_residual' is to follow
                # elif ((self.eigenvalue_ordering=='maximise coefficients')
                      # or (self.eigenvalue_ordering=='minimise residual' and self.experimental_level is not None)):
                    # c = eigvects.real**2 # fractional character
                    # for i in np.argsort(np.max(c,axis=1)): # loop through columns, beginnign with column containing the largest c
                        # j = np.argmax(c[i,:])              # index of largest c in this column
                        # ii = list(range(len(c)))           # index of columns
                        # ii[i],ii[j] = j,i                  # swap largest c into diagonal position 
                        # c = c[:,ii]                        # swap for all columns
                        # eigvals,eigvects = eigvals[ii],eigvects[:,ii] # and eigvalues
                    # ## reorder to match experiment if there is experimental data
                    # if self.eigenvalue_ordering=='minimise residual':
                        # if self.experimental_level is not None:
                            # ## rearrange to best match experimenatl data. Loop
                            # ## through experimental data and swap model levels to
                            # ## get best match.
                            # if self.eigenvalue_ordering=='minimise residual':
                                # texp,tmod = np.meshgrid(self._exp['T'][iJ,ivalid],eigvals)
                                # Δ = np.abs(texp-tmod)
                                # while not np.isnan(Δ).all():
                                    # with np.warnings.catch_warnings(): # this particular nan warning will be silenced for this block, occurs when some rows of Δ are all NaN
                                        # np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
                                        # i = np.nanargmin(np.nanmin(Δ,1))
                                        # j = np.nanargmin(Δ[i,:])
                                    # eigvals[i]   ,eigvals[j]    = eigvals[j]   ,eigvals[i] 
                                    # eigvects[:,i],eigvects[:,j] = eigvects[:,j],eigvects[:,i]
                                    # Δ[i,:],Δ[j,:] = Δ[j,:],Δ[i,:]
                                    # Δ[j,:] = Δ[:,j] = np.nan
                # elif self.eigenvalue_ordering=='minimise residual' and self.experimental_level is None:
                    # raise Exception("Cannot reorder eigenvalues too minimise resiudal, there is no experimental data.")
                # else:
                    # raise Exception("Unknown value for eigenvalue_ordering: "+repr(self.eigenvalue_ordering))
                # ## expand to full size of states, inverting the above
                # ## process where nonexisting levels (nans) were neglected
                # self.eigvals[iJ,ivalid] = eigvals
                # ## self.eigvects[ief,iJ][np.meshgrid(ivalid,ivalid)] = eigvects.real
                # self.eigvects[iJ][np.ix_(ivalid,ivalid)] = (eigvects.real).transpose()
        # ## return residual for optimisation if requested
        # if self.experimental_level is not None:
            # residuals = np.concatenate((self.rotational_level['Tresidual'],self.rotational_level['Γresidual'],))
            # residuals = residuals[~np.isnan(residuals)]
        # else:
            # residuals = []
        # return(residuals)

    # def _get_rotational_level(self):
        # """Construct a level object from object data."""
        # ## data alreaduy collected
        # if len(self._rotational_level)>0:
            # return(self._rotational_level)
        # ## ## update self if necessary ## causes infinite recurseion
        # ## if self.has_changed():
        # ##     self.construct()
        # ## collect data
        # if self.eigvals is not None:
            # for i,state in enumerate(self.states):
                # T = self.eigvals[:,i].real
                # Γ = self.eigvals[:,i].imag
                # j = ~np.isnan(T)
                # if np.any(j):
                    # self._rotational_level.append(J=state.J[j],T=T[j],Γ=Γ[j],**state.qn)
            # ## get residual errors perhaps
            # if self.experimental_level is not None:
                # i,j = find_common(self._rotational_level,self.experimental_level)
                # # if np.sum(i)>0:
                # ## set Tresidual, dTresidual, Γresidual, and dΤresidual in
                # ## model Rotational_Level if possible
                # for key in ('T','Γ'):
                    # self._rotational_level[key+'residual'] = self._rotational_level['d'+key+'residual'] = np.nan
                    # if self.experimental_level.is_known(key):
                        # self._rotational_level[key+'residual'][i] = self.experimental_level[key][j]-self._rotational_level[key][i] # 
                        # if self.experimental_level.is_known('d'+key):
                            # self._rotational_level['d'+key+'residual'][i] = self.experimental_level['d'+key][j]
                        # else:
                            # self._rotational_level['d'+key+'residual'][i] = np.nan
        # self._rotational_level['partition_source'] = 'self'
        # return(self._rotational_level)

    # rotational_level = property(_get_rotational_level)

    # def _get_vibrational_level(self):
        # """Get a Vibrational_Level containing all spin-manifold
        # parameters. This could somehow be a property or be a static
        # object."""
        # if len(self._vibrational_level) == 0:
            # ## creat a new Vibrational_Level and set Eref if possible
            # self._vibrational_level['description'] = 'List of vibronic level constants.'
            # self._vibrational_level['Eref'] = self.Eref
            # ## add data from vibronic_spin_manifolds
            # for t in self._vibronic_spin_manifolds:
                # keys_vals = copy(t['qn'])
                # for name,p in t['parameter_set'].items():
                    # if not np.isnan(p.p):
                        # keys_vals[name+'v'] = p.p
                        # if not np.isnan(p.dp):
                            # keys_vals['d'+name+'v'] = p.dp
                # ## ensure defaults are set to prevent incomplete data in some rows
                # self._vibrational_level.set_default(*keys_vals)
                # ## add a new row
                # self._vibrational_level.append(**keys_vals)
        # return(self._vibrational_level)

    # vibrational_level = property(_get_vibrational_level)

    # def get_vibronic_interactions(self):
        # """Get a Vibrational_Transition containing mixing energy for all
        # interacting levels."""
        # retval = Vibrational_Transition(description='List of nonradiative couplings.',Eref=self.Eref)
        # for vibronic_interaction in self._vibronic_interactions:
            # keys_vals = copy(vibronic_interaction['qn'])
            # for name,p in vibronic_interaction['parameter_set'].items():
                # keys_vals[name+'v'] = p.p
                # keys_vals['d'+name+'v'] = p.dp
            # ## find relevant state spin_manifolds if they exist
            # qnp,qnpp = separate_upper_lower_quantum_numbers(vibronic_interaction['qn'])
            # for t_vibronic_spin_manifold in self._vibronic_spin_manifolds:
                # if match_quantum_numbers(qnp,t_vibronic_spin_manifold['qn']):
                    # for key,val in t_vibronic_spin_manifold['parameter_set'].items():
                        # keys_vals[key+'vp'] = val 
            # for t_vibronic_spin_manifold in self._vibronic_spin_manifolds:
                # if match_quantum_numbers(qnpp,t_vibronic_spin_manifold['qn']):
                    # for key,val in t_vibronic_spin_manifold['parameter_set'].items():
                        # keys_vals[key+'vpp'] = val 
            # ## add line data
            # ## retval.set_default(*keys_vals)
            # retval.append(**keys_vals)
        # return retval
   #  
    # def set_width(self,name='',Γ0=0,Γ1=0,Γ2=0,**qn):
        # """Set zero-order width of states matching name, Ω, ef (Ω and ef can
        # be lists of levels to match. Γ are polynomial is a constant or
        # a list of polynomial coefficiens in J(J+1) or else a function
        # in terms of J.  This method should be called after real part
        # of this states Hamiltonian is defined."""
        # all_qn = decode_level(name)
        # all_qn.update(qn)
        # all_qn['species'] = self.species
        # p = self.add_parameter_set(
            # ## note=f'set_width {all_qn}',
            # note=f'set_width {encode_level(**all_qn)}',
            # Γ0=Γ0,Γ1=Γ1,Γ2=Γ2,step_default=dict(Γ0=1e-2,Γ1=1e-4,Γ2=1e-6))
        # self.format_input_functions.append(lambda: f'{self.name}.set_width({repr(name)},{p.format_input(neglect_fixed_zeros=True)},{my.dict_to_kwargs(qn)})')
        # def Γf(J,p=p): # Γ as a function of J
            # return(np.polyval([p['Γ2'],p['Γ1'],p['Γ0']],J*(J+1)))
        # ## add imaginarcy width to existing Vibronic_State
        # for state in self.get_matching_states(name,**all_qn):
            # state.fH = lambda J,fH=state.fH,Γf=Γf: fH(J) + 1j*Γf(J)

    # def set_width_spline(
            # self,
            # name='',
            # Js=[0,100],        # list defining spline know J
            # Γs=[1,1],          # list Γ at knots (or a constant value)
            # vary=True,         # constant or a list
            # step=0.01,         # constant or a list
            # order=1,           # spline order -- default linear 
            # **qn
    # ):
        # all_qn = decode_level(name)
        # all_qn.update(qn)
        # all_qn['species'] = self.species
        # if np.isscalar(Γs):
            # Γs = Γs*np.ones(len(Js)) # default amplitudes to list of hge same length
        # Js,Γs = np.array(Js),np.array(Γs)
        # p = self.add_parameter_list(f'set_width_spline',Γs,vary,step) # add to optimsier
        # self.format_input_functions.append(lambda: f"{self.name}.set_width_spline({repr(name)},Js={repr(list(Js))},Γs={repr(p.plist)},vary={repr(vary)},step={repr(step)},order={order},{my.dict_to_kwargs(qn)})")
        # def Γf(J): # Γ as a function of J
            # Γ = np.full(J.shape,0.0)
            # i = (J>=Js.min())&(J<=Js.max())
            # Γ[i] = my.spline(Js[i],p.plist,order=order)
            # return(Γ)
        # ## add imaginary width to existing Vibronic_State
        # for state in self.get_matching_states(name,**all_qn):
            # state.fH = lambda J,fH_old=state.fH: fH_old(J) + 1j*my.spline(Js,p.plist,J,order=order)

    # def get_matching_states(self,name='',**qn):
        # """Find states by quantum number."""
        # qn = copy(qn)
        # for key,val in decode_level(name).items():
            # qn.setdefault(key,val)
        # for key in qn: 
            # qn[key] = my.ensure_iterable(qn[key])
        # retval = []
        # for t in self.states:
            # for key,val in qn.items():
                # if key not in t.qn or t.qn[key] not in val:
                    # break
            # else:
                # retval.append(t)
        # if len(retval)==0:
            # warnings.warn(f'no matching_states found for {name}, {qn}')
        # return(retval)

    # def get_state(self,name=None,raise_exception_on_not_found=True,**qn):
        # """FInd one and only one states."""
        # if name is not None:
            # for key,val in decode_level(name).items(): qn.setdefault(key,val)
        # states = self.get_matching_states(**qn)
        # if len(states)==1: return(states[0])
        # if len(states)==0:
            # if raise_exception_on_not_found:
                # raise Exception('State not found: '+repr(name)+' '+repr(qn))
            # else:
                # return(None)
        # if len(states)>1:  raise Exception('State not unique: '+repr(name)+' '+repr(qn))

    # def add_coupling(
            # self,
            # name1,name2,
            # ξ=0,ξD=0,           # L-uncoupling interaction parameters
            # η=0,ηD=0,           # spin-orbit interaction parameters
            # qn1=None,qn2=None,
            # # print_latex_Hamiltonian=False,
            # verbose=False,
            # # suboptimiser=None, # dictionary like, containing, ξ,η,ξD,ηD   e.g., another optimiser
    # ):
        # """Calculate algebraic part of Σli.si operator and optimise the
        # unknown part.  Not very accurate? ±0.01cm-1?""" 
        # verbose = verbose or self.verbose
        # ## take care of inputs
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # input_qn1,input_qn2 = copy(qn1),copy(qn2) # store before addition
        # qn1 = self._combine_quantum_numbers(name1,species=self.species,**qn1)
        # qn2 = self._combine_quantum_numbers(name2,species=self.species,**qn2)
        # ## Add parameter to optimisation unless externally provided,
        # ## in which case p_external_optimisation is dictionary-like
        # ## providing ξ,η,ξD,ηD
        # p = self.add_parameter_set(
            # note=f'add_coupling {encode_transition(qnp=qn1,qnpp=qn2)}',
            # ξ=ξ,ξD=ξD,η=η,ηD=ηD,
            # step_default=dict(ξ=1e-2,ξD=1e-4,η=1e-2,ηD=1e-4),) # add adjustable parameters to optimiser
        # self.format_input_functions.append(
            # lambda name1=name1,name2=name2,qn1=input_qn1,qn2=input_qn2: f'{self.name}.add_coupling({repr(name1)},{repr(name2)},\n{p.format_multiline(neglect_fixed_zeros=True)},\n   qn1={repr(qn1)},qn2={repr(qn2)})')
        # self._vibronic_interactions.append(
            # {'qn':join_upper_lower_quantum_numbers(qn1,qn2), 'parameter_set':p})
        # if verbose:
            # print('\n\n',self.format_input_functions[-1]()) # monitor verbosely
        # state1s,state2s = self.get_matching_states(**qn1),self.get_matching_states(**qn2) # find interacting states
        # # assert len(state1s)>0 and len(state2s)>0, 'No interacting states found. name1='+repr(name1)+' name2='+repr(name2)+' qn1='+repr(qn1)+' qn2='+repr(qn2)
        # if len(state1s)==0:
            # print(f'warning: No interacting state found: {repr(name1)}')
            # return
        # if len(state2s)==0:
            # print(f'warning: No interacting state found: {repr(name2)}')
            # return
        # ## get case (a) quantum numbers etc, and symbolic rotational and spin-orbit interaction matrices
        # casea,JLef,JSef = get_rotational_coupling_matrix(qn1=qn1,qn2=qn2)
        # casea,LSef = get_spin_orbit_coupling_matrix(qn1=qn1,qn2=qn2)
        # if verbose:
            # print( "Case (a) quantum numbers e/f parity basis:\n")
            # for it,t in enumerate(casea['qnef'].iter_data_collection()):
                # print( it+1,encode_level(**t))
            # print( "\nHJL e/f parity basis:\n")
            # pprint(JLef)
            # print( "\nHLS e/f parity basis:\n")
            # pprint(LSef)
        # η,ηD,ξ,ξD = sympy.Symbol("η"),sympy.Symbol("ηD"),sympy.Symbol("ξ"),sympy.Symbol("ξD")
        # NNJLef = (JLef*casea['NNef']+casea['NNef']*JLef)/2                
        # NNLSef = (LSef*casea['NNef']+casea['NNef']*LSef)/2
        # H = -ξ*JLef - ξD*NNJLef + η*LSef + ηD*NNLSef
        # ## lambdify symbolic function and insert fitting parameters
        # for i,qni in enumerate(casea['qnef']): # bra
            # for j,qnj in enumerate(casea['qnef']): # ket
                # if j<=i: continue                  # symmetric
                # if H[i,j]==0: continue # no interaction
                # if qni['ef']!=qnj['ef']: continue
                # for statei in self.get_matching_states(name1,Σ=qni['Σ'],ef=qni['ef']):
                    # for statej in self.get_matching_states(name2,Σ=qnj['Σ'],ef=qni['ef']):
                        # ## new cached_lambdify_sympy_expression -- get
                        # ## cached pure function, then make another
                        # ## function subsituting in varied parameters
                        # raw_function =_tools.cached_lambdify_sympy_expression(('J','η','ηD','ξ','ξD'),H[i,j])
                        # def fH(J,f=raw_function,p=p):
                            # return f(J,η=p['η'],ηD=p['ηD'],ξ=p['ξ'],ξD=p['ξD'])
                        # self.interactions.append(Vibronic_Interaction(statei,statej,fH=fH))
                        # ## ## old lambdify_sympy_expression
                        # ## self.interactions.append(Vibronic_Interaction(
                        # ##     statei,statej,
                        # ##     fH=_tools.lambdify_sympy_expression(H[i,j],('J',),{},p)))

    # def add_L_uncoupling(
            # self,
            # name1,name2,
            # p=0,pD=0,           # L-uncoupling interaction parameters
            # qn1=None,qn2=None,
            # verbose=False,
    # ):
        # """USE THIS OR USE AND ADD_SPIN_ORBIT_COUPLING OR USE ADD_COUPLING?""" 
        # verbose = verbose or self.verbose
        # ## take care of inputs
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # input_qn1,input_qn2 = copy(qn1),copy(qn2) # store before addition
        # qn1 = self._combine_quantum_numbers(name1,species=self.species,**qn1)
        # qn2 = self._combine_quantum_numbers(name2,species=self.species,**qn2)
        # ## Add parameter to optimisation unless externally provided,
        # ## in which case p_external_optimisation is dictionary-like
        # ## providing p,pD
        # parameter_set = self.add_parameter_set(
            # note=f'add_L_uncoupling {encode_transition(**{key+"p":val for key,val in qn1.items()},**{key+"pp":val for key,val in qn2.items()})}',
            # p=p,pD=pD,step_default=dict(p=1e-2,pD=1e-4),)
        # self.format_input_functions.append(lambda name1=name1,name2=name2,qn1=input_qn1,qn2=input_qn2: f'{self.name}.add_L_uncoupling({repr(name1):15},{repr(name2):15},{parameter_set.format_input(neglect_fixed_zeros=True)},qn1={repr(qn1)},qn2={repr(qn2)})')
        # self._vibronic_interactions.append({'qn':join_upper_lower_quantum_numbers(qn1,qn2), 'parameter_set':parameter_set})
        # # if verbose: print('\n\n',self.format_input_functions[-1]()) # monitor verbosely
        # state1s,state2s = self.get_matching_states(**qn1),self.get_matching_states(**qn2) # find interacting states
        # if len(state1s)==0:
            # print(f'warning: No interacting state found: {repr(name1)}')
            # return
        # if len(state2s)==0:
            # print(f'warning: No interacting state found: {repr(name2)}')
            # return
        # ## get case (a) quantum numbers etc, and symbolic rotational and spin-orbit interaction matrices
        # casea,JLef,JSef = get_rotational_coupling_matrix(qn1=qn1,qn2=qn2)
        # # if verbose:
            # # print( "Case (a) quantum numbers e/f parity basis:\n")
            # # for it,t in enumerate(casea['qnef'].iter_data_collection()):
                # # print( it+1,encode_level(**t))
            # # print( "\nHJL e/f parity basis:\n")
            # # pprint(JLef)
        # ## monitor verbosely
        # p,pD = sympy.Symbol("p"),sympy.Symbol("pD")
        # NNJLef = (JLef*casea['NNef']+casea['NNef']*JLef)/2                
        # H = -p*JLef - pD*NNJLef
        # ## print verbosely
        # if verbose:
            # print( )
            # print(self.format_input_functions[-1]())
            # print_matrix_elements(casea['qnef'],H,'L-uncoupling')
        # ## lambdify symbolic function and insert fitting parameters
        # for i,qni in enumerate(casea['qnef']): # bra
            # for j,qnj in enumerate(casea['qnef']): # ket
                # if j<=i: continue                  # symmetric
                # if H[i,j]==0: continue # no interaction
                # if qni['ef']!=qnj['ef']: continue
                # for statei in self.get_matching_states(name1,Σ=qni['Σ'],ef=qni['ef']):
                    # for statej in self.get_matching_states(name2,Σ=qnj['Σ'],ef=qni['ef']):
                        # self.interactions.append(Vibronic_Interaction(
                            # statei,statej,
                            # fH=_tools.lambdify_sympy_expression(H[i,j],('J',),{},parameter_set)))

    # def add_S_uncoupling(
            # self,
            # name1,name2,
            # p=0,pD=0,           # L-uncoupling interaction parameters
            # qn1=None,qn2=None,
            # verbose=False,
    # ):
        # """USE THIS OR USE AND ADD_SPIN_ORBIT_COUPLING OR USE ADD_COUPLING?"""
        # print('warning:  add_S_uncoupling has incorrect phases relative to PGopher, at least for a 3Σ+/3Σ- interaction.')
        # verbose = verbose or self.verbose
        # ## take care of inputs
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # input_qn1,input_qn2 = copy(qn1),copy(qn2) # store before addition
        # qn1 = self._combine_quantum_numbers(name1,species=self.species,**qn1)
        # qn2 = self._combine_quantum_numbers(name2,species=self.species,**qn2)
        # ## Add parameter to optimisation unless externally provided,
        # ## in which case p_external_optimisation is dictionary-like
        # ## providing p,pD
        # parameter_set = self.add_parameter_set(
            # note=f'add_S_uncoupling {encode_transition(**{key+"p":val for key,val in qn1.items()},**{key+"pp":val for key,val in qn2.items()})}',
            # p=p,pD=pD,step_default=dict(p=1e-2,pD=1e-4),)
        # self.format_input_functions.append(lambda name1=name1,name2=name2,qn1=input_qn1,qn2=input_qn2: f'{self.name}.add_S_uncoupling({repr(name1):15},{repr(name2):15},{parameter_set.format_input(neglect_fixed_zeros=True)},qn1={repr(qn1)},qn2={repr(qn2)})')
        # self._vibronic_interactions.append({'qn':join_upper_lower_quantum_numbers(qn1,qn2), 'parameter_set':parameter_set})
        # state1s,state2s = self.get_matching_states(**qn1),self.get_matching_states(**qn2) # find interacting states
        # if len(state1s)==0:
            # print(f'warning: No interacting state found: {repr(name1)}')
            # return
        # if len(state2s)==0:
            # print(f'warning: No interacting state found: {repr(name2)}')
            # return
        # ## get case (a) quantum numbers etc, and symbolic rotational and spin-orbit interaction matrices
        # # casea,JLef,JSef = get_rotational_coupling_matrix(qn1=qn1,qn2=qn2)
        # casea,JLef,JSef = get_rotational_coupling_matrix(qn1=qn1,qn2=qn2,verbose=False)
        # p,pD = sympy.Symbol("p"),sympy.Symbol("pD")
        # NNJSef = (JSef*casea['NNef']+casea['NNef']*JSef)/2                
        # H = -(p*JSef + pD*NNJSef)
        # ## monitor verbosely
        # if verbose:
            # print( )
            # print(self.format_input_functions[-1]())
            # print_matrix_elements(casea['qnef'],H,'S-uncoupling')
        # ## lambdify symbolic function and insert fitting parameters
        # for i,qni in enumerate(casea['qnef']): # bra
            # for j,qnj in enumerate(casea['qnef']): # ket
                # if j<=i: continue                  # symmetric
                # if H[i,j]==0: continue # no interaction
                # if qni['ef']!=qnj['ef']: continue
                # for statei in self.get_matching_states(name1,Σ=qni['Σ'],ef=qni['ef']):
                    # for statej in self.get_matching_states(name2,Σ=qnj['Σ'],ef=qni['ef']):
                        # self.interactions.append(Vibronic_Interaction(
                            # statei,statej,
                            # fH=_tools.lambdify_sympy_expression(H[i,j],('J',),{},parameter_set)))

    # def add_LS_coupling(
            # self,
            # name1,name2,
            # p=0,pD=0,           # spin-orbit interaction parameters
            # qn1=None,qn2=None,
            # verbose=False,
            # suboptimiser=None, # dictionary like, containing, p,pD   e.g., another optimiser
    # ):
        # """USE THIS OR USE AND ADD_ROTATIONAL_COUPLING OR USE ADD_COUPLING?""" 
        # verbose = verbose or self.verbose
        # ## take care of inputs
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # input_qn1,input_qn2 = copy(qn1),copy(qn2) # store before addition
        # qn1 = self._combine_quantum_numbers(name1,species=self.species,**qn1)
        # qn2 = self._combine_quantum_numbers(name2,species=self.species,**qn2)
        # ## Add parameter to optimisation unless externally provided,
        # ## in which case p_external_optimisation is dictionary-like
        # ## providing p,pD
        # parameter_set = self.add_parameter_set(
            # note=f'add_LS_coupling {encode_transition(**{key+"p":val for key,val in qn1.items()},**{key+"pp":val for key,val in qn2.items()})}',
            # p=p,pD=pD,step_default=dict(p=1e-2,pD=1e-4),) # add adjustable parameters to optimiser
        # self.format_input_functions.append(lambda name1=name1,name2=name2,qn1=input_qn1,qn2=input_qn2: f'{self.name}.add_LS_coupling( {repr(name1):15},{repr(name2):15},{parameter_set.format_input(neglect_fixed_zeros=True)},qn1={repr(qn1)},qn2={repr(qn2)})')
        # self._vibronic_interactions.append({'qn':join_upper_lower_quantum_numbers(qn1,qn2), 'parameter_set':parameter_set})
        # ## find states that interact
        # state1s,state2s = self.get_matching_states(**qn1),self.get_matching_states(**qn2) # find interacting states
        # if len(state1s)==0:
            # print(f'warning: No interacting state found: {repr(name1)}')
            # return
        # if len(state2s)==0:
            # print(f'warning: No interacting state found: {repr(name2)}')
            # return
        # ## get case (a) quantum numbers etc, and symbolic rotational and spin-orbit interaction matrices
        # casea,LSef = get_spin_orbit_coupling_matrix(qn1=qn1,qn2=qn2,verbose=False)
        # # if verbose:
            # # print( "Case (a) quantum numbers e/f parity basis:\n")
            # # for it,t in enumerate(casea['qnef'].iter_data_collection()):
                # # print( it+1,encode_level(**t))
            # # print( "\nHLS e/f parity basis:\n")
            # # pprint(LSef)
        # p,pD = sympy.Symbol("p"),sympy.Symbol("pD")
        # NNLSef = (LSef*casea['NNef']+casea['NNef']*LSef)/2
        # H = p*LSef + pD*NNLSef
        # ## monitor verbosely
        # if verbose:
            # print( )
            # print(self.format_input_functions[-1]())
            # print_matrix_elements(casea['qnef'],H,'LS')
        # ## lambdify symbolic function and insert fitting parameters
        # for i,qni in enumerate(casea['qnef']): # bra
            # for j,qnj in enumerate(casea['qnef']): # ket
                # if j<=i: continue                  # symmetric
                # if H[i,j]==0: continue # no interaction
                # if qni['ef']!=qnj['ef']: continue
                # for statei in self.get_matching_states(name1,Σ=qni['Σ'],ef=qni['ef']):
                    # for statej in self.get_matching_states(name2,Σ=qnj['Σ'],ef=qni['ef']):
                        # self.interactions.append(Vibronic_Interaction(
                            # statei,statej,
                            # fH=_tools.lambdify_sympy_expression(H[i,j],('J',),{},parameter_set)))

    # def add_coupling_function(
            # self,
            # name1,name2,
            # ef,Σ1,Σ2,
            # f,                  # function of (J,ef,Σ1,Σ2)
            # qn1=None,qn2=None,
    # ):
        # """Defining a coupling between particular (ef,Σ1,Σ2) levels
        # with an arbitrary function of J."""
        # ## take care of inputs
        # input_qn1,input_qn2 = copy(qn1),copy(qn2) # store before addition
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # qn1 = self._combine_quantum_numbers(name1,species=self.species,**qn1,ef=ef,Σ=Σ1)
        # qn2 = self._combine_quantum_numbers(name2,species=self.species,**qn2,ef=ef,Σ=Σ2)
        # for state1 in self.get_matching_states(**qn1):
            # for state2 in self.get_matching_states(**qn2):
                # self.interactions.append(Vibronic_Interaction(state1,state2,f))

    # def add_electrostatic_interaction(self,name1,name2,He=0,qn1=None,qn2=None):
        # """Add electrostatic intercation, the same for all (Σ,ef) pairs."""
        # p = self.add_parameter_set(name1+'-'+name2,He=He,step_default=dict(He=1e-2),) # add variable parameter to optimiser
        # ## make repr function
        # if qn1 is None: qn1 = {} # quantum numbers
        # if qn2 is None: qn2 = {} # quantum numbers
        # self.format_input_functions.append(lambda name1=name1,name2=name2,qn1=copy(qn1),qn2=copy(qn2): '\n    '.join([self.name+".add_electrostatic_interaction("+ repr(name1)+','+repr(name2)+',', str(p).replace('\n','\n    '), 'qn1='+repr(qn1)+', qn2='+repr(qn2)+',)']))
        # if self.verbose:
            # print( )
            # print(self.format_input_functions[-1]()) # monitor verbosely
        # ## get quantum numbers from name
        # qn1 = self._combine_quantum_numbers(name1,**qn1)
        # qn2 = self._combine_quantum_numbers(name2,**qn2)
        # assert qn1['Λ']==qn2['Λ'],'Selection rule violated: Λ1!=Λ2'
        # assert qn1['s']==qn2['s'],'Selection rule violated: s1!=s2'
        # assert qn1['S']==qn2['S'],'Selection rule violated: S1!=S2'
        # ## add interactions to optimisation
        # state1s,state2s = self.get_matching_states(**qn1),self.get_matching_states(**qn2) # find interacting states
        # assert len(state1s)>0 and len(state2s)>0, 'No interacting states found. name1='+repr(name1)+' name2='+repr(name2)+' qn1='+repr(qn1)+' qn2='+repr(qn2)
        # for state1,state2 in itertools.product(state1s,state2s):
            # if state1.qn['ef']!=state2.qn['ef']: continue # e ⟵/⟶ f
            # if state1.qn['Σ']!=state2.qn['Σ']: continue # ΔΣ=0
            # self.interactions.append(Vibronic_Interaction(state1,state2,fH=lambda J:p['He']))
               #  
    # def plot_levels(self,xkey='J',ykey='T',ax=None,reduce_coefficients=(0.,),plot_legend=False,annotate_lines= True,**plot_kwargs):
        # if ax is None: ax = plt.gca()
        # if len(self.level)==0: return
        # unique_labels = list(self.level.unique('label'))
        # lines = {'e':[],'f':[]}
        # for isublevel,(qn,sublevel) in enumerate(self.level.unique_sublevels()):
            # sublevel = deepcopy(sublevel)
            # for key in ('Σ','Ω','SR'):
                # if sublevel.is_known(key):
                    # qn[key] = sublevel[key][0]
            # label = encode_level(**qn)
            # iplot_kwargs = copy(plot_kwargs)
            # # iplot_kwargs.setdefault('color',my.newcolor(isublevel))
            # iplot_kwargs.setdefault('color',my.newcolor(unique_labels.index(sublevel['label'][0])))
            # iplot_kwargs.setdefault('mfc',iplot_kwargs['color'])
            # iplot_kwargs.setdefault('mec',iplot_kwargs['color'])
            # iplot_kwargs.setdefault('marker','')
            # iplot_kwargs.setdefault('markersize',10)
            # iplot_kwargs.setdefault('mew',1)
            # iplot_kwargs.setdefault('ls',('-' if sublevel['ef'][0]=='e' else ':'))
            # iplot_kwargs.setdefault('alpha',1)
            # # iplot_kwargs.setdefault('label',label+'_Σ='+format(sublevel['Σ'][0])+'_Ω='+format(sublevel['Ω'][0]))
            # iplot_kwargs.setdefault('label',label)
            # ## plot model
            # l = ax.plot(sublevel[xkey],
                        # sublevel[ykey] - np.polyval(reduce_coefficients,sublevel[xkey]*(sublevel[xkey]+1)),
                        # **iplot_kwargs)
            # lines[sublevel['ef'][0]].append(l)
            # ## plot experiment
            # if self.experimental_level is not None:
                # t = self.experimental_level.matches(
                    # species=sublevel[0]['species'],
                    # label=sublevel[0]['label'],
                    # v=sublevel[0]['v'],
                    # F=sublevel[0]['F'],
                    # ef=sublevel[0]['ef'],)
                # ax.errorbar(
                    # t[xkey],
                    # t[ykey]-np.polyval(reduce_coefficients,t[xkey]*(t[xkey]+1)),
                    # (t['d'+ykey] if t.is_known('d'+ykey) else np.zeros(len(t))),
                    # ls='',
                    # marker=('o' if sublevel[0]['ef']=='e' else 'x'),
                    # markersize=10,
                    # mew=1,
                    # color=iplot_kwargs['color'],
                    # # mfc=iplot_kwargs['color'],
                    # mfc='none',
                    # mec=iplot_kwargs['color'],
                    # label='',
                # )
        # ax.grid(True,color='grey')
        # if plot_legend: my.legend()
        # if annotate_lines:
            # my.annotate_line(line=lines['e'],xpos='max',ypos='center',fontsize='x-small',xoffset=10,yoffset=5)
            # my.annotate_line(line=lines['f'],xpos='max',ypos='center',fontsize='x-small',xoffset=10,yoffset=-5,alpha=0.5)
            # ax.set_xlim(max(-0.1,ax.get_xlim()[0]),ax.get_xlim()[1]+10)
               #  
    # def plot(
            # self,
            # fig=None,
            # Treduce=None,      # coeffiecient to reuced term values
            # Tresidual_ylim = None,
            # **limit_to_qn,
    # ):
        # ## prepare figures
        # if fig is None:
            # fig = plt.gcf()
        # ## construct or get up to date
        # if self.has_changed():
            # self.construct()
        # ## nothing to plot
        # if len(self.rotational_level)==0:
            # return
        # fig.clf()
        # axT = axTresidual = axΓ = axΓresidual = None
        # ## get a T reduction fucntion
        # if Treduce is None:
            # freduce = lambda J:0.
        # else:
            # freduce = lambda J:np.polyval(Treduce,J*(J+1))
        # t = self.rotational_level.matches(**limit_to_qn)
        # for isublevel,(qn,sublevel) in enumerate(t.unique_sublevels()):
            # ## plot model term values 
            # if axT is None: axT = my.subplot(fig=fig)
            # axT.plot(sublevel['J'],sublevel['T']-freduce(sublevel['J']),ls='-',marker='',label=encode_level(**qn),color=my.newcolor(isublevel))
            # ## plot experimental and residual term values
            # if self.experimental_level is not None and self.experimental_level.is_known('T'):
                # t = self.experimental_level.matches(**qn)
                # if t.is_known('dT'):
                    # axT.errorbar(t['J'],t['T']-freduce(t['J']),t['dT'],ls='',marker='o',mfc='none',markersize=7,color=my.newcolor(isublevel))
                # else:
                    # axT.plot(t['J'],t['T']-freduce(t['J']),ls='',marker='o',mfc='none',markersize=7,color=my.newcolor(isublevel))
                # if axTresidual is None: axTresidual = my.subplot(fig=fig)
                # ## axTresidual.plot(sublevel['J'],sublevel['Tresidual'],ls='-',marker='o',mfc='none',markersize=7,label=encode_level(**qn),color=my.newcolor(isublevel))
                # if t.is_known('dTresidual'):
                    # axTresidual.errorbar(sublevel['J'],sublevel['Tresidual'],sublevel['dTresidual'],ls='-',marker='o',mfc='none',markersize=7,label=encode_level(**qn),color=my.newcolor(isublevel))
                # else:
                    # axTresidual.plot(sublevel['J'],sublevel['Tresidual'],ls='-',marker='o',mfc='none',markersize=7,label=encode_level(**qn),color=my.newcolor(isublevel))
                # if Tresidual_ylim is not None:
                    # axTresidual.set_ylim(*Tresidual_ylim)
            # ## plot model linewidths
            # if np.any(sublevel['Γ']>0):
                # if axΓ is None: axΓ = my.subplot(fig=fig)
                # axΓ.plot(sublevel['J'],sublevel['Γ'],ls='-',marker='',label=encode_level(**qn),color=my.newcolor(isublevel))
                # ## experimental and residual linewidths
                # if self.experimental_level is not None and self.experimental_level.is_known('Γ'):
                    # t = self.experimental_level.matches(**qn)
                    # axΓ.errorbar(t['J'],t['Γ'],t['dΓ'],ls='',marker='o',mfc='none',markersize=7,color=my.newcolor(isublevel))
                    # axΓ.set_yscale('log')
                    # if axΓresidual is None: axΓresidual = my.subplot(fig=fig)
                    # axΓresidual.plot(sublevel['J'],sublevel['Γresidual'],ls='-',marker='o',mfc='none',markersize=7,label=encode_level(**qn),color=my.newcolor(isublevel))
        # ## end plot
        # my.legend(ax=fig.gca(),fontsize='small')

    # def get_vibrational_overlap(self):
        # """Calculate vibrational overlap ⟨vi|vj⟩ for all combinations of
        # levels."""
        # overlap = np.full((len(self.J),len(self.states),len(self.states)),np.nan)
        # for (i,statei),(j,statej) in itertools.product(enumerate(self.states), enumerate(self.states)):
            # if statei.wavefunction is None or statej.wavefunction is None: continue
            # for iJ,J in enumerate(self.J):
                # overlap[iJ,i,j] = integrate.trapz(statei.wavefunction[iJ]*statej.wavefunction[iJ],self.R)
        # return(overlap)

    # def print_vibrational_overlap(self,namei=None,namej=None,J=None):
        # """Print Franck-Condon factors."""
        # overlap = self.get_vibrational_overlap()
        # if J is None: J = self.J
        # for Ji in J:
            # iJ = self.J==Ji
            # for i,statei, in enumerate(self.states):
                # for j in range(i,len(self.states)):
                    # statej = self.states[j]
                    # if namei is not None and statei.name!=namei: continue
                    # if namej is not None and statej.name!=namej: continue
                    # if statei.qn['label']==statej.qn['label']: continue # not interested in factors within an electronic state
                    # if np.isnan(overlap[iJ,i,j]): continue
                    # print('{:30} {:30} {:10.3e}'.format(
                        # statei.name+'_J='+format(Ji,'g'),
                        # statej.name+'_J='+format(Ji,'g'),
                        # float(overlap[iJ,i,j])))
                       #  
    # def get_coefficients(self,J):
        # """Get coefficient matrix at a particular J"""
        # return(np.column_stack([s.c[:,s.Jf==J].squeeze() for s in self.states]))
       #  
    # def plot_coefficients(self,fig=None,**plot_states_with_matching_qn):
        # """Plots nice picture of mixing coefficients."""
        # if fig is None:
            # fig = plt.gcf()
            # fig.clf()
        # matching_states = [t for t in self.states
                           # if match_quantum_numbers(plot_states_with_matching_qn,t.qn)]
        # for i,statei in enumerate(matching_states):
            # ax = my.subplot(i)
            # # ax.set_title(statei.name+' Ω='+str(statei.qn['Ω']),fontsize='medium')
            # ax.set_title(encode_level(**statei.qn),fontsize='medium')
            # for ief,ef in enumerate(('e','f')):
                # iefmatch = my.find(np.array([t.qn['ef']==ef for t in self.states]))
                # for j in iefmatch:
                    # statej = self.states[j]
                    # ax.plot(
                        # self.J,
                        # self.eigvects[:,i,j]**2,
                        # marker=('o' if ef=='e' else 'x'),
                        # linestyle=('-' if ef=='e' else ':'),
                        # color=my.newcolor(self.states.index(statej)),
                        # mew=1, mfc='none',
                        # # label=statej.name+str(statei.qn['Ω'])+' Σ='+str(statei.qn['Σ']),)
                        # # label=statej.name,
                    # )
            # ax.grid(True,color='grey')
            # ax.set_ylim(1e-6,1)
            # ax.set_xlim(xmin=-0.5)
        # ## legend in final axes
        # # for i,state in enumerate(self.states):
        # for i,state in enumerate(matching_states):
            # ax.plot([],[],marker='o',ls='', color=my.newcolor(i),
                    # # label=state.name+' Ω='+str(state.qn['Ω']))
                    # label=encode_level(**state.qn))
        # my.legend_colored_text(ax=ax,fontsize='medium',loc='upper left')
        # ax.grid(False)
        # ax.xaxis.set_ticks([])
        # ax.yaxis.set_ticks([])
        # ax.set_frame_on(False)
        # return(fig)

    # def exchange_levels(self,J=None,iname='',jname='',**qn):
        # """Change assignemtn by exchanging levels. Give quantm numbers as
        # er..g, Λi=2, Σj=4. If not subscript, i.e., v=5, then applies to both
        # states"""
        # self._rotational_level.clear()
        # ## get quantum numbers
        # qni,qnj = decode_level(iname),decode_level(jname)
        # for key,val in qn.items():
            # if len(key)>1 and key[-1]=='i':
                # qni[key[:-1]] = val
            # elif len(key)>1 and key[-1]=='j':
                # qnj[key[:-1]] = val
            # else:
                # qni[key] = qnj[key] = val
        # if J is None:
            # iJ = range(len(self.J))
        # else:
            # iJ = my.inrange(self.J,my.ensure_iterable(J))
        # ## get states -- must be unique
        # i = self.states.index(self.get_state(iname,**qni))
        # j = self.states.index(self.get_state(jname,**qnj))
        # self.eigvals[iJ,i],self.eigvals[iJ,j] = self.eigvals[iJ,j],self.eigvals[iJ,i]
        # self.eigvects[iJ,i,:],self.eigvects[iJ,j,:] = self.eigvects[iJ,j,:],self.eigvects[iJ,i,:]

    # def set_exchange_levels(self,*args,**kwargs):
        # self.format_input_functions.append(f'{self.name}.set_exchange_levels({my.repr_args_kwargs(*args,**kwargs)})')
        # self.construct_functions.append(lambda: self.exchange_levels(*args,**kwargs))


@lru_cache
def _get_linear_H(S,Λ,s):
    """Compute symbolic and functional Hamiltonian for the spin manifold
    of a linear molecule."""
    ## symbolic variables, Note that expected value of ef is +1 or -1 for 'e' and 'f'
    p = {key:sympy.Symbol(key) for key in (
        'Tv','Bv','Dv','Hv','Av','ADv','λv','λDv','λHv', 'γv','γDv','ov','pv','pDv','qv','qDv')}
    J = sympy.Symbol('J')
    case_a = quantum_numbers.get_case_a_basis(Λ,s,S,print_output=False)
    efs = case_a['qnef']['ef']
    Σs = case_a['qnef']['Σ']
    NN = case_a['NNef']
    NS = case_a['NSef']
    ## construct some convenient matrices
    def anticommutate(X,Y):
        return(X*Y+Y*X)
    I  = sympy.eye(case_a['n']) # unit matrix
    ## Equation 18 of brown1979
    H = p['Tv']*I + p['Bv']*NN - p['Dv']*NN**2 + p['Hv']*NN**3       # T + BN**2 - DN**4 + HN**6
    if S>0:
        if Λ>0: H += anticommutate(p['Av']*I+p['ADv']*NN,sympy.diag(*[float(Λ*Σ) for Σ in Σs]))/2 # 1/2[A+AN2,LzSz]+
        H += (p['γv']*I+p['γDv']*NN)*NS # (γ+γD.N**2)N.S
        H += anticommutate(p['λv']*I+p['λDv']*NN+p['λHv']*NN**2,sympy.diag(*[float(Σ**2) for Σ in Σs])-I/3*S*(S+1)) # [λ+λD.N**2),Sz**2-1/3*S**2]+
    ## add Λ-doubling terms here, element-by-element.
    for i,(Σi,efi) in enumerate(zip(Σs,efs)):
        for j,(Σj,efj) in enumerate(zip(Σs,efs)):
            if efi!=efj: continue
            ef = (1 if efi=='e' else -1) # Here ef is +1 for 'e' and -1 for 'f'
            ## 1Π state
            if Λ>0 and S==0: 
                if efi=='f' and efj=='f': H[i,j] += p['qv']*J*(J+1)   # coefficient
            ## is ef=1 for e, ef=-1 for f 2Π states, amiot1981
            ## table II, there are more distortion constants,
            ## which I have not included here, but could be
            ## done. Perhaps these could be included with N-matrix
            ## multiplication?
            elif Λ==1 and S==0.5:
                for i,(Σi,efi) in enumerate(zip(Σs,efs)):
                    efi = (1 if efi=='e' else -1)
                    for j,(Σj,efj) in enumerate(zip(Σs,efs)):
                        efj = (1 if efj=='e' else -1)
                        if efi!=efj: continue
                        ## diagonal elseement for level 2 in amiot1981
                        if   Σi==-0.5 and Σj==-0.5:
                            H[i,j] += efi*( -0.5*(J+0.5)*p['pv']
                                             -(J+0.5)*p['qv']
                                             -0.5*(J+0.5)*((J-0.5)*(J+0.5)+2)*p['pDv']
                                             -0.5*(3*(J-0.5)*(J+0.5)+4)*(J+0.5)*p['qDv']    )
                        ## diagonal element for level 1 in amiot1981
                        elif Σi==+0.5 and Σj==+0.5:
                            H[i,j] += efi*( -0.5*(J-0.5)*(J+1.5)*(J+0.5)*p['qv']
                                             -0.5*(J-0.5)*(J+0.5)*(J+0.5)*p['qDv']  )
                        ## off-diagonal element
                        elif Σi==-0.5 and Σj==+0.5:
                            H[i,j] += efi*(  0.5*((J-0.5)*(J+1.5))**0.5*(J+0.5)*p['qv']
                                              + 0.25*((J-0.5)*(J+0.5))**0.5*(J+0.5)*p['pDv']
                                              +0.5*((J-0.5)*(J+0.5))**(0.5)*((J-0.5)*(J+0.5)+2)*(J+0.5)*p['qDv'] )
            ## 3Π states, from Table I brown_merer1979            
            elif Λ==1 and S==1:
                ## diagonal elements
                if   Σi==-1 and Σj==-1:
                    H[i,j] += -ef*(p['ov']+p['pv']+p['qv'])
                elif Σi== 0 and Σj== 0:
                    H[i,j] += -ef*p['qv']*J*(J+1)/2
                elif Σi==+1 and Σj==+1:
                    H[i,j] += 0
                ## off-diagonal elements
                elif Σi==-1 and Σj==+0:
                    H[i,j] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*ef
                    H[j,i] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*ef
                elif Σi==-1 and Σj==+1:
                    H[i,j] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*ef
                    H[j,i] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*ef
                elif Σi==+0 and Σj==+1:
                    pass # zero
            # else:
                # if 'ov' in kwargs or 'pv' in kwargs or 'qv' in kwargs:
                    # raise Exception("Cannot use ov, pv, or qv because Λ-doubling not specified for state with Λ="+repr(Λ)+" and S="+repr(S))
    ## simplify and print Hamiltonian if requested
    # if self.verbose:
        # print( )
        # print(self.format_input_functions[-1]())
        # print_matrix_elements(case_a['qnef'],H,'H')
    ## Convert elements of symbolic Hamiltonian into lambda functions
    fH = np.full(H.shape,None)
    for i in range(len(Σs)):
        for j in range(i,-1,-1):
            fH[i,j] = fH[j,i] = tools.lambdify_sympy_expression(
                H[i,j],'J',**{key:0 for key in p})
    return efs,Σs,H,fH


class VibLine(Optimiser):
    
    """Calculate and optimally fit the line strengths of a band between
    two states defined by LocalDeperturbation objects. Currently only
    for single-photon transitions. """

    def __init__(
            self,
            name,
            level_u,level_l,
            J_l=None,ΔJ=None,
            # # experimental_transition=None,
            # verbose=None,
    ):
        ## add upper and lower levels
        self.name = name
        self.level_u = level_u
        self.level_l = level_l
        self.species = self.level_l.species
        tkwargs = {
            'permit_auto_defaults':True, 
            'permit_nonprototyped_data':False,
        }
        self.rotational_line = lines.Diatomic(**tkwargs)
        self.vibrational_line = lines.Diatomic(**tkwargs)
        self.vibrational_spin_line = lines.Diatomic(**tkwargs)
        self.μ = None


        # self.vibrational_line.generate_from_levels(
            # self.level_u.vibrational_level,
            # self.level_l.vibrational_level)
        # self.vibrational_line['μv'] = 0

        # self.vibrational_spin_line.generate_from_levels(
            # self.level_u.vibrational_spin_level,
            # self.level_l.vibrational_spin_level)

        # assert statep.Tref==statepp.Tref, 'statep and statepp Tref do not match..'
        # self.Tref = statep.Tref
        ## determine a name
        # if name is None:
            # self.name = self.statep.name+'_'+self.statepp.name
        # else:
            # self.name = name
        # self.transition_moments = []
        # self._scalar_transition_moments = [] # stored for the benefit of get_vibrational_transition
        # self.experimental_transition = experimental_transition        # a Transition object for comparing model results to 
        # self.transition = Rotational_Transition(
            # Name=f'{self.name}.transition',
            # description='Constructed by an Interacting_Vibronic_Transition object.',
            # Tref=self.Tref,
            # partition_source = 'self',
        # )
        # self.transition.format_input_functions.clear() # delete the constructor line
        # self.rotational_transition = self.transition # alternative name
        # ## how much to print
        # self.verbose = (False if verbose is None else verbose)
        ## Decide on lower and upper state J values, and ΔJ transitions
        if J_l is not None: 
            self.J_l = np.array(J_l,ndmin=1)
        elif J_l is None and self.level_l.J is not None:
            self.J_l = self.level_l.J
        elif J_l is None and self.level_l.J is None:
            if self.level_u.states[0].qn['S']%1==0:
                self.J_l = np.arange(0,30)
            else:
                self.J_l = np.arange(0.5,30)
            self.level_l.J = self.J_l
        if ΔJ is None:  
            self.ΔJ = (-1,0,+1)
        else:
            self.ΔJ = np.array(ΔJ,ndmin=1)
        self.level_l.J = self.J_l
        self.level_u.J = np.unique(np.concatenate([self.J_l+ΔJ for ΔJ in self.ΔJ]))
        self.level_u.J = self.level_u.J[self.level_u.J>=0]
            
        # self.statep.J = np.unique(np.concatenate([self.Jpp+t for t in self.ΔJ]))
        # ## check for bad quantum numbers
        # assert self.statepp.states[0].qn['S']%1==self.statepp.J[0]%1,'Jpp should be / should not be half-integer.'
        # assert self.statep.states[0].qn['S']%1==self.statep.J[0]%1,'Jp should be / should not be half-integer.'
        # ## define R coordinate in case wavefunctions are used
        # assert np.all(self.statep.R==self.statepp.R),'statep and statepp must have identical R coordinate'
        # self.R = self.statep.R
        ## construct optimiser -- inheriting from states
        Optimiser.__init__(self,name=self.name)
        # self.format_input_functions = [] # no formatted input
        self.add_suboptimiser(self.level_u,self.level_l)
        self.add_construct_function(self._initialise_construct)
        self.add_post_construct_function(self._finalise_construct)

        self._μ = Dataset()
        # self.preconstruct_functions = []
        # self.construct_functions.append(self.construct_transition)
        # # self.construct_functions.append(self.get_residuals) 
        # self.output_to_directory_functions.append(lambda directory: self.transition.save_to_file(directory+'/Rotational_Transition.h5'))
        # self.output_to_directory_functions.append(lambda directory: self.get_vibrational_transition().save_to_file(directory+'/Vibrational_Transition.h5'))
        # # self.format_input_functions.append(f"\nfrom anh import *")
        # # self.format_input_functions.append(lambda: f"{self.name} = spectra.Interacting_Vibronic_Transition({self.statep.name},{self.statepp.name},name={repr(self.name)},Jpp={repr(self.Jpp if isinstance(self.Jpp,range) else list(self.Jpp))},ΔJ={repr(list(self.ΔJ))})")
        # def f():
            # retval = f"{self.name} = spectra.Interacting_Vibronic_Transition({self.statep.name},{self.statepp.name}"
            # if name is not None:
                # retval += f',name={repr(self.name)}'
            # if Jpp is not None:
                # retval += f',Jpp={repr(Jpp)}'
            # if ΔJ is not None:
                # retval += f',ΔJ={repr(ΔJ)}'
            # if experimental_transition is not None:
                # retval += f',experimental_transition={repr(experimental_transition.name)}'
            # if verbose is not None:
                # retval += f',verbose={repr(verbose)}'
            # return(retval+')')
        # self.format_input_functions.append(f)
        # self.transition.suboptimisers.append(self)
        # self._cache = {}

    # def _initialise_construct(self):
    #     self.μ = np.full((
    #         len(self.J_l),
    #         len(self.ΔJ),
    #         len(self.level_u.vibrational_spin_level),
    #         len(self.level_l.vibrational_spin_level)
    #     ),0.)
        

    # def _finalise_construct(self):
    #     ## initialise arrays
    #     Sij = np.full(self.μ.shape,0.)
    #     E_u = np.full(self.μ.shape,0.)
    #     E_l = np.full(self.μ.shape,0.)
    #     J_u = np.full(self.μ.shape,0.)
    #     J_l = np.full(self.μ.shape,0.)
    #     ## loop over all rotational transitions
    #     for (iJ_l,J_li),(iΔJ,ΔJi) in itertools.product(enumerate(self.J_l),enumerate(self.ΔJ)):
    #         ## get upper lower level indices
    #         J_ui = J_li+ΔJi
    #         iJ_u = tools.find(self.level_u.J==J_ui)
    #         iJ_l = tools.find(self.level_l.J==J_li)
    #         if len(iJ_u)==0: continue
    #         if len(iJ_l)==0: continue
    #         iJ_u,iJ_l = iJ_u[0],iJ_l[0]
    #         ## get mixing coefficients
    #         c_u  =  self.level_u.c[iJ_u ,:,:]
    #         c_l  =  self.level_l.c[iJ_l ,:,:]
    #         ## get mixed line strengths
    #         Sij[iJ_l,iΔJ,:,:] = np.dot(c_u,np.dot(self.μ[iJ_l,iΔJ,:,:],np.transpose(c_l)))**2
    #         ## get upper and lower energy levels
    #         E_u[iJ_l,iΔJ,:,:] = np.column_stack([self.level_u.E[iJ_u].real for t in range(Sij.shape[3])])
    #         E_l[iJ_l,iΔJ,:,:] = np.row_stack([self.level_l.E[iJ_l].real for t in range(Sij.shape[2])])
    #         J_u[iJ_l,iΔJ,:,:] = J_ui
    #         J_l[iJ_l,iΔJ,:,:] = J_li
    #     ## collect data in  rotational line object
    #     self.rotational_line['Sij'] = np.ravel(Sij)
    #     self.rotational_line['E_u'] = np.ravel(E_u)
    #     self.rotational_line['E_l'] = np.ravel(E_l)
    #     self.rotational_line['J_u'] = np.ravel(J_u)
    #     self.rotational_line['J_l'] = np.ravel(J_l)
    #     for key in self.vibrational_spin_line:
    #         if self.vibrational_spin_line.is_scalar(key):
    #             self.rotational_line[key] = self.vibrational_spin_line[key]
    #         else:
    #             self.rotational_line[key] = np.ravel(np.tile(self.vibrational_spin_line[key],len(self.J_l)*len(self.ΔJ)))
    #     # self.rotational_line.remove_match(Sij=0)

    def _initialise_construct(self):
        ## cache this stuff
        self.μ0 = np.full((
            len(self.J_l),
            len(self.ΔJ),
            len(self.level_u.vibrational_spin_level),
            len(self.level_l.vibrational_spin_level),),0.)

    def _finalise_construct(self):

        # ## initialise rotational_line and cache it
        # self.rotational_line.clear()
        # for iJ_l,J_l in enumerate(self.J_l):
            # for iΔJ,ΔJ in enumerate(self.ΔJ):
                # J_u = J_l + ΔJ
                # if J_u not in self.level_u.J:
                    # continue
                # qn_l = self.level_l.vibrational_spin_level[self.level_l.iallowed[J_l]].as_dict()
                # qn_u = self.level_u.vibrational_spin_level[self.level_u.iallowed[J_u],].as_dict()
                # self.rotational_line.extend(
                    # J_u=J_u, J_l=J_l,
                    # E_u=np.nan, E_l=np.nan, μ=np.nan,
                    # **{key+'_u':val for key,val in qn_u.items()},
                    # **{key+'_l':val for key,val in qn_l.items()},)

        # μ = np.full(self.μ0.shape,0.)
        nlines = 0
        ## could vectorise linalg with np.dot
        for iJ_l,J_l in enumerate(self.J_l):
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                J_u = J_l + ΔJ
                if J_u not in self.level_u.J:
                    continue
                μ0 = self.μ0[iJ_l,iΔJ,:,:]
                c_l = self.level_l.c[J_l]
                c_u = self.level_u.c[J_u]
                # iallowed_l = self.level_l.iallowed[J_l]
                # iallowed_u = self.level_u.iallowed[J_u]
                ## get mixed line strengths
                μ = np.dot(c_u,np.dot(μ0,np.transpose(c_l)))
                # μ = μ[iallowed_u,:][:,iallowed_l]

                # iallowed_l = self.level_l.iallowed[J_l]
                # iallowed_u = self.level_u.iallowed[J_u]
                # nallowed_l = np.sum(iallowed_l)
                # nallowed_u = np.sum(iallowed_u)
                # kw_l = self.level_l.vibrational_spin_level[iallowed_l].as_dict()
                # kw_u = self.level_u.vibrational_spin_level[iallowed_u].as_dict()
                # kw_l['E'] = self.level_l.E[J_l][iallowed_l]
                # kw_u['E'] = self.level_u.E[J_u][iallowed_u]
                kw_l = self.level_l.vibrational_spin_level.as_dict()
                kw_u = self.level_u.vibrational_spin_level.as_dict()
                kw_l['E'] = self.level_l.E[J_l]
                kw_u['E'] = self.level_u.E[J_u]
                self.rotational_line.extend(
                    J_l=J_l,J_u=J_u,
                    μ=np.ravel(μ),
                    **{key+'_u':np.tile(val,len(self.level_l.vibrational_spin_level)) for key,val in kw_u.items()},
                    **{key+'_l':np.repeat(val,len(self.level_u.vibrational_spin_level)) for key,val in kw_l.items()},
                )

                
       
    @auto_construct_method('add_transition_moment')
    def add_transition_moment(self,name=None,μv=1,**extra_qn):
        """Add constant transition moment. μv can be optimised."""
        """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        transitions. μv should be in atomic units and can be specifed
        as a value (optimisable), a function of R or a suboptimiser
        given ['μ']."""
        ## get all quantum numbers
        qn = quantum_numbers.decode_transition(name)
        qn.update(extra_qn)
        self.vibrational_line.append(μv=μv,**qn)
        qn_u,qn_l = quantum_numbers.separate_upper_lower_quantum_numbers(qn)
        ## get transition moment functions for all ef/Σ
        ## combinations and add optimisable parameter to functions
        ef_u,Σ_u,ef_l,Σ_l,fμ = _get_linear_transition_moment(qn['S_u'],qn['Λ_u'],qn['s_u'], qn['S_l'],qn['Λ_l'],qn['s_l'],)
        for i in range(fμ.shape[0]):
            for j in range(fμ.shape[1]):
                if fμ[i,j] is not None:
                    fμ[i,j]=lambda J,ΔJ,f=fμ[i,j]: f(J,ΔJ)*μv
        ## compute μ0
        cache = {}
        def construct_function():
            for i,(ef_ui,Σ_ui) in enumerate(zip(ef_u,Σ_u)):
                for j,(ef_li,Σ_li) in enumerate(zip(ef_l,Σ_l)):
                    row_u,i_u = self.level_u.vibrational_spin_level.matching_row(ef=ef_ui,Σ=Σ_ui,**qn_u,return_index=True)
                    row_l,i_l = self.level_l.vibrational_spin_level.matching_row(ef=ef_li,Σ=Σ_li,**qn_l,return_index=True)
                    if fμ[i,j] is None:
                        continue
                    for iΔJ,ΔJ in enumerate(self.ΔJ):
                        self.μ0[:,iΔJ,i_u,i_l] = fμ[i,j](self.J_l,ΔJ)
        return construct_function

    # def add_transition_moment(
            # self,
            # namep,              # name to match upper state too, also used to augment quantum numbers
            # namepp,             # lower state
            # qnp=None,           # upper states matched to these quantum numbers
            # qnpp=None,          # lower state
            # μ=None,             # a constant value of Parameter input arguments or an Electronic_Transition_Moment
            # verbose=False,
    # ):
        # """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        # transitions. μ should be in atomic units and can be specifed
        # as a value (optimisable), a function of R or a suboptimiser
        # given ['μ']."""
        # print('warning: add_transition_moment is being deprecated, perhaps use add_transition_moment_constant?')
        # ## get full set of quantum numbesr
        # if qnp  is None: qnp = {} 
        # if qnpp is None: qnpp = {}
        # qnpin = qnp; qnp = copy(qnpin)
        # qnppin = qnpp; qnpp = copy(qnppin)
        # for key,val in  decode_level(namep).items():
            # qnp.setdefault(key,val) 
        # for key,val in decode_level(namepp).items():
            # qnpp.setdefault(key,val)
        # qnp['species']  = self.statep.species
        # qnpp['species'] = self.statepp.species
        # ## Decide what μ is. Ultimately a dicationary-like variable p must be created
        # ## indexable with element  p['μ'] that is a scalar or vector
        # ## transition moment. 
        # ##
        # ## An Electronic_Transition_Moment, callable as an R-dependent
        # ## function
        # if isinstance(μ,Electronic_Transition_Moment):
            # p = {'μ':None}
            # def f(p=p):
                # p['μ'] = μ(self.R)
            # μ.add_construct(f)
            # self.suboptimisers.append(μ)
        # ## A function of R -- not optimisable, for tha use an
        # ## Electronic_Transition_Moment
        # elif my.isfunction(μ):
            # p = {'μ':μ(self.R)}
        # ## a regular scalar parameter (or Optimised_Parameter)
        # else:
            # p = self.add_parameter_set(μ=μ,note=f'add_transition_moment {encode_transition(qnp,qnpp)}')
            # μ = p               # back reference for format_input_functions
            # self._scalar_transition_moments.append(
                # {'qn':join_upper_lower_quantum_numbers(qnp,qnpp), 'parameter_set':p})
        # ## format input function
        # def f(μ=μ,qnp=copy(qnp),qnpp=copy(qnpp)):
            # retval = f'{self.name}.add_transition_moment({repr(namep):15},{repr(namepp):15},'
            # if qnpin is not None and len(qnpin)>0:
                # retval += f'qnp={repr(qnpin)},'
            # if qnppin is not None and len(qnppin)>0:
                # retval += f'qnpp={repr(qnppin)},'
            # if μ is not None:
                # if isinstance(μ,Electronic_Transition_Moment):
                    # retval += f'μ={μ.name},'
                # elif my.isfunction(μ):
                    # retval += f'μ={repr(μ)},'
                # else:
                    # retval += f'{μ.format_input()},'
            # retval += ')'
            # return(retval)
        # self.format_input_functions.append(f)
        # ## check some selection rules
        # if ((qnp['Λ']==0 and qnpp['Λ']==0 and qnp['s']!=qnpp['s'])
            # or (np.abs(qnp['Λ']-qnpp['Λ'])>1)
            # or (qnpp['S']!=qnp['S'])):
            # raise Exception(f"Forbidden transition: {repr(namep)} to {repr(namepp)}")
        # ## Get signed and e/f parity quantum numbers and transformation matrices
        # caseap  = get_case_a_basis( qnp['Λ'], qnp['s'], qnp['S']) 
        # caseapp = get_case_a_basis(qnpp['Λ'],qnpp['s'],qnpp['S'])
        # Mefp  = np.array( caseap['Mef'].evalf())
        # Mefpp = np.array(caseapp['Mef'].evalf())
        # ## for each ΔΩ e/f transition compute the contribution signed
        # ## ΔΩ transitions. Previously I did this symbolically using
        # ## formulae for Honl-London factors, but I could never get
        # ## fully consistent phase factors (still can't really) and so
        # ## I rewrote this using Wigner-3J coefficients (see
        # ## hansson2005) which are solved numerically. This means
        # ## rather than merely computing symbolic signed-Ω transition
        # ## moemnt and then using matrix multiplication to get e/f
        # ## parity moement the quadruple-loop below and fucntion
        # ## addition below is needed.
        # fM = {}                 # dictionary of (Σp,efp,Σpp,efpp) transition moments
        # for (ip,qnpef),(ipp,qnppef) in itertools.product(
                # enumerate(caseap['qnef']), enumerate(caseapp['qnef']),):
            # ## loop over signed-Ω combinations of upper and lower
            # ## states and compute a transition moment functions for
            # ## each
            # fsigned = []            
            # for (jp,qnppm),(jpp,qnpppm) in itertools.product(
                    # enumerate(caseap['qnpm']),enumerate(caseapp['qnpm'])):
                # if qnppm['Σ']!=qnpppm['Σ']: continue # not allowed!
                # c = Mefp[ip,jp]*Mefpp[ipp,jpp] # compute coefficient of thes signed-Ω wavefunctions to their respective ef-states
                # if c==0: continue
                # ## compute change in sign if reversed transition moment.
                # μsign = +1
                # if (qnppm['Λ']+qnpppm['Λ']) < 0: μsign = caseap['σvpm']*caseapp['σvpm']
                # if (qnppm['Λ']+qnpppm['Λ']) == 0 and (qnppm['Σ']+qnpppm['Σ']) < 0:
                    # μsign = caseap['σvpm']*caseapp['σvpm']
                # ## this computes every part of the linestrength apart
                # ## from the adjustable transition moment, with a cache
                # ## for speed
                # @functools.lru_cache(maxsize=4096)
                # def fsignedi_scalar_no_μ(Jpp,ΔJ, Ωp=qnppm['Ω'],Ωpp=qnpppm['Ω'], Λp=qnppm['Λ'],Λpp=qnpppm['Λ'], μsign=float(μsign), c=float(c)):
                    # return(
                        # c       # contribution to ef-basis state
                        # *np.sqrt((2*Jpp+1)*(2*(Jpp+ΔJ)+1)) # see hansson2005 eq. 13
                        # *(-1)**(Jpp+ΔJ-Ωp)  # phase factor --see hansson2005 eq. 13
                        # *(-1 if Λp==0 else 1)   # phase factor, a hack that should be understood
                        # *μsign # transition moment phase factor (+1 or -1)
                        # *wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj') # Wigner 3J line strength factor vectorised over Jpp
                    # )
                # ## ## add the transiton moment and vectorise over Jpp
                # ## def fsignedi(Jpp,ΔJ,fsignedi_scalar_no_μ=fsignedi_scalar_no_μ):
                # ##     return(np.array([fsignedi_scalar_no_μ(Jppi,ΔJ)*p['μ'] for Jppi in my.ensure_iterable(Jpp)]))
                # ## add the transiton moment
                # def fsignedi(Jpp,ΔJ,fsignedi_scalar_no_μ=fsignedi_scalar_no_μ):
                    # return fsignedi_scalar_no_μ(Jpp,ΔJ)*p['μ']
                # fsigned.append(fsignedi)
            # ## sum over all sign combinations and save this
            # ## Σp,efp,Σpp,efpp transition moment
            # fM[qnpef['Σ'],qnpef['ef'],qnppef['Σ'],qnppef['ef'],] = lambda Jpp,ΔJ,fsigned=fsigned: np.sum([f(Jpp,ΔJ) for f in fsigned],axis=0)
        # ## for all matching transition states find the corret
        # ## transition moment, if their is one
        # stateps = self.statep.get_matching_states(**qnp)
        # statepps = self.statepp.get_matching_states(**qnpp)
        # for tstates,tqn in ((stateps,qnp),(statepps,qnpp)):
            # if len(tstates)==0:
                # print(f'warning: add_transition_moment: No states found matching {repr(tqn)}')
        # for statep,statepp in itertools.product(stateps, statepps):
            # qns = (statep.qn['Σ'],statep.qn['ef'],statepp.qn['Σ'],statepp.qn['ef'])
            # if qns not in fM: continue
            # self.transition_moments.append(Vibronic_Transition_Moment(statep,statepp,fM[qns]))
    # def add_transition_moment(
            # self,
            # namep,              # name to match upper state too, also used to augment quantum numbers
            # namepp,             # lower state
            # qnp=None,           # upper states matched to these quantum numbers
            # qnpp=None,          # lower state
            # μ=None,             # a constant value of Parameter input arguments or an Electronic_Transition_Moment
            # verbose=False,
    # ):
        # """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        # transitions. μ should be in atomic units and can be specifed
        # as a value (optimisable), a function of R or a suboptimiser
        # given ['μ']."""
        # ## get full set of quantum numbesr
        # if qnp  is None: qnp = {} 
        # if qnpp is None: qnpp = {}
        # qnpin = qnp; qnp = copy(qnpin)
        # qnppin = qnpp; qnpp = copy(qnppin)
        # for key,val in  decode_level(namep).items():
            # qnp.setdefault(key,val) 
        # for key,val in decode_level(namepp).items():
            # qnpp.setdefault(key,val)
        # qnp['species']  = self.statep.species
        # qnpp['species'] = self.statepp.species
        # ## Decide what μ is. Ultimately a dicationary-like variable p must be created
        # ## indexable with element  p['μ'] that is a scalar or vector
        # ## transition moment. 
        # ##
        # ## An Electronic_Transition_Moment, callable as an R-dependent
        # ## function
        # if isinstance(μ,Electronic_Transition_Moment):
            # p = {'μ':None}
            # def f(p=p):
                # p['μ'] = μ(self.R)
            # μ.add_construct(f)
            # self.suboptimisers.append(μ)
        # ## A function of R -- not optimisable, for tha use an
        # ## Electronic_Transition_Moment
        # elif my.isfunction(μ):
            # p = {'μ':μ(self.R)}
        # ## a regular scalar parameter (or Optimised_Parameter)
        # else:
            # p = self.add_parameter_set(μ=μ,note=f'add_transition_moment {encode_transition(qnp,qnpp)}')
            # μ = p               # back reference for format_input_functions
            # self._scalar_transition_moments.append(
                # {'qn':join_upper_lower_quantum_numbers(qnp,qnpp), 'parameter_set':p})
        # ## format input function
        # def f(μ=μ,qnp=copy(qnp),qnpp=copy(qnpp)):
            # retval = f'{self.name}.add_transition_moment({repr(namep):15},{repr(namepp):15},'
            # if qnpin is not None and len(qnpin)>0:
                # retval += f'qnp={repr(qnpin)},'
            # if qnppin is not None and len(qnppin)>0:
                # retval += f'qnpp={repr(qnppin)},'
            # if μ is not None:
                # if isinstance(μ,Electronic_Transition_Moment):
                    # retval += f'μ={μ.name},'
                # elif my.isfunction(μ):
                    # retval += f'μ={repr(μ)},'
                # else:
                    # retval += f'{μ.format_input()},'
            # retval += ')'
            # return(retval)
        # self.format_input_functions.append(f)
        # ## check some selection rules
        # if ((qnp['Λ']==0 and qnpp['Λ']==0 and qnp['s']!=qnpp['s'])
            # or (np.abs(qnp['Λ']-qnpp['Λ'])>1)
            # or (qnpp['S']!=qnp['S'])):
            # raise Exception(f"Forbidden transition: {repr(namep)} to {repr(namepp)}")
        # ## Get signed and e/f parity quantum numbers and transformation matrices
        # caseap  = get_case_a_basis( qnp['Λ'], qnp['s'], qnp['S']) 
        # caseapp = get_case_a_basis(qnpp['Λ'],qnpp['s'],qnpp['S'])
        # Mefp  = np.array( caseap['Mef'].evalf())
        # Mefpp = np.array(caseapp['Mef'].evalf())
        # ## for each ΔΩ e/f transition compute the contribution signed
        # ## ΔΩ transitions. Previously I did this symbolically using
        # ## formulae for Honl-London factors, but I could never get
        # ## fully consistent phase factors (still can't really) and so
        # ## I rewrote this using Wigner-3J coefficients (see
        # ## hansson2005) which are solved numerically. This means
        # ## rather than merely computing symbolic signed-Ω transition
        # ## moemnt and then using matrix multiplication to get e/f
        # ## parity moement the quadruple-loop below and fucntion
        # ## addition below is needed.
        # fM = {}                 # dictionary of (Σp,efp,Σpp,efpp) transition moments
        # for (ip,qnpef),(ipp,qnppef) in itertools.product(
                # enumerate(caseap['qnef']), enumerate(caseapp['qnef']),):
            # ## loop over signed-Ω combinations of upper and lower
            # ## states and compute a transition moment functions for
            # ## each
            # fsigned = []            
            # for (jp,qnppm),(jpp,qnpppm) in itertools.product(
                    # enumerate(caseap['qnpm']),enumerate(caseapp['qnpm'])):
                # if qnppm['Σ']!=qnpppm['Σ']: continue # not allowed!
                # c = Mefp[ip,jp]*Mefpp[ipp,jpp] # compute coefficient of thes signed-Ω wavefunctions to their respective ef-states
                # if c==0: continue
                # ## compute change in sign if reversed transition moment.
                # μsign = +1
                # if (qnppm['Λ']+qnpppm['Λ']) < 0: μsign = caseap['σvpm']*caseapp['σvpm']
                # if (qnppm['Λ']+qnpppm['Λ']) == 0 and (qnppm['Σ']+qnpppm['Σ']) < 0:
                    # μsign = caseap['σvpm']*caseapp['σvpm']
                # ## this computes every part of the linestrength apart
                # ## from the adjustable transition moment, with a cache
                # ## for speed
                # @functools.lru_cache(maxsize=4096)
                # def fsignedi_scalar_no_μ(Jpp,ΔJ, Ωp=qnppm['Ω'],Ωpp=qnpppm['Ω'], Λp=qnppm['Λ'],Λpp=qnpppm['Λ'], μsign=float(μsign), c=float(c)):
                    # return(
                        # c       # contribution to ef-basis state
                        # *np.sqrt((2*Jpp+1)*(2*(Jpp+ΔJ)+1)) # see hansson2005 eq. 13
                        # *(-1)**(Jpp+ΔJ-Ωp)  # phase factor --see hansson2005 eq. 13
                        # *(-1 if Λp==0 else 1)   # phase factor, a hack that should be understood
                        # *μsign # transition moment phase factor (+1 or -1)
                        # *wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj') # Wigner 3J line strength factor vectorised over Jpp
                    # )
                # ## ## add the transiton moment and vectorise over Jpp
                # ## def fsignedi(Jpp,ΔJ,fsignedi_scalar_no_μ=fsignedi_scalar_no_μ):
                # ##     return(np.array([fsignedi_scalar_no_μ(Jppi,ΔJ)*p['μ'] for Jppi in my.ensure_iterable(Jpp)]))
                # ## add the transiton moment
                # def fsignedi(Jpp,ΔJ,fsignedi_scalar_no_μ=fsignedi_scalar_no_μ):
                    # return fsignedi_scalar_no_μ(Jpp,ΔJ)*p['μ']
                # fsigned.append(fsignedi)
            # ## sum over all sign combinations and save this
            # ## Σp,efp,Σpp,efpp transition moment
            # fM[qnpef['Σ'],qnpef['ef'],qnppef['Σ'],qnppef['ef'],] = lambda Jpp,ΔJ,fsigned=fsigned: np.sum([f(Jpp,ΔJ) for f in fsigned],axis=0)
        # ## for all matching transition states find the corret
        # ## transition moment, if their is one
        # stateps = self.statep.get_matching_states(**qnp)
        # statepps = self.statepp.get_matching_states(**qnpp)
        # for tstates,tqn in ((stateps,qnp),(statepps,qnpp)):
            # if len(tstates)==0:
                # print(f'warning: add_transition_moment: No states found matching {repr(tqn)}')
        # for statep,statepp in itertools.product(stateps, statepps):
            # qns = (statep.qn['Σ'],statep.qn['ef'],statepp.qn['Σ'],statepp.qn['ef'])
            # if qns not in fM: continue
            # self.transition_moments.append(Vibronic_Transition_Moment(statep,statepp,fM[qns]))


    # def add_transition_moment_Jpp_spline(
            # self,
            # namep,namepp,
            # *Jpp_μ_spline_points, # e.g., (0,0.1),(5,0.2),(20,(0.5,True))
            # qnp=None,qnpp=None,
            # order=3,            # spline order
            # ΔJ=None             # which ΔJ to apply this to, None for all
    # ):
        # """Add transition moment as a spline function of Jpp and
        # independent of ΔJ. Specified μ can be optimised."""
        # ## make adjusted μ into Parameters
        # Jpp_μ_spline_points = [
            # [Jpp, (self.add_parameter('μ',*μ,note=f'add_transition_moment_Jpp_spline Jpp={Jpp}')
                   # if my.isiterable(μ) else μ)]
            # for Jpp,μ in Jpp_μ_spline_points]
        # ## make a functional form of μ
        # ΔJ_to_include = (None if ΔJ is None else my.ensure_iterable(ΔJ))
        # def μf(Jpp,ΔJ):
            # if ΔJ_to_include is not None and ΔJ not in ΔJ_to_include:
                # return 0
            # else:
                # return my.spline(
                    # [Jpp for Jpp,μ in Jpp_μ_spline_points],
                    # [float(μ) for Jpp,μ in Jpp_μ_spline_points],
                    # Jpp,order=order)
        # ## format input function
        # def f(qnp=copy(qnp),qnpp=copy(qnpp)):
            # retval = [f'{self.name}.add_transition_moment_Jpp_spline({repr(namep):15},{repr(namepp):15}']
            # retval.extend([repr(t) for t in Jpp_μ_spline_points])
            # if qnp is not None:
                # retval.append(f'qnp={repr(qnp)}')
            # if qnpp is not None:
                # retval.append(f'qnpp={repr(qnpp)}')
            # if ΔJ is not None:
                # retval.append(f'ΔJ={repr(ΔJ)}')
            # return ','.join(retval)+')'
        # self.format_input_functions.append(f)
        # ## implement transition
        # self._add_transition_moment_internal(namep,namepp,μf,qnp,qnpp)


    # def get_transition_moment(self,statep,statepp):
        # for t in self.transition_moments:
            # if t.statep==statep and t.statepp==statep:
                # return(t)
        # else: return(None)    

    # def get_residuals(self):
        # """Get list of integrated cross section differences."""
        # # raise ImplementationError()
        # if self.experimental_transition is None: return(None)
        # i,j = spectra.find_common_transitions_levels(self.transition,self.experimental_transition)
        # # return(np.log(self.transition['σ'][i]/self.experimental_transition['σ'][j]))
        # # t = np.log2(self.transition['Sij'][i]/self.experimental_transition['Sij'][j])
        # # return(np.log2(self.transition['Sij'][i]/self.experimental_transition['Sij'][j]))
        # return(self.experimental_transition['ν'][j]-self.transition['ν'][i])
        # # return(t)
        # # return(self.transition['Sij'][i]-self.experimental_transition['Sij'][j])

    # def construct_transition(
            # self,
            # # remove_lines_fractionally_smaller_than=1e-10,
    # ):
        # """Calculate line strengths from mixing of zeroth-order
        # transition moments -- only works for one-photon absorption."""
        # for f in self.preconstruct_functions: f()
        # Jpp,ΔJ = np.array(self.Jpp),np.array(self.ΔJ)
        # ## build unmixed M indexed by [statep,statepp,efpp,Jpp,ΔJ]
        # if self.verbose:
            # print( )
            # print('constructing transition moments')
        # M = np.zeros((len(self.statep.states),len(self.statepp.states),len(Jpp),len(ΔJ))) 
        # for t in self.transition_moments:
            # M[self.statep.states.index(t.statep),self.statepp.states.index(t.statepp),:,:] += t.construct(Jpp,ΔJ)
        # ## determine all mixed transitions and save in a Transition object
        # if self.verbose:
            # print( 'building transition')
        # Tp,Tpp = np.full(M.shape,np.nan,dtype=float),np.full(M.shape,np.nan,dtype=float) 
        # Γp,Γpp = np.full(M.shape,np.nan,dtype=float),np.full(M.shape,np.nan,dtype=float) 
        # Sij = np.zeros(M.shape)
        # for (iJpp,Jppi),(iΔJ,ΔJi) in itertools.product(enumerate(Jpp),enumerate(ΔJ)):
            # # if self.verbose and iΔJ==0 and iJpp%10==0: print(f'Jppi = {Jppi}, Jppi max = {np.max(Jpp)}')
            # ## determine other transition quantum numbers
            # Jpi = Jppi+ΔJi
            # iJp = my.find(self.statep.J==Jpi)
            # if len(iJp)==0: continue
            # iJp = int(iJp)
            # ## determine which transition connect allowed levels
            # eigvalp  =  self.statep.eigvals[ iJp,:] # relevant upper state energy levels computed above
            # eigvalpp = self.statepp.eigvals[iJpp,:] # relevant lower state energy levels computed above
            # ivalidp,ivalidpp = my.find(~np.isnan(eigvalp)),my.find(~np.isnan(eigvalpp)) # NaN energy levels indicate levels not allowed
            # eigvalp,eigvalpp = eigvalp[ivalidp],eigvalpp[ivalidpp] # only compute transitions between allowed levels
            # if len(ivalidp)==0 or len(ivalidpp)==0: continue       # no valid levels for any transition to occur between
            # ## get zeroth-order matrix and mixing coefficients
            # Mi = M[:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)]
            # cp  =  self.statep.eigvects[iJp ,:,:][ivalidp ,:][:,ivalidp ]  
            # cpp = self.statepp.eigvects[iJpp,:,:][ivalidpp,:][:,ivalidpp]
            # ## get mixed intensities
            # Mi = np.dot(cp,np.dot(Mi,np.transpose(cpp)))
            # ## get arrays of transition data -- all expanded to full
            # ## size of all data -- efficient cpu, more memory, its
            # ## actually pretty hard to do this efficiently
            # Sij[:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = Mi**2
            # t = my.repmat_vector( eigvalp,(len(eigvalpp),),0)
            # Tp [:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = t.real
            # Γp [:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = t.imag
            # t = my.repmat_vector(eigvalpp,(len( eigvalp),),1)
            # Tpp[:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = t.real
            # Γpp[:,:,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = t.imag
            # ## Sij[:,:,iefpp,iJpp,iΔJ][np.ix_(ivalidp,ivalidpp)] = Mi**2
            # ## Sij[:,:,iefpp,iJpp,iΔJ][ivalidp,:][:,ivalidpp] = Mi**2
            # ## t = my.repmat_vector( eigvalp,(len(eigvalpp),),0)
            # ## Tp [:,:,iefpp,iJpp,iΔJ][ivalidp,:][:,ivalidpp] = t.real
            # ## Γp [:,:,iefpp,iJpp,iΔJ][ivalidp,:][:,ivalidpp] = t.imag
            # ## t = my.repmat_vector(eigvalpp,(len( eigvalp),),1)
            # ## Tpp[:,:,iefpp,iJpp,iΔJ][ivalidp,:][:,ivalidpp] = t.real
            # ## Γpp[:,:,iefpp,iJpp,iΔJ][ivalidp,:][:,ivalidpp] = t.imag
        # ## collect all new data, unfortunately the order of qn etc has probably changed
        # new_data = dict(
            # Sij=np.ravel(Sij),
            # Tp=np.ravel(Tp),
            # Tpp=np.ravel(Tpp),
            # Γp=np.ravel(Γp),
            # Γpp=np.ravel(Γpp),
            # Jpp=np.ravel(my.repmat_vector(Jpp,(len(self.statep.states),len(self.statepp.states),len(ΔJ)),2)),
            # ΔJ=np.ravel(my.repmat_vector(ΔJ,  (len(self.statep.states),len(self.statepp.states),len(Jpp)),3)),
        # )
        # ## first run compute all quantum numbers and append to Transition. Afterards just insert new data.
        # if len(self.transition)==0:
            # ## limt to certain strength
            # ## ikeep = new_data['Sij']>(new_data['Sij'].max()*remove_lines_fractionally_smaller_than)
            # ikeep = new_data['Sij']>0
            # ## expand all other quantum numbers into full size array data
            # qnp = {}
            # for key,val in self.statep.states[0].qn.items():
                # qnp[key+'p'] = np.zeros(Sij.shape,dtype=Rotational_Level._class_vector_data[key].dtype)
                # for i,state in enumerate(self.statep.states):
                    # qnp[key+'p'][i,:,:,:] = state.qn[key]
                # qnp[key+'p'] = np.ravel(qnp[key+'p'])
            # qnpp = {}
            # for key,val in self.statepp.states[0].qn.items():
                # qnpp[key+'pp'] = np.zeros(Sij.shape,dtype=Rotational_Level._class_vector_data[key].dtype)
                # for i,state in enumerate(self.statepp.states):
                    # qnpp[key+'pp'][:,i,:,:] = state.qn[key]
                # qnpp[key+'pp'] = np.ravel(qnpp[key+'pp'])
            # ## put new data in transition
            # self.transition.clear(clear_scalar_data=False)
            # self.transition.append( 
                # **{key:val[ikeep] for key,val in new_data.items()}, 
                # **{key:val[ikeep] for key,val in qnp.items()}, 
                # **{key:val[ikeep] for key,val in qnpp.items()}, 
            # )
            # self._cache['construct_transition_ikeep'] = ikeep
            # self._cache['construct_transition_qnp']   = qnp
            # self._cache['construct_transition_qnpp']  = qnpp
        # else:
            # ikeep = self._cache['construct_transition_ikeep']
            # qnp   = self._cache['construct_transition_qnp']
            # qnpp  = self._cache['construct_transition_qnpp']
            # for t in (new_data,qnp,qnpp):
                # for key,val in t.items():
                    # self.transition[key] = val[ikeep]
        # ## compute resiudals
        # if self.experimental_transition is not None:
            # return(self.get_residuals())
        # if self.verbose:
            # print('number of lines computed:',len(self.transition))

    # def get_vibrational_transition(self):
        # """Get a Vibrational_Transition containing transition quantum
        # numbers and transition moment."""
        # ## intialise transition object
        # retval = Vibrational_Transition(
            # description='Constants and transition moment of two vibronic levels.',Tref=self.Tref)
        # ## add row for each transition moment
        # for transition_moment in self._scalar_transition_moments:
            # qn = transition_moment['qn']
            # μ = transition_moment['parameter_set'].p['μ']
            # dμ = transition_moment['parameter_set'].dp['μ']
            # ## find relevant state spin_manifolds if they exist
            # qnp,qnpp = separate_upper_lower_quantum_numbers(qn)
            # parametersp,parameterspp = {},{}
            # for t in self.statep._vibronic_spin_manifolds:
                # if match_quantum_numbers(qnp,t['qn']):
                    # for name,p in t['parameter_set'].items():
                        # parametersp[name+'vp'] = p.p
                        # parametersp['d'+name+'vp'] = p.dp
            # for t in self.statepp._vibronic_spin_manifolds:
                # if match_quantum_numbers(qnpp,t['qn']):
                    # for name,p in t['parameter_set'].items():
                        # parametersp[name+'vpp'] = p.p
                        # parametersp['d'+name+'vpp'] = p.dp
            # ## add line data
            # keys_vals = dict(**qn,**parametersp,**parameterspp,μ=μ,dμ=dμ)
            # retval.set_default(*keys_vals)
            # retval.append(**keys_vals)
        # return(retval)

    # vibrational_transition = property(get_vibrational_transition)

    # def plot(self):
        # """Convenience quick plot."""
        # my.fig(1)
        # self.plot_transitions()
        # my.fig(2)
        # self.plot_spectrum()

    # def plot_transitions(self,ykey='Sij',xkey='Jpp',fig=None,**plot_kwargs):
        # """Plot linestrengths or fvalues. Uses a whole figure."""
        # if fig is None: fig = plt.gcf()
        # fig.clf()
        # for i0,(d0,t0) in enumerate(self.transition.unique_dicts_matches('labelp','labelpp','vp','vpp')):
            # ax = my.subplot(i0)
            # my.annotate_corner(t0['labelp'][0]+'('+str(int(t0['vp'][0]))+')$-$'+t0['labelpp'][0]+'('+str(int(t0['vpp'][0]))+')',ax=ax,loc='top right')
            # for i1,(d1,t1) in enumerate(t0.unique_dicts_matches('Fp','Fpp')):
                # for i2,(d2,t2) in enumerate(t1.unique_dicts_matches('ΔJ')):
                    # kwargs = dict(
                        # color=my.newcolor(i1),
                        # label=t2['branch'][0]+' ΔΣ='+format(int(t2['Σp'][0]-t2['Σpp'][0])),
                        # # ls=my.newlinestyle(i2),
                        # ls={0:'-',1:'--',-1:':'}[t2['ΔJ'][0]],
                        # mfc='none',
                        # mec=my.newcolor(i1),
                        # mew=1)
                    # kwargs.update(plot_kwargs)
                    # ax.plot(t2[xkey],t2[ykey],**kwargs)
                    # if self.experimental_transition is not None:
                        # i,j = spectra.find_common_transitions_levels(t2,self.experimental_transition)
                        # texp = self.experimental_transition[j]
                        # kwargs['label'] = None
                        # kwargs['ls'] = ''
                        # marker=my.newmarker(i2),
                        # if any(texp):
                            # if texp.is_known('d'+ykey):
                                # ax.errorbar(texp[xkey],texp[ykey],t['d'+ykey],**kwargs)
                            # else:
                                # ax.plot(texp[xkey],texp[ykey],**kwargs)
        # for ax in fig.axes:
            # ax.set_ylim(ymin=0)
            # # ax.set_yscale('log')
            # ax.set_xlabel(xkey)
            # ax.set_ylabel(ykey)
            # my.legend(loc='lower right',ax=ax,fontsize='medium',show_style=True)
        # return(fig)

    # def plot_spectrum(self,temperaturepp=300,ax=None,**plot_cross_section_kwargs):
        # """Plot a simulated cross section."""
        # transition = self.transition
        # transition['temperaturepp'] = temperaturepp
        # if ax is None: ax = plt.gca()
        # return(transition.plot_spectrum(**plot_cross_section_kwargs))

@lru_cache
def _get_linear_transition_moment(Sp,Λp,sp,Spp,Λpp,spp):

    ## check some selection rules
    if ((Λp==0 and Λpp==0 and sp!=spp)
        or (np.abs(Λp-Λpp)>1)
        or (Sp!=Spp)):
        raise Exception(f"Forbidden transition")

    ## Get signed and e/f parity quantum numbers and transformation matrices
    caseap  = quantum_numbers.get_case_a_basis( Λp, sp, Sp) 
    caseapp = quantum_numbers.get_case_a_basis(Λpp,spp,Spp)
    Mefp  = np.array( caseap['Mef'].evalf())
    Mefpp = np.array(caseapp['Mef'].evalf())
    efp = caseap['qnef']['ef']
    efpp = caseapp['qnef']['ef']
    Σp = caseap['qnef']['Σ']
    Σpp = caseapp['qnef']['Σ']
    
    fμrot = np.full((len(Mefp),len(Mefpp)),None)
    for (ip,qnpef),(ipp,qnppef) in itertools.product(
            enumerate(caseap['qnef'].rows()),
            enumerate(caseapp['qnef'].rows()),):
        fi = []                # add +- basis componetns
        for (jp,qnppm),(jpp,qnpppm) in itertools.product(
                enumerate(caseap['qnpm'].rows()),
                enumerate(caseapp['qnpm'].rows())):


            if qnppm['Σ'] != qnpppm['Σ']:
                continue
            
            c = Mefp[ip,jp]*Mefpp[ipp,jpp] # compute coefficient of thes signed-Ω wavefunctions to their respective ef-states
            if c==0:
                continue
            
            ## compute change in sign if reversed transition moment.
            μsign = +1
            if (qnppm['Λ']+qnpppm['Λ']) < 0:
                μsign = caseap['σvpm']*caseapp['σvpm']
            if (qnppm['Λ']+qnpppm['Λ']) == 0 and (qnppm['Σ']+qnpppm['Σ']) < 0:
                μsign = caseap['σvpm']*caseapp['σvpm']


            ## this computes every part of the linestrength apart
            ## from the adjustable transition moment, with a cache
            ## for speed
            def fij(Jpp,ΔJ,
                    Ωp=qnppm['Ω'],Ωpp=qnpppm['Ω'],
                    Λp=qnppm['Λ'],Λpp=qnpppm['Λ'],
                    μsign=float(μsign), c=float(c)):
                retval = np.full(len(Jpp),nan)
                i = (Jpp>=qnpppm['Ω'])|((Jpp+ΔJ)>=qnppm['Ω'])
                Jpp = Jpp[i]
                retval[i] = (
                    c       # contribution to ef-basis state
                    *np.sqrt((2*Jpp+1)*(2*(Jpp+ΔJ)+1)) # see hansson2005 eq. 13
                    *(-1)**(Jpp+ΔJ-Ωp)  # phase factor --see hansson2005 eq. 13
                    *(-1 if Λp==0 else 1)   # phase factor, a hack that should be understood
                    *μsign # transition moment phase factor (+1 or -1)
                    *quantum_numbers.wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj') # Wigner 3J line strength factor vectorised over Jpp
                )
                return retval
            fi.append(fij)
        fμrot[ip,ipp] = lambda Jpp,ΔJ,fi=fi: np.sum([fij(Jpp,ΔJ) for fij in fi])
    return efp,Σp,efpp,Σpp,fμrot

    # ## for each ΔΩ e/f transition compute the contribution signed
    # ## ΔΩ transitions. Previously I did this symbolically using
    # ## formulae for Honl-London factors, but I could never get
    # ## fully consistent phase factors (still can't really) and so
    # ## I rewrote this using Wigner-3J coefficients (see
    # ## hansson2005) which are solved numerically. This means
    # ## rather than merely computing symbolic signed-Ω transition
    # ## moemnt and then using matrix multiplication to get e/f
    # ## parity moement the quadruple-loop below and fucntion
    # ## addition below is needed.
    # fM = {}                 # dictionary of (Σp,efp,Σpp,efpp) transition moments
    # for (ip,qnpef),(ipp,qnppef) in itertools.product(
            # enumerate(caseap['qnef']), enumerate(caseapp['qnef']),):
        # ## loop over signed-Ω combinations of upper and lower
        # ## states and compute a transition moment functions for
        # ## each
        # fsigned = []            
        # for (jp,qnppm),(jpp,qnpppm) in itertools.product(
                # enumerate(caseap['qnpm'].rows()),enumerate(caseapp['qnpm'].rows())):
            # if qnppm['Σ']!=qnpppm['Σ']: continue # not allowed!
            # c = Mefp[ip,jp]*Mefpp[ipp,jpp] # compute coefficient of thes signed-Ω wavefunctions to their respective ef-states
            # if c==0: continue
            # ## compute change in sign if reversed transition moment.
            # μsign = +1
            # if (qnppm['Λ']+qnpppm['Λ']) < 0: μsign = caseap['σvpm']*caseapp['σvpm']
            # if (qnppm['Λ']+qnpppm['Λ']) == 0 and (qnppm['Σ']+qnpppm['Σ']) < 0:
                # μsign = caseap['σvpm']*caseapp['σvpm']
            # ## this computes every part of the linestrength apart
            # ## from the adjustable transition moment, with a cache
            # ## for speed
            # @lru_cache(maxsize=4096)
            # def fsignedi_scalar_no_μ(
                    # Jpp,ΔJ,
                    # Ωp=qnppm['Ω'],Ωpp=qnpppm['Ω'], Λp=qnppm['Λ'],Λpp=qnpppm['Λ'],
                    # μsign=float(μsign), c=float(c),
            # ):
                # return(
                    # c       # contribution to ef-basis state
                    # *np.sqrt((2*Jpp+1)*(2*(Jpp+ΔJ)+1)) # see hansson2005 eq. 13
                    # *(-1)**(Jpp+ΔJ-Ωp)  # phase factor --see hansson2005 eq. 13
                    # *(-1 if Λp==0 else 1)   # phase factor, a hack that should be understood
                    # *μsign # transition moment phase factor (+1 or -1)
                    # *wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj') # Wigner 3J line strength factor vectorised over Jpp
                # )
            # ## add the transiton moment
            # def fsignedi(Jpp,ΔJ,fsignedi_scalar_no_μ=fsignedi_scalar_no_μ):
                # return fsignedi_scalar_no_μ(Jpp,ΔJ)*μ(Jpp,ΔJ)
            # fsigned.append(fsignedi)

        # # ## sum over all sign combinations and save this
        # # ## Σp,efp,Σpp,efpp transition moment
        # # fM[qnpef['Σ'],qnpef['ef'],qnppef['Σ'],qnppef['ef'],] = lambda Jpp,ΔJ,fsigned=fsigned: np.sum([f(Jpp,ΔJ) for f in fsigned],axis=0)
    # # ## for all matching transition states find the corret
    # # ## transition moment if there is one
    # # stateps = self.statep.get_matching_states(**qn_u)
    # # statepps = self.statepp.get_matching_states(**qn_l)
    # # for tstates,tqn in ((stateps,qn_u),(statepps,qn_l)):
        # # if len(tstates)==0:
            # # print(f'warning: add_transition_moment: No states found matching {repr(tqn)}')
    # # for statep,statepp in itertools.product(stateps, statepps):
        # # qns = (statep.qn['Σ'],statep.qn['ef'],statepp.qn['Σ'],statepp.qn['ef'])
        # # if qns not in fM: continue
        # # self.transition_moments.append(Vibronic_Transition_Moment(statep,statepp,fM[qns]))

