from copy import copy,deepcopy
from pprint import pprint
from functools import lru_cache
import itertools

import numpy as np
from numpy import nan,array
import sympy
from scipy import linalg

from . import levels,lines
from . import quantum_numbers
from . import tools
from . import dataset
from .dataset import Dataset
from .tools import find,cache
from .optimise import Optimiser,P,optimise_method,format_input_method
from .kinetics import get_species,Species



class VibLevel(Optimiser):
    """A vibronic interaction matrix."""

    def __init__(
            self,
            name='viblevel',
            species='14N2',
            J=None,                       # compute on this, or default added
            # ef=None,                      # compute for these e/f parities , default ('e','f')
            experimental_level=None,      # a Level object for optimising relative to 
            # eigenvalue_ordering='maximise coefficients', # for deciding on the quantum number assignments of mixed level, options are 'minimise residual', 'maximise coefficients', 'preserve energy ordering' or None
            Eref=0.,       # energy reference relative to equilibrium energy, defaults to 0. -- not well defined
    ):
        self.name = name          # a nice name
        self.species = get_species(species)
        tkwargs = {'Eref':Eref, 'permit_auto_defaults':True,
                   'permit_nonprototyped_data':False,}
        self.manifolds = {}                                   
        self.level = levels.Diatomic(name=f'{self.name}.level',**tkwargs)
        self.level.pop_format_input_function()
        self.level.add_suboptimiser(self)
        self.level.pop_format_input_function()
        # self.vibrational_level = levels.Diatomic(**tkwargs)
        self.vibrational_spin_level = levels.Diatomic(**tkwargs) 
        self.interactions = Dataset() 
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
        self.verbose = False
        ## various options for how to assign the perturbed levels: None
        # self.eigenvalue_ordering = eigenvalue_ordering
        ## inputs / outputs of diagonalisation
        self._H_subblocks = []
        self.H = None
        self.eigvals = None
        self.eigvects = None
        ## a Level object containing data, better access through level property
        self._exp = None # an array of experimental data matching the model data in shape
        ## the optimiser
        Optimiser.__init__(self,name=self.name)
        self.pop_format_input_function()
        self.automatic_format_input_function(
            multiline=False,
            limit_to_args=('name', 'species', 'J', 'Eref',))
        def f(directory):
            self.level.save(directory+'/level.h5')
        self.add_save_to_directory_function(f)
        self.add_post_construct_function(self.construct_levels)

    def construct_levels(self):
        """The actual matrix diagonlisation is done last."""
        ## if first run or model changed then construct Hamiltonian
        ## and blank rotational level, and determine which levels are
        ## actually allowed
        if True or self.H is None or self._last_add_construct_function_time > self._last_construct_time:
            ## construct Hamiltonian
            self.H = np.full((len(self.J),len(self.vibrational_spin_level),len(self.vibrational_spin_level)),0.)
            ## create a rotational level with all quantum numbers
            ## inserted and limit to allowed levels
            self.level.clear()
            self.level['J'] = np.repeat(self.J,len(self.vibrational_spin_level))
            for key in self.vibrational_spin_level:
                self.level[key] = np.tile(self.vibrational_spin_level[key], len(self.J))
        ## compute upper-triangular part of H from subblocks
        self.H[:] = 0.0
        for ibeg,jbeg,fH in self._H_subblocks:
            for (i,j),fHi in np.ndenumerate(fH):
                if ibeg == jbeg:
                    ## manifold 
                    self.H[:,i+ibeg,j+jbeg] += fHi(self.J)
                else:
                    ## coupling between manifolds
                    t = fHi(self.J)
                    self.H[:,i+ibeg,j+jbeg] += t
                    self.H[:,j+jbeg,i+ibeg] += t
        ## nothing to be done
        if len(self.vibrational_spin_level) == 0:
            return
        ## compute mixed energies and mixing coefficients
        self.eigvals = {}             # eignvalues
        self.eigvects = {}             # mixing coefficients
        je = self.vibrational_spin_level['ef'] == +1
        jf = self.vibrational_spin_level['ef'] == -1
        if np.sum(je) == 0:
            je = None
        if np.sum(jf) == 0:
            jf = None
        for iJ,J in enumerate(self.J):
            H = self.H[iJ,:,:]
            H[np.isnan(H)] = 0.0 # better to solve at source
            ## compute e- and f-parity matrcies separatesly.  Might be
            ## nice to automatically convert matrix into blocks
            eigvals = np.zeros(self.H.shape[2],dtype=complex)
            eigvects = np.zeros((self.H.shape[2],self.H.shape[2]),dtype=float)
            if je is not None:
                eigvals_e,eigvects_e = linalg.eigh(H[np.ix_(je,je)])
                eigvals_e,eigvects_e = _diabaticise_eigenvalues(eigvals_e,eigvects_e)
                eigvals[je] = eigvals_e
                eigvects[np.ix_(je,je)] = eigvects_e
            if jf is not None:
                eigvals_f,eigvects_f = linalg.eigh(H[np.ix_(jf,jf)])
                eigvals_f,eigvects_f = _diabaticise_eigenvalues(eigvals_f,eigvects_f)
                eigvals[jf] = eigvals_f
                eigvects[np.ix_(jf,jf)] = eigvects_f
            ## diagonaliase all at one
            # eigvals,eigvects = linalg.eigh(H)
            # eigvals,eigvects = _diabaticise_eigenvalues(eigvals,eigvects)
            self.eigvals[J] = eigvals.real
            self.eigvects[J] = eigvects.real
        ## insert energies into level
        self.level['E'] = np.concatenate(list(self.eigvals.values()))
        self.level.index((self.level['J']>self.level['Ω']))

    @optimise_method(add_construct_function=False)
    def add_level(self,name,**kwargs):
        """Add a new electronic vibrational level. kwargs contains fitting
        parameters and optionally extra quantum numbers."""
        ## all quantum numbers and molecular parameters
        kw = quantum_numbers.decode_level(name) | kwargs 
        kw['species'] = self.species.isotopologue
        if 'S' not in kw or 's' not in kw or 'Λ' not in kw:
            raise Exception('Quantum numbers S, s, and Λ are required.')
        ## Checks that integer/half-integer nature of J corresponds to
        ## quantum number S
        if kw['S']%1!=self.J[0]%1:
            raise Exception(f'Integer/half-integer nature of S and J do not match: {S%1} and {self.J[0]%1}')
        ## get Hamiltonian and insert adjustable parameters into
        ## functions
        ef,Σ,sH,fH = _get_linear_H(kw['S'],kw['Λ'],kw['s'])
        fH = [[lambda J,f=fH[i,j]: f(J,**kw)
               for i in range(fH.shape[0])]
              for j in range(fH.shape[1])]
        ## add to self
        ibeg = len(self.vibrational_spin_level)
        self.vibrational_spin_level.extend(
            ef=ef,Σ=Σ,
            **{key:kw[key] for key in ('species','label','S','Λ','s','v')})
        self._H_subblocks.append((ibeg,ibeg,fH))
        if name in self.manifolds:
            raise Exception(f'Non-unique name: {repr(name)}')
        self.manifolds[name] = dict(ibeg=ibeg,ef=ef,Σ=Σ,n=len(ef),**kw) 

    @format_input_method()
    def add_LS_coupling(self,name1,name2,ηv=0,ηDv=0):
        kw1 = self.manifolds[name1]
        kw2 = self.manifolds[name2]
        ## get coupling matrices -- cached
        JL,JS,LS,NNJL,NNJS,NNLS = _get_offdiagonal_coupling(
            kw1['S'],kw1['Λ'],kw1['s'],kw2['S'],kw2['Λ'],kw2['s'],verbose=self.verbose)
        ## substitute in adjustable parameter
        H = np.full(LS.shape,None)
        for i in np.ndindex(JL.shape):
            H[i] = lambda J,i=i: ηv*LS[i](J) + ηDv*NNLS[i](J)
        self._H_subblocks.append((kw1['ibeg'],kw2['ibeg'],H))

    @format_input_method()
    def add_JL_coupling(self,name1,name2,ξv=0,ξDv=0):
        kw1 = self.manifolds[name1]
        kw2 = self.manifolds[name2]
        ## get coupling matrices -- cached
        JL,JS,LS,NNJL,NNJS,NNLS = _get_offdiagonal_coupling(
            kw1['S'],kw1['Λ'],kw1['s'],kw2['S'],kw2['Λ'],kw2['s'],verbose=self.verbose)
        ## substitute in adjustable parameter
        H = np.full(JL.shape,None)
        for i in np.ndindex(H.shape):
            H[i] = lambda J,i=i: -ξv*JL[i](J) - ξDv*NNJL[i](J)
        self._H_subblocks.append((kw1['ibeg'],kw2['ibeg'],H))

    @format_input_method()
    def add_JS_coupling(self,name1,name2,pv=0):
        kw1 = self.manifolds[name1]
        kw2 = self.manifolds[name2]
        ## get coupling matrices -- cached
        JL,JS,LS,NNJL,NNJS,NNLS = _get_offdiagonal_coupling(
            kw1['S'],kw1['Λ'],kw1['s'],kw2['S'],kw2['Λ'],kw2['s'],verbose=self.verbose)
        ## substitute in adjustable parameter
        H = np.full(JL.shape,None)
        for i in np.ndindex(H.shape):
            H[i] = lambda J,i=i: pv*JS[i](J)
        self._H_subblocks.append((kw1['ibeg'],kw2['ibeg'],H))

    def plot(self,**kwargs):
        kwargs.setdefault('xkey','J')
        kwargs.setdefault('ykeys','E')
        return self.level.plot(**kwargs)
        
class VibLine(Optimiser):
    
    """Calculate and optimally fit the line strengths of a band between
    two states defined by LocalDeperturbation objects. Currently only
    for single-photon transitions. """

    def __init__(
            self,
            name,
            level_u,
            level_l,
            J_l=None,
            ΔJ=None,
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
        self.line = lines.Diatomic(name=f'{self.name}.line',**tkwargs)
        self.line.pop_format_input_function()
        self.line.add_suboptimiser(self)
        self.line.pop_format_input_function()
        # self.vibrational_line = lines.Diatomic(**tkwargs)
        # self.vibrational_spin_line = lines.Diatomic(**tkwargs)
        # self.vibrational_spin_line.add_suboptimiser(self)
        # self.μ = None
        self._transition_moment_functions = []
        ## decide on lower and upper state J values, and ΔJ
        ## transitions
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
        ## construct optimiser -- inheriting from states
        Optimiser.__init__(self,name=self.name)
        self.pop_format_input_function()
        self.automatic_format_input_function(
            multiline=False,
            limit_to_args=('name', 'level_u', 'level_l', 'J_l', 'ΔJ',))
        self.add_suboptimiser(self.level_u,self.level_l,add_format_function=False)
        def f(directory):
            self.line.save(directory+'/line.h5')
        self.add_save_to_directory_function(f)
        self.add_post_construct_function(self.construct_lines)

    def construct_lines(self):
        ## initialise μ0 and line arrays if model has changed
        if self._last_add_construct_function_time > self._last_construct_time:
            ## initialise μ0
            self.μ0 = np.full((
                len(self.J_l),
                len(self.ΔJ),
                len(self.level_u.vibrational_spin_level),
                len(self.level_l.vibrational_spin_level),),0.)
            ## add quantum numbers to line
            self.line.clear()
            for iJ_l,J_l in enumerate(self.J_l):
                for iΔJ,ΔJ in enumerate(self.ΔJ):
                    J_u = J_l + ΔJ
                    if J_u not in self.level_u.J:
                        continue
                    n_l = len(self.level_l.vibrational_spin_level)
                    n_u = len(self.level_u.vibrational_spin_level)
                    self.line.extend(
                        J_u=J_u,
                        J_l=J_l,
                        # E_u=np.repeat(self.level_u.eigvals[J_u],n_l),
                        # E_l=np.tile(self.level_l.eigvals[J_l],n_u),
                        **{key+'_u':np.repeat(val,n_l) for key,val in self.level_u.vibrational_spin_level.items()},
                        **{key+'_l':np.tile(val,n_u) for key,val in self.level_l.vibrational_spin_level.items()},)
        ## build μ0
        for iu,jl,fμ in self._transition_moment_functions:
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                self.μ0[:,iΔJ,iu,jl] = fμ(self.J_l,ΔJ)
        ## could vectorise linalg with np.dot
        μs = []
        E_us = []
        E_ls = []
        for iJ_l,J_l in enumerate(self.J_l):
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                J_u = J_l + ΔJ
                if J_u not in self.level_u.J:
                    continue
                μ0 = self.μ0[iJ_l,iΔJ,:,:]
                c_l = self.level_l.eigvects[J_l]
                c_u = self.level_u.eigvects[J_u]
                ## get mixed line strengths
                μ = np.dot(np.transpose(c_u),np.dot(μ0,c_l))
                μs.append(μ)
                ## get energy levels for this J_u,J_l transition
                E_us.append(np.repeat(self.level_u.eigvals[J_u],len(self.level_l.vibrational_spin_level)))
                E_ls.append(np.tile(  self.level_l.eigvals[J_l],len(self.level_u.vibrational_spin_level)))
        ## add all new data to rotational line
        self.line['μ'] = np.ravel(μs)
        self.line['E_u'] = np.ravel(E_us)
        self.line['E_l'] = np.ravel(E_ls)
        ## remove forbidden lines
        # self.line.index(
            # ## levels do not exist
            # (self.line['J_l'] >= self.line['Ω_l'])
            # & (self.line['J_u'] >= self.line['Ω_u'])
            # # ## no Q branch
            # # & ((self.line['Λ_u'] > 0)
            # #    | (self.line['Λ_l'] > 0)
            # #    | (self.line['J_u'] != self.line['J_l']))
            # ## no Q branch
            # & ((self.line['Λ_u'] > 0)
               # | (self.line['Λ_l'] > 0)
               # | (self.line['J_u'] != self.line['J_l']))
        # )
        # self.line.remove_match(Sij=0)


    @format_input_method()
    def add_transition_moment(self,name_u,name_l,μv=1):
        """Add constant transition moment. μv can be optimised."""
        """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        transitions. μv should be in atomic units and can be specifed
        as a value (optimisable), a function of R or a suboptimiser
        given ['μ']."""

        ## get all quantum numbers
        kwu = self.level_u.manifolds[name_u]
        kwl = self.level_l.manifolds[name_l]
        ## get transition moment functions for all ef/Σ combinations
        ## and add optimisable parameter to functions
        fμ = _get_linear_transition_moment(kwu['S'],kwu['Λ'],kwu['s'],kwl['S'],kwl['Λ'],kwl['s'],verbose=self.verbose)
        ## find indices of transitioning state and add to
        ## self._transition_moment_functions
        for i,j in np.ndindex(fμ.shape):
            if fμ[i,j] is None:
                continue
            self._transition_moment_functions.append((
                i+kwu['ibeg'],j+kwl['ibeg'],lambda J,ΔJ,f=fμ[i,j]: f(J,ΔJ)*μv))

    def plot(self,**kwargs):
        kwargs.setdefault('xkey','ν')
        kwargs.setdefault('ykey','Sij')
        return self.line.plot_stick_spectrum(**kwargs)

@lru_cache
def _get_linear_H(S,Λ,s):
    """Compute symbolic and functional Hamiltonian for the spin manifold
    of a linear molecule."""
    ## symbolic variables, Note that expected value of ef is +1 or -1 for 'e' and 'f'
    p = {key:sympy.Symbol(key) for key in (
        'Tv','Bv','Dv','Hv','Av','ADv','λv','λDv','λHv','γv','γDv','ov','pv','pDv','qv','qDv')}
    J = sympy.Symbol('J')
    case_a = quantum_numbers.get_case_a_basis(S,Λ,s,print_output=False)
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
            if efi != efj:
                continue
            # ef = (1 if efi=='e' else -1) # Here ef is +1 for 'e' and -1 for 'f'
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
                    H[i,j] += -efi*(p['ov']+p['pv']+p['qv'])
                elif Σi== 0 and Σj== 0:
                    H[i,j] += -efi*p['qv']*J*(J+1)/2
                elif Σi==+1 and Σj==+1:
                    H[i,j] += 0
                ## off-diagonal elements
                elif Σi==-1 and Σj==+0:
                    H[i,j] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*efi
                    H[j,i] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*efi
                elif Σi==-1 and Σj==+1:
                    H[i,j] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*efi
                    H[j,i] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*efi
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
    # print( H)                  #  DEBUG
    ## Convert elements of symbolic Hamiltonian into lambda functions
    fH = np.full(H.shape,None)
    for i in range(len(Σs)):
        for j in range(i,-1,-1):
            fH[i,j] = fH[j,i] = tools.lambdify_sympy_expression(
                H[i,j],'J',**{key:0 for key in p})
    return efs,Σs,H,fH

@lru_cache
def _get_offdiagonal_LS_coupling_matrix(S1,Λ1,s1,S2,Λ2,s2):
    """Compute symbolic and functional Hamiltonian for the spin manifold
    of a linear molecule between two different electronic-vibrational
    states."""
    ## get case a quantum numbers and symmmetrising matrices for both
    ## states and combine into one
    casea1 = quantum_numbers.get_case_a_basis(S1,Λ1,s1)
    casea2 = quantum_numbers.get_case_a_basis(S2,Λ2,s2)
    qnpm = casea1['qnpm'] + casea2['qnpm']
    qnef = casea1['qnef'] + casea2['qnef']
    Mef = linalg.block_diag(casea1['Mef'],casea2['Mef'],)
    NNef = linalg.block_diag(casea1['NNef'],casea2['NNef'])
    n = len(qnpm)
    ## get LS matrix -- off-diagonal elements only
    LS = sympy.zeros(n) # in signed-Ω basis
    from sympy.physics.wigner import wigner_3j
    for i1,qn1 in enumerate(casea1['qnpm'].rows()): # bra
        for i2,qn2 in enumerate(casea2['qnpm'].rows()): # ket
            i,j = i1,i2+casea1['n']
            ## build spin-orbit interaction matrix
            if (qn1['Λ']-qn2['Λ'])==(qn2['Σ']-qn1['Σ']) and np.abs(qn1['Λ']-qn2['Λ'])==1: # selection rules ΔΛ=-ΔΣ=±1, ΔΩ=0
                if (qn1['Λ']-qn2['Λ'])==+1:
                    LS[i,j] = LS[j,i] = np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L+S-
                if (qn1['Λ']-qn2['Λ'])==-1:
                    LS[i,j] = LS[j,i] = qn1['σv']*qn2['σv']*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L-S+ calculated from L+S- using σv equivalence
            elif (qn1['Λ']==qn2['Λ']) and (qn2['Σ']==qn1['Σ']) : # selection rules, ΔΛ=ΔΣ=ΔΩ=0
                phase = 1
                if qn1['Λ']<0 or (qn1['Λ']==0 and qn1['Σ']<0):
                    phase *= qn1['σv']
                if qn2['Λ']<0 or (qn2['Λ']==0 and qn2['Σ']<0):
                    phase *= qn2['σv']
                LS[i,j] = LS[j,i] = phase*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # LzSz
            else:
                pass
    ## transform to e/f basis
    LSef = Mef*LS*Mef.T # transform to e/f basis
    ## get upper off-diagonal block
    LSef = LSef[casea1['n']:,:casea1['n']]
    η = sympy.Symbol("η")
    LSef = η*LSef
    ## lambdify
    H = np.full(LSef.shape,None)
    for i,LSefi in np.ndenumerate(LSef):
        H[i] = tools.lambdify_sympy_expression(η*LSefi,'J','η')
    return H

@lru_cache
def _get_offdiagonal_coupling(S1,Λ1,s1,S2,Λ2,s2,verbose=False):
    """Compute symbolic and functional Hamiltonian for the spin manifold
    of a linear molecule between two different electronic-vibrational
    states.  ⟨1|H|2⟩"""
    ## get case a quantum numbers and symmmetrising matrices for both
    ## states and combine into one
    casea1 = quantum_numbers.get_case_a_basis(S1,Λ1,s1)
    casea2 = quantum_numbers.get_case_a_basis(S2,Λ2,s2)
    Mef = linalg.block_diag(casea1['Mef'],casea2['Mef'],)
    NNef = linalg.block_diag(casea1['NNef'],casea2['NNef'])
    n = casea1['n'] + casea2['n']
    ## get matrix elements in signed-Ω basis
    JL = sympy.zeros(n)         # L-uncoupling matrix
    JS = sympy.zeros(n)         # S-uncoupling matrix
    LS = sympy.zeros(n)         # spin-orbit matrix
    J = sympy.Symbol("J")
    from sympy.physics.wigner import wigner_3j
    for i1,qn1 in enumerate(casea1['qnpm'].rows()): # bra
        for i2,qn2 in enumerate(casea2['qnpm'].rows()): # ket
            ## indicies in full size matrix containing both manifolds
            i,j = i1,i2+casea1['n']
            ## add term in L-uncoupling matrix, JL[i,j] =  ⟨i(Λ+1)SΣJ(Ω+1)|J-L+|jΛSΣJΩ⟩
            if (qn2['S']==qn1['S']
                and np.abs(qn2['Λ']-qn1['Λ'])==1
                and (qn2['Ω']-qn1['Ω'])==(qn2['Λ']-qn1['Λ'])): # test for satisfying selection rulese i=i, S=S, ΔΛ=1, ΔΛ=ΔΩ
                if qn1['Λ']>qn2['Λ']:
                    JL[i,j] = JL[j,i] = sympy.sqrt(J*(J+1)-qn2['Ω']*(qn2['Ω']+1)) # J-L+
                else:
                    JL[i,j] = JL[j,i] = qn1['σv']*qn2['σv']*sympy.sqrt(J*(J+1)--qn2['Ω']*(-qn2['Ω']+1)) # J+L- by symmetry from J-L+
            ## add term in S-uncoupling matrix, JS[i,j] = ⟨iΛS(Σ+1)J(Ω+1)|J-S+|jΛSΣJΩ⟩
            if (qn2['S']==qn1['S']
                and np.abs(qn2['Σ']-qn1['Σ'])==1
                and (qn2['Ω']-qn1['Ω'])==(qn2['Σ']-qn1['Σ'])): # test for satisfying selection rulese i=i, S=S, ΔΣ=1, ΔΣ=ΔΩ
                ## correct phases here?
                if qn1['Σ']>qn2['Σ']:
                    JS[i,j] = JS[j,i] = sympy.sqrt(qn2['S']*(qn2['S']+1)-qn2['Σ']*(qn2['Σ']+1))*sympy.sqrt(J*(J+1)-qn2['Ω']*(qn2['Ω']+1)) # J-S+
                else:
                    JS[i,j] = JS[j,i] = qn1['σv']*qn2['σv']*sympy.sqrt(qn2['S']*(qn2['S']+1)--qn2['Σ']*(-qn2['Σ']+1))*sympy.sqrt(J*(J+1)--qn2['Ω']*(-qn2['Ω']+1))  # J+S- by symmetry from J-S+
            ## add term in spin-orbit interaction matrix - selection rules ΔΛ=-ΔΣ=±1, ΔΩ=0
            if np.abs(qn1['Λ']-qn2['Λ'])==1 and (qn1['Λ']-qn2['Λ'])==(qn2['Σ']-qn1['Σ']): # 
                if (qn1['Λ']-qn2['Λ'])==+1:
                    LS[i,j] = LS[j,i] = np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L+S-
                if (qn1['Λ']-qn2['Λ'])==-1:
                    LS[i,j] = LS[j,i] = qn1['σv']*qn2['σv']*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L-S+ calculated from L+S- using σv equivalence
            ## add term in spin-orbit interaction matrix - selection rules ΔΛ=ΔΣ=ΔΩ=0
            if (qn1['Λ']==qn2['Λ']) and (qn2['Σ']==qn1['Σ']) :
                phase = 1
                if qn1['Λ']<0 or (qn1['Λ']==0 and qn1['Σ']<0):
                    phase *= qn1['σv']
                if qn2['Λ']<0 or (qn2['Λ']==0 and qn2['Σ']<0):
                    phase *= qn2['σv']
                LS[i,j] = LS[j,i] = phase*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # LzSz
    ## transform to e/f basis
    JLef = Mef*JL*Mef.T
    JSef = Mef*JS*Mef.T
    JSef = np.abs(JSef)         # HACK
    LSef = Mef*LS*Mef.T
    ## get rotationally commutated versions -- could compute higher
    ## orders here
    NNJLef = (JLef*NNef+NNef*JLef)/2                
    NNJSef = (JSef*NNef+NNef*JSef)/2                
    NNLSef = (LSef*NNef+NNef*LSef)/2                
    ## get upper off-diagonal block
    i = (slice(0,casea1['n']),slice(casea1['n'],n))
    JLef = JLef[i]
    JSef = JSef[i]
    LSef = LSef[i]
    NNJLef = NNJLef[i]
    NNJSef = NNJSef[i]
    NNLSef = NNLSef[i]
    ## print matrix elements
    if verbose:
        for operator,matrix in (('JL',JLef), ('JS',JSef), ('LS',LSef),):
            for i,qn1 in enumerate(casea1['qnef'].rows()):
                for j,qn2 in enumerate(casea2['qnef'].rows()):
                    if matrix[i,j] != 0:
                        print(_format_matrix_element(qn1,qn2,operator,matrix[i,j]))
    ## lambdify and add parameter 
    JL = np.full(JLef.shape,None)
    JS = np.full(JSef.shape,None)
    LS = np.full(LSef.shape,None)
    NNJL = np.full(NNJLef.shape,None)
    NNJS = np.full(NNJSef.shape,None)
    NNLS = np.full(NNLSef.shape,None)
    for i in np.ndindex(LSef.shape):
        JL[i] = tools.lambdify_sympy_expression(JLef[i],'J')
        JS[i] = tools.lambdify_sympy_expression(JSef[i],'J')
        LS[i] = tools.lambdify_sympy_expression(LSef[i],'J')
        NNJL[i] = tools.lambdify_sympy_expression(NNJLef[i],'J')
        NNJS[i] = tools.lambdify_sympy_expression(NNJSef[i],'J')
        NNLS[i] = tools.lambdify_sympy_expression(NNLSef[i],'J')
    ## return
    return JL,JS,LS,NNJL,NNJS,NNLS


    # ## symbolic variables, Note that expected value of ef is +1 or -1 for 'e' and 'f'
    # casea1 = quantum_numbers.get_casea_basis(Λ1,s1,S1,print_output=False)
    # casea1 = quantum_numbers.get_casea_basis(Λ1,s,S2,print_output=False)
    # efs = casea['qnef']['ef']
    # Σs = casea['qnef']['Σ']
    # NN = casea['NNef']
    # NS = casea['NSef']
    # ## construct some convenient matrices
    # def anticommutate(X,Y):
        # return(X*Y+Y*X)
    # I  = sympy.eye(casea['n']) # unit matrix
    # ## Equation 18 of brown1979
    # H = p['Tv']*I + p['Bv']*NN - p['Dv']*NN**2 + p['Hv']*NN**3       # T + BN**2 - DN**4 + HN**6
    # if S>0:
        # if Λ>0: H += anticommutate(p['Av']*I+p['ADv']*NN,sympy.diag(*[float(Λ*Σ) for Σ in Σs]))/2 # 1/2[A+AN2,LzSz]+
        # H += (p['γv']*I+p['γDv']*NN)*NS # (γ+γD.N**2)N.S
        # H += anticommutate(p['λv']*I+p['λDv']*NN+p['λHv']*NN**2,sympy.diag(*[float(Σ**2) for Σ in Σs])-I/3*S*(S+1)) # [λ+λD.N**2),Sz**2-1/3*S**2]+
    # ## add Λ-doubling terms here, element-by-element.
    # for i,(Σi,efi) in enumerate(zip(Σs,efs)):
        # for j,(Σj,efj) in enumerate(zip(Σs,efs)):
            # if efi!=efj: continue
            # ef = (1 if efi=='e' else -1) # Here ef is +1 for 'e' and -1 for 'f'
            # ## 1Π state
            # if Λ>0 and S==0: 
                # if efi=='f' and efj=='f': H[i,j] += p['qv']*J*(J+1)   # coefficient
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
                        # ## diagonal elseement for level 2 in amiot1981
                        # if   Σi==-0.5 and Σj==-0.5:
                            # H[i,j] += efi*( -0.5*(J+0.5)*p['pv']
                                             # -(J+0.5)*p['qv']
                                             # -0.5*(J+0.5)*((J-0.5)*(J+0.5)+2)*p['pDv']
                                             # -0.5*(3*(J-0.5)*(J+0.5)+4)*(J+0.5)*p['qDv']    )
                        # ## diagonal element for level 1 in amiot1981
                        # elif Σi==+0.5 and Σj==+0.5:
                            # H[i,j] += efi*( -0.5*(J-0.5)*(J+1.5)*(J+0.5)*p['qv']
                                             # -0.5*(J-0.5)*(J+0.5)*(J+0.5)*p['qDv']  )
                        # ## off-diagonal element
                        # elif Σi==-0.5 and Σj==+0.5:
                            # H[i,j] += efi*(  0.5*((J-0.5)*(J+1.5))**0.5*(J+0.5)*p['qv']
                                              # + 0.25*((J-0.5)*(J+0.5))**0.5*(J+0.5)*p['pDv']
                                              # +0.5*((J-0.5)*(J+0.5))**(0.5)*((J-0.5)*(J+0.5)+2)*(J+0.5)*p['qDv'] )
            # ## 3Π states, from Table I brown_merer1979            
            # elif Λ==1 and S==1:
                # ## diagonal elements
                # if   Σi==-1 and Σj==-1:
                    # H[i,j] += -ef*(p['ov']+p['pv']+p['qv'])
                # elif Σi== 0 and Σj== 0:
                    # H[i,j] += -ef*p['qv']*J*(J+1)/2
                # elif Σi==+1 and Σj==+1:
                    # H[i,j] += 0
                # ## off-diagonal elements
                # elif Σi==-1 and Σj==+0:
                    # H[i,j] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*ef
                    # H[j,i] += -sympy.sqrt(2*J*(J+1))*-1/2*(p['pv']+2*p['qv'])*ef
                # elif Σi==-1 and Σj==+1:
                    # H[i,j] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*ef
                    # H[j,i] += -sympy.sqrt(J*(J+1)*(J*(J+1)-2))*1/2*p['qv']*ef
                # elif Σi==+0 and Σj==+1:
                    # pass # zero
            # # else:
                # # if 'ov' in kwargs or 'pv' in kwargs or 'qv' in kwargs:
                    # # raise Exception("Cannot use ov, pv, or qv because Λ-doubling not specified for state with Λ="+repr(Λ)+" and S="+repr(S))
    # ## simplify and print Hamiltonian if requested
    # # if self.verbose:
        # # print( )
        # # print(self.format_input_functions[-1]())
        # # print_matrix_elements(casea['qnef'],H,'H')
    # ## Convert elements of symbolic Hamiltonian into lambda functions
    # fH = np.full(H.shape,None)
    # for i in range(len(Σs)):
        # for j in range(i,-1,-1):
            # fH[i,j] = fH[j,i] = tools.lambdify_sympy_expression(
                # H[i,j],'J',**{key:0 for key in p})
    # return efs,Σs,H,fH

@lru_cache
def _get_linear_transition_moment(Sp,Λp,sp,Spp,Λpp,spp,verbose=False):
    """Matrix elements for electric-diople transition between two
    Hund's case (a) manifolds."""
    ## check some selection rules
    if ((Λp==0 and Λpp==0 and sp!=spp)
        or (np.abs(Λp-Λpp)>1)
        or (Sp!=Spp)):
        raise Exception(f"Forbidden transition")
    ## Get signed and e/f parity quantum numbers and transformation matrices
    caseap  = quantum_numbers.get_case_a_basis( Sp, Λp, sp) 
    caseapp = quantum_numbers.get_case_a_basis(Spp,Λpp,spp)
    Mefp  = np.array( caseap['Mef'].evalf())
    Mefpp = np.array(caseapp['Mef'].evalf())
    # efp = caseap['qnef']['ef']
    # efpp = caseapp['qnef']['ef']
    # Σp = caseap['qnef']['Σ']
    # Σpp = caseapp['qnef']['Σ']
    ## get matrix elements as symbolic functions of lower state J.
    ## e/f-parity symmetrised matrix elements are a linear
    ## combinantion of signed-Ω (pm) matrix elements
    fμ = np.full((len(Mefp),len(Mefpp)),None)
    for (ip,qnpef),(ipp,qnppef) in itertools.product(
            enumerate(caseap['qnef'].rows()),
            enumerate(caseapp['qnef'].rows()),):
        fi = []                # add +- basis componetns
        for (jp,qnppm),(jpp,qnpppm) in itertools.product(
                enumerate(caseap['qnpm'].rows()),
                enumerate(caseapp['qnpm'].rows())):
            ## skip -- not allowed
            if qnppm['Σ'] != qnpppm['Σ']:
                continue
            ## compute coefficient of thes signed-Ω wavefunctions to
            ## their respective ef-states
            c = Mefp[ip,jp]*Mefpp[ipp,jpp]
            if c==0:
                continue
            ## compute change in sign if reversed transition moment.
            if qnppm['Ω'] >= qnpppm['Ω']:
                μsign = 1.
            elif qnppm['Ω'] < qnpppm['Ω']:
                ## reversed transition moment
                μsign = qnppm['σv']*qnpppm['σv']
            ## this computes every part of the linestrength apart
            ## from the adjustable transition moment
            def fij(Jpp,ΔJ,
                    Ωp=qnppm['Ω'],Ωpp=qnpppm['Ω'],
                    Λp=qnppm['Λ'],Λpp=qnpppm['Λ'],
                    μsign=float(μsign), c=float(c)):

                tJpp = (2*Jpp+1)*(2*(Jpp+ΔJ)+1)
                tJpp[tJpp<0] = 0 # avoid sqrt <0 warnings
                retval = (
                    c       # contribution to ef-basis state
                    *np.sqrt(tJpp) # see hansson2005 eq. 13
                    *(-1)**(Jpp+ΔJ-Ωp)  # phase factor --see hansson2005 eq. 13
                    *(-1 if Λp==0 else 1)   # phase factor, a hack that should be understood
                    *μsign # transition moment phase factor (+1 or -1)
                    *quantum_numbers.wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj')  # Wigner 3J line strength factor vectorised over Jpp
                )
                retval[np.isnan(retval)] = 0. # not allowed
                # print('DEBUG:', Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp)
                # print('DEBUG:', quantum_numbers.wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj'))
                # print('DEBUG:', retval)
                return retval
            fi.append(fij)
        fμ[ip,ipp] = lambda Jpp,ΔJ,fi=fi: np.sum([fij(Jpp,ΔJ) for fij in fi],axis=0)
        if verbose:
            for ΔJ in (-1,0,1):
                if (val:=float(fμ[ip,ipp](array([10]),ΔJ))) != 0:
                    print(_format_matrix_element(qnpef,qnppef,f'μ(ΔJ={ΔJ:+2},Jpp=10)',format(val,'+0.4f')))
    return fμ

def encode_level(**kwargs):
    """Turn quantum numbers etc (as in decode_level) into a string name. """
    ## Test code
    ## t = encode_level(species='N2',label='cp4',Λ=0,s=0,gu='u',S=0,v=5,F=1,ef='e',Ω=0)
    ## print(t)
    ## pprint(decode_level(t))
    kwargs = copy(kwargs)  # all values get popped below, so make a ocpy
    retval = ''                 # output string
    ## electronic state label and then get symmetry symbol, first get whether Σ+/Σ-/Π etc, then add 2S+1 then add g/u
    if 'Λ' in kwargs:
        Λ = kwargs.pop('Λ')
        if Λ==0:
            retval = 'Σ'
            ## get + or - superscript
            if 's' in kwargs:
                retval += ('+' if kwargs.pop('s')==0 else '-')
        elif Λ==1:
            retval = 'Π'
        elif Λ==2:
            retval = 'Δ'
        elif Λ==3:
            retval = 'Φ'
        else:
            raise Exception('Λ>3 not implemented')
        if Λ>0 and 's' in kwargs: kwargs.pop('s') # not needed
        if 'S' in kwargs: retval = str(int(2*float(kwargs.pop('S'))+1))+retval
        if 'gu' in kwargs: retval += kwargs.pop('gu')
    ## add electronic state label
    if 'label' in kwargs and retval=='':
        retval =  kwargs.pop('label')
    elif 'label' in kwargs:
        retval = kwargs.pop('label')+'.'+retval
    ## prepend species
    if 'species' in kwargs and retval=='':
        retval =  kwargs.pop('species')
    elif 'species' in kwargs:
        retval =  kwargs.pop('species')+'_'+retval
    ## append all other quantum numbers in parenthese
    if len(kwargs)>0:
        t = []
        for key in kwargs:
            # if key not in Level.key_data['qn']: continue # only include defining quantum numbers
            # from .levels_transitions import Rotational_Level # HACK -- circular import
            # if key not in Rotational_Level._class_key_data['qn']: continue # only include defining quantum numbers
            if key in ('v','F'): # ints
                t.append(key+'='+str(int(kwargs[key])))
            elif key in ('Ω','Σ','SR'): 
                t.append(key+'='+format(kwargs[key],'g'))
            else:
                t.append(key+'='+str(kwargs[key]))
        retval = retval + '('+','.join(t)+')'
    return(retval)
    
def encode_bra_op_ket(qn1,operator,qn2):
    """Output a nicely encode ⟨A|O|B⟩."""
    return('⟨'+encode_level(**qn1)+'|'+operator+'|'+encode_level(**qn2)+'⟩')

def _format_matrix_element(qn1,qn2,operator=None,value=None):
    retval = f'⟨ {" ".join([format(key+"="+format(val,"g"),"5") for key,val in qn1.items()])} |' # bra
    if operator is not None:
        retval += f' {operator} '
    retval += f'| {" ".join([format(key+"="+format(val,"g"),"5") for key,val in qn2.items()])} ⟩' # ket
    if value is not None:
        retval += f' = {value}'
    return retval

def _diabaticise_eigenvalues(eigvals,eigvects):
    ## fractional character array
    c = eigvects.real**2
    ## mask of levels without confirmed assignments
    not_found = np.full(len(c), True) 
    ## find all mixing coefficients greater than
    ## 0.5 and fix their assignments
    for i in range(len(c)):
        j = np.argsort(-c[i,:])
        if c[i,j[0]] > 0.5:
            j = j[0]
            ii = list(range(len(c)))           # index of columns
            ii[i],ii[j] = j,i                  # swap largest c into diagonal position 
            c = c[:,ii]                        # swap for all columns
            eigvals,eigvects = eigvals[ii],eigvects[:,ii] # and eigvalues
            not_found[i] = False 
    ## trial all permutations of remaining
    ## unassigned levels, looking for the one
    ## which gives the best assignments
    if np.any(not_found):
        ## limit to mixing coefficient array of unassigned levels
        c_not_found = c[not_found,:][:,not_found]
        number_not_found = len(c_not_found)
        if number_not_found > 10:
            warnings.warn(f'Trialling all permutations of {number_not_found} levels.')
        ## loop through all permutations, looking for globally best metric
        best_permutation = None
        best_metric = 0
        for permutation in itertools.permutations(range(number_not_found)):
            ## metric is the smallest diagonal coefficient
            metric = np.min([tci[pi] for tci,pi in zip(c_not_found,permutation)])
            if metric > best_metric:
                best_permutation = permutation
                best_metric = metric
        ## reorder arrays to match best permuation
        best_permutation = np.array(best_permutation)
        c[:,not_found] = (c[:,not_found])[:,best_permutation]
        eigvects[:,not_found] = (eigvects[:,not_found])[:,best_permutation]
        eigvals[not_found] = (eigvals[not_found])[best_permutation]
    return eigvals,eigvects
