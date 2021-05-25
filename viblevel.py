from copy import copy,deepcopy
from pprint import pprint
from functools import lru_cache
import itertools
from time import perf_counter as timestamp
import warnings

import numpy as np
from numpy import nan,array
import sympy
from scipy import linalg

from . import levels,lines
from . import quantum_numbers
from . import tools
from . import dataset
from . import plotting
from .dataset import Dataset
from .tools import find,cache
from .optimise import Optimiser,P,optimise_method
from .kinetics import get_species,Species



class VibLevel(Optimiser):
    """A vibronic interaction matrix."""

    def __init__(
            self,
            name='viblevel',
            species='[14N][14N]',
            J=None,                       # compute on this, or default added
            # ef=None,                      # compute for these e/f parities , default ('e','f')
            experimental_level=None,      # a Level object for optimising relative to 
            # eigenvalue_ordering='maximise coefficients', # for deciding on the quantum number assignments of mixed level, options are 'minimise residual', 'maximise coefficients', 'preserve energy ordering' or None
            Eref=0.,       # energy reference relative to equilibrium energy, defaults to 0. -- not well defined
    ):
        self.name = name          # a nice name
        self.species = get_species(species)
        tkwargs = {'auto_defaults':True,'permit_nonprototyped_data':False}
        self.manifolds = {}
        self._shifts = []       # used to shift individual levels after diagonalisation
        self.level = levels.LinearDiatomic(name=f'{self.name}.level',**tkwargs)
        self.level.pop_format_input_function()
        self.level.add_suboptimiser(self)
        self.level.pop_format_input_function()
        self.vibrational_spin_level = levels.LinearDiatomic(**tkwargs) 
        self.interactions = Dataset() 
        self.J = J
        self.verbose = False
        self.sort_eigvals = True # try to reorder eigenvalues/eigenvectors after diagonalisation into diabatic levels
        ## inputs / outputs of diagonalisation
        self.eigvals = None
        self.eigvects = None
        ## a Level object containing data, better access through level property
        ## the optimiser
        Optimiser.__init__(self,name=self.name)
        self.pop_format_input_function()
        self.automatic_format_input_function(
            multiline=False,
            limit_to_args=('name', 'species', 'J', 'Eref',))
        self.add_save_to_directory_function(
            lambda directory: self.level.save(f'{directory}/level.h5'))
        ## compute residual error if a experimental level is provided
        self.experimental_level = experimental_level
        self._experimental_level_cache = {}
        if self.experimental_level is not None:
            self.add_suboptimiser(self.experimental_level)
        ## finalise construction
        self._initialise_construct()
        self.add_post_construct_function(self._finalise_construct)

    def _get_J(self):
        return self._J

    def _set_J(self,J):
        J_is_half_integer = self.species['nelectrons']%2==1
        if J is None:
            if J_is_half_integer:
                J = np.arange(0.5,30.5,1)
            else:
                J = np.arange(31)
        J = np.asarray(J)
        if J_is_half_integer:
            assert np.all(np.mod(J,1)==0.5),f'Half-integer J required for {repr(species)}'
        else:
            assert np.all(np.mod(J,1)==0),f'Integer J required for {repr(species)}'
        self._J = J
        self._clean_construct = True

    J = property(_get_J,_set_J)

    @optimise_method(add_format_input_function=False)
    def _initialise_construct(self,_cache=None):
        """Make a new array if a clean construct, or set to zero."""
        if self._clean_construct:
            self.H = np.full((len(self.J),len(self.vibrational_spin_level),len(self.vibrational_spin_level)),0.,dtype=complex)
        else:
            self.H *= 0    

    def _finalise_construct(self):
        """The actual matrix diagonlisation is done last."""
        ## if first run or model changed then construct Hamiltonian
        ## and blank rotational level, and determine which levels are
        ## actually allowed
        if self._clean_construct:
            ## levels that exist
            self.allowed = np.stack([self.vibrational_spin_level['Ω'] <= J for J in self.J])
            ## create a rotational level with all quantum numbers
            ## inserted and limit to allowed levels
            self.level.clear()
            self.level['J'] = np.concatenate([np.full(np.sum(i),J) for J,i in zip(self.J,self.allowed)])
            for key in self.vibrational_spin_level:
                self.level[key] = np.concatenate([self.vibrational_spin_level[key][i] for i in self.allowed])
            if self.experimental_level is not None:
                ## get all experimental_levels for defined
                ## levels, nan for missing
                self.level['Eref'] = nan
                self.level['Γref'] = nan
                iexp,imod = dataset.find_common(self.experimental_level,self.level,keys=self.level.defining_qn)
                self.level['Eref'][imod] = self.experimental_level['E'][iexp]
                self.level['Eref','unc'][imod] = self.experimental_level['E','unc'][iexp]
                self.level['Γref'][imod] = self.experimental_level['Γ'][iexp]
                self.level['Γref','unc'][imod] = self.experimental_level['Γ','unc'][iexp]
            self._finalise_construct_cache = dict(iexp=iexp,imod=imod)
        else:
            iexp = self._finalise_construct_cache['iexp']
            imod = self._finalise_construct_cache['imod']

        ## nothing to be done
        if len(self.vibrational_spin_level) == 0:
            return

        ## compute mixed energies and mixing coefficients
        self.eigvals = {}             # eignvalues
        self.eigvects = {}             # mixing coefficients
        for iJ,J in enumerate(self.J):
            H = self.H[iJ,:,:]
            iallowed = self.allowed[iJ]
            iforbidden = ~iallowed
            H[iforbidden,:] = H[:,iforbidden] = 0

            ## diagonalise independent block submatrices separately
            eigvals = np.zeros(self.H.shape[2],dtype=complex)
            eigvects = np.zeros(self.H.shape[1:3],dtype=float)
            for i in tools.find_blocks(H!=0,error_on_empty_block=False):
                He = H[np.ix_(i,i)]
                eigvalsi,eigvectsi = linalg.eig(He)
                if self.sort_eigvals:
                    eigvalsi,eigvectsi = _diabaticise_eigenvalues(eigvalsi,eigvectsi)
                eigvals[i] = eigvalsi
                eigvects[np.ix_(i,i)] = np.real(eigvectsi)
            eigvals = eigvals[iallowed]
            eigvects = eigvects[np.ix_(iallowed,iallowed)]

            ## reorder to get minimial differencd with reference data
            ## -- approximately
            if self.sort_eigvals:
                for ef in (+1,-1):
                    i = self.vibrational_spin_level.match(ef=ef)
                    i = i[self.vibrational_spin_level.match(Ω_max=J)]
                    j = self.level.match(J=J,ef=ef,Ω_max=J)
                    # print('DEBUG:', )
                    # print('DEBUG:', J,ef)
                    k = _permute_to_minimise_difference(eigvals[i],self.level.get('Eref',j))
                    # print('DEBUG:', k)
                    eigvals[i] = eigvals[i][k]
                    eigvects[np.ix_(i,i)] = eigvects[np.ix_(i,i)][np.ix_(k,k)]

            ## save eigenvalues
            self.eigvals[J] = eigvals
            self.eigvects[J] = eigvects

        ## insert energies into level
        t = np.concatenate(list(self.eigvals.values()))
        self.level['E'] = t.real
        self.level['Γ'] = t.imag

        ## compute residual if possible
        residual = np.concatenate((self.level['Eres'],self.level['Γres']),dtype=float)
        residual = residual[~np.isnan(residual)]
        return residual

    @optimise_method()
    def add_level(self,name,Γv=0,_cache=None,**kwargs):
        """Add a new electronic vibrational level. kwargs contains fitting
        parameters and optionally extra quantum numbers."""
        ## process inputs
        if self._clean_construct:
            ## all quantum numbers and molecular parameters
            kw = quantum_numbers.decode_linear_level(name) | kwargs
            kw['species'] = self.species.isotopologue
            if 'S' not in kw or 's' not in kw or 'Λ' not in kw:
                raise Exception('Quantum numbers S, s, and Λ are required.')
            ## check kwargs contains necessary quantum numbers
            for key in ('species','label','S','Λ','s','v'):
                if key not in kw:
                    raise Exception(f'Required quantum number: {key}')
            ## check kwargs contains only defined data
            for key in kw:
                if key not in (
                        'species','label','S','Λ','s','v','gu',
                        'Tv','Bv','Dv','Hv','Lv','Mv',
                        'Av','ADv','λv','λDv','λHv','γv','γDv',
                        'ov','pv','pDv','qv','qDv',
                        ):
                    raise Exception(f'Keyword argument not a known quantum number of Hamiltonian parameter: {repr(key)}')
            _cache['kw'] = kw
        kw = _cache['kw']
        ## Checks that integer/half-integer nature of J corresponds to
        ## quantum number S
        if kw['S']%1!=self.J[0]%1:
            raise Exception(f'Integer/half-integer nature of S and J do not match: {S%1} and {self.J[0]%1}')
        ## get quantum numbers and Hamiltonian
        if 'fH' not in _cache:
            ## get Hamiltonian and insert adjustable parameters into
            ## functions, including complex width
            ef,Σ,sH,fH = _get_linear_H(kw['S'],kw['Λ'],kw['s'])
            n = len(ef)
            ibeg = len(self.vibrational_spin_level)
            iend = ibeg + n
            _cache['n'],_cache['ef'],_cache['Σ'],_cache['fH'],_cache['ibeg'],_cache['iend'] = n,ef,Σ,fH,ibeg,iend
            ## add manifold data and list of vibrational_spin_levels
            if name in self.manifolds:
                raise Exception(f'Non-unique name: {repr(name)}')
            self.manifolds[name] = dict(ibeg=ibeg,iend=iend,ef=ef,Σ=Σ,n=len(ef),**kw) 
            self.vibrational_spin_level.extend(
                keys='new',ef=ef,Σ=Σ,
                **{key:kw[key] for key in ('species','label','S','Λ','s','v','gu') if key in kw},)
            ## make H bigger
            tH = np.full((len(self.J),len(self.vibrational_spin_level),len(self.vibrational_spin_level)),0.,dtype=complex)
            tH[:,:ibeg,:ibeg] = self.H
            self.H = tH
        n,ef,Σ,fH,ibeg,iend = _cache['n'],_cache['ef'],_cache['Σ'],_cache['fH'],_cache['ibeg'],_cache['iend']
        ## update H
        for i,j in np.ndindex((n,n)):
            self.H[:,i+ibeg,j+ibeg] = fH[i,j](self.J,**kw) + 1j*Γv

    @optimise_method()
    def add_spline_width(self,name,knots,ef=None,Σ=None,order=3):
        """Add complex width to manifold 'name' according to the given spline knots.. If ef and Σ are not none then set these levels only."""
        ## load data about this level from name
        kw = self.manifolds[name]
        ibeg = kw['ibeg']
        ## get indices to add width to
        i = np.full(kw['n'],True)
        if ef is not None:
            i &= kw['ef'] == ef
        if Σ is not None:
            i &= kw['Σ'] == Σ
        i = ibeg + tools.find(i)
        ## get spline points and J range
        xs = [knot[0] for knot in knots]
        ys = [knot[1] for knot in knots]
        Jbeg,Jend = np.min(xs),np.max(xs)
        iJ = (self.J>=Jbeg) & (self.J<=Jend)
        ## insert imaginary spline widths
        t = tools.spline(xs,ys,self.J[iJ],order=order)*1j
        for ii in i:
            self.H[iJ,i,i] += t

    @optimise_method()
    def add_coupling(
            self,
            name1,name2,        
            ηv=0,ηDv=0,         # LS -- spin-orbit coupling
            ξv=0,ξDv=0,         # JL -- L-uncoupling
            pv=0,pDv=0,         # JS -- S-uncoupling
            He=0,               # electronic coupling
            _cache=None):
        """Add spin-orbit coupling of two manifolds."""
        ## get matrix cache of matrix elements
        if self._clean_construct:
            kw1 = self.manifolds[name1]
            kw2 = self.manifolds[name2]
            ## get coupling matrices -- cached
            JL,JS,LS,NNJL,NNJS,NNLS = _get_offdiagonal_coupling(
                kw1['S'],kw1['Λ'],kw1['s'],kw2['S'],kw2['Λ'],kw2['s'],verbose=self.verbose)
            ibeg,jbeg = kw1['ibeg'],kw2['ibeg']
            _cache['ibeg'],_cache['jbeg'],_cache['JL'],_cache['JS'],_cache['LS'],_cache['NNJL'],_cache['NNLS'],_cache['NNLS'] = ibeg,jbeg,JL,JS,LS,NNJL,NNJS,NNLS
        ibeg,jbeg,JL,JS,LS,NNJL,NNJS,NNLS = _cache['ibeg'],_cache['jbeg'],_cache['JL'],_cache['JS'],_cache['LS'],_cache['NNJL'],_cache['NNLS'],_cache['NNLS']
        ## substitute into Hamiltonian (both upper and lowe diagonals, treated as real)
        for i,j in np.ndindex(JL.shape):
            t = (
                ηv*LS[i,j](self.J) 
                + ηDv*NNLS[i,j](self.J)
                + -ξv*JL[i,j](self.J) - ξDv*NNJL[i,j](self.J)
                + pv*JS[i,j](self.J) 
                + float(He)
            )
            self.H[:,i+ibeg,j+jbeg] += t
            self.H[:,j+jbeg,i+ibeg] += np.conj(t)

    def plot(self,fig=None,**plot_kwargs):
        if fig is None:
            fig = plotting.gcf()
        fig.clf()
        ax0 = plotting.subplot(0,fig=fig)
        ax0.set_title('E')
        legend_data = []
        for ilevel,(qn,m) in enumerate(self.level.unique_dicts_matches('species','label','v')):
            for isublevel,(qn2,m2) in enumerate(m.unique_dicts_matches('Σ','ef')):
                plot_kwargs |= dict(
                    color=plotting.newcolor(ilevel),
                    linestyle=plotting.newlinestyle(isublevel),
                    marker= plotting.newmarker(isublevel),
                    fillstyle='none',
                )
                tkwargs = plot_kwargs | {'label':quantum_numbers.encode_linear_level(**qn,**qn2) ,}
                legend_data.append(tkwargs)
                if self.experimental_level is None:
                    ax0.plot(m2['J'],m2['E'],**tkwargs)
                else:
                    ax1 = plotting.subplot(1,fig=fig)
                    ax1.set_title('Eres')
                    tkwargs = plot_kwargs | {'marker':'',}
                    ax0.plot(m2['J'],m2['E'],**tkwargs)
                    tkwargs = plot_kwargs | {'linestyle':'',}
                    ax0.errorbar(m2['J'],m2['Eref'],m2['Eref','unc'],**tkwargs)
                    ax1.errorbar(m2['J'],m2['Eres'],m2['Eres','unc'],**tkwargs)

        plotting.legend(*legend_data,show_style=True,ax=ax0)

        
class VibLine(Optimiser):
    
    """Calculate and optimally fit the line strengths of a band between
    two states defined by LocalDeperturbation objects. Currently only
    for single-photon transitions. """

    def __init__(self,name,level_u,level_l,J_l=None,ΔJ=None):
        ## add upper and lower levels
        self.name = name
        self.level_u = level_u
        self.level_l = level_l
        self.species = self.level_l.species
        tkwargs = {'auto_defaults':True, 'permit_nonprototyped_data':False,}
        self.line = lines.LinearDiatomic(name=f'{self.name}.line',**tkwargs)
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
        self.initialise_construct()

    @optimise_method(add_format_input_function=False)
    def initialise_construct(self,_cache=None):
        if self._clean_construct:
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
                        keys='new',
                        **{key+'_u':np.repeat(val,n_l) for key,val in self.level_u.vibrational_spin_level.items()},
                        **{key+'_l':np.tile(val,n_u) for key,val in self.level_l.vibrational_spin_level.items()},)


    def construct_lines(self):
        """Finalise construct."""
        ## build μ0
        for iu,jl,fμ in self._transition_moment_functions:
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                self.μ0[:,iΔJ,iu,jl] = fμ(self.J_l,ΔJ)
        ## could vectorise linalg with np.dot
        μs,E_us,E_ls,Γ_us,Γ_ls = [],[],[],[],[]
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
                E_us.append(np.repeat(self.level_u.eigvals[J_u].real,len(self.level_l.vibrational_spin_level)))
                E_ls.append(np.tile(  self.level_l.eigvals[J_l].real,len(self.level_u.vibrational_spin_level)))
                Γ_us.append(np.repeat(self.level_u.eigvals[J_u].imag,len(self.level_l.vibrational_spin_level)))
                Γ_ls.append(np.tile(  self.level_l.eigvals[J_l].imag,len(self.level_u.vibrational_spin_level)))
        ## add all new data to rotational line
        self.line['μ'] = np.ravel(μs)
        self.line['E_u'] = np.ravel(E_us)
        self.line['E_l'] = np.ravel(E_ls)
        self.line['Γ_u'] = np.ravel(Γ_us)
        self.line['Γ_l'] = np.ravel(Γ_ls)
        # ## remove forbidden lines
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

    @optimise_method()
    def add_transition_moment(self,name_u,name_l,μv=1,_cache=None):
        """Add constant transition moment. μv can be optimised."""
        """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        transitions. μv should be in atomic units and can be specifed
        as a value (optimisable), a function of R or a suboptimiser
        given ['μ']."""
        if self._clean_construct:
            ## get all quantum numbers
            kwu = self.level_u.manifolds[name_u]
            kwl = self.level_l.manifolds[name_l]
            ## get transition moment functions for all ef/Σ combinations
            ## and add optimisable parameter to functions
            fμ = _get_linear_transition_moment(kwu['S'],kwu['Λ'],kwu['s'],kwl['S'],kwl['Λ'],kwl['s'],verbose=self.verbose)
            _cache['kwu'],_cache['kwl'],_cache['fμ'] = kwu,kwl,fμ
        else:
            kwu,kwl,fμ = _cache['kwu'],_cache['kwl'],_cache['fμ']
        ## Add transition moment to μ0 array
        for i,j in np.ndindex(fμ.shape):
            if fμ[i,j] is None:
                continue
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                self.μ0[:,iΔJ,i+kwu['ibeg'],j+kwl['ibeg']] += fμ[i,j](self.J_l,ΔJ)*μv

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
        'Tv','Bv','Dv','Hv','Lv','Mv','Av','ADv','λv','λDv','λHv','γv','γDv','ov','pv','pDv','qv','qDv')}
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
    H = p['Tv']*I + p['Bv']*NN - p['Dv']*NN**2 + p['Hv']*NN**3 + p['Lv']*NN**4 + p['Mv']*NN**5
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

def _diabaticise_eigenvalues_in_blocks(eigvals,eigvects):
    """Diabaticise eigvects after first dividing into independent
    blocks."""
    ## find indices of independent blocks
    blocks = tools.find_blocks(np.abs(eigvects.real)>1e-1)
    ## build new arrays out of diabaticised independent blocks
    retval_eigvals = []
    retval_eigvects = []
    for i in blocks:
        t0,t1 = _diabaticise_eigenvalues(eigvals[i],eigvects[i,:][:,i])
        retval_eigvals.append(t0)
        retval_eigvects.append(t1)
    retval_eigvals = np.concatenate(retval_eigvals)
    retval_eigvects =  linalg.block_diag(*retval_eigvects)
    return retval_eigvals,retval_eigvects
        
def _diabaticise_eigenvalues(eigvals,eigvects):
    """Re-order eigvals/eigvects to maximise eigvects diagonal."""
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
            warnings.warn(f'Trialing all permutations of {number_not_found} levels.')
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

def _permute_to_minimise_difference(x,y):
    """Get rearrangment of x that approximately minimises its rms
    different relative to y."""
    t0,t1 = np.meshgrid(x,y)
    Δ = np.abs(t0-t1)
    k = np.arange(len(Δ))
    while np.any(~np.isnan(Δ)):
        with np.warnings.catch_warnings(): # this particular nan warning will be silenced for this block, occurs when some rows of Δ are all NaN
            np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            i = np.nanargmin(np.nanmin(Δ,1))
            j = np.nanargmin(Δ[i,:])
            if i != j:
                Δ[i,:],Δ[j,:] = Δ[j,:],Δ[i,:]
                k[i],k[j] = k[j],k[i]
            Δ[j,:] = Δ[:,j] = np.nan
    return k

def calc_viblevel(
        name=None,
        species=None,J=None,
        levels=None,         # {name:add_level_kwargs,...}
        couplings=None, # None or {name1,name2:add_coupling_kwargs,...}
        spline_widths=None, # None or {name:add_spline_width_kwargs,...}
):
    """Compute a VibLevel model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_level, add_spline_width etc."""
    v = VibLevel(name=name,species=species,J=J)
    if levels is not None:
        for name,kwargs in levels.items():
            v.add_level(name,**kwargs)
    if couplings is not None:
        for (name1,name2),kwargs in couplings.items():
            v.add_coupling(name1,name2,**kwargs)
    if spline_widths is not None:
        for name,kwargs in spline_widths.items():
            v.add_spline_width(name,**kwargs)
    v.construct()
    return v

def calc_level(*args_viblevel,**kwargs_viblevel):
    """Compute a VibLevel model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_level, add_spline_width etc."""
    v = calc_viblevel(*args_viblevel,**kwargs_viblevel)
    return v.level

def calc_line(
        species=None,J_l=None,ΔJ=None,
        upper=None,           # kwargs for calc_level
        lower=None,           # kwargs for calc_level
        transition_moments=None,# None or {name_u,name_l:add_transition_moment,...}
):
    """Compute a VibLevel model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_level, add_spline_width etc."""
    upper['species'] = lower['species'] = species
    upper = calc_viblevel(**upper)
    lower = calc_viblevel(**lower)
    v = VibLine('vibline',upper,lower,J_l=J_l,ΔJ=ΔJ)
    if transition_moments is not None:
        for (name1,name2),kwargs in transition_moments.items():
            v.add_transition_moment(name1,name2,**kwargs)
    v.construct()
    return v.line

