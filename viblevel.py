from copy import copy,deepcopy
from pprint import pprint
from functools import lru_cache
import itertools
import warnings

import numpy as np
from numpy import nan,array,arange
import sympy
from scipy import linalg

from . import levels,lines
from . import quantum_numbers
from . import tools
from . import dataset
from . import database
from .exceptions import DatabaseException
from . import plotting
from .dataset import Dataset
from .tools import find,cache,timestamp
from .optimise import Optimiser,P,Parameter,optimise_method,format_input_method,format_input_class
from .database import get_species_property

@format_input_class()
class Level(Optimiser):
    """A vibronic interaction matrix."""

    def __init__(
            self,
            name='viblevel',
            species=None,
            J=None,                       # compute on this, or default added
            # ef=None,                      # compute for these e/f parities , default ('e','f')
            experimental_level=None,      # a Level object for optimising relative to 
            # eigenvalue_ordering='maximise coefficients', # for deciding on the quantum number assignments of mixed level, options are 'minimise residual', 'maximise coefficients', 'preserve energy ordering' or None
            Eref=0.,       # energy reference relative to equilibrium energy, defaults to 0. -- not well defined
            Zsource='self',
            sort_diabaticise=True,
            sort_manifolds=True,
            sort_match_experiment=False,
    ):
        Optimiser.__init__(self,name=name)
        self.species = get_species_property(species,'isotopologue_formula')
        self.Zsource = Zsource
        self.Eref = Eref
        self._manifolds = {}
        self._shifts = []       # used to shift individual levels after diagonalisation
        self._level = levels.Diatom(name=f'{self.name}.level')
        self._level.pop_format_input_function()
        self._level.add_suboptimiser(self,
                                     # construct_now=False,
                                             )
        self.vibrational_spin_level = levels.Diatom()
        self.interactions = {}
        self.verbose = False
        ##  try to reorder eigenvalues/eigenvectors after
        ##  diagonalisation into diabatic levels
        self.sort_diabaticise = sort_diabaticise
        self.sort_manifolds = sort_manifolds
        self.sort_match_experiment = sort_match_experiment
        ## inputs / outputs of diagonalisation
        self.eigvals = None
        self.eigvects = None
        self.add_save_to_directory_function(
            lambda directory: self._level.save(f'{directory}/level.h5'))
        ## set J
        J_is_half_integer = get_species_property(self.species,'nelectrons')%2==1
        if J is None:
            if J_is_half_integer:
                J = np.arange(0.5,30.5,1)
            else:
                J = np.arange(31)
        J = np.asarray(J)
        if J_is_half_integer:
            if not  np.all(np.mod(J,1)==0.5):
                raise Exception(f'Half-integer J required for {self.species!r}')
        else:
            if not np.all(np.mod(J,1)==0):
                raise Exception(f'Integer J required for {self.species!r}')
        self.J = J
        ## compute residual error if a experimental level is provided
        self.experimental_level = experimental_level
        self._experimental_level_cache = {}
        if self.experimental_level is not None:
            self.add_suboptimiser(self.experimental_level)
        ## finalise construction
        self._initialise_construct()
        self.add_post_construct_function(self._finalise_construct)

    ## make sure constructed before accessing level object
    def _get_level(self):
        return self._level
    level = property(_get_level)

    def get_electronic_vibrational_level(self):
        retval = levels.Diatom()
        for i,m in enumerate(self._manifolds.values()):
            retval.append(
                {key:m[key] for key in m.keys()
                 if key not in ('ef','ibeg','iend','n','Σ','Ω',)},)
        return retval

    def get_electronic_vibrational_interactions(self):
        print( self.interactions)
        retval = lines.Diatom()
        for (name1,name2),d in self.interactions.items():
            retval.append(
                encoded_qn=f'{name1}–{name2}',
                **{key:val for key,val in d.items() if val!=0},)
        return retval

    def get_pgopher_linearmanifold(self):
        """Return an xml fragment describing this viblevel as a pgopher LinearManifold."""
        lines = []
        lines.append(f'<LinearManifold Name="{self.name}" LimitSearch="True">')
        ## add manifolds
        level = self.get_electronic_vibrational_level()
        encode_lambda = {(0,0):'Sigma+', (0,1):'Sigma-',
                         (1,0):'Pi', (2,0):'Delta',
                         (3,0):'Phi', (4,0):'Gamma',}
        for i in range(len(level)):
            lines.append('')
            line = []
            line.append(r'<Linear Name="'+level.encode_qn({key:level[key,i] for key in ('label','S','Λ','s','gu','v') if level.is_known(key)})+'"')
            line.append('Lambda="'+encode_lambda[level['Λ',i],level['s',i]]+'"')
            line.append('S="'+format(level['S',i]*2,'g')+'"')
            line.append('>')
            lines.append(' '.join(line))
            for key1,key2 in (
                    ('Tv','Origin'), ('Bv','B'), ('Dv','D'),
                    ('Hv','H'), ('Lv','L'), ('Mv','M'), ('Nv','N'), ('Ov','O'),
                    ('Av','A'), ('ADv','AD'),
                    ('λv','LambdaSS'), ('λD','LambdaD'), ('λH','LambdaH'),
                    ('γv','gamma'), ('γDv','gammaD'), ('γHv','gammaH'), ('γLv','gammaL'),
                    ('ov','o'), ('oDv','oD'), ('oHv','oH'), ('oLv','oL'),
                    ('pv','p'), ('pDv','pD'), ('pHv','pH'), ('pLv','pL'),
                    ('qv','q'), ('qDv','qD'), ('qHv','qH'), ('qLv','qL'),
                    ):
                if level.is_set(key1):
                    lines.append(f'  <Parameter Name="{key2}" Value="{level[key1,i]}"/>',)
            lines.append(r'</Linear>')
        ## add interactions
        for (name1,name2),d in self.interactions.items():
            for key1,key2 in (
                    ('ηv','LS'),
                    ('ξv','Luncouple'),
            ):
                if key1 in d and d[key1] !=0:
                    lines.append('')
                    lines.append(f'<LinearPerturbation Op="{key2}" Bra="{name1}" Ket="{name2}">')
                    lines.append(f'  <Parameter Name="Value" Value="{d[key1]}"/>')
                    lines.append(f'</LinearPerturbation>')
        ## join lines
        lines.append('')
        lines.append(f'</LinearManifold>')
        retval = '\n'.join(lines)
        return retval

    

    @optimise_method(add_format_input_function=False)
    def _initialise_construct(self,_cache=None):
        """Make a new array if a clean construct, or set to zero."""
        if self._clean_construct:
            self.vibrational_spin_level.clear()
            self._manifolds.clear()
            ## initialise Hamiltonian matrices
            self.H = np.full(
                (len(self.J),
                 len(self.vibrational_spin_level),
                 len(self.vibrational_spin_level)),
                0.0,
                dtype=complex)
        else:
            self.H *= 0    

    def _finalise_construct(self):
        """The actual matrix diagonlisation is done last."""
        ## if first run or model changed then construct Hamiltonian
        ## and blank rotational level, and determine which levels are
        ## actually allowed
        if self._clean_construct:
            ## create a rotational level with all quantum numbers
            ## inserted and limit to allowed levels
            self._level.clear()
            self._level['Eref'] = self.Eref
            self._level['Zsource'] =self.Zsource
            self._level['J'] = np.repeat(self.J,len(self.vibrational_spin_level))
            for key in self.vibrational_spin_level:
                self._level[key] = np.tile(self.vibrational_spin_level[key],len(self.J))
            ## limit to allowed levels
            self._iallowed = self._level['J'] >= self._level['Ω']
            self._level.index(self._iallowed)
            if self.experimental_level is not None:
                ## get all experimental_levels for defined
                ## levels, nan for missing
                self._level['Eexp'] = nan
                self._level['Γexp'] = nan
                iexp,imod = dataset.find_common(self.experimental_level,self._level,keys=self._level.defining_qn)
                self._level['Eexp'][imod] = self.experimental_level['E'][iexp]
                self._level['Eexp','unc'][imod] = self.experimental_level['E','unc'][iexp]
                if self.experimental_level.is_known('Γ'):
                    self._level['Γexp'][imod] = self.experimental_level['Γ'][iexp]
                    self._level['Γexp','unc'][imod] = self.experimental_level['Γ','unc'][iexp]
                self._finalise_construct_cache = dict(iexp=iexp,imod=imod)
        else:
            if self.experimental_level is not None:
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
            # H[np.isnan(H)] = 0.0
            iallowed = ~np.isnan(np.diag(H).real)
            Hallowed = H[np.ix_(iallowed,iallowed)]
            ## diagonalise independent block submatrices separately
            eigvals = np.zeros(self.H.shape[2],dtype=complex)
            eigvects = np.zeros(self.H.shape[1:3],dtype=float)
            for nblock,iblock in enumerate(tools.find_blocks(Hallowed!=0,error_on_empty_block=False)):
                k = find(iallowed)[iblock]
                Hblock = Hallowed[np.ix_(iblock,iblock)]
                eigvalsi,eigvectsi = linalg.eig(Hblock)
                if self.sort_diabaticise:# and self._clean_construct:
                    eigvalsi,eigvectsi = _diabaticise_eigenvalues_sort_coefficients(eigvalsi,eigvectsi)
                ## index of this block into vibrational_spin_levels
                ## energy sort each vibrational_spin_level, already
                ## blocked by ef symmetry
                if self.sort_manifolds:
                    E0 = np.diag(Hblock).real # unperturbed energies of this block
                    for qn,m in tools.unique_combinations_masks(
                            self.vibrational_spin_level['label'][k],
                            self.vibrational_spin_level['v'][k]):
                        if np.sum(m) > 1:
                            i = np.argsort(eigvalsi[m])
                            j = np.argsort(np.argsort(E0[m]))
                            l = i[j]
                            eigvalsi[m] = eigvalsi[m][l]
                            eigvectsi[:,m] = eigvectsi[:,m][:,l]
                ## save block eigvals
                eigvals[k] = eigvalsi
                eigvects[np.ix_(k,k)] = np.real(eigvectsi)
            ## reorder to get approximate minimal difference with
            ## respect reference data
            if self.sort_match_experiment and self.experimental_level is not None:
                for ef in (+1,-1):
                    i = self.vibrational_spin_level.match(ef=ef,Ω_max=J)
                    j = self._level.match(J=J,ef=ef,Ω_max=J)
                    k = _permute_to_minimise_difference(eigvals[i],self._level['Eexp',j])
                    eigvals[i] = eigvals[i][k]
                    eigvects[np.ix_(i,i)] = eigvects[np.ix_(i,i)][np.ix_(k,k)]
            ## save eigenvalues
            self.eigvals[J] = eigvals
            self.eigvects[J] = eigvects
        ## insert energies into allowed levels
        t = np.concatenate(list(self.eigvals.values()))
        t = t[self._iallowed]
        self._level['E'] = t.real
        self._level['Γ'] = t.imag
        ## compute residual if possible
        if self.experimental_level is not None:
            residual = np.concatenate((self._level['Eres'],self._level['Γres']),dtype=float)
            residual = residual[~np.isnan(residual)]
            return residual

    @optimise_method()
    def add_manifold(self,name=None,Γv=0,_cache=None,**kwargs):
        """Add a new electronic vibrational level. kwargs contains fitting
        parameters and optionally extra quantum numbers."""
        ## process inputs
        if self._clean_construct:
            ## collect all quantum numbers and molecular parameters
            kw = {}
            if name is not None:
                kw |= quantum_numbers.decode_linear_level(name) 
            kw |= kwargs
            kw['species'] = self.species
            for key in ('S','s','Λ'):
                if key not in kw:
                    if 'species' in kw and 'label' in kw:
                        from . import database
                        try:
                            kw[key] = database.get_electronic_state_property(
                                kw['species'],kw['label'],key)
                        except DatabaseException as err:
                            raise Exception(f'Quantum number {key!r} is required and could not be computed from the database.')
                    else:
                        raise Exception(f'Quantum number {key!r} is required or "species" and "label" to use the database.')
            ## ## if name not given, then generate from quantum numbers
            ## if name is None:
            ##     name = quantum_numbers.encode_linear_level(
            ##         {key:kw[key]
            ##          for key in ('label','Λ','S','s','gu','v',)
            ##          if key in kw})
            ## check kwargs contains necessary quantum numbers
            for key in ('species','label','S','Λ','s','v'):
                if key not in kw:
                    raise Exception(f'Required quantum number: {key}')
            ## check kwargs contains only defined data
            allowed_kwargs = (
                'species','label','S','Λ','s','v','gu',
                'Tv','Bv','Dv','Hv','Lv','Mv','Nv','Ov',
                'Av','ADv','λv','λDv','λHv','γv','γDv',
                'ov','pv','pDv','qv','qDv',)
            for key in kw:
                if key not in allowed_kwargs:
                    raise Exception(f'Keyword argument {repr(key)} is not a known quantum number of Hamiltonian parameter. Allowed kwargs: {allowed_kwargs} ')
            ## set differentiation stepsize
            stepsizes = {'Tv':1e-3, 'Bv':1e-6, 'Dv':1e-9, 'Hv':1e-13,
                         'Lv':1e-15, 'Mv':1e-17, 'Nv':1e-19, 'Ov':1e-21,
                         'Av':1e-3, 'ADv':1e-7, 'λv':1e-5,
                         'λDv':1e-8, 'λHv':1e-11, 'γv':1e-3, 'γDv':1e-7, 'ov':1e-3,
                         'pv':1e-3, 'pDv':1e-7, 'qv':1e-6, 'qDv':1e-7,}
            for key in kw:
                if isinstance(kw[key],Parameter) and key in stepsizes:
                    kw[key].step = stepsizes[key]
            _cache['kw'] = kw
            ## Checks that integer/half-integer nature of J corresponds to
            ## quantum number S
            if kw['S']%1!=self.J[0]%1:
                raise Exception(f'Integer/half-integer nature of S and J do not match: {S%1} and {self.J[0]%1}')
            ## get Hamiltonian and insert adjustable parameters into
            ## functions, including complex width
            ef,Σ,sH,fH = _get_linear_H(kw['S'],kw['Λ'],kw['s'])
            n = len(ef)
            ibeg = len(self.vibrational_spin_level)
            iend = ibeg + n
            Ω = kw['Λ'] + Σ
            _cache['n'],_cache['ef'],_cache['Σ'],_cache['fH'],_cache['ibeg'],_cache['iend'] = n,ef,Σ,fH,ibeg,iend
            ## add manifold data and list of vibrational_spin_levels
            if name in self._manifolds:
                raise Exception(f'Non-unique name: {repr(name)}')
            self._manifolds[name] = dict(ibeg=ibeg,iend=iend,ef=ef,Σ=Σ,n=len(ef),Ω=Ω,**kw)
            ## set values missing in existing data to 0
            for key in kw:
                if key not in self.vibrational_spin_level:
                    self.vibrational_spin_level[key] = 0
            self.vibrational_spin_level.extend(ef=ef,Σ=Σ, **{key:kw[key] for key in kw},)
            ## make H bigger
            tH = np.full((len(self.J),len(self.vibrational_spin_level),len(self.vibrational_spin_level)),0.,dtype=complex)
            tH[:,:ibeg,:ibeg] = self.H
            self.H = tH
            _cache['fH'] = fH
            ## cache indices of valid J
            _cache['iJ'] = [self.J>=Ωi for Ωi in Ω]
        n,ef,Σ,fH,ibeg,iend,iJ,kw = _cache['n'],_cache['ef'],_cache['Σ'],_cache['fH'],_cache['ibeg'],_cache['iend'],_cache['iJ'],_cache['kw']
        ## update H
        for i,j in np.ndindex((n,n)):
            k = iJ[i] & iJ[j]
            self.H[k,i+ibeg,j+ibeg] = fH[i,j](self.J[k],**kw) + 1j*Γv
            self.H[~k,i+ibeg,j+ibeg] = nan

    add_level = add_manifold    # deprecated


    @optimise_method()
    def add_spline_width(self,name,knots,ef=None,Σ=None,order=3):
        """Add complex width to manifold 'name' according to the given spline knots.. If ef and Σ are not none then set these levels only."""
        ## load data about this level from name
        kw = self._manifolds[name]
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

    @optimise_method(format_multi_line=4)
    def add_coupling(
            self,
            name1,name2,        
            ηv=0,ηDv=0,         # LS -- spin-orbit coupling
            ξv=0,ξDv=0,         # JL -- L-uncoupling
            HJSv=0,HJSDv=0,         # JS -- S-uncoupling
            Hev=0,               # electronic coupling
            λvtest=0,               # spin-spin off-diagonal coupling, not final formulation
            _cache=None):
        """Add spin-orbit coupling of two manifolds."""
        assert HJSDv == 0, 'not implemented'
        ## save all interactions
        self.interactions[name1,name2] = {
            'ηv':ηv, 'ηDv':ηDv,
            'ξv':ξv, 'ξDv':ξDv,
            'HJSv':HJSv, 'HJSDv':HJSDv,
            'Hev':Hev,
        }
        ## get matrix cache of matrix elements
        if self._clean_construct:
            kw1 = self._manifolds[name1]
            kw2 = self._manifolds[name2]
            S1,S2 = kw1['S'],kw2['S'],
            s1,s2 = kw1['s'],kw2['s'],
            Λ1,Λ2 = kw1['Λ'],kw2['Λ'],
            Σ1,Σ2 = kw1['Σ'],kw2['Σ']
            Ω1,Ω2 = kw1['Ω'],kw2['Ω']
            ef1,ef2 = kw1['ef'],kw2['ef'],
            ibeg,jbeg = kw1['ibeg'],kw2['ibeg']
            ## get coupling matrices -- cached
            JL,JS,LS,NNJL,NNJS,NNLS = _get_offdiagonal_coupling(
                S1,Λ1,s1,S2,Λ2,s2,verbose=self.verbose)
            ## find indices of valid J
            iJ1 = [self.J>=Ω for Ω in Ω1]
            iJ2 = [self.J>=Ω for Ω in Ω2]
            ## get mask for diagonal (Λ,S,Σ,ef,)~(Λ,S,Σ,ef) transitions
            ΛSΣefdiag = np.array([[
                    S1==S2 and Λ1==Λ2 and Σ1i==Σ2i and ef1i==ef2i
                        for (Σ2i,ef2i) in zip(Σ2,ef2)]
                  for (Σ1i,ef1i) in zip(Σ1,ef1)])
            ## get spin-spin 3-j coefficients while applying other
            ## selection rules, see Sec. 3.4.4 and Eq. 3.4.49 of
            ## lefebvre-brion_field2004
            SS = np.full((len(Σ1),len(Σ2)),0.0)
            for i,(Σ1i,ef1i,Ω1i) in enumerate(zip(Σ1,ef1,Ω1)):
                for j,(Σ2j,ef2j,Ω2j) in enumerate(zip(Σ2,ef2,Ω2)):
                    if (Λ1==0 and S1<=1/2) or (Λ2==0 and S2<=1/2):
                        continue
                    if 'gu' in kw1 and kw1['gu']!=kw2['gu']:
                        continue
                    if Λ1==0 and Λ2==0 and s1==s2:
                        continue
                    if ef1i != ef2j:
                        continue
                    if Ω1i != Ω2j:
                        continue
                    SS[i,j] = quantum_numbers.wigner3j(S1,2,S2,-Σ1i,Σ1i-Σ2j,Σ2j)
            ## save cache
            _cache |= dict(
                ibeg=ibeg,jbeg=jbeg,
                JL=JL,JS=JS,LS=LS,
                NNJL=NNJL,NNJS=NNJS,NNLS=NNLS,
                iJ1=iJ1,iJ2=iJ2,ΛSΣefdiag=ΛSΣefdiag,SS=SS)
        ## load cache
        ibeg,jbeg,JL,JS,LS,NNJL,NNJS,NNLS,iJ1,iJ2,ΛSΣefdiag,SS = (
            _cache['ibeg'],_cache['jbeg'],
            _cache['JL'],_cache['JS'],_cache['LS'],
            _cache['NNJL'],_cache['NNJS'],_cache['NNLS'],
            _cache['iJ1'],_cache['iJ2'],_cache['ΛSΣefdiag'],_cache['SS'])
        ## substitute into Hamiltonian (both upper and lowe diagonals, treated as real)
        for i,j in np.ndindex(JL.shape):
            iJ = iJ1[i] & iJ2[j]
            J = self.J[iJ]
            ## sum up all possible nonzero interactions
            H = 0.0
            for pi,Hi in (
                (ηv, ηv*LS[i,j](J)), # LS
                (ηDv, ηDv*NNLS[i,j](J)), # LS centrifugal
                (ξv, -ξv*JL[i,j](J)),    # JL
                (ξDv, -ξDv*NNJL[i,j](J)), # JL centrifugal
                (HJSv, HJSv*JS[i,j](J)), # JS, better symbol needed
                (Hev, float(Hev)*ΛSΣefdiag[i,j]), # electronic
                (λvtest, λvtest*SS[i,j]),               # spin-spin off-diagonal coupling, not final formulation
            ):
                if isinstance(pi,Parameter) or pi!=0:
                    H += Hi
            ## add to self
            self.H[iJ,i+ibeg,j+jbeg] += H
            self.H[iJ,j+jbeg,i+ibeg] += np.conj(H)

    def load_from_pgopher(
            self,
            filename,
            match=None,
            not_keys=(),
            **load_pgopher_constants_kwargs
    ):
        ## load molecular constants and perturbations
        level = levels.Diatom()
        level.load_pgopher_constants(filename,**load_pgopher_constants_kwargs)
        line = lines.Diatom()
        line.load_pgopher_constants(filename,**load_pgopher_constants_kwargs)
        ## limit to matches
        level.limit_to_match(match)
        tmatch = {}
        for key,val in match.items():
            tmatch[f'{key}_u'] = tmatch[f'{key}_l'] = val
        line.limit_to_match(tmatch)
        ## limit to keys if specified
        for key in not_keys:
            if key in level:
                level.unset(key)
            if key in line:
                line.unset(key)
        ## add to self
        self.add_manifolds_from_level(level)
        self.add_couplings_from_line(line)

    def add_manifolds_from_level(self,level):
        ## add mainfolds to self
        for row in level.rows():
            qn,p ={},{}
            for key in row:
                if key in ('label','v','Λ','S','s','gu',):
                    qn[key] = row[key]
                elif row[key] != 0:
                    p[key] = row[key]
            self.add_manifold(name=level.encode_qn(qn),**p)

    def add_couplings_from_line(self,line):
        """Add couplings in line object to self."""
        for row in line.rows():
            ## separate qn and matrix elements
            qnu,qnl,p = {},{},{}
            for key in row:
                if key in  ('label_u','v_u','Λ_u','S_u','s_u','gu_u'):
                    qnu[key[:-2]] = row[key]
                elif key in  ('label_l','v_l','Λ_l','S_l','s_l','gu_l'):
                    qnl[key[:-2]] = row[key]
                elif row[key] != 0:
                    p[key] = row[key]
            ## find levels corresponding to his interaction
            name1 = name2 = None
            for name,t in self._manifolds.items():
                if np.all([t[key] == qnu[key] for key in qnu]):
                    assert name1 is None
                    name1 = name
                if np.all([t[key] == qnl[key] for key in qnl]):
                    assert name2 is None
                    name2 = name
            ## add coupling
            self.add_coupling(name1=name1,name2=name2,**p)

    def plot(
            self,
            fig=None,
            ylim_E=None,
            ylim_Eresidual=None,
            plot_errorbars=True,
            reduce_coefficients=(0,),
            match=None,
            ## plot resiudal as histogram
            plot_histogram=True,
            normalise_histogram=False,
            nbins_histogram=None,
            **plot_kwargs,):
        """Plot data and residual error."""
        self.construct()
        if fig is None:
            fig = plotting.gcf()
        fig.clf()
        ## plot energy levels
        axE = plotting.subplot(0,fig=fig)
        axE.set_title('E')
        legend_data = []
        ## reducing to plot only matching levels
        level = self.level
        if match is not None:
            level = level.matches(match)
        for ilevel,(qn,m) in enumerate(level.unique_dicts_matches('species','label','v')):
            for isublevel,(qn2,m2) in enumerate(m.unique_dicts_matches('Σ','ef')):
                plot_kwargs |= dict(
                    color=plotting.newcolor(ilevel),
                    linestyle=plotting.newlinestyle(isublevel),
                    marker= plotting.newmarker(int(qn2['Σ'])-m['Σ'].min()),
                    fillstyle=('bottom' if qn2['ef']==1 else 'top'))
                tkwargs = plot_kwargs | {'label':quantum_numbers.encode_linear_level(**qn,**qn2) ,}
                legend_data.append(tkwargs)
                ΔEreduce = np.polyval(reduce_coefficients,m2['J']*(m2['J']+1))
                if self.experimental_level is None:
                    tkwargs = plot_kwargs
                    axE.plot(m2['J'],m2['E']-ΔEreduce,**tkwargs)
                else:
                    ## plot E residual
                    axEres = plotting.subplot(1,fig=fig)
                    axEres.set_title('Eres')
                    tkwargs = plot_kwargs | {'marker':'',}
                    axE.plot(m2['J'],m2['E']-ΔEreduce,**tkwargs)
                    Ekwargs = plot_kwargs | {'linestyle':'',}
                    Ereskwargs = plot_kwargs | {'linestyle':'-',}
                    if plot_errorbars:
                        i = ~np.isnan(m2['Eexp'])
                        axE.errorbar(m2['J'][i],m2['Eexp'][i]-ΔEreduce[i],m2['Eexp','unc'][i],**Ekwargs)
                        i = ~np.isnan(m2['Eres'])
                        axEres.errorbar(m2['J'][i],m2['Eres'][i],m2['Eres','unc'][i],**Ereskwargs)
                    else:
                        axE.plot(m2['J'],m2['Eexp']-ΔEreduce,**Ekwargs)
                        axEres.plot(m2['J'],m2['Eres'],**Ereskwargs)
                    if ylim_Eresidual is not None:
                        if np.isscalar(ylim_Eresidual):
                            ylim_Eresidual = (-ylim_Eresidual,ylim_Eresidual)
                        axEres.set_ylim(ylim_Eresidual)
                if ylim_E is not None:
                    axE.set_ylim(ylim_E)
                ## plot linewidths
                if np.any(level['Γ'] > 0):
                    axΓ = plotting.subplot(2,fig=fig)
                    axΓ.set_title('Γ')
                    if self.experimental_level is None:
                        tkwargs = plot_kwargs
                        axΓ.plot(m2['J'],m2['Γ'],**tkwargs)
                    else:
                        axΓres = plotting.subplot(3,fig=fig)
                        axΓres.set_title('Γres')
                        tkwargs = plot_kwargs | {'marker':'',}
                        axΓ.plot(m2['J'],m2['Γ'],**tkwargs)
                        tkwargs = plot_kwargs | {'linestyle':'',}
                        if plot_errorbars:
                            axΓ.errorbar(m2['J'],m2['Γexp'],m2['Γexp','unc'],**tkwargs)
                            axΓres.errorbar(m2['J'],m2['Γexp'],m2['Γexp','unc'],**tkwargs)
                        else:
                            axΓ.plot(m2['J'],m2['Γexp'],**tkwargs)
                            axΓres.plot(m2['J'],m2['Γexp'],**tkwargs)
        ## plot E residual histogram
        if plot_histogram and self.experimental_level is not None:
            ax = plotting.subplot(fig=fig)
            if nbins_histogram is None:
                nbins_histogram = int(len(level)/20)
            if normalise_histogram:
                ax.set_title('Eres (normalised)')
                ax.hist(level['Eres']/level['Eres','unc'],nbins_histogram)
            else:
                ax.set_title('Eres')
                ax.hist(level['Eres'],nbins_histogram)
        plotting.legend(*legend_data,show_style=True,ax=axE)

        
@format_input_class()
class Line(Optimiser):
    
    """Calculate and optimally fit the line strengths of a band between
    two states defined by LocalDeperturbation objects. Currently only
    for single-photon transitions. """

    def __init__(self,name,level,ΔJ=(-1,0,+1),Eref=0,Zsource='self'):
        ## add upper and lower levels
        self.name = name
        self.level = self.level_u = self.level_l = level
        self.species = self.level_l.species
        self.Zsource = Zsource
        self.level_u.Eref = self.level_l.Eref = self.Eref = Eref
        self.ΔJ = ΔJ
        self.J_l = level.J
        ## construct optimiser -- inheriting from states
        Optimiser.__init__(self,name=self.name)
        ## internal line.Diatom object containing compute transitions
        self._line = lines.Diatom(name=f'{self.name}.line')
        self._line.pop_format_input_function()
        self._line.add_suboptimiser(self)
        self.add_suboptimiser(self.level)
        def f(directory): 
            self._line.save(directory+'/line.h5')
        self.add_save_to_directory_function(f)
        self.add_post_construct_function(self._finalise_construct)
        self._initialise_construct()

    ## make sure constructed before accessing line object
    def _get_line(self):
        self.construct()
        return self._line
    line = property(lambda self:self._get_line())

    @optimise_method(add_format_input_function=False)
    def _initialise_construct(self,_cache=None):
        if self._clean_construct:
            ## initialise μ0
            self.μ0 = np.full(
                (len(self.J_l),
                 len(self.ΔJ),
                 len(self.level_u.vibrational_spin_level),
                 len(self.level_l.vibrational_spin_level),),
                0.)
            ## add quantum numbers to line
            self._line.clear()
            for iJ_l,J_l in enumerate(self.J_l):
                for iΔJ,ΔJ in enumerate(self.ΔJ):
                    J_u = J_l + ΔJ
                    if J_u not in self.level_u.J:
                        continue
                    ## add levels
                    self._line.extend(
                        J_u=J_u,
                        J_l=J_l,
                        **{key+'_u':np.repeat(val,len(self.level_l.vibrational_spin_level))
                           for key,val in self.level_u.vibrational_spin_level.items()},
                        **{key+'_l':  np.tile(val,len(self.level_u.vibrational_spin_level))
                           for key,val in self.level_l.vibrational_spin_level.items()},)
            ## limit to levels that exist
            self._iallowed = (
                ## levels exist
                ((self._line['J_l'] >= self._line['Ω_l']) & (self._line['J_u'] >= self._line['Ω_u']))
                ## ef/ΔJ transition selection rules
                & ((       ((self._line['J_u']-self._line['J_l']) == 0) & (self._line['ef_u'] != self._line['ef_l']))
                   |((np.abs(self._line['J_u']-self._line['J_l']) == 1) & (self._line['ef_u'] == self._line['ef_l']))))
            self._line.index(self._iallowed)
            ## set some more things
            self._line['Zsource'] = self.Zsource
            self._line['Eref'] = self.Eref
        else:
            self.μ0 *= 0.

    def _finalise_construct(self):
        """Finalise construct."""
        ## could vectorise linalg with np.dot
        μs,E_us,E_ls,Γ_us,Γ_ls = [],[],[],[],[]
        for iJ_l,J_l in enumerate(self.J_l):
            for iΔJ,ΔJ in enumerate(self.ΔJ):
                J_u = J_l + ΔJ
                if J_u not in self.level_u.J:
                    continue
                ## compute mixed line strengths
                c_l = self.level_l.eigvects[J_l]
                c_u = self.level_u.eigvects[J_u]
                μ0 = self.μ0[iJ_l,iΔJ,:,:]
                μ = np.dot(np.transpose(c_u),np.dot(μ0,c_l))
                μs.append(μ)
                ## get energy levels for this J_u,J_l transition
                E_us.append(np.repeat(self.level_u.eigvals[J_u].real,len(c_l)))
                E_ls.append(  np.tile(self.level_l.eigvals[J_l].real,len(c_u)))
                Γ_us.append(np.repeat(self.level_u.eigvals[J_u].imag,len(c_l)))
                Γ_ls.append(  np.tile(self.level_l.eigvals[J_l].imag,len(c_u)))
        ## add all new data to rotational line
        self._line['μ']   = np.ravel(μs)[self._iallowed]
        self._line['E_u'] = np.ravel(E_us)[self._iallowed]
        self._line['E_l'] = np.ravel(E_ls)[self._iallowed]
        self._line['Γ_u'] = np.ravel(Γ_us)[self._iallowed]
        self._line['Γ_l'] = np.ravel(Γ_ls)[self._iallowed]
        
    @optimise_method(format_multi_line=99)
    def add_transition_moment(self,name_u,name_l,μv=1,_cache=None):
        """Add constant transition moment. μv can be optimised."""
        """Following Sec. 6.1.2.1 for lefebvre-brion_field2004. Spin-allowed
        transitions. μv should be in atomic units and can be specifed
        as a value (optimisable), a function of R or a suboptimiser
        given ['μ']."""
        if self._clean_construct:
            ## get all quantum numbers
            kwu = self.level_u._manifolds[name_u]
            kwl = self.level_l._manifolds[name_l]
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
                self.μ0[:,iΔJ,i+kwu['ibeg'],j+kwl['ibeg']] += fμ[i,j](self.J_l,ΔJ)*float(μv)

    def plot(
            self,
            Teq=300,            # equilibrium temperature
            match=None,         # plot some lines only
            **kwargs,           # passed to plot_stick_spectrum
    ):
        """Plot line intensities with a stick spectrum."""
        kwargs.setdefault('xkey','ν') # 
        kwargs.setdefault('zkeys',None)
        if Teq is None:
            kwargs.setdefault('ykey','Sij')
        else:
            kwargs.setdefault('ykey','σ')
            self.line['Teq'] = Teq
            self.line['Zsource'] = 'self'
        tline = self.line[self.line['Sij'] > 0]
        if match is not None:
            tline.limit_to_match(match)
        fig = tline.plot_stick_spectrum(**kwargs)
        return fig

@lru_cache
def _get_linear_H(S,Λ,s):
    """Compute symbolic and functional Hamiltonian for the spin manifold
    of a linear molecule."""
    ## symbolic variables, Note that expected value of ef is +1 or -1 for 'e' and 'f'
    p = {key:sympy.Symbol(key) for key in (
        'Tv','Bv','Dv','Hv','Lv','Mv','Nv','Ov','Av','ADv','λv','λDv','λHv','γv','γDv','ov','pv','pDv','qv','qDv')}
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
    H = (p['Tv']*I + p['Bv']*NN - p['Dv']*NN**2 + p['Hv']*NN**3
         + p['Lv']*NN**4 + p['Mv']*NN**5 + p['Nv']*NN**6 + p['Ov']*NN**7)
    if S>0:
        if Λ>0: H += anticommutate(
                p['Av']*I+p['ADv']*NN,
                sympy.diag(*[float(Λ*Σ) for Σ in Σs]))/2 # 1/2[A+AN2,LzSz]+
        H += (p['γv']*I+p['γDv']*NN)*NS # (γ+γD.N**2)N.S
        H += anticommutate(
            p['λv']*I+p['λDv']*NN+p['λHv']*NN**2,
            sympy.diag(*[float(Σ**2) for Σ in Σs])-I/3*S*(S+1)) # [λ+λD.N**2),Sz**2-1/3*S**2]+
    ## add Λ-doubling terms here, element-by-element.
    for i,(Σi,efi) in enumerate(zip(Σs,efs)):
        for j,(Σj,efj) in enumerate(zip(Σs,efs)):
            if efi != efj:
                continue
            ## 1Π state
            if Λ>0 and S==0:
                if efi==-1 and efj==-1:
                    H[i,j] += p['qv']*J*(J+1)   # coefficient
            ## is ef=1 for e, ef=-1 for f 2Π states, amiot1981
            ## table II, there are more distortion constants,
            ## which I have not included here, but could be
            ## done. Perhaps these could be included with N-matrix
            ## multiplication?
            elif Λ==1 and S==0.5:
                for i,(Σi,efi) in enumerate(zip(Σs,efs)):
                    efi = (1 if efi==+1 else -1)
                    for j,(Σj,efj) in enumerate(zip(Σs,efs)):
                        efj = (1 if efj==+1 else -1)
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
                    # *quantum_numbers.wigner3j(Jpp+ΔJ,1,Jpp,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj')  # Wigner 3J line strength factor vectorised over Jpp
                    *np.array([quantum_numbers.wigner3j(Jppi+ΔJ,1,Jppi,-Ωp,Ωp-Ωpp,Ωpp,method='py3nj') for Jppi in Jpp]) # Wigner 3J line strength factor vectorised over Jpp
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
        
def _diabaticise_eigenvalues_by_E0(eigvals,eigvects,E0):
    """Re-order eigvals/eigvects to match the energy ordering of
    deperturbed levels E0."""
    i = np.argsort(eigvals)
    j = np.argsort(np.argsort(E0))
    index = i[j]
    eigvals = eigvals[index]
    eigvects = eigvects[:,index]
    return eigvals,eigvects,index
        
def _diabaticise_eigenvalues_sort_coefficients(eigvals,eigvects):
    """put largest mixing coefficients on the diagonal, beginning with
    the smallest"""
    c = eigvects.real**2
    i = list(range(len(c)))
    j = list(range(len(c)))
    while len(j) > 0:
        ci = c[np.ix_(j,j)]
        imax = np.argmax(np.max(ci,1))
        jmax = np.argmax(ci[imax])
        if imax!=jmax:
            c[:,j[imax]],c[:,j[jmax]] = c[:,j[jmax]].copy(),c[:,j[imax]].copy() 
            i[j[imax]],i[j[jmax]] = i[j[jmax]],i[j[imax]]
        j.pop(imax)
    eigvals = eigvals[i]
    eigvects = eigvects[:,i]
    return eigvals,eigvects

def _diabaticise_eigenvalues(eigvals,eigvects):
    """Re-order eigvals/eigvects to maximise eigvects diagonal."""
    index = arange(len(eigvals))
    ## fractional character array
    c = eigvects.real**2
    ## mask of levels without confirmed assignments
    not_found = np.full(len(c), True) 
    ## find all mixing coefficients greater than 0.5 and fix their
    ## assignments.  If its too many to test permutations of remainder
    ## then accept coefficients min_frac_diff bigger than the next
    ## coefficient
    min_frac_diff = 2
    while sum(not_found) > 7:
        for i in range(len(c)):
            j = np.argsort(-c[i,:])
            ## accept if greater than 0.5 coefficient, or 5% bigger than next coefficient
            if c[i,j[0]] > 0.5 or c[i,j[0]]/c[i,j[1]] > min_frac_diff:
                j = j[0]
                ii = list(range(len(c)))           # index of columns
                ii[i],ii[j] = j,i                  # swap largest c into diagonal position 
                c = c[:,ii]                        # swap for all columns
                eigvals,eigvects = eigvals[ii],eigvects[:,ii] # and eigvalues
                index = index[ii]
                not_found[i] = False
        min_frac_diff -= 0.1
    ## trial all permutations of remaining
    ## unassigned levels, looking for the one
    ## which gives the best assignments
    if np.any(not_found):
        ## limit to mixing coefficient array of unassigned levels
        c_not_found = c[not_found,:][:,not_found]
        number_not_found = len(c_not_found)
        if number_not_found > 8:
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
        index[not_found] = (index[not_found])[best_permutation]
    return eigvals,eigvects,index

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
        name='viblevel',
        species=None,J=None,
        levels=None,         # {name:add_manifold_kwargs,...}
        couplings=None, # None or {name1,name2:add_coupling_kwargs,...}
        spline_widths=None, # None or {name:add_spline_width_kwargs,...}
):
    """Compute a Level model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_manifold, add_spline_width etc."""
    v = Level(name=name,species=species,J=J)
    if levels is not None:
        for name,kwargs in levels.items():
            v.add_manifold(name,**kwargs)
    if couplings is not None:
        for (name1,name2),kwargs in couplings.items():
            v.add_coupling(name1,name2,**kwargs)
    if spline_widths is not None:
        for name,kwargs in spline_widths.items():
            v.add_spline_width(name,**kwargs)
    v.construct()
    return v

def calc_level(*args_viblevel,match=None,**kwargs_viblevel):
    """Compute a Level model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_manifold, add_spline_width etc."""
    v = calc_viblevel(*args_viblevel,**kwargs_viblevel)
    if match is not None:
        retval = dataset.make(v.level.classname)
        retval.copy_from_and_optimise(v.level,match=match)
        return retval
    else:
        return v.level

def calc_line(
        species=None,J_l=None,ΔJ=None,
        upper=None,           # kwargs for calc_level
        lower=None,           # kwargs for calc_level
        transition_moments=None,# None or {name_u,name_l:add_transition_moment,...}
):
    """Compute a Level model and return the generated level
    object. levels and splinewidths etc are lists of kwargss for
    add_manifold, add_spline_width etc."""
    upper['species'] = lower['species'] = species
    upper = calc_viblevel(**upper)
    lower = calc_viblevel(**lower)
    v = Line('vibline',upper,lower,J_l=J_l,ΔJ=ΔJ)
    if transition_moments is not None:
        for (name1,name2),kwargs in transition_moments.items():
            v.add_transition_moment(name1,name2,**kwargs)
    v.construct()
    return v.line


## deprecated
VibLevel = Level
VibLine = Line
