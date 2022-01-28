import re
from copy import copy
import functools
import itertools
from pprint import pprint

import numpy as np
from scipy import linalg
import sympy
# from immutabledict import immutabledict as idict

from .dataset import Dataset
from . import tools
from . import kinetics
from .tools import vectorise
from .exceptions import DatabaseException,InvalidEncodingException

def vectorise_decode(function):
    """Input function takes a single string argument and returns a
    dictionary. If a list is given then convert function to return a dict
    of lists."""
    @functools.wraps(function)
    def vectorised_function(arg):
        if isinstance(arg,str):
            return function(arg)
        else:
            retval = {}
            for i,argi in enumerate(arg):
                for key,val in function(argi).items():
                    if i == 0:
                        retval[key] = [val]
                    else:
                        retval[key].append(val)
        return retval
    return vectorised_function

def separate_upper_lower(qn):
    """Divide quantum numbers into upper and lower level dictionaries and
    remove 'p' and 'pp' suffices."""
    qn = copy(qn)
    qnl,qnu = {},{}
    for key in list(qn):
        if len(key)>2 and key[-2:]=='_l':
            qnl[key[:-2]] = qn.pop(key)
        elif len(key)>2 and key[-2:]=='_u':
            qnu[key[:-2]] = qn.pop(key)
    for key in list(qn):
        if key in ('species',):
            qnl[key] = qnu[key] = qn.pop(key)
        elif r:=re.match(r'Δ(.+)',key):
            tkey = r.group(1)
            if tkey in qnu and tkey not in qnl:
                qnl[tkey] = qnu[tkey] - qn.pop(key)
            elif tkey not in qnu and tkey in qnl:
                qnu[tkey] = qnl[tkey] + qn.pop(key)
    # if len(qn) > 0:
        # raise Exception(f'Did not use qn: {qn}')
    return qnu,qnl

def join_upper_lower(qn_u,qn_l):
    """Upper and lower level quantum numnbers get 'p' or 'pp' suffices
    added and returned as a single dictionary."""
    qn = {}
    qn.update({key+'_u':val for key,val in qn_u.items()})
    qn.update({key+'_l':val for key,val in qn_l.items()})
    return qn

def normalise_term_symbol(term):
    """Convert atomic and molecular term symbols into a standard
    form."""
    ## not implemented
    return term

# @vectorise_decode
# @tools.cache
# def decode_rotational_transition(branch):
    # """Expect e.g., P13ee, P11, P1, P. Return as dict."""
    # branch = branch.strip()
    # if r:=re.match(r'^([OPQRS])([0-9])([0-9])([ef])([ef])$',branch):
        # return {'ΔJ': decode_ΔJ(r.group(1)),
                # 'Fi_u': int(r.group(2)),
                # 'Fi_l': int(r.group(3)),
                # 'ef_u': decode_ef(r.group(4)),
                # 'ef_l': decode_ef(r.group(5))}
    # elif r:=re.match(r'^([OPQRS])([0-9])([0-9])$',branch):
        # return {'ΔJ': decode_ΔJ(r.group(1)),
                # 'Fi_u': int(r.group(2)),
                # 'Fi_l': int(r.group(3)),}
    # elif r:=re.match(r'^([OPQRS])$',branch):
        # return {'ΔJ': decode_ΔJ(r.group(1)),}
    # elif r:=re.match(r'^([OPQRSopqrs])([OPQRS])([0-9])([0-9])([ef])([ef])$',branch):
        # return {
            # 'ΔN': decode_ΔJ(r.group(1)),
            # 'ΔJ': decode_ΔJ(r.group(2)),
            # 'Fi_u': int(r.group(3)),
            # 'Fi_l': int(r.group(4)),
            # 'ef_u': decode_ef(r.group(5)),
            # 'ef_l': decode_ef(r.group(6)),
        # }
    # elif r:=re.match(r'^([OPQRSopqrs])([OPQRS])([0-9])([0-9])$',branch):
        # return {
            # 'ΔN': decode_ΔJ(r.group(1)),
            # 'ΔJ': decode_ΔJ(r.group(2)),
            # 'Fi_u': int(r.group(3)),
            # 'Fi_l': int(r.group(4)),
        # }
    # elif r:=re.match(r'^([OPQRSopqrs])([OPQRS])$',branch):
        # return {
            # 'ΔN': decode_ΔJ(r.group(1)),
            # 'ΔJ': decode_ΔJ(r.group(2)),
        # }
    # else:
        # raise InvalidEncodingException(f"Cannot decode branch: {repr(branch)}")

# decode_branch = decode_rotational_transition # deprecated


def encode_rotational_transition(qn,mathtext=False):
    """Encode rotational transition for ΔJ, J_u, J_l, ef_l, ef_u, Fi_l,
    and Fi_u"""
    if 'ΔJ' not in qn and 'J_l' in qn and 'J_u' in qn:
        qn['ΔJ'] = qn['J_u']-qn['J_l']
    ΔJ = _encode_ΔJ[qn['ΔJ']]
    FFefef = ''
    if 'Fi_u' and 'Fi_l' in qn:
        FFefef += format(qn['Fi_u'],'g')+format(qn['Fi_l'],'g')
    if 'ef_u' and 'ef_l' in qn:
        FFefef += encode_ef(qn['ef_u'])+encode_ef(qn['ef_l'])
    if mathtext:
        retval = f'${ΔJ}_{{{FFefef}}}$'
    else:
        retval = f'{ΔJ}{FFefef}'
    return retval

_decode_ΔJ = {'O':-2,'P':-1,'Q':0,'R':1,'S':2,'o':-2,'p':-1,'q':0,'r':1,'s':2}
@vectorise(dtype=int)
def decode_ΔJ(encoded):
    return _decode_ΔJ[encoded]

_encode_ΔJ = {-2:'O',-1:'P',0:'Q',1:'R',2:'S'}
@vectorise(dtype='<U1')
def encode_ΔJ(decode) :
    return _encode_ΔJ[decoded]

_decode_Λ = {'Σ':0,'Σ⁺':0,'Σ⁻':0,'Π':1,'Φ':2,'Γ':3}
@vectorise(dtype=int)
def decode_Λ(encoded):
    return _decode_Λ[encoded]

_decode_Λs = {'Σ+':(0,0),'Σ-':(0,1), 'Σ⁺':(0,0),'Σ⁻':(0,1), 'Π':(1,0),'Δ':(2,0),'Φ':(3,0),'Γ':(4,0)}
def decode_Λs(encoded):
    return _decode_Λs[encoded]
    
_decode_ef = {'e':+1,'f':-1}
@vectorise(dtype=int)
def decode_ef(encoded):
    return _decode_ef[encoded]
    
_encode_ef = {+1:'e',-1:'f'}
@vectorise(dtype='<U1')
def encode_ef(x):
    return _encode_ef[x]
    
_decode_gu = {'g':+1,'u':-1,+1:+1,-1:-1}
@vectorise(dtype=int)
def decode_gu(encoded):
    return _decode_gu[encoded]

_encode_gu = {'g':'g','u':'u',+1:'g',-1:'u'}
@vectorise(dtype='<U1')
def encode_gu(decoded):
    return _encode_gu[decoded]

_decode_sa = {'s':+1,'a':-1}
@vectorise(dtype=int)
def decode_sa(encoded):
    return _decode_sa[encoded]

def decode_comma_separated_equalities(encoded):
    """E.g,. decode 'a=1,b=2.53' into {'a':1,'b':2.53}. COULD IMPLEMENT
    NON-NUMERIC DATA."""
    retval = {}
    for pair in encoded.split(','):
        key,val = pair.split('=')
        try:
            retval[key.strip()] = int(val)
        except ValueError:
            try:
                retval[key.strip()] = float(val)
            except ValueError:
                raise InvalidEncodingException(f'Could not decode comma separated equalities: {repr(encoded)}')
    return retval

def decode_rotational_transition(encoded):
    """Expect e.g., P13ee25, P, P13ee, P25. Return as dict."""
    ## e.g, P, P5, P13ee, P13ee5, P13ee5.5 
    if r:=re.match(r'^([opqrsOPQRS])?([OPQRS])([0-9][0-9][ef][ef])?([0-9]+(?:\.[05])?)?$',encoded):
        qn = {}
        if r.group(1) is not None:
            qn['ΔN'] = decode_ΔJ(r.group(1))
        qn['ΔJ'] = decode_ΔJ(r.group(2))
        if r.group(3) is not None:
            qn['Fi_u'] = int(r.group(3)[0])
            qn['Fi_l'] = int(r.group(3)[1])
            qn['ef_u'] = decode_ef(r.group(3)[2])
            qn['ef_l'] = decode_ef(r.group(3)[3])
        if r.group(4) is not None:
            try:
                qn['J_l'] = int(r.group(4))
            except ValueError:
                qn['J_l'] = float(r.group(4))
    else:
        raise InvalidEncodingException(f'Could not decode rotational transition: {repr(encoded)}')
    return qn

@vectorise()
def decode_linear_level(encoded):
    """Decode something of the form 32S16O_A.3Π(v=0,Ω=1,J=5) etc."""
    if '–' in encoded:
        raise Exception(f'Encoded linear level contains "–", perhaps it is a line: {repr(encoded)}')
    ## 32S16O_A.3Π or A.3Π or 32S16O_A.3Π(v=0,Ω=1,J=5) or
    ## A.3Π(v=0,Ω=1,J=5) or 32S16O_(v=0,Ω=1,J=5) or (v=0,Ω=1,J=5)
    if r:=re.match(r'^(.+_)?([^(]+)?(\(.*\))?$',encoded):
        qn = {}
        ## get species
        if r.group(1) is not None:
            qn['species'] = decode_species(r.group(1)[:-1])
        ## decode term.
        if r.group(2) is not None:
            term = r.group(2)
            ## e.g., Ap, c3, c₃, c′₄
            if r2:=re.match(r'^([A-Za-z]+[′″]?[0-9₀₁₂₃₄₅₆₇₈₉]*)?$',term):
                qn['label'] = r2.group(1)
            ## e.g., Ap.3Σ+g
            elif r2:=re.match(r'^([A-Za-z]+[′″]?)?\.([0-9])?(Σ\+|Σ-|Π|Δ|Γ)([gu])?$',term):
                qn['label'] = r2.group(1)
                if r2.group(2) is not None:
                    qn['S'] = (float(r2.group(2))-1)/2
                qn['Λ'],qn['s'] = decode_Λs(r2.group(3))
                if r2.group(4) is not None:
                    qn['gu'] = decode_gu(r2.group(4))
            ## e.g., Ap³Σ⁺g
            elif r2:=re.match(r'^([A-Za-z]+[′″]?)?([⁰¹²³⁴⁵⁶⁷⁸⁹])?(Σ⁺|Σ⁻|Π|Δ|Γ)([gu])?$',term):
                qn['label'] = r2.group(1)
                if r2.group(2) is not None:
                    qn['S'] = (float(tools.regularise_unicode(r2.group(2)))-1)/2
                qn['Λ'],qn['s'] = decode_Λs(r2.group(3))
                if r2.group(4) is not None:
                    qn['gu'] = decode_gu(r2.group(4))
        ## decode data in parentheses
        if r.group(3) is not None:
            qn |= decode_comma_separated_equalities(r.group(3)[1:-1])
    else:
        raise Exception(f'Could not decode_linear_level from {encoded!r}')
    return qn

def encode_linear_level(qn=None,**more_qn):
    """Turn quantum numbers etc (as in decode_level) into a string name. """
    ## Test code
    ## t = encode_level(species='N2',label='cp4',Λ=0,s=0,gu='u',S=0,v=5,F=1,ef='e',Ω=0)
    ## print(t)
    ## pprint(decode_level(t))
    if qn is None:
        qn = {}
    qn = qn | more_qn
    retval = ''                 # output string
    ## electronic state label and then get symmetry symbol, first get whether Σ+/Σ-/Π etc, then add 2S+1 then add g/u
    if 'Λ' in qn:
        Λ = qn.pop('Λ')
        if Λ==0:
            retval = 'Σ'
            ## get + or - superscript
            if 's' in qn:
                retval += tools.superscript_numerals('+' if qn.pop('s')==0 else '-')
        elif Λ==1:
            retval = 'Π'
        elif Λ==2:
            retval = 'Δ'
        elif Λ==3:
            retval = 'Φ'
        else:
            raise InvalidEncodingException('Λ>3 not implemented')
        if Λ>0 and 's' in qn:
            qn.pop('s') # not needed
        if 'S' in qn:
            retval = tools.superscript_numerals(str(int(2*float(qn.pop('S'))+1)))+retval
        if 'gu' in qn:
            gu = encode_gu(qn.pop('gu'))
    ## add electronic state label
    if 'label' in qn and retval=='':
        retval =  qn.pop('label')
    elif 'label' in qn:
        retval = qn.pop('label')+retval
    ## prepend species
    if 'species' in qn and retval=='':
        retval =  qn.pop('species')
    elif 'species' in qn:
        retval =  qn.pop('species')+'_'+retval
    ## append all other quantum numbers in parenthese
    if len(qn)>0:
        t = []
        for key,val in qn.items():
            if not isinstance(val,str):
                val = format(val,'g')
            t.append(f'{key}={val}')
        retval = retval + '('+','.join(t)+')'
    return retval
    
def decode_species(species):
    """Try to normalise a species name."""
    return kinetics.get_species(species).name

@vectorise_decode
def decode_linear_line(encoded):
    """Name must be in the form
    species_levelp-levelpp_rotationaltransition. Matched from beginning,
    and returns when matching runs out."""
    ## e.g., species_upper–lower_rot, or u←l or u→l
    if r:=re.match(r'([^_]+_)?([^–_]+)[–←→]([^–_]+)(_.+)?$',encoded):
        qn = {}
        if r.group(1) is not None:
            qn |= {'species':decode_species(r.group(1)[:-1])}
        qn |= {f'{key}_u':val for key,val in decode_linear_level(r.group(2)).items()}
        qn |= {f'{key}_l':val for key,val in decode_linear_level(r.group(3)).items()}
        if r.group(4) is not None:
            qn |= decode_rotational_transition(r.group(4)[1:])
    else:
        if re.match(r'[–←→]',encoded):
            raise InvalidEncodingException(f'Encoded linear line does not contain "–": {repr(encoded)}')
        else:
            raise InvalidEncodingException(f"Could not decode linear line: {repr(encoded)}")
    return qn

def encode_linear_line(qn=None,qnl=None,qnu=None,mathtext=False):
    ## decode branch
    if 'branch' in qn:
        qn |= decode_rotational_transition(qn.pop('branch'))
    ## get all upper and lower qn
    if qnl is None:
        qnl = {}
    else:
        qnl = copy(qnl)
    if qnu is None:
        qnu = {}
    else:
        qnu = copy(qnu)
    if False or 'ΔJ' in qn:
        rotational_transition = encode_rotational_transition(
            {key:qn.pop(key) for key in ('ΔJ','Fi_u','Fi_l','ef_u','ef_l') if key in qn},
        mathtext=mathtext)
    else:
        rotational_transition = None
    if qn is not None:
        for key,val in qn.items():
            if key=='species':
                qnu.setdefault(key,val)
            if len(key)>2:
                if key[-2:]=='_u':
                    qnu.setdefault(key[:-2],val)
                if key[-2:]=='_l':
                    qnl.setdefault(key[:-2],val)
    if 'species' in qnl and 'species' in qnu and qnu['species']==qnl['species']:
        qnl.pop('species')
    upper_level = encode_linear_level(qnu)
    lower_level = encode_linear_level(qnl)
    retval = ''
    if len(upper_level) > 0 or len(lower_level) > 0:
        retval = f'{upper_level}–{lower_level}'
    if rotational_transition is not None:
        if len(retval) > 0: 
            retval += '_'
        retval += rotational_transition
    return retval
    
def encode_latex_term_symbol(**qn):
    """Beginning only."""
    t = {0:r'\Sigma', 1:r'\Pi', 2:r'\Delta', 3:r'\Phi'}
    retval = f'{{}}^{int(2*qn["S"]+1)}{t[qn["Λ"]]}'
    if 's' in qn and qn['Λ']==0: retval += ('^+' if qn['s']==0 else  '^-')
    subscript = ''
    if 'gu' in qn: subscript += qn['gu']
    if 'F'  in qn: subscript += 'F_'+str(int(qn['F']))
    if 'Ω'  in qn:
        if qn['Ω']%1==0: subscript += str(int(qn['Ω']))
        else:            subscript += r'\frac{{{}}}{{2}}'.format(int(qn['Ω']*2))
    if 'ef' in qn: subscript += qn['ef']
    if len(subscript)>0: retval += r'_{'+subscript+r'}'
    return(retval)
    
_encode_Λ_data = {0:'Σ',1:'Π',2:'Δ',3:'Φ',4:'Γ'}
def encode_Λ(Λ):
    return(_encode_Λ_data[Λ])

@vectorise_decode
def decode_transition(transition):
    """Transition must be in the form
    species_levelp-levelpp_rotationaltransition. Matched from beginning,
    and returns when matching runs out."""
    original_transition = transition
    transition = transition.replace(' ','') # all white space removed
    qn = dict()                 # determined quantum numbers
    ## look for rotational transition as e.g., ..._P11fe23, if found
    ## decode and remove from transition
    rot_qn_upper = rot_qn_lower = rot_qn_Δ = None
    r = re.match(r'^(.*)_([^_-]+)$',transition)
    if r:
        try:
            rot_qn_upper,rot_qn_lower,rot_qn_Δ  = decode_rotational_transition(r.group(2),return_separately=True)
            transition = r.group(1)
        except InvalidEncodingException: # not a valid rotational transition
            pass
    ## split upper and lower level
    transition = transition.replace('Σ','Σ').replace('Pi','Π').replace('Delta','Δ').replace('Phi','Φ') # hack to make greek symbols compatible
    transition = transition.replace('Σ-','SigmaMinus') # hack to temporarily protect minus sign
    if transition.count('--')==1:                       # upper-lower
        upper,lower = transition.split('--')            # upper (lower not given)
    elif transition.count('-')==1:                       # upper-lower
        upper,lower = transition.split('-')            # upper (lower not given)
    elif transition.count('-')==0:
        # upper,lower = transition,''
        raise InvalidEncodingException(f'Is this an encoded transition? "-" not found: {repr(original_transition)}')
    else:
        raise InvalidEncodingException(f'Require only one "-" in an encoded transition: {repr(original_transition)}')
    upper,lower = upper.replace('SigmaMinus','Σ-'),lower.replace('SigmaMinus','Σ-') # put this back after split
    upper,lower = decode_level(upper),decode_level(lower)
    ## add rotataional qn if found above
    if rot_qn_lower is not None:
        lower.update(rot_qn_lower)
    if rot_qn_upper is not None:
        upper.update(rot_qn_upper)
    ## assume common species if only given once
    if 'species' in upper and 'species' not in lower:
        lower['species'] = upper['species'] 
    ## return
    retval = join_upper_lower(upper,lower)
    if rot_qn_Δ is not None:
        for key,val in rot_qn_Δ.items():
            retval['Δ'+key] = val
    return retval

def encode_transition(
        qn=None,
        qnl=None,
        qnu=None,
):
    ## get all upper and lower qn
    if qnl is None:
        qnl = {}
    else:
        qnl = copy(qnl)
    if qnu is None:
        qnu = {}
    else:
        qnu = copy(qnu)
    if qn is not None:
        for key,val in qn.items():
            if key=='species':
                qnu.setdefault(key,val)
            if len(key)>2:
                if key[-2:]=='_u':
                    qnu.setdefault(key[:-2],val)
                if key[-2:]=='_l':
                    qnl.setdefault(key[:-2],val)
    if 'species' in qnl and 'species' in qnu and qnu['species']==qnl['species']:
        qnl.pop('species')
    retval = encode_level(qnu)+'-'+encode_level(qnl)
    if 'ΔJ' in qn:
        retval += '_'+_encode_ΔJ[qn['ΔJ']] 
    return retval
    
def encode_level(qn):
    """Turn quantum numbers etc (as in decode_level) into a string name. """
    ## Test code
    ## t = encode_level(species='N2',label='cp4',Λ=0,s=0,gu='u',S=0,v=5,F=1,ef='e',Ω=0)
    ## print(t)
    ## pprint(decode_level(t))
    qn = copy(qn)  # all values get popped below, so make a ocpy
    retval = ''                 # output string
    ## electronic state label and then get symmetry symbol, first get whether Σ+/Σ-/Π etc, then add 2S+1 then add g/u
    if 'Λ' in qn:
        Λ = qn.pop('Λ')
        if Λ==0:
            retval = 'Σ'
            ## get + or - superscript
            if 's' in qn:
                retval += ('+' if qn.pop('s')==0 else '-')
        elif Λ==1:
            retval = 'Π'
        elif Λ==2:
            retval = 'Δ'
        elif Λ==3:
            retval = 'Φ'
        else:
            raise Exception('Λ>3 not implemented')
        if Λ>0 and 's' in qn: qn.pop('s') # not needed
        if 'S' in qn: retval = str(int(2*float(qn.pop('S'))+1))+retval
        if 'gu' in qn: retval += qn.pop('gu')
    ## add electronic state label
    if 'label' in qn and retval=='':
        retval =  qn.pop('label')
    elif 'label' in qn:
        retval = qn.pop('label')+'.'+retval
    ## prepend species
    if 'species' in qn and retval=='':
        retval =  qn.pop('species')
    elif 'species' in qn:
        retval =  qn.pop('species')+'_'+retval
    ## append all other quantum numbers in parenthese
    if len(qn)>0:
        t = []
        for key in qn:
            if key in ('v','F'): # ints
                t.append(key+'='+str(int(qn[key])))
            elif key in ('Ω','Σ','SR'): 
                t.append(key+'='+format(qn[key],'g'))
            else:
                t.append(key+'='+str(qn[key]))
        retval = retval + '('+','.join(t)+')'
    return retval

  
def encode_latex_term_symbol(**qn):
    """Beginning only."""
    t = {0:r'\Sigma', 1:r'\Pi', 2:r'\Delta', 3:r'\Phi'}
    retval = f'{{}}^{int(2*qn["S"]+1)}{t[qn["Λ"]]}'
    if 's' in qn and qn['Λ']==0: retval += ('^+' if qn['s']==0 else  '^-')
    subscript = ''
    if 'gu' in qn: subscript += qn['gu']
    if 'F'  in qn: subscript += 'F_'+str(int(qn['F']))
    if 'Ω'  in qn:
        if qn['Ω']%1==0: subscript += str(int(qn['Ω']))
        else:            subscript += r'\frac{{{}}}{{2}}'.format(int(qn['Ω']*2))
    if 'ef' in qn: subscript += qn['ef']
    if len(subscript)>0: retval += r'_{'+subscript+r'}'
    return(retval)
    
_encode_Λ_data = {0:'Σ',1:'Π',2:'Δ',3:'Φ',4:'Γ'}
def encode_Λ(Λ):
    return(_encode_Λ_data[Λ])

@functools.lru_cache
def get_case_a_basis(S,Λ,s,verbose=False,**kwargs):
    """Determine wavefunctions of a case a state in signed-Ω (pm, +/-)
    and e/f bases. As well as transformation matrices between them."""
    assert Λ>=0 and (s==0 or s==1) and S>=0,'Some quantum number has an invalid value.'
    ## signed-Ω wavefunction
    qnpm = [dict(Λ=Λi,s=s,S=S,Σ=Σ,Ω=Λi+Σ) for (Λi,Σ) in itertools.product(((Λ,-Λ) if Λ>0 else (Λ,)),np.arange(S,-S-0.1,-1))]
    # σvpm = (-1)**(-S+S%1+s)     # phase change when σv operation on converts +Ω to -Ω states
    σvpm = (-1)**(S-S%1+s)     # phase change when σv operation on converts +Ω to -Ω states
    for t in qnpm:
        t['σv'] = (-1)**(S-S%1+s) # symmetry of signed-Ω wavefunctions under inversion
    ## ef symmetrised wavefunctions (maybe not both)
    qnef = []
    for Σ in (np.arange(S,-0.1,-1) if Λ==0 else np.arange(S,-S-0.1,-1)):
        if Λ==0 and Σ==0:
            qnef.append(dict(Λ=Λ,s=s,S=S,Σ=Σ,Ω=np.abs(Λ+Σ),ef=(-1)**(s+S+S%1)))
        else:
            qnef.extend([dict(Λ=Λ,s=s,S=S,Σ=Σ,Ω=np.abs(Λ+Σ),ef=+1), dict(Λ=Λ,s=s,S=S,Σ=Σ,Ω=np.abs(Λ+Σ),ef=-1)])
    ## matrix to convert signed to ef
    Mef = sympy.zeros(len(qnef),len(qnpm))
    for ipm,pm in enumerate(qnpm):
        for ief,ef in enumerate(qnef):
            if np.abs(ef['Ω'])!=np.abs(pm['Ω']) or np.abs(ef['Σ'])!=np.abs(pm['Σ']): continue # no mixing
            if Λ>0:
                if pm['Λ']>0:   Mef[ief,ipm] = +1/sympy.sqrt(2)
                else:           Mef[ief,ipm] = pm['σv']/sympy.sqrt(2)*ef['ef']
            else:
                if ef['Σ']==0:  Mef[ief,ipm] = +1
                else:
                    if pm['Σ']>0:   Mef[ief,ipm] = +1/sympy.sqrt(2)
                    else:           Mef[ief,ipm] = pm['σv']/sympy.sqrt(2)*ef['ef']
    Mpm = Mef.inv() # matrix to convert from ef to pm
    ## get NNpm matrix including spin-rotation interaction
    NNpm = sympy.zeros(Mpm.shape[0])
    NSpm = sympy.zeros(Mpm.shape[0])
    J = sympy.Symbol("J")
    for i,qn1 in enumerate(qnpm): # bra
        for j,qn2 in enumerate(qnpm): # ket
            if i==j:         # diagonal elements
                NNpm[i,i] = J*(J+1)+qn1['S']*(qn1['S']+1)-2*qn1['Ω']*qn1['Σ']
                NSpm[i,i] = qn1['Ω']*qn1['Σ']-qn1['S']*(qn1['S']+1)
                continue
            if j<=i: continue          # symmetric elements taken care of below
            if (qn1['Σ']-qn2['Σ']) != (qn1['Ω']-qn2['Ω']): continue # selection rule ΔΣ=ΔΩ
            ## S-uncoupling,  ⟨iΛS(Σ+1)J(Ω+1)|J-S+|iΛSΣJΩ⟩ = sqrt[(S(S+1)-Σ(Σ+1))*(J(J+1)-Ω(Ω+1))]                if i1==i2 and (Σj-Σi)=+1: 
            if (qn1['Σ']-qn2['Σ'])==+1:   NNpm[i,j] = NNpm[j,i] = -sympy.sqrt(qn2['S']*(qn2['S']+1)-qn2['Σ']*(qn2['Σ']+1))*sympy.sqrt(J*(J+1)-qn2['Ω']*(qn2['Ω']+1)) # J-S+
            if (qn1['Σ']-qn2['Σ'])==-1:   NNpm[i,j] = NNpm[j,i] = -qn1['σv']*qn2['σv']*sympy.sqrt(qn2['S']*(qn2['S']+1)--qn2['Σ']*(-qn2['Σ']+1))*sympy.sqrt(J*(J+1)--qn2['Ω']*(-qn2['Ω']+1)) # J+S- calculated from J-S+ and using σv operator to determine any sign change -- actually always symmetric with respect to inversion since it is the same electronic state
            NSpm[i,j] = NSpm[j,i] = -NNpm[i,j]/2 
    NNef = Mef*NNpm*Mef.T
    NSef = Mef*NSpm*Mef.T
    ## simplify -- slow to execute. Does it run faster when using the
    ## simplified expressions later?
    # NNpm.simplify();NNef.simplify();NSpm.simplify();NSef.simplify()
    ## convert qn to recarrays
    # qnpm = my.dict_to_recarray({key:[t[key] for t in qnpm] for key in qnpm[0]})
    # qnef = my.dict_to_recarray({key:[t[key] for t in qnef] for key in qnef[0]})
    qnpm = Dataset(**{key:[t[key] for t in qnpm] for key in qnpm[0]})
    qnef = Dataset(**{key:[t[key] for t in qnef] for key in qnef[0]})
    if verbose:
        print('\nqnpm:');print(qnpm)
        print('\nqnef:'); print(qnef)
        print('\nMpm:'); pprint(Mpm)
        print('\nMef:'); pprint(Mef)
        print('\nNNpm:'); pprint(NNpm)
        print('\nNNef:'); pprint(NNef)
        print('\nNSpm:'); pprint(NSpm)
        print('\nNSef:'); pprint(NSef)
    # return idict(
    #     n=len(qnpm),
    #     qnpm=qnpm,qnef=qnef,
    #     Mpm=Mpm,Mef=Mef,
    #     NNpm=NNpm,NNef=NNef,
    #     NSpm=NSpm,NSef=NSef,
    #     σvpm=σvpm,
    # )
    return {'n':len(qnpm), 'qnpm':qnpm, 'qnef':qnef, 'Mpm':Mpm, 'Mef':Mef,
            'NNpm':NNpm, 'NNef':NNef, 'NSpm':NSpm, 'NSef':NSef, 'σvpm':σvpm,}

# def get_case_a_to_case_b_transformation(Λ,s,S,J):
    # """Calculate matrices converting case (a) vector to case (b). Matrices
    # included for signed-Ω and e/f symmetry case (a) vectors. COULD ADD
    # LOGIC TO IDENFITY e AND f PARITY CASE B LEVELS AND INCLUDE THIS IN
    # THEIR QUANTUM NUMBERS."""
    # J = float(J)
    # assert S%1==J%1,'Both S and J must be integer or half-integer'
    # ##  get signed-Ω and e/f-parity case (a) bases, with quantum numbers and transformation between them
    # t = get_case_a_basis(Λ,s,S)
    # qnapm,qnaef,Maef = t['qnpm'],t['qnef'],np.matrix(t['Mef']).astype(float)
    # ## get case(b) quantum numbers
    # qnb = deepcopy(qnapm)
    # qnb['SR'] = qnb['Σ']
    # qnb['J'] = J
    # qnb['N'] = qnb['J'] - qnb['SR']
    # qnb.remove_key('Σ','Ω','σv')
    # ## calculate case (a) signed-Ω to case (b) transformation matrix
    # Mbpm = np.matrix([[spectra.clebsch_gordan(J,Λ+Σ,S,-Σ,J-SR,Λ)
                      # for Σ in qnapm['Σ']] for SR in qnb['SR']])
    # ## calculate case (a) e/f to case (b) transformation matrix
    # Mbef = Mbpm*np.transpose(Maef)
    # return(dict(n=len(qnapm), qnapm=qnapm,qnaef=qnaef, qnb=qnb,
                # Maef=Maef, Mbpm=Mbpm, Mbef=Mbef,))

# # @my.lru_cache_copy
# # @my.immutabledict_args
# @functools.lru_cache
# def get_multistate_case_a_basis(*states_qn):
    # """Determine wavefunctions of case a states in signed-Ω and e/f
    # bases. As well as transformation matrices between them. All states
    # are noninteracting and are given as a list of dictionaries with
    # keys label,Λ,s,S. """
    # ## get individual state data and make combined lists and block
    # ## diagonal matrices
    # qnpm,qnef = Dataset(),Dataset()
    # Mpm,Mef,NNpm,NNef = [],[],[],[]
    # for i,state in enumerate(states_qn):
        # d = get_case_a_basis(*state)
        # qnpm.extend(state_index=i,**d['qnpm'])
        # qnef.extend(state_index=i,**d['qnef'])
        # Mpm.append(d['Mpm'])
        # Mef.append(d['Mef'])
        # NNpm.append(d['NNpm'])
        # NNef.append(d['NNef'])
    # ## combine
    # Mef = linalg.block_diag(*Mef)
    # Mpm = linalg.block_diag(*Mpm)
    # NNpm = linalg.block_diag(*NNpm)
    # NNef = linalg.block_diag(*NNef)
    # return immutabledict(n=len(qnpm),qnpm=qnpm,qnef=qnef,Mpm=Mpm,Mef=Mef,NNpm=NNpm,NNef=NNef)

# def get_rotational_coupling_matrix(name1='',name2='', qn1=None,qn2=None,verbose=False):
    # """Compute a symbolic matrix rotationally coupling two spin manifolds."""
    # for key,val in decode_level(name1).items(): qn1.setdefault(key,val)
    # for key,val in decode_level(name2).items(): qn2.setdefault(key,val)
    # casea = get_multistate_case_a_basis(qn1,qn2,verbose=False) # get case (a) wavefunctions and transformation
    # JL = sympy.zeros(casea['n'])                               # L-uncoupling matrix
    # JS = sympy.zeros(casea['n'])                               # S-uncoupling matrix
    # J = sympy.Symbol("J")
    # from sympy.physics.wigner import wigner_3j
    # for i,qn1 in enumerate(casea['qnpm']): # bra
        # for j,qn2 in enumerate(casea['qnpm']): # ket
            # if j<=i: continue          # symmetric elements taken care of below
            # ## add term in L-uncoupling matrix, JL[i,j] =  ⟨i(Λ+1)SΣJ(Ω+1)|J-L+|jΛSΣJΩ⟩
            # if (qn2['label']!=qn1['label']
                # and qn2['S']==qn1['S']
                # and np.abs(qn2['Λ']-qn1['Λ'])==1
                # and (qn2['Ω']-qn1['Ω'])==(qn2['Λ']-qn1['Λ'])): # test for satisfying selection rulese i=i, S=S, ΔΛ=1, ΔΛ=ΔΩ
                # if qn1['Λ']>qn2['Λ']:
                    # JL[i,j] = JL[j,i] = sympy.sqrt(J*(J+1)-qn2['Ω']*(qn2['Ω']+1)) # J-L+
                # else:
                    # JL[i,j] = JL[j,i] = qn1['σv']*qn2['σv']*sympy.sqrt(J*(J+1)--qn2['Ω']*(-qn2['Ω']+1)) # J+L- by symmetry from J-L+
            # ## add term in S-uncoupling matrix, JS[i,j] = ⟨iΛS(Σ+1)J(Ω+1)|J-S+|jΛSΣJΩ⟩
            # if (qn2['label']!=qn1['label']
                # and qn2['S']==qn1['S']
                # and np.abs(qn2['Σ']-qn1['Σ'])==1
                # and (qn2['Ω']-qn1['Ω'])==(qn2['Σ']-qn1['Σ'])): # test for satisfying selection rulese i=i, S=S, ΔΣ=1, ΔΣ=ΔΩ
                # ## correct phases here?
                # if qn1['Σ']>qn2['Σ']:
                    # JS[i,j] = JS[j,i] = sympy.sqrt(qn2['S']*(qn2['S']+1)-qn2['Σ']*(qn2['Σ']+1))*sympy.sqrt(J*(J+1)-qn2['Ω']*(qn2['Ω']+1)) # J-S+
                # else:
                    # JS[i,j] = JS[j,i] = qn1['σv']*qn2['σv']*sympy.sqrt(qn2['S']*(qn2['S']+1)--qn2['Σ']*(-qn2['Σ']+1))*sympy.sqrt(J*(J+1)--qn2['Ω']*(-qn2['Ω']+1))  # J+S- by symmetry from J-S+ 
    # ## transform to e/f basis
    # JLef = casea['Mef']*JL*casea['Mef'].T 
    # JSef = casea['Mef']*JS*casea['Mef'].T
    # if verbose:
        # print('get_rotational_coupling_matrix')
        # print('Signed-Ω basis:')
        # print_matrix_elements(casea['qnpm'],JS,'JS')
        # print('Definite e/f-parity basis:')
        # print_matrix_elements(casea['qnef'],JSef,'JS')
    # return(casea,JLef,JSef)

# def get_spin_orbit_coupling_matrix(name1='',name2='', qn1=None,qn2=None,verbose=False):
    # """Calculate algebraic part of Σli.si operator and optimise the
    # unknown part.  Not very accurate? ±0.01cm-1?"""
    # for key,val in decode_level(name1).items():
        # qn1.setdefault(key,val)
    # for key,val in decode_level(name2).items():
        # qn2.setdefault(key,val)
    # casea = get_multistate_case_a_basis(qn1,qn2,verbose=False) # get case (a) wavefunctions and transformation
    # LS = sympy.zeros(casea['n']) # in signed-Ω basis
    # from sympy.physics.wigner import wigner_3j
    # for i,qn1 in enumerate(casea['qnpm']): # bra
        # for j,qn2 in enumerate(casea['qnpm']): # ket
            # if j<=i: continue          # symmetric elements taken care of below
            # ## build spin-orbit interaction matrix
            # if qn1['label']!=qn2['label'] and (qn1['Λ']-qn2['Λ'])==(qn2['Σ']-qn1['Σ']) and np.abs(qn1['Λ']-qn2['Λ'])==1: # selection rules ΔΛ=-ΔΣ=±1, ΔΩ=0
                # if (qn1['Λ']-qn2['Λ'])==+1:
                    # LS[i,j] = LS[j,i] = np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L+S-
                # if (qn1['Λ']-qn2['Λ'])==-1:
                    # LS[i,j] = LS[j,i] = qn1['σv']*qn2['σv']*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # L-S+ calculated from L+S- using σv equivalence
            # elif qn1['label']!=qn2['label'] and (qn1['Λ']==qn2['Λ']) and (qn2['Σ']==qn1['Σ']) : # selection rules, ΔΛ=ΔΣ=ΔΩ=0
                # phase = 1
                # if qn1['Λ']<0 or (qn1['Λ']==0 and qn1['Σ']<0):
                    # phase *= qn1['σv']
                # if qn2['Λ']<0 or (qn2['Λ']==0 and qn2['Σ']<0):
                    # phase *= qn2['σv']
                # LS[i,j] = LS[j,i] = phase*np.abs(wigner_3j(qn1['S'],1,qn2['S'],-qn1['Σ'],qn1['Σ']-qn2['Σ'],+qn2['Σ'])) # LzSz
            # else:
                # pass
    # ## transform to e/f basis
    # LSef = casea['Mef']*LS*casea['Mef'].T # transform to e/f basis
    # ## describe verbosely
    # if verbose:
        # print('get_spin_orbit_coupling_matrix')
        # print('Signed-Ω basis:')
        # print_matrix_elements(casea['qnpm'],LS,'LS')
        # print('Definite e/f-parity basis:')
        # print_matrix_elements(casea['qnef'],LSef,'LS')
    # return(casea,LSef)

# def pm_to_ef_coefficients(Λ,s,S,Σ,symbolic=False):
    # """Return a dictionary whose elements e.g., ('e','+1') are the
    # coefficients needed to convert a bases in terms of +Λ and -Λ
    # into a e/f parity symmetrised states. Return as numerical
    # values or unevaluated irrational numbers (symbolic)."""
    # if symbolic:   invsqrt2 = 1/sympy.sqrt(2)
    # else:          invsqrt2 = 1/np.sqrt(2)
    # if Λ==0:     
        # if Σ==0: # Ω=0 Σ± states, only one parity, the other has coefficients of zero
            # if (s==0 and S%2==0) or (s==1 and S%2==1):   return({'e':{1:1}})
            # else:                                        return({'f':{1:1}})
        # else:               # Ω>0 Σ± states, reverse signed if Σ-
            # if s==0:  return(dict(e={+1:+invsqrt2,-1:+invsqrt2,},f={+1:+invsqrt2,-1:-invsqrt2,}))
            # else:     return(dict(e={+1:+invsqrt2,-1:-invsqrt2,},f={+1:+invsqrt2,-1:+invsqrt2,}))
    # else:                   # Λ>0 states
        # return(dict(e={+1:+invsqrt2,-1:+invsqrt2,},f={+1:+invsqrt2,-1:-invsqrt2,}))

@tools.vectorise(dtype=float,cache=True)
def honl_london_factor(Ωp,Ωpp,Jp,Jpp,return_zero_on_fail=False):
    """Calculate Honl London factor for an arbitrary one-photon
    transition. Returns an float, or a array of floats as
    required. Selection rule violating transitions return nan. """
    return 3*M_indep_direction_cosine_matrix_elements(Ωp,Ωpp,Jp,Jpp,return_zero_on_fail)**2

def M_indep_direction_cosine_matrix_elements(Ωp,Ωpp,Jp,Jpp,return_zero_on_fail=False):
    """Calculate using formulae of lefebvre-brion_field2004 Tab 6.1"""
    ## handle vector data case -- requires all quantum numbers be arrays of the same length
    if not np.isscalar(Ωp):
        return(np.array([M_indep_direction_cosine_matrix_elements(Ωpi,Ωppi,Jpi,Jppi,return_zero_on_fail) for (Ωpi,Ωppi,Jpi,Jppi) in zip(Ωp,Ωpp,Jp,Jpp)]))
    ## if original not iterable, return as not iterable, but must be
    ## arrays for convenience during calculation
    J,Ω = Jpp,Ωpp
    if   (Jp==Jpp+1) and (Ωp==Ωpp+1):
        retval = -np.sqrt( ((J+Ω+1)*(J+Ω+2)) / (3*(J+1)) )
    elif (Jp==Jpp+1) and (Ωp==Ωpp):
        retval = np.sqrt( ((J+Ω+1)*(J-Ω+1)) / (3*(J+1)) )
    elif (Jp==Jpp+1) and (Ωp==Ωpp-1):
        retval = np.sqrt( ((J-Ω+1)*(J-Ω+2)) / (3*(J+1)) )
    elif (Jp==Jpp)   and (Ωp==Ωpp+1):
        retval = np.sqrt( ((2*J+1)*(J-Ω)*(J+Ω+1)) / (3*J*(J+1)) )
    elif (Jp==Jpp)   and (Ωp==Ωpp):
        retval = np.sqrt( ((2*J+1)) / (3*J*(J+1)) ) 
    elif (Jp==Jpp)   and (Ωp==Ωpp-1):
        retval = np.sqrt( ((2*J+1)*(J+Ω)*(J-Ω+1)) / (3*J*(J+1)) )
    elif (Jp==Jpp-1) and (Ωp==Ωpp+1):
        retval = +np.sqrt( ((J-Ω)*(J-Ω-1)) / (3*J) )
    elif (Jp==Jpp-1) and (Ωp==Ωpp):
        retval = np.sqrt( ((J+Ω)*(J-Ω)) / (3*J) )
    elif (Jp==Jpp-1) and (Ωp==Ωpp-1):
        retval = -np.sqrt( ((J+Ω)*(J+Ω-1)) / (3*J) )
    else:
        ## forbidden transitions get a strength of zero, or raise an
        ## error
        if return_zero_on_fail:
            return 0.
        else:
            raise ValueError('Could not find correct Honl-London case.') 
    ## convert bad values (nans) to zero and make integer
    if np.isnan(retval):
        retval=0.
    return retval

@tools.cache
def wigner3j(j1,j2,j3,m1,m2,m3,
             # method='sympy',  # symbolic calc with sympy
             # method='sympy_numeric',  # numericisation of symbolic calc from sympy
             method='py3nj',  # numeric calc in fortran (fast) or symbolic calc with sympy, if j1 is vector the solution is vector
):
    """Calculate wigner 3j symbol."""
    if method=='sympy':
        from sympy.physics.wigner import wigner_3j
        return(wigner_3j(j1,j2,j3,m1,m2,m3))
    elif method=='sympy_numeric':
        from sympy.physics.wigner import wigner_3j
        return(float(wigner_3j(j1,j2,j3,m1,m2,m3)))
    elif method=='py3nj':       # vectorised
        import py3nj
        if j1 < 0 or j2 < 0 or j3 < 0:
            return 0. 
        if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
            return 0.
        return py3nj.wigner3j(int(j1*2),int(j2*2),int(j3*2),int(m1*2),int(m2*2),int(m3*2))
    else:
        raise Exception(f"Unknown method: {repr(method)}")

def wigner3j_vector(j1,j2,j3,m1,m2,m3):
    """Calculate wigner 3j symbol."""
    j1 = np.asarray(j1*2,dtype=int)
    length = len(j1)
    j2 = (np.full(length,j2*2,dtype=int) if np.isscalar(j2) else np.asarray(j2*2,dtype=int))
    j3 = (np.full(length,j3*2,dtype=int) if np.isscalar(j3) else np.asarray(j3*2,dtype=int))
    m1 = (np.full(length,m1*2,dtype=int) if np.isscalar(m1) else np.asarray(m1*2,dtype=int))
    m2 = (np.full(length,m2*2,dtype=int) if np.isscalar(m2) else np.asarray(m2*2,dtype=int))
    m3 = (np.full(length,m3*2,dtype=int) if np.isscalar(m3) else np.asarray(m3*2,dtype=int))
    retval = np.zeros(length,dtype=float)
    i = (j1>=0)&(j2>=0)&(j3>=0)&(j1>=np.abs(m1))&(j2>=np.abs(m2))&(j3>=np.abs(m3))
    if np.any(i):
        retval[i] = py3nj.wigner3j(j1[i],j2[i],j3[i],m1[i],m2[i],m3[i])
    return retval 

# def clebsch_gordan(j1,m1,j2,m2,J,M):
    # """Clebsch-Gordan coefficient. Gives coefficent allowing for the
    # summation of |j1,m1,j2,m2⟩ states in to a pure |JM⟩ state."""
    # return((-1)**(-j1+j2-M)*np.sqrt(2*J+1)*spectra.wigner3j(j1,j2,J,m1,m2,-M))
          #  
