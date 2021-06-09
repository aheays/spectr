import functools
from functools import lru_cache,wraps
import re
import os
import warnings
from copy import copy
from pprint import pprint
import itertools
import io

from scipy import interpolate,constants,integrate,linalg,stats
import csv
import glob as glob_module
import numpy as np
from numpy import array
import h5py
from sympy.printing.pycode import pycode

from . import plotting
from .plotting import *


class AutoDict:

    """Kind of like a dictionary of lists. If an unknown key is given then
    creates an empty list."""

    def __init__(self,default_value):
        self._dict = dict()
        self.default_value = default_value

    def __getitem__(self,key):
        if key not in self._dict:
            self._dict[key] = copy(self.default_value)
        return(self._dict[key])

    def __setitem__(self,key,val):
        self._dict[key] = val
    
    def __str__(self):
        return(str(self._dict))
    
    def __repr__(self):
        return(repr(self._dict))

    def __iter__(self):
        for key in self._dict.keys():
            yield key

    def keys(self):
        for key in self._dict.keys():
            yield key

    def values(self):
        for val in self._dict.values():
            yield val

    def items(self):
        for key,val in self._dict.items():
            yield key,val


#######################################################
## decorators / decorator factories / function tools ##
#######################################################

cache = lru_cache

def vectorise(vargs=None,dtype=None,cache=False):
    """Vectorise a scalar-argument scalar-return value function.  If all
    arguments are scalar return a scalar result. If args is None
    vectorise all arguments. If a list of indices vectorise only those
    arguments. If dtype is given return value is an array of this
    type, else a list is returned. If cache is True then cache
    indivdual scalar function calls."""
    def actual_decorator(function):

        ## get a cached version fo the function if requested
        if cache:

            # ## works with multiprocessing -- can be dill pickled
            # _cache_max_len = 10000
            # _cache = {}
            # def function_maybe_cached(*args):
                # hashed_args = hash(tuple(args))
                # if hashed_args not in _cache:
                    # retval = function(*args)
                    # if len(_cache) < _cache_max_len:
                        # _cache[hashed_args] = retval
                    # else:
                        # warnings.warn(f'Need to implement a limited cache: {function.__name__}')
                # return _cache[hashed_args]

            ## will not work with dill for some reason
            function_maybe_cached = lru_cache(function)

        else:
            function_maybe_cached = function

        @wraps(function)
        def vectorised_function(*args):
            args = list(args)
            ## get list of arg indices that should be vectorised
            if vargs is None:
                vector_arg_indices = list(range(len(args)))
            else:
                vector_arg_indices = list(vargs)
            ## check for scalar args and consistent length for vector
            ## args
            length = None
            vector_args = []
            for i in copy(vector_arg_indices):
                if isiterable(args[i]):
                    vector_args.append(args[i])
                    if length is None:
                        length = len(args[i])
                    else:
                        assert len(args[i])==length,'Nonconstant length of vector arguments.'
                else:
                    vector_arg_indices.remove(i)
            if length is None:
                ## all scalar, do scalar calc
                return function_maybe_cached(*args)
            else:
                ## compute for each vectorised arg combination
                if dtype is None:
                    retval = []
                else:
                    retval = np.empty(length,dtype=dtype)
                for i in range(length):
                    for j,k in enumerate(vector_arg_indices):
                        args[k] = vector_args[j][i]
                    iretval = function_maybe_cached(*args)
                    if dtype is None:
                        retval.append(iretval)
                    else:
                        retval[i] = iretval
            return retval
        return vectorised_function
    return actual_decorator

def vectorise_arguments(function):
    """Convert all arguments to arrays of the same length.  If all
    original input arguments are scalar then convert the result back
    to scalar."""
    @functools.wraps(function)
    def function_with_vectorised_arguments(*args):
        arglen = 0
        for arg in args:
            if np.isscalar(arg):
                continue
            if arglen==0:
                arglen = len(arg)
            else:
                assert arglen == len(arg),'Mismatching lengths of vector arguments.'
        if arglen==0:
            ## all scalar -- make length 1 and compute, returning as
            ## scalar
            return function(*[np.array([arg]) for arg in args])[0]
        else:
            ## ensure all arguments vector and compute a vector
            ## result
            return function(*[np.full(arglen,arg) if np.isscalar(arg) else np.asarray(arg) for arg in args])
    return function_with_vectorised_arguments

# def frozendict_args(f):
    # """A decorator that aims to be like functools.lru_cache except it
    # deepcopies the retrun value, so that mutable object can safely
    # cached. """
    # from frozendict import frozendict
    # def fnew(*args,**kwargs):
        # return(f(*[frozendict(arg) if isinstance(arg,dict) else arg for arg in args],
                 # **{key:(frozendict(val) if isinstance(val,dict) else val) for key,val in kwargs.items()}))
    # return(fnew)

# def lru_cache_copy(f,*lru_cache_args,**lru_cache_kwargs):
    # """A decorator that aims to be like functools.lru_cache except it
    # deepcopies the retrun value, so that mutable object can safely
    # cached. """
    # import functools
    # import copy
    # @functools.lru_cache(*lru_cache_args,**lru_cache_kwargs)    
    # def fcached(*args,**kwargs):
        # return(f(*args,**kwargs))
    # def fcached_copied(*args,**kwargs):
        # return(copy.deepcopy(fcached(*args,**kwargs)))
    # return(fcached_copied)

# @functools.lru_cache
# def cached_pycode(*args,**kwargs):
    # return(pycode(*args,**kwargs))

# def pick_function_kwargs(kwargs,function):
    # """Find which kwargs belong to this function, return as a dict."""
    # import inspect
    # ## get a list of acceptable args, special case for plot (and others?)
    # if function.__name__ == 'plot':    # special case for plot since args not in function definition line
        # args = ('agg_filter','alpha','animated','antialiased','aa','clip_box','clip_on','clip_path','color','c','contains','dash_capstyle','dash_joinstyle','dashes','drawstyle','figure','fillstyle','gid','label','linestyle','ls','linewidth','lw','marker','markeredgecolor','mec','markeredgewidth','mew','markerfacecolor','mfc','markerfacecoloralt','mfcalt','markersize','ms','markevery','path_effects','picker','pickradius','rasterized','sketch_params','snap','solid_capstyle','solid_joinstyle','transform','url','visible','xdata','ydata','zorder')
    # else:
        # # args,varargs,keywords,defaults = inspect.getargspec(function)
        # t = inspect.signature(function)
        # print( t)
        # print( dir(t))
        # t = inspect.getfullargspec(function)
        # print( t)
        # args,varargs,keywords,defaults = inspect.signature(function)
    # ## pick these into a new dict
    # picked_kwargs = dict()
    # for key,val in kwargs.items():
        # if key in args: picked_kwargs[key] = val
    # other_kwargs = dict()
    # for key,val in kwargs.items():
        # if key not in picked_kwargs:
            # other_kwargs[key] = val
    # return(picked_kwargs,other_kwargs)

# def repr_args_kwargs(*args,**kwargs):
    # """Format args and kwargs into evaluable args and kwargs."""
    # retval = ''
    # if len(args)>0:
        # retval += ','.join([repr(arg) for arg in args])+','
    # if len(kwargs)>0:
        # retval += ','.join([str(key)+'='+repr(val) for key,val in kwargs.items()])+','
    # return(retval)

def dict_to_kwargs(d,keys=None):
    """Expand a dict into evaluable kwargs. Default to all keys."""
    if keys is None:
        keys = d.keys() # default to all keys
    return(','.join([key+'='+repr(d[key]) for key in keys]))

def dict_expanded_repr(d,indent='',maxdepth=1,separate_with_blanks_depth=-1,_depth=0):
    """pprint dict recursively but repr non-dict elements."""
    indent = '    '
    lines = ['{']
    for i,(key,val) in enumerate(d.items()):
        if separate_with_blanks_depth >= _depth:
            prefix = '\n'+indent
        else:
            prefix = indent
        if (
                not isinstance(val,dict) # not a dict
                or _depth >= maxdepth    # already too deep
                or len(val) == 0         # empty dict
                or (len(val) == 1 and not any([isinstance(t,dict) for t in val.values()])) # dict contains no other dicts
            ):
            ## put on one line
            lines.append(f'{prefix}{repr(key):20}: {repr(val)},')
        else:
            ## expand as subdict
            subdict = dict_expanded_repr(val,indent+"    ",_depth=_depth+1,maxdepth=maxdepth,separate_with_blanks_depth=separate_with_blanks_depth)
            lines.append(f'{prefix}{repr(key):10}: {subdict},')
    lines.append('}')
    lines = [indent*_depth+t for t in lines]
    retval = '\n'.join(lines)
    return retval

def compute_matrix_of_function(A,*args,**kwargs):
    """2D only"""
    retval = np.matrix([[Aij(*args,**kwargs) for Aij in Ai] for Ai in A])
    return retval


############################
## mathematical functions ##
############################

# def kronecker_delta(x,y):
    # """1 if x==y else 0."""
    # if np.isscalar(x) and np.isscalar(y): return(1 if x==y else 0) # scalar case
    # if np.isscalar(x) and not np.isscalar(y): x,y = y,x            # one vector, get in right order
    # retval = np.zeros(x.shape)
    # retval[x==y] = 1
    # return(retval)              # vector case


def tanh_transition(x,xa,xb,center,width):
    """Creates a smooth match between extreme values xa and xb on grid x.
    Uses a hyperbolic tangent centred at center with the given transition
    width."""
    return (np.tanh((x-center)/width)+1)/2*(xb-xa)+xa

def tanh_hat(x,xa,xb,center,ramp_width,top_width):
    """Creates a smooth match between extreme values xa and xb on grid x.
    Uses a hyperbolic tangent centred at center with the given transition
    width."""
    return (
        tanh_transition(x,xa,xb,center-top_width/2,ramp_width)
        -tanh_transition(x,xa,xb,center+top_width/2,ramp_width)
    )
    # if np.isscalar(x):
        # if x<=center:
            # return (np.tanh((x-center+top_width)/ramp_width)+1)/2*(xb-xa)+xa 
        # else:
            # return (np.tanh((center+top_width-x)/ramp_width)+1)/2*(xb-xa)+xa
    # else:
        # i = x<center
        # retval = np.empty(x.shape,dtype=float)
        # retval[i] = (np.tanh((x[i]-center-top_width)/ramp_width)+1)/2*(xb-xa)+xa
        # retval[~i] = (np.tanh((center+top_width-x[~i])/ramp_width)+1)/2*(xb-xa)+xa
        # return retval
    

def leastsq(func,
            x0,
            dx,
            R=100.,
            print_error_mesg=True,
            error_only=False,
            xtol=1.49012e-8,
            rms_noise=None,     # for calculation of uncertaintes use this noise level rather than calculate from fit residual. This is useful in the case of an imperfect fit.
):
    """
    Rejig the inputs of scipy.optimize.leastsq so that they do what I
    want them to.
    \nInputs:\n
      func -- The same as for leastsq.
      x0 -- The same as for leastsq.
      dx -- A sequence of the same length as x0 containing the desired
      absolute stepsize to use when calculating the finite difference
      Jacobean.
      R -- The ratio of two step sizes: Dx/dx. Where Dx is the maximum
      stepsize taken at any time. Note that this is only valid for the
      first iteration, after which leastsq appears to approximately
      double the 'factor' parameter.
      print_error_mesg -- if True output error code and message if failure
    \nOutputs: (x,sigma)\n
    x -- array of fitted parameters
    sigma -- error of these
    The reason for doing this is that I found it difficult to tweak
    the epsfcn, diag and factor parametres of leastsq to do what I
    wanted, as far as I can determine these behave in the following
    way:
    dx = x*sqrt(epsfcn) ; x!=0,
    dx = 1*sqrt(epsfcn) ; x==0.
    Default epsfcn=2.2e-16 on scucomp2.
    Dx = abs(x*100)      ; x!=0, factor is not set,
    Dx = abs(x*factor)   ; x!=0, factor is set,
    Dx = abs(factor)     ; x==0, factor is set,
    Dx = 100             ; x==0, factor is not set, diag is not set,
    Dx = abs(100/diag)   ; x==0, factor is not set, diag is set,
    Dx = abs(factor/diag); x==0, factor is set, diag is set.
    Many confusing cases, particularly annoying when initial x==0 and
    it is not possible to control dx or Dx individually for each
    parameter.
    My solution was to add a large value to each parameter so that
    there is little or no chance it will change magnitude during the
    course of the optimisation. This value was calculated separately
    for each parameter giving individual control over dx. I did not
    think of a way to also control Dx individually, instead the ratio
    R=Dx/dx may be globally set.
    """
    from scipy import optimize
    ## limit the number of evaluation to a minimum number to compute
    ## the uncertainty from the second derivative - make R small to
    ## improve performance? - Doesn't work for very large number of
    ## parameters - errors are all nan, probably because of a bug in
    ## leastsq?
    if error_only:
        maxfev = len(x0)+1
        R = 1.
    else:
        maxfev = 0
    ## try and wangle actual inputs of numpy.leastsq to get the right
    ## step sizes
    x0=np.array(x0)
    if np.isscalar(dx): dx = np.full(x0.shape,dx)
    dx=np.array(dx)
    epsfcn = 1e-15              # required that sqrt(epsfcn)<<dp/p
    xshift = x0+dx/np.sqrt(epsfcn)    # required that xshift>>p
    factor = R*np.sqrt(epsfcn)
    x = x0-xshift
    ## perform optimisation. try block is for the case where failure
    ## to calculte error
    try:
        (x,cov_x,info,mesg,success)=optimize.leastsq(
            lambda x:func(x+xshift),
            x,
            epsfcn=epsfcn,
            factor=factor,
            full_output=True,
            maxfev=maxfev,
            xtol = xtol,
            )
    except ValueError as err:
        if str(err)=='array must not contain infs or NaNs':
            raise Exception('Bad covariance matrix in error calculation, residual independent of some variable?')
        else:
            raise
    ## check if any parameters have zero effect on the fit, and raise
    ## a warning if so. This will prevent the calculation of the
    ## uncertainty.
    if np.min(np.sum(np.abs(info['fjac']),1))==0: # a test for no effect
        ## calculate finite difference derivative
        reference_residual = np.array(func(x+xshift))
        diff = []
        for i,dxi in enumerate(dx):
            x1 = copy(x)
            x1[i] += dxi
            diff.append(np.max(np.abs(reference_residual-np.array(func(x1+xshift)))))
        diff = np.array(diff)
        ## warn about those that have no difference
        for j in find(diff==0):
            print(f'warning: Parameter has no effect: index={j}, value={float(x[j]+xshift[j])}')
    ## warn on error if requested
    if (not success) & print_error_mesg:
        warnings.warn("leastsq exit code: "+str(success)+mesg)
    ## sometimes this is not an array
    if not np.iterable(x): x=[x]
    ## attempt to calculate covariance of parameters
    if cov_x is None: 
        sigma_x = np.nan*np.ones(len(x))
    else:
        ## calculate noise rms from the resiudal if not explicitly provided
        if rms_noise is None:
            chisq=sum(info["fvec"]*info["fvec"])
            dof=len(info["fvec"])-len(x)+1        # degrees of freedom
            ## assumes unweighted data with experimental uncertainty
            ## deduced from fitted residual. ref gavin2011.
            std_y = np.sqrt(chisq/dof)
        else:
            std_y = rms_noise   # note that the degrees of freedom is not considered here
        sigma_x = np.sqrt(cov_x.diagonal())*std_y
    return(x+xshift,sigma_x)

def rms(x):
    """Calculate rms."""
    return np.sqrt(np.mean(np.array(x)**2))

def nanrms(x):
    """Calculate rms, ignoring NaN data."""
    return np.sqrt(np.nanmean(np.array(x)**2))

def randn(shape=None):
    """Return a unit standard deviation normally distributed random
    float, or array of given shape if provided."""
    if shape == None:
        return float(np.random.standard_normal((1)))
    else:
        return np.random.standard_normal(shape)


###########################
## convenience functions ##
###########################

def uniquify_strings(strings):
    repeats = {}
    for s in strings:
        if s in repeats:
            repeats[s] +=1
        else:
            repeats[s] = 1
    retval = []
    counts = {}
    for s in strings:
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
        if repeats[s] == 1:
            retval.append(s)
        else:
            retval.append(s+'_'+str(counts[s]))
    return retval

def convert_to_bool_vector_array(x):
    retval = array(x,ndmin=1)
    if retval.dtype.kind == 'b':
        return retval
    elif retval.dtype.kind == 'U':
        t = []
        for xi in retval:
            if xi=='True':
                t.append(True)
            elif xi=='False':
                t.append(False)
            else:
                raise Exception("Valid boolean string values are 'True' or 'False'")
        retval = array(t,ndmin=1,dtype=bool)
        return retval
    else:
        ## use numpy directly
        try:
            return np.asarray(x,dtype=bool,ndmin=1)
        except:
            return array([bool(t) for t in tools.ensure_iterable(x)],dtype=bool)

def warnings_off():
    warnings.simplefilter("ignore")

def date_string():
    """Get string representing date in ISO format."""
    import datetime
    t = datetime.datetime.now()
    return('-'.join([str(t.year),format(t.month,'02d'),format(t.day,'02d')]))

# def dump(o,f):
    # """Like pickle.dump except that f can string filename."""
    # if type(f)==str:
        # f=file(f,'w')
        # pickle.dump(o,f)
        # f.close()
    # else: pickle.dump(o,f)

# def load(f):
    # """Like pickle.load except that f can string filename."""
    # if type(f)==str:
        # f=file(f,'r')
        # o = pickle.load(f)
        # f.close()
    # else: o = pickle.load(f)
    # return o

# def take_from_list(l,b):
    # """Return part of a list.\n\nReturn elements of l for which b is
    # true. l and b must be the same length."""
    # return [ll for (ll,bb) in zip(l,b) if bb]

def isiterable(x):
    """Test if x is iterable, False for strings."""
    if isinstance(x,str): 
        return False
    try:
        iter(x)
    except TypeError:
        return False
    return True

# def isiterable_not_string(x):
    # """Test if x is iterable, False for strings."""
    # if isinstance(x,str): 
        # return False
    # try:
        # iter(x)
    # except TypeError:
        # return False
    # return True

def indices(arr):
    """Generator return all combinations of indices for an array."""
    for i in itertools.product(
            *[range(n) for n in
              np.asarray(arr).shape]):
        yield i

########################
## file manipulations ##
########################

def expand_path(path):
    """Shortcut to os.path.expanduser(path). Returns a
    single file only, first in list of matching."""
    import os
    return os.path.expanduser(path)

def tmpfile():
    """Create a secure named temporary file which will be
    automatically deleted. Object is returned."""
    return tempfile.NamedTemporaryFile()

def tmpdir():
    """Create a temporary directory which will not be
    automatically deleted. Pathname is returned."""
    return tempfile.mkdtemp()

# def cp(src,dest):
    # return(shutil.copy(expand_path(src),expand_path(dest)))

# def cptree(src,dest):
    # """"""
    # return(shutil.copytree(expand_path(src),expand_path(dest)))

def trash(filename):
    """Put file in the trash can. Silence on error. No filename expansion."""
    import shlex
    os.system('trash-put '+shlex.quote(filename)+' > /dev/null 2>&1')

def mkdir(*directories,trash_existing=False):
    """Create directory tree (or multiple) if it doesn't exist."""
    ## if multiple loop through them
    if len(directories)>1:
        for directory in directories:
            mkdir(directory)
        return
    ## if single then do it
    directory = expand_path(directories[0])
    if os.path.isdir(directory):
        if trash_existing: # deletes contents--keeps directory
            for t in glob(f'{directory}/*'):
                trash(t)
        return
    ## walk parent directories making if necessary
    partial_directories = directory.split('/')
    for i in range(len(partial_directories)):
        partial_directory = '/'.join(partial_directories[0:i+1])
        if partial_directory=='' or os.path.isdir(partial_directory):
            continue
        else:
            if os.path.exists(partial_directory): raise Exception("Exists and is not a directory: "+partial_directory)
            os.mkdir(partial_directory)

#####################
## text formatting ##
#####################

def make_valid_python_symbol_name(string):
    """Substitute characters in string so that it can be used as a symbol
    name."""
    string = re.sub(r'[-<!^.+|&/%]','_',string)
    return string

def regularise_unicode(s):
    """Turn unicode symbols into something more ascii"""
    ## superscripts / subscripts 
    for x,y in ( ('⁰','0'), ('¹','1'), ('²','2'), ('³','3'),
                 ('⁴','4'), ('⁵','5'), ('⁶','6'), ('⁷','7'), ('⁸','8'),
                 ('⁹','9'), ('⁺','+'), ('⁻','-'), ('₀','0'), ('₁','1'),
                 ('₂','2'), ('₃','3'), ('₄','4'), ('₅','5'), ('₆','6'),
                 ('₇','7'), ('₈','8'), ('₉','9'), ):
        if x in s:
            s = s.replace(x,y)
    return s

def superscript_numerals(s):
    """Turn unicode symbols into something more ascii"""
    ## superscripts / subscripts 
    for x,y in ( ('⁰','0'), ('¹','1'), ('²','2'), ('³','3'),
                 ('⁴','4'), ('⁵','5'), ('⁶','6'), ('⁷','7'), ('⁸','8'),
                 ('⁹','9'), ('⁺','+'), ('⁻','-'), ):
        if y in s:
            s = s.replace(y,x)
    return s

def subscript_numerals(s):
    """Turn unicode symbols into something more ascii"""
    ## superscripts / subscripts 
    for x,y in (('₀','0'), ('₁','1'),
                 ('₂','2'), ('₃','3'), ('₄','4'), ('₅','5'), ('₆','6'),
                 ('₇','7'), ('₈','8'), ('₉','9'), ):
        if y in s:
            s = s.replace(y,x)
    return s

# def decode_format_string(s):
    # """Get the different arts of a format string, return as dictionary."""
    # g = re.match(r'([<>+-]*)([0-9]*).?([0-9]*)([fgsed])',s).groups()
    # return(dict(prefix=g[0],length=g[1],precision=g[2],type=g[3]))


# def parentheses_style_errors_format(
        # x,s,
        # error_significant_figures=2,
        # tex=False,
        # treat_zero_as_nan=False,
        # default_sig_figs=3,     # if not error to indicate uncertainty use this many significant figures
        # max_leading_zeros=3,    # before use scientific notation
        # fmt='f',                # or 'e'
        # # nan_data_as_blank=False, # do not print nans, does something else instead
        # nan_substitute=None,
# ):
    # """
    # Convert a value and its error in to the form 1234.456(7) where the
    # parantheses digit is the error in the least significant figure
    # otherwise. If bigger than 1 gives 1200(400). If error_significant_figures>1
    # print more digits on the error.
    # \nIf tex=True make nicer still.
    # """
    # ## vectorise
    # if np.iterable(x):
        # return [
            # format_parentheses_style_errors(
                # xi,si,
                # error_significant_figures,
                # tex,
                # treat_zero_as_nan,
                # default_sig_figs,
                # max_leading_zeros,
                # fmt,
                # # nan_data_as_blank=nan_data_as_blank,
                # nan_substitute=nan_substitute,
            # ) for (xi,si) in zip(x,s)]
    # assert fmt in ('e','f'),'Only "e" and "f" formatting implemented.'
    # ## deal with zeros
    # if treat_zero_as_nan:
        # if x==0.: x=np.nan
        # if s==0.: s=np.nan
    # ## data is nan
    # if np.isnan(x):
        # if nan_substitute is None:
            # retval = 'nan(nan)'
        # else:
            # retval = nan_substitute
    # ## data exists but uncertainty is nan, just return with default_sig_figs
    # elif np.isnan(s):
        # if fmt=='f':
            # retval = format_float_with_sigfigs(x,default_sig_figs)
        # elif fmt=='e':
            # retval = format(x,f'0.{default_sig_figs-1}e')
    # ## data and uncertainty -- computed parenthetical error
    # else:
        # if 'f' in fmt:
            # ## format data string 'f'
            # ## format error string
            # t=format(s,'0.'+str(error_significant_figures-1)+'e') ## string rep in form +1.3e-11
            # precision = int(re.sub('.*e','',t))-error_significant_figures+1
            # s = t[0:1+error_significant_figures].replace('.','').replace('e','')
            # x=(round(x/10**precision)+0.1)*10**precision
            # x=format(x,('+0.30f' if fmt[0]=='+' else '0.30f'))
            # i=x.find('.')
            # if precision < 0:
                # x=x[:i-precision+1]
            # elif precision >0:
                # x=x[:i]
                # for j in range(precision): s=s+'0'
            # elif precision==0:
                # x=x[:i]
            # retval = x+'('+s+')'
        # elif 'e' in fmt:
            # ## format data string 'e'
            # r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(x,'0.50e'))
            # value_precision = int(r.group(2))
            # r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(s,'0.50e'))
            # error_precision = int(r.group(2))
            # error_string = r.group(1).replace('.','')[:error_significant_figures]
            # value_string = str(int(np.round(x*10**(-error_precision+error_significant_figures-1))))
            # ## add decimal dot
            # if value_string[0] in '+-':
                # if len(value_string)>2: value_string = value_string[0:2]+'.'+value_string[2:]
            # else:
                # if len(value_string)>1: value_string = value_string[0]+'.'+value_string[1:]
            # error_string = str(int(np.round(s*10**(-error_precision+error_significant_figures-1))))
            # retval = f'{value_string}({error_string})e{value_precision}'
        # else:
            # raise Exception('fmt must be "e" or "f"')
    # if tex:
        # ## separate thousands by \,, unless less than ten thousand
        # # if not re.match(r'[0-9]{4}[.(,]',retval): 
        # #     while re.match(r'[0-9]{4,}[.(,]',retval):
        # #         retval = re.sub(r'([0-9]+)([0-9]{3}[,.(].*)',r'\1,\2',retval)
        # #     retval = retval.replace(',','\,')
        # while True:
            # r = re.match(r'^([+-]?[0-9,]*[0-9])([0-9]{3}(?:$|[.(]).*)',retval)
            # if not r:
                # break
            # retval = r.group(1)+','+r.group(2)
        # retval = retval.replace(',','\\,')
        # ## separate decimal thousands by \,, unless less than ten thousandths
        # r = re.match(r'(.*\.)([0-9]{5,})(.*)',retval)
        # if r:
            # beg,decimal_digits,end = r.groups()
            # retval = beg+r'\,'.join([decimal_digits[i:i+3] for i in range(0,len(decimal_digits),3)])+end
        # ## replace nan with --
        # # retval = retval.replace('nan','--')
        # # retval = retval.replace('--(--)','--')
        # t = re.match(r'(.*)e([+-])([0-9]+)(.*)',retval)
        # if t:
            # beg,exp_sign,exp,end = t.groups()
            # exp = str(int(exp)) # get to single digit int
            # if exp==0:  exp_sign = '' # no e-0
            # if exp_sign=='+': exp_sign = '' # no e+10
            # retval = f'{beg}\\times10^{{{exp_sign}{exp}}}{end}'
        # # retval = re.sub(r'e([+-]?)0?([0-9]+)',r'\\times10^{\1\2}',retval)
        # # retval = re.sub(r'e([+-]?[0-9]+)',r'\\times10^{\1}',retval)
        # retval = '$'+retval.replace('--','-')+'$' # encompass
    # return(retval)
# format_parentheses_style_errors = parentheses_style_errors_format # deprecated
# parentheses_style_errors = parentheses_style_errors_format # deprecated name

# def parentheses_style_errors_decode(string):
    # """Convert string of format '##.##(##)' where parentheses contain
    # an uncertainty esitmate of the least significant digit of the
    # preceding value. Returns this value and an error as floats."""
    # ## if vector of strings given return an array of decoded values
    # if not np.isscalar(string):
        # return(np.array([parentheses_style_errors_decode(t) for t in string]))
    # ## 
    # m = re.match(r'([0-9.]+)\(([0-9]+)\)',str(string))
    # if m is None:
        # warnings.warn('Could not decode `'+str(string)+"'")
        # return np.nan,np.nan
    # valstr,errstr = m.groups()
    # tmpstr = list(re.sub('[0-9]','0',valstr,))
    # i = len(tmpstr)
    # for j in reversed(errstr):
        # i=i-1
        # while tmpstr[i]=='.': i=i-1
        # tmpstr[i] = j
    # return float(valstr),float(''.join(tmpstr))

# def format_tex_scientific(
        # x,
        # sigfigs=2,
        # include_math_environment=True,
        # nan_behaviour='--',     # None for error on NaN, else a replacement string
# ):
    # """Convert a string to scientific notation in tex code."""
    # if np.isnan(x):
        # if nan_behaviour is None:
            # raise Exception("NaN not handled.")
        # else:
            # return('--')
    # s = format(x,'0.'+str(sigfigs-1)+'e')
    # m = re.match(r'(-?[0-9.]+)[eEdD]\+?(-)?0*([0-9]+)',s)
    # digits,sign,exponent =  m.groups()
    # if sign is None:
        # sign=''
    # if include_math_environment:
        # delim = '$'
    # else:
        # delim = ''
    # return r'{delim}{digits}\times10^{{{sign}{exponent}}}{delim}'.format(
        # digits=digits,sign=sign,exponent=exponent,delim=delim)

# def format_numprint(x,fmt='0.5g',nan_behaviour='--'):
    # """Make an appropriate latex numprint formatting command. Fomra number
    # with fmt first. nan_behaviour # None for error on NaN, else a
    # replacement string """
    # if np.isnan(x):
        # if nan_behaviour is None:
            # raise Exception("NaN not handled.")
        # else:
            # return('--')
    # return(f'\\np{{{format(x,fmt)}}}')

# def format_float_with_sigfigs(
        # x,
        # sigfigs,
        # tex=False,
        # fmt='f',                # or 'e'
# ):
    # """Convert a float to a float format string with a certain number of
    # significant figures. This is different to numpy float formatting
    # which controls the number of decimal places."""
    # assert sigfigs>0
    # if tex:
        # thousands_separator = r'\,'
    # else:
        # thousands_separator = ''
    # ## get number of sigfigs rounded and printed into a string, special case for x=0
    # if x!=0:
        # x = float(x)
        # sign_x = np.sign(x)
        # x = np.abs(x)
        # x = float(x)
        # exponent = int(np.floor(np.log10(x)))
        # decimal_part = max(0,sigfigs-exponent-1)
        # s = format(sign_x*np.round(x/10**(exponent+1-sigfigs))*10**(exponent+1-sigfigs),'0.{0:d}f'.format(decimal_part))
    # else:
        # if sigfigs==1:
            # s = '0'
        # else:
            # s = '0.'+''.join(['0' for t in range(sigfigs-1)])
    # ## Split into >=1. <1 components
    # if s.count('.')==1:
        # greater,lesser = s.split('.')
        # sep = '.'
    # else:
        # greater,sep,lesser = s,'',''
    # ## add thousands separator, if number is bigger than 9999
    # if len(greater)>4:
        # indices_to_add_thousands_separator = list(range(len(greater)-3,0,-3))[-1::-1]
        # greater = thousands_separator.join([greater[a:b] for (a,b) in zip(
            # [0]+indices_to_add_thousands_separator,
            # indices_to_add_thousands_separator+[len(greater)])])
    # ## recomprise
    # s = greater+sep+lesser
    # return(s)

# def format_fixed_width(x,sigfigs,width=None):
    # """Creates a exponential form floating point number with given
    # significant figures and total width. If this fails then return a
    # str form of the given width. Default width is possible."""
    # if width is None: 
        # width = sigfigs+6
    # try:
        # return format(x,'>+{}.{}e'.format(width,sigfigs-1))
    # except ValueError:
        # return format(x,'>'+str(width))

def regularise_string_to_symbol(x):
    """Turn an arbitrary string into a valid python symbol. NOT
FINISHED!!  Check out https://github.com/Ghostkeeper/Luna/blob/d69624cd0dd5648aec2139054fae4d45b634da7e/plugins/data/enumerated/enumerated_type.py#L91"""
    x = x.replace('.','_')
    x = x.replace('-','_')
    if re.match('[0-9]',x[0]):
        x = 'x_'+x
    return x
    
def format_string_or_general_numeric(x):
    """Return a string which is the input formatted 'g' if it is numeric or else is str representation."""
    try:
        return format(x,'g')
    except ValueError:
        return(str(x))

# def format_as_disjoint_ranges(x,separation=1,fmt='g'):
    # """Identify ranges and gaps in ranges and print as e.g, 1-5,10,12-13,20"""
    # x = np.sort(x)
    # i = [0]+list(find(np.diff(x)>separation)+1)+[len(x)] # range edges
    # ## Print each range as a-b if range is required or a if single
    # ## value is disjoin. Separate with commas.
    # return(','.join([
        # format(x[a],fmt)+'-'+format(x[b-1],fmt) if (b-a)>1 else format(x[a],fmt)
                    # for (a,b) in zip(
                            # [0]+list(find(np.diff(x)>separation)+1),
                            # list(find(np.diff(x)>separation)+1)+[len(x)])]))



def format_columns(
        data,                   # list or dict (for labels)
        fmt='>11.5g',
        labels=None,
        header=None,
        record_separator='\n',
        delimiter=' ',
        comment_string='# ',
):
    """Print args in with fixed column width. Labels are column
    titles.  NOT QUITE THERE YET"""
    ## if data is dict, reinterpret appropriately
    if hasattr(data,'keys'):
        labels = data.keys()
        data = [data[key] for key in data]
    ## make formats a list as long as data
    if isinstance(fmt,str):
        fmt = [fmt for t in data]
    ## get string formatting for labels and failed formatting
    fmt_functions = []
    for f in fmt:
        def fcn(val,f=f):
            if isinstance(val,str):
                ## default to a string of that correct length
                r = re.match(r'[^0-9]*([0-9]+)(\.[0-9]+)?[^0-9].*',f)
                return(format(val,'>'+r.groups()[0]+'s'))
            elif val is None:
                ## None -- print as None
                r = re.match(r'[^0-9]*([0-9]+)(\.[0-9]+)?[^0-9].*',f)
                return(format(repr(val),'>'+r.groups()[0]+'s'))
            else:
                ## return in given format if possible
                return(format(val,f)) 
        fmt_functions.append(fcn)
    ## begin output records
    records = []
    ## add header if required
    if header is not None:
        records.append(comment_string+header.strip().replace('\n','\n'+comment_string))
    ## labels if required
    if labels!=None:
        records.append(comment_string+delimiter.join([f(label) for (f,label) in zip(fmt_functions,labels)]))
    ## compose formatted data columns
    comment_pad = ''.join([' ' for t in comment_string])
    records.extend([comment_pad+delimiter.join([f(field) for (f,field) in zip(fmt_functions,record)]) for record in zip(*data)])
    t = record_separator.join(records)
    return(record_separator.join(records))

def printcols(*columns):
    """Print the data into readable columns heuristically."""
    print(format_columns(columns))

# def dict_array_to_str(d,keys=None,fmts=None,**kwargs_for_make_table):
    # """Return a string listing the contents of a dictionary made up of
    # arrays of the same length. If no keys specified, print all keys."""
    # if keys is None: keys = list(d.keys())
    # if fmts is None:
        # fmts = [max('12',len(key)) for key in keys]
    # elif isinstance(fmts,str):
        # fmts = [fmts for key in keys]
    # columns = [Value_Array(name=key,val=d[key],fmt=fmt) for (key,fmt) in zip(keys,fmts)]
    # return make_table(columns,**kwargs_for_make_table)

# def recarray_to_str(d,*args,**kwargs):
    # kwargs.setdefault('headers',d.dtype.names)
    # return(tabulate(d,*args,**kwargs))
                            

# def make_latex_table(columns,environment='tabular',preamble='',postscript='',):
    # """Convenient method to create a simple LaTeX table from data.

    # Columns is a list of Value_Arrays or dictionaries with the following keys:

    # val -- an array or somesuch, not necessarily numerical
    # fmt -- (optional?) format string, or an arbitrary function that
           # converts elements of data to a string
    # latex_table_alignment -- (optional?) latex table alignment code
    # name -- (optional?) To go at top of table.

    # Other options:

    # environment -- enclosing latex environment, i.e., tabular or longtable
    # preamble -- appears before environment
    # postscript -- appears after environment """
    # ## convert dictionary to Value_Array if necessary
    # for i in range(len(columns)):
        # if isinstance(columns[i],dict):
            # columns[i] = Value_Array(**columns[i])
    # string = '\n'.join([
        # preamble,
        # r'\begin{'+environment+'}{'+''.join([col.latex_table_alignment for col in columns])+'}',
        # # r'\hline\hline',
        # r'\toprule',
        # ' & '.join([col.name for col in columns])+' \\\\',
        # # r'\hline',
        # r'\midrule',
        # make_table(columns,field_separator=' & ',row_separator=' \\\\\n',comment='',print_description=False,print_keys=False)+' \\\\',
        # # r'\hline\hline',
        # r'\bottomrule',
        # r'\end{'+environment+'}',
        # postscript,
    # ])
    # return string

# def make_table(columns,header=None,row_separator='\n',
               # field_separator=' ', comment='#',
               # print_description=True, print_keys=True):
    # """Convenient method to create a simple table from data.\n
    # Columns is a list of Value_Array objects\n
    # Other options:\n
    # header -- put at top of table
    # """
    # ## keys and data
    # rows = list(zip(*[t.get_formatted_list(include_column_label=print_keys) for t in columns]))
    # rows = [field_separator.join(row) for row in rows]
    # rows[0] = comment+' '+rows[0] # comment keys
    # ## add space to align commented keys with data
    # comment_space = ''.join([' ' for t in comment])
    # for i in range(len(comment)):
        # comment_space = comment_space+' '
    # rows[1:] = [comment_space+t for t in rows[1:]]
    # ## header and description
    # if print_description:
        # for col in columns[-1::-1]:
            # rows.insert(0,comment+' '+col.name.strip()+': '+col.description)
    # if header!=None:
        # for header_line in header.split('\n')[-1::-1]:
            # rows.insert(0,comment+' '+header_line)
    # return row_separator.join(rows)

# def tabulate(tabular_data, headers=[], tablefmt="plain",
             # floatfmt="g", numalign="decimal", stralign="left",
             # missingval=""):
    # """From tabulate.tabulate with "csv" tablefmt added."""
    # import tabulate as _tabulate
    # _tabulate._table_formats["csv"] = _tabulate.TableFormat(
        # lineabove=None, linebelowheader=None,
        # linebetweenrows=None, linebelow=None,
        # headerrow=_tabulate.DataRow("", ", ", ""),
        # datarow=_tabulate.DataRow("", ", ", ""),
        # padding=0, with_header_hide=None)
    # return(_tabulate.tabulate(tabular_data,headers,tablefmt,floatfmt,numalign,stralign,missingval))

#####################################
## save / load /convert array data ##
#####################################

def find_inverse(index_array,total_shape):
    """Convert an integer index array into a boolean mask."""
    retval = np.full(total_shape,False)
    retval[index_array] = True
    return retval

def cast_abs_float_array(x):
    """Return as 1D array of absolute floating point values."""
    return np.abs(np.asarray(x,dtype=float))

# def cast_normalise_species_string_array(x):
    # """Return as 1D array of strings of normalised species names."""
    # from . import database
    # return np.array(database.normalise_species(x),dtype=str)

def repmat_vector(x,repeats=(),axis=-1):
    """x must be 1D. Expand to as many other dimension as length of
    repeats. Put x variability on axis. All other dimensions just
    repeat this one. """
    x = np.array(x)
    assert len(x.shape)==1,'1D only'
    retval = np.tile(x,list(repeats)+[1])
    if not (axis==-1 or axis==(retval.ndim-1)):
        return(np.moveaxis(np.tile(x,list(repeats)+[1]),-1,axis))
    return(retval)

def repmat(x,repeats):
    """Copy and expand matrix in various directions. Length of repeats
    must match dimension of x. If greater, then extra dimension are
    PREPENDED to x. Each (integer>0) element of repeat determines how
    many copies to make along that axis. """
    ## ensure types
    repeats = ensure_iterable(repeats)
    x = np.array(x)
    ## ensure all dimensions are included -- otherwise assume 1
    if len(repeats)<x.ndim:               #ensure long enough to match array ndim
        repeats.extend(np.ones(len(repeats-x.ndim)))
    ## ensure array has enough dimensions -- prepending empty dimensions
    if x.ndim<len(repeats):               
        x = np.reshape(x,[1 for t in range(len(repeats)-x.ndim)]+list(x.shape))
    ## for each non-unity repeat increase size of the array
    for axis,repeat in enumerate(repeats):
        if repeat==1: continue
        x = np.concatenate(tuple([x for t in range(repeat)]),axis=axis)
    return(x)

# def transpose(x):
    # """Take any 2D interable and transpose it. Returns a list of
    # tuples. \n\nnumpy.transpose is similar but returns an array cast to the
    # most general type."""
    # return list(zip(*tuple(x)))

def file_to_array_unpack(*args,**kwargs):
    """Same as file_to_array but unpack data by default."""
    kwargs.setdefault('unpack',True)
    return file_to_array(*args,**kwargs)

# def txt_to_array_via_hdf5(path,**kwargs):
    # """Loads ASCII text to an array, converting to hdf5 on the way
    # because it is faster.  Also deletes commented rows. MASSIVE
    # HACK."""
    # import os,tempfile,subprocess
    # ## default comment char is #
    # kwargs.setdefault('comments','#')
    # comments = kwargs.pop('comments')
    # path = os.path.expanduser(path)
    # ## expand path if possible and ensure exists - or else awk will hang
    # if not os.path.lexists(path):
        # raise IOError(1,'file does not exist',path)
    # tmpfile=tempfile.NamedTemporaryFile()
    # command = 'cat "'+path+'" | sed "/^ *'+comments+'/d" | h5fromtxt '+tmpfile.name
    # (status,output)=subprocess.getstatusoutput(command)
    # assert status==0,"Conversion to hdf5 failed:\n"+output
    # retval = hdf5_to_array(tmpfile.name,**kwargs)
    # return retval

def hdf5_to_array(filename,unpack=False,dataset=None,usecols=None):
    """Loads dataset dataset in hdf5 file into an array. If None then look
    for 'data', if its not there load first dataset. """
    import sys,h5py,os
    try:
        f = h5py.File(os.path.expanduser(filename),'r')
    except IOError:
        sys.stderr.write('Tried opening file: '+filename+'\n')
        raise
    if dataset is None:
        if 'data' in f:
            x = np.array(f['data']) # 'data'
        else:
            x = np.array(list(f.items())[0][1]) # first 
    else:
        x = np.array(f[dataset]) # dataset
    f.close()
    if usecols is not None:
        x = x[:,usecols]
    if unpack==True:
        x = x.transpose()
    return(x)

def hdf5_get_attributes(filename):
    """Get top level attributes."""
    with h5py.File(expand_path(filename_or_hdf5_object),'r') as fid:
        return {key:val for key,val in fid.attrs.items()}

def hdf5_to_numpy(value):
    if not np.isscalar(value):
        value = value[()]
    ## convert bytes string to unicode
    if np.isscalar(value):
        if isinstance(value,bytes):
            value = value.decode()
    else:
        ## this is a test for bytes string (kind='S') but for
        ## some reason sometimes (always?) loads as object
        ## type
        if value.dtype.kind in ('S','O'):
            # value = np.asarray(value,dtype=str)
            value = np.asarray([t.decode() for t in value],dtype=str)
    return value

def numpy_to_hdf5(value):
    ## deal with missing unicode type in hdft
    ## http://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
    if not np.isscalar(value) and value.dtype.kind=='U':
        value = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))
    return value
    
def hdf5_to_dict(fid):
    """Load all elements in hdf5 into a dictionary. Groups define
    subdictionaries. Scalar data set as attributes."""
    ## open file if necessary
    if isinstance(fid,str):
        with h5py.File(expand_path(fid),'r') as fid2:
            return hdf5_to_dict(fid2)
    retval = {}            # the output data
    ## load attributes
    for tkey,tval in fid.attrs.items():
        retval[str(tkey)] = hdf5_to_numpy(tval)
    ## load data and subdicts
    for key,val in fid.items():
        if isinstance(val,h5py.Dataset):
            retval[str(key)] = hdf5_to_numpy(val)
        else:
            retval[str(key)] = hdf5_to_dict(val)
    return retval

def dict_to_hdf5(fid,data,compress=False,verbose=True):
    """Save all elements of a dictionary as datasets, attributes, or
    subgropus in an hdf5 file."""
    if isinstance(fid,str):
        ## open file if necessary
        fid = expand_path(fid)
        mkdir(dirname(fid)) # make leading directories if not currently there
        with h5py.File(fid,mode='w') as new_fid:
            dict_to_hdf5(new_fid,data,compress,verbose)
            return
    ## add data
    for key,val in data.items():
        if isinstance(val,dict):
            ## recursively create groups
            group = fid.create_group(key)
            dict_to_hdf5(group,val,compress,verbose)
        else:
            if isinstance(val,np.ndarray):
                ## add arrays as a dataset
                if compress:
                    kwargs={'compression':"gzip",'compression_opts':9}
                else:
                    kwargs = {}
                fid.create_dataset(key,data=numpy_to_hdf5(val),**kwargs)
            else:
                ## add non-array data as attribute
                try:
                    fid.attrs.create(key,val)
                except TypeError as error:
                    if verbose:
                        raise error
                        print(error)

def append_to_hdf5(filename,**keys_vals):
    """Added key=val to hdf5 file."""
    import h5py
    with h5py.File(expand_path(filename),'a') as d:
        for key,val in keys_vals.items() :
            d[key] = val


# def print_hdf5_tree(filename_or_hdf5_object,make_print=True):
    # """Print out a tree of an hdf5 object or file."""
    # if not isinstance(filename_or_hdf5_object,h5py.HLObject):
        # filename_or_hdf5_object = h5py.File(expand_path(filename_or_hdf5_object),'r')
    # retval = []
    # for key in filename_or_hdf5_object.keys():
        # if isinstance(filename_or_hdf5_object[key],h5py.Dataset):
            # retval.append('['+repr(key)+']')
        # else:
            # sub_retval = hdf5_print_tree(filename_or_hdf5_object[key],make_print=False)
            # retval.extend(['['+repr(key)+']'+t for t in sub_retval])
    # if make_print:
        # ## original call -- print
        # print('\n'.join(retval))
    # else:
        # ## recursive call -- return data
        # return(retval)

# def print_dict_tree(d):
    # print(format_dict_key_tree(d))

# def pprint_dict_recursively(d,max_indent_level=None,indent_level=0):
    # """Actual works on anything with an 'items' method."""
    # indent = ''.join(['  ' for t in range(indent_level)])
    # for (key,val) in d.items():
        # if hasattr(val,'items') and (max_indent_level is None or indent_level < max_indent_level):
            # print(indent+str(key))
            # pprint_dict_recursively(val,max_indent_level,indent_level=indent_level+1)
        # else:
            # print(indent+str(key)+': '+repr(val))

# def walk_dict_items(d,maxdepth=np.inf):
    # """A generator that walks through dictionary d (depth first) and any
    # subdictionary returning keys and values ones by one."""
    # if maxdepth<0: return
    # for (key,val) in d.items():
        # yield(key,val)
        # if isinstance(val,dict):
            # for tkey,tval in walk_dict_items(val,maxdepth=maxdepth-1):
                # yield(tkey,tval)

# def format_dict_key_tree(d,prefix='└─ '):
    # """Print out a tree of a dicts keys, not the values though."""
    # s = []
    # for i,key in enumerate(d):
            # # for t in range(depth): prefix = '│  '+prefix
        # s.append(prefix+key)
        # if hasattr(d[key],'keys'):
            # s.append(format_dict_key_tree(d[key],'    '+prefix))
    # return('\n'.join(s))

# def recarray_to_dict(ra):
    # """Convert a record array to a dictionary. There may be a builtin
    # way to do this."""
    # retval = dict()
    # for key in ra.dtype.names:
        # retval[key] = np.array(ra[key])
    # return(retval)
    # # return({key:np.array(ra[key]) for key in ra.dtype.names})

def dict_to_recarray(d):
    """Convert a dictionary of identically sized arrays into a
    recarray. Names are dictionary keys. Add some dictionary-like
    methods to this particular instance of a recarray."""
    if len(d)==0:
        ra = np.recarray((0),float) # no data
    else:
        ra = np.rec.fromarrays([np.array(d[t]) for t in d], names=list(d.keys()),)
    return(ra)

def make_recarray(**kwargs):
    """kwargs are key=val pair defining arrays of equal length from
    which to make recarray."""
    ra = np.rec.fromarrays(kwargs.values(),names=list(kwargs.keys()))
    return(ra)

# def append_fields_to_recarray(recarray,**keyvals):
    # """Add a new field of name name to recarray. All values of keyvals must
    # be the same length as recarray."""
    # d = recarray_to_dict(recarray)
    # for key,val in keyvals.items():
        # if np.isscalar(val): val = np.full(recarray.shape,val) # expand scalara data
        # d[key] = val
    # return(dict_to_recarray(d))

# def recarray_concatenate_fields(*recarrays):
    # """Join recarrays into one. Both must be the same length.  Common keys
    # causes an error."""
    # for t in recarrays: assert len(t)==len(recarrays[0]),'Recarrays not all same length' # check lengths equal
    # ## check keys are unique
    # all_keys = []
    # for t in recarrays: all_keys.extend(t.dtype.names)
    # assert len(all_keys)==len(np.unique(all_keys)),f'keys not unique: {repr(all_keys)}'
    # ## join into one recarray
    # keys_vals = dict()
    # for t in recarrays:
        # for key in t.dtype.names:
            # keys_vals[key] = t[key]
    # return(dict_to_recarray(keys_vals))

# def concatenate_recarrays_unify_dtype(recarrays_list,casting='safe',**concatenate_kwargs):
    # """Concatenate recarrays, but first try and align dtype to avoid
    # promotion errors. Various heuristic design decisions in here."""
    # assert(len(recarrays_list)>0),'one at recarray to contenate at least'
    # ## get all dtypes as a list of list of strings to manipulate
    # dtypes = [
        # [(t.names[i],t[i].str) for i in range(len(t))]
        # for t in [t.dtype for t in recarrays_list]]
    # ## determine a unifying dtype to cast all arrays to
    # dtype_cast = dtypes[0]      # initialise to first
    # for dtype in dtypes[1:]:    # update with others
        # if dtype==dtype_cast: continue # already compliant
        # for idtype,((namei,stri),(namej,strj)) in enumerate(zip(dtype,dtype_cast)): # check individual dtypes, order must be the same
            # assert namei==namej,f'Names do not match {repr(namei)} {repr(namej)}' # names do not match
            # if stri==strj:
                # continue
            # elif stri[0:2]=='<U' and strj[0:2]=='<U': # if strings are different lengths, cast to the longer
                # max_length = max(int(stri[2:]),int(strj[2:]))
                # dtype_cast[idtype] = (namei,'<U'+str(max_length))
            # elif (stri[0:2]=='<i' and strj[0:2]=='<f') or (strj[0:2]=='<i' and stri[0:2]=='<f'): # i and f, cast to f8
                # dtype_cast[idtype] = (namei,'<f8')
            # else:               # this combination not implemented
                # raise Exception(f"dtype rectification not implemented: {repr((namei,stri))} {repr((namej,strj))}")
    # return(                     # return concatenated array
        # np.concatenate(         # concatenate 
            # [t.astype(dtype_cast,casting=casting) for t in recarrays_list], # cast individudal arrays
            # **concatenate_kwargs))

# def recarray_remove(recarray,*keys):
    # """Return a copy of recarray with keys removed."""
    # return(dict_to_recarray({key:val for key,val in recarray_to_dict(recarray).items() if key not in keys}))

# def append_to_recarray(recarray,**kwargs):
    # """Make a new and longer recarray. kwargs must match the complete dtype
    # of the recarray.  WOULDN'T CONCATENAT WORK WELL ENOUGH?"""
    # assert set(recarray.dtype.names)==set(kwargs), 'dtype.names in old and new recarrays do not match. Old: '+repr(set(recarray.dtype.names))+' new: '+repr(set(kwargs))
    # new_recarray = np.rec.fromarrays([kwargs[key] for key in recarray.dtype.names],dtype=recarray.dtype)
    # return(np.append(recarray,new_recarray))

# def recarray_to_hdf5(filename,recarray,header=None,**kwargs):
    # """Used dict_to_hdf5 to save a recarray to hdf5 file."""
    # d = recarray_to_dict(recarray)
    # if header is not None: d['README'] = str(header)
    # return(dict_to_hdf5(filename,d,**kwargs))

# def kwargs_to_hdf5(filename,**kwargs):
    # return(dict_to_hdf5(filename,dict(**kwargs)))


# def kwargs_to_directory(directory, **dict_to_directory_kwargs,):
    # dict_to_directory(directory,dict_to_directory_kwargs)

# def dict_to_directory(
        # directory,
        # dictionary,
        # array_format='h5',
        # remove_string_margin=True,
        # make_directory=True
# ):
    # """Create a directory and save contents of dictionary into it."""
    # if make_directory: mkdir(directory)
    # for key,val in dictionary.items():
        # ## save strings to text files
        # if isinstance(val,str):
            # if remove_string_margin:
                # val = re.sub(r'(^|\n)\s+\b',r'\1',val) # delete white space at beginning of all lines
            # string_to_file(directory+'/'+str(key),val)
        # ## save numpy arrays in binary format, or hdf5, or text file
        # elif isinstance(val,np.ndarray):
            # if   array_format == 'npy':  array_to_file(directory+'/'+str(key)+'.npy',val)
            # elif array_format == 'npz':  array_to_file(directory+'/'+str(key)+'.npz',val)
            # elif array_format == 'text': array_to_file(directory+'/'+str(key),val)
            # elif array_format == 'h5':   array_to_file(directory+'/'+str(key)+'.h5',val)
            # else:   raise Exception('array_format must be one of "npy", "npz", "text", "hdf5"')
        # ## save dictionaries as subdirectories
        # elif isinstance(val,dict):
            # dict_to_directory(directory+'/'+str(key),val,array_format)
        # ##
        # else:
            # raise Exception('Do not know how to save: key: '+repr(key)+' val: '+repr(val))

# def directory_to_dict(directory):
    # """Load all contents of a directory into a dictiionary, recursive."""
    # directory = expand_path(directory)
    # directory = re.sub(r'(.*[^/])/*',r'\1',directory) # remove trailing /
    # retval = {}
    # for filename in glob(directory+'/*'):
        # filename = filename[len(directory)+1:]
        # extension = os.path.splitext(filename)[1]
        # ## load subdirectories as dictionaries
        # if os.path.isdir(directory+'/'+filename):
            # retval[filename] = directory_to_dict(directory+'/'+filename)
        # ## load binary data
        # if extension in ('.npy','.h5','.hdf5'):
            # # retval[filename[:-4]] = np.load(diarectory+'/'+filename)
            # retval[filename[:-len(extension)]] = file_to_array(directory+'/'+filename)
        # ## read README as string
        # elif filename in ('README',):
            # retval[filename] = file_to_string(directory+'/'+filename)
        # ## else assume text data
        # elif filename in ('README',):
            # retval[filename] = file_to_array(filename)
    # return(retval)


# class Data_Directory:
    # """Data is stored in a directory and accessed by key."""


    # def __init__(self,directory_path):
        # self.root = expand_path(re.sub(r'(.*[^/])/*',r'\1',directory_path))
        # self._cache = {}

    # def __getitem__(self,key):
        # ## return from cache if possible
        # if key in self._cache: return(self._cache[key])
        # ## get full filename. If it does not exist look for a version
        # ## with a unique extension
        # filename = f'{self.root}/{key}'
        # if not os.path.exists(filename):
            # try:
                # filename = glob_unique(f'{self.root}/{key}.*')
            # except FileNotFoundError:
                # raise KeyError(f"Cannot find file with or without extension: {self.root}/{key}")
        # ## if recursive subdir access
        # if '/' in key:
            # i = key.find('/')
            # retval = self[key[:i]][key[i+1:]]
        # ## load subdirectories as dictionaries
        # elif os.path.isdir(filename):
            # retval = Data_Directory(filename)
        # ## read README as string
        # elif filename in ('README',):
            # retval = file_to_string(filename)
        # ## load array data
        # else:
            # retval = file_to_array(filename)
        # ## save to cache and return
        # self._cache[key] = retval
        # return(retval)

    # def keys_deep(self):
        # """Keys expanded to all levels."""
        # retval = []
        # for filename in glob(f'{self.root}/*'):
            # key = filename[len(self.root)+1:]
            # if os.path.isdir(filename):
                # retval.extend([f'{key}/{t}' for t in self[key].keys()])
            # else:
                # retval.append(key)
        # return(retval)

    # def keys(self):
        # """Keys in top level."""
        # return([t[len(self.root)+1:] for t in glob(f'{self.root}/*')])

    # def __len__(self):
        # return(len(self.keys()))

    # def __str__(self):
        # return(str(self.keys()))



# def percentDifference(x,y):
    # """Calculate percent difference, i.e. (x-y)/mean(x,y)."""
    # return 100.*2.*(x-y)/(x+y)

# def notNan(arg):
    # """Return part of arg that is not nan."""
    # return arg[~np.isnan(arg)]

# def bestPermutation(x,y):
    # """Find permuation of x that best matches y in terms of
    # rms(x-y). A brute force approach"""
    # assert len(x)==len(y), 'Inputs must have same shape.'
    # x = copy(np.array(x))
    # y = copy(np.array(y))
    # permutations = itertools.permutations(list(range(len(x))))
    # bestRms = np.inf
    # bestPermutation = None
    # while True:
        # try:
            # permutation = np.array(next(permutations))
            # nextRms = rms(x[permutation]-y)
            # if nextRms < bestRms:
                # bestPermutation = permutation
                # bestRms = nextRms
        # except StopIteration:
            # break
    # return bestPermutation

# def minimise_differences(a,b):
    # """Returns an index which reorders a to best match b. This is done by
    # finding the best possible match, fixing that, then the second etc. So
    # may not give the best summed RMS or whatever."""
    # assert len(a)==len(b), "Nonmatching array lengths."
    # # ## if one of the arrays is incomplete then pad with infs. As a
    # # ## side effect of numpy argmin these will retain their current
    # # ## ordering, which may not be reliable in the future.
    # # if len(a)<len(b):
        # # a = np.concatenate((a,np.inf*np.ones(len(b)-len(a)),))
    # # if len(b)<len(a):
        # # b = np.concatenate((b,np.inf*np.ones(len(a)-len(b)),))
    # x,y = np.meshgrid(a,b)          
    # t = np.abs(x-y)                    # all possible differences
    # ifinal = np.ones(a.shape,dtype=int)*-1          # final sorted ai indices
    # ilist = list(range(len(a))) 
    # jlist = list(range(len(b)))
    # ## find best match, reducing size of t iteratively
    # while t.shape[0]>0 and t.shape[1]>0:
        # i,j = np.unravel_index(np.argmin(t),t.shape) # minimum difference
        # ifinal[ilist[i]] = jlist[j] # save this index
        # ilist.pop(i)                # remove the minimum values from further consideration
        # jlist.pop(j)                # remove the minimum values from further consideration
        # t = t[list(range(0,i))+list(range(i+1,t.shape[0])),:] # shrink array
        # t = t[:,list(range(0,j))+list(range(j+1,t.shape[1]))] # shrink array
    # return(ifinal)

# def user_string():
    # """Return string describing user."""
    # import getpass
    # return getpass.getuser()+'@'+os.uname()[1]

# def sum_in_quadrature(*args):
    # """returns sqrt(sum(args**2))"""
    # return(np.sqrt(sum([np.square(t) for t in args])))

# def triangle_sum(x,axis=None):
    # """Return sqrt of sum of squares.

    # If input is an array then calculates according to axis, like other
    # numpy functions.  Else, triangle sums all elements of x, vector or
    # not (axis kwarg ignored).
    # """
    # x = np.asarray(x)
    # ## Tries to sum as numpy array.
    # try:
        # return (np.sqrt((x**2).sum(axis=axis)))
    # ## Upon failure, sums elements
    # except:
        # retval = 0.
        # for xx in x: retval = retval + xx**2
        # return np.sqrt(retval)

# def sum_in_quadrature(*args):
    # return(np.sqrt(np.sum([t**2 for t in args],axis=0)))

# def combine_product_errors(product,factors,errors):
    # """Combine uncorrelated normally distributed errors dx, dy, dz etc
    # to get error of x*y*z etc"""
    # return(product*np.sqrt(np.sum([(error/factor)**2 for (error,factor) in zip(errors,factors)])))

# def set_dict_default(dictionary,default_values_dictionary):
    # """Copy all keys,vals from default_values_dictionary to dictionary
    # unless key is already present in dictionary. """
    # for key,val in default_values_dictionary.items():
        # if not key in dictionary:
            # dictionary[key] = val

# def update_dict(dictionary,update_values_dictionary):
    # """Update dictionary with all keys,vals in
    # update_values_dictionary. Raise error if key in
    # update_values_dictionary is not in dictionary."""
    # for key,val in update_values_dictionary.items():
        # if key in dictionary:
            # dictionary[key] = val
        # else:
            # raise Exception("Updating key not in dictionary: "+repr(key))

# def safeUpdate(a,b):
    # """Update dictionary a from dictionary b. If any keys in b not
    # found in a an error is raised. Update of a done in place."""
    # i = isin(list(b.keys()),list(a.keys()))
    # if not all(i):
        # raise Exception('Bad keys: '+ str([key for (key,ii) in zip(list(b.keys()),i) if not ii]))
    # a.update(b)

# def safe_update_attr(a,b):
    # """Update attributes of a from dictionary b. If any keys in b not
    # found in a an error is raised. Update of a done in place."""
    # for (key,val) in list(b.items()):
        # if hasattr(a,key):
            # setattr(a,key,val)
        # else:
            # raise AttributeError('Bad attr: '+key)

# def index_dict_array(d,i):
    # """Assumes all elements of dictionary d are arrays which can be
    # indexed by i. Then returns a copy of d containing such
    # subarrays."""
    # dnew = {}
    # for key in d:
        # dnew[key] = d[key][i]
    # return dnew

# def match_array_to_regexp(array,regexp,strip=True):
    # """Match an array of strings to a regular expression and return a
    # boolean array. If strip=True strip whitespace from strings."""
    # # assert array.ndim==1, "Currently only implemented for 1D arrays."
    # if strip:
        # return(np.array([bool(re.match(regexp,t.strip())) for t in array]))
    # else:
        # return(np.array([bool(re.match(regexp,t)) for t in array]))

# def pause(message="Press any key to continue..."):
    # """Wait for use to press enter. Not usable outsdie linux."""
    # input(message)

def get_clipboard():
    """Get a string from clipboard."""
    status,output = subprocess.getstatusoutput("xsel --output --clipboard")
    assert status==0, 'error getting clipboard: '+output
    return output

def set_clipboard(string):
    """Send a string to clipboard."""
    pipe=os.popen(r'xsel --input --clipboard','w');
    pipe.write(string)
    pipe.close()

def cl(x,fmt='0.15g'):
    """Take array or scalar x and convert to string and put on clipboard."""
    if np.isscalar(x):
        if isinstance(x,str):
            set_clipboard(x)
        else:
            set_clipboard(format(x,fmt))
    else:
        set_clipboard(array_to_string(x,fmt=fmt))

def pa():
    """Get string from clipboard. If possible convert to an array."""
    x = get_clipboard()
    try:
        return string_to_array(x)
    except:
        return x

# def wget(url):
    # """Downloads data from url and returns it."""
    # import subprocess
    # code,data = subprocess.getstatusoutput("wget --restrict-file-names=nocontrol --quiet -O /dev/stdout '"+url+"'")
    # if code!=0: raise Exception("wget failed: "+str(code)+" "+str(data))
    # return(data)

# def printfmt(*args,fmt='8g'):
    # """Print all args with the same format."""
    # print(' '.join([format(arg,fmt) for arg in args]))

# def printcols(*args,fmt='15',labels=None):
    # """Print args in with fixed column width. Labels are column
    # titles."""
    # if not np.iterable(args) or isinstance(args,str): 
        # args = (args,)
    # else:
        # args = [arg for arg in args]
        # for i in range(len(args)): 
            # if not np.iterable(args[i]) or isinstance(args[i],str):
                # args[i] = [args[i]]
    # if labels!=None:
        # assert len(labels)==len(args),'Not enough/too many labels.'
        # print((' '.join(format(t,fmt) for t in labels)))
    # print((array2str(list(zip(*args)),fmt=fmt)))

# def recarray_to_file(filename,recarray,*args,**kwargs):
    # """Output a recarray to a text file."""
    # return(string_to_file(filename,recarray_to_string(recarray,*args,**kwargs)))

# def dict_array_to_file(filename,d,**kwargs_for_dict_array_to_str):
    # """Write dictionary of arrays to a file."""
    # kwargs_for_dict_array_to_str.setdefault('print_description',False)
    # f = open(os.path.expanduser(filename),'w')
    # f.write(dict_array_to_str(d,**kwargs_for_dict_array_to_str))
    # f.close()

def glob(path='*',regexp=None):
    """Shortcut to glob.glob(os.path.expanduser(path)). Returns a list
    of matching paths. Also sed alphabetically. If re is provided filter names accordingly."""
    retval = sorted(glob_module.glob(os.path.expanduser(path)))
    if regexp is not None:
        retval = [t for t in retval if re.match(regexp,t)]
    return(retval)

def glob_unique(path):
    """Match glob and return as one file. If zero or more than one file
    matched raise an error."""
    filenames = glob_module.glob(os.path.expanduser(path))
    if len(filenames)==1:
        return(filenames[0])
    elif len(filenames)==0:
        raise FileNotFoundError('No files matched path: '+repr(path))
    elif len(filenames)>1:
        raise Exception('Multiple files matched path: '+repr(path))

# def grep(regexp,string=None,filename=None):
    # """Poor approx to grep."""
    # if string is not None and filename is None:
        # pass
    # elif string is None and filename is not None:
        # string = file_to_string(filename)
        # pass
    # else:
        # raise Exception('Require string or filename, but not both.')
    # return('\n'.join([t for t in string.split('\n') if re.match(regexp,t)]))

# def rootname(path,recurse=False):
    # """Returns path stripped of leading directories and final
    # extension. Set recurse=True to remove all extensions."""
    # path = os.path.splitext(os.path.basename(path))[0]
    # if not recurse or path.count('.')+path.count('/') == 0:
        # return path
    # else:
        # return rootname(path,recurse=recurse)

# def get_filename_extension(path):
    # """Return extension, or return None if none."""
    # t = os.path.splitext(path)
    # if t[1]=='': return(None)
    # return(t[1][1:])

def basename(path):
    """Remove all leading directories. If the path is a directory strip
    final '/'."""
    if path[-1]=='/':
        return(basename(path[:-1]))
    else:
        return(os.path.basename(path))

def dirname(path):
    """Return full directory prefix."""
    try:
        i = path[-1::-1].index('/')
        return(path[0:len(path)-i])
    except ValueError:
        return('./')

def polyfit(
        x,y,
        dy=None,
        order=0,
        fixed=None,
        do_print=False,
        do_plot=False,
        error_on_missing_dy=True,
        # plot_kwargs=None,
):
    """
    Polyfit with weights calculated from provided standard
    deviation. Will ignore data with NaNs in any of x, y, or dy. If
    dy=None, or a dy is a constant, or dy is all 0., then a constant
    value (default 1) will be used. If some dy is 0, then these will
    be set to NaN and ignored.
    \nInputs:\n
    x - independent variables
    y - dependent variable
    dy - standard error of y
    order - order of polynomial to fit
    fixed - parameter to not vary, fixed values in dict
            indexed by order, e.g. {0:100,} fixes constant term
    extended_output - output more, default is False
    print_output - print some data, default is False
    plot_output - issue plotting commands, default is False
    return_style - 'list' or 'dict'.
    plotkwargs - a dictionary of kwargs passed to plot
    \nOutputs:\n
    p - the polynomial coefficients
    If extended_output=True then also returns:
    dp - standard error in p, will only be accurate if order is correct
    f - a function representing this polynomial
    residuals - of fit
    chisqprob - probability of arriving at these residuals 
                (or greater ones given the standard errors dy with the
                proposed polynomial model.
    """
    x,y = np.array(x),np.array(y) # ensure types
    if dy is None: dy = np.full(x.shape,1.,dtype=float) # default uncertainty if Noneprovided
    if type(dy) in [float,int,np.float64,np.int64]: dy = dy*np.ones(y.shape) # vectorise dy if constant given
    dy = np.array(dy)           # ensure array
    if error_on_missing_dy and (np.any(np.isnan(dy)) or np.any(dy==0)): raise Exception("Incomplete dy data, zero or nan.") # raise error bad dy 
    xin,yin,dyin = copy(x),copy(y),copy(dy) # save original data
    ## deal with nan or zero data by not fitting to them. If all zero or nan then set error to 1.
    dy[dy==0] = np.nan
    i = ~(np.isnan(x)|np.isnan(y)|np.isnan(dy))
    if np.any(i):
        x,y,dy = x[i],y[i],dy[i] # reduce data to known uncertianteis
    else:
        dy = np.full(dy.shape,1.) # set all uncertainties to 1 if none are known
        i = np.full(dy.shape,True)
    ## reduce polynomial order to match data length
    order = min(order,len(x)-1)
    ## solve linear least squares equation, do not include fixed parameters in matrix
    W = np.diag(1/dy)
    if fixed is None:
        At = np.array([x**n for n in range(order+1)])
        A = At.transpose()
        AtW = np.dot(At,W)
        AtWA = np.dot(AtW,A)
        invAtWA = linalg.inv(AtWA)
        invAtWAdotAtW  = np.dot(invAtWA,AtW)
        p = np.dot(invAtWAdotAtW,y)
    else:
        y_reduced = copy(y)
        At = []
        for n in range(order+1):
            if n in fixed:  y_reduced = y_reduced - fixed[n]*x**n
            else:           At.append(x**n)
        At = np.array([t for t in At])
        A = At.transpose()
        AtW = np.dot(At,W)
        AtWA = np.dot(AtW,A)
        invAtWA = linalg.inv(AtWA)
        invAtWAdotAtW  = np.dot(invAtWA,AtW)
        p_reduced = list(np.dot(invAtWAdotAtW,y_reduced))
        p = []
        for n in range(order+1):
            if n in fixed:
                p.append(fixed[n])
            else: 
                p.append(p_reduced.pop(0))
    p = np.flipud(p) # to conform to polyfit convention
    ## function
    f = lambda x: np.polyval(p,x)
    ## fitted values
    yf = f(xin)
    ## residuals
    residuals = yin-yf
    ## chi-square probability
    if dy is None:
        chisq = chisqnorm = chisqprob = None
    else:
        chisq = (residuals[i]**2/dy**2).sum()
        chisqnorm = chisq/(len(x)-order-1-1)
        # chisqprob = stats.chisqprob(chisq,len(x)-order-1-1)
        chisqprob = stats.distributions.chi2.sf(chisq,len(x)-order-1-1)
    ## stdandard error paramters (IGNORING CORRELATION!)
    if dy is None:
        dp = None
    else:
        dp = np.sqrt(np.dot(invAtWAdotAtW**2,dy**2))
        dp = np.flipud(dp) # to conform to polyfit convention
    ## a nice summary message
    if do_print:
        print(('\n'.join(
            ['             p             dp']
            +[format(a,'14.7e')+' '+format(b,'14.7e') for (a,b) in zip(p,dp)]
            +['chisq: '+str(chisq),'chisqprob: '+str(chisqprob),
              'rms: '+str(rms(residuals)),
              'max_abs_residual: '+str(abs(residuals).max())]
            )))
    ## a nice plot
    if do_plot:
        fig=plotting.plt.gcf()
        fig.clf()
        ax = subplot(0)
        ax.errorbar(x,y,dy,label='data') 
        ax.errorbar(x,yf,label='fit') 
        plotting.legend()
        ax = subplot(1)
        ax.errorbar(x,residuals,dy,label='residual error')
        plotting.legend()
    ## return 
    return dict(
        x=xin,y=yin,dy=dyin,
        p=p,dp=dp,
        yf=f(xin),f=f,
        residuals=residuals,
        chisqprob=chisqprob,chisq=chisq,chisqnorm=chisqnorm,
        fixed=fixed)

# def dot(x,y):
    # """Like numpy dot but sums over last dimension of first argument
    # and first dimension of last argument."""
    # return np.tensordot(x,y,(-1,0))

def ensure_iterable(x):
    """If input is not iterable enclose it as a list."""
    if np.isscalar(x): 
        return (x,)
    else: 
        return x

# def as_scalar_or_first_value(x):
    # """Return x if scalar, else return its first value."""
    # if np.isscalar(x): 
        # return(x)
    # else: 
        # return(x[0])

# def flip(x):
    # """ 
    # Flipud 1D arrays or 2D arrays where 2nd dimension is length 1.
    # Fliplr 2D arrays where first dimension is length 1.
    # """
    # if x.ndim == 1:
        # return np.flipud(x)
    # if x.ndim==2:
        # if shape(x)[0]==1: return np.fliplr(x)
        # if shape(x)[1]==1: return np.flipud(x)
    # raise Exception("Could not flip array, shape is wrong probably.")

def spline(
        xi,yi,x,s=0,order=3,
        check_bounds=True,
        set_out_of_bounds_to_zero=True,
        sort_data=True,
        ignore_nan_data=False,
):
    """Evaluate spline interpolation of (xi,yi) at x. Optional argument s
    is spline tension. Order is degree of spline. Silently defaults to 2 or 1
    if only 3 or 2 data points given.
    """
    order = min(order,len(xi)-1)
    xi,yi,x = np.array(xi,ndmin=1),np.array(yi,ndmin=1),np.array(x,ndmin=1)
    if ignore_nan_data:
        i = np.isnan(xi)|np.isnan(yi)
        if any(i):
            xi,yi = xi[~i],yi[~i]
    if sort_data:
        i = np.argsort(xi)
        xi,yi = xi[i],yi[i]
    if set_out_of_bounds_to_zero:
        i = (x>=xi[0])&(x<=xi[-1])
        y = np.zeros(x.shape)
        if any(i):
            y[i] = spline(xi,yi,x[i],s=s,order=order,set_out_of_bounds_to_zero=False,sort_data=False)
        return(y)
    if check_bounds:
        assert x[0]>=xi[0],'Splined lower limit outside data range: '+str(x[0])+' < '+str(xi[0])
        assert x[-1]<=xi[-1],'Splined upper limit outside data range: '+format(x[-1],'0.10g')+' > '+format(xi[-1],'0.10g')
    return interpolate.UnivariateSpline(xi,yi,k=order,s=s)(x)

# def splinef(xi,yi,s=0,order=3,sort_data=True):
    # """Return spline function for points (xi,yi). Will return order
    # or (less if fewer points). Sort data for convenience (takes time)."""
    # order = min(order,len(xi)-1)
    # xi,yi = np.array(xi),np.array(yi)
    # if sort_data:
        # i = np.argsort(xi)
        # xi,yi = xi[i],yi[i]
    # return interpolate.UnivariateSpline(xi,yi,k=order,s=s)

# def spline_with_smooth_ends(xi,yi,x):
    # """Evaluate spline defined by xi,yi at x. The first and last
    # intevals defined by xi are replaced with a 5th order polynomial
    # with C2 continuity with internal points and endpoint derivatives
    # set to zero.  Smooth end intervals by substituting a fifth order
    # polynomial that has 1-2nd derivatives zero at it's outer boundary
    # and matches the spline at the inner boundary.
    # Second output is the spline function without the edge smoothing.
    # """
    # from scipy import interpolate, linalg, integrate
    # xi = np.squeeze(xi);yi=np.squeeze(yi);x=np.squeeze(x)
    # ## generate spline function
    # spline_function=interpolate.UnivariateSpline(xi,yi,s=0)
    # ## evaluate spline function
    # y=spline_function(x)
    # ## change last interval to have smooth finish
    # ## internal point spline coeffs
    # a = spline_function.derivatives(xi[-2]) 
    # b = spline_function.derivatives(xi[-1]) 
    # ## range of last interval
    # i=np.argwhere((x>=xi[-2])&(x<=xi[-1])) 
    # ## new poly coordinate - range [0,1] -need to adjust derivs
    # L = (x[i[-1]]-x[i[0]])
    # ab = (x[i]-x[i[0]])/L
    # a[1] = a[1]*L
    # a[2] = a[2]*L**2
    # ## calc 5th order poly with desired properties
    # A = np.matrix([[1,1,1],[3,4,5],[6,12,20]])
    # z = np.matrix([[b[0]-a[0]-a[1]-a[2]],[-a[1]-2*a[2]],[-a[2]]])
    # v=linalg.inv(A)*z
    # ## new polynomial coefficients
    # anew = np.arange(0.,6.)
    # anew[0:3] = a[0:3]
    # anew[3:6] = np.squeeze(np.array(v))
    # ## substitute into potential array
    # y[i]=np.polyval(flip(anew),ab)
    # ##
    # ## change first interval to have smooth finish
    # a = spline_function.derivatives(xi[1]) 
    # b = spline_function.derivatives(xi[0]) 
    # ## range of last interval
    # i=np.argwhere((x>=xi[0])&(x<=xi[1])) 
    # ## new poly coordinate - range [0,1] -need to adjust derivs
    # L = (x[i[-1]]-x[i[0]])
    # ab = 1.-(x[i]-x[i[0]])/L
    # a[1] = -a[1]*L
    # a[2] = a[2]*L**2
    # ## calc 5th order poly with desired properties
    # A = np.matrix([[1,1,1],[3,4,5],[6,12,20]])
    # z = np.matrix([[b[0]-a[0]-a[1]-a[2]],[-a[1]-2*a[2]],[-a[2]]])
    # v=linalg.inv(A)*z
    # ## new polynomial coefficients
    # anew = np.arange(0.,6.)
    # anew[0:3] = a[0:3]
    # anew[3:6] = np.squeeze(np.array(v))
    # ## substitute into potential array
    # y[i]=np.polyval(flip(anew),ab)
    # ## 
    # return(y)

# def piecewise_linear_interpolation_and_extrapolation(xa,ya,x):
    # """Linearly interpolate and extrapolate) points xa and ya over domain x."""
    # y = np.zeros(x.shape,dtype=float)
    # ## interpolate
    # for (x0,x1,y0,y1) in zip(xa[0:-1],xa[1:],ya[0:-1],ya[1:]):
        # p = np.polyfit([x0,x1],[y0,y1],1)
        # i = (x>=x0)&(x<=x1)
        # if any(i): y[i] = np.polyval(p,x[i])
    # ## extrapolate
    # p = np.polyfit([xa[0],xa[1]],[ya[0],ya[1]],1)
    # i = x<=xa[0]
    # if any(i): y[i] = np.polyval(p,x[i])
    # p = np.polyfit([xa[-2],xa[-1]],[ya[-2],ya[-1]],1)
    # i = x>=xa[-1]
    # if any(i): y[i] = np.polyval(p,x[i])
    # return(y)

# def interpolate_to_mesh(x,y,z,xbins=100,ybins=100):
    # """Takes coordinates (x,y,z) and interpolates to a mesh divided
    # into bins (xbins,ybins) between min and max of x and y. Retrun
    # arrays (x,y,z) of shape (xbins,ybins)."""
    # xnew = np.linspace(min(x),max(x),xbins)
    # ynew = np.linspace(min(y),max(y),ybins)
    # f = interpolate.interp2d(x,y,z,kind='linear')
    # znew = f(xnew,ynew)
    # xnew,ynew = np.meshgrid(xnew,ynew)
    # return(xnew,ynew,znew)

# def array_to_txt_via_hdf5(path,*args,**kwargs):
    # """Loads ASCII text to an array, converting to hdf5 on the way
    # because it is faster.  Also deletes commented rows. MASSIVE
    # HACK."""
    # path = os.path.expanduser(path)
    # tmpfile=tempfile.NamedTemporaryFile()
    # array_to_hdf5(tmpfile.name,*args)
    # # command = 'cat "'+path+'" |sed "/^ *'+comments+'/d" | h5fromtxt '+tmpfile.name
    # command = 'h5totxt -s " " {hdf5file:s} > {txtfile:s}'.format(hdf5file=tmpfile.name,txtfile=path)
    # (status,output)=subprocess.getstatusoutput(command)
    # assert status==0,"Conversion from hdf5 failed:\n"+output

def array_to_hdf5(filename,*args,**kwargs):
    """Column stack arrays in args and save in an hdf5 file. In a
    single data set named 'data'. Overwrites existing files."""
    filename = expand_path(filename)
    ## kwargs.setdefault('compression',"gzip") # slow
    ## kwargs.setdefault('compression_opts',9) # slow
    if os.path.exists(filename):
        assert not os.path.isdir(filename),'Will not overwrite directory: '+filename
        os.unlink(filename)
    f = h5py.File(filename,'w')
    f.create_dataset('data',data=np.column_stack(args),**kwargs)
    f.close()

# def savetxt(filename,*args,**kwargs):
    # """Column-stacks arrays given as *args and saves them to
    # filename. 
    # A short cut for savetxt(filename,column_stack(args)). Kwargs passed to
    # np.savetxt.
    # If *args consists of one string, write this to file, ignore
    # everything else.
    # Also writes to 
    # """
    # ## This roundabout method is used in case this file is beind
    # ## watched by a graphing program which might fail if the writing
    # ## takes a long time. I.e. kst2, but probably not actually
    # ## important at all
    # # tmpfd,tmpfilename  = tempfile.mkstemp()
    # fid = open(expand_path(filename),'w')
    # if 'header' in kwargs:
        # fid.write(kwargs.pop('header')+'\n')
    # # np.savetxt(tmpfilename,np.column_stack(args),**kwargs)
    # np.savetxt(fid,np.column_stack(args),**kwargs)
    # # tmpfd.close()
    # # shutil.copyfile(tmpfilename,os.path.expanduser(filename))

# def savetxt_append(filename,*args,**kwargs):
    # """Open file filename. Append text made from column_stack of remaining args."""
    # f = open(filename,'a')
    # np.savetxt(f,np.column_stack(args),**kwargs)
    # f.close()
# savetxtAppend=savetxt_append
# save_append=savetxtAppend

# def solve_linear_least_squares_symbolic_equations(
        # system_of_equations,
        # plot_residual=False,
# ):
    # """Solve an overspecified system of linear equations. This is encoded pretry strictly e.g.,:
    # 1*x +  2*y =  4
    # 1*x +  3*y =  8
    # 0*x + -1*y = -3
    # Important separators are: newline, =, + , *.
    # """
    # ## get system of equations
    # equations = []
    # for t in system_of_equations.split('\n'):
        # t = t.split('#')[0]            # eliminate trailling comments
        # if len(t.strip())==0: continue # blank line
        # equations.append(t)
    # ## decode into terms
    # Aij,bi,variables = [],[],[]
    # for i,equation in enumerate(equations):
        # lhs,rhs = equation.split('=')
        # for term in lhs.split('+'):
            # coeff,var = term.split('*')
            # coeff = float(coeff)
            # var = var.strip()
            # if var not in variables: variables.append(var)
            # Aij.append((i,variables.index(var),coeff))
        # bi.append(float(rhs))
    # ## form matrices
    # A = np.zeros((len(equations),len(variables)))
    # for i,j,x in Aij: A[i,j] = x
    # b = np.array(bi)
    # ## solve. If homogeneous assume first variable==1
    # homogeneous = True if np.all(b==0) else False
    # if homogeneous:
        # b = -A[:,0].squeeze()
        # A = A[:,1:]
    # x = np.dot( np.linalg.inv(np.dot(np.transpose(A),A)),   np.dot(np.transpose(A),b))
    # if homogeneous:
        # x = np.concatenate(([1],x))
        # A = np.column_stack((-b,A))
        # b = np.zeros(len(equations))
    # if plot_residual:
        # fig = plt.gcf()
        # fig.clf()
        # ax = fig.gca()
        # ax.plot(b-np.dot(A,x),marker='o')
    # ## return solution dictionary
    # return({var:val for var,val in zip(variables,x)})

# def leastsq_model(y,f,v,monitor_rms=True,optimise=True,print_result=False,xtol=1e-12):
    # """
    # Optimises a least squares model. Higher level than leastsq.
    # Inputs:
    # y -- Array of experimental data to model.
    # f -- Function that generates model data. Input arguments are kwargs
         # corresponding to keys in dictionary v.
    # v -- Dictionary. Each key is a function argument of f which is
         # either a real number or an array of real numbers. Each key indexes
         # a subdictionary containing elements: 
             # pin  -- initial value of real number or array, 
             # grad -- gradient calculation step estimate, same dimension size as pin,
             # vary -- boolean array whether to vary or fix elements the same dimension 
                     # and size as pin.
   # monitor_rms -- print whenever rms is lowered during optimisation
   # optimise    -- set to False to return function using initial parameters
   # Outputs (yf,r,v):
   # yf -- model data
   # r  -- residual exp - mod data
   # v  -- modified v dictionary, elements now contain new elements:
           # p     -- fitted parameters
           # sigma -- estimated fit uncertainty.
    # """
    # ## prep v dictionary in some convenient ways
    # v = copy(v)            # make local
    # vkeys = list(v.keys())          # get here in case key order changes in dictionary
    # for key in vkeys:
        # if np.iterable(v[key]['pin']):
            # v[key]['pin'] = np.array(v[key]['pin'])
            # if not np.iterable(v[key]['grad']):
                # v[key]['grad'] = v[key]['grad']*np.ones(v[key]['pin'].shape,dtype=float)
            # if not np.iterable(v[key]['vary']):
                # v[key]['vary'] = v[key]['vary']*np.ones(v[key]['pin'].shape,dtype=bool)
            # v[key]['vary'] = np.array(v[key]['vary'])
            # v[key]['grad'] = np.array(v[key]['grad'])
    # ## encode varied parameters and their gradient calculation step size
    # p,grad = [],[]
    # for key in vkeys:
        # if np.iterable(v[key]['pin']):
            # p.extend(v[key]['pin'][v[key]['vary']])
            # grad.extend(v[key]['grad'][v[key]['vary']])
        # elif v[key]['vary']:
                # p.append(v[key]['pin'])
                # grad.append(v[key]['grad'])
    # lowest_rms_thus_far = [np.inf]
    # def calc_residual(p):
        # residual = y-calc_function(p)
        # if monitor_rms:
            # if rms(residual) < lowest_rms_thus_far[0]:
                # lowest_rms_thus_far[0] = rms(residual)
                # print('RMS:',lowest_rms_thus_far[0])
        # return residual
    # def calc_function(p):
        # ## call function, return value. Need to build kwargs for
        # ## function.
        # decode_p(p)
        # kwargs = {key:v[key]['p'] for key in vkeys}
        # return f(**kwargs)
    # def decode_p(p):
        # ## loop through variables, extracts varied parametres
        # ## from p and shortens p accordingly
        # p = list(p)
        # for key in vkeys:
            # v[key]['p'] = copy(v[key]['pin'])
            # ## if a list
            # if np.iterable(v[key]['pin']):
                # v[key]['p'][v[key]['vary']] = p[:sum(v[key]['vary'])]
                # p = p[sum(v[key]['vary']):]
            # ## if a varied float
            # elif v[key]['vary']:
                # v[key]['p'] = p.pop(0)
            # ## if not varied
            # else:
                # v[key]['p'] = v[key]['pin']
            # ## ensure abslute value if required
            # try:
                # if v[key]['strictly_positive']:
                    # v[key]['p'] = np.abs(v[key]['p'])
            # except KeyError:    # implies strictly_positive not defined
                # pass
    # def decode_sigma(sigma):
        # ## similar to decode_p
        # sigma = np.abs(sigma)   # make all positive
        # for key in vkeys:
            # if np.iterable(v[key]['pin']):
                # v[key]['sigma'] = np.nan*np.ones(v[key]['pin'].shape)
                # v[key]['sigma'][v[key]['vary']] = sigma[:sum(v[key]['vary'])]
                # sigma = sigma[sum(v[key]['vary']):]
            # elif v[key]['vary']:
                # v[key]['sigma'] = sigma[0]
                # sigma = sigma[1:]
            # else:
                # v[key]['sigma'] = np.nan
    # ## actual least squares fit
    # if optimise:
        # p,sigma = leastsq(calc_residual,p,grad,print_error_mesg=True,error_only=False,xtol=xtol)
    # else:
        # sigma = np.zeros(len(p))
    # decode_p(p)
    # decode_sigma(sigma)
    # r = calc_residual(p)
    # yf = calc_function(p)
    # ## print summary of fitted parameters
    # if print_result:
        # print(('\n'.join([
            # '{:20s} = {:20s} +- {:8s}'.format(key,str(val['p']),str(val['sigma']))
                                              # for (key,val) in list(v.items())])))
    # return(yf,r,v)

# def weighted_mean(
        # x,
        # dx,
        # error_on_nan_zero=True, # otherwise edit out affected values
# ):
    # """Calculate weighted mean and its variance. If
    # error_on_nan_zero=False then remove data with NaN x or dx, or 0
    # dx."""
    # # ## if ufloat, separate into nominal and error parts -- return as ufloat
    # # if isinstance(x[0],uncertainties.AffineScalarFunc):
        # # (mean,variance) = weighted_mean(*decompose_ufloat_array(x))
        # # return(ufloat(mean,variance))
    # x,dx = np.array(x,dtype=float),np.array(dx,dtype=float) 
    # ## trim data to non nan if these are to be neglected, or raise an error
    # i = np.isnan(x)
    # if np.any(i):
        # if error_on_nan_zero:
            # raise Exception('NaN values present.') 
        # else:
            # x,dx = x[~i],dx[~i]
    # i = np.isnan(dx)
    # if np.any(i):
        # if error_on_nan_zero:
            # raise Exception('NaN errors present.') 
        # else:
            # x,dx = x[~i],dx[~i]
    # i = (dx==0)
    # if np.any(i):
        # if error_on_nan_zero:
            # raise Exception('NaN errors present.') 
        # else:
            # x,dx = x[~i],dx[~i]
    # ## make weighed mean
    # weights = dx**-2           # assuming dx is variance of normal pdf
    # weights = weights/sum(weights) # normalise
    # mean = np.sum(x*weights)
    # variance = np.sqrt(np.sum((dx*weights)**2))
    # return (mean,variance)

# def average(x,dx,axis=1,returned=True,warningsOn=True):
    # """ 
    # Based on numpy.average.\n
    # Return weighted mean along a particular. Nan data x, or nan or zero
    # errors dx are ignored.\n
    # May not word for >2D.
    # """
    # ## avoid overwriting originals - slows things down probably
    # x = np.array(x)
    # dx = np.array(dx)
    # ## calculate weights from standard error
    # weights = dx.astype(float)**-2.
    # ## set weight of zero error or nan data to zero
    # i = np.isnan(dx)|(dx==0)|np.isnan(x)
    # x[i] = 0.
    # weights[i] = 0.
    # ## if all data invalid avoid a ZeroDivisionError
    # i = weights.sum(axis)==0
    # if axis==0:
        # weights[:,i] = 1.
    # elif axis==1:
        # weights[i,:] = 1.
    # ## the actual averaging
    # m,tmp = np.average(x,weights=weights,returned=returned,axis=axis)
    # s = np.sqrt(tmp**-1)
    # ## correct for the avoided ZeroDivisionError
    # m[i] = np.nan
    # s[i] = np.nan
    # return m,s

# def common_weighted_mean(*data):
    # """Take a weighted mean of all arguments which are arrays of form
    # [x,y,dy].  Those arguments for which certain values of x are
    # missing are not included in mean of that value. Alternatively you
    # could provide the arrays already concatenated into one big array
    # with some repeating x values.\n\nZero or NaN error data is removed
    # and forgotten. """
    # ## one 2D array (three colummns (x,y,dy)
    # if len(data)==1:
        # x,y,dy = data[0]
    # ## join all input data given as three inputs, x,y,dy
    # else:        
        # x=np.concatenate([d[0] for d in data])
        # y=np.concatenate([d[1] for d in data])
        # dy=np.concatenate([d[2] for d in data])
    # ## remove NaN x or y data
    # i = ~(np.isnan(y)|np.isnan(x))
    # x,y,dy = x[i],y[i],dy[i]
    # ## zero or NaN errors are increased so as to remove this freom the
    # ## weighting - POTENTIAL BUG
    # i = (dy==0)|np.isnan(dy)
    # ## if all are 0 or nan - set to some random value, otherwise make
    # ## sufficiently large
    # if all(i): dyDefault = 1
    # else:      dyDefault = dy[~i].max()*1e5
    # dy[i] = dyDefault
    # ## prepare arrays for output data
    # xout=np.unique(x)
    # yout=np.zeros(xout.shape)
    # dyout=np.zeros(xout.shape)
    # ## take various means
    # for i in range(len(xout)):
        # ii = np.argwhere(x==xout[i])
        # (yout[i],dyout[i])=weighted_mean(y[ii],dy[ii])
    # ## return zero error
    # dyout[dyout>dyDefault*1e-2] = np.nan
    # return(xout,yout,dyout)

# def mean_ignore_missing(x):
    # """Calc unweighted mean of columns of a 2D array. Any nan values
    # are ignored."""
    # return np.array([float(t[~np.isnan(t)].mean()) for t in x])

# def equal_or_none(x,y):
    # if x is None and y is None:
        # return(True)
    # elif x==y:
        # return(True)
    # else:
        # return(False)

# def nanequal(x,y):
    # """Return true if x and y are equal or both NaN. If they are vector,
    # do this elementwise."""
    # if np.isscalar(x) and np.isscalar(y):
        # if x==y:
            # return(True)
        # else:
            # try:
                # if np.isnan(x) and np.isnan(y):
                    # return(True)
            # except TypeError:
                # pass
            # return(False)
    # elif not np.isscalar(x) and not np.isscalar(y):
        # if x.dtype.kind!='f' or x.dtype.kind!='f':
            # return(x==y)
        # else:
            # return((x==y)|(np.isnan(x)&np.isnan(y)))
    # else:
        # raise Exception('Not implemented')

# def nancumsum(x,*args,**kwargs):
    # """Calculate cumsum, first set nans to zero."""
    # x = np.asarray(x)
    # x[np.isnan(x)] = 0.
    # return np.cumsum(x,*args,**kwargs)

def cumtrapz(y,
             x=None,               # if None assume unit xstep
             reverse=False,        # or backwards
):
    """Cumulative integral, with first point equal to zero, same length as
    input."""
    if reverse:
        y = y[::-1]
        if x is not None: 
            x = x[::-1]
    yintegrated = np.concatenate(([0],integrate.cumtrapz(y,x)))
    if reverse:
        yintegrated = -yintegrated[::-1] # minus sign to account for change in size of dx when going backwards, which is probably not intended
    return yintegrated 

# def cumtrapz_reverse(y,x):
    # """Return a cumulative integral ∫y(x) dx from high to low limit."""
    # x,i = np.unique(x,return_index=True)
    # y = y[i]
    # return(integrate.cumtrapz(y[-1::-1],-x[-1::-1])[-1::-1])


# def power_spectrum(x,y,make_plot=False,fit_peaks=False,fit_radius=1,**find_peaks_kwargs):
    # """Return (frequency,power) after spectral analysis of y. Must be on a
    # uniform x grid."""
    # dx = np.diff(x)
    # assert np.abs(dx.max()/dx.min()-1)<1e-5,'Uniform grid required.'
    # dx = dx[0]
    # F = np.fft.fft(y)          # Fourier transform
    # F = np.real(F*np.conj(F))         # power spectrum
    # F = F[:int((len(F-1))/2+1)] # keep up to Nyquist frequency
    # f = np.linspace(0,1/dx/2.,len(F)+1)[:-1] # frequency scale
    # if make_plot:
        # ax = plt.gca()
        # ax.plot(f,F,color=newcolor(0))
        # ax.set_xlabel('f')
        # ax.set_ylabel('F')
        # ax.set_yscale('log')
    # if not fit_peaks:
        # return(f,F)
    # else:
        # import spectra
        # resonances = spectra.data_structures.Dynamic_Recarray()
        # for i in find_peaks(F,f,**find_peaks_kwargs):
            # ibeg = max(0,i-fit_radius)
            # iend = min(i+fit_radius+1,len(f))
            # ft,Ft = f[ibeg:iend],F[ibeg:iend]
            # p,yf = spectra.lineshapes.fit_lorentzian(ft,Ft,x0=f[i],S=F[i],Γ=dx)
            # resonances.append(f0=p['x0'], λ0=1/p['x0'], S=p['S'],Γ=p['Γ'])
            # if make_plot:
                # ax.plot(ft,yf,color=newcolor(1))
        # return(f,F,resonances)

# def find_in_recarray(recarray,**key_value):
    # """Find elements of recarray which match (key,value) pairs."""
    # # return(np.prod([recarray[key]==value for (key,value) in key_value.items()],axis=0))
    # return(np.prod([recarray[key]==value for (key,value) in key_value.items()],axis=0,dtype=bool))

# def unique(x):
#     """Returns unique elements."""
#     if not np.iterable(x): return([x])
#     return(list(set(x)))

def unique(x,preserve_ordering=False):
    """Returns unique elements. preserve_ordering is likely slower"""
    if preserve_ordering:
        x = list(x)
        for t in copy(x):
            while x.count(t)>1:
                x.remove(t)
        return(x)
    else:
        return(list(set(x)))

# def argunique(x):
    # """Find indices of unique elements of x. Picks first such
    # element. Does not sort."""
    # y = np.unique(x)
    # x = list(x)
    # return np.array([x.index(yi) for yi in y])

# def unique_iterable(x):
    # """Find unique elements of x assuming each element is an iterable,
    # returns as a tuple of tuples. E.g., [[1,2],[1,2,],[1,1,]]] returns
    # ((1,2),(1,1,))."""
    # return(tuple(           # return as tuple
            # set(            # do the uniquifying by using a set object
                # tuple(tuple(t) for t in x)))) # convert 2D elements into immutable tuple of tuples

def unique_combinations(*args):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    return(set(zip(*args)))

# def unique_array_combinations(*arrs,return_mask=False):
#     """All are iterables of the same length. Finds row-wise combinations of
#     args that are unique. Elements of args must be hashable."""
#     ra = np.rec.fromarrays(arrs)
#     unique_values = np.unique(ra)
#     if return_mask:
#         return([(t,ra==t) for t in unique_values])
#     else:
#         return(unique_values)

def unique_combinations_masks(*arrs):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    # ra = np.rec.fromarrays(arrs)
    ra = np.rec.fromarrays(arrs)
    unique_values = np.unique(ra)
    return [(t,ra==t) for t in unique_values]

def unique_combinations_first_index(*arrs):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    ra = np.rec.fromarrays(arrs)
    unique_values,first_index = np.unique(ra,return_index=True)
    return unique_values,first_index

def sortall(x,*others,reverse=False):
    """Sort x and return sorted. Also return others sorted according
    to x."""
    x = np.asarray(x)
    i = np.argsort(x)
    retval = [x[i]]
    for y in others:
        retval.append(np.asarray(y)[i])
    if reverse:
        retval = [t[::-1] for t in retval]
    return retval

# def sortByFirst(*args,**kwargs):
    # """Returns iterable args all sorted by the elements of the first
    # arg. Must be same length of course.  Optional key-word argument
    # outputType=type will cast outputs into type. Otherwise tuples are
    # returned regardless of the input types."""

    # outputs = list(zip(*(sorted(zip(*args),key=lambda x:x[0]))))
    # if 'outputType' in kwargs:
        # outputs = (kwargs['outputType'](output) for output in outputs)
    # return outputs

# def sortByFirstInPlace(*args):
    # """Sorts all arrays in args in place, according to sorting of
    # first arg. Returns nothing. Must be arrays or something else that
    # supports fancy indexing."""
    # i = np.argsort(args[0])
    # for j in range(len(args)):
        # args[j][:] = args[j][i]

# def vectorise_function(fcn,*args,**kwargs):
    # """Run function multiple times if any argument is vector, possibly
    # broadcasting the rest. If all arguments are scalar return None."""
    # ## get length of data or if needs to be vectorise
    # length = None
    # for arg in list(args)+list(kwargs.values()):
        # if not np.isscalar(arg) and arg is not None:
            # if length is None:
                # length = len(arg)
            # else:
                # assert len(arg)==length,'Cannot vectorise, argument lengths mismatch.'
    # ## indicates no vectorisation needed
    # if length is None:
        # return(None)
    # ## extend short data
    # if length is not None:
        # args = list(args)
        # for i,arg in enumerate(args):
            # if np.isccalar(arg) or arg is None:
                # args[i] = [arg for t in range(length)]
        # for key,val in kwargs.items():
            # if np.isscalar(val) or val is None:
                # kwargs[key] = [val for t in range(length)]
    # ## run and return list of results
    # return([fcn(
        # *[arg[i] for arg in args],
        # **{key:val[i] for key,val in kwargs.items()})
            # for i in range(length)])

# def vectorise_dicts(*dicts):
    # """Input arguments are multiple dicts with the keys. Output argument
    # is one dict with the same keys and all input values in lists."""
    # if len(dicts)==0:
        # return({})
    # retval = {}
    # for key in dicts[0]:
        # retval[key] = [t[key] for t in dicts]
    # return(retval)

# def common(x,y,use_hash=False):
    # """Return indices of common elements in x and y listed in the order
    # they appear in x. Raises exception if repeating multiple matches
    # found."""
    # if not use_hash:
        # ix,iy = [],[]
        # for ixi,xi in enumerate(x):
            # iyi = find([xi==t for t in y])
            # if len(iyi)==1: 
                # ix.append(ixi)
                # iy.append(iyi[0])
            # elif len(iyi)==0:
                # continue
            # else:
                # raise Exception('Repeated value in y for: '+repr(xi))
        # if len(np.unique(iy))!=len(iy):
            # raise Exception('Repeated value in x for something.')
        # return(np.array(ix),np.array(iy))
    # else:
        # xhash = np.array([hash(t) for t in x])
        # yhash = np.array([hash(t) for t in y])
        # ## get sorted hashes, checking for uniqueness
        # xhash,ixhash = np.unique(xhash,return_index=True)
        # assert len(xhash)==len(x),f'Non-unique values in x.'
        # yhash,iyhash = np.unique(yhash,return_index=True)
        # assert len(yhash)==len(y),f'Non-unique values in y.'
        # ## use np.searchsorted to find one set of hashes in the other
        # iy = np.arange(len(yhash))
        # ix = np.searchsorted(xhash,yhash)
        # ## remove y beyond max of x
        # i = ix<len(xhash)
        # ix,iy = ix[i],iy[i]
        # ## requires removing hashes that have no search sorted partner
        # i = yhash[iy]==xhash[ix]
        # ix,iy = ix[i],iy[i]
        # ## undo the effect of the sorting above
        # ix,iy = ixhash[ix],iyhash[iy]
        # ## sort by index of first array -- otherwise sorting seems to be arbitrary
        # i = np.argsort(ix)
        # ix,iy = ix[i],iy[i]
        # return(ix,iy)


# def get_common(x,y):
    # """Return common subsets of x and y."""
    # i,j = common(x,y)
    # return(x[i],y[i])

# def sort_to_match(x,y):
    # """Return a copy of x sorted into the smae dissaray as y. That is
    # the same reordering will sort both y and the return value."""
    # x = np.array(x,ndmin=1)
    # y = np.array(y,ndmin=1)
    # return x[np.argsort(x)[np.argsort(np.argsort(y))]]

# def argsort_to_match(x,y):
    # """Returns indices which will sort x to give the same dissaray as y."""
    # x = np.array(x,ndmin=1)
    # y = np.array(y,ndmin=1)
    # return np.argsort(x)[np.argsort(np.argsort(y))]

def isin(x,y):
    """Return arrays of booleans same size as x, True for all those
    elements that exist in y."""
    return np.array([i in y for i in x],dtype=bool,ndmin=1)

# def find_overlap(x,y):
    # """Return boolean arrays (i,j) indicating (x,y) that cover an
    # overlapping region. Outermost points are taken from x, i.e., x[i]
    # encompasses y[j]. Assumes x and y are ordered. """
    # i = (x>=y[0])&(x<=y[-1])
    # j = (y>=x[0])&(y<=x[-1])
    # if any(i):                  # if not overlap then don't proceed with this
        # if not i[0]  and  x[i][0]!=y[j][0] : i[my.find(i)[0] -1]  = True
        # if not i[-1] and x[i][-1]!=y[j][-1]: i[my.find(i)[-1]+1] = True
    # return(i,j)

# def argminabs(x): return(np.argmin(np.abs(x)))

# def argmaxabs(x): return(np.argmax(np.abs(x)))

def find(x):
    """Convert boolean array to array of True indices."""
    return(np.where(x)[0])

def find_unique(x):
    """Convert boolean array to array of True indices."""
    i = find(x)
    assert len(i)>0,'No match found'
    assert len(i)<2,'Multiple matches found'
    return i[0]

def findin_unique(x,y):
    """Find one only match of x in y. Else raise an error."""
    i = findin(x,y)
    assert len(i)!=1,'No match found'
    assert len(i)<2,'Multiple matches found'
    return i[0]

def findin(x,y):
    """Find x in y and return a list of the matching y indices. If an
    element of x cannot be found in y, or if multiple found, an error
    is raised."""
    x = ensure_iterable(x)
    y = ensure_iterable(y)
    i = np.zeros(len(x),dtype='int')
    for j,xj in enumerate(x):
        ii = find(y==xj)
        if len(ii) != 1:
            if len(ii) == 0:
                raise Exception(f'Element not found in y: {repr(xj)}')
            if len(ii) > 1:
                raise Exception(f'Element non-unique in y: {repr(xj)}')
        i[j] = ii[0]
    return i

def find_nearest(x,y):
    """Find nearest match of x in y and return a list of the nearest-match
    y indices. If multiple x match the same y an error is raised."""
    y = np.asarray(y)
    i = np.array([np.argmin(abs(y-xi)) for xi in ensure_iterable(x)])
    assert len(i) == len(np.unique(i)),'Multiple values in x nearest-match the same value in y.'
    return i

def findin_numeric(x,y,tolerance=1e-10):
    """Use compiled code to findin with numeric data only."""
    ix,iy = np.argsort(x),np.argsort(y) # sort data
    tx,ty = np.asarray(x,dtype=float)[ix],np.asarray(y,dtype=float)[iy]
    i = np.full(tx.shape,-1,dtype=int)
    myf.findin_sorted(tx,ty,tolerance,i)
    if i[0]==-1: raise Exception('Some value of x not found in y within tolerance.') # hack of an error code
    i = i[np.argsort(ix)]                 # undo x sort
    i = np.argsort(iy)[i]                 # undo y sort
    return(i)

# def inv(x):
    # """Invert a symmetric matrix."""
    # x = np.array(x,order='F',dtype=float)
    # y = copy(x)
    # myf.inv(x,y)
    # return(y)

# def integrate_trapz_uniform(y,dx=1.):
    # """Trapezium integration on a uniform grid. If x is 2D then integrate
    # along the first axis. Seems to be about 3x faster than
    # scipy.integrate.trapz. Speed up if 2D array is order='F'."""
    # if y.ndim==1:
        # yint = np.zeros(1,dtype=float)
        # myf.integrate_trapz_uniform_grid(y.astype(float),dx,yint)
        # return(float(yint))
    # elif y.ndim==2:
        # yint = np.zeros(y.shape[1],dtype=float)
        # myf.integrate_trapz_uniform_grid_two_dimensional(np.asfortranarray(y),dx,yint)
        # return(yint)
    # else:
        # raise Exception('Only implemented for 1D and 2D arrays.')

def find_blocks(b,error_on_empty_block=True):
    """Find boolean index arrays that divide boolean array b into
    independent True blocks."""
    i = np.full(len(b),True)    # not yet in a block
    blocks = []                 # final list of blocks
    while np.any(i):            # until all rows in a block
        ## start new block with first remaining row
        block = b[find(i)[0],:]
        if np.sum(block) == 0:
            ## a block is all zero -- connected to nothing
            if error_on_empty_block:
                ## raise an error
                raise Exception('empty block found')
            else:
                ## return as one connected block
                blocks.append(block|True)
                break
        ## add coupled elements to this block until no new ones found
        while np.any((t:=np.any(b[block,:],0)) & ~block):
            block |= t
            t = np.any(b[block,:],0)
        ## record found block and blocked rows
        blocks.append(block)
        i &= ~block
    return blocks

def inrange(x,xbeg,xend=None):
    """Return arrays of booleans same size as x, True for all those
    elements that xbeg<=x<=xend.\n\nIf xend is none and xbeg is an
    array, find elements of x in the range of y. """
    if xend is None:
        return (x>=np.min(xbeg))&(x<=np.max(xbeg))
    else:
        return (x>=xbeg)&(x<=xend)

def limit_to_range(beg,end,x,*other_arrays):
    """Limit x to range between beg and end (using np.searchsorted, must
    be sorted.  Also index other_arrays and return all arrays."""
    i = np.searchsorted(x,(beg,end))
    return tuple([t[i[0]:i[1]] for t in [x]+list(other_arrays)])

# def common_range(x,y):
    # """Return min max of values in both x and y (may not be actual
    # values in x and y)."""
    # return(max(min(x),min(y)),min(max(x),max(y)))

# def in_common_range(x,y):
    # """Return indices of arrays x and y that line inside their common range."""
    # t0,t1 = common_range(x,y)
    # return(inrange(x,t0,t1),inrange(y,t0,t1))

# def find_in_range_sorted(x,x0,x1):
    # """Find return i0,i1, indices of x bounding x0 and x1. Assumes
    # x1>x0 and x is sorted., NOT COMLETELY ACCURATE -- MIGHT GET
    # EDGTEST SLIGHTLY WRONGE"""
    # ## prepare fortran inputs
    # x = np.array(x,dtype=float,ndmin=1)
    # n = np.array(len(x),dtype=int)
    # x0 = np.array(x0,dtype=float)
    # x1 = np.array(x1,dtype=float)
    # i0 = np.array(0,dtype=int)
    # i1 = np.array(1,dtype=int)
    # ## call compiled coe
    # myf.find_in_range_sorted(x,x0,x1,i0,i1)
    # return(int(i0),int(i1))

# # def find_regexp(regexp,x):
# #     """Returns boolean array of elements of x whether or not they match
# #     regexp."""
# #     return np.array([bool(re.match(regexp,t)) for t in x])


def match_regexp(regexp,x):
    """Returns boolean array of elements of x whether or not they match
    regexp."""
    return np.array([bool(re.match(regexp,t)) for t in x])

def find_regexp(regexp,x):
    return find(match_regexp(regexp,x))

# def meshgrid(*args):
    # """ meshgrid(arr1,arr2,arr3,...)
    # Expand 1D arrays arr1,... into multiple multidimensional arrays
    # that loop over the values of arr1, ....  Similar to matlab/octave
    # meshgrid. Sorry about the poor documentation, an example:
    # meshgrid(np.array([1,2]),np.array([3,4]),)
    # returns
    # (array([[1, 1],[2, 2]]), array([[3, 4],[3, 4]]))
    # """
    # ## a sufficiently confusing bit of code its probably easier to
    # ## rewrite than figure out how it works
    # n = len(args)
    # assert n>=2, 'requires at least two arrays'
    # l = [len(arg) for arg in args]
    # ret = []
    # for i in range(n):
        # x = np.array(args[i])
        # for j in range(n):
            # if i==j: continue
            # x = np.expand_dims(x,j).repeat(l[j],j)
        # ret.append(x)
    # return tuple(ret)

# def sum_with_nans_as_zero(args,**kwargs):
    # """Add arrays in args, as if nan values are actually zero. If
# revert_to_nans is True, then turn all zeros back to nans after
# summation."""
    # kwargs.setdefault('revert_to_nans',False)
    # is_scalar = np.isscalar(args[0])
    # args = [np.array(arg,ndmin=1) for arg in args]
    # retval = np.zeros(args[0].shape)
    # for arg in args:
        # i = ~np.isnan(arg)
        # retval[i] += arg[i]
    # if is_scalar: 
        # retval = float(retval)
    # if kwargs['revert_to_nans']:
        # retval[retval==0.] = np.nan
    # return retval

# def mystack(*args): 
    # return(column_stack(args))

# def flatten(*args):
    # """
    # All args are flattened into 1D and concatentated into one 1D
    # array. Wont flatten strings.
    # """
    # return(np.array([x for x in mpl.cbook.flatten(args)]))

# def unpack(*args):
    # """Flatten all args and join together into tuple."""
    # return tuple(x for x in matplotlib.cbook.flatten(args))

def normal_distribution(x,μ=0.,σ=1):
    """Normal distribution."""
    return(1/np.sqrt(constants.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2)))

def fit_normal_distribution(x,bins=None,figure=None):
    """Fit a normal distribution in log space to a polynomial."""
    if bins is None:
        ## estimate good amount of binning
        bins = max(10,int(len(x)/200))
    count,edges = np.histogram(x,bins)
    centres = (edges[:-1]+edges[1:])/2
    logcount = np.log(count)
    i = ~np.isinf(logcount)
    p = np.polyfit(centres[i],logcount[i],2)
    σ = np.sqrt(-1/p[0]/2)
    μ = p[1]*σ**2
    if figure is not None:
        ## make a plot
        figure.clf()
        ax = plotting.subplot(0,fig=figure)
        ax.plot(centres,logcount)
        ax.plot(centres,np.polyval(p,centres))
        ax = plotting.subplot(1,fig=figure)
        ax.set_title("Fit in log space")
        ax.plot(centres,count)
        nd = normal_distribution(centres,μ,σ)
        ax.plot(centres,nd/np.mean(nd)*np.mean(count))
        ax.set_title("Fit in linear space")
    return μ,σ

def gaussian(x,fwhm=1.,mean=0.,norm='area'):
    """
    y = gaussian(x[,fwhm,mean]). 
    Produces a gaussian with area normalised to one.
    If norm='peak' peak is equal to 1.
    If norm='sum' sums to 1.
    Default fwhm = 1. Default mean = 0.
    """
    fwhm,mean = float(fwhm),float(mean)
    if norm=='area':
        ## return 1/fwhm*np.sqrt(4*np.log(2)/constants.pi)*np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2);
        return 1/fwhm*0.9394372786996513*np.exp(-(x-mean)**2*2.772588722239781/fwhm**2);
    elif norm=='peak':
        return np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2)
    elif norm=='sum':
        t = np.exp(-(x-mean)**2*4*np.log(2)/fwhm**2)
        return t/t.sum()
    else:
        raise Exception('normalisation method '+norm+' not known')

def convolve_with_gaussian(x,y,fwhm,fwhms_to_include=10,regrid_if_necessary=False):
    """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    convolution after a certain number of fwhms. x must be on a
    regular grid."""
    dx = (x[-1]-x[0])/(len(x)-1)
    ## check on regular grid, if not then spline to a new one
    t = np.diff(x)
    regridded = False
    if (t.max()-t.min())>dx/100.:
        if regrid_if_necessary:
            regridded = True
            x_original = x
            xstep = t.min()
            x = np.linspace(x[0],x[-1],(x[-1]-x[0])/xstep)
            y = spline(x_original,y,x)
        else:
            raise Exception("Data not on a regular x grid")
    ## add padding to data
    xpad = np.arange(dx,fwhms_to_include*fwhm,dx)
    x = np.concatenate((x[0]-xpad[-1::-1],x,x[-1]+xpad))
    y = np.concatenate((np.full(xpad.shape,y[0]),y,np.full(xpad.shape,y[-1])))
    ## convolve
    gx = np.arange(-fwhms_to_include*fwhm,fwhms_to_include*fwhm,dx)
    if (len(gx)%2) == 0:        # is even
        gx = gx[0:-1]
    gx = gx-gx.mean()
    gy = gaussian(gx,fwhm=fwhm,mean=0.,norm='sum')
    assert len(y)>len(gy), 'Data vector is shorter than convolving function.'
    y = np.convolve(y,gy,mode='same')
    ## remove padding
    y = y[len(xpad):-len(xpad)]
    x = x[len(xpad):-len(xpad)]
    ## return to original grid if regridded
    if regridded:
        y = spline(x,y,x_original)
    return y

# def convolve_with_gaussian_to_grid(x,y,xout,fwhm):
    # """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    # convolution after a certain number of fwhms. x must be on a
    # regular grid."""
    # yout = np.zeros(xout.shape)
    # myf.convolve_with_gaussian(x,y,xout,yout,float(fwhm))
    # return(yout)

# def convolve_with_lorentzian(x,y,fwhm,fwhms_to_include=50,regrid_if_necessary=False):
    # """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    # convolution after a certain number of fwhms. x must be on a
    # regular grid."""
    # dx = (x[-1]-x[0])/(len(x)-1)
    # ## check on regular grid, if not then spline to a new one
    # t = np.diff(x)
    # regridded = False
    # if (t.max()-t.min())>dx/1000.:
        # if regrid_if_necessary:
            # regridded = True
            # x_original = x
            # xstep = t.min()
            # # x = np.arange(x[0],x[-1]+xstep/2.,xstep)
            # x = np.linspace(x[0],x[-1],(x[-1]-x[0])/xstep)
            # y = spline(x_original,y,x)
        # else:
            # raise Exception("Data not on a regular x grid")
    # ## add padding to data
    # xpad = np.arange(dx,fwhms_to_include*fwhm,dx)
    # x = np.concatenate((x[0]-xpad[-1::-1],x,x[-1]+xpad))
    # y = np.concatenate((np.full(xpad.shape,y[0]),y,np.full(xpad.shape,y[-1])))
    # ## convolve
    # gx = np.arange(-fwhms_to_include*fwhm,fwhms_to_include*fwhm,dx)
    # if iseven(len(gx)): gx = gx[0:-1]
    # gx = gx-gx.mean()
    # gy = lorentzian(gx,k0=0,S=1,Gamma=fwhm,norm='sum')
    # assert len(y)>len(gy), 'Data vector is shorter than convolving function.'
    # y = np.convolve(y,gy,mode='same')
    # ## remove padding
    # y = y[len(xpad):-len(xpad)]
    # x = x[len(xpad):-len(xpad)]
    # ## return to original grid if regridded
    # if regridded:
        # y = spline(x,y,x_original)
    # return y

# def convolve_with_gaussian_with_prebinning(
        # x,y,
        # gaussian_FWHM,
        # bins_per_gaussian_FWHM=10,
# ):
    # """Convolve data x,y with a Gaussian of full-width half-maximum
    # gaussian_FWHM. To speed up the process first bin data by an amount
    # dictated by bins_per_gaussian_FWHM. Then return a new pair (x,y)
    # on the binned grid and convolved by the Gaussian."""
    # ## calculate bin centres
    # binwidth = gaussian_FWHM/bins_per_gaussian_FWHM
    # xbin = np.arange(x.min()+binwidth/2.,x.max(),binwidth)
    # ## for each bin get the mean value (using trapezium integration)
    # ## of input function over that range
    # ybin = np.zeros(xbin.shape)
    # for i,xi in enumerate(xbin):
        # j = (x>(xi-binwidth/2))&(x<=(xi+binwidth/2))
        # try:
            # ybin[i] = integrate.trapz(y[j],x[j])/(x[j].max()-x[j].min())
            # if np.isnan(ybin[i]): ybin[i] = 0 # probably should do this more intelligently -- or raise an exception
        # except ValueError:      # catch empty j -- much faster than prechecking
            # ybin[i] = 0
    # ## convolve with gaussian
    # ybin = convolve_with_gaussian(xbin,ybin,gaussian_FWHM)
    # return(xbin,ybin)

# def convolve_with_sinc(x,y,fwhm,fwhms_to_include=10,):
    # """Convolve function y(x) with a sinc of FWHM fwhm. Truncate
    # convolution after a certain number of fwhms. x must be on a
    # regular grid."""
    # ## add padding to data
    # dx = (x[-1]-x[0])/(len(x)-1)
    # xpad = np.arange(dx,fwhms_to_include*fwhm,dx)
    # x = np.concatenate((x[0]-xpad[-1::-1],x,x[-1]+xpad))
    # y = np.concatenate((np.full(xpad.shape,y[0]),y,np.full(xpad.shape,y[-1])))
    # ## get sinc to convolve with
    # gx = np.arange(-fwhms_to_include*fwhm,fwhms_to_include*fwhm,dx)
    # if iseven(len(gx)): gx = gx[0:-1]
    # gx = gx-gx.mean()
    # gy = sinc(gx,fwhm=fwhm,mean=0.,norm='sum')
    # ## convolve
    # y = np.convolve(y,gy,mode='same')
    # ## remove padding
    # y = y[len(xpad):-len(xpad)]
    # x = x[len(xpad):-len(xpad)]
    # return(y)

# def lorentzian(k,k0=0,S=1,Gamma=1,norm='area'):
    # """Lorentzian profile.
    # Inputs:
        # k     energy
        # k0    resonance central energy
        # S     integrated cross-section
        # Gamma full-width half-maximum
    # """
    # if norm=='area':
        # return S*Gamma/2./3.1415926535897931/((k-k0)**2+Gamma**2/4.)
    # elif norm=='sum':
        # t = S*Gamma/2./3.1415926535897931/((k-k0)**2+Gamma**2/4.)
        # t = t/t.sum()
        # return(t)
    # else:
        # raise Exception('normalisation other than area not implemented, easy to do though.')

# def voigt_fwhm(gaussian_width,lorentzian_width):
    # """Approximate calculationg from wikipedia"""
    # return 0.5346*lorentzian_width + np.sqrt(0.2166*lorentzian_width**2 + gaussian_width**2)

# def _voigt_cachetools_hashkey(*args,**kwargs):
    # """A bespoke cachetoosl memoizing cache key. Uses arary beg,end,len
    # instead of array itself as a hash."""
    # args = list(args)
    # for i,arg in enumerate(args): # convert arrays into three value hashes
        # if isinstance(arg,np.ndarray):
            # if len(arg)==0:
                # args[i]=None
            # else:
                # args[i] = (arg[0],arg[-1],len(arg))
    # return(cachetools.keys.hashkey(*args,**kwargs))

# # @cachetools.cached(cache=cachetools.LRUCache(1e4),key=_voigt_cachetools_hashkey)
# def voigt(
        # k,                      # assumes k sorted
        # k0=0,
        # strength=1.,
        # gaussian_width=1.,
        # lorentzian_width=1, 
        # # method='mclean_etal1994',normalisation='area',
        # # method='wofz',
        # # method='wofz_parallel',
        # method='wofz_approx_long_range',
        # normalisation='area',
        # minimum_width=0.,
        # long_range_gaussian_cutoff_widths=10,
        # widths_cutoff=None,
# ):
    # """Approximates a voigt-profile.

    # Inputs:

    # k -- Array at which to evaluate function.
    # k0 -- Center of peak.
    # strength -- The meaning of this depends on normalisation.
    # method -- 'whiting1968', 'whiting1968 fortran', 'convolution',  
              # 'mclean_etal1994', 'wofz'
    # normalisation -- 'area', 'peak', 'sum', 'none'
    # minimum_width -- For Gaussian or Lorentzian widths below this just, ignore this
                    # component and return a pure Lorentzian or Gaussian.

    # Outputs:

    # v -- Array same size as k.

    # All methods are approximate, even convolution because of the
    # finite grid step employed.

    # References:

    # E. E. Whiting 1968, Journal Of Quantitative Spectroscopy &
    # Radiative Transfer 8:1379.

    # A. B. McLean et al. 1994, Journal of Electron Spectrosocpy and
    # Related Phenomena, 69:125.

     # Notes:

    # McLean et al. is more accurate than Whiting and slightly faster,
    # and convolution should be most accurate for sufficiently small
    # step size, but is very slow. Fortran version of Whiting is the
    # fastest, fotran version of Mclean is slightly slower!

    # Whiting method leads to zeros at small values (underflow
    # somewhere?).

    # Some of the analytical approximations for voigt profiles below are
    # area normalised by their definition, so I do not explicitly
    # normalise these, this leads to possible errors in their
    # normalisation of perhaps 0.5% or below.

    # wofz -- Use the real part of the Fadeeva function

    # wofz_approx_long_range -- Use the real part of the Fadeeva
           # function, after a certain range, just use a pure
           # Lorentzian.

    # """
    # if widths_cutoff is not None:
        # krange = (gaussian_width+lorentzian_width)*widths_cutoff
        # if (k[0]-k0)>krange or (k0-k[-1])>krange: return(np.zeros(k.shape)) # no line in range -- return zero
        # if np.abs(k[0]-k0)<krange and np.abs(k0-k[-1])<krange:          # all k is in range of line -- calculate all
            # return(voigt(k,k0,strength,gaussian_width,lorentzian_width,method,normalisation,minimum_width,widths_cutoff=None,))
        # ## else search for important range
        # ibeg,iend = k.searchsorted(k0-krange),k.searchsorted(k0+krange)
        # v = np.zeros(k.shape)
        # v[ibeg:iend] = voigt(k[ibeg:iend],k0,strength,gaussian_width,lorentzian_width,method,normalisation,minimum_width,widths_cutoff=None,)
        # return(v)
    # ## short cuts for negligible widths one way or the other
    # if   gaussian_width<=minimum_width:   method = 'lorentzian'
    # elif lorentzian_width<=minimum_width: method = 'gaussian'
    # ## Calculates profile.
    # if method == 'lorentzian':
        # v = lorentzian(k,k0,1.,lorentzian_width,norm='area')
    # elif method == 'gaussian':
        # v = gaussian(k,gaussian_width,k0,norm='area')
    # elif method == 'wofz':
        # from scipy import special
        # norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        # b = 0.8325546111576 # np.sqrt(np.log(2))
        # v = special.wofz((2.*(k-k0)+1.j*lorentzian_width)*b/gaussian_width).real/norm
    # elif method == 'wofz_parallel':
        # from scipy import special
        # norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        # b = 0.8325546111576 # np.sqrt(np.log(2))
        # results = multiprocessing.Pool().map(special.wofz,(2.*(k-k0)+1.j*lorentzian_width)*b/gaussian_width)
        # v = np.array(results).real/norm
    # elif method == 'wofz_approx_long_range':
        # from scipy import special
        # # i = np.abs(k-k0)<(gaussian_width+lorentzian_width)*long_range_gaussian_cutoff_widths
        # kcutoff = (gaussian_width+lorentzian_width)*long_range_gaussian_cutoff_widths
        # ibeg,iend = np.searchsorted(k,[k0-kcutoff,k0+kcutoff])
        # v = np.zeros(k.shape)
        # norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        # b = 0.8325546111576 # np.sqrt(np.log(2))
        # v[ibeg:iend] = special.wofz((2.*(k[ibeg:iend]-k0)+1.j*lorentzian_width)*b/gaussian_width).real/norm
        # v[:ibeg] = lorentzian(k[:ibeg],k0,1.,lorentzian_width,norm='area')
        # v[iend:] = lorentzian(k[iend:],k0,1.,lorentzian_width,norm='area')
    # elif method == 'whiting1968':
        # ## total FWHM - approximate formula
        # voigt_width=np.abs(lorentzian_width)/2.+np.sqrt(lorentzian_width**2/4+gaussian_width**2)
        # ## adjust for mean and width
        # x=np.abs(k-k0)/voigt_width
        # ## ratio of widths
        # a = abs(lorentzian_width)/voigt_width
        # ## lorentzian width and area normalised - approximate formula
        # v = (  ((1-a)*np.exp(-2.772*x**2) + a/(1+4*x**2) +  
                # 0.016*(1-a)*a*(np.exp(-0.4*x**2.25)-10/(10+x**2.25)))/  
               # (voigt_width*(1.065 + 0.447*a + 0.058*a**2))  )
    # elif method == 'whiting1968 fortran':
        # import pyvoigt
        # v = np.zeros(k.shape)
        # pyvoigt.voigt_whiting1968(lorentzian_width,gaussian_width,50.,k0,1.,k,v)
    # elif method == 'convolution':
        # ## convolution becomes displaced one grid point unless length k odd
        # if iseven(len(k)):
            # ktmp = k[1::]
            # l = lorentzian(ktmp,k0,1.,lorentzian_width)
            # g = gaussian(ktmp-ktmp[(len(ktmp)-1)/2],gaussian_width)
            # v = np.convolve(l,g,mode='same')
            # v = np.concatenate((v[0:1],v))
        # else:
            # l = lorentzian(k,0,1.,lorentzian_width)
            # g = gaussian(k-k[(len(k)-1)/2].mean(),gaussian_width,0.)
            # v = np.convolve(l,g,mode='same')
    # elif method == 'mclean_etal1994':
        # ## The method is inherently area normalised.
        # ## 0.939437278 = 2/pi*sqrt(pi*np.log(2))
        # A=[-1.2150,-1.3509,-1.2150,-1.3509]
        # B=[1.2359, 0.3786, -1.2359, -0.3786,]
        # C = [ -0.3085, 0.5906, -0.3085, 0.5906, ]
        # D = [0.0210, -1.1858, -0.0210, 1.1858, ]
        # X = 1.665109/gaussian_width*(k-k0)
        # Y = 0.8325546*lorentzian_width/gaussian_width
        # v = ( (C[0]*(Y-A[0])+D[0]*(X-B[0]))/((Y-A[0])**2+(X-B[0])**2)
              # + (C[1]*(Y-A[1])+D[1]*(X-B[1]))/((Y-A[1])**2+(X-B[1])**2)
              # + (C[2]*(Y-A[2])+D[2]*(X-B[2]))/((Y-A[2])**2+(X-B[2])**2)
              # + (C[3]*(Y-A[3])+D[3]*(X-B[3]))/((Y-A[3])**2+(X-B[3])**2) ) *0.939437278/gaussian_width
    # elif method == 'mclean_etal1994 fortran':
        # import pyvoigt
        # v = np.zeros(k.shape)
        # pyvoigt.voigt(lorentzian_width,gaussian_width,k0,1.,k,v)
    # else:
        # raise Exception(f'Unknown method: {repr(method)}')
    # ## Normalises and returns the profile.  Some methods are already
    # ## area normalised by their formulation.
    # if normalisation == 'none':
        # v = v*strength
    # elif normalisation == 'sum':
        # v = v/v.sum()*strength
    # elif normalisation == 'peak':
        # v = v/v.max()*strength
    # elif normalisation == 'area':
        # ## some methods are automatically area normalised
        # if method in ['whiting1968','whiting1968 fortran','mclean_etal1994',
                      # 'mclean_etal1994 fortran','lorentzian','gaussian',
                      # 'wofz','wofz_approx_long_range']:
            # v = v*strength
        # else:
            # v = v/integrate.simps(v,k)*strength
    # return(v)

# # def cross_correlate(x,y,max_shift=None,return_shift=False):
    # # """Normalised cross correlation."""
    # # retval = np.correlate(x,y,mode='same')
    # # n = len(retval)
    # # imid = int((n-1)/2)
    # # t = np.arange(n-imid,n,1)
    # # norm = np.concatenate((t,[n],t[::-1]))
    # # retval /= norm
    # # if return_shift:
        # # shift = np.arange(len(retval))-imid
        # # return(shift,retval)
    # # else:
        # # return(retval)

# def cross_correlate(
        # x0,y0,                  # x,y data one
        # x1,y1,                  # more x,y data
        # conv_width=None,          # how wide (in terms of x) to cross correlation
        # max_shift=None,           # maximum shift of cross corelation
# ):
    # """Normalised cross correlation."""
    # ## spline all data to min grid step
    # dx = min(np.min(x0[1:]-x0[:-1]),np.min(x1[1:]-x1[:-1]))
    # ##
    # tx0 = np.arange(x0.min(),x0.max(),dx)
    # y0 = spline(x0,y0,tx0)
    # x0 = tx0
    # tx1 = np.arange(x1.min(),x1.max(),dx)
    # y1 = spline(x1,y1,tx1)
    # x1 = tx1
    # ## mid points -- might drop a half pointx
    # imid0 = np.floor((len(x0)-1)/2)
    # imid1 = np.floor((len(x1)-1)/2)
    # ## get defaults conv_width and max_shift -- whole domain when added
    # if conv_width is None:
        # conv_width = dx*(min(imid0,imid1)-1)/2
    # if max_shift is None:
        # max_shift = conv_width
    # ## initalise convolved grid
    # xc = np.arange(0,max_shift,dx)
    # xc = np.concatenate((-xc[::-1],xc[1:]))
    # yc = np.full(xc.shape,0.0)
    # imidc = int((len(xc)-1)/2)
    # ## convert conv_width, max_shift to indexes, ensuring compatible
    # ## with data length
    # iconv_width = int(conv_width/dx) - 1
    # imax_shift = min(int(max_shift/dx),imid0-1,imid1-1) 
    # myf.cross_correlate(y0,y1,yc,imax_shift,iconv_width)
    # return(xc,yc)

def autocorrelate(x,nmax=None):
    if nmax is None:
        retval = np.correlate(x, x, mode='full')
        retval  = retval[int(len(retval-1)/2):]
    else:
        retval = np.empty(nmax,dtype=float)
        for i in range(nmax):
            retval[i] = np.sum(x[i:]*x[:len(x)-i])/np.sum(x[i:]**2)
    return retval

# def sinc(x,fwhm=1.,mean=0.,strength=1.,norm='area',):
    # """ Calculate sinc function. """
    # t = np.sinc((x-mean)/fwhm*1.2)*1.2/fwhm # unit integral normalised
    # if norm=='area':
        # return strength*t
    # elif norm=='sum':
        # return strength*t/t.sum()
    # elif norm=='peak':
        # return strength*t/np.sinc(0.)*1.2/fwhm

# def isfloat(a):
    # """Test if input is a floating point number - doesn't distinguish
    # between various different kinds of floating point numbers like a
    # simple test would."""
    # return type(a) in [float,float64]

# def isint(a):
    # """Test if input is an integer - doesnt distinguish between
    # various different kinds of floating point numbers like a simple
    # test would."""
    # return type(a) in [int,np.int64]

def isnumeric(a):
    """Test if constant numeric value."""
    return type(a) in [int,np.int64,float,np.float64]

# def iseven(x):
    # """Test if argument is an even number."""
    # return np.mod(x,2)==0

# def loadtxt(path,**kwargs):
    # """Sum as numpy.loadtxt but sets unpack=True. And expands '~' in
    # path."""
    # path = os.path.expanduser(path)
    # kwargs.setdefault('unpack',True)
    # return np.loadtxt(path,**kwargs)

# def loadawk(path,script,use_hdf5=False,**kwargs):
    # """Load text file as array after parsing through an
    # awkscript. Data is saved in a temporary file and all kwargs are
    # passed to loadtxt."""
    # path = expand_path(path)
    # tmpfile = pipe_through_awk(path,script)
    # if os.path.getsize(tmpfile.name)==0:
        # raise IOError(None,'No data found',path)
    # if use_hdf5:
        # output = txt_to_array_via_hdf5(tmpfile,**kwargs)
    # else:
        # output = np.loadtxt(tmpfile,**kwargs)
    # return output


def rootname(path,recurse=False):
    """Returns path stripped of leading directories and final
    extension. Set recurse=True to remove all extensions."""
    path = os.path.splitext(os.path.basename(path))[0]
    if not recurse or path.count('.')+path.count('/') == 0:
        return path
    else:
        return rootname(path,recurse=recurse)

def array_to_file(filename,*args,make_leading_directories=True,**kwargs):
    """Use filename to decide whether to attempt to save as an hdf5 file
    or ascii data.\n\nKwargs:\n\n mkdir -- If True, create all leading
    directories if they don't exist. """
    filename = expand_path(filename)
    extension = os.path.splitext(filename)[1]
    if mkdir:
        mkdir(dirname(filename))
    if extension in ('.hdf5','.h5'):
        array_to_hdf5(filename,*args,**kwargs)
    elif extension=='.npy':
        np.save(filename,args,**kwargs)
    elif extension=='.npz':
        np.savez_compressed(filename,args,**kwargs)
    else:
        if not any(isin(('fmt','header',),kwargs)):     # can't use via hdf5 for formatting
            try:
                return Array_to_txt_via_hdf5(filename,*args,**kwargs)
            except:
                pass
        np.savetxt(filename, np.column_stack(args),**kwargs) # fall back

# def load_and_spline(filename,x,missing_data=0.):
    # """Load data file using file_to_array. Resplines the results to a
    # grid x. Missing data set to zero. Returns splined y."""
    # xn,yn = file_to_array(filename,unpack=True)
    # y = np.zeros(x.shape)
    # i = (x>xn.min())&(x<xn.max())
    # y[i] = spline(xn,yn,x[i])
    # y[~i] = missing_data
    # return y

def loadxy(
        filename,
        xkey=None,ykey=None,
        xcol=None,ycol=None,
        **kwargs
):
    """Load x and y data from a file."""
    if xkey is not None:
        ## load by key
        from .dataset import Dataset
        d = Dataset()
        d.load(filename,**kwargs)
        x,y = d['x'],d['y']
    else:
        if xcol is None:
            xcol = 0
        if ycol is None:
            ycol = 1
        d = file_to_array(filename,**kwargs)
        x,y = d[:,xcol],d[:,ycol]
    return x,y

def file_to_array(
        filename,
        xmin=None,xmax=None,    # only load this range
        sort=False,             #
        check_uniform=False,
        awkscript=None,
        unpack=False,
        filetype=None,
        **kwargs,               # passed to function depending on filetype
):
    """Use filename to decide whether to attempt to load as an hdf5
    file or ascii data. xmin/xmax data ranges to load."""
    ## dealt with filename and type
    filename = expand_path(filename)
    if filetype is None:
        filetype = infer_filetype(filename)
    ## default kwargs
    kwargs.setdefault('comments','#')
    kwargs.setdefault('encoding','utf8')
    kwargs.setdefault('delimiter',' ')
    ## load according to filetype
    if filetype=='hdf5':
        hdf5_kwargs = copy(kwargs)
        for key,key_hdf5 in (
                ('comments',None),
                ('delimiter',None),
                ('encoding',None),
                ('skip_header',None),
        ):
            if key_hdf5 is None:
                if key in hdf5_kwargs: hdf5_kwargs.pop(key)
            else:
                hdf5_kwargs[key_hdf5] = hdf5_kwargs.pop(key)
        d = hdf5_to_array(filename,**hdf5_kwargs)
    elif filetype=='numpy':
        d = np.load(filename)
    elif filetype in ('opus', 'opus_spectrum', 'opus_background'):
        from . import bruker
        data = bruker.OpusData(filename)
        if filetype in ('opus','opus_spectrum') and data.has_spectrum():
            d = np.column_stack(data.get_spectrum())
        elif data.has_background():
            d = np.column_stack(data.get_background())
        else:
            raise Exception(f"Could not load opus data {filename=}")
    else:
        ## fallback try text
        np_kwargs = copy(kwargs)
        if len(filename)>4 and filename[-4:] in ('.csv','.CSV'):
            np_kwargs['delimiter'] = ','
        if 'delimiter' in np_kwargs and np_kwargs['delimiter']==' ':
            np_kwargs.pop('delimiter')
        d = np.genfromtxt(filename,**np_kwargs)
    ## post processing
    d = np.squeeze(d)
    if xmin is not None:
        d = d[d[:,0]>=xmin]
    if xmax is not None:
        d = d[d[:,0]<=xmax]
    if sort:
        d = d[np.argsort(d[:,0])]
    if check_uniform:
        Δd = np.diff(d[:,0])
        fractional_tolerance = 1e-5
        Δdmax,Δdmin = np.max(Δd),np.min(Δd)
        assert (Δdmax-Δdmin)/Δdmax<fractional_tolerance,f'{check_uniform=} and first column of data is not uniform within {fractional_tolerance=}'
    if unpack:
        d = d.transpose()
    return(d)

# def file_to_xml_tree(filename):
    # """Load an xml file using standard library 'xml'."""
    # from xml.etree import ElementTree
    # return(ElementTree.parse(expand_path(filename)))

# def pipe_through_awk(original_file_path, awk_script):
    # """
    # Pass file path through awk, and return the temporary file where
    # the result is sent. Doesn't load the file into python memory.
    # """
    # ## expand path if possible and ensure exists - or else awk will hang
    # original_file_path = expand_path(original_file_path)
    # if not os.path.lexists(original_file_path):
        # raise IOError(1,'file does not exist',original_file_path)
    # # if new_file is None:
    # new_file=tempfile.NamedTemporaryFile(mode='w+',encoding='utf-8')
    # # command = 'awk '+'\''+awk_script+'\' '+original_file_path+'>'+new_file_path
    # # # (status,output)=commands.getstatusoutput(command)
    # status = subprocess.call(['awk',awk_script,original_file_path],stdout=new_file.file)
    # assert status==0,"awk command failed"
    # new_file.seek(0)
    # return new_file

# def loadsed(path,script,**kwargs):
    # """
# `Load text file as array after parsing through an
    # sedscript. Data is saved in a temporary file and all kwargs are
    # passed to loadtxt.
    # """
    # tmpfile = sedFilteredFile(path,script)
    # output = np.loadtxt(tmpfile,**kwargs)
    # return output

# def sedFilteredFile(path,script):
    # """Load text file as array after parsing through a sed
    # script. Data is saved in a temporary file and all kwargs are
    # passed to loadtxt."""
    # tmpfile=tempfile.NamedTemporaryFile()
    # command = 'sed '+'\''+script+'\' '+path+'>'+tmpfile.name
    # (status,output)=subprocess.getstatusoutput(command)
    # if status!=0: raise Exception("sed command failed:\n"+output)
    # return tmpfile

# def ensureArray(arr):
    # """Return an at least 1D array version of input if not already one."""
    # if type(arr) != np.array:
        # if np.iterable(arr):
            # arr = np.array(arr)
        # else:
            # arr = np.array([arr])
    # return arr

# def string_to_list(string):
    # """Convert string of numbers separated by spaces, tabs, commas, bars,
    # and newlines into an array. Empty elements are replaced with NaN
    # if tabs are used as separators. If spaces are used then the excess
    # is removed, including all leading and trailing spaces."""
    # retval = [t.strip() for t in re.split(r'\n',string)
             # if not (re.match(r'^ *#.*',t) or re.match(r'^ *$',t))] # remove blank and # commented lines
    # retval = flatten([re.split(r'[ \t|,]+',t) for t in retval]) # split each x on space | or ,
    # retval = [try_cast_to_numerical(t) for t in retval]
    # return(retval)

def string_to_array(s,**array_kwargs):
    """Convert string of numbers separated by spaces, tabs, commas, bars,
    and newlines into an array. Empty elements are replaced with NaN
    if tabs are used as separators. If spaces are used then the excess
    is removed, including all leading and trailing spaces."""
    ## remove all data after an '#'
    s,count = re.subn(r' *#[^\n]*\n','\n',s)
    ## replace commas and bars with spaces
    s,count = re.subn(r'[|,]',' ',s)
    ## remove leading and trailing and excess spaces, and leading and
    ## trailing newlines
    s,count = re.subn(' {2,}',' ',s)
    s,count = re.subn(r'^[ \n]+|[ \n]+$','',s)
    # s,count = re.subn(r'^\n+|\n+$','',s)
    s,count = re.subn(r' *\n *','\n',s)
    ## spaces to tabs - split on tabs
    s = s.replace(' ','\t')
    ## split on \n
    s = s.splitlines()
    ## strip whitespace
    s = [t.strip() for t in s]
    ## remove blank lines
    s = [t for t in s if len(t)>0]
    ## split each line on tab, stripping each
    s = [[t0.strip() for t0 in t1.split('\t')] for t1 in s]
    ## convert missing values to NaNs
    for i in range(len(s)):
        if s[i] == []: s[i] = ['NaN']
        for j in range(len(s[i])):
            if s[i][j] == '': s[i][j] = 'NaN'
    ## convert to array of numbers, failing that array of strings
    try: 
        s = np.array(s,dtype=np.number,**array_kwargs)
    except ValueError:
        s = np.array(s,dtype=str,**array_kwargs)
        ## also try to convert into complex array -- else leave as string
        try:
            s = np.array(s,dtype=complex,**array_kwargs)
        except ValueError:
            pass
            # # warnings.warn('Nonnumeric value, return string array')
    ## if 2 dimensional transpose so that columns in text file are first index
    # if s.ndim==2: s=s.transpose()
    ## squeeze to smallest possible ndim
    # return s.squeeze()
    return(s.squeeze())

def string_to_array_unpack(s):
    '''return(string_to_array(s).transpose())'''
    return(string_to_array(s).transpose())

def array_to_string(*arrays,fmt='g',field_sep=' ',record_sep='\n'):
    """Convert array to a string format. Input arrays are concatenatd
    column wise.Nicer output than numpy.array2string, only works on 0,
    1 or 2D arrays. Format fmt can be a single string or a list of
    strings corresponding to each column."""
    a = np.column_stack(arrays)
    ## 0D
    if a.ndim==0: return(format(a[0],fmt))
    ## make 1D array 2D
    if a.ndim==1: a = a.reshape((-1,1))
    ## if fmt is a fmt string expand to list with same length as 2nd D
    ## of array
    if isinstance(fmt,str):
        fmt = fmt.strip()
        fmt = fmt.split()
        ## same fmt for all columsn else must be the same length as
        ## columns
        if len(fmt)==1: fmt = [fmt[0] for t in range(a.shape[1])]
    ## build string and return
    return(record_sep.join(
        [field_sep.join(
            [format(t0,t1) for (t0,t1) in zip(record,fmt)]
        ) for record in a]))

def string_to_file(
        filename,
        string,
        mode='w',
        encoding='utf8',
):
    """Write string to file_name."""
    filename = expand_path(filename)
    mkdir(dirname(filename))
    with open(filename,mode=mode,encoding=encoding) as f: 
        f.write(string)

# def str2range(string):
    # """Convert string of integers like '1,2,5:7' to an array of
    # values."""
    # x = string.split(',')
    # r = []
    # for y in x:
        # try:
            # r.append(int(y))
        # except ValueError:
            # y = y.split(':')
            # r.extend(list(range(int(y[0]),int(y[1])+1)))
    # return r

# def derivative(x,y=None,n=1):
    # """Calculate d^ny/dx^n using central difference - end points are
    # extrapolated. Endpoints could use a better formula."""
    # if y is None:
        # x,y = np.arange(len(x),dtype=float),x
    # if n==0:
        # return(y)
    # if n>1:
        # y = derivative(x,y,n-1)
    # d = np.zeros(y.shape)
    # d[1:-1] = (y[2::]-y[0:-2:])/(x[2::]-x[0:-2:])
    # d[0] = (y[1]-y[0])/(x[1]-x[0])
    # d[-1] = (y[-2]-y[-1])/(x[-2]-x[-1])
    # return d

# def curvature(x,y):
    # """Calculate curvature of function."""
    # d=derivative(x,y);  # 1st diff
    # dd=derivative(x,d); # 2nd diff
    # return dd/((1.+d**2.)**(3./2.)) 


# def execfile(filepath):
    # """Execute the file in current namespace. Not identical to python2 execfile."""
    # with open(filepath, 'rb') as fid:
        # # exec(compile(fid.read(),filepath,'exec'))
        # exec(fid.read())

# def file_to_string(filename):
    # with open(expand_path(filename),mode='r',errors='replace') as fid:
        # string = fid.read(-1)
    # return(string)

# def file_to_lines(filename,**open_kwargs):
    # """Split file data on newlines and return as a list."""
    # fid = open(expand_path(filename),'r',**open_kwargs)
    # string = fid.read(-1)
    # fid.close()
    # return(string.split('\n'))

# def file_to_tokens(filename,**open_kwargs):
    # """Split file on newlines and whitespace, then return as a list of
    # lists."""
    # return([line.split() for line in file_to_string(filename).split('\n')])

# def file_to_regexp_matches(filename,regexp):
    # """Return match objects for each line in filename matching
    # regexp."""
    # with open(expand_path(filename),'r') as fid:
        # matches = []
        # for line in fid:
            # match = re.match(regexp,line)
            # if match: matches.append(match)
    # return(matches)

def file_to_dict(filename,*args,filetype=None,**kwargs):
    """Convert text file to dictionary.
    \nKeys are taken from the first uncommented record, or the last
    commented record if labels_commented=True. Leading/trailing
    whitespace and leading commentStarts are stripped from keys.\n
    This requires that all elements be the same length. Header in hdf5
    files is removed."""
    filename = expand_path(filename)
    if filetype is None:
        filetype = infer_filetype(filename)
    if filetype == 'text':
        d = txt_to_dict(filename,*args,**kwargs)
    elif filetype=='npz':
        d = dict(**np.load(filename))
        ## avoid some problems later whereby 0D  arrays are not scalars
        for key,val in d.items():
            if val.ndim==0:
                d[key] = np.asscalar(val)
    elif filetype == 'hdf5':
        d = hdf5_to_dict(filename)
        if 'header' in d: d.pop('header') # special case header, not data 
        if 'README' in d: d.pop('README') # special case header, not data 
    elif filetype in ('csv','ods'):
        ## load as spreadsheet, set # as comment char
        kwargs.setdefault('comment','#')
        d = sheet_to_dict(filename,*args,**kwargs)
    elif filetype == 'rs':
        ## my convention -- a ␞ separated file
        kwargs.setdefault('comment_regexp','#')
        kwargs.setdefault('delimiter','␞')
        d = txt_to_dict(filename,*args,**kwargs)
    elif filetype == 'org':
        ## load as table in org-mode file
        d = org_table_to_dict(filename,*args,**kwargs)
    elif filetype == 'opus':
        raise ImplementationError()
        # ## load as table in org-mode file
        # d = org_table_to_dict(filename,*args,**kwargs)
    elif filetype == 'directory':
        ## load as data directory
        d = Data_Directory(filename)
    else:
        ## fall back try text
        d = txt_to_dict(filename,*args,**kwargs)
    return(d)



def infer_filetype(filename):
    """Determine type of datafile from the name or possibly its
    contents."""
    extension = os.path.splitext(filename)[1]
    if extension=='.npz':
        return 'npz'
    elif extension in ('.hdf5','.h5'): # load as hdf5
        return 'hdf5'
    elif extension == '.ods':
        return 'ods'
    elif extension in ('.csv','.CSV'):
        return 'csv'
    elif extension == '.rs':
        return 'rs'
    elif basename(filename) == 'README' or extension == '.org':
        return 'org'
    elif extension in ('.txt','.dat'):
        return 'text'
    elif re.match(r'.*\.[0-9]+$',basename(filename)):
        return 'opus'
    elif os.path.exists(filename) and os.path.isdir(filename):
        return 'directory'
    else:
        return None
    return(d)
    
def file_to_string(filename):
    with open(expand_path(filename),mode='r',errors='replace') as fid:
        string = fid.read(-1)
    return(string)

def file_to_lines(filename,**open_kwargs):
    """Split file data on newlines and return as a list."""
    fid = open(expand_path(filename),'r',**open_kwargs)
    string = fid.read(-1)
    fid.close()
    return(string.split('\n'))

def file_to_tokens(filename,**open_kwargs):
    """Split file on newlines and whitespace, then return as a list of
    lists."""
    return([line.split() for line in file_to_string(filename).split('\n')])

def org_table_to_dict(filename,table_name=None):
    """Load a table into a dicationary of arrays. table_name is used to
    find a #+NAME: tag."""
    with open(filename,'r') as fid:
        if table_name is not None:
            ## scan to beginning of table
            for line in fid:
                if re.match(r'^ *#\+NAME: *'+re.escape(table_name)+' *$',line):
                    break
            else:
                raise Exception("Could not find table_name "+repr(table_name)+" in file "+repr(filename))
        ## skip other metadata
        for line in fid:
            if not re.match(r'^ *#',line): break
        ## load lines of table
        table_lines = []
        for line in fid:
            ## skip horizontal lines
            if re.match(r'^ *\|-',line): continue
            ## end of table
            if not re.match(r'^ *\|',line): break
            ## remove new line
            line = line[:-1]
            ## remove leading and pipe character
            line = re.sub(r'^ *\| *',r'',line)
            ## remove empty following fields
            line = re.sub(r'^[ |]*',r'',line[-1::-1])[-1::-1]
            table_lines.append(line.split('|'))
        ## turn into an array of dicts
        return(stream_to_dict(iter(table_lines)))

def string_to_dict(string,**kwargs_txt_to_dict):
    """Convert a table in string from into a dict. Keys taken from
    first row. """
    ## turn string into an IO object and pass to txt_to_dict to decode the lines. 
    # string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    import io
    kwargs_txt_to_dict.setdefault('labels_commented',False)
    return(txt_to_dict(io.StringIO(string),**kwargs_txt_to_dict))

# def string_to_recarray(string):
    # """Convert a table in string from into a recarray. Keys taken from
    # first row. """
    # ## turn string into an IO object and pass to txt_to_dict to decode the lines. 
    # string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    # import io
    # return(dict_to_recarray(txt_to_dict(io.StringIO(string), labels_commented=False,)))

# # def string_to_dynamic_recarray(string):
    # # string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    # # import io
    # # d = txt_to_dict(io.StringIO(string), labels_commented=False,)
    # # from data_structures import Dynamic_Recarray
    # # return(Dynamic_Recarray(**d))

def file_to_recarray(filename,*args,**kwargs):
    """Convert text file to record array, converts from dictionary
    returned by file_to_dict."""
    return(dict_to_recarray(file_to_dict(filename,*args,**kwargs)))

def file_to_dataset(*args,**kwargs):
    from . import dataset
    return dataset.Dataset(**file_to_dict(*args,**kwargs))

# def decompose_ufloat_array(x):
    # """Return arrays of nominal_values and std_devs of a ufloat array."""
    # return(np.array([t.nominal_value for t in x],dtype=float),np.array([t.std_dev for t in x],dtype=float),)

# def item(x):
    # """Recursively penetrate layers of iterable containers to get at
    # the one number inside. Also tries to get value out of zero
    # dimensional array"""
    # if iterable(x): return(item(x[0]))  # first value only
    # elif hasattr(x,'item'): return x.item()
    # else: return x

# def jj(x): return x*(x+1)

# def txt2multiarray(path):
    # """Text file contains 1 or 2D arrays which are separated by blank
    # lines or a line beginning with a comment character. These are read
    # into separate arrays and returned."""
    # f = open(expand_path(path),'r')
    # s = f.read()
    # f.close()
    # s,n = re.subn('^(( *($|\n)| *#.*($|\n)))+','',s)
    # s,n = re.subn('((\n *($|\n)|\n *#.*($|\n)))+','#',s)
    # if s=='':
        # s = []
    # else:
        # s = s.split('#')
    # ## return, also removes empty arrays
    # return [str2array(tmp).transpose() for tmp in s if len(tmp)!=0]

# def fwhm(x,y,plot=False,return_None_on_error=False):
    # """Roughly calculate full-width half-maximum of data x,y. Linearly
    # interpolates nearest points to half-maximum to get
    # full-width. Requires single peak only in view. """
    # hm = (np.max(y)-np.min(y))/2.
    # i = find((y[1:]-hm)*(y[:-1]-hm)<0)
    # if len(i)!=2:
        # if return_None_on_error:
            # return(None)
        # else:
            # raise Exception("Poorly defined peak, cannot find fwhm.")
    # x0 = (hm-y[i[0]])*(x[i[0]]-x[i[0]+1])/(y[i[0]]-y[i[0]+1]) + x[i[0]]
    # x1 = (hm-y[i[1]])*(x[i[1]]-x[i[1]+1])/(y[i[1]]-y[i[1]+1]) + x[i[1]]
    # if plot==True:
        # ax = plt.gca()
        # ax.plot(x,y)
        # ax.plot([x0,x1],[hm,hm])
    # return x1-x0

# def estimate_fwhm(
        # x,y,
        # imax=None,              # index of peak location -- else uses maximum of y
        # plot=False,             
# ):
    # """Roughly calculate full-width half-maximum of data x,y. Linearly
    # interpolates nearest points to half-maximum to get
    # full-width. Looks around tallest peak."""
    # if imax is None: imax = np.argmax(y)
    # half_maximum = y[imax]/2.
    # ## index of half max nearest peak on left
    # if all(y[1:imax+1]>half_maximum): raise Exception('Could not find lower half maximum.')
    # i = find((y[1:imax+1]>half_maximum)&(y[:imax+1-1]<half_maximum))[-1]
    # ## position of half max on left
    # x0 = (half_maximum-y[i])*(x[i]-x[i+1])/(y[i]-y[i+1]) + x[i]
    # ## index of half max nearest peak on left
    # if all(y[imax:]>half_maximum): raise Exception('Could not find upper half maximum.')
    # i = find((y[imax+1:]<half_maximum)&(y[imax:-1]>half_maximum))[0]+imax
    # ## position of half max on left
    # x1 = (half_maximum-y[i])*(x[i]-x[i+1])/(y[i]-y[i+1]) + x[i]
    # if plot==True:
        # ax = plt.gca()
        # ax.plot(x,y)
        # ax.plot([x0,x1],[half_maximum,half_maximum])
    # return x1-x0

# def fixedWidthString(*args,**kwargs):
    # """Return a string joining *args in g-format with given width, and
    # separated by one space."""
    # if 'width' in kwargs: width=kwargs['width']
    # else: width=13
    # return ' '.join([format(s,str(width)+'g') for s in args])

# def string_to_number(s,default=None):
    # """ Attempt to convert string to either int or float. If fail use
    # default, or raise error if None."""
    # ## if container, then recursively operate on elements instead
    # if np.iterable(s) and not isinstance(s,str):
        # return([str2num(ss) for ss in s])
    # elif not isinstance(s,str):
        # raise Exception(repr(s)+' is not a string.')
    # ## try to return as int
    # try:
        # return int(s)
    # except ValueError:
        # ## try to return as float
        # try:
            # return float(s)
        # except ValueError:
            # if default is not None:
                # return(default)
            # raise Exception(f'Could not convert string to number: {repr(s)}')

def string_to_number_if_possible(s):
    """ Attempt to convert string to either int or float. If fail use
    default, or raise error if None."""
    ## if container, then recursively operate on elements instead
    if np.iterable(s) and not isinstance(s,str):
        return([string_to_number_if_possible(ss) for ss in s])
    elif not isinstance(s,str):
        raise Exception(repr(s)+' is not a string.')
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s 

# def txt_to_recarray(*args,**kwargs):
    # """See txt_to_dict."""
    # return(dict_to_recarray(txt_to_dict(*args,**kwargs)))

def txt_to_dict(
        filename,              # path or open data stream (will be closed)
        labels=None,
        delimiter=None,
        skiprows=0,
        comment_regexp='#',
        labels_commented=True,
        awkfilter=None,
        filter_function=None,
        filter_regexp=None,
        ignore_blank_lines=True,
        replacement_for_blank_elements='nan'
):
    """Convert text file to dictionary. Keys are taken from the first
    uncommented record, or the last commented record if
    labels_commented=True. Leading/trailing whitespace and leading
    comment_starts are stripped from keys.
    filter_function: if not None run lines through this function before
    parsing.
    filter_regexp: if not None then must be (pattern,repl) and line
    run through re.sub(pattern,repl,line) before parsing.
    """
    ## If filename is a path name, open it, else assumed is already an open file.
    if type(filename)==str:
        filename = expand_path(filename)
        if awkfilter is not None:
            filename = pipe_through_awk(filename,awkfilter)
        else:
            filename=open(filename,'r', encoding='utf8')
    ## Reads all data, filter, and split
    lines = []
    last_line_in_first_block_of_commented_lines = None
    first_block_commented_lines_passed = False
    number_of_columns = None
    for i,line in enumerate(filename.readlines()):
        if i<skiprows:
            continue
        line = line.strip()     # remove leading/trailing whitespace
        if ignore_blank_lines and len(line)==0:
            continue
        if filter_function is not None:
            line = filter_function(line)
        if filter_regexp is not None:
            line = re.sub(filter_regexp[0],filter_regexp[1],line)
        line = (line.split() if delimiter is None else line.split(delimiter)) # split line
        if comment_regexp is not None and re.match(comment_regexp,line[0]): # commented line found
            if not first_block_commented_lines_passed:
                line[0] = re.sub(comment_regexp,'',line[0]) # remove comment start
                if len(line[0])==0:
                    line = line[1:] # first element was comment only,
                last_line_in_first_block_of_commented_lines = line
            continue
        first_block_commented_lines_passed = True # look for no more key labels
        if number_of_columns is None:
            number_of_columns = len(line) 
        else:
            assert len(line)==number_of_columns,f'Wrong number of column on line {i}'
        lines.append(line)      # store this line, it contains data
    filename.close()
    if number_of_columns is None: return({}) # no data
    ## get data labels
    if labels is None:          # look for labels if not given
        if labels_commented:    # expect last line of initial unbroken comment block
            if last_line_in_first_block_of_commented_lines is None:
                labels = ['column'+str(i) for i in range(number_of_columns)] # get labels as column indices
            else:
                labels = [t.strip() for t in last_line_in_first_block_of_commented_lines] # get labels from commented line
        else:
            labels = [t.strip() for t in lines.pop(0)] # get from first line of data
    assert len(set(labels))==len(labels),'Non-unique data labels: '+repr(labels)
    assert len(labels)==number_of_columns,f'Number of labels ({len(labels)}) does not match number of columns ({number_of_columns})'
    if len(lines)==0: return({t:[] for t in key}) #  no data
    ## get data from rest of file, and convert to arrays
    data = dict()
    for key,column in zip(labels,zip(*lines)):
        column = [(t.strip() if len(t.strip())>0 else replacement_for_blank_elements) for t in column]
        data[key] = try_cast_to_numerical_array(column)
    return data 
            
# def txt_to_array_unpack(filename,skiprows=0,comment='#'):
    # """Read a text file of 2D data into a seperate array for each
    # column. Attempt to cast each column independently."""
    # ## Reads all data.
    # fid = open(filename,'r')
    # lines = fid.readlines()
    # fid.close()
    # ## remove blank and commented lines
    # lines = [l for l in lines if not re.match(r'^\s*$|^ *'+comment,l)]
    # ## skiprows
    # for i in range(skiprows): lines.pop(0)
    # ## initialise data lists
    # data = [[] for t in lines[0].split()]
    # ## get data
    # for line in lines:
        # for x,y in zip(data,line.split()):
            # x.append(y)
    # ## cast to arrays, numerical if possiblec
    # data = [try_cast_to_numerical_array(t) for t in data]
    # return(data)
            
# def org_table_to_recarray(filename,table_name):
    # """Read org-mode table from a file, convert data into a recarray, as
# numbers if possible, else strings. The table_name is expected to be an
# org-mode name: e.g., #+NAME: table_name"""
    # fid = open(expand_path(filename),'r')
    # data = dict()
    # ## read file to table found
    # for line in fid:
        # if re.match('^ *#\\+NAME\\: *'+re.escape(table_name.strip())+' *$',line): break
    # else:
        # raise Exception('table: '+table_name+' not found in file: '+filename)
    # ## get keys line and initiate list in data dict
    # for line in fid:
        # if re.match(r'^ *\|-',line): continue # skip hlines
        # line = line.strip(' |\n')
        # keys = [key.strip('| ') for key in line.split('|')]
        # break
    # for key in keys: data[key] = []
    # ## loop through until blank line (end of table) adding data
    # for line in fid:
        # if line[0]!='|': break                # end of table
        # if re.match(r'^ *\|-',line): continue # skip hlines
        # line = line.strip(' |')
        # vals = [val.strip() for val in line.split('|')]
        # for (key,val) in zip(keys,vals): data[key].append(val)
    # for key in data: data[key] = try_cast_to_numerical_array(data[key]) # cast to arrays, numerical if possiblec
    # return(dict_to_recarray(data))

def try_cast_to_numerical(string):
    """Try to cast into a numerical type, or else return as a string."""
    try:                 return(int(string))
    except ValueError:   pass
    try:                 return(float(string))
    except ValueError:   pass
    try:                 return(complex(string))
    except ValueError:   pass
    return(str(string))

def try_cast_to_numerical_array(x):
    """Try to cast an interator into an array of ints. On failure try
    floats. On failure return as array of strings."""
    try:
        return np.array(x,dtype=float)
    except ValueError:
        return np.array(x,dtype=str)

# def prefix_postfix_lines(string,prefix='',postfix=''):
    # """Prefix and postfix every line of string."""
    # return(prefix+string.replace('\n',postfix+'\n'+prefix)+postfix)

# def extractHeader(path):
    # """
    # Extract info of the from XXX=YYY from a txt or csv file returning
    # a dictionary with all keys and values given by strings.

    # Header indicated by comment character '#'.

    # All quotes are stripped.

    # Wildly unfinished.
    # """
    # retDict = {}
    # fid = open(path,'r')
    # while True:
        # line = fid.readline()
        # ## eliminate all quotes
        # line = line.replace("'","")
        # line = line.replace('"','')
        # ## break once header pased
        # if not re.match(r'^\s*#',line): break
        # ## eliminate leading comments
        # line = re.sub(r'^\s*#*','',line)
        # ## tokenise on space and ','
        # toks = re.split(r' +| *, *',line)
        # ## find tokens containing '=' and extract key/vals as
        # ## dictionary
        # for tok in toks:
            # if re.match(r'.*=.*',tok):
                # key,val = tok.split('=')
                # retDict[key.strip()] = val.strip()
    # fid.close()
    # return retDict


# def odsReader(fileName,tableIndex=0):
    # """
    # Opens an odf spreadsheet, and returns a generator that will
    # iterate through its rows. Optional argument table indicates which
    # table within the spreadsheet. Note that retures all data as
    # strings, and not all rows are the same length.
    # """
    # import odf.opendocument,odf.table,odf.text
    # ## common path expansions
    # fileName = expand_path(fileName)
    # ## loads sheet
    # sheet = odf.opendocument.load(fileName).spreadsheet
    # ## Get correct table. If 'table' specified as an integer, then get
    # ## from numeric ordering of tables. If specified as a string then
    # ## search for correct table name.
    # if isinstance(tableIndex,int):
        # ## get by index
        # table = sheet.getElementsByType(odf.table.Table)[tableIndex]
    # elif isinstance(tableIndex,str):
        # ## search for table by name, if not found return error
        # for table in sheet.getElementsByType(odf.table.Table):
            # # if table.attributes[(u'urn:oasis:names:tc:opendocument:xmlns:table:1.0', u'name')]==tableIndex:
            # if table.getAttribute('name')==tableIndex: break
        # else:
            # raise Exception('Table `'+str(tableIndex)+'\' not found in `'+str(fileName)+'\'')
    # else:
        # raise Exception('Table name/index`'+table+'\' not understood.')
    # ## divide into rows
    # rows = table.getElementsByType(odf.table.TableRow)
    # ## For each row divide into cells and then insert new cells for
    # ## those that are repeated (multiple copies are not stored in ods
    # ## format). The number of multiple copies is stored as a string of
    # ## an int.
    # for row in rows:
        # cellStrs = []
        # for cell in row.getElementsByType(odf.table.TableCell):
            # cellStrs.append(str(cell))
            # if cell.getAttribute('numbercolumnsrepeated')!=None:
                # for j in range(int(cell.getAttribute('numbercolumnsrepeated'))-1):
                    # cellStrs.append(str(cell))
        # ## yield each list of cells to make a generator
        # yield cellStrs

# def loadsheet(path,tableIndex=0):
    # """Converts contents of a spreadsheet to a list of lists or
    # strings. For spreadsheets with multiple tables, this can be
    # specified with the optional argument."""
    # ## opens file according to extension
    # if path[-4:]=='.csv' or path[-4:]=='.CSV':
        # assert tableIndex==0, 'Multiple tables not defined for csv files.'
        # fid = open(path,'r')
        # sheet = csv.reader(fid)
    # elif path[-4:]=='.ods':
        # sheet = odsReader(path,tableIndex=tableIndex)
    # ## get data rows into list of lists
    # ret = [row for row in sheet]
    # ## close file if necessary
    # if path[-4:]=='.csv' or path[-4:]=='.CSV':
        # fid.close()
    # return(ret)      

# def sheet_to_recarray(*args,**kwargs):
    # return(dict_to_recarray(sheet_to_dict(*args,**kwargs)))

# def sheet_to_dataframe(*args,**kwargs):
    # import pandas as pd
    # return(pd.DataFrame(sheet_to_dict(*args,**kwargs)))

def sheet_to_dict(path,return_all_tables=False,skip_header=None,**kwargs):
    """Converts csv or ods file, or list of lists to a dictionary.\n\nFor
    csv files, path can be open file object or path. For ods it must
    be a path\n\nKeys read from first row unless skiprows
    specified.\n\nIf tableName is supplied string then keys and data
    are read betweenen first column flags <tableName> and
    <\\tableName>. Other wise reads to end of file.\n\nConversions
    specify a dictionary of (key,function) pairs where function is
    used to convert the string from of each element of key, rather
    than str2num.\n\nFurther kwargs are passed to csv.reader if a csv
    file is used, or for ods files ignored.\n\nIf there is missing
    data the line might get ignored.  \nLeading/trailing white space
    and leading commentChar.\n\nSpecify ods/xls sheet with
    sheet_name=name.\n\nIf return_all_tables, return a dict of dicts,
    with keys given by all table names found in sheet. """
    ## deprecated kwargs
    if 'tableName' in kwargs:
        kwargs['table_name'] = kwargs.pop('tableName')
    if 'commentChar' in kwargs:
        kwargs['comment'] = kwargs.pop('commentChar')
    ## open generator reader according to file extension
    fid = None
    if isinstance(path,list):
        reader = (line for line in path)
    ## some common path expansions
    elif isinstance(path,str) and (path[-4:]=='.csv' or path[-4:]=='.CSV'):
        fid=open(expand_path(path),'r')
        reader=csv.reader(
            fid,
            skipinitialspace=True,
            quotechar=(kwargs.pop('quotechar') if 'quotechar' in kwargs else '"'),)
    elif isinstance(path,str) and path[-4:] =='.ods':
        kwargs.setdefault('sheet_name',0)
        reader=odsReader(expand_path(path),tableIndex=kwargs.pop('sheet_name'))
    elif isinstance(path,str) and path[-5:] =='.xlsx':
        assert 'sheet_name' not in kwargs,'Not implemented'
        import openpyxl
        data = openpyxl.open(path,read_only=True,data_only=True,keep_links=False)
        print( data)
        reader=odsReader(expand_path(path),tableIndex=kwargs.pop('sheet_name'))
    elif isinstance(path,io.IOBase):
        reader=csv.reader(expand_path(path),)
    else:
        raise Exception("Failed to open "+repr(path))
    ## if skip_header is set this is the place to pop the first few recrods of the reader objects
    if skip_header is not None:
        for t in range(skip_header): next(reader)
    ## if requested return all tables. Fine all names and then call
    ## sheet2dict separately for all found tables.
    if return_all_tables:
        return_dict = dict()
        for line in reader:
            if len(line)==0: continue
            r = re.match(r'<([^\\][^>]*)>',line[0],)
            if r:
                table_name = r.groups()[0]
                return_dict[table_name] = sheet_to_dict(path,table_name=table_name,**kwargs)
        return return_dict
    ## load the data into a dict
    data = stream_to_dict(reader,**kwargs)
    ## close file if necessary
    if fid!=None: fid.close()
    ## return
    return data

def stream_to_dict(
        stream,
        split=None,             # a string to split rows on
        comment=None,           # a string to remove from beginnig of rows (regexp is comment+)
        table_name=None,
        conversions={},
        skip_rows=0,
        error_on_missing_data=False,
        types = None,           # a dictionary of data keys will cast them as this type
        cast_types=True,        # attempt to convert strings to numbers
):
    r"""Read a stream (line-by-line iterator) into a dictionary. First
    line contains keys for columns. An attempt is made to cast as
    numeric data.\n\nsplit -- if not None, split line on this character
    comment -- remove from keys if not None
    table_name -- stop reading at end of <\table_name> or <\>
    conversions -- convert data belonging to keys of conversions by a function"""
    ## get keys first line must contain keys, split if requested, else already
    ## iterable, remove trailing new line if necessary
    def get_line():
        line = next(stream)
        ## split if split string given
        if split!=None:
            if line[-1]=='\n':
                line = line[:-1]
            line = line.split(split)
        ## else check if already split (i.e., in a list) or make a
        ## list with one element
        else:
            if np.isscalar(line):
                line = [line]
        ## blank lines -- skip (recurse)
        if len(line)==0:
            line = get_line()
        ## if a comment string is defined then skip this line
        ## (recurse) if it begins with a comment
        if comment!=None:
            if re.match(r'^ *'+re.escape(comment),line[0]): line = get_line()
        return line
    ## skip rows if requested
    for i in range(skip_rows): 
        next(reader)
    ## if requested, scan through file until table found
    if table_name!=None:
        while True:
            try:
                line = get_line()
                if line==[]: continue   # blank line continue
            except StopIteration:
                raise Exception('table_name not found: '+repr(table_name))
            ## if table specified stop reading at the end of it and dont'
            ## store data before it
            except:
                raise               # an actual error
            if len(line)>0 and str(line[0]).strip()=='<'+table_name+'>':
                break # table found
    ## this line contains dict keys
    keys = get_line()           
    ## eliminate blank keys and those with leading/trailing space and
    ## comment char from keys
    if comment != None:
        keys = [re.sub(r'^ *'+comment+r' *',r'',key) for key in keys]
    keys = [key.strip() for key in keys] # remove trailing/leading white space around keys
    nonBlankKeys=[]
    nonBlankKeys = [i for i in range(len(keys)) if keys[i] not in ['','None']]
    keys = [keys[i] for i in nonBlankKeys]
    ## check no repeated keys
    assert len(keys)==len(np.unique(keys)),'repeated keys'
    ## initialise dictionary of lists
    data = dict()
    for key in keys: data[key] = []
    ## read line-by-line, collecting data
    while True:
        try:
            line = get_line()
            if line==[]: continue   # blank line conitinue
        except StopIteration: break # read until end of file
        except: raise               # an actual error
        ## if table specified stop reading at the end of it and dont'
        ## store data before it
        if table_name!=None:
            if '<\\'+table_name+'>'==str(line[0]): break
            if str(line[0]).strip() in ('<\\>','<\\'+table_name+'>',): break
        if len(line)==0 or (len(line)==1 and line[0]==''): continue # skip empty data
        ## if partially missing data pad with blanks or raise an error
        if len(line)<len(keys):
            if error_on_missing_data:
                raise Exception('Length data less than length keys: '+str(line))
            else:
                line.extend(['' for t in range(len(keys)-len(line))])
        line = np.take(line,nonBlankKeys)
        ## add data to lists - loop through each cell and try to cast
        ## it appropriately, if a conversions is explicitly given for
        ## each key, then use that instead
        for (key,cell) in zip(keys,line):
            if key in conversions:
                data[key].append(conversions[key](cell))
            else:
                cell = cell.strip() # remove end blanks
                ## data[key].append(str2num(cell,default_to_nan=False,blank_to_nan=True))
                if cell=="": cell = "nan" # replace empty string with "nan" to facilitate possible numerical convervsion
                data[key].append(cell)
    ## Convert lists to arrays of numbers or whatever. If type given
    ## in types use that, esle try to cast as int, on failure try
    ## float, on failure revert to str.
    if cast_types:
        for key in keys:
            if types is not None and key in types:
                data[key] = np.array(data[key],dtype=types[key])
            else:
                data[key] = try_cast_to_numerical_array(data[key])
    return data

# def read_structured_data_file(filename):
    # """Read a file containing tables and key-val pairs"""
    # d = {'header':[]}
    # with open(expand_path(filename)) as f:
        # # for i in range(50):
        # while True:
            # l = f.readline()    # read a line
            # if l=='': break     # end of text file
            # l = re.sub('^ *# *(.*)',r'\1',l) # remove comments
            # l = l[:-1]          # remove newline
            # ## look for lines like: a = b
            # r = re.match('^ *([^=]*[^= ]) *= *(.+) *$',l)
            # if r:
                # d[r.groups()[0]] = r.groups()[1]
                # d['header'].append(l)
                # continue
            # ## look for tables like: <tablename> ...
            # r = re.match('^ *< *(.+) *> *$',l)
            # if r:
                # table_name = r.groups()[0]
                # d[table_name] = stream_to_dict(f,split=' ',table_name=table_name,comment='#')
                # continue
            # ## rest is just text of some kind
            # d['header'].append(l)
    # return d 

# def csv2array(fileObject,tableName=None,skiprows=0,**kwargs):
    # """Convert csv file to array.
    # \nFile can be open file object or path.
    # \nIf tableName is supplied string then keys and data are read
    # between first column flags <tableName> and <\\tableName>. Other
    # wise reads to end of file.
    # \nFurther kwargs are passed to csv.reader."""
    # ## open csv.reader
    # if type(fileObject)==str:
        # f=open(expand_path(fileObject),'r')
        # reader=csv.reader(f,**kwargs)
    # else: 
        # reader=csv.reader(fileObject,**kwargs)
    # ## skip to specific table if requested
    # while tableName!=None:
        # try: 
            # line = next(reader)
        # except StopIteration: 
            # raise Exception('Table `'+tableName+'\' not found in file '+str(fileObject))
        # ## skip empty lines
        # if len(line)==0:
            # continue
        # ## table found, break loop
        # if '<'+tableName+'>'==str(line[0]): 
            # break
    # ## skiprows - IF READING TABLE AND GET TO END OF TABLE IN HERE THERE WILL BE A BUG!!!!
    # for i in range(skiprows): 
        # next(reader)
    # ## initialise list of lists
    # data = []
    # ## read data
    # while True:
        # ## break on end of file
        # try:    line = next(reader)
        # except: break
        # ## if table specified stop reading at the end of it
        # if tableName!=None and '<\\'+tableName+'>'==str(line[0]): break
        # if tableName!=None and '<\\>'==str(line[0]): break
        # ## add new row to data and convert to numbers
        # data.append([str2num(cell) for cell in line])
    # ## convert lists to array
    # if isiterable(data[0]):
        # ## if 2D then pad short rows
        # data = array2DPadded(data)
    # else:
        # data = np.array(data)
    # ## close file if necessary
    # if type(fileObject)==str: f.close()
    # ## return
    # return data

# def loadcsv(path,**kwargs):
    # """Convert csv file to array. Further kwargs are passed to
    # csv.reader."""
    # ## open csv.reader
    # f = open(expand_path(path),'r')
    # reader = csv.reader(f,**kwargs)
    # ## initialise list of lists
    # data = []
    # ## read data
    # while True:
        # ## break on end of file
        # try:    line = next(reader)
        # except: break
        # ## add new row to data and convert to numbers
        # data.append([str2num(cell) for cell in line])
    # ## convert lists to array
    # # if isiterable(data[0]):
        # # ## if 2D then pad short rows
        # # data = np.array2DPadded(data)
    # # else:
    # data = np.array(data)

    # ## close file if necessary
    # f.close()
    # ## return
    # return data

# sheet2array = csv2array         # doesn't currently work with ods, fix one day

# def writeCSV(path,data,**kwargs):
    # """
    # Writes data to path as a CSV file.\n\nPath is filename.
    # Data is a 2D iterable.
    # Kwargs are passed as options to csv.writer, with sensible defaults.
    # """
    # kwargs.setdefault('skipinitialspace',True)
    # kwargs.setdefault('quoting',csv.QUOTE_NONNUMERIC)
    # fid = open(path,'w')
    # writer = csv.writer(fid,**kwargs)
    # writer.writerows(data)
    # fid.close()

# def isuvalue(x):
    # """Test if uvalue in a fairly general way."""
    # if x in (uncertainties.Variable,ufloat): return(True)
    # if isinstance(x,uncertainties.Variable): return(True)
    # if isinstance(x,uncertainties.AffineScalarFunc): return(True)
    # return(False)

# def array2DPadded(x):
    # """Take 2D nested iterator x, and turn it into array, extending
    # any dim=1 arrays by NaNs to make up for missing values."""
    # dim1 = max([len(xi) for xi in x])
    # for xi in x:
        # xi.extend([np.nan for i in range(dim1-len(xi))])
    # return np.array(x)

###################
## miscellaneous ##
###################

# def digitise_postscript_figure(
        # filename,
        # xydpi_xyvalue0 = None,  # ((xdpi,ydpi),(xvalue,yvalue)) for fixing axes
        # xydpi_xyvalue1 = None,  # 2nd point for fixing axes

# ):
    # """Get all segments in a postscript file. That is, an 'm' command
    # followed by an 'l' command. Could find points if 'm' without an 'l' or
    # extend to look for 'moveto' and 'lineto' commands."""
    # data = file_to_string(filename).split() # load as list split on all whitespace
    # retval = []                  # line segments
    # ## loop through looking for line segments
    # i = 0
    # while (i+3)<len(data):
        # if data[i]=='m' and data[i+3]=='l': # at least one line segment
            # x,y = [float(data[i-1])],[-float(data[i-2])]
            # while (i+3)<len(data) and data[i+3]=='l':
                # x.append(float(data[i+2]))
                # y.append(-float(data[i+1]))
                # i += 3
            # retval.append([x,y])
        # i += 1
    # ## make into arrays
    # for t in retval:
        # t[0],t[1] = np.array(t[0],ndmin=1),np.array(t[1],ndmin=1)
    # ## transform to match axes if possible
    # if xydpi_xyvalue0 is not None:
        # a0,b0,a1,b1 = xydpi_xyvalue0[0][0],xydpi_xyvalue0[1][0],xydpi_xyvalue1[0][0],xydpi_xyvalue1[1][0]
        # m = (b1-b0)/(a1-a0)
        # c = b0-a0*m
        # xtransform = lambda t,c=c,m=m:c+m*t
        # a0,b0,a1,b1 = xydpi_xyvalue0[0][1],xydpi_xyvalue0[1][1],xydpi_xyvalue1[0][1],xydpi_xyvalue1[1][1]
        # m = (b1-b0)/(a1-a0)
        # c = b0-a0*m
        # ytransform = lambda t,c=c,m=m:c+m*t
        # for t in retval:
            # t[0],t[1] = xtransform(t[0]),ytransform(t[1])
    # return(retval)

# def bibtex_file_to_dict(filename):
    # """Returns a dictionary with a pybtex Fielddict, indexed by bibtex
    # file keys."""
    # from pybtex.database import parse_file
    # database  = parse_file(filename)
    # entries = database.entries
    # retval_dict = dict()
    # for key in entries:
        # fields = entries[key].rich_fields
        # retval_dict[key] = {key:str(fields[key]) for key in fields}
    # return(retval_dict)



##############################################################
## functions for time series / spectra / cross sections etc ##
##############################################################

def fit_noise_level(x,y,order=3,plot=False,fig=None):
    """Fit a polynomial through some noisy data and calculate statistics on
    the residual.  Set plot=True to see how well this works."""
    p = np.polyfit(x,y,order)
    yf = np.polyval(p,x)
    r = y-yf
    nrms = rms(r)
    if plot:
        if fig is None:
            fig = plotting.qfig()
        else:
            fig.clf()
        ax = fig.gca()
        ax.plot(x,y)
        ax.plot(x,yf)
        ax = plotting.subplot()
        ax.plot(x,r)
        plotting.hist_with_normal_distribution(r, ax=plotting.subplot())
    return nrms

def fit_background(
        x, y,
        fit_min_or_max='max',           # 'max' to fit absorption data, 'min' for emission
        x1=3, # spline points for initial fit maximum value fit -- or an interval for evenly spaced
        x2=None, # spline points for final least squares fit -- or an interval for evenly spaced, None to skip
        trim=(0,1), # fractional interval of data to keep ordered by initial fit residual
        figure=None,            # a Figure to plot
        spline_order=3,
):
    """Initially fit background of a noisy spectrum to a spline fixed to
    maximum (minimum) values in an interval around points x1.  Then
    select data in trim interval based on the resulting residual
    error. Then least-squares spline-fit trimmed data at points x2 (or
    uniform grid if this is an interval)."""
    ## first estimate remove polynomial and discard points
    yfit = fit_spline_to_extrema(x,y,'max',x1,1/4,order=spline_order)
    yresidual = y-yfit
    ## trim residual maxima and minima 
    itrim = np.full(x.shape,False)
    itrim[int(len(x)*trim[0]):int(len(x)*trim[1])] = True
    itrim = itrim[np.argsort(np.argsort(yresidual))] # put in x-order
    itrim[0] = itrim[-1] = True                      # do not trim endpoints
    ## get fitted statistics
    μ,σ = fit_normal_distribution(yresidual[itrim])
    ## refit trimmed data
    if x2 is not None:
        xspline,yspline = fit_least_squares_spline(x[itrim],y[itrim],x2)
        yfit = spline(xspline,yspline,x,order=spline_order)
    ## adjust for missing noise due to trimming
    yfit += μ
    if figure is not None:
        ## plot somem stuff
        ax0 = subplot(0,fig=figure)
        ax0.plot(x,y)
        ax0.plot(x[itrim],y[itrim])
        ax0.plot(x,yfit,lw=3)
        ax1 = subplot(1,fig=figure)
        n,b,p = ax1.hist(yresidual[itrim],max(10,int(len(y)/200)),density=True)
        b = (b[1:]+b[:-1])/2
        t = normal_distribution(b,μ,σ)
        ax1.plot(b,t/np.mean(t)*np.mean(n))
        ax1.set_title('Fitted normal distribution')
    return yfit

def fit_least_squares_spline(
        x,                      # x data in spetrum -- sorted
        y,                      # y data in spectra
        xspline=10,             # spline points or an interval
        order=3,
):
    """Fit least squares spline coefficients at xspline to (x,y)."""
    ## get x spline points
    xbeg,xend = x[0],x[-1]
    if np.isscalar(xspline):
        xspline = np.linspace(xbeg,xend,max(2,int((xend-xbeg)/xspline)))
    xspline = np.asarray(xspline,dtype=float)
    ## get initial y spline points
    yspline = np.array([y[np.argmin(np.abs(x-xsplinei))] for xsplinei in xspline])
    print( f'optimising {len(yspline)} spline points...')
    yspline,dyspline = leastsq(lambda yspline:y-spline(xspline,yspline,x), yspline, yspline*1e-5,)
    return xspline,yspline

def fit_spline_to_extrema(
        x,                      # x data in spetrum - must be sorted
        y,                      # y data in spectra
        fit_min_or_max='max',
        xi=10, # x values to fit spline points, or interval of evenly spaced points
        interval_fraction=1/4,  # select a value in this interval around xi
        order=3,s=0,            # spline parameters
):
    """Fit a spline to data defined at points near xi. Exact points
    are selected as the maximum (minimum) in an interval around xi
    defined as the fraction bounded by neighbouring xi."""
    assert fit_min_or_max in ('min','max')
    ## get xi spline points
    xbeg,xend = x[0],x[-1]
    if np.isscalar(xi):
        xi = np.linspace(xbeg,xend,max(2,int((xend-xbeg)/xi)))
    xi = np.asarray(xi,dtype=float)
    assert np.all(np.sort(xi)==xi),'Spline points not monotonically increasing'
    ## get y spline points
    interval_beg = np.concatenate((xi[0:1], xi[1:]-(xi[1:]-xi[:-1])*interval_fraction))
    interval_end = np.concatenate(((xi[:-1]+(xi[1:]-xi[:-1])*interval_fraction,x[-1:])))
    xspline,yspline = [],[]
    for begi,endi in zip(interval_beg,interval_end):
        if begi>x[-1] or endi<x[0]:
            ## out of bounds of data
            continue
        iinterval = (x>=begi)&(x<=endi)
        if fit_min_or_max == 'min':
            ispline = find(iinterval)[np.argmin(y[iinterval])]
        elif fit_min_or_max == 'max':
            ispline = find(iinterval)[np.argmax(y[iinterval])]
        xspline.append(x[ispline])
        yspline.append(y[ispline])
    # xspline,yspline = np.array(xspline),np.array(yspline)
    xspline[0],xspline[-1] = x[0],x[-1] # ensure endpoints are included
    yf = spline(xspline,yspline,x,order=order,s=s,check_bounds=False)
    return yf

# def localmax(x):
    # """Return array indices of all local (internal) maxima. The first point is returned
    # for adjacent equal points that form a maximum."""
    # j = np.append(np.argwhere(x[1:]!=x[0:-1]),len(x)-1)
    # y = x[j]
    # i = np.squeeze(np.argwhere((y[1:-1]>y[0:-2])&(y[1:-1]>y[2:]))+1)
    # return np.array(j[i],ndmin=1)

# def localmin(x):
    # """Return array indices of all local (internal) minima. The first point is returned
    # for adjacent equal points that form a minimum."""
    # return localmax(-x)

# # def down_sample(x,y,dx):
    # # """Down sample into bins dx wide.
    # # An incomplete final bin is discarded.
    # # Reduce the number of points in y by factor, summing
    # # n-neighbours. Any remaining data for len(y) not a multiple of n is
    # # discarded. If x is given, returns the mean value for each bin, and
    # # return (y,x)."""
    # # if x is None:
        # # return np.array([np.sum(y[i*n:i*n+n]) for i in range(int(len(y)/n))])
    # # else:
        # # return np.array(
            # # [(np.sum(y[i*n:i*n+n]),np.mean(x[i*n:i*n+n])) for i in range(int(len(y)/n))]).transpose()

# def average_to_grid(x,y,xgrid):
    # """Average y on grid-x to a new (sparser) grid. This might be useful
    # when you want to average multiple noisy traces onto a common
    # grid. Splining wont work because of the noise."""
    # i = ~np.isnan(y);x,y = x[i],y[i] # remove bad data, i.e., nans
    # ygrid = np.ones(xgrid.shape)*np.nan
    # ## loop over output grid - SLOW!
    # for i in np.argwhere((xgrid>x[0])&(xgrid<x[-1])):
        # if i==0:
            # ## first point, careful about bounds
            # j = False*np.ones(x.shape,dtype=bool)
        # elif i==len(xgrid)-1:
            # ## first point, careful about bounds
            # j = False*np.ones(x.shape,dtype=bool)
        # else:
            # ## find original data centred around current grid point
            # j = (x>0.5*(xgrid[i]+xgrid[i-1]))&(x<=0.5*(xgrid[i]+xgrid[i+1]))
        # if j.sum()>0:
            # ygrid[i] = y[j].sum()/j.sum()
    # return ygrid

def bin_data(y,n,x=None):
    """Reduce the number of points in y by factor, summing
    n-neighbours. Any remaining data for len(y) not a multiple of n is
    discarded. If x is given, returns the mean value for each bin, and
    return (y,x)."""
    if x is None:
        return np.array([np.sum(y[i*n:i*n+n]) for i in range(int(len(y)/n))])
    else:
        return np.array(
            [(np.sum(y[i*n:i*n+n]),np.mean(x[i*n:i*n+n])) for i in range(int(len(y)/n))]).transpose()

def resample(xin,yin,xout):
    """One particular way to spline or bin (as appropriate) (x,y) data to
    a given xout grid. Trapezoidally-integrated value is preserved."""
    assert np.all(xin==np.unique(xin)),'Input x-data not monotonically increasing.'
    assert all(yin>=0),'Negative cross section in input data'
    assert not np.any(np.isnan(yin)),'NaN cross section in input data'
    assert xout[0]>=xin[0],'Output x minimum less than input.'
    assert xout[-1]<=xin[-1],'Output x maximum greater than input.'
    ## integration region boundary points -- edge points and mid
    ## points of xout
    xbnd = np.concatenate((xout[0:1],(xout[1:]+xout[:-1])/2,xout[-1:]))
    ## linear spline data to original and boundary points
    xfull = np.unique(np.concatenate((xin,xbnd)))
    yfull = spline(xin,yin,xfull,order=1)
    ## indentify boundary pointsin full 
    ibnd = np.searchsorted(xfull,xbnd)
    ## compute trapezoidal cumulative integral 
    ycum = np.concatenate(([0],integrate.cumtrapz(yfull,xfull)))
    ## output cross section points are integrated values between
    ## bounds
    yout = (ycum[ibnd[1:]]-ycum[ibnd[:-1]])/(xfull[ibnd[1:]]-xfull[ibnd[:-1]])
    return yout

def resample_out_of_bounds_to_zero(xin,yin,xout):
    """Like resample but can handle out of bounds by setting this to
    zero."""
    yout = np.zeros(xout.shape,dtype=float)
    i = (xout>=xin[0])&(xout<=xin[-1])
    if sum(i)>0:
        yout[i] = resample(xin,yin,xout[i])
    return yout

# def locate_peaks(
        # y,x=None,
        # minX=0.,
        # minY=0.,
        # fitMaxima=True, fitMinima=False,
        # plotResult=False,
        # fitSpline=False,
        # search_width=1,
        # convolve_with_gaussian_of_width=None,
# ):
    # """Find the maxima, minima, or both of a data series. If x is not
    # specified, then replace with indices.\n
    # Points closer than minX will be reduced to one extremum, points
    # less than minY*noise above the mean will be rejected. The mean is
    # determined from a tensioned spline, unless fitSpline=False.\n
    # If plotResult then issue matplotlib commands on the current axis.\n\n
    # """
    # ## x defaults to indices
    # if x is None: x = np.arange(len(y))
    # ## sort by x
    # x = np.array(x)
    # y = np.array(y)
    # i = np.argsort(x)
    # x,y = x[i],y[i]
    # ## smooth with gaussianconvolution if requested
    # if convolve_with_gaussian_of_width is not None:
        # y = convolve_with_gaussian(x,y,convolve_with_gaussian_of_width,regrid_if_necessary=True)
    # ## fit smoothed spline if required
    # if fitSpline:
        # fs = spline(x,y,x,s=1)
        # ys = y-fs
    # else:
        # fs = np.zeros(y.shape)
        # ys = y
    # ## fit up or down, or both
    # if fitMaxima and fitMinima:
        # ys = np.abs(ys)
    # elif fitMinima:
        # ys = -ys
    # ## if miny!=0 reject those too close to the noise
    # # if minY!=0:
        # # minY = minY*np.std(ys)
        # # i =  ys>minY
        # # # ys[i] = 0.
        # # x,y,ys,fs = x[i],y[i],ys[i],fs[i]
        # # # x,ys = x[i],ys[i]
    # ## find local maxima
    # i =  list(find( (ys[1:-1]>ys[0:-2]) & (ys[1:-1]>ys[2:]) )+1)
    # ## find equal neighbouring points that make a local maximum
    # # j = list(np.argwhere(ys[1:]==ys[:-1]).squeeze())
    # j = list(find(ys[1:]==ys[:-1]))
    # while len(j)>0:
        # jj = j.pop(0)
        # kk = jj + 1
        # if kk+1>=len(ys): break
        # while ys[kk+1]==ys[jj]:
            # j.pop(0)
            # kk = kk+1
            # if kk+1>=len(ys):break
        # if jj==0: continue
        # if kk+1>=len(ys): continue
        # if (ys[jj]>ys[jj-1])&(ys[kk]>ys[kk+1]):
            # i.append(int((jj+kk)/2.))
    # i = np.sort(np.array(i))
    # ## if minx!=0 reject one of each pair which are too close to one
    # ## if miny!=0 reject those too close to the noise
    # if minY!=0:
        # minY = minY*np.std(ys)
        # i = [ii for ii in i if ys[ii]>minY]
    # ## another, taking the highest
    # if minX!=0:
        # while True:
            # jj = find(np.diff(x[i]) < minX)
            # if len(jj)==0: break
            # for j in jj:
                # if ys[j]>ys[j+1]:
                    # i[j+1] = -1
                # else:
                    # i[j] = -1
            # i = [ii for ii in i if ii!=-1]
    # ## plot
    # if plotResult:
        # fig = plt.gcf()
        # ax = fig.gca()
        # ax.plot(x,y,color='red')
        # if minY!=0:
            # ax.plot(x,fs+minY,color='lightgreen')
            # ax.plot(x,fs-minY,color='lightgreen')
        # ax.plot(x[i],y[i],marker='o',ls='',color='blue',mew=2)
    # ## return
    # return np.array(i,dtype=int)

def find_peaks(
        y,
        x=None,                    # if x is None will use index
        peak_type='maxima',     # can be 'maxima', 'minima', 'both'
        peak_min=None, # minimum height of trough between adjacent peaks as a fraction of the lowest neighbouring peak height. I.e., 0.9 would be a very shallow trough.
        ybeg = None,       # peaks below this will be ignored
        yend = None,      # peaks below this will be ignored
        xbeg = None,       # peaks below this will be ignored
        xend = None,      # peaks below this will be ignored
        x_minimimum_separation = None, # two peaks closer than this will be reduced to the taller
        return_coords=True,
):
    """A reworked version of locate_peaks with difference features. Does
not attempt to fit the background with a tensioned spline, isntead
this should already be reduced to zero for the fractional_trough_depth
part of the algorithm to work. """
    ## find both maxima and minima
    assert peak_type in ('maxima', 'minima', 'both')
    if peak_type=='both':
        maxima = find_peaks(y,x,'maxima',fractional_trough_depth,ybeg,yend,xbeg,xend,x_minimimum_separation)
        minima = find_peaks(y,x,'minima',fractional_trough_depth,ybeg,yend,xbeg,xend,x_minimimum_separation)
        return np.concatenate(np.sort(maxima,minima))
    if ybeg is None:
        ybeg = -np.inf
    if yend is None:
        yend = np.inf
    ## get data in correct array format
    y = np.array(y,ndmin=1)             # ensure y is an array
    if x is None:
        x = np.arange(len(y)) # default x to index
    assert all(np.diff(x)>0), 'Data not sorted or unique with respect to x.'
    ## in case of minima search
    if peak_type=='minima':
        y *= -1
        ybeg,yend = -1*yend,-1*ybeg
    ## shift for conveniences
    shift = np.min(y)
    y -= shift
    ybeg -= shift
    yend -= shift
    ## find all peaks
    ipeak = find((y[:-2]<=y[1:-1])&(y[2:]<=y[1:-1]))+1
    ## limit to minima/maxima
    if ybeg is not None:
        ipeak = ipeak[y[ipeak]>=ybeg]
    if yend is not None:
        ipeak = ipeak[y[ipeak]<=yend]
    if xbeg is not None:
        ipeak = ipeak[x[ipeak]>=xbeg]
    if xend is not None:
        ipeak = ipeak[x[ipeak]<=xend]
    ## compare with next point to see if trough is deep enough to
    ## count as tow peaks. If not keep the tallest peak.
    if peak_min is not None:
        ipeak = list(ipeak)
        i = 0
        while i < (len(ipeak)-1):
            ## index of minimum between maxima
            j = ipeak[i] + np.argmin(y[ipeak[i]:ipeak[i+1]+1])
            
            if (min(y[ipeak[i]],y[ipeak[i+1]])-y[j]) > peak_min:
                ## no problem, proceed to next maxima
                i += 1
            else:
                ## not happy, delete the lowest height maxima
                if y[ipeak[i]]>y[ipeak[i+1]]:
                    ipeak.pop(i+1)
                else:
                    ipeak.pop(i)
        ipeak = np.array(ipeak)
    ## if any peaks are closer than x_minimimum_separation then keep the taller.
    if x_minimimum_separation is not None:
        while True:
            jj = find(np.diff(x[ipeak]) < x_minimimum_separation)
            if len(jj)==0: break
            for j in jj:
                if y[j]>y[j+1]:
                    ipeak[j+1] = -1
                else:
                    ipeak[j] = -1
            ipeak = [ii for ii in ipeak if ii!=-1]
    ipeak = np.array(ipeak)
    return ipeak


#################
## sympy stuff ##
#################

# @functools.lru_cache
# def cached_pycode(*args,**kwargs):
    # return(pycode(*args,**kwargs))

@functools.lru_cache
def lambdify_sympy_expression(
        sympy_expression,
        *args,                  # strings denoting input arguments of lambda function
        **kwargs,               # key=val, set to kwarg arguments of lambda function
): 
    """An alternative to sympy lambdify.  ."""
    ## make into a python string
    # t =  cached_pycode(sympy_expression,fully_qualified_modules=False)
    t =  pycode(sympy_expression,fully_qualified_modules=False)
    ## replace math functions
    for t0,t1 in (('sqrt','np.sqrt'),):
        t = t.replace(t0,t1)
    ## build argument list into expression
    arglist = list(args) + [f'{key}={repr(val)}' for key,val in kwargs.items()] 
    eval_expression = f'lambda {",".join(arglist)},**kwargs: np.asarray({t},dtype=float)' # NOTE: includes **kwargs
    return eval(eval_expression)
