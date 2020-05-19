################
## decorators ##
################

def frozendict_args(f):
    """A decorator that aims to be like functools.lru_cache except it
    deepcopies the retrun value, so that mutable object can safely
    cached. """
    from frozendict import frozendict
    def fnew(*args,**kwargs):
        return(f(*[frozendict(arg) if isinstance(arg,dict) else arg for arg in args],
                 **{key:(frozendict(val) if isinstance(val,dict) else val) for key,val in kwargs.items()}))
    return(fnew)

def lru_cache_copy(f,*lru_cache_args,**lru_cache_kwargs):
    """A decorator that aims to be like functools.lru_cache except it
    deepcopies the retrun value, so that mutable object can safely
    cached. """
    import functools
    import copy
    @functools.lru_cache(*lru_cache_args,**lru_cache_kwargs)    
    def fcached(*args,**kwargs):
        return(f(*args,**kwargs))
    def fcached_copied(*args,**kwargs):
        return(copy.deepcopy(fcached(*args,**kwargs)))
    return(fcached_copied)


############################
## mathematical functions ##
############################

def kronecker_delta(x,y):
    """1 if x==y else 0."""
    if np.isscalar(x) and np.isscalar(y): return(1 if x==y else 0) # scalar case
    if np.isscalar(x) and not np.isscalar(y): x,y = y,x            # one vector, get in right order
    retval = np.zeros(x.shape)
    retval[x==y] = 1
    return(retval)              # vector case

#########
## myc ##
#########

def myc_test_function_array(x):
    """An interface to one of the c-code functions."""
    ## import myc
    import ctypes
    myc = ctypes.CDLL(os.path.abspath("myc.so"))

    ## copy x if it is the wrong type, otherwise use as is -- risk of
    ## mutation in c code
    if x.dtype != np.dtype(float) or x.ndim!=1:
        x = np.array(x,dtype=ctypes.c_double,ndmin=1)
    assert x.ndim==1
    ## generate return array
    retval = np.empty(x.shape,dtype=ctypes.c_double)
    ## call function -- operating on return array in place
    myc.test_function_array(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        retval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(x))
    return(retval)


################################
## save / load convert arrays ##
################################
def expand_path(path):
    """Shortcut to os.path.expanduser(path). Returns a
    single file only, first in list of matching."""
    import os
    return os.path.expanduser(path)

def file_to_array_unpack(*args,**kwargs):
    """Same as file_to_array but unpack data by default."""
    kwargs.setdefault('unpack',True)
    return(file_to_array(*args,**kwargs))

def date_string():
    """Get string representing date in ISO format."""
    import datetime
    t = datetime.datetime.now()
    return('-'.join([str(t.year),format(t.month,'02d'),format(t.day,'02d')]))

def txt_to_array_via_hdf5(path,**kwargs):
    """Loads ASCII text to an array, converting to hdf5 on the way
    because it is faster.  Also deletes commented rows. MASSIVE
    HACK."""
    import os,tempfile,subprocess
    ## default comment char is #
    kwargs.setdefault('comments','#')
    comments = kwargs.pop('comments')
    path = os.path.expanduser(path)
    ## expand path if possible and ensure exists - or else awk will hang
    if not os.path.lexists(path):
        raise IOError(1,'file does not exist',path)
    tmpfile=tempfile.NamedTemporaryFile()
    command = 'cat "'+path+'" | sed "/^ *'+comments+'/d" | h5fromtxt '+tmpfile.name
    (status,output)=subprocess.getstatusoutput(command)
    assert status==0,"Conversion to hdf5 failed:\n"+output
    retval = hdf5_to_array(tmpfile.name,**kwargs)
    return retval

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

def hdf5_to_dict(filename_or_hdf5_object):
    """Load all elements in hdf5 into a dictionary. Groups define
    subdictionaries."""
    ## decide if filename and open -- STR TEST MAY NOT BE A GOOD TEST
    if isinstance(filename_or_hdf5_object,str):
        filename_or_hdf5_object = h5py.File(expand_path(filename_or_hdf5_object),'r')
    retval_dict = {}            # the output data
    ## recurse through object loading data 
    for key in filename_or_hdf5_object.keys():
        ## make a new subdict recursively
        if isinstance(filename_or_hdf5_object[key],h5py.Dataset):
            retval_dict[str(key)] = filename_or_hdf5_object[key][()]
        ## add data
        else:
            retval_dict[str(key)] = hdf5_to_dict(filename_or_hdf5_object[key])
    return(retval_dict)

def print_hdf5_tree(filename_or_hdf5_object,make_print=True):
    """Print out a tree of an hdf5 object or file."""
    if not isinstance(filename_or_hdf5_object,h5py.HLObject):
        filename_or_hdf5_object = h5py.File(expand_path(filename_or_hdf5_object),'r')
    retval = []
    for key in filename_or_hdf5_object.keys():
        if isinstance(filename_or_hdf5_object[key],h5py.Dataset):
            retval.append('['+repr(key)+']')
        else:
            sub_retval = hdf5_print_tree(filename_or_hdf5_object[key],make_print=False)
            retval.extend(['['+repr(key)+']'+t for t in sub_retval])
    if make_print:
        ## original call -- print
        print('\n'.join(retval))
    else:
        ## recursive call -- return data
        return(retval)

def print_dict_tree(d):
    print(format_dict_key_tree(d))
    
def format_dict_key_tree(d,prefix='└─ '):
    """Print out a tree of a dicts keys, not the values though."""
    s = []
    for i,key in enumerate(d):
            # for t in range(depth): prefix = '│  '+prefix
        s.append(prefix+key)
        if hasattr(d[key],'keys'):
            s.append(format_dict_key_tree(d[key],'    '+prefix))
    return('\n'.join(s))

def pprint_dict_recursively(d,max_indent_level=None,indent_level=0):
    """Actual works on anything with an 'items' method."""
    indent = ''.join(['  ' for t in range(indent_level)])
    for (key,val) in d.items():
        if hasattr(val,'items') and (max_indent_level is None or indent_level < max_indent_level):
            print(indent+str(key))
            pprint_dict_recursively(val,max_indent_level,indent_level=indent_level+1)
        else:
            print(indent+str(key)+': '+repr(val))

def walk_dict_items(d,maxdepth=np.inf):
    """A generator that walks through dictionary d (depth first) and any
    subdictionary returning keys and values ones by one."""
    if maxdepth<0: return
    for (key,val) in d.items():
        yield(key,val)
        if isinstance(val,dict):
            for tkey,tval in walk_dict_items(val,maxdepth=maxdepth-1):
                yield(tkey,tval)

def recarray_to_dict(ra):
    """Convert a record array to a dictionary. There may be a builtin
    way to do this."""
    retval = collections.OrderedDict()
    for key in ra.dtype.names:
        retval[key] = np.array(ra[key])
    return(retval)
    # return({key:np.array(ra[key]) for key in ra.dtype.names})

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

def append_fields_to_recarray(recarray,**keyvals):
    """Add a new field of name name to recarray. All values of keyvals must
    be the same length as recarray."""
    d = recarray_to_dict(recarray)
    for key,val in keyvals.items():
        if np.isscalar(val): val = np.full(recarray.shape,val) # expand scalara data
        d[key] = val
    return(dict_to_recarray(d))

def recarray_concatenate_fields(*recarrays):
    """Join recarrays into one. Both must be the same length.  Common keys
    causes an error."""
    for t in recarrays: assert len(t)==len(recarrays[0]),'Recarrays not all same length' # check lengths equal
    ## check keys are unique
    all_keys = []
    for t in recarrays: all_keys.extend(t.dtype.names)
    assert len(all_keys)==len(np.unique(all_keys)),f'keys not unique: {repr(all_keys)}'
    ## join into one recarray
    keys_vals = collections.OrderedDict()
    for t in recarrays:
        for key in t.dtype.names:
            keys_vals[key] = t[key]
    return(dict_to_recarray(keys_vals))

def concatenate_recarrays_unify_dtype(recarrays_list,casting='safe',**concatenate_kwargs):
    """Concatenate recarrays, but first try and align dtype to avoid
    promotion errors. Various heuristic design decisions in here."""
    assert(len(recarrays_list)>0),'one at recarray to contenate at least'
    ## get all dtypes as a list of list of strings to manipulate
    dtypes = [
        [(t.names[i],t[i].str) for i in range(len(t))]
        for t in [t.dtype for t in recarrays_list]]
    ## determine a unifying dtype to cast all arrays to
    dtype_cast = dtypes[0]      # initialise to first
    for dtype in dtypes[1:]:    # update with others
        if dtype==dtype_cast: continue # already compliant
        for idtype,((namei,stri),(namej,strj)) in enumerate(zip(dtype,dtype_cast)): # check individual dtypes, order must be the same
            assert namei==namej,f'Names do not match {repr(namei)} {repr(namej)}' # names do not match
            if stri==strj:
                continue
            elif stri[0:2]=='<U' and strj[0:2]=='<U': # if strings are different lengths, cast to the longer
                max_length = max(int(stri[2:]),int(strj[2:]))
                dtype_cast[idtype] = (namei,'<U'+str(max_length))
            elif (stri[0:2]=='<i' and strj[0:2]=='<f') or (strj[0:2]=='<i' and stri[0:2]=='<f'): # i and f, cast to f8
                dtype_cast[idtype] = (namei,'<f8')
            else:               # this combination not implemented
                raise Exception(f"dtype rectification not implemented: {repr((namei,stri))} {repr((namej,strj))}")
    return(                     # return concatenated array
        np.concatenate(         # concatenate 
            [t.astype(dtype_cast,casting=casting) for t in recarrays_list], # cast individudal arrays
            **concatenate_kwargs))

def recarray_remove(recarray,*keys):
    """Return a copy of recarray with keys removed."""
    return(dict_to_recarray({key:val for key,val in recarray_to_dict(recarray).items() if key not in keys}))

def append_to_recarray(recarray,**kwargs):
    """Make a new and longer recarray. kwargs must match the complete dtype
    of the recarray.  WOULDN'T CONCATENAT WORK WELL ENOUGH?"""
    assert set(recarray.dtype.names)==set(kwargs), 'dtype.names in old and new recarrays do not match. Old: '+repr(set(recarray.dtype.names))+' new: '+repr(set(kwargs))
    new_recarray = np.rec.fromarrays([kwargs[key] for key in recarray.dtype.names],dtype=recarray.dtype)
    return(np.append(recarray,new_recarray))

def recarray_to_hdf5(filename,recarray,header=None,**kwargs):
    """Used dict_to_hdf5 to save a recarray to hdf5 file."""
    d = recarray_to_dict(recarray)
    if header is not None: d['README'] = str(header)
    return(dict_to_hdf5(filename,d,**kwargs))

def kwargs_to_hdf5(filename,**kwargs):
    return(dict_to_hdf5(filename,dict(**kwargs)))

def dict_to_hdf5(
        filename,
        dictionary,
        keys=None,
        overwrite=True,
        compress_data=True,
):
    """Save all elements of a dictionary as datasets in an hdf5 file.
    Compression options a la h5py, e.g., 'gzip' or True for defaults"""
    import h5py
    import os
    filename = expand_path(filename)
    mkdir_if_necessary(dirname(filename)) # make leading directories if not currently there
    if keys is None:
        keys = list(dictionary.keys()) # default add all keys to datasets
    if overwrite:
        if os.path.isdir(filename):
            raise Exception("Is directory: "+filename)
        if os.path.lexists(filename):
            os.unlink(filename)
    else:
        if os.path.lexists(filename):
            raise Exception("File exists: "+filename)
    f = h5py.File(filename,mode='w')
    for key in keys:
        kwargs = {}
        if compress_data and not np.isscalar(dictionary[key]):
            kwargs['compression'] = "gzip"
            kwargs['compression_opts'] = 9
        data = np.asarray(dictionary[key])
        ## deal with missing unicode type in hdft http://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
        if not np.isscalar(data) and data.dtype.kind=='U':
            data = np.array(data, dtype=h5py.string_dtype(encoding='utf-8'))
        f.create_dataset(key,data=data,**kwargs)
    f.close()
    
def kwargs_to_directory(directory, **dict_to_directory_kwargs,):
    dict_to_directory(directory,dict_to_directory_kwargs)
data_to_directory = kwargs_to_directory # deprecated
    
def dict_to_directory(
        directory,
        dictionary,
        array_format='h5',
        remove_string_margin=True,
        make_directory=True
):
    """Create a directory and save contents of dictionary into it."""
    if make_directory: mkdir_if_necessary(directory)
    for key,val in dictionary.items():
        ## save strings to text files
        if isinstance(val,str):
            if remove_string_margin:
                val = re.sub(r'(^|\n)\s+\b',r'\1',val) # delete white space at beginning of all lines
            string_to_file(directory+'/'+str(key),val)
        ## save numpy arrays in binary format, or hdf5, or text file
        elif isinstance(val,np.ndarray):
            if   array_format == 'npy':  array_to_file(directory+'/'+str(key)+'.npy',val)
            elif array_format == 'npz':  array_to_file(directory+'/'+str(key)+'.npz',val)
            elif array_format == 'text': array_to_file(directory+'/'+str(key),val)
            elif array_format == 'h5':   array_to_file(directory+'/'+str(key)+'.h5',val)
            else:   raise Exception('array_format must be one of "npy", "npz", "text", "hdf5"')
        ## save dictionaries as subdirectories
        elif isinstance(val,dict):
            dict_to_directory(directory+'/'+str(key),val,array_format)
        ##
        else:
            raise Exception('Do not know how to save: key: '+repr(key)+' val: '+repr(val))

def directory_to_dict(directory):
    """Load all contents of a directory into a dictiionary, recursive."""
    directory = expand_path(directory)
    directory = re.sub(r'(.*[^/])/*',r'\1',directory) # remove trailing /
    retval = {}
    for filename in myglob(directory+'/*'):
        filename = filename[len(directory)+1:]
        extension = os.path.splitext(filename)[1]
        ## load subdirectories as dictionaries
        if os.path.isdir(directory+'/'+filename):
            retval[filename] = directory_to_dict(directory+'/'+filename)
        ## load binary data
        if extension in ('.npy','.h5','.hdf5'):
            # retval[filename[:-4]] = np.load(diarectory+'/'+filename)
            retval[filename[:-len(extension)]] = file_to_array(directory+'/'+filename)
        ## read README as string
        elif filename in ('README',):
            retval[filename] = file_to_string(directory+'/'+filename)
        ## else assume text data
        elif filename in ('README',):
            retval[filename] = file_to_array(filename)
    return(retval)


class Data_Directory:
    """Data is stored in a directory and accessed by key."""
    
    def __init__(self,directory_path):
        self.root = expand_path(re.sub(r'(.*[^/])/*',r'\1',directory_path))
        self._cache = {}

    def __getitem__(self,key):
        ## return from cache if possible
        if key in self._cache: return(self._cache[key])
        ## get full filename. If it does not exist look for a version
        ## with a unique extension
        filename = f'{self.root}/{key}'
        if not os.path.exists(filename):
            try:
                filename = glob_unique(f'{self.root}/{key}.*')
            except FileNotFoundError:
                raise KeyError(f"Cannot find file with or without extension: {self.root}/{key}")
        ## if recursive subdir access
        if '/' in key:
            i = key.find('/')
            retval = self[key[:i]][key[i+1:]]
        ## load subdirectories as dictionaries
        elif os.path.isdir(filename):
            retval = Data_Directory(filename)
        ## read README as string
        elif filename in ('README',):
            retval = file_to_string(filename)
        ## load array data
        else:
            retval = file_to_array(filename)
        ## save to cache and return
        self._cache[key] = retval
        return(retval)

    def keys_deep(self):
        """Keys expanded to all levels."""
        retval = []
        for filename in myglob(f'{self.root}/*'):
            key = filename[len(self.root)+1:]
            if os.path.isdir(filename):
                retval.extend([f'{key}/{t}' for t in self[key].keys()])
            else:
                retval.append(key)
        return(retval)

    def keys(self):
        """Keys in top level."""
        return([t[len(self.root)+1:] for t in myglob(f'{self.root}/*')])

    def __len__(self):
        return(len(self.keys()))

    def __str__(self):
        return(str(self.keys()))


def pick_function_kwargs(kwargs,function):
    """Find which kwargs belong to this function, return as a dict."""
    import inspect
    ## get a list of acceptable args, special case for plot (and others?)
    if function.__name__ == 'plot':    # special case for plot since args not in function definition line
        args = ('agg_filter','alpha','animated','antialiased','aa','clip_box','clip_on','clip_path','color','c','contains','dash_capstyle','dash_joinstyle','dashes','drawstyle','figure','fillstyle','gid','label','linestyle','ls','linewidth','lw','marker','markeredgecolor','mec','markeredgewidth','mew','markerfacecolor','mfc','markerfacecoloralt','mfcalt','markersize','ms','markevery','path_effects','picker','pickradius','rasterized','sketch_params','snap','solid_capstyle','solid_joinstyle','transform','url','visible','xdata','ydata','zorder')
    else:
        args,varargs,keywords,defaults = inspect.getargspec(function)
    ## pick these into a new dict
    picked_kwargs = collections.OrderedDict()
    for key,val in kwargs.items():
        if key in args: picked_kwargs[key] = val
    other_kwargs = collections.OrderedDict()
    for key,val in kwargs.items():
        if key not in picked_kwargs:
            other_kwargs[key] = val
    return(picked_kwargs,other_kwargs)

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
        j = find(diff==0)
        warnings.warn('Parameter has no effect: parameter indices: '+str(j)+' with fitted values: '+str(x[j]))
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
    """Calculate rms, ignoring NaN data."""
    return np.sqrt(np.nanmean(np.array(x)**2))


########################
## file manipulations ##
########################

def tmpfile():
    """Create a secure named temporary file which will be
    automatically deleted. Object is returned."""
    return tempfile.NamedTemporaryFile()

def tmpdir():
    """Create a temporary directory which will not be
    automatically deleted. Pathname is returned."""
    return tempfile.mkdtemp()

def cp(src,dest):
    return(shutil.copy(expand_path(src),expand_path(dest)))

def cptree(src,dest):
    """"""
    return(shutil.copytree(expand_path(src),expand_path(dest)))

def trash(filename):
    """Put file in the trash can. Silence on error. No filename expansion."""
    import shlex
    os.system('trash-put '+shlex.quote(filename)+' > /dev/null 2>&1')

def mkdir(*directories,trash_existing_directory=False):
    """Create directory tree (or multiple) if it doesn't exist."""
    ## if multiple loop through them
    if len(directories)>1:
        for directory in directories:
            mkdir_if_necessary(directory)
        return
    ## if single then do it
    directory = expand_path(directories[0])
    if os.path.isdir(directory):
        if trash_existing_directory: # deletes contents--keeps directory
            for t in myglob(f'{directory}/*'):
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

mkdir_if_necessary = mkdir      # deprecated name

def randn(shape=None):
    """Return a unit standard deviation normally distributed random
    float, or array of given shape if provided."""
    if shape == None:
        return float(np.random.standard_normal((1)))
    else:
        return np.random.standard_normal(shape)

def percentDifference(x,y):
    """Calculate percent difference, i.e. (x-y)/mean(x,y)."""
    return 100.*2.*(x-y)/(x+y)

def notNan(arg):
    """Return part of arg that is not nan."""
    return arg[~np.isnan(arg)]

def bestPermutation(x,y):
    """Find permuation of x that best matches y in terms of
    rms(x-y). A brute force approach"""
    assert len(x)==len(y), 'Inputs must have same shape.'
    x = copy(np.array(x))
    y = copy(np.array(y))
    permutations = itertools.permutations(list(range(len(x))))
    bestRms = np.inf
    bestPermutation = None
    while True:
        try:
            permutation = np.array(next(permutations))
            nextRms = rms(x[permutation]-y)
            if nextRms < bestRms:
                bestPermutation = permutation
                bestRms = nextRms
        except StopIteration:
            break
    return bestPermutation

def minimise_differences(a,b):
    """Returns an index which reorders a to best match b. This is done by
    finding the best possible match, fixing that, then the second etc. So
    may not give the best summed RMS or whatever."""
    assert len(a)==len(b), "Nonmatching array lengths."
    # ## if one of the arrays is incomplete then pad with infs. As a
    # ## side effect of numpy argmin these will retain their current
    # ## ordering, which may not be reliable in the future.
    # if len(a)<len(b):
        # a = np.concatenate((a,np.inf*np.ones(len(b)-len(a)),))
    # if len(b)<len(a):
        # b = np.concatenate((b,np.inf*np.ones(len(a)-len(b)),))
    x,y = np.meshgrid(a,b)          
    t = np.abs(x-y)                    # all possible differences
    ifinal = np.ones(a.shape,dtype=int)*-1          # final sorted ai indices
    ilist = list(range(len(a))) 
    jlist = list(range(len(b)))
    ## find best match, reducing size of t iteratively
    while t.shape[0]>0 and t.shape[1]>0:
        i,j = np.unravel_index(np.argmin(t),t.shape) # minimum difference
        ifinal[ilist[i]] = jlist[j] # save this index
        ilist.pop(i)                # remove the minimum values from further consideration
        jlist.pop(j)                # remove the minimum values from further consideration
        t = t[list(range(0,i))+list(range(i+1,t.shape[0])),:] # shrink array
        t = t[:,list(range(0,j))+list(range(j+1,t.shape[1]))] # shrink array
    return(ifinal)

def user_string():
    """Return string describing user."""
    import getpass
    return getpass.getuser()+'@'+os.uname()[1]

def sum_in_quadrature(*args):
    """returns sqrt(sum(args**2))"""
    return(np.sqrt(sum([np.square(t) for t in args])))
    
def triangle_sum(x,axis=None):
    """Return sqrt of sum of squares.

    If input is an array then calculates according to axis, like other
    numpy functions.  Else, triangle sums all elements of x, vector or
    not (axis kwarg ignored).
    """
    x = np.asarray(x)
    ## Tries to sum as numpy array.
    try:
        return (np.sqrt((x**2).sum(axis=axis)))
    ## Upon failure, sums elements
    except:
        retval = 0.
        for xx in x: retval = retval + xx**2
        return np.sqrt(retval)

def sum_in_quadrature(*args):
    return(np.sqrt(np.sum([t**2 for t in args],axis=0)))

def combine_product_errors(product,factors,errors):
    """Combine uncorrelated normally distributed errors dx, dy, dz etc
    to get error of x*y*z etc"""
    return(product*np.sqrt(np.sum([(error/factor)**2 for (error,factor) in zip(errors,factors)])))

def set_dict_default(dictionary,default_values_dictionary):
    """Copy all keys,vals from default_values_dictionary to dictionary
    unless key is already present in dictionary. """
    for key,val in default_values_dictionary.items():
        if not key in dictionary:
            dictionary[key] = val

def update_dict(dictionary,update_values_dictionary):
    """Update dictionary with all keys,vals in
    update_values_dictionary. Raise error if key in
    update_values_dictionary is not in dictionary."""
    for key,val in update_values_dictionary.items():
        if key in dictionary:
            dictionary[key] = val
        else:
            raise Exception("Updating key not in dictionary: "+repr(key))
        
def safeUpdate(a,b):
    """Update dictionary a from dictionary b. If any keys in b not
    found in a an error is raised. Update of a done in place."""
    i = isin(list(b.keys()),list(a.keys()))
    if not all(i):
        raise Exception('Bad keys: '+ str([key for (key,ii) in zip(list(b.keys()),i) if not ii]))
    a.update(b)
    
def safe_update_attr(a,b):
    """Update attributes of a from dictionary b. If any keys in b not
    found in a an error is raised. Update of a done in place."""
    for (key,val) in list(b.items()):
        if hasattr(a,key):
            setattr(a,key,val)
        else:
            raise AttributeError('Bad attr: '+key)

def index_dict_array(d,i):
    """Assumes all elements of dictionary d are arrays which can be
    indexed by i. Then returns a copy of d containing such
    subarrays."""
    dnew = {}
    for key in d:
        dnew[key] = d[key][i]
    return dnew
        
def match_array_to_regexp(array,regexp,strip=True):
    """Match an array of strings to a regular expression and return a
    boolean array. If strip=True strip whitespace from strings."""
    # assert array.ndim==1, "Currently only implemented for 1D arrays."
    if strip:
        return(np.array([bool(re.match(regexp,t.strip())) for t in array]))
    else:
        return(np.array([bool(re.match(regexp,t)) for t in array]))

def pause(message="Press any key to continue..."):
    """Wait for use to press enter. Not usable outsdie linux."""
    input(message)
    
def getClipboard():
    """Get a string from clipboard."""
    status,output = subprocess.getstatusoutput("xsel --output --clipboard")
    assert status==0, 'error getting clipboard: '+output
    return output

def setClipboard(string):
    """Send a string to clipboard."""
    pipe=os.popen(r'xsel --input --clipboard','w');
    pipe.write(string)
    pipe.close()

def cl(x,fmt='0.15g'):
    """Take array or scalar x and convert to string and put on clipboard."""
    if np.isscalar(x):
        if isinstance(x,str):
            setClipboard(x)
        else:
            setClipboard(format(x,fmt))
    else:
        setClipboard(array_to_string(x,fmt=fmt))
    
def pa():
    """Get string from clipboard. If possible convert to an array."""
    x = getClipboard()
    try:
        return str2array(x)
    except:
        return x

def wget(url):
    """Downloads data from url and returns it."""
    import subprocess
    code,data = subprocess.getstatusoutput("wget --restrict-file-names=nocontrol --quiet -O /dev/stdout '"+url+"'")
    if code!=0: raise Exception("wget failed: "+str(code)+" "+str(data))
    return(data)

def printfmt(*args,fmt='8g'):
    """Print all args with the same format."""
    print(' '.join([format(arg,fmt) for arg in args]))
    
def printcols(*args,fmt='15',labels=None):
    """Print args in with fixed column width. Labels are column
    titles."""
    if not np.iterable(args) or isinstance(args,str): 
        args = (args,)
    else:
        args = [arg for arg in args]
        for i in range(len(args)): 
            if not np.iterable(args[i]) or isinstance(args[i],str):
                args[i] = [args[i]]
    if labels!=None:
        assert len(labels)==len(args),'Not enough/too many labels.'
        print((' '.join(format(t,fmt) for t in labels)))
    print((array2str(list(zip(*args)),fmt=fmt)))

def format_columns(
        data,                   # list or dict (for labels)
        fmt='>10.5g',
        labels=None,
        header=None,
        record_separator='\n',
        field_separator=' ',
        comment_string='# ',
):
    """Print args in with fixed column width. Labels are column
    titles.  NOT QUITE THERE YET"""
    ## if data is dict, reinterpret appropriately
    if hasattr(data,'keys'):
        labels = data.keys()
        data = [data[key] for key in data]
    ## make formats a list as long as data
    if isinstance(fmt,str): fmt = [fmt for t in data]
    ## get string formatting for labels and failed formatting
    fmt_functions = []
    for f in fmt:
        def fcn(val,f=f):
            if isinstance(val,str):
                ## default to a string of that correct length
                r = re.match(r'[^0-9]*([0-9]+)(\.[0-9]+)?[^0-9].*',f)
                return(format(val,'>'+r.groups()[0]+'s'))
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
        records.append(comment_string+field_separator.join([f(label) for (f,label) in zip(fmt_functions,labels)]))
    ## compose formatted data columns
    comment_pad = ''.join([' ' for t in comment_string])
    records.extend([comment_pad+field_separator.join([f(field) for (f,field) in zip(fmt_functions,record)]) for record in zip(*data)])
    return(record_separator.join(records))

def print_columns(data,**kwargs):
    """Print the data into readable columns heuristically."""
    if isinstance(data,np.recarray):
        print(recarray_to_str(data,**kwargs))
    elif isinstance(data,dict):
        print(dict_array_to_str(data,**kwargs))
    else:
        print(format_columns(data,**kwargs))
    
def dict_array_to_str(d,keys=None,fmts=None,**kwargs_for_make_table):
    """Return a string listing the contents of a dictionary made up of
    arrays of the same length. If no keys specified, print all keys."""
    if keys is None: keys = list(d.keys())
    if fmts is None:
        fmts = [max('12',len(key)) for key in keys]
    elif isinstance(fmts,str):
        fmts = [fmts for key in keys]
    columns = [Value_Array(name=key,val=d[key],fmt=fmt) for (key,fmt) in zip(keys,fmts)]
    return make_table(columns,**kwargs_for_make_table)
    
def recarray_to_str(d,*args,**kwargs):
    kwargs.setdefault('headers',d.dtype.names)
    return(tabulate(d,*args,**kwargs))
        
def recarray_to_file(filename,recarray,*args,**kwargs):
    """Output a recarray to a text file."""
    return(string_to_file(filename,recarray_to_string(recarray,*args,**kwargs)))
        
def dict_array_to_file(filename,d,**kwargs_for_dict_array_to_str):
    """Write dictionary of arrays to a file."""
    kwargs_for_dict_array_to_str.setdefault('print_description',False)
    f = open(os.path.expanduser(filename),'w')
    f.write(dict_array_to_str(d,**kwargs_for_dict_array_to_str))
    f.close()
    
def myglob(path='*',regexp=None):
    """Shortcut to glob.glob(os.path.expanduser(path)). Returns a list
    of matching paths. Also sed alphabetically. If re is provided filter names accordingly."""
    retval = sorted(glob.glob(os.path.expanduser(path)))
    if regexp is not None:
        retval = [t for t in retval if re.match(regexp,t)]
    return(retval)

def glob_unique(path):
    """Match glob and return as one file. If zero or more than one file
    matched raise an error."""
    filenames = glob.glob(os.path.expanduser(path))
    if len(filenames)==1:
        return(filenames[0])
    elif len(filenames)==0:
        raise FileNotFoundError('No files matched path: '+repr(path))
    elif len(filenames)>1:
        raise Exception('Multiple files matched path: '+repr(path))

def grep(regexp,string=None,filename=None):
    """Poor approx to grep."""
    if string is not None and filename is None:
        pass
    elif string is None and filename is not None:
        string = file_to_string(filename)
        pass
    else:
        raise Exception('Require string or filename, but not both.')
    return('\n'.join([t for t in string.split('\n') if re.match(regexp,t)]))

def rootname(path,recurse=False):
    """Returns path stripped of leading directories and final
    extension. Set recurse=True to remove all extensions."""
    path = os.path.splitext(os.path.basename(path))[0]
    if not recurse or path.count('.')+path.count('/') == 0:
        return path
    else:
        return rootname(path,recurse=recurse)

def get_filename_extension(path):
    """Return extension, or return None if none."""
    t = os.path.splitext(path)
    if t[1]=='': return(None)
    return(t[1][1:])
    
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

def polyfit(x,y,dy=None,order=0,fixed=None,extended_output=True,
            print_output=False,plot_output=False,return_style='dict',
            error_on_missing_dy=True,**plotkwargs):
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
    ## calc extra information
    if extended_output or print_output or plot_output:
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
    if print_output:
        print(('\n'.join(
            ['             p             dp']
            +[format(a,'14.7e')+' '+format(b,'14.7e') for (a,b) in zip(p,dp)]
            +['chisq: '+str(chisq),'chisqprob: '+str(chisqprob),
              'rms: '+str(rms(residuals)),
              'max_abs_residual: '+str(abs(residuals).max())]
            )))
    ## a nice plot
    if plot_output:
        ax=plt.gca()
        ax.errorbar(x,residuals,dy,**plotkwargs) # plot residual
    ## what to return
    if extended_output:
        if return_style=='list':
            return(p,dp,f,residuals,chisqprob)
        elif return_style=='dict':
            return(dict(x=xin,y=yin,dy=dyin,p=p,dp=dp,yf=f(xin),f=f,residuals=residuals,
                        chisqprob=chisqprob,chisq=chisq,chisqnorm=chisqnorm,fixed=fixed))
        else: raise Exception()
    else:
        return(p)

def dot(x,y):
    """Like numpy dot but sums over last dimension of first argument
    and first dimension of last argument."""
    return np.tensordot(x,y,(-1,0))

def ensure_iterable(x):
    """If input is not iterable enclose it as a list."""
    if np.isscalar(x): 
        return (x,)
    else: 
        return x

def as_scalar_or_first_value(x):
    """Return x if scalar, else return its first value."""
    if np.isscalar(x): 
        return(x)
    else: 
        return(x[0])

def flip(x):
    """ 
    Flipud 1D arrays or 2D arrays where 2nd dimension is length 1.
    Fliplr 2D arrays where first dimension is length 1.
    """
    if x.ndim == 1:
        return np.flipud(x)
    if x.ndim==2:
        if shape(x)[0]==1: return np.fliplr(x)
        if shape(x)[1]==1: return np.flipud(x)
    raise Exception("Could not flip array, shape is wrong probably.")

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

def splinef(xi,yi,s=0,order=3,sort_data=True):
    """Return spline function for points (xi,yi). Will return order
    or (less if fewer points). Sort data for convenience (takes time)."""
    order = min(order,len(xi)-1)
    xi,yi = np.array(xi),np.array(yi)
    if sort_data:
        i = np.argsort(xi)
        xi,yi = xi[i],yi[i]
    return interpolate.UnivariateSpline(xi,yi,k=order,s=s)
    
def spline_with_smooth_ends(xi,yi,x):
    """Evaluate spline defined by xi,yi at x. The first and last
    intevals defined by xi are replaced with a 5th order polynomial
    with C2 continuity with internal points and endpoint derivatives
    set to zero.  Smooth end intervals by substituting a fifth order
    polynomial that has 1-2nd derivatives zero at it's outer boundary
    and matches the spline at the inner boundary.
    Second output is the spline function without the edge smoothing.
    """
    from scipy import interpolate, linalg, integrate
    xi = np.squeeze(xi);yi=np.squeeze(yi);x=np.squeeze(x)
    ## generate spline function
    spline_function=interpolate.UnivariateSpline(xi,yi,s=0)
    ## evaluate spline function
    y=spline_function(x)
    ## change last interval to have smooth finish
    ## internal point spline coeffs
    a = spline_function.derivatives(xi[-2]) 
    b = spline_function.derivatives(xi[-1]) 
    ## range of last interval
    i=np.argwhere((x>=xi[-2])&(x<=xi[-1])) 
    ## new poly coordinate - range [0,1] -need to adjust derivs
    L = (x[i[-1]]-x[i[0]])
    ab = (x[i]-x[i[0]])/L
    a[1] = a[1]*L
    a[2] = a[2]*L**2
    ## calc 5th order poly with desired properties
    A = np.matrix([[1,1,1],[3,4,5],[6,12,20]])
    z = np.matrix([[b[0]-a[0]-a[1]-a[2]],[-a[1]-2*a[2]],[-a[2]]])
    v=linalg.inv(A)*z
    ## new polynomial coefficients
    anew = np.arange(0.,6.)
    anew[0:3] = a[0:3]
    anew[3:6] = np.squeeze(np.array(v))
    ## substitute into potential array
    y[i]=np.polyval(flip(anew),ab)
    ##
    ## change first interval to have smooth finish
    a = spline_function.derivatives(xi[1]) 
    b = spline_function.derivatives(xi[0]) 
    ## range of last interval
    i=np.argwhere((x>=xi[0])&(x<=xi[1])) 
    ## new poly coordinate - range [0,1] -need to adjust derivs
    L = (x[i[-1]]-x[i[0]])
    ab = 1.-(x[i]-x[i[0]])/L
    a[1] = -a[1]*L
    a[2] = a[2]*L**2
    ## calc 5th order poly with desired properties
    A = np.matrix([[1,1,1],[3,4,5],[6,12,20]])
    z = np.matrix([[b[0]-a[0]-a[1]-a[2]],[-a[1]-2*a[2]],[-a[2]]])
    v=linalg.inv(A)*z
    ## new polynomial coefficients
    anew = np.arange(0.,6.)
    anew[0:3] = a[0:3]
    anew[3:6] = np.squeeze(np.array(v))
    ## substitute into potential array
    y[i]=np.polyval(flip(anew),ab)
    ## 
    return(y)

def piecewise_linear_interpolation_and_extrapolation(xa,ya,x):
    """Linearly interpolate and extrapolate) points xa and ya over domain x."""
    y = np.zeros(x.shape,dtype=float)
    ## interpolate
    for (x0,x1,y0,y1) in zip(xa[0:-1],xa[1:],ya[0:-1],ya[1:]):
        p = np.polyfit([x0,x1],[y0,y1],1)
        i = (x>=x0)&(x<=x1)
        if any(i): y[i] = np.polyval(p,x[i])
    ## extrapolate
    p = np.polyfit([xa[0],xa[1]],[ya[0],ya[1]],1)
    i = x<=xa[0]
    if any(i): y[i] = np.polyval(p,x[i])
    p = np.polyfit([xa[-2],xa[-1]],[ya[-2],ya[-1]],1)
    i = x>=xa[-1]
    if any(i): y[i] = np.polyval(p,x[i])
    return(y)

def interpolate_to_mesh(x,y,z,xbins=100,ybins=100):
    """Takes coordinates (x,y,z) and interpolates to a mesh divided
    into bins (xbins,ybins) between min and max of x and y. Retrun
    arrays (x,y,z) of shape (xbins,ybins)."""
    xnew = np.linspace(min(x),max(x),xbins)
    ynew = np.linspace(min(y),max(y),ybins)
    f = interpolate.interp2d(x,y,z,kind='linear')
    znew = f(xnew,ynew)
    xnew,ynew = np.meshgrid(xnew,ynew)
    return(xnew,ynew,znew)

def array_to_txt_via_hdf5(path,*args,**kwargs):
    """Loads ASCII text to an array, converting to hdf5 on the way
    because it is faster.  Also deletes commented rows. MASSIVE
    HACK."""
    path = os.path.expanduser(path)
    tmpfile=tempfile.NamedTemporaryFile()
    array_to_hdf5(tmpfile.name,*args)
    # command = 'cat "'+path+'" |sed "/^ *'+comments+'/d" | h5fromtxt '+tmpfile.name
    command = 'h5totxt -s " " {hdf5file:s} > {txtfile:s}'.format(hdf5file=tmpfile.name,txtfile=path)
    (status,output)=subprocess.getstatusoutput(command)
    assert status==0,"Conversion from hdf5 failed:\n"+output
        
def array_to_hdf5(filename,*args,**kwargs):
    """Column stack arrays in args and save in an hdf5 file. In a
    single data set named 'data'. Overwrites existing files."""
    filename = expand_path(filename)
    kwargs.setdefault('compression',"gzip")
    kwargs.setdefault('compression_opts',9)
    if os.path.exists(filename):
        assert not os.path.isdir(filename),'Wont overwrite directory: '+filename
        os.unlink(filename)
    f = h5py.File(filename,'w')
    f.create_dataset('data',data=np.column_stack(args),**kwargs)
    f.close()

def savetxt(filename,*args,**kwargs):
    """Column-stacks arrays given as *args and saves them to
    filename. 
    A short cut for savetxt(filename,column_stack(args)). Kwargs passed to
    np.savetxt.
    If *args consists of one string, write this to file, ignore
    everything else.
    Also writes to 
    """
    ## This roundabout method is used in case this file is beind
    ## watched by a graphing program which might fail if the writing
    ## takes a long time. I.e. kst2, but probably not actually
    ## important at all
    # tmpfd,tmpfilename  = tempfile.mkstemp()
    fid = open(expand_path(filename),'w')
    if 'header' in kwargs:
        fid.write(kwargs.pop('header')+'\n')
    # np.savetxt(tmpfilename,np.column_stack(args),**kwargs)
    np.savetxt(fid,np.column_stack(args),**kwargs)
    # tmpfd.close()
    # shutil.copyfile(tmpfilename,os.path.expanduser(filename))

def savetxt_append(filename,*args,**kwargs):
    """Open file filename. Append text made from column_stack of remaining args."""
    f = open(filename,'a')
    np.savetxt(f,np.column_stack(args),**kwargs)
    f.close()
savetxtAppend=savetxt_append
save_append=savetxtAppend

def solve_linear_least_squares_symbolic_equations(
        system_of_equations,
        plot_residual=False,
):
    """Solve an overspecified system of linear equations. This is encoded pretry strictly e.g.,:
    1*x +  2*y =  4
    1*x +  3*y =  8
    0*x + -1*y = -3
    Important separators are: newline, =, + , *.
    """
    ## get system of equations
    equations = []
    for t in system_of_equations.split('\n'):
        t = t.split('#')[0]            # eliminate trailling comments
        if len(t.strip())==0: continue # blank line
        equations.append(t)
    ## decode into terms
    Aij,bi,variables = [],[],[]
    for i,equation in enumerate(equations):
        lhs,rhs = equation.split('=')
        for term in lhs.split('+'):
            coeff,var = term.split('*')
            coeff = float(coeff)
            var = var.strip()
            if var not in variables: variables.append(var)
            Aij.append((i,variables.index(var),coeff))
        bi.append(float(rhs))
    ## form matrices
    A = np.zeros((len(equations),len(variables)))
    for i,j,x in Aij: A[i,j] = x
    b = np.array(bi)
    ## solve. If homogeneous assume first variable==1
    homogeneous = True if np.all(b==0) else False
    if homogeneous:
        b = -A[:,0].squeeze()
        A = A[:,1:]
    x = np.dot( np.linalg.inv(np.dot(np.transpose(A),A)),   np.dot(np.transpose(A),b))
    if homogeneous:
        x = np.concatenate(([1],x))
        A = np.column_stack((-b,A))
        b = np.zeros(len(equations))
    if plot_residual:
        fig = plt.gcf()
        fig.clf()
        ax = fig.gca()
        ax.plot(b-np.dot(A,x),marker='o')
    ## return solution dictionary
    return({var:val for var,val in zip(variables,x)})

def leastsq_model(y,f,v,monitor_rms=True,optimise=True,print_result=False,xtol=1e-12):
    """
    Optimises a least squares model. Higher level than leastsq.
    Inputs:
    y -- Array of experimental data to model.
    f -- Function that generates model data. Input arguments are kwargs
         corresponding to keys in dictionary v.
    v -- Dictionary. Each key is a function argument of f which is
         either a real number or an array of real numbers. Each key indexes
         a subdictionary containing elements: 
             pin  -- initial value of real number or array, 
             grad -- gradient calculation step estimate, same dimension size as pin,
             vary -- boolean array whether to vary or fix elements the same dimension 
                     and size as pin.
   monitor_rms -- print whenever rms is lowered during optimisation
   optimise    -- set to False to return function using initial parameters
   Outputs (yf,r,v):
   yf -- model data
   r  -- residual exp - mod data
   v  -- modified v dictionary, elements now contain new elements:
           p     -- fitted parameters
           sigma -- estimated fit uncertainty.
    """
    ## prep v dictionary in some convenient ways
    v = copy(v)            # make local
    vkeys = list(v.keys())          # get here in case key order changes in dictionary
    for key in vkeys:
        if np.iterable(v[key]['pin']):
            v[key]['pin'] = np.array(v[key]['pin'])
            if not np.iterable(v[key]['grad']):
                v[key]['grad'] = v[key]['grad']*np.ones(v[key]['pin'].shape,dtype=float)
            if not np.iterable(v[key]['vary']):
                v[key]['vary'] = v[key]['vary']*np.ones(v[key]['pin'].shape,dtype=bool)
            v[key]['vary'] = np.array(v[key]['vary'])
            v[key]['grad'] = np.array(v[key]['grad'])
    ## encode varied parameters and their gradient calculation step size
    p,grad = [],[]
    for key in vkeys:
        if np.iterable(v[key]['pin']):
            p.extend(v[key]['pin'][v[key]['vary']])
            grad.extend(v[key]['grad'][v[key]['vary']])
        elif v[key]['vary']:
                p.append(v[key]['pin'])
                grad.append(v[key]['grad'])
    lowest_rms_thus_far = [np.inf]
    def calc_residual(p):
        residual = y-calc_function(p)
        if monitor_rms:
            if rms(residual) < lowest_rms_thus_far[0]:
                lowest_rms_thus_far[0] = rms(residual)
                print('RMS:',lowest_rms_thus_far[0])
        return residual
    def calc_function(p):
        ## call function, return value. Need to build kwargs for
        ## function.
        decode_p(p)
        kwargs = {key:v[key]['p'] for key in vkeys}
        return f(**kwargs)
    def decode_p(p):
        ## loop through variables, extracts varied parametres
        ## from p and shortens p accordingly
        p = list(p)
        for key in vkeys:
            v[key]['p'] = copy(v[key]['pin'])
            ## if a list
            if np.iterable(v[key]['pin']):
                v[key]['p'][v[key]['vary']] = p[:sum(v[key]['vary'])]
                p = p[sum(v[key]['vary']):]
            ## if a varied float
            elif v[key]['vary']:
                v[key]['p'] = p.pop(0)
            ## if not varied
            else:
                v[key]['p'] = v[key]['pin']
            ## ensure abslute value if required
            try:
                if v[key]['strictly_positive']:
                    v[key]['p'] = np.abs(v[key]['p'])
            except KeyError:    # implies strictly_positive not defined
                pass
    def decode_sigma(sigma):
        ## similar to decode_p
        sigma = np.abs(sigma)   # make all positive
        for key in vkeys:
            if np.iterable(v[key]['pin']):
                v[key]['sigma'] = np.nan*np.ones(v[key]['pin'].shape)
                v[key]['sigma'][v[key]['vary']] = sigma[:sum(v[key]['vary'])]
                sigma = sigma[sum(v[key]['vary']):]
            elif v[key]['vary']:
                v[key]['sigma'] = sigma[0]
                sigma = sigma[1:]
            else:
                v[key]['sigma'] = np.nan
    ## actual least squares fit
    if optimise:
        p,sigma = leastsq(calc_residual,p,grad,print_error_mesg=True,error_only=False,xtol=xtol)
    else:
        sigma = np.zeros(len(p))
    decode_p(p)
    decode_sigma(sigma)
    r = calc_residual(p)
    yf = calc_function(p)
    ## print summary of fitted parameters
    if print_result:
        print(('\n'.join([
            '{:20s} = {:20s} +- {:8s}'.format(key,str(val['p']),str(val['sigma']))
                                              for (key,val) in list(v.items())])))
    return(yf,r,v)

def weighted_mean(
        x,
        dx,
        error_on_nan_zero=True, # otherwise edit out affected values
):
    """Calculate weighted mean and its variance. If
    error_on_nan_zero=False then remove data with NaN x or dx, or 0
    dx."""
    # ## if ufloat, separate into nominal and error parts -- return as ufloat
    # if isinstance(x[0],uncertainties.AffineScalarFunc):
        # (mean,variance) = weighted_mean(*decompose_ufloat_array(x))
        # return(ufloat(mean,variance))
    x,dx = np.array(x,dtype=float),np.array(dx,dtype=float) 
    ## trim data to non nan if these are to be neglected, or raise an error
    i = np.isnan(x)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN values present.') 
        else:
            x,dx = x[~i],dx[~i]
    i = np.isnan(dx)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN errors present.') 
        else:
            x,dx = x[~i],dx[~i]
    i = (dx==0)
    if np.any(i):
        if error_on_nan_zero:
            raise Exception('NaN errors present.') 
        else:
            x,dx = x[~i],dx[~i]
    ## make weighed mean
    weights = dx**-2           # assuming dx is variance of normal pdf
    weights = weights/sum(weights) # normalise
    mean = np.sum(x*weights)
    variance = np.sqrt(np.sum((dx*weights)**2))
    return (mean,variance)

def average(x,dx,axis=1,returned=True,warningsOn=True):
    """ 
    Based on numpy.average.\n
    Return weighted mean along a particular. Nan data x, or nan or zero
    errors dx are ignored.\n
    May not word for >2D.
    """
    ## avoid overwriting originals - slows things down probably
    x = np.array(x)
    dx = np.array(dx)
    ## calculate weights from standard error
    weights = dx.astype(float)**-2.
    ## set weight of zero error or nan data to zero
    i = np.isnan(dx)|(dx==0)|np.isnan(x)
    x[i] = 0.
    weights[i] = 0.
    ## if all data invalid avoid a ZeroDivisionError
    i = weights.sum(axis)==0
    if axis==0:
        weights[:,i] = 1.
    elif axis==1:
        weights[i,:] = 1.
    ## the actual averaging
    m,tmp = np.average(x,weights=weights,returned=returned,axis=axis)
    s = np.sqrt(tmp**-1)
    ## correct for the avoided ZeroDivisionError
    m[i] = np.nan
    s[i] = np.nan
    return m,s

def common_weighted_mean(*data):
    """Take a weighted mean of all arguments which are arrays of form
    [x,y,dy].  Those arguments for which certain values of x are
    missing are not included in mean of that value. Alternatively you
    could provide the arrays already concatenated into one big array
    with some repeating x values.\n\nZero or NaN error data is removed
    and forgotten. """
    ## one 2D array (three colummns (x,y,dy)
    if len(data)==1:
        x,y,dy = data[0]
    ## join all input data given as three inputs, x,y,dy
    else:        
        x=np.concatenate([d[0] for d in data])
        y=np.concatenate([d[1] for d in data])
        dy=np.concatenate([d[2] for d in data])
    ## remove NaN x or y data
    i = ~(np.isnan(y)|np.isnan(x))
    x,y,dy = x[i],y[i],dy[i]
    ## zero or NaN errors are increased so as to remove this freom the
    ## weighting - POTENTIAL BUG
    i = (dy==0)|np.isnan(dy)
    ## if all are 0 or nan - set to some random value, otherwise make
    ## sufficiently large
    if all(i): dyDefault = 1
    else:      dyDefault = dy[~i].max()*1e5
    dy[i] = dyDefault
    ## prepare arrays for output data
    xout=np.unique(x)
    yout=np.zeros(xout.shape)
    dyout=np.zeros(xout.shape)
    ## take various means
    for i in range(len(xout)):
        ii = np.argwhere(x==xout[i])
        (yout[i],dyout[i])=weighted_mean(y[ii],dy[ii])
    ## return zero error
    dyout[dyout>dyDefault*1e-2] = np.nan
    return(xout,yout,dyout)

def mean_ignore_missing(x):
    """Calc unweighted mean of columns of a 2D array. Any nan values
    are ignored."""
    return np.array([float(t[~np.isnan(t)].mean()) for t in x])

def equal_or_none(x,y):
    if x is None and y is None:
        return(True)
    elif x==y:
        return(True)
    else:
        return(False)

def nanequal(x,y):
    """Return true if x and y are equal or both NaN. If they are vector,
    do this elementwise."""
    if np.isscalar(x) and np.isscalar(y):
        if x==y:
            return(True)
        else:
            try:
                if np.isnan(x) and np.isnan(y):
                    return(True)
            except TypeError:
                pass
            return(False)
    elif not np.isscalar(x) and not np.isscalar(y):
        if x.dtype.kind!='f' or x.dtype.kind!='f':
            return(x==y)
        else:
            return((x==y)|(np.isnan(x)&np.isnan(y)))
    else:
        raise Exception('Not implemented')

def nancumsum(x,*args,**kwargs):
    """Calculate cumsum, first set nans to zero."""
    x = np.asarray(x)
    x[np.isnan(x)] = 0.
    return np.cumsum(x,*args,**kwargs)

def cumtrapz(y,
             x=None,               # if None assume unit xstep
             direction='forwards', # or backwards
):
    """Cumulative integral, with first point equal to zero, same length as
    input."""
    assert direction in ('forwards','backwards'),f'Bad direction: {repr(direction)}'
    if direction=='backwards':
        y = y[::-1]
        if x is not None: x = x[::-1]
    yintegrated = np.concatenate(([0],integrate.cumtrapz(y,x)))
    if direction=='backwards': yintegrated = -yintegrated[::-1] # minus sign to account for change in size of dx when going backwards, which is probably not intended
    return(yintegrated)

def cumtrapz_reverse(y,x):
    """Return a cumulative integral ∫y(x) dx from high to low limit."""
    x,i = np.unique(x,return_index=True)
    y = y[i]
    return(integrate.cumtrapz(y[-1::-1],-x[-1::-1])[-1::-1])


def power_spectrum(x,y,make_plot=False,fit_peaks=False,fit_radius=1,**find_peaks_kwargs):
    """Return (frequency,power) after spectral analysis of y. Must be on a
    uniform x grid."""
    dx = np.diff(x)
    assert np.abs(dx.max()/dx.min()-1)<1e-5,'Uniform grid required.'
    dx = dx[0]
    F = np.fft.fft(y)          # Fourier transform
    F = np.real(F*np.conj(F))         # power spectrum
    F = F[:int((len(F-1))/2+1)] # keep up to Nyquist frequency
    f = np.linspace(0,1/dx/2.,len(F)+1)[:-1] # frequency scale
    if make_plot:
        ax = plt.gca()
        ax.plot(f,F,color=newcolor(0))
        ax.set_xlabel('f')
        ax.set_ylabel('F')
        ax.set_yscale('log')
    if not fit_peaks:
        return(f,F)
    else:
        import spectra
        resonances = spectra.data_structures.Dynamic_Recarray()
        for i in find_peaks(F,f,**find_peaks_kwargs):
            ibeg = max(0,i-fit_radius)
            iend = min(i+fit_radius+1,len(f))
            ft,Ft = f[ibeg:iend],F[ibeg:iend]
            p,yf = spectra.lineshapes.fit_lorentzian(ft,Ft,x0=f[i],S=F[i],Γ=dx)
            resonances.append(f0=p['x0'], λ0=1/p['x0'], S=p['S'],Γ=p['Γ'])
            if make_plot:
                ax.plot(ft,yf,color=newcolor(1))
        return(f,F,resonances)

def find_in_recarray(recarray,**key_value):
    """Find elements of recarray which match (key,value) pairs."""
    # return(np.prod([recarray[key]==value for (key,value) in key_value.items()],axis=0))
    return(np.prod([recarray[key]==value for (key,value) in key_value.items()],axis=0,dtype=bool))

# def unique(x):
#     """Returns unique elements."""
#     if not np.iterable(x): return([x])
#     return(list(set(x)))

def unique(*x,preserve_ordering=False):
    """Returns unique elements. preserve_ordering is likely slower"""
    if preserve_ordering:
        x = list(x)
        for t in copy(x):
            while x.count(t)>1:
                x.pop(t)
        return(x)
    else:
        return(list(set(x)))

def argunique(x):
    """Find indices of unique elements of x. Picks first such
    element. Does not sort."""
    y = np.unique(x)
    x = list(x)
    return np.array([x.index(yi) for yi in y])

def unique_iterable(x):
    """Find unique elements of x assuming each element is an iterable,
    returns as a tuple of tuples. E.g., [[1,2],[1,2,],[1,1,]]] returns
    ((1,2),(1,1,))."""
    return(tuple(           # return as tuple
            set(            # do the uniquifying by using a set object
                tuple(tuple(t) for t in x)))) # convert 2D elements into immutable tuple of tuples

def unique_combinations(*args):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    return(set(zip(*args)))

def unique_array_combinations(*arrs,return_mask=False):
    """All are iterables of the same length. Finds row-wise combinations of
    args that are unique. Elements of args must be hashable."""
    ra = np.rec.fromarrays(arrs)
    unique_values = np.unique(ra)
    if return_mask:
        return([(t,ra==t) for t in unique_values])
    else:
        return(unique_values)

def sortByFirst(*args,**kwargs):
    """Returns iterable args all sorted by the elements of the first
    arg. Must be same length of course.  Optional key-word argument
    outputType=type will cast outputs into type. Otherwise tuples are
    returned regardless of the input types."""
    
    outputs = list(zip(*(sorted(zip(*args),key=lambda x:x[0]))))
    if 'outputType' in kwargs:
        outputs = (kwargs['outputType'](output) for output in outputs)
    return outputs

def sortByFirstInPlace(*args):
    """Sorts all arrays in args in place, according to sorting of
    first arg. Returns nothing. Must be arrays or something else that
    supports fancy indexing."""
    i = np.argsort(args[0])
    for j in range(len(args)):
        args[j][:] = args[j][i]

def vectorise_function(fcn,*args,**kwargs):
    """Run function multiple times if any argument is vector, possibly
    broadcasting the rest. If all arguments are scalar return None."""
    ## get length of data or if needs to be vectorise
    length = None
    for arg in list(args)+list(kwargs.values()):
        if not np.isscalar(arg) and arg is not None:
            if length is None:
                length = len(arg)
            else:
                assert len(arg)==length,'Cannot vectorise, argument lengths mismatch.'
    ## indicates no vectorisation needed
    if length is None:
        return(None)
    ## extend short data
    if length is not None:
        args = list(args)
        for i,arg in enumerate(args):
            if np.isccalar(arg) or arg is None:
                args[i] = [arg for t in range(length)]
        for key,val in kwargs.items():
            if np.isscalar(val) or val is None:
                kwargs[key] = [val for t in range(length)]
    ## run and return list of results
    return([fcn(
        *[arg[i] for arg in args],
        **{key:val[i] for key,val in kwargs.items()})
            for i in range(length)])

def vectorise_dicts(*dicts):
    """Input arguments are multiple dicts with the keys. Output argument
    is one dict with the same keys and all input values in lists."""
    if len(dicts)==0:
        return({})
    retval = {}
    for key in dicts[0]:
        retval[key] = [t[key] for t in dicts]
    return(retval)
    
def common(x,y,use_hash=False):
    """Return indices of common elements in x and y listed in the order
    they appear in x. Raises exception if repeating multiple matches
    found."""
    if not use_hash:
        ix,iy = [],[]
        for ixi,xi in enumerate(x):
            iyi = find([xi==t for t in y])
            if len(iyi)==1: 
                ix.append(ixi)
                iy.append(iyi[0])
            elif len(iyi)==0:
                continue
            else:
                raise Exception('Repeated value in y for: '+repr(xi))
        if len(np.unique(iy))!=len(iy):
            raise Exception('Repeated value in x for something.')
        return(np.array(ix),np.array(iy))
    else:
        xhash = np.array([hash(t) for t in x])
        yhash = np.array([hash(t) for t in y])
        ## get sorted hashes, checking for uniqueness
        xhash,ixhash = np.unique(xhash,return_index=True)
        assert len(xhash)==len(x),f'Non-unique values in x.'
        yhash,iyhash = np.unique(yhash,return_index=True)
        assert len(yhash)==len(y),f'Non-unique values in y.'
        ## use np.searchsorted to find one set of hashes in the other
        iy = np.arange(len(yhash))
        ix = np.searchsorted(xhash,yhash)
        ## remove y beyond max of x
        i = ix<len(xhash)
        ix,iy = ix[i],iy[i]
        ## requires removing hashes that have no search sorted partner
        i = yhash[iy]==xhash[ix]
        ix,iy = ix[i],iy[i]
        ## undo the effect of the sorting above
        ix,iy = ixhash[ix],iyhash[iy]
        ## sort by index of first array -- otherwise sorting seems to be arbitrary
        i = np.argsort(ix)
        ix,iy = ix[i],iy[i]
        return(ix,iy)
    

def get_common(x,y):
    """Return common subsets of x and y."""
    i,j = common(x,y)
    return(x[i],y[i])
    
def sort_to_match(x,y):
    """Return a copy of x sorted into the smae dissaray as y. That is
    the same reordering will sort both y and the return value."""
    x = np.array(x,ndmin=1)
    y = np.array(y,ndmin=1)
    return x[np.argsort(x)[np.argsort(np.argsort(y))]]

def argsort_to_match(x,y):
    """Returns indices which will sort x to give the same dissaray as y."""
    x = np.array(x,ndmin=1)
    y = np.array(y,ndmin=1)
    return np.argsort(x)[np.argsort(np.argsort(y))]

def isin(x,y):
    """Return arrays of booleans same size as x, True for all those
    elements that exist in y."""
    return np.array([i in y for i in x])

def find_overlap(x,y):
    """Return boolean arrays (i,j) indicating (x,y) that cover an
    overlapping region. Outermost points are taken from x, i.e., x[i]
    encompasses y[j]. Assumes x and y are ordered. """
    i = (x>=y[0])&(x<=y[-1])
    j = (y>=x[0])&(y<=x[-1])
    if any(i):                  # if not overlap then don't proceed with this
        if not i[0]  and  x[i][0]!=y[j][0] : i[my.find(i)[0] -1]  = True
        if not i[-1] and x[i][-1]!=y[j][-1]: i[my.find(i)[-1]+1] = True
    return(i,j)

def argminabs(x): return(np.argmin(np.abs(x)))

def argmaxabs(x): return(np.argmax(np.abs(x)))

def find(x):
    """Convert boolean array to array of True indices."""
    return(np.where(x)[0])

def findin(x,y):
    """Find indices of x that appear in y, in the order they appear in
    y. If an element of x cannot be found in y, or if multiple found,
    an error is raised."""
    x = ensureIterable(x)
    y = ensureIterable(y)
    i = np.zeros(len(x),dtype='int')
    for j in range(len(x)):
        ii = find(y==x[j])
        if len(ii)!=1:
            raise Exception('Element `'+str(x[j])+'\' should have 1 version, '+str(len(ii))+' found.')
        i[j] = ii
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

def inv(x):
    """Invert a symmetric matrix."""
    x = np.array(x,order='F',dtype=float)
    y = copy(x)
    myf.inv(x,y)
    return(y)

def integrate_trapz_uniform(y,dx=1.):
    """Trapezium integration on a uniform grid. If x is 2D then integrate
    along the first axis. Seems to be about 3x faster than
    scipy.integrate.trapz. Speed up if 2D array is order='F'."""
    if y.ndim==1:
        yint = np.zeros(1,dtype=float)
        myf.integrate_trapz_uniform_grid(y.astype(float),dx,yint)
        return(float(yint))
    elif y.ndim==2:
        yint = np.zeros(y.shape[1],dtype=float)
        myf.integrate_trapz_uniform_grid_two_dimensional(np.asfortranarray(y),dx,yint)
        return(yint)
    else:
        raise Exception('Only implemented for 1D and 2D arrays.')

def inrange(x,xbeg,xend=None):
    """Return arrays of booleans same size as x, True for all those
    elements that xbeg<=x<=xend.\n\nIf xend is none and xbeg is an
    array, find elements of x in the range of y. """
    if xend is None:
        return (x>=np.min(xbeg))&(x<=np.max(xbeg))
    else:
        return (x>=xbeg)&(x<=xend)

def common_range(x,y):
    """Return min max of values in both x and y (may not be actual
    values in x and y)."""
    return(max(min(x),min(y)),min(max(x),max(y)))

def in_common_range(x,y):
    """Return indices of arrays x and y that line inside their common range."""
    t0,t1 = common_range(x,y)
    return(inrange(x,t0,t1),inrange(y,t0,t1))

def find_in_range_sorted(x,x0,x1):
    """Find return i0,i1, indices of x bounding x0 and x1. Assumes
    x1>x0 and x is sorted., NOT COMLETELY ACCURATE -- MIGHT GET
    EDGTEST SLIGHTLY WRONGE"""
    ## prepare fortran inputs
    x = np.array(x,dtype=float,ndmin=1)
    n = np.array(len(x),dtype=int)
    x0 = np.array(x0,dtype=float)
    x1 = np.array(x1,dtype=float)
    i0 = np.array(0,dtype=int)
    i1 = np.array(1,dtype=int)
    ## call compiled coe
    myf.find_in_range_sorted(x,x0,x1,i0,i1)
    return(int(i0),int(i1))

# def find_regexp(regexp,x):
#     """Returns boolean array of elements of x whether or not they match
#     regexp."""
#     return np.array([bool(re.match(regexp,t)) for t in x])


def match_regexp(regexp,x):
    """Returns boolean array of elements of x whether or not they match
    regexp."""
    return np.array([bool(re.match(regexp,t)) for t in x])

def find_regexp(regexp,x):
    return(find(match_regexp(regexp,x)))

def meshgrid(*args):
    """ meshgrid(arr1,arr2,arr3,...)
    Expand 1D arrays arr1,... into multiple multidimensional arrays
    that loop over the values of arr1, ....  Similar to matlab/octave
    meshgrid. Sorry about the poor documentation, an example:
    meshgrid(np.array([1,2]),np.array([3,4]),)
    returns
    (array([[1, 1],[2, 2]]), array([[3, 4],[3, 4]]))
    """
    ## a sufficiently confusing bit of code its probably easier to
    ## rewrite than figure out how it works
    n = len(args)
    assert n>=2, 'requires at least two arrays'
    l = [len(arg) for arg in args]
    ret = []
    for i in range(n):
        x = np.array(args[i])
        for j in range(n):
            if i==j: continue
            x = np.expand_dims(x,j).repeat(l[j],j)
        ret.append(x)
    return tuple(ret)

def localmax(x):
    """Return array indices of all local (internal) maxima. The first point is returned
    for adjacent equal points that form a maximum."""
    j = np.append(np.argwhere(x[1:]!=x[0:-1]),len(x)-1)
    y = x[j]
    i = np.squeeze(np.argwhere((y[1:-1]>y[0:-2])&(y[1:-1]>y[2:]))+1)
    return np.array(j[i],ndmin=1)

def localmin(x):
    """Return array indices of all local (internal) minima. The first point is returned
    for adjacent equal points that form a minimum."""
    return localmax(-x)

# def down_sample(x,y,dx):
    # """Down sample into bins dx wide.
    # An incomplete final bin is discarded.
    # Reduce the number of points in y by factor, summing
    # n-neighbours. Any remaining data for len(y) not a multiple of n is
    # discarded. If x is given, returns the mean value for each bin, and
    # return (y,x)."""
    # if x is None:
        # return np.array([np.sum(y[i*n:i*n+n]) for i in range(int(len(y)/n))])
    # else:
        # return np.array(
            # [(np.sum(y[i*n:i*n+n]),np.mean(x[i*n:i*n+n])) for i in range(int(len(y)/n))]).transpose()

def average_to_grid(x,y,xgrid):
    """Average y on grid-x to a new (sparser) grid. This might be useful
    when you want to average multiple noisy traces onto a common
    grid. Splining wont work because of the noise."""
    i = ~np.isnan(y);x,y = x[i],y[i] # remove bad data, i.e., nans
    ygrid = np.ones(xgrid.shape)*np.nan
    ## loop over output grid - SLOW!
    for i in np.argwhere((xgrid>x[0])&(xgrid<x[-1])):
        if i==0:
            ## first point, careful about bounds
            j = False*np.ones(x.shape,dtype=bool)
        elif i==len(xgrid)-1:
            ## first point, careful about bounds
            j = False*np.ones(x.shape,dtype=bool)
        else:
            ## find original data centred around current grid point
            j = (x>0.5*(xgrid[i]+xgrid[i-1]))&(x<=0.5*(xgrid[i]+xgrid[i+1]))
        if j.sum()>0:
            ygrid[i] = y[j].sum()/j.sum()
    return ygrid

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

def locate_peaks(
        y,x=None,
        minX=0.,
        minY=0.,
        fitMaxima=True, fitMinima=False,
        plotResult=False,
        fitSpline=False,
        search_width=1,
        convolve_with_gaussian_of_width=None,
):
    """Find the maxima, minima, or both of a data series. If x is not
    specified, then replace with indices.\n
    Points closer than minX will be reduced to one extremum, points
    less than minY*noise above the mean will be rejected. The mean is
    determined from a tensioned spline, unless fitSpline=False.\n
    If plotResult then issue matplotlib commands on the current axis.\n\n
    """
    ## x defaults to indices
    if x is None: x = np.arange(len(y))
    ## sort by x
    x = np.array(x)
    y = np.array(y)
    i = np.argsort(x)
    x,y = x[i],y[i]
    ## smooth with gaussianconvolution if requested
    if convolve_with_gaussian_of_width is not None:
        y = convolve_with_gaussian(x,y,convolve_with_gaussian_of_width,regrid_if_necessary=True)
    ## fit smoothed spline if required
    if fitSpline:
        fs = spline(x,y,x,s=1)
        ys = y-fs
    else:
        fs = np.zeros(y.shape)
        ys = y
    ## fit up or down, or both
    if fitMaxima and fitMinima:
        ys = np.abs(ys)
    elif fitMinima:
        ys = -ys
    ## if miny!=0 reject those too close to the noise
    # if minY!=0:
        # minY = minY*np.std(ys)
        # i =  ys>minY
        # # ys[i] = 0.
        # x,y,ys,fs = x[i],y[i],ys[i],fs[i]
        # # x,ys = x[i],ys[i]
    ## find local maxima
    i =  list(find( (ys[1:-1]>ys[0:-2]) & (ys[1:-1]>ys[2:]) )+1)
    ## find equal neighbouring points that make a local maximum
    # j = list(np.argwhere(ys[1:]==ys[:-1]).squeeze())
    j = list(find(ys[1:]==ys[:-1]))
    while len(j)>0:
        jj = j.pop(0)
        kk = jj + 1
        if kk+1>=len(ys): break
        while ys[kk+1]==ys[jj]:
            j.pop(0)
            kk = kk+1
            if kk+1>=len(ys):break
        if jj==0: continue
        if kk+1>=len(ys): continue
        if (ys[jj]>ys[jj-1])&(ys[kk]>ys[kk+1]):
            i.append(int((jj+kk)/2.))
    i = np.sort(np.array(i))
    ## if minx!=0 reject one of each pair which are too close to one
    ## if miny!=0 reject those too close to the noise
    if minY!=0:
        minY = minY*np.std(ys)
        i = [ii for ii in i if ys[ii]>minY]
    ## another, taking the highest
    if minX!=0:
        while True:
            jj = find(np.diff(x[i]) < minX)
            if len(jj)==0: break
            for j in jj:
                if ys[j]>ys[j+1]:
                    i[j+1] = -1
                else:
                    i[j] = -1
            i = [ii for ii in i if ii!=-1]
    ## plot
    if plotResult:
        fig = plt.gcf()
        ax = fig.gca()
        ax.plot(x,y,color='red')
        if minY!=0:
            ax.plot(x,fs+minY,color='lightgreen')
            ax.plot(x,fs-minY,color='lightgreen')
        ax.plot(x[i],y[i],marker='o',ls='',color='blue',mew=2)
    ## return
    return np.array(i,dtype=int)

def find_peaks(
        y,
        x=None,                    # if x is None will use index
        peak_type='maxima',     # can be 'maxima', 'minima', 'both'
        fractional_trough_depth=None, # minimum height of trough between adjacent peaks as a fraction of the lowest neighbouring peak height. I.e., 0.9 would be a very shallow trough.
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
        return(np.concatenate(np.sort(maxima,minima)))
    ## get data in correct array format
    y = np.array(y,ndmin=1)             # ensure y is an array
    if x is None: x = np.arange(len(y)) # default x to index
    assert all(np.diff(x)>0), 'Data not sorted or unique with respect to x.'
    ## in case of minima search
    if peak_type=='minima':
        y *= -1
        ybeg,yend = -1*yend,-1*ybeg
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
    if fractional_trough_depth is not None:
        i = 0
        while i < (len(ipeak)-1):
            ## index of minimum between maxima
            j = ipeak[i]+np.argmin(y[ipeak[i]:ipeak[i+1]+1])
            if (y[j]/min(y[ipeak[i]],y[ipeak[i+1]]) < fractional_trough_depth):
                ## no problem, proceed to next maxima
                i += 1
            else:
                ## not happy, delete the lowest height maxima
                if y[ipeak[i]]>y[ipeak[i+1]]:
                    ipeak.pop(i+1)
                else:
                    ipeak.pop(i)
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
    return(ipeak)

def sum_with_nans_as_zero(args,**kwargs):
    """Add arrays in args, as if nan values are actually zero. If
revert_to_nans is True, then turn all zeros back to nans after
summation."""
    kwargs.setdefault('revert_to_nans',False)
    is_scalar = np.isscalar(args[0])
    args = [np.array(arg,ndmin=1) for arg in args]
    retval = np.zeros(args[0].shape)
    for arg in args:
        i = ~np.isnan(arg)
        retval[i] += arg[i]
    if is_scalar: 
        retval = float(retval)
    if kwargs['revert_to_nans']:
        retval[retval==0.] = np.nan
    return retval

def mystack(*args): 
    return(column_stack(args))

def flatten(*args):
    """
    All args are flattened into 1D and concatentated into one 1D
    array. Wont flatten strings.
    """
    return(np.array([x for x in mpl.cbook.flatten(args)]))

def unpack(*args):
    """Flatten all args and join together into tuple."""
    return tuple(x for x in matplotlib.cbook.flatten(args))

def normal_distribution(x,μ=0.,σ=1):
    """Normal distribution."""
    return(1/np.sqrt(constants.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2)))

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
    if iseven(len(gx)): gx = gx[0:-1]
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

def convolve_with_gaussian_to_grid(x,y,xout,fwhm):
    """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    convolution after a certain number of fwhms. x must be on a
    regular grid."""
    yout = np.zeros(xout.shape)
    myf.convolve_with_gaussian(x,y,xout,yout,float(fwhm))
    return(yout)

def convolve_with_lorentzian(x,y,fwhm,fwhms_to_include=50,regrid_if_necessary=False):
    """Convolve function y(x) with a gaussian of FWHM fwhm. Truncate
    convolution after a certain number of fwhms. x must be on a
    regular grid."""
    dx = (x[-1]-x[0])/(len(x)-1)
    ## check on regular grid, if not then spline to a new one
    t = np.diff(x)
    regridded = False
    if (t.max()-t.min())>dx/1000.:
        if regrid_if_necessary:
            regridded = True
            x_original = x
            xstep = t.min()
            # x = np.arange(x[0],x[-1]+xstep/2.,xstep)
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
    if iseven(len(gx)): gx = gx[0:-1]
    gx = gx-gx.mean()
    gy = lorentzian(gx,k0=0,S=1,Gamma=fwhm,norm='sum')
    assert len(y)>len(gy), 'Data vector is shorter than convolving function.'
    y = np.convolve(y,gy,mode='same')
    ## remove padding
    y = y[len(xpad):-len(xpad)]
    x = x[len(xpad):-len(xpad)]
    ## return to original grid if regridded
    if regridded:
        y = spline(x,y,x_original)
    return y

def convolve_with_gaussian_with_prebinning(
        x,y,
        gaussian_FWHM,
        bins_per_gaussian_FWHM=10,
):
    """Convolve data x,y with a Gaussian of full-width half-maximum
    gaussian_FWHM. To speed up the process first bin data by an amount
    dictated by bins_per_gaussian_FWHM. Then return a new pair (x,y)
    on the binned grid and convolved by the Gaussian."""
    ## calculate bin centres
    binwidth = gaussian_FWHM/bins_per_gaussian_FWHM
    xbin = np.arange(x.min()+binwidth/2.,x.max(),binwidth)
    ## for each bin get the mean value (using trapezium integration)
    ## of input function over that range
    ybin = np.zeros(xbin.shape)
    for i,xi in enumerate(xbin):
        j = (x>(xi-binwidth/2))&(x<=(xi+binwidth/2))
        try:
            ybin[i] = integrate.trapz(y[j],x[j])/(x[j].max()-x[j].min())
            if np.isnan(ybin[i]): ybin[i] = 0 # probably should do this more intelligently -- or raise an exception
        except ValueError:      # catch empty j -- much faster than prechecking
            ybin[i] = 0
    ## convolve with gaussian
    ybin = convolve_with_gaussian(xbin,ybin,gaussian_FWHM)
    return(xbin,ybin)
        
def convolve_with_sinc(x,y,fwhm,fwhms_to_include=10,):
    """Convolve function y(x) with a sinc of FWHM fwhm. Truncate
    convolution after a certain number of fwhms. x must be on a
    regular grid."""
    ## add padding to data
    dx = (x[-1]-x[0])/(len(x)-1)
    xpad = np.arange(dx,fwhms_to_include*fwhm,dx)
    x = np.concatenate((x[0]-xpad[-1::-1],x,x[-1]+xpad))
    y = np.concatenate((np.full(xpad.shape,y[0]),y,np.full(xpad.shape,y[-1])))
    ## get sinc to convolve with
    gx = np.arange(-fwhms_to_include*fwhm,fwhms_to_include*fwhm,dx)
    if iseven(len(gx)): gx = gx[0:-1]
    gx = gx-gx.mean()
    gy = sinc(gx,fwhm=fwhm,mean=0.,norm='sum')
    ## convolve
    y = np.convolve(y,gy,mode='same')
    ## remove padding
    y = y[len(xpad):-len(xpad)]
    x = x[len(xpad):-len(xpad)]
    return(y)

def lorentzian(k,k0=0,S=1,Gamma=1,norm='area'):
    """Lorentzian profile.
    Inputs:
        k     energy
        k0    resonance central energy
        S     integrated cross-section
        Gamma full-width half-maximum
    """
    if norm=='area':
        return S*Gamma/2./3.1415926535897931/((k-k0)**2+Gamma**2/4.)
    elif norm=='sum':
        t = S*Gamma/2./3.1415926535897931/((k-k0)**2+Gamma**2/4.)
        t = t/t.sum()
        return(t)
    else:
        raise Exception('normalisation other than area not implemented, easy to do though.')

def voigt_fwhm(gaussian_width,lorentzian_width):
    """Approximate calculationg from wikipedia"""
    return 0.5346*lorentzian_width + np.sqrt(0.2166*lorentzian_width**2 + gaussian_width**2)

def _voigt_cachetools_hashkey(*args,**kwargs):
    """A bespoke cachetoosl memoizing cache key. Uses arary beg,end,len
    instead of array itself as a hash."""
    args = list(args)
    for i,arg in enumerate(args): # convert arrays into three value hashes
        if isinstance(arg,np.ndarray):
            if len(arg)==0:
                args[i]=None
            else:
                args[i] = (arg[0],arg[-1],len(arg))
    return(cachetools.keys.hashkey(*args,**kwargs))

# @cachetools.cached(cache=cachetools.LRUCache(1e4),key=_voigt_cachetools_hashkey)
def voigt(
        k,                      # assumes k sorted
        k0=0,
        strength=1.,
        gaussian_width=1.,
        lorentzian_width=1, 
        # method='mclean_etal1994',normalisation='area',
        # method='wofz',
        # method='wofz_parallel',
        method='wofz_approx_long_range',
        normalisation='area',
        minimum_width=0.,
        long_range_gaussian_cutoff_widths=10,
        widths_cutoff=None,
):
    """Approximates a voigt-profile.

    Inputs:

    k -- Array at which to evaluate function.
    k0 -- Center of peak.
    strength -- The meaning of this depends on normalisation.
    method -- 'whiting1968', 'whiting1968 fortran', 'convolution',  
              'mclean_etal1994', 'wofz'
    normalisation -- 'area', 'peak', 'sum', 'none'
    minimum_width -- For Gaussian or Lorentzian widths below this just, ignore this
                    component and return a pure Lorentzian or Gaussian.

    Outputs:

    v -- Array same size as k.

    All methods are approximate, even convolution because of the
    finite grid step employed.

    References:

    E. E. Whiting 1968, Journal Of Quantitative Spectroscopy &
    Radiative Transfer 8:1379.

    A. B. McLean et al. 1994, Journal of Electron Spectrosocpy and
    Related Phenomena, 69:125.

     Notes:

    McLean et al. is more accurate than Whiting and slightly faster,
    and convolution should be most accurate for sufficiently small
    step size, but is very slow. Fortran version of Whiting is the
    fastest, fotran version of Mclean is slightly slower!

    Whiting method leads to zeros at small values (underflow
    somewhere?).

    Some of the analytical approximations for voigt profiles below are
    area normalised by their definition, so I do not explicitly
    normalise these, this leads to possible errors in their
    normalisation of perhaps 0.5% or below.

    wofz -- Use the real part of the Fadeeva function

    wofz_approx_long_range -- Use the real part of the Fadeeva
           function, after a certain range, just use a pure
           Lorentzian.

    """
    if widths_cutoff is not None:
        krange = (gaussian_width+lorentzian_width)*widths_cutoff
        if (k[0]-k0)>krange or (k0-k[-1])>krange: return(np.zeros(k.shape)) # no line in range -- return zero
        if np.abs(k[0]-k0)<krange and np.abs(k0-k[-1])<krange:          # all k is in range of line -- calculate all
            return(voigt(k,k0,strength,gaussian_width,lorentzian_width,method,normalisation,minimum_width,widths_cutoff=None,))
        ## else search for important range
        ibeg,iend = k.searchsorted(k0-krange),k.searchsorted(k0+krange)
        v = np.zeros(k.shape)
        v[ibeg:iend] = voigt(k[ibeg:iend],k0,strength,gaussian_width,lorentzian_width,method,normalisation,minimum_width,widths_cutoff=None,)
        return(v)
    ## short cuts for negligible widths one way or the other
    if   gaussian_width<=minimum_width:   method = 'lorentzian'
    elif lorentzian_width<=minimum_width: method = 'gaussian'
    ## Calculates profile.
    if method == 'lorentzian':
        v = lorentzian(k,k0,1.,lorentzian_width,norm='area')
    elif method == 'gaussian':
        v = gaussian(k,gaussian_width,k0,norm='area')
    elif method == 'wofz':
        from scipy import special
        norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        b = 0.8325546111576 # np.sqrt(np.log(2))
        v = special.wofz((2.*(k-k0)+1.j*lorentzian_width)*b/gaussian_width).real/norm
    elif method == 'wofz_parallel':
        from scipy import special
        norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        b = 0.8325546111576 # np.sqrt(np.log(2))
        results = multiprocessing.Pool().map(special.wofz,(2.*(k-k0)+1.j*lorentzian_width)*b/gaussian_width)
        v = np.array(results).real/norm
    elif method == 'wofz_approx_long_range':
        from scipy import special
        # i = np.abs(k-k0)<(gaussian_width+lorentzian_width)*long_range_gaussian_cutoff_widths
        kcutoff = (gaussian_width+lorentzian_width)*long_range_gaussian_cutoff_widths
        ibeg,iend = np.searchsorted(k,[k0-kcutoff,k0+kcutoff])
        v = np.zeros(k.shape)
        norm = 1.0644670194312262*gaussian_width # sqrt(2*pi/8/log(2))
        b = 0.8325546111576 # np.sqrt(np.log(2))
        v[ibeg:iend] = special.wofz((2.*(k[ibeg:iend]-k0)+1.j*lorentzian_width)*b/gaussian_width).real/norm
        v[:ibeg] = lorentzian(k[:ibeg],k0,1.,lorentzian_width,norm='area')
        v[iend:] = lorentzian(k[iend:],k0,1.,lorentzian_width,norm='area')
    elif method == 'whiting1968':
        ## total FWHM - approximate formula
        voigt_width=np.abs(lorentzian_width)/2.+np.sqrt(lorentzian_width**2/4+gaussian_width**2)
        ## adjust for mean and width
        x=np.abs(k-k0)/voigt_width
        ## ratio of widths
        a = abs(lorentzian_width)/voigt_width
        ## lorentzian width and area normalised - approximate formula
        v = (  ((1-a)*np.exp(-2.772*x**2) + a/(1+4*x**2) +  
                0.016*(1-a)*a*(np.exp(-0.4*x**2.25)-10/(10+x**2.25)))/  
               (voigt_width*(1.065 + 0.447*a + 0.058*a**2))  )
    elif method == 'whiting1968 fortran':
        import pyvoigt
        v = np.zeros(k.shape)
        pyvoigt.voigt_whiting1968(lorentzian_width,gaussian_width,50.,k0,1.,k,v)
    elif method == 'convolution':
        ## convolution becomes displaced one grid point unless length k odd
        if iseven(len(k)):
            ktmp = k[1::]
            l = lorentzian(ktmp,k0,1.,lorentzian_width)
            g = gaussian(ktmp-ktmp[(len(ktmp)-1)/2],gaussian_width)
            v = np.convolve(l,g,mode='same')
            v = np.concatenate((v[0:1],v))
        else:
            l = lorentzian(k,0,1.,lorentzian_width)
            g = gaussian(k-k[(len(k)-1)/2].mean(),gaussian_width,0.)
            v = np.convolve(l,g,mode='same')
    elif method == 'mclean_etal1994':
        ## The method is inherently area normalised.
        ## 0.939437278 = 2/pi*sqrt(pi*np.log(2))
        A=[-1.2150,-1.3509,-1.2150,-1.3509]
        B=[1.2359, 0.3786, -1.2359, -0.3786,]
        C = [ -0.3085, 0.5906, -0.3085, 0.5906, ]
        D = [0.0210, -1.1858, -0.0210, 1.1858, ]
        X = 1.665109/gaussian_width*(k-k0)
        Y = 0.8325546*lorentzian_width/gaussian_width
        v = ( (C[0]*(Y-A[0])+D[0]*(X-B[0]))/((Y-A[0])**2+(X-B[0])**2)
              + (C[1]*(Y-A[1])+D[1]*(X-B[1]))/((Y-A[1])**2+(X-B[1])**2)
              + (C[2]*(Y-A[2])+D[2]*(X-B[2]))/((Y-A[2])**2+(X-B[2])**2)
              + (C[3]*(Y-A[3])+D[3]*(X-B[3]))/((Y-A[3])**2+(X-B[3])**2) ) *0.939437278/gaussian_width
    elif method == 'mclean_etal1994 fortran':
        import pyvoigt
        v = np.zeros(k.shape)
        pyvoigt.voigt(lorentzian_width,gaussian_width,k0,1.,k,v)
    else:
        raise Exception(f'Unknown method: {repr(method)}')
    ## Normalises and returns the profile.  Some methods are already
    ## area normalised by their formulation.
    if normalisation == 'none':
        v = v*strength
    elif normalisation == 'sum':
        v = v/v.sum()*strength
    elif normalisation == 'peak':
        v = v/v.max()*strength
    elif normalisation == 'area':
        ## some methods are automatically area normalised
        if method in ['whiting1968','whiting1968 fortran','mclean_etal1994',
                      'mclean_etal1994 fortran','lorentzian','gaussian',
                      'wofz','wofz_approx_long_range']:
            v = v*strength
        else:
            v = v/integrate.simps(v,k)*strength
    return(v)

# def cross_correlate(x,y,max_shift=None,return_shift=False):
    # """Normalised cross correlation."""
    # retval = np.correlate(x,y,mode='same')
    # n = len(retval)
    # imid = int((n-1)/2)
    # t = np.arange(n-imid,n,1)
    # norm = np.concatenate((t,[n],t[::-1]))
    # retval /= norm
    # if return_shift:
        # shift = np.arange(len(retval))-imid
        # return(shift,retval)
    # else:
        # return(retval)

def cross_correlate(
        x0,y0,                  # x,y data one
        x1,y1,                  # more x,y data
        conv_width=None,          # how wide (in terms of x) to cross correlation
        max_shift=None,           # maximum shift of cross corelation
):
    """Normalised cross correlation."""
    ## spline all data to min grid step
    dx = min(np.min(x0[1:]-x0[:-1]),np.min(x1[1:]-x1[:-1]))
    ##
    tx0 = np.arange(x0.min(),x0.max(),dx)
    y0 = spline(x0,y0,tx0)
    x0 = tx0
    tx1 = np.arange(x1.min(),x1.max(),dx)
    y1 = spline(x1,y1,tx1)
    x1 = tx1
    ## mid points -- might drop a half pointx
    imid0 = np.floor((len(x0)-1)/2)
    imid1 = np.floor((len(x1)-1)/2)
    ## get defaults conv_width and max_shift -- whole domain when added
    if conv_width is None:
        conv_width = dx*(min(imid0,imid1)-1)/2
    if max_shift is None:
        max_shift = conv_width
    ## initalise convolved grid
    xc = np.arange(0,max_shift,dx)
    xc = np.concatenate((-xc[::-1],xc[1:]))
    yc = np.full(xc.shape,0.0)
    imidc = int((len(xc)-1)/2)
    ## convert conv_width, max_shift to indexes, ensuring compatible
    ## with data length
    iconv_width = int(conv_width/dx) - 1
    imax_shift = min(int(max_shift/dx),imid0-1,imid1-1) 
    myf.cross_correlate(y0,y1,yc,imax_shift,iconv_width)
    return(xc,yc)

def autocorrelate(x):
    """Calculates an autocorrelation y of a a 1D array, x.  That is,
correlates x with itself as a function of relative shift, truncates
the vectors rather than pad with zeros (like convolve
does). Normalised by the square root of the sum of squres for the two
sections of x being correlated.
    SLOW: USES A LOOP!!
    """
    x = np.asarray(x)
    ks = np.arange(0,int(len(x)/2)+1,dtype=int)
    y = np.zeros(ks.shape,dtype=float)
    for (i,k) in enumerate(ks):
        a = x[k:-1]
        b = x[0:-k-1]
        y[i] = (a*b).sum() / ( np.sqrt( np.sum(a**2)*np.sum(b**2) )  )
    return y

def sinc(x,fwhm=1.,mean=0.,strength=1.,norm='area',):
    """ Calculate sinc function. """
    t = np.sinc((x-mean)/fwhm*1.2)*1.2/fwhm # unit integral normalised
    if norm=='area':
        return strength*t
    elif norm=='sum':
        return strength*t/t.sum()
    elif norm=='peak':
        return strength*t/np.sinc(0.)*1.2/fwhm

def isfloat(a):
    """Test if input is a floating point number - doesn't distinguish
    between various different kinds of floating point numbers like a
    simple test would."""
    return type(a) in [float,float64]

def isint(a):
    """Test if input is an integer - doesnt distinguish between
    various different kinds of floating point numbers like a simple
    test would."""
    return type(a) in [int,np.int64]

def isnumeric(a):
    """Test if constant numeric value."""
    return type(a) in [int,np.int64,float,np.float64]

def iseven(x):
    """Test if argument is an even number."""
    return np.mod(x,2)==0

def loadtxt(path,**kwargs):
    """Sum as numpy.loadtxt but sets unpack=True. And expands '~' in
    path."""
    path = os.path.expanduser(path)
    kwargs.setdefault('unpack',True)
    return np.loadtxt(path,**kwargs)

def loadawk(path,script,use_hdf5=False,**kwargs):
    """Load text file as array after parsing through an
    awkscript. Data is saved in a temporary file and all kwargs are
    passed to loadtxt."""
    path = expand_path(path)
    tmpfile = pipe_through_awk(path,script)
    if os.path.getsize(tmpfile.name)==0:
        raise IOError(None,'No data found',path)
    if use_hdf5:
        output = txt_to_array_via_hdf5(tmpfile,**kwargs)
    else:
        output = np.loadtxt(tmpfile,**kwargs)
    return output

def array_to_file(filename,*args,mkdir=False,**kwargs):
    """Use filename to decide whether to attempt to save as an hdf5 file
    or ascii data.\n\nKwargs:\n\n mkdir -- If True, create all leading
    directories if they don't exist. """
    filename = expand_path(filename)
    extension = os.path.splitext(filename)[1]
    if mkdir: mkdir_if_necessary(dirname(filename))
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

def load_and_spline(filename,x,missing_data=0.):
    """Load data file using file_to_array. Resplines the results to a
    grid x. Missing data set to zero. Returns splined y."""
    xn,yn = file_to_array(filename,unpack=True)
    y = np.zeros(x.shape)
    i = (x>xn.min())&(x<xn.max())
    y[i] = spline(xn,yn,x[i])
    y[~i] = missing_data
    return y

def file_to_array(
        filename,
        xmin=None,xmax=None,
        sort=False,
        awkscript=None,
        unpack=False,
        filetype=None,
        **kwargs,               # passed to function depending on filetype
):
    """Use filename to decide whether to attempt to load as an hdf5
    file or ascii data. xmin/xmax data ranges to load."""
    ## dealt with filename and type
    filename = expand_path(filename)
    extension = os.path.splitext(filename)[1]
    if filetype==None:
        if extension in ('.hdf5','.h5'):
            filetype='hdf5'
        elif extension in ('.npy','.npz'):
            filetype='numpy array'
        elif extension in ('.0','.1','.2','.3'):
            filetype='opus'
        else:
            filetype='text'
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
    elif filetype=='numpy array':
        d = np.load(filename)
    elif filetype=='opus':
        import spectra
        x,y,header = spectra.lib_molecules.load_bruker_opus_spectrum(filename)
        d = np.column_stack((x,y))
    elif filetype=='text':
        np_kwargs = copy(kwargs)
        # np_kwargs.pop('encoding')
        if len(filename)>4 and filename[-4:] in ('.csv','.CSV'): np_kwargs['delimiter'] = ','
        if 'delimiter' in np_kwargs and np_kwargs['delimiter']==' ': np_kwargs.pop('delimiter')
        d = np.genfromtxt(filename,**np_kwargs)
    else:
        raise Exception(f'Unknown filetype: {repr(filetyep)}')
    ## post processing
    d = np.squeeze(d)
    if xmin is not None:  d = d[d[:,0]>=xmin]
    if xmax is not None:  d = d[d[:,0]<=xmax]
    if sort:              d = d[np.argsort(d[:,0])]
    if unpack:            d = d.transpose()
    return(d)

def file_to_xml_tree(filename):
    """Load an xml file using standard library 'xml'."""
    from xml.etree import ElementTree
    return(ElementTree.parse(expand_path(filename)))

def pipe_through_awk(original_file_path, awk_script):
    """
    Pass file path through awk, and return the temporary file where
    the result is sent. Doesn't load the file into python memory.
    """
    ## expand path if possible and ensure exists - or else awk will hang
    original_file_path = expand_path(original_file_path)
    if not os.path.lexists(original_file_path):
        raise IOError(1,'file does not exist',original_file_path)
    # if new_file is None:
    new_file=tempfile.NamedTemporaryFile(mode='w+',encoding='utf-8')
    # command = 'awk '+'\''+awk_script+'\' '+original_file_path+'>'+new_file_path
    # # (status,output)=commands.getstatusoutput(command)
    status = subprocess.call(['awk',awk_script,original_file_path],stdout=new_file.file)
    assert status==0,"awk command failed"
    new_file.seek(0)
    return new_file

def loadsed(path,script,**kwargs):
    """
`Load text file as array after parsing through an
    sedscript. Data is saved in a temporary file and all kwargs are
    passed to loadtxt.
    """
    tmpfile = sedFilteredFile(path,script)
    output = np.loadtxt(tmpfile,**kwargs)
    return output

def sedFilteredFile(path,script):
    """Load text file as array after parsing through a sed
    script. Data is saved in a temporary file and all kwargs are
    passed to loadtxt."""
    tmpfile=tempfile.NamedTemporaryFile()
    command = 'sed '+'\''+script+'\' '+path+'>'+tmpfile.name
    (status,output)=subprocess.getstatusoutput(command)
    if status!=0: raise Exception("sed command failed:\n"+output)
    return tmpfile

def ensureArray(arr):
    """Return an at least 1D array version of input if not already one."""
    if type(arr) != np.array:
        if np.iterable(arr):
            arr = np.array(arr)
        else:
            arr = np.array([arr])
    return arr

def string_to_list(string):
    """Convert string of numbers separated by spaces, tabs, commas, bars,
    and newlines into an array. Empty elements are replaced with NaN
    if tabs are used as separators. If spaces are used then the excess
    is removed, including all leading and trailing spaces."""
    retval = [t.strip() for t in re.split(r'\n',string)
             if not (re.match(r'^ *#.*',t) or re.match(r'^ *$',t))] # remove blank and # commented lines
    retval = flatten([re.split(r'[ \t|,]+',t) for t in retval]) # split each x on space | or ,
    retval = [try_cast_to_numerical(t) for t in retval]
    return(retval)

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

def string_to_array_transpose(s):
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
            
def string_to_file(filename,string,mode='w',encoding='utf8',make_directory=False):
    """Write string to file_name."""
    filename = expand_path(filename)
    if make_directory:
        mkdir_if_necessary(dirname(filename)) 
    with open(filename,mode=mode,encoding=encoding) as f: 
        f.write(string)

def str2range(string):
    """Convert string of integers like '1,2,5:7' to an array of
    values."""
    x = string.split(',')
    r = []
    for y in x:
        try:
            r.append(int(y))
        except ValueError:
            y = y.split(':')
            r.extend(list(range(int(y[0]),int(y[1])+1)))
    return r

def derivative(x,y=None,n=1):
    """Calculate d^ny/dx^n using central difference - end points are
    extrapolated. Endpoints could use a better formula."""
    if y is None:
        x,y = np.arange(len(x),dtype=float),x
    if n==0:
        return(y)
    if n>1:
        y = derivative(x,y,n-1)
    d = np.zeros(y.shape)
    d[1:-1] = (y[2::]-y[0:-2:])/(x[2::]-x[0:-2:])
    d[0] = (y[1]-y[0])/(x[1]-x[0])
    d[-1] = (y[-2]-y[-1])/(x[-2]-x[-1])
    return d

def curvature(x,y):
    """Calculate curvature of function."""
    d=derivative(x,y);  # 1st diff
    dd=derivative(x,d); # 2nd diff
    return dd/((1.+d**2.)**(3./2.)) 


def execfile(filepath):
    """Execute the file in current namespace. Not identical to python2 execfile."""
    with open(filepath, 'rb') as fid:
        # exec(compile(fid.read(),filepath,'exec'))
        exec(fid.read())

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

def file_to_regexp_matches(filename,regexp):
    """Return match objects for each line in filename matching
    regexp."""
    with open(expand_path(filename),'r') as fid:
        matches = []
        for line in fid:
            match = re.match(regexp,line)
            if match: matches.append(match)
    return(matches)
    
def file_to_dict(filename,*args,**kwargs):
    """Convert text file to dictionary.
    \nKeys are taken from the first uncommented record, or the last
    commented record if labels_commented=True. Leading/trailing
    whitespace and leading commentStarts are stripped from keys.\n
    This requires that all elements be the same length. Header in hdf5
    files is removed."""
    filename = expand_path(filename)
    file_extension = os.path.splitext(filename)[1]
    if file_extension=='.npz':
        d = dict(**np.load(filename))
        ## avoid some problems later whereby 0D  arrays are not scalars
        for key,val in d.items():
            if val.ndim==0:
                d[key] = np.asscalar(val)
    elif file_extension in ('.hdf5','.h5'): # load as hdf5
        d = hdf5_to_dict(filename)
        if 'header' in d: d.pop('header') # special case header, not data 
        if 'README' in d: d.pop('README') # special case header, not data 
    elif file_extension in ('.ods','.csv','.CSV'): # load as spreadsheet, set # as comment char
        kwargs.setdefault('comment','#')
        d = sheet_to_dict(filename,*args,**kwargs)
    elif file_extension in ('.rs',): # my convention -- a ␞ separated file
        kwargs.setdefault('comment_regexp','#')
        kwargs.setdefault('delimiter','␞')
        d = txt_to_dict(filename,*args,**kwargs)
    elif rootname(filename) in ('README',): # load as org mode
        d = org_table_to_dict(filename,*args,**kwargs)
    elif os.path.isdir(filename): # load as directory
        d = Data_Directory(filename)
    else: # load as text
        d = txt_to_dict(filename,*args,**kwargs)
    return(d)

def org_table_to_dict(filename,table_name):
    """Load a table into a dicationary of arrays. table_name is used to
    find a #+NAME: tag."""
    with open(filename,'r') as fid:
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

def string_to_recarray(string):
    """Convert a table in string from into a recarray. Keys taken from
    first row. """
    ## turn string into an IO object and pass to txt_to_dict to decode the lines. 
    string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    import io
    return(dict_to_recarray(txt_to_dict(io.StringIO(string), labels_commented=False,)))

# def string_to_dynamic_recarray(string):
    # string = re.sub(r'^[ \n]*','',string) # remove leading blank lines
    # import io
    # d = txt_to_dict(io.StringIO(string), labels_commented=False,)
    # from data_structures import Dynamic_Recarray
    # return(Dynamic_Recarray(**d))

def file_to_recarray(filename,*args,**kwargs):
    """Convert text file to record array, converts from dictionary
    returned by file_to_dict."""
    return(dict_to_recarray(file_to_dict(filename,*args,**kwargs)))

def decompose_ufloat_array(x):
    """Return arrays of nominal_values and std_devs of a ufloat array."""
    return(np.array([t.nominal_value for t in x],dtype=float),np.array([t.std_dev for t in x],dtype=float),)

def item(x):
    """Recursively penetrate layers of iterable containers to get at
    the one number inside. Also tries to get value out of zero
    dimensional array"""
    if iterable(x): return(item(x[0]))  # first value only
    elif hasattr(x,'item'): return x.item()
    else: return x

def jj(x): return x*(x+1)

def txt2multiarray(path):
    """Text file contains 1 or 2D arrays which are separated by blank
    lines or a line beginning with a comment character. These are read
    into separate arrays and returned."""
    f = open(expand_path(path),'r')
    s = f.read()
    f.close()
    s,n = re.subn('^(( *($|\n)| *#.*($|\n)))+','',s)
    s,n = re.subn('((\n *($|\n)|\n *#.*($|\n)))+','#',s)
    if s=='':
        s = []
    else:
        s = s.split('#')
    ## return, also removes empty arrays
    return [str2array(tmp).transpose() for tmp in s if len(tmp)!=0]

def fwhm(x,y,plot=False,return_None_on_error=False):
    """Roughly calculate full-width half-maximum of data x,y. Linearly
    interpolates nearest points to half-maximum to get
    full-width. Requires single peak only in view. """
    hm = (np.max(y)-np.min(y))/2.
    i = find((y[1:]-hm)*(y[:-1]-hm)<0)
    if len(i)!=2:
        if return_None_on_error:
            return(None)
        else:
            raise Exception("Poorly defined peak, cannot find fwhm.")
    x0 = (hm-y[i[0]])*(x[i[0]]-x[i[0]+1])/(y[i[0]]-y[i[0]+1]) + x[i[0]]
    x1 = (hm-y[i[1]])*(x[i[1]]-x[i[1]+1])/(y[i[1]]-y[i[1]+1]) + x[i[1]]
    if plot==True:
        ax = plt.gca()
        ax.plot(x,y)
        ax.plot([x0,x1],[hm,hm])
    return x1-x0

def estimate_fwhm(
        x,y,
        imax=None,              # index of peak location -- else uses maximum of y
        plot=False,             
):
    """Roughly calculate full-width half-maximum of data x,y. Linearly
    interpolates nearest points to half-maximum to get
    full-width. Looks around tallest peak."""
    if imax is None: imax = np.argmax(y)
    half_maximum = y[imax]/2.
    ## index of half max nearest peak on left
    if all(y[1:imax+1]>half_maximum): raise Exception('Could not find lower half maximum.')
    i = find((y[1:imax+1]>half_maximum)&(y[:imax+1-1]<half_maximum))[-1]
    ## position of half max on left
    x0 = (half_maximum-y[i])*(x[i]-x[i+1])/(y[i]-y[i+1]) + x[i]
    ## index of half max nearest peak on left
    if all(y[imax:]>half_maximum): raise Exception('Could not find upper half maximum.')
    i = find((y[imax+1:]<half_maximum)&(y[imax:-1]>half_maximum))[0]+imax
    ## position of half max on left
    x1 = (half_maximum-y[i])*(x[i]-x[i+1])/(y[i]-y[i+1]) + x[i]
    if plot==True:
        ax = plt.gca()
        ax.plot(x,y)
        ax.plot([x0,x1],[half_maximum,half_maximum])
    return x1-x0

def fixedWidthString(*args,**kwargs):
    """Return a string joining *args in g-format with given width, and
    separated by one space."""
    if 'width' in kwargs: width=kwargs['width']
    else: width=13
    return ' '.join([format(s,str(width)+'g') for s in args])
    
def string_to_number(s,default=None):
    """ Attempt to convert string to either int or float. If fail use
    default, or raise error if None."""
    ## if container, then recursively operate on elements instead
    if np.iterable(s) and not isinstance(s,str):
        return([str2num(ss) for ss in s])
    elif not isinstance(s,str):
        raise Exception(repr(s)+' is not a string.')
    ## try to return as int
    try:
        return int(s)
    except ValueError:
        ## try to return as float
        try:
            return float(s)
        except ValueError:
            if default is not None:
                return(default)
            raise Exception(f'Could not convert string to number: {repr(s)}')
    
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
    return(s)

def txt_to_recarray(*args,**kwargs):
    """See txt_to_dict."""
    return(dict_to_recarray(txt_to_dict(*args,**kwargs)))
                          
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
    for i,line  in enumerate(filename.readlines()):
        if i<skiprows: continue
        line = line.strip()     # remove leading/trailing whitespace
        if ignore_blank_lines and len(line)==0: continue
        if filter_function is not None: line = filter_function(line)
        if filter_regexp is not None:   line = re.sub(filter_regexp[0],filter_regexp[1],line)
        line = (line.split() if delimiter is None else line.split(delimiter)) # split line
        if comment_regexp is not None and re.match(comment_regexp,line[0]): # commented line found
            if not first_block_commented_lines_passed:
                line[0] = re.sub(comment_regexp,'',line[0]) # remove comment start
                if len(line[0])==0: line = line[1:] # first element was comment only,
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
    data = collections.OrderedDict()
    for key,column in zip(labels,zip(*lines)):
        column = [(t.strip() if len(t.strip())>0 else replacement_for_blank_elements) for t in column]
        data[key] = try_cast_to_numerical_array(column)
    return(data)
            
def txt_to_array_unpack(filename,skiprows=0,comment='#'):
    """Read a text file of 2D data into a seperate array for each
    column. Attempt to cast each column independently."""
    ## Reads all data.
    fid = open(filename,'r')
    lines = fid.readlines()
    fid.close()
    ## remove blank and commented lines
    lines = [l for l in lines if not re.match(r'^\s*$|^ *'+comment,l)]
    ## skiprows
    for i in range(skiprows): lines.pop(0)
    ## initialise data lists
    data = [[] for t in lines[0].split()]
    ## get data
    for line in lines:
        for x,y in zip(data,line.split()):
            x.append(y)
    ## cast to arrays, numerical if possiblec
    data = [try_cast_to_numerical_array(t) for t in data]
    return(data)
            
def org_table_to_recarray(filename,table_name):
    """Read org-mode table from a file, convert data into a recarray, as
numbers if possible, else strings. The table_name is expected to be an
org-mode name: e.g., #+NAME: table_name"""
    fid = open(expand_path(filename),'r')
    data = collections.OrderedDict()
    ## read file to table found
    for line in fid:
        if re.match('^ *#\+NAME\: *'+re.escape(table_name.strip())+' *$',line): break
    else:
        raise Exception('table: '+table_name+' not found in file: '+filename)
    ## get keys line and initiate list in data dict
    for line in fid:
        if re.match(r'^ *\|-',line): continue # skip hlines
        line = line.strip(' |\n')
        keys = [key.strip('| ') for key in line.split('|')]
        break
    for key in keys: data[key] = []
    ## loop through until blank line (end of table) adding data
    for line in fid:
        if line[0]!='|': break                # end of table
        if re.match(r'^ *\|-',line): continue # skip hlines
        line = line.strip(' |')
        vals = [val.strip() for val in line.split('|')]
        for (key,val) in zip(keys,vals): data[key].append(val)
    for key in data: data[key] = try_cast_to_numerical_array(data[key]) # cast to arrays, numerical if possiblec
    return(dict_to_recarray(data))

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

def prefix_postfix_lines(string,prefix='',postfix=''):
    """Prefix and postfix every line of string."""
    return(prefix+string.replace('\n',postfix+'\n'+prefix)+postfix)

def extractHeader(path):
    """
    Extract info of the from XXX=YYY from a txt or csv file returning
    a dictionary with all keys and values given by strings.

    Header indicated by comment character '#'.
    
    All quotes are stripped.

    Wildly unfinished.
    """
    retDict = {}
    fid = open(path,'r')
    while True:
        line = fid.readline()
        ## eliminate all quotes
        line = line.replace("'","")
        line = line.replace('"','')
        ## break once header pased
        if not re.match(r'^\s*#',line): break
        ## eliminate leading comments
        line = re.sub(r'^\s*#*','',line)
        ## tokenise on space and ','
        toks = re.split(r' +| *, *',line)
        ## find tokens containing '=' and extract key/vals as
        ## dictionary
        for tok in toks:
            if re.match(r'.*=.*',tok):
                key,val = tok.split('=')
                retDict[key.strip()] = val.strip()
    fid.close()
    return retDict


def odsReader(fileName,tableIndex=0):
    """
    Opens an odf spreadsheet, and returns a generator that will
    iterate through its rows. Optional argument table indicates which
    table within the spreadsheet. Note that retures all data as
    strings, and not all rows are the same length.
    """
    import odf.opendocument,odf.table,odf.text
    ## common path expansions
    fileName = expand_path(fileName)
    ## loads sheet
    sheet = odf.opendocument.load(fileName).spreadsheet
    ## Get correct table. If 'table' specified as an integer, then get
    ## from numeric ordering of tables. If specified as a string then
    ## search for correct table name.
    if isinstance(tableIndex,int):
        ## get by index
        table = sheet.getElementsByType(odf.table.Table)[tableIndex]
    elif isinstance(tableIndex,str):
        ## search for table by name, if not found return error
        for table in sheet.getElementsByType(odf.table.Table):
            # if table.attributes[(u'urn:oasis:names:tc:opendocument:xmlns:table:1.0', u'name')]==tableIndex:
            if table.getAttribute('name')==tableIndex: break
        else:
            raise Exception('Table `'+str(tableIndex)+'\' not found in `'+str(fileName)+'\'')
    else:
        raise Exception('Table name/index`'+table+'\' not understood.')
    ## divide into rows
    rows = table.getElementsByType(odf.table.TableRow)
    ## For each row divide into cells and then insert new cells for
    ## those that are repeated (multiple copies are not stored in ods
    ## format). The number of multiple copies is stored as a string of
    ## an int.
    for row in rows:
        cellStrs = []
        for cell in row.getElementsByType(odf.table.TableCell):
            cellStrs.append(str(cell))
            if cell.getAttribute('numbercolumnsrepeated')!=None:
                for j in range(int(cell.getAttribute('numbercolumnsrepeated'))-1):
                    cellStrs.append(str(cell))
        ## yield each list of cells to make a generator
        yield cellStrs

def loadsheet(path,tableIndex=0):
    """Converts contents of a spreadsheet to a list of lists or
    strings. For spreadsheets with multiple tables, this can be
    specified with the optional argument."""
    ## opens file according to extension
    if path[-4:]=='.csv' or path[-4:]=='.CSV':
        assert tableIndex==0, 'Multiple tables not defined for csv files.'
        fid = open(path,'r')
        sheet = csv.reader(fid)
    elif path[-4:]=='.ods':
        sheet = odsReader(path,tableIndex=tableIndex)
    ## get data rows into list of lists
    ret = [row for row in sheet]
    ## close file if necessary
    if path[-4:]=='.csv' or path[-4:]=='.CSV':
        fid.close()
    return(ret)      

def sheet_to_recarray(*args,**kwargs):
    return(dict_to_recarray(sheet_to_dict(*args,**kwargs)))

def sheet_to_dataframe(*args,**kwargs):
    import pandas as pd
    return(pd.DataFrame(sheet_to_dict(*args,**kwargs)))

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
    if 'tableName' in kwargs:   kwargs['table_name'] = kwargs.pop('tableName')
    if 'commentChar' in kwargs: kwargs['comment'] = kwargs.pop('commentChar')
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
    elif isinstance(path,str) and path[-4:]=='.ods':
        kwargs.setdefault('sheet_name',0)
        reader=odsReader(expand_path(path),tableIndex=kwargs.pop('sheet_name'))
    elif isinstance(path,file):
        reader=csv.reader(expand_path(path),)
    else:
        raise Exception("Failed to open "+repr(path))
    ## if skip_header is set this is the place to pop the first few recrods of the reader objects
    if skip_header is not None:
        for t in range(skip_header): next(reader)
    ## if requested return all tables. Fine all names and then call
    ## sheet2dict separately for all found tables.
    if return_all_tables:
        return_dict = collections.OrderedDict()
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
    """Read a stream (line-by-line iterator) into a dictionary. First
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
    for i in range(skip_rows): next(reader)
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
    data = collections.OrderedDict()
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

def read_structured_data_file(filename):
    """Read a file containing tables and key-val pairs"""
    d = {'header':[]}
    with open(expand_path(filename)) as f:
        # for i in range(50):
        while True:
            l = f.readline()    # read a line
            if l=='': break     # end of text file
            l = re.sub('^ *# *(.*)',r'\1',l) # remove comments
            l = l[:-1]          # remove newline
            ## look for lines like: a = b
            r = re.match('^ *([^=]*[^= ]) *= *(.+) *$',l)
            if r:
                d[r.groups()[0]] = r.groups()[1]
                d['header'].append(l)
                continue
            ## look for tables like: <tablename> ...
            r = re.match('^ *< *(.+) *> *$',l)
            if r:
                table_name = r.groups()[0]
                d[table_name] = stream_to_dict(f,split=' ',table_name=table_name,comment='#')
                continue
            ## rest is just text of some kind
            d['header'].append(l)
    return d 

def csv2array(fileObject,tableName=None,skiprows=0,**kwargs):
    """Convert csv file to array.
    \nFile can be open file object or path.
    \nIf tableName is supplied string then keys and data are read
    between first column flags <tableName> and <\\tableName>. Other
    wise reads to end of file.
    \nFurther kwargs are passed to csv.reader."""
    ## open csv.reader
    if type(fileObject)==str:
        f=open(expand_path(fileObject),'r')
        reader=csv.reader(f,**kwargs)
    else: 
        reader=csv.reader(fileObject,**kwargs)
    ## skip to specific table if requested
    while tableName!=None:
        try: 
            line = next(reader)
        except StopIteration: 
            raise Exception('Table `'+tableName+'\' not found in file '+str(fileObject))
        ## skip empty lines
        if len(line)==0:
            continue
        ## table found, break loop
        if '<'+tableName+'>'==str(line[0]): 
            break
    ## skiprows - IF READING TABLE AND GET TO END OF TABLE IN HERE THERE WILL BE A BUG!!!!
    for i in range(skiprows): 
        next(reader)
    ## initialise list of lists
    data = []
    ## read data
    while True:
        ## break on end of file
        try:    line = next(reader)
        except: break
        ## if table specified stop reading at the end of it
        if tableName!=None and '<\\'+tableName+'>'==str(line[0]): break
        if tableName!=None and '<\\>'==str(line[0]): break
        ## add new row to data and convert to numbers
        data.append([str2num(cell) for cell in line])
    ## convert lists to array
    if isiterable(data[0]):
        ## if 2D then pad short rows
        data = array2DPadded(data)
    else:
        data = np.array(data)
    ## close file if necessary
    if type(fileObject)==str: f.close()
    ## return
    return data

def loadcsv(path,**kwargs):
    """Convert csv file to array. Further kwargs are passed to
    csv.reader."""
    ## open csv.reader
    f = open(expand_path(path),'r')
    reader = csv.reader(f,**kwargs)
    ## initialise list of lists
    data = []
    ## read data
    while True:
        ## break on end of file
        try:    line = next(reader)
        except: break
        ## add new row to data and convert to numbers
        data.append([str2num(cell) for cell in line])
    ## convert lists to array
    # if isiterable(data[0]):
        # ## if 2D then pad short rows
        # data = np.array2DPadded(data)
    # else:
    data = np.array(data)
        
    ## close file if necessary
    f.close()
    ## return
    return data

sheet2array = csv2array         # doesn't currently work with ods, fix one day

def writeCSV(path,data,**kwargs):
    """
    Writes data to path as a CSV file.\n\nPath is filename.
    Data is a 2D iterable.
    Kwargs are passed as options to csv.writer, with sensible defaults.
    """
    kwargs.setdefault('skipinitialspace',True)
    kwargs.setdefault('quoting',csv.QUOTE_NONNUMERIC)
    fid = open(path,'w')
    writer = csv.writer(fid,**kwargs)
    writer.writerows(data)
    fid.close()

def isiterable(x):
    """Test if x is iterable, true for strings."""
    try:
        iter(x)
    except TypeError:
        return False
    return True

def isiterable_not_string(x):
    """Test if x is iterable, False for strings."""
    if isinstance(x,str): 
        return False
    try:
        iter(x)
    except TypeError:
        return False
    return True

def isuvalue(x):
    """Test if uvalue in a fairly general way."""
    if x in (uncertainties.Variable,ufloat): return(True)
    if isinstance(x,uncertainties.Variable): return(True)
    if isinstance(x,uncertainties.AffineScalarFunc): return(True)
    return(False)

def array2DPadded(x):
    """Take 2D nested iterator x, and turn it into array, extending
    any dim=1 arrays by NaNs to make up for missing values."""
    dim1 = max([len(xi) for xi in x])
    for xi in x:
        xi.extend([np.nan for i in range(dim1-len(xi))])
    return np.array(x)

def castArrayToFloats(arr,replacement=np.nan):
    """Cast an array to astype(float) - replacing failed casts with
    *replacement*"""
    try:
        return arr.astype(float)
    except (TypeError,ValueError):
        arrout = np.ones(arr.shape,dtype=float)
        for i in range(len(arr)):
            try:
                arrout[i] = float(arr[i])
            except (TypeError,ValueError):
                arrout[i] = replacement
        return arrout

def repeat(x,repeats,axis=None):
    """Just like numpy repeat, but will extend the dimension of the
    array first if necessary."""
    x = np.asarray(x)
    if axis!=None and axis>=len(x.shape):
        number_of_axes_to_add = axis-len(x.shape)+1
        x = x.reshape(list(x.shape)+[1 for t in range(number_of_axes_to_add)])
    return np.repeat(x,repeats,axis)

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

def transpose(x):
    """Take any 2D interable and transpose it. Returns a list of
    tuples. \n\nnumpy.transpose is similar but returns an array cast to the
    most general type."""
    return list(zip(*tuple(x)))
    
def dump(o,f):
    """Like pickle.dump except that f can string filename."""
    if type(f)==str:
        f=file(f,'w')
        pickle.dump(o,f)
        f.close()
    else: pickle.dump(o,f)

def load(f):
    """Like pickle.load except that f can string filename."""
    if type(f)==str:
        f=file(f,'r')
        o = pickle.load(f)
        f.close()
    else: o = pickle.load(f)
    return o

def sumFunctions(*args):
    """Return a function that is the sum of output of a series of functions.
    \nInputs:
    \n f1,f2,f3,... -- list of function objects, each with one input and one summable output 
    \nOutputs:
    \n fout(x) = f1(x)+f2(x)+f3(x)+...
    """
    return lambda x: sum([f(x) for f in args])

def repr_args_kwargs(*args,**kwargs):
    """Format args and kwargs into evaluable args and kwargs."""
    retval = ''
    if len(args)>0:
        retval += ','.join([repr(arg) for arg in args])+','
    if len(kwargs)>0:
        retval += ','.join([str(key)+'='+repr(val) for key,val in kwargs.items()])+','
    return(retval)

def dict_to_kwargs(d,keys=None):
    """Expand a dict into evaluable kwargs. Default to all keys."""
    if keys is None: keys = d.keys() # default to all keys
    # if d is None: return('')
    return(','.join([key+'='+repr(d[key]) for key in keys]))

class Dict_Object:
    """An object which takes dictionary like arguments and exposes
    them as attributes."""
    def __init__(self,*args,**kwargs):
        ## each member of args must be a dictionary and are added to
        ## __dict__
        for arg in args:
            self.__dict__.update(arg)
        ## kwargs are passed straight to __dict__
        for (key,val) in list(kwargs.items()):
            self.__dict__[key] = val
    ## otherwise appear as a dictionary
    def __getitem__(self,*args,**kwargs):
        return self.__dict__.__getitem__(*args,**kwargs)
    def __setitem__(self,*args,**kwargs):
        return self.__dict__.__setitem__(*args,**kwargs)
    def __str__(self):
        return self.__dict__.__str__()
    def __repr__(self):
        return self.__dict__.__repr__()

def take_from_list(l,b):
    """Return part of a list.\n\nReturn elements of l for which b is
    true. l and b must be the same length."""
    return [ll for (ll,bb) in zip(l,b) if bb]

def digitise_postscript_figure(
        filename,
        xydpi_xyvalue0 = None,  # ((xdpi,ydpi),(xvalue,yvalue)) for fixing axes
        xydpi_xyvalue1 = None,  # 2nd point for fixing axes
        
):
    """Get all segments in a postscript file. That is, an 'm' command
    followed by an 'l' command. Could find points if 'm' without an 'l' or
    extend to look for 'moveto' and 'lineto' commands."""
    data = file_to_string(filename).split() # load as list split on all whitespace
    retval = []                  # line segments
    ## loop through looking for line segments
    i = 0
    while (i+3)<len(data):
        if data[i]=='m' and data[i+3]=='l': # at least one line segment
            x,y = [float(data[i-1])],[-float(data[i-2])]
            while (i+3)<len(data) and data[i+3]=='l':
                x.append(float(data[i+2]))
                y.append(-float(data[i+1]))
                i += 3
            retval.append([x,y])
        i += 1
    ## make into arrays
    for t in retval:
        t[0],t[1] = np.array(t[0],ndmin=1),np.array(t[1],ndmin=1)
    ## transform to match axes if possible
    if xydpi_xyvalue0 is not None:
        a0,b0,a1,b1 = xydpi_xyvalue0[0][0],xydpi_xyvalue0[1][0],xydpi_xyvalue1[0][0],xydpi_xyvalue1[1][0]
        m = (b1-b0)/(a1-a0)
        c = b0-a0*m
        xtransform = lambda t,c=c,m=m:c+m*t
        a0,b0,a1,b1 = xydpi_xyvalue0[0][1],xydpi_xyvalue0[1][1],xydpi_xyvalue1[0][1],xydpi_xyvalue1[1][1]
        m = (b1-b0)/(a1-a0)
        c = b0-a0*m
        ytransform = lambda t,c=c,m=m:c+m*t
        for t in retval:
            t[0],t[1] = xtransform(t[0]),ytransform(t[1])
    return(retval)
   
################################################################################
## things for plotting #########################################################
################################################################################

## standard papersize for figures - in inches
papersize=Dict_Object(dict(
    a4=(8.3,11.7),
    a4_portrait=(8.3,11.7),
    a4_landscape=(11.7,8.3),
    a5=(5.87,8.3),
    a5landscape=(8.3,5.87),
    letter=(8.5,11),
    letter_portrait=(8.5,11),
    letter_landscape=(11,8.5),
    latexTiny=(constants.golden*1.2,1.2),
    latexSmall=(constants.golden*1.7,1.7),
    latexMedium=(constants.golden*3.,3.),
    latexLarge=(constants.golden*4.,4.),
    squareMedium=(5.,5.),
    squareLarge=(8.,8.),
    article_full_page_width=6.77,
    # article_single_column_width=3.27,
        article_single_column_width=3.5,
    article_full_page_height=8.66,
))

def presetRcParams(
        preset='base',      # name of the preset to use
        make_fig=False, # make a figure and axes and return (fig,ax)
        **params, # a dictionay containing any valid rcparams, or abbreviated by removing xxx.yyy. etc
):
    """Call this function wit some presets before figure object is
    created. If make_fig = True return (fig,ax) figure and axis
    objects. Additional kwargs are applied directly to rcParams"""
    ## try and get screen size
    try:
        xscreen,yscreen = get_screensize() # pts
        xscreen,yscreen = xscreen/mpl.rcParams['figure.dpi'],yscreen/mpl.rcParams['figure.dpi'] # inches
    except:
        warnings.warn("could not get screensize")
        xscreen,yscreen = 5,5
    ## dicitionary of dictionaries containing keyval pairs to
    ## substitute into rcParams
    presets = dict()
    ## the base params
    presets['base'] = {
        'legend.handlelength'  :1.5,
        'legend.handletextpad' :0.4,
        'legend.labelspacing'  :0.,
        # 'legend.loc'           :'best', # setting this to best makes things slooooow
        'legend.numpoints'     :1,
        'font.family'          :'serif',
        'text.usetex'          :False,
        'text.latex.preamble'  :[
            r'\usepackage{mhchem}',
            r'\usepackage[np]{numprint}\npdecimalsign{\ensuremath{.}} \npthousandsep{\,} \npproductsign{\times} \npfourdigitnosep ',
        ],
        'mathtext.fontset'     :'cm',
        'lines.markeredgewidth': 1,
        # 'axes.prop_cycle': cycler('color',linecolors_colorblind_safe),
        'axes.prop_cycle': cycler('color',linecolors_print),
        # 'axes.color_cycle': linecolors_colorblind_safe,
        'patch.edgecolor': 'none',
        'xtick.minor.top': True,
        'xtick.minor.bottom': True,
        'xtick.minor.visible': True ,
        'xtick.top': True ,
        'xtick.bottom': True ,
        'ytick.minor.right': True,
        'ytick.minor.left': True,
        'ytick.minor.visible': True ,
        'ytick.right': True ,
        'ytick.left': True ,
        'path.simplify'      :False, # whether or not to speed up plots by joining line segments
        'path.simplify_threshold' :1, # how much to do so
        'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
        ## set up axes tick marks and labels
        'axes.formatter.limits' : (-3, 6), # use scientific notation if log10 of the axis range is smaller than the first or larger than the second
        'axes.formatter.use_mathtext' : True, # When True, use mathtext for scientific notation.
        'axes.formatter.useoffset'      : False,    # If True, the tick label formatter # will default to labeling ticks relative # to an offset when the data range is # small compared to the minimum absolute # value of the data.
        'axes.formatter.offset_threshold' : 4,     # When useoffset is True, the offset # will be used when it can remove # at least this number of significant # digits from tick labels.
        }

    presets['screen']=deepcopy(presets['base'])
    presets['screen'].update({
        'figure.figsize'     :(xscreen,yscreen),
        # 'figure.figsize'     :(10,10),
        'figure.subplot.left':0.05,
        'figure.subplot.right':0.95,
        'figure.subplot.bottom':0.05,
        'figure.subplot.top':0.95,
        'figure.subplot.wspace':0.2,
        'figure.subplot.hspace':0.2,
        'figure.autolayout'  : True, # reset tight_layout everytime figure is redrawn -- seems to cause problems with long title and label strings
        # 'toolbar'  :'none' , # hides toolbar but also disables keyboard shortcuts
        'legend.handlelength':4,
        'text.usetex'        :False,
        'lines.linewidth'    : 1,
        'lines.markersize'   : 10.0,
        'grid.alpha'         : 1.0,
        'grid.color'         : 'gray',
        'grid.linestyle'     : ':',
        'legend.fontsize'    :9.,
        'axes.titlesize'     :14.,
        'axes.labelsize'     :14.,
        'xtick.labelsize'    :14.,
        'ytick.labelsize'    :14.,
        'font.size'          :14.,
        'axes.prop_cycle'    : cycler('color',linecolors_screen),
        'path.simplify'      :True , # whether or not to speed up plots by joining line segments
        'path.simplify_threshold' :1, # how much to do so
        'agg.path.chunksize': 10000,  # antialisin speed up -- does not seem to do anything over path.simplify
    })

    presets['article_single_column']=deepcopy(presets['base'])
    presets['article_single_column'].update({
        'text.usetex'          :False,
        'figure.figsize'       :(papersize['article_single_column_width'],papersize['article_single_column_width']/constants.golden_ratio),
        # 'lines.linewidth'    : 0.5,
        'lines.linewidth'    : 1,
        'figure.subplot.left'  :0.15,
        'figure.subplot.right' :0.96,
        'figure.subplot.bottom':0.19,
        'figure.subplot.top'   :0.92,
        'figure.subplot.wspace':0.35,
        'figure.subplot.hspace':0.3,
        'legend.fontsize'      :9.,
        'axes.titlesize'       :10.,
        'axes.labelsize'       :10.,
        'lines.markersize'     :4.,
        'xtick.labelsize'      :9.,
        'ytick.labelsize'      :9.,
        'font.size'            :10.,
        # 'axes.formatter.use_mathtext': True, # use math text for scientific notation . i.e,. not 1e-9
    })

    presets['articleSingleColumn'] = presets['article_single_column'] # deprecated
    presets['articleSingleColumnSmall']=deepcopy(presets['articleSingleColumn'])
    presets['articleSingleColumnSmall'].update({
            'figure.figsize':(3.,2.),
            })

    presets['article_single_column_one_third_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_one_third_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']/3.),
            'figure.subplot.bottom':0.15,
            'figure.subplot.top'   :0.95,
            })

    presets['article_single_column_half_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_half_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']/2.),
            'figure.subplot.bottom':0.1,
            'figure.subplot.top'   :0.95,
            })
    presets['articleSingleColumnTwoSubfigures'] = presets['article_single_column_half_page'] # deprecated

    presets['article_single_column_two_thirds_page']=deepcopy(presets['article_single_column_half_page'])
    presets['article_single_column_two_thirds_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']*2./3.),
            'figure.subplot.bottom':0.1,
            'figure.subplot.top'   :0.95,
            })

    presets['article_single_column_full_page']=deepcopy(presets['article_single_column'])
    presets['article_single_column_full_page'].update({
            'figure.figsize':(papersize['article_single_column_width'],papersize['article_full_page_height']),
            'figure.subplot.bottom':0.05,
            'figure.subplot.top'   :0.97,
            })

    presets['article_double_column']=deepcopy(presets['article_single_column'])
    presets['article_double_column'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_single_column_width']),
            'figure.subplot.left':0.1,
            'lines.linewidth'    : 0.5,
            'figure.subplot.bottom':0.14,})
    presets['articleDoubleColumn'] = presets['article_double_column']
    
    presets['article_double_column_half_page']=deepcopy(presets['article_double_column'])
    presets['article_double_column_half_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']/2.),
            # 'figure.figsize':(6.,5.),
            'figure.subplot.left':0.1,
            'figure.subplot.right':0.95,
            'figure.subplot.bottom':0.10,
            'figure.subplot.top':0.95,
            'figure.subplot.wspace':0.3,
            'figure.subplot.hspace':0.3,})

    presets['article_double_column_two_thirds_page']=deepcopy(presets['article_double_column'])
    presets['article_double_column_two_thirds_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']*2./3.),
            'figure.subplot.left':0.1,
            'figure.subplot.right':0.95,
            'figure.subplot.bottom':0.07,
            'figure.subplot.top':0.95,
            'figure.subplot.wspace':0.3,
            'figure.subplot.hspace':0.3,
            })

    presets['article_full_page']=deepcopy(presets['article_double_column'])
    presets['article_full_page'].update({
            'figure.figsize':(papersize['article_full_page_width'],papersize['article_full_page_height']),
            'figure.figsize':(6.9,9.2),
            'figure.subplot.bottom':0.05,
            'figure.subplot.top'   :0.97,
            'figure.subplot.hspace':0.15,
            'figure.subplot.left':0.1,
            })

    presets['article_full_page_landscape']=deepcopy(presets['article_full_page'])
    presets['article_full_page_landscape'].update({
            'figure.figsize':(papersize['article_full_page_height'],papersize['article_full_page_width']),
            'figure.subplot.left':0.05,
            'figure.subplot.right'   :0.97,
            'figure.subplot.bottom':0.07,
            })

    presets['beamer_base']=deepcopy(presets['base'])
    presets['beamer_base'].update({
        'text.usetex'          :False,
        'font.size'            :8,
        'xtick.labelsize'      :8,
        'ytick.labelsize'      :8,
        'ytick.labelsize'      :8,
        'lines.linewidth'      :0.5,
        'lines.markersize'     :4,
    })

    ## good for a simgle image slide
    presets['beamer_large']=deepcopy(presets['base'])
    presets['beamer_large'].update({
        'figure.figsize'       :(4.5,2.5), # 5.0393701,3.7795276 beamer size
        'figure.subplot.left':0.15,
        'figure.subplot.right':0.92,
        'figure.subplot.bottom':0.17,
        'figure.subplot.top'   :0.93,
        'figure.subplot.wspace':0.20,
        'figure.subplot.hspace':0.37,
        'xtick.labelsize'      :8.,
        'ytick.labelsize'      :8.,
        'ytick.labelsize'      :8.,
    })

    presets['beamer_large_twinx']=deepcopy(presets['beamer_large'])
    presets['beamer_large_twinx'].update({
        'figure.figsize'       :(4.5,2.5), # 5.0393701,3.7795276 beamer size
        'figure.subplot.left':0.15,
        'figure.subplot.right':0.85,
    })

    ## good for single imag ewith more text
    presets['beamer_medium']=deepcopy(presets['beamer_base'])
    presets['beamer_medium'].update({
            'figure.figsize'       :(constants.golden*1.8,1.8),
            'figure.subplot.left'  :0.18,
            'figure.subplot.right' :0.95,
            'figure.subplot.bottom':0.2,
            'figure.subplot.top'   :0.9,
            'figure.subplot.wspace':0.20,
            'figure.subplot.hspace':0.37,
            })

    ## good to fill one quadrant
    presets['beamer_small']=deepcopy(presets['beamer_base'])
    presets['beamer_small'].update({
        'figure.figsize'       :(2.25,2.25/constants.golden),
        'figure.subplot.left'  :0.25,
        'figure.subplot.right' :0.95,
        'figure.subplot.bottom':0.25,
        'figure.subplot.top'   :0.95,
        'figure.subplot.wspace':0.20,
        'figure.subplot.hspace':0.37,
        'axes.labelpad': 1,
    })

    ## maximise
    presets['beamer_entire_slide']=deepcopy(presets['beamer_large'])
    presets['beamer_entire_slide'].update({
        'figure.figsize'       :(5.0393701,3.7795276), 
    })

    ## fit more text beside
    presets['beamer_wide']=deepcopy(presets['beamer_large'])
    presets['beamer_wide'].update({
            'figure.figsize'       :(4.5,1.5),
            'figure.subplot.bottom':0.25,
            })

    ## fit more text beside
    presets['beamer_tall']=deepcopy(presets['beamer_large'])
    presets['beamer_tall'].update({
            'figure.figsize'       :(2.2,3.2),
            'figure.subplot.left':0.25,
            'figure.subplot.bottom':0.14,
            'figure.subplot.top':0.95,
            })

    ## good for two column
    presets['beamer_half_width']=deepcopy(presets['beamer_base'])
    presets['beamer_half_width'].update({
            'figure.figsize'       :(2.25,2.5),
            'figure.subplot.left':0.2,
            'figure.subplot.bottom':0.15,
            'figure.subplot.top':0.95,
            'figure.subplot.right':0.95,
            })

    presets['a4_portrait'] = deepcopy(presets['base'])
    presets['a4_portrait'].update({
            'text.usetex'          :False,
            'figure.figsize':papersize.a4_portrait,
            'figure.subplot.left':0.11,
            'figure.subplot.right':0.92,
            'figure.subplot.top':0.94,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            'lines.markersize':2.,
            'legend.fontsize':'large',
            'font.size':10,
            })

    presets['a4_landscape'] = deepcopy(presets['a4_portrait'])
    presets['a4_landscape'].update({
            'figure.figsize':papersize.a4_landscape,
            'figure.subplot.left':0.07,
            'figure.subplot.right':0.95,
            'figure.subplot.top':0.95,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })



    presets['letter_portrait'] = deepcopy(presets['a4_portrait'])
    presets['letter_portrait'].update({
            'figure.figsize':papersize.letter_portrait,
            'figure.subplot.left':0.11,
            'figure.subplot.right':0.92,
            'figure.subplot.top':0.94,
            'figure.subplot.bottom':0.08,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })

    presets['letter_landscape'] = deepcopy(presets['a4_portrait'])
    presets['letter_landscape'].update({
            'figure.figsize':papersize.letter_landscape,
            'figure.subplot.left':0.08,
            'figure.subplot.right':0.93,
            'figure.subplot.top':0.92,
            'figure.subplot.bottom':0.1,
            'figure.subplot.wspace':0.2,
            'figure.subplot.hspace':0.2,
            })

    ## synonyms 
    presets['a4'] = presets['a4_portrait']
    presets['letter'] = presets['letter_portrait']
    presets['a4landscape'] = presets['a4_landscape'] # deprecated
    presets['a4portrait'] = presets['a4_portrait']   # deprecated
    ## find key in presets to match the requested preset
    for key in presets[preset]:
        mpl.rcParams[key] = presets[preset][key]
    ## extra keys -- potentially with assuming prefixes in rcParams
    ## hierarchy until an existing key is found
    for key,val in params.items():
        for prefix in ('','figure.','figure.subplot.','axes.',
                        'lines.','font.','xtick.','ytick.',): 
            if prefix+key in mpl.rcParams:
                mpl.rcParams[prefix+key] = val
                break
        else:
            raise Exception(f"Could not interpret rcParam: {repr(key)}")
    ## override paerasm defined as kwargs, shortcuts for common option
    # if figsize    is not None: mpl.rcParams['figure.figsize']        = figsize
    # if usetex     is not None: mpl.rcParams['text.usetex']           = usetex
    # if left       is not None: mpl.rcParams['figure.subplot.left']   = left
    # if right      is not None: mpl.rcParams['figure.subplot.right']  = right
    # if bottom     is not None: mpl.rcParams['figure.subplot.bottom'] = bottom
    # if top        is not None: mpl.rcParams['figure.subplot.top']    = top
    # if wspace     is not None: mpl.rcParams['figure.subplot.wspace'] = wspace
    # if hspace     is not None: mpl.rcParams['figure.subplot.hspace'] = hspace
    # if linewidth  is not None: mpl.rcParams['lines.linewidth']       = linewidth
    # if autolayout is not None: mpl.rcParams['figure.autolayout']     = autolayout
    ## create figure and axes objects
    if make_fig:
        fig = plt.figure()
        ax = fig.gca()
        return(fig,ax,)

def newfig():
    """Make a new figure"""
    n = 1
    while plt.fignum_exists(n):
        n = n+1
    fig(n=n)
    
def fig(
        n=None,
        preset_rcparams='screen',
        figsize=None,
        hide_toolbar=True,
        # tight_layout=True,
        clear_figure=True,
        fullscreen=False,
        **preset_rcparams_kwargs):
    """quick figure preparation."""
    presetRcParams(preset_rcparams,**preset_rcparams_kwargs)
    if n is None:
        n = plt.gcf().number + 1 
    fig = plt.figure(n);
    if fullscreen:
        set_figsize_fullscreen()
    if clear_figure:
        fig.clf()
    # fig.set_tight_layout(tight_layout)
    fig._my_fig = True          # use this as a marker that this figure was created by this function 
    newcolor(reset=True)
    newlinestyle(reset=True)
    newmarker(reset=True)
    # extra_interaction()
    if figsize=='full screen':
        set_figsize_fullscreen(fig=fig)
    elif figsize is not None:
        set_figsize_in_pixels(*figsize,fig=fig)
    if hide_toolbar:
        # from PyQt5 import QtWidgets 
        # from PyQt4 import QtGui as QtWidgets 
        try:
            win = fig.canvas.manager.window
        except AttributeError:
            win = fig.canvas.window()
        # toolbar = win.findChild(QtWidgets.QToolBar)
        # toolbar.setVisible(False)
    ax = fig.gca()
    def format_coord(x,y):
        if x<1e-5 or x>1e5: xstr = f'{x:0.18e}'
        else:               xstr = f'{x:0.18f}'
        if y<1e-5 or y>1e5: ystr = f'{y:0.18e}'
        else:               ystr = f'{y:0.18f}'
        return(f'x={xstr:<25s} y={ystr:<25s}')
    ax.format_coord = format_coord
    if preset_rcparams=='scree':
        ax.grid(True,color='gray')
    return(fig,ax)

def figax(*args,**kwargs):
    f = fig(*args,**kwargs)
    ax = f.gca()
    return(f,ax)

def get_screensize():
    """In pixels."""
    status,output = subprocess.getstatusoutput(r"set_window.py get_current_screen_dimensions")
    if status!=0:
        raise Exception("Could not determine screensize")
    x,y = output.split()
    return(int(x),int(y))

def set_figsize_fullscreen(fig=None):
    """Set figsize in pixels for screen display aiming for full
    screen."""
    try:
        x,y = get_screensize()
        return(set_figsize_in_pixels(
            x=x,
            y=y-55,        # a bit less to fit in toolbar
            fig=fig))
    except:
        # pass
        warnings.warn('set_figsize_fullscreen failed')
        

def set_figsize_in_pixels(x,y,fig=None):
    """Set figsize in pixels for screen display."""
    if fig is None:
        fig = plt.gcf()
    dpi = fig.get_dpi()
    figsize=(x/dpi,y/dpi)
    fig.set_size_inches(*figsize)
    return(figsize)

def extra_interaction(fig=None,lines_picker=5):
    """Call this to customise the matplotlib interactive gui experience
    for this figure."""
    ## define a bunch of functions, the action comes at the end 
    ## select an axes
    def select_axes(axes):
        if axes is None: return # sometimes happens for some reason
        ## initialise dictionrary to store useful data
        if not hasattr(axes,'my_extra_interaction'):
            axes.my_extra_interaction = dict(selected_line = None, selected_line_annotation = None, currently_selecting_points = False,) 
        ## set all lines to requested picker values
        if lines_picker is not None:
            for line in axes.lines:
                line.set_picker(lines_picker)
        ## set this as selected axes
        fig.my_extra_interaction['axes'] = axes
    ## some data stored in figure to facilitate actions below
    ## what to do when a deselected line is picked
    def select_line(line):
        print(line.get_label())
        axes = fig.my_extra_interaction['axes']
        axes.my_extra_interaction['selected_line_annotation'] = annotate_corner(line.get_label(), ax=axes, fontsize='large')
        axes.my_extra_interaction['selected_line'] = line
        line.set_linewidth(line.get_linewidth()*2)
        line.set_markersize(line.get_markersize()*2)
        plt.draw()
    ## what to do when a selected line is picked
    def deselect_line(line):    
        axes = fig.my_extra_interaction['axes']
        line.set_linewidth(line.get_linewidth()/2)
        line.set_markersize(line.get_markersize()/2)
        if axes.my_extra_interaction['selected_line_annotation']!=None:
            axes.my_extra_interaction['selected_line_annotation'].remove()
        axes.my_extra_interaction['selected_line_annotation'] = None
        axes.my_extra_interaction['selected_line'] = None
    ## if selecting previously selected line, unselect it, else select it
    def on_button_press(event):
        if event.inaxes: fig.sca(event.inaxes) # set clicked axes to gca
        select_axes(event.inaxes)
    ## on picking of line etc
    def on_pick(event):
        axes = fig.my_extra_interaction['axes']
        line = event.artist
        if axes.my_extra_interaction['currently_selecting_points'] is True:
            pass
        elif axes.my_extra_interaction['selected_line'] is None:
            select_line(line)
        elif line==axes.my_extra_interaction['selected_line']:
            deselect_line(line)
        else:
            deselect_line(axes.my_extra_interaction['selected_line'])
            select_line(line)
        plt.draw()
        return
    ## key options
    def on_key(event):
        axes = fig.my_extra_interaction['axes']
        ## delete line
        if event.key=='d':
            if axes.my_extra_interaction['selected_line'] is not None:
                line = axes.my_extra_interaction['selected_line']
                deselect_line(line)
                line.set_visible(False)
        ## autoscale
        elif event.key=='a':
            axes.autoscale(enable=True,axis='both')
        ## zoom to all data
        elif event.key=='z': 
            lines  = (axes.get_lines()
                      if axes.my_extra_interaction['selected_line'] is None
                      else [axes.my_extra_interaction['selected_line']])  # use all lines if not selected
            xmin,xmax,ymin,ymax = np.inf,-np.inf,np.inf,-np.inf
            for line in lines:
                if not line.get_visible(): continue
                if not isinstance(line.get_xdata(),np.ndarray): continue # hack to avoid things I dont know what they are
                xmin = min(xmin,line.get_xdata().min())
                xmax = max(xmax,line.get_xdata().max())
                ymin = min(ymin,line.get_ydata().min())
                ymax = max(ymax,line.get_ydata().max())
            if not np.isinf(ymin): axes.set_ylim(ymin=ymin)
            if not np.isinf(ymax): axes.set_ylim(ymax=ymax)
        ## zoom to full yscale
        elif event.key=='y':
            lines  = (axes.get_lines()
                      if axes.my_extra_interaction['selected_line'] is None
                      else [axes.my_extra_interaction['selected_line']])  # use all lines if not selected
            xmin,xmax = axes.get_xlim()
            ymin,ymax = np.inf,-np.inf
            for line in lines:
                if not line.get_visible(): continue
                if not isinstance(line.get_xdata(),np.ndarray): continue # hack to avoid things I dont know what they are
                i = find((line.get_xdata()>=xmin)&(line.get_xdata()<=xmax))
                if not any(i): continue
                ymin = min(ymin,(line.get_ydata()[i]).min())
                ymax = max(ymin,(line.get_ydata()[i]).max())
            if not np.isinf(ymin): axes.set_ylim(ymin=ymin)
            if not np.isinf(ymax): axes.set_ylim(ymax=ymax)
        ## zoom to full xscale 
        elif event.key=='x': 
            lines  = (axes.get_lines()
                      if axes.my_extra_interaction['selected_line'] is None
                      else [axes.my_extra_interaction['selected_line']])  # use all lines if not selected
            xmin,xmax = np.inf,-np.inf
            ymin,ymax = axes.get_ylim()
            for line in lines:
                if not line.get_visible(): continue
                if not isinstance(line.get_xdata(),np.ndarray): continue # hack to avoid things I dont know what they are
                i = (line.get_ydata()>=ymin)&(line.get_ydata()<=ymax) # get only data in current ylim
                if not any(i): continue
                xmin = min(xmin,np.min(line.get_xdata()[i])) # get new limits
                xmax = max(xmax,np.max(line.get_xdata()[i]))
            if not np.isinf(xmin): axes.set_xlim(xmin=xmin) # set limits
            if not np.isinf(xmax): axes.set_xlim(xmax=xmax)
        ## slect points
        # elif event.key=='p':
            # if axes.my_extra_interaction['currently_selecting_points'] is False:
                # axes.my_extra_interaction['currently_selecting_points'] = True
            # elif axes.my_extra_interaction['currently_selecting_points'] is True:
                # axes.my_extra_interaction['currently_selecting_points'] = False
            # points = fig.ginput(n=-1,timeout=-1,show_clicks=True)
            # print('\n'.join(['{:0.15g} {:0.15g}'.format(*t) for t in points]))
            # axes.my_extra_interaction['currently_selecting_points'] = False
        ## move with arrow keys
        elif event.key=='right':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)*0.2
            axes.set_xlim(xmin+shift,xmax+shift)
        elif event.key=='left':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)*0.2
            axes.set_xlim(xmin-shift,xmax-shift)
        elif event.key=='up':
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)*0.2
            axes.set_ylim(ymin+shift,ymax+shift)
        elif event.key=='down':
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)*0.2
            axes.set_ylim(ymin-shift,ymax-shift)
        ## zoom with arrow keys
        elif event.key=='shift+right':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)/2.
            axes.set_xlim(xmin-shift,xmax+shift)
        elif event.key=='shift+left':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)/4.
            axes.set_xlim(xmin+shift,xmax-shift)
        elif event.key=='shift+up':
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)/2.
            axes.set_ylim(ymin-shift,ymax+shift)
        elif event.key=='shift+down':
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)/4.
            axes.set_ylim(ymin+shift,ymax-shift)
        ## zoom with +/=/- keys
        elif event.key=='+' or event.key=='=':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)/4.
            axes.set_xlim(xmin+shift,xmax-shift)
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)/4.
            axes.set_ylim(ymin+shift,ymax-shift)
        elif event.key=='-':
            xmin,xmax = axes.get_xlim()
            shift = (xmax-xmin)/2.
            axes.set_xlim(xmin-shift,xmax+shift)
            ymin,ymax = axes.get_ylim()
            shift = (ymax-ymin)/2.
            axes.set_ylim(ymin-shift,ymax+shift)
        ## redraw
        plt.draw()
        return
    ## select figure
    if fig is None: fig = plt.gcf() # determine figure object
    if hasattr(fig,'my_extra_interaction'):
        return                  # extra interaction already set up, do not add again
    else:
        fig.my_extra_interaction = {}
        select_axes(fig.gca())
    ## watch for events
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

def autoscale_y(ax=None):
    """Autoscale y axis to fit data within current x limits."""
    if ax==None: ax = plt.gca()
    ax.set_ylim(
        np.nanmin([np.nanmin(line.get_ydata()[inrange(line.get_xdata(),*ax.get_xlim())]) for line in ax.lines]),
        np.nanmax([np.nanmax(line.get_ydata()[inrange(line.get_xdata(),*ax.get_xlim())]) for line in ax.lines]),
        )
    
_newcolor_nextcolor=0
linecolors_screen=(
    'red',
    'blue',
    'green',
    'black',
    'orange',
    'magenta',
    'aqua',
    'indigo',
    'brown',
    ## 'grey',
    ## 'aliceblue',
    ## 'aquamarine',
    ## 'azure',
    ## 'beige',
    ## 'bisque',
    ## 'blanchedalmond',
    ## 'blue',
    ## 'blueviolet',
    ## 'brown',
    'burlywood',
    'cadetblue',
    'chartreuse',
    'chocolate',
    'coral',
    'cornflowerblue',
    ## 'cornsilk',
    'crimson',
    'cyan',
    # 'darkblue',
    # 'darkcyan',
    # 'darkgoldenrod',
    # 'darkgray',
    # 'darkgreen',
    # 'darkkhaki',
    # 'darkmagenta',
    # 'darkolivegreen',
    # 'darkorange',
    # 'darkorchid',
    # 'darkred',
    # 'darksalmon',
    # 'darkseagreen',
    # 'darkslateblue',
    # 'darkslategray',
    # 'darkturquoise',
    # 'darkviolet',
    # 'deeppink',
    # 'deepskyblue',
    # 'dimgray',
    # 'dodgerblue',
    # 'firebrick',
    # 'forestgreen',
    # 'fuchsia',
    # 'gainsboro',
    # 'gold',
    # 'goldenrod',
    # 'gray',
    # 'green',
    # 'greenyellow',
    # 'honeydew',
    # 'hotpink',
    # 'indianred',
    # 'indigo',
    # 'ivory',
    # 'khaki',
    # 'lavender',
    # 'lavenderblush',
    # 'lawngreen',
    # 'lemonchiffon',
    # 'lightblue',
    # 'lightcoral',
    # 'lightcyan',
    # 'lightgoldenrodyellow',
    # 'lightgreen',
    # 'lightgray',
    # 'lightpink',
    # 'lightsalmon',
    # 'lightseagreen',
    # 'lightskyblue',
    # 'lightslategray',
    # 'lightsteelblue',
    # 'lightyellow',
    # 'lime',
    # 'limegreen',
    # 'linen',
    # 'magenta',
    # 'maroon',
    # 'mediumaquamarine',
    # 'mediumblue',
    # 'mediumorchid',
    # 'mediumpurple',
    # 'mediumseagreen',
    # 'mediumslateblue',
    # 'mediumspringgreen',
    # 'mediumturquoise',
    # 'mediumvioletred',
    # 'midnightblue',
    # 'mintcream',
    # 'mistyrose',
    # 'moccasin',
    # 'navajowhite',
    # 'olive',
    # 'olivedrab',
    # 'orange',
    # 'orangered',
    # 'orchid',
    # 'palegoldenrod',
    # 'palegreen',
    # 'paleturquoise',
    # 'palevioletred',
    # 'papayawhip',
    # 'peachpuff',
    # 'peru',
    # 'pink',
    # 'plum',
    # 'powderblue',
    # 'purple',
    # 'red',
    # 'rosybrown',
    # 'royalblue',
    # 'saddlebrown',
    # 'salmon',
    # 'sandybrown',
    # 'seagreen',
    # 'seashell',
    # 'sienna',
    # 'silver',
    # 'skyblue',
    # 'slateblue',
    # 'slategray',
    # 'snow',
    # 'springgreen',
    # 'steelblue',
    # 'tan',
    # 'teal',
    # 'thistle',
    # 'tomato',
    # 'turquoise',
    # 'violet',
    # 'wheat',
    # 'yellow',
    # 'yellowgreen',
    # # 'floralwhite', 'ghostwhite', 'navy','oldlace', 'white','whitesmoke','antiquewhite',
)

linecolors_colorblind_safe=(
    (204./256.,102./256.,119./256.),
    (61./256., 170./256.,153./256.),
    (51./256., 34./256., 136./256.),
    ## (17./256., 119./256.,51./256.),
    (170./256.,68./256., 153./256.),
    ## (136./256.,34./256., 85./256.),
    (153./256.,153./256.,51./256.),
    (136./256.,204./256.,238./256.),
    (221./256.,204./256.,199./256.),
    (51./256., 102./256.,170./256.),
    (17./256., 170./256.,153./256.),
    (102./256.,170./256.,85./256.),
    (153./256.,34./256., 136./256.),
    (238./256.,51./256., 51./256.),
    (238./256.,119./256.,34./256.),
    ## (204./256.,204./256.,85./256.),
    ## (255./256.,238./256.,51./256.),
    ## (119./256.,119./256.,119./256.),
)   ## from http://www.sron.nl/~pault/

## from http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=6
linecolors_print=(
    # ## attempt1
    # '#a50026',
    # '#f46d43',
    # '#fdae61',
    # '#fee090',
    # '#74add1',
    # '#4575b4',
    # '#4575b4',
    # '#313695',
    # '#d73027',
    # '#abd9e9',
    # '#e0f3f8',
    ## attempt 2
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#a65628',
    '#f781bf',
    '#ffff33', # light yellow
)

# linecolors = mpl.rcParams['axes.color_cycle']
linecolors = [f['color'] for f in mpl.rcParams['axes.prop_cycle']]

def newcolor(index=None,reset=None,linecolors=None):
    """Retuns a color string, different to the last one, from the list
    linecolors. If reset is set, returns to first element of
    linecolors, no color is returned. If index (int) is supplied
    return this color. If index is supplied and reset=True, set index
    to this color. """
    global _newcolor_nextcolor
    if linecolors is None:
        linecolors = [f['color'] for f in mpl.rcParams['axes.prop_cycle']]
        # linecolors = [f for f in mpl.rcParams['axes.color_cycle']]
    if reset!=None or index in ['None','none','']:
        _newcolor_nextcolor=0
        return
    if index is not None:
        ## index should be an int -- but make it work for anything
        try:
            index = int(index)
        except (TypeError,ValueError):
             # index = id(index)
             index = hash(index)
        if reset:
            _newcolor_nextcolor = (index) % len(linecolors)
        return(linecolors[(index) % len(linecolors)])

    retval = linecolors[_newcolor_nextcolor]
    _newcolor_nextcolor = (_newcolor_nextcolor+1) % len(linecolors)
    return retval


_newlinestyle_nextstyle=0
linestyles=('solid','dashed','dotted','dashdot')
def newlinestyle(index=None,reset=None):
    """Retuns a style string, different to the last one, from the list
    linestyles. If reset is set, returns to first element of
    linestyles, no style is returned."""
    global linestyles,_newlinestyle_nextstyle
    if reset!=None:
        _newlinestyle_nextstyle=0
        return
    if index is not None:
        _newlinestyle_nextstyle = (index) % len(linestyles)
    retval = linestyles[_newlinestyle_nextstyle]
    _newlinestyle_nextstyle = (_newlinestyle_nextstyle+1) % len(linestyles)
    return retval

_newmarker_nextmarker=0
# markers=('o','x','t','d')
markers=("o","s","d","p","+","x","v","^","<",">","1","2","3","4","8","*","h","H","D","|","_",)
def newmarker(index=None,reset=None):
    """Retuns a marker type string, different to the last one, from
    the list markers. If reset is set, returns to first element of
    markers, no style is returned."""
    global markers,_newmarker_nextmarker
    if reset!=None:
        _newmarker_nextmarker=0
        return 
    if index is not None:
        index = int(index)
        _newmarker_nextmarker = (index) % len(markers)
    retval = markers[_newmarker_nextmarker]
    _newmarker_nextmarker = (_newmarker_nextmarker+1) % len(markers)
    return retval

def newcolorlinestyle(index=None,reset=None):
    """Returns a new color and line style if necessary. Cycles colors
first."""
    ## reset built in counters and do nothing else
    if reset is not None:
        return(newcolor(reset=reset),newlinestyle(reset=reset))
    ## return combination according to an index
    if index is not None:
        color_index = index%len(linecolors)
        linestyle_index = (index-color_index)%len(linestyles)
        return(linecolors[color_index],linestyles[linestyle_index])
    ## iterate and get a new combination
    if _newcolor_nextcolor==len(linecolors)-1:
        return(newcolor(),newlinestyle())
    else:
        return(newcolor(),linestyles[_newlinestyle_nextstyle])
    
# def newcolormarker(reset=None):
#     """Returns a new color and line style if necessary."""
#     if reset is not None:
#         return(newcolor(reset=reset),newmarker(reset=reset))
#     if _newcolor_nextcolor==len(linecolors)-1:
#         return(newcolor(),newmarker())
#     else:
#         return(newcolor(),markers[_newmarker_nextmarker])

def newcolormarker(reset=None):
    """Returns a new color and line style if necessary."""
    if reset is not None:
        newcolor(reset=reset)
        newmarker(reset=reset)
    if _newcolor_nextcolor==len(linecolors)-1:
        color,marker = newcolor(),newmarker()
    else:
        color,marker = newcolor(),markers[_newmarker_nextmarker]
    return({'color':color,'marker':marker})

# def subplot(*args,**kwargs):
    # """Work out reasonable dimensions for an array of subplots with
    # total number n.\n
    # subplot(i,j,n) - nth subplot of i,j grid, like normal\n
    # subplot(i*j,n)   - switch to nth subplot out of a total of i*j in a sensible arrangement\n
    # subplot((i,j)) - add next subplot in i,j grid\n
    # subplot(i*j)   - total number of subplots, guess a good arrangment\n
    # All other kwargs are passed onto pyplot.subplot."""
    # assert len(args) in [1,2,3], 'Bad number of inputs.'
    # if 'fig' in kwargs:
        # fig = kwargs.pop('fig')
    # else:
        # fig=plt.gcf();
    # ## determine next subplot from how many are already drawn
    # if len(args)==1: 
        # if isinstance(args[0],tuple):
            # args=(args[0][0],args[0][1],len(fig.axes)+1,)
        # else:
            # args=(args[0],len(fig.axes)+1,)
    # ## send straight to subplot
    # if len(args)==3:
        # return fig.add_subplot(*args,**kwargs)
    # ## nice rounded dimensions
    # nsubplots,isubplot = args
    # rows = int(np.floor(np.sqrt(nsubplots)))
    # columns = int(np.ceil(float(nsubplots)/float(rows)))
    # ## if landscapish reverse rows/columns
    # if fig.get_figheight()>fig.get_figwidth(): (rows,columns,)=(columns,rows,)
    # return fig.add_subplot(rows,columns,isubplot,**kwargs)
# add_subplot=mysubplot=subplot
# mySubplot=subplot

def subplot(
        n=None,                 # subplot index, begins at 1, if None adds a new subplot
        ncolumns=None,          # how many colums (otherwise adaptive)
        nrows=None,          # how many colums (otherwise adaptive)
        fig=None,               
        **add_subplot_kwargs
):
    """Return axes n from figure fig containing subplots.\n If subplot
    n does not exist, return a new axes object, possibly shifting all
    subplots to make room for ti. If axes n already exists, then
    return that. If ncolumns is specified then use that value,
    otherwise use internal heuristics.  n IS ZERO INDEXED"""
    if fig is None:
        fig=plt.gcf() # if figure object not provided use this
    old_axes = fig.axes           # list of axes originally in figure
    old_nsubplots = len(fig.axes) # number of subplots originally in figure
    if n is None:
        n = old_nsubplots+1
    else:
        n = n+1                           #set to 1-indexed in convention of subplots
    ## indexes an already existing subplot - return that axes
    if n<=old_nsubplots:
        ax = fig.axes[n-1]
    ## higher than existing subplots, creating empty intervening axes then add ths new one
    elif n>old_nsubplots+1:
        for i in range(old_nsubplots,n):
            ax = subplot(i,ncolumns,nrows,fig,**add_subplot_kwargs)
    ## need to add a new subplot
    else:
        nsubplots = old_nsubplots+1
        if ncolumns is not None and nrows is not None:
            columns,rows = ncolumns,nrows
        elif ncolumns is None and nrows is None:
            rows = int(np.floor(np.sqrt(nsubplots)))
            columns = int(np.ceil(float(nsubplots)/float(rows)))
            if fig.get_figheight()>fig.get_figwidth(): 
                rows,columns = columns,rows
        elif ncolumns is None and nrows is not None:
            raise ImplementationError()
        else:
            columns = ncolumns
            rows = int(np.ceil(float(nsubplots)/float(columns)))
        ## adjust old axes to new grid of subplots
        for i in range(0,old_nsubplots):
            i_old,j_old,n_old = fig.axes[i].get_geometry()
            old_axes[i].change_geometry(rows,columns,n_old)
        ## create and return new subplot
        ax = fig.add_subplot(rows,columns,nsubplots,**add_subplot_kwargs)
        # return(axes)
    plt.sca(ax)      # set to current axes
    ## set some other things if a quick figure
    if hasattr(fig,'_my_fig') and fig._my_fig is True:
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.10g'))
        ax.grid(True)
    return(ax)

def get_axes_position(ax=None):
    """Get coordinates of top left and bottom right corner of
    axes. Defaults to gca(). Returns an array not a Bbox."""
    if ax is None: ax = plt.gca()
    return np.array(ax.get_position())
    
def set_axes_position(x0,y0,x1,y1,ax=None):
    """Set coordinates of bottom left and top right corner of
    axes. Defaults to gca()."""
    if ax is None: ax = plt.gca()
    return ax.set_position(matplotlib.transforms.Bbox(np.array([[x0,y0],[x1,y1]])))

def transform_points_into_axis_fraction(x,y,ax=None):
    """Convert width and height in points to an axes fraction."""
    if ax is None: ax = plt.gca()
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # size of window in inches
    width, height = bbox.width*72, bbox.height*72 # size of window in pts
    return(x/width,y/height)

def transform_points_into_data_coords(x,y,ax=None):
    """Convert width and height in points to to data coordinates."""
    if ax is None: ax = plt.gca()
    xaxes,yaxes = transform_points_into_axis_fraction(x,y,ax)
    return(xaxes*np.abs(np.diff(ax.get_xlim())),yaxes*np.abs(np.diff(ax.get_ylim())))

def transform_axis_fraction_into_data_coords(x,y,ax=None):
    """Convert width and height in points to to data coordinates."""
    if ax is None: ax = plt.gca()
    (xbeg,xend),(ybeg,yend) = ax.get_xlim(),ax.get_ylim()
    xout = xbeg + x*(xend-xbeg)
    yout = ybeg + y*(yend-ybeg)
    return(xout,yout)

def subplotsCommonAxes(fig=None,):
    """Issue this comman with a optional Figure object and all x and y
    ticks and labels will be turned off for all subplots except the
    leftmost and bottommost. (Hopefully.)"""
    ## havent yet implemented turning off labels
    if fig is None: fig = plt.gcf()
    for ax in fig.axes:
        bbox = ax.get_position()
        if abs(bbox.x0-fig.subplotpars.left)>1e-2:
            ax.set_yticklabels([],visible=False)
        if abs(bbox.y0-fig.subplotpars.bottom)>1e-2:
            ax.set_xticklabels([],visible=False)

def add_xaxis_alternative_units(
        ax,
        transform,              # a function, from existing units to new
        inverse_transform=None,
        fmt='0.3g',             # for tick labels
        label='',               # axis label
        labelpad=None,               # axis label
        ticks=None,             # in original units
        minor=False,            # by default minor tick labels are turned off
        **set_tick_params_kwargs
):
    """Make an alternative x-axis (on top of plot) with units
    transformed by a provided function. Scale will always be linear."""
    ax2 = ax.twiny()
    if ticks is not None:
        assert inverse_transform is not None,'inverse_transform required if ticks are specified'
        ax2.xaxis.set_ticks(np.sort(inverse_transform(np.array(ticks))))
    ax2.xaxis.set_tick_params(**set_tick_params_kwargs)
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.xaxis.get_ticklocs()])
    if not minor: ax2.xaxis.set_ticks([],minor=True)
    ax2.set_xlabel(label,labelpad=labelpad)   # label
    return(ax2)

def add_yaxis_alternative_units(
        ax,
        transform,              # a function, from existing units to new
        fmt='0.3g',             # for tick labels
        label='',               # axis label
        labelpad=None,               # axis label
        ticks=None,             # in original units
        minor=False,            # by default minor tick labels are turned off
        **set_tick_params_kwargs
):
    """Make an alternative x-axis (on top of plot) with units
    transformed by a provided function. Scale will always be linear."""
    ax2 = ax.twinx(sharex=True)
    if ticks is not None: ax2.yaxis.set_ticks(np.sort(ticks))
    ax2.yaxis.set_tick_params(**set_tick_params_kwargs)
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.yaxis.get_ticklocs()])
    if not minor: ax2.yaxis.set_ticks([],minor=True)
    ax2.set_ylabel(label,labelpad=labelpad)   # label
    return(ax2)

def make_axes_limits_even(ax=None,beg=None,end=None,square=None):
    """Make x and y limits the same"""
    if ax is None:
        ax = plt.gca()
    if square is not None:
        beg,end = -square,square
    if beg is None:
        beg = min(ax.get_xlim()[0],ax.get_ylim()[0])
        end = max(ax.get_xlim()[1],ax.get_ylim()[1])
    ax.set_xlim(beg,end)
    ax.set_ylim(beg,end)
    return(beg,end)

# def add_yaxis_alternative_units(ax,transform,label='',fmt='0.3g',ticks=None,minor=False):
    # """Make an alternative y-axis (on top of plot) with units
    # transformed by a provided function."""
    # ax2 = ax.twinx()
    # if ticks!=None:
        # ax2.yaxis.set_ticks(ticks)
    # ax2.set_ylim(ax.get_ylim())
    # ax2.yaxis.set_ticklabels([format(transform(t),fmt) for t in ax2.yaxis.get_ticklocs()])
    # if not minor: ax2.yaxis.set_ticks([],minor=True)
    # ax2.set_ylabel(label)   # label
    # return(ax2)
            
def get_range_of_lines(line):
    """Find the maximum of a list of line2D obejts. I.e. a plotted
    line or errorbar, or an axes.. Returns (xmin,ymin,xmax,ymax)."""
    ymin = xmin = ymax = xmax = np.nan
    for t in line:
        x = t.get_xdata()
        y = t.get_ydata()
        i = ~np.isnan(y*x)
        if not any(i): continue
        x,y = x[i],y[i]
        if np.isnan(ymin):
            ymin,ymax = y.min(),y.max()
            xmin,xmax = x.min(),x.max()
        else:
            ymax,ymin = max(y.max(),ymax),min(y.min(),ymin)
            xmax,xmin = max(x.max(),xmax),min(x.min(),xmin)
    return(xmin,ymin,xmax,ymax)

def texEscape(s):
    """Escape TeX special characters in string. For filenames and
    special characters when using usetex."""
    return (re.sub(r'([_${}\\])', r'\\\1', s))
tex_escape=texEscape # remove one day

# def legend_from_kwargs(*plot_kwargs,ax=None,**legend_kwargs):
    # """Make an arbitrary legend satisfying plot_kwargs. Labels are taken
    # from plot_kwargs['label']."""
    # legend_kwargs.setdefault('loc','best')
    # legend_kwargs.setdefault('frameon',False)
    # legend_kwargs.setdefault('framealpha',1)
    # legend_kwargs.setdefault('edgecolor','black')
    # legend_kwargs.setdefault('fontsize','medium')
    # labels = [t.pop('label') for t in plot_kwargs]
    # leg = plt.legend(
        # labels=labels,
        # handles=[ax.plot([],[],**t)[0] for t in  plot_kwargs],
        # **legend_kwargs)
    # if ax is None: ax = plt.gca()
    # ax.add_artist(leg)
 #    return(leg)

# def legend(
        # *args,                  # passed to plt.legend as handles=args
        # # fontsize='medium',
        # ha='left',
        # ax=None,
        # **legend_kwargs):
    # """If no args, then enumerate lines."""
    # # legend_kwargs.setdefault('labelspacing',0.2)
    # # legend_kwargs.setdefault('columnspacing',0.5)
    # legend_kwargs.setdefault('loc','best')
    # legend_kwargs.setdefault('frameon',False)
    # legend_kwargs.setdefault('framealpha',1)
    # legend_kwargs.setdefault('edgecolor','black')
    # legend_kwargs.setdefault('fontsize','medium')
    # # legend_kwargs.setdefault('loc','best')
    # if ax is None: ax = plt.gca()
    # if not any(ax.get_lines()): return
    # ## Default labels are _line1, _line2, etc. If no explicit labels
    # ## are set then use these instead. Need to get rid of underscore
    # ## so that these will actually appear in legend.
    # if all([re.match(r'_line[0-9]+',l.get_label()) for l in ax.lines]):
        # for l in ax.lines:
            # if re.match(r'_line[0-9]+',l.get_label()):
                # l.set_label(l.get_label()[1:])
    # try:
        # ## try draw with current labels
        # leg = ax.legend(handles=(args if len(args)>0 else None),**legend_kwargs)
    # except AttributeError:
        # ## else invent some labels
        # leg = ax.legend(np.arange(len(ax.lines)),**legend_kwargs)
    # except IndexError: 
        # return
    # # ## remove frame
    # # texts = leg.get_texts()
    # # for t in texts: t.set_fontsize(fontsize)
    # ## right aligned
    # if ha=='right':
        # vp = leg._legend_box._children[-1]._children[0]
        # for c in vp._children:
            # c._children.reverse()
        # vp.align="right"
    # return(leg)

def legend(
        *plot_kwargs_or_lines,  # can be dicts of plot kwargs including label
        ax=None,                # axis to add legend to
        include_ax_lines=True, # add current lines in axis to legend
        color_text= True,     # color the label text
        show_style=False,      # hide the markers
        in_layout=False,       # constraining tight_layout or not
        **legend_kwargs,        # passed to legend
):
    """Make a legend and add to axis. Operates completely outside the
    normal scheme."""
    if ax is None: ax = plt.gca()
    def _reproduce_line(line): # Makes a new empty line with the properties of the input 'line'
        new_line = plt.Line2D([],[]) #  the new line to fill with properties
        for key in ('alpha','color','fillstyle','label',
                    'linestyle','linewidth','marker',
                    'markeredgecolor','markeredgewidth','markerfacecolor',
                    'markerfacecoloralt','markersize','markevery',
                    'solid_capstyle','solid_joinstyle',): # add all these properties
            if hasattr(line,'get_'+key):                  # if the input line has this property
                getattr(new_line,'set_'+key)(getattr(line,'get_'+key)()) # copy it to the new line
            elif hasattr(line,'get_children'): # if it does not but has children (i.e., and errorbar) then search in them for property
                for child in line.get_children():
                    if hasattr(child,'get_'+key): # property found
                        try:                      # try to set in new line, if ValueError then it has an invalid valye for a Line2D
                            getattr(new_line,'set_'+key)(getattr(child,'get_'+key)())
                            break # property added successfully search no more children
                        except ValueError:
                            pass
                    else:       # what to do if property not found anywhere
                        pass    # nothing!
        return(new_line)
    ## collect line handles and labels
    handles,labels = [],[]
    ## add existing lines in axis to legend
    if include_ax_lines:
        for handle,label in zip(*ax.get_legend_handles_labels()):
            if label!='_nolegend':
                labels.append(label)
                handles.append(_reproduce_line(handle))
    ## add get input lines or kwargs
    for i,t in enumerate(plot_kwargs_or_lines):
        if isinstance(t,matplotlib.lines.Line2D) or isinstance(t,mpl.container.ErrorbarContainer):
            raise Exception("Does not currently work for some reason.")
            t = t[0]
            if t.get_label()!='_nolegend':
                labels.append(t.get_label())
                handles.append(_reproduce_line(t))
        elif isinstance(t,dict):
            if t['label']!='_nolegend_':
                labels.append(t['label'])
                handles.append(plt.Line2D([],[],**t))
        else:
            raise Exception(f'Unhandled plot container type: {type(t)}')
    ## hide markers if desired
    if not show_style:
        for t in handles:
            t.set_linestyle('')
            t.set_marker('')
        legend_kwargs['handlelength'] = 0
        legend_kwargs['handletextpad'] = 0
    ## make a legend
    legend_kwargs.setdefault('handlelength',2)
    legend_kwargs.setdefault('loc','best')
    legend_kwargs.setdefault('frameon',False)
    legend_kwargs.setdefault('framealpha',1)
    legend_kwargs.setdefault('edgecolor','black')
    legend_kwargs.setdefault('fontsize','medium')
    if len(labels)==0: return(None)
    leg = ax.legend(labels=labels,handles=handles,**legend_kwargs)
    leg.set_in_layout(False)
    ## color the text
    if color_text:
        for text,handle in zip(leg.get_texts(),handles):
            if isinstance(handle,mpl.container.ErrorbarContainer):
                color = handle[0].get_color()
            else:
                color = handle.get_color()
            text.set_color(color)
    ## add to axis
    # ax.add_artist(leg)
    return(leg)

legend_from_kwargs = legend
legend_colored_text = legend


def supylabel(text,fig=None,x=0.01,y=0.5,**kwargs):
    """Set ylabel for entire figure at bottom. x,y to adjust position
    in figure fraction."""
    kwargs.setdefault('va','center')
    kwargs.setdefault('ha','left')
    kwargs.setdefault('rotation',90)
    # kwargs.setdefault('fontsize','large')
    if fig is None:
        fig = plt.gcf()
    fig.text(x,y,text,**kwargs)

def supxlabel(text,fig=None,x=0.5,y=0.02,loc='bottom',**kwargs):
    """Set xlabel for entire figure at bottom. x,y to adjust position
    in figure fraction."""
    if loc=='bottom':
        x,y=0.5,0.01
        kwargs.setdefault('va','bottom')
        kwargs.setdefault('ha','center')
    if loc=='top':
        x,y=0.95,0.98
        kwargs.setdefault('va','top')
        kwargs.setdefault('ha','center')
    kwargs.setdefault('rotation',0)
    # kwargs.setdefault('fontsize','large')
    if fig is None: fig = plt.gcf()
    fig.text(x,y,text,**kwargs)

def suplegend(fig=None,lines=None,labels=None,
              ax=None,loc='below',**legend_kwargs):
    """Plot a legend for the entire figure. This is useful when
    several subplots have the same series and common legend is more
    efficient. """
    if fig    is None: fig    = plt.gcf()
    if ax     is None: ax     = plt.gca()
    if lines  is None: lines  = [l for l in ax.get_lines() if l.get_label() != '_nolegend_']
    if labels is None: labels = [l.get_label() for l in lines]
    legend_kwargs.setdefault('numpoints',1)
    legend_kwargs.setdefault('fontsize','medium')
    legend_kwargs.setdefault('borderaxespad',1)   #try to squeeze between edge of figure and axes title
    ## not accpeted directly by fig.legend
    fontsize = legend_kwargs.pop('fontsize')   
    ## put legend
    if loc in ('below','bottom'):
        legend_kwargs.setdefault('ncol',3)
        legend_kwargs['loc'] = 'lower center'
    elif loc in ('above','top'):
        legend_kwargs.setdefault('ncol',3)
        legend_kwargs['loc'] = 'upper center'
    elif loc=='right':
        legend_kwargs.setdefault('ncol',1)
        legend_kwargs['loc'] = 'center right'
    else:
        legend_kwargs['loc'] = loc
    leg = fig.legend(lines,labels,**legend_kwargs)
    ## remove frame and resize text
    # leg.draw_frame(False)
    for t in leg.get_texts():  t.set_fontsize(fontsize)
    return(leg)

def turnOffOffsetTicklabels(ax=None):
    if ax is None: ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
    

def noTickMarksOrLabels(axes=None,axis=None):
    """Turn off tick labels and marks, axis='x' or axis='y' affects
    only that axis."""
    if axes is None: axes=gca()
    if (axis is None) or (axis=='x'):
        axes.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        axes.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if (axis is None) or (axis=='y'):
        axes.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
noTicMarksOrLabels = noTickMarksOrLabels

def simple_tick_labels(ax=None,axis=None,fmt='%g'):
    """Turn off fancy - but confusing scientific notation. Optional
    axis='x' or axis='y' to affect one axis o. Also sets the format of
    the mouse indicator to be 0.12g. """
    if ax is None: ax=plt.gca()
    # if (axis is None) | (axis=='x'):
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
        # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # if (axis is None) | (axis=='y'):
        # ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
        # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax.format_coord = lambda x,y : "x={:0.12g} y={:0.12g}".format(x, y) 
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(fmt))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(fmt))
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')
        
def tick_label_format(fmt,axis='both',ax=None):
    """Use a standard string format to format tick labels. Requires %
    symbol. Axis can be 'both', 'x',' or 'y'. Axes defaults to gca."""
    if ax is None: ax=plt.gca()
    if (axis=='both') | (axis=='x'):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    if (axis=='both') | (axis=='y'):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))

def noTickLabels(axes=None,axis=None):
    """Turn off tick labels, axis='x' or axis='y' affects only that axis."""
    if axes is None: axes=plt.gca()
    if (axis is None) | (axis=='x'):
        axes.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if (axis is None) | (axis=='y'):
        axes.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def connect_zoom_in_axis(ax1,ax2,**line_kwargs):
    """Join two plots, one a zoom in of the other, by lines indicating
    the zoom. Requires ax1 to be located above ax2 on the
    figure. Requires that the xlim of one axis (either one) is
    completely within the xlim of the other."""
    ## Defining the line to draw
    line_kwargs.setdefault('lw',1)
    line_kwargs.setdefault('color','red')
    line_kwargs.setdefault('ls','-')
    line_kwargs.setdefault('zorder',-5)
    ## get locations in figure coordinates then draw line
    fig = ax1.figure
    transFigure = fig.transFigure.inverted()
    data1_beg,data1_end = ax1.get_xlim()
    data2_beg,data2_end = ax2.get_xlim()
    coord1 = transFigure.transform(ax1.transAxes.transform([(max(data1_beg,data2_beg)-data1_beg)/(data1_end-data1_beg),0]))
    coord2 = transFigure.transform(ax2.transAxes.transform([(max(data1_beg,data2_beg)-data2_beg)/(data2_end-data2_beg),1]))
    line1 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=fig.transFigure,**line_kwargs)
    fig.lines.append(line1)
    coord1 = transFigure.transform(ax1.transAxes.transform([(min(data1_end,data2_end)-data1_beg)/(data1_end-data1_beg),0]))
    coord2 = transFigure.transform(ax2.transAxes.transform([(min(data1_end,data2_end)-data2_beg)/(data2_end-data2_beg),1]))
    line2 = matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=fig.transFigure,**line_kwargs)
    fig.lines.append(line2)
    ## add vertical lines
    line_kwargs['lw'] = 0
    if (data1_beg>data2_beg)&(data1_end<data2_end):
        ax2.axvline(data2_beg,**line_kwargs)
        ax2.axvline(data2_end,**line_kwargs)
    if (data2_beg>data1_beg)&(data2_end<data1_end):
        # ax1.axvline(data2_beg,**line_kwargs)
        # ax1.axvline(data2_end,**line_kwargs)
        
        ax1.axvspan(data2_beg,data2_end,**line_kwargs)
        # ax1.axvline(data2_end,**line_kwargs)
    return((line1,line2))
        
def arrow(x1y1,x2y2,
          arrowstyle='->',
          # label=None,
          xcoords='data',      # affects both ends of arrow
          ycoords='data',      # affects both ends of arrow
          ax=None,
          color='black',
          labelpad=10,
          fontsize=10,
          **arrow_kwargs):
    """
    Trying to make a nice simple arrow drawing function.  kwargs are
    passed directly to matplotlib.Patches.FancyArrowPatch. There are
    some custom defaults for these.
    """
    arrowParams={
        'linewidth':1.,
        'linestyle':'solid',
        'edgecolor':color, # also colors line
        'facecolor':color,
        'arrowstyle':arrowstyle, # simple,wedge,<->,-] etc
        'mutation_scale':10, # something to with head size
        'shrinkA':0.0,       # don't truncate ends
        'shrinkB':0.0,       # don't truncate ends
        }
    arrowParams.update(arrow_kwargs)
    if ax is None: ax = plt.gca()
    if xcoords=='data':
        xtransform = ax.transData
    elif xcoords=='axes fraction':
        xtransform = ax.transAxes
    else:
        raise Exception("unkonwn xcoords "+repr(xcoords))
    if ycoords=='data':
        ytransform = ax.transData
    elif ycoords=='axes fraction':
        ytransform = ax.transAxes
    else:
        raise Exception("unkonwn ycoords "+repr(ycoords))
    arrow = ax.add_patch(
        matplotlib.patches.FancyArrowPatch(
            x1y1,x2y2,
            transform=mpl.transforms.blended_transform_factory(xtransform,ytransform),
            **arrowParams),)
    ## add label parallel to arrow
    # if label is not None:
        # ## transform to display coordinates
        # x1,y1 = ax.transData.transform((x1,y1))
        # x2,y2 = ax.transData.transform((x2,y2))
        # midpoint = (0.5*(x1+x2),0.5*(y1+y2))
        # try:
            # angle  = np.arctan((y2-y1)/(x2-x1))
        # except ZeroDivisionError:
            # angle = np.pi/2.
        # ax.annotate(str(label),
                    # ax.transData.inverted().transform(midpoint),
                    # xycoords='data',
                    # xytext=(-labelpad*fontsize*np.sin(angle),labelpad*fontsize*np.cos(angle)),
                    # # xytext=(0,0),
                    # textcoords='offset points',
                    # rotation=angle/np.pi*180,
                    # ha='center',va='center',fontsize=fontsize,color=color)
    return(arrow)

myArrow=arrow

def annotate_corner(string,loc='top left',ax=None,fig=None,xoffset=5,yoffset=5,**kwargs):
    """Put string in the corner of the axis. Location as in
    legend. xoffset and yoffset in points."""
    if fig is None: fig = plt.gcf()
    if ax is None: ax = fig.gca()
    ## get x and y offsets from axes edge 
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #bbox in inches
    dx = xoffset/(bbox.width*72)          #in axes fraction
    dy = yoffset/(bbox.height*72)         #in axes fraction
    if loc in ['top left','tl','upper left']: xy,ha,va = (dx,1.-dy),'left','top'
    elif loc in ['bottom left','bl','lower left']: xy,ha,va = (dx,dy),'left','bottom'
    elif loc in ['top right','tr','upper right']: xy,ha,va = (1-dx,1-dy),'right','top'
    elif loc in ['bottom right','br','lower right']: xy,ha,va = (1-dx,dy),'right','bottom'
    elif loc in ['center left' ,'centre left' ]: xy,ha,va = (dx,0.5),'left','center'
    elif loc in ['top center','top centre','center top','centre top','upper center','upper centre','center upper','centre upper']: xy,ha,va = (0.5,1-dy),'center','top'
    elif loc in ['bottom center','bottom centre','center bottom','centre bottom','lower center','lower centre','center lower','centre lower']: xy,ha,va = (1-dx,0.5),'center','bottom'
    elif loc in ['center right','centre right']: xy,ha,va = (1-dx,0.5),'right','center'
    else: ValueError('Bad input loc')
    return ax.annotate(string,xy,xycoords='axes fraction',ha=ha,va=va,**kwargs)

def annotate_line(string=None,xpos='ymax',ypos='above',
                  line=None,ax=None,color=None,
                  xoffset=0,yoffset=0,   # pts
                  **annotate_kwargs):
    """Put a label above or below a line. Defaults to legend label and
    all lines in axes.  xpos can be [min,max,left,right,peak,a
    value]. ypos in [above,below,a value].  First plots next to first
    point, last to last. Left/right fixes to left/rigth axis."""
    ## default to current axes
    if ax is None and line is None:
        ax = plt.gca()
    ## no lines, get list from current axes
    if line is None:
        line = ax.get_lines()
    ## list of lines, annotate each individually
    if np.iterable(line):
        return([annotate_line(string=string,xpos=xpos,ypos=ypos,line=l,ax=ax,color=color,xoffset=xoffset,yoffset=yoffset,**annotate_kwargs) for l in line])
    if ax is None: ax = line.axes
    if string is None:
        if re.match('_line[0-9]+',line.get_label()): return None
        string = line.get_label()
    ## a shift to make space around text
    text_x_offset,text_y_offset = transform_points_into_axis_fraction(mpl.rcParams['font.size']/2,mpl.rcParams['font.size']/2)
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    xdata,ydata = line.get_data()
    if len(xdata)==0: return    # no line
    if xpos in (None,'left'):
        annotate_kwargs['xycoords'] = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        xpos = text_x_offset
        annotate_kwargs.setdefault('ha','left')
    elif xpos in (None,'right'):
        annotate_kwargs['xycoords'] = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)
        xpos = 1-text_x_offset
        annotate_kwargs.setdefault('ha','right')
    elif xpos in ('min','xmin',):
        xpos = xdata.min()
        annotate_kwargs.setdefault('ha','right')
    elif xpos in ('max','xmax'):
        xpos = xdata.max()
        annotate_kwargs.setdefault('ha','left')
    elif xpos in ('peak','ymax',):
        xpos = line.get_xdata()[np.argmax(ydata)]
        annotate_kwargs.setdefault('ha','center')
    elif xpos in ('minimum','ymin',):
        xpos = line.get_xdata()[np.argmin(ydata)]
        annotate_kwargs.setdefault('ha','center')
    elif isnumeric(xpos):
        annotate_kwargs.setdefault('ha','center')
    else:
        raise Exception('bad xpos: ',repr(xpos))
    i = np.argmin(np.abs(xdata-xpos))
    if ypos in ['above','top']:
        annotate_kwargs.setdefault('va','bottom')
        ypos = ydata[i] + text_y_offset
    elif ypos in ['below','bottom']:
        annotate_kwargs.setdefault('va','top')
        ypos = ydata[i] - text_y_offset
    elif ypos in ['center']:
        annotate_kwargs.setdefault('va','center')
        ypos = ydata[i]
    elif ypos == None or isnumeric(ypos):
        annotate_kwargs.setdefault('va','center')
    else:
        raise Exception('bad ypos: ',repr(ypos))
    if color is None: 
        color = line.get_color()
    ## draw label
    if string=='_nolegend_': string=''
    xoffset,yoffset = transform_points_into_data_coords(xoffset,yoffset)
    annotate_kwargs.setdefault('in_layout',False)
    return(ax.annotate(string,(float(xpos+xoffset),float(ypos+yoffset)),color=color,**annotate_kwargs))

def annotate(*args,**kwargs):
    """Adds default arrow style to kwargs, otherwise the same as
    numpy.annotate, which by default doesn't draw an arrow
    (annoying!)."""
    ## default arrow style, if no arrows coords then draw no arrow
    kwargs.setdefault('arrowprops',dict(arrowstyle="->",))
    if 'linewidth' in kwargs:
        kwargs['arrowprops']['linewidth']=kwargs.pop('linewidth')
    if 'color' in kwargs:
        kwargs['arrowprops']['edgecolor']=kwargs['color']
        kwargs['arrowprops']['facecolor']=kwargs['color']
    if len(args)<3:
        kwargs.pop('arrowprops')
    else:
        if not (isiterableNotString(args[1]) and isiterableNotString(args[2])):
            kwargs.pop('arrowprops')
    ax = plt.gca()
    a =  ax.annotate(*args,**kwargs)
    a.set_in_layout(False)
    return(a)

def annotate_point(label,x,y,ax=None,fontsize='medium',marker='o',linestyle='',color='black',**plot_kwargs):
    if ax is None:
        ax = plt.gca()
    l = ax.plot(x,y,marker=marker,linestyle=linestyle,color=color,**plot_kwargs)
    a = ax.annotate(str(label),(x,y),fontsize=fontsize,color=color)
    return(l,a)

def clginput(return_coord='both',with_comma=False):
    """Get ginput and add to clipboard. return_coord can be 'both', 'x', or 'y'."""
    x = np.array(plt.ginput(-1))
    if return_coord=='both': cl(x)
    elif return_coord=='x':
        if with_comma:
            cl('\n'.join([format(t,'#.15g')+',' for t in x[:,0]]))
        else:
            cl(x[:,0])
    elif return_coord=='y': cl(x[:,1])
    else: raise Exception("Unknown return_coord: ",repr(return_coord))

def ginput_spline_curve(x,y):
    """Interactively add points to a figure, whilst updating a spline
    curve connecting them. Returns selected points. Hit 'enter' to
    break the interactive loop. Middle button to select a point."""
    ## create figure and axes, and draw curve
    presetRcParams('screen',autolayout=True)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x,y,color=newcolor(0))
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    ## to break loop below on close figure window
    # def f(event): raise KeyboardInterrupt
    # fig.canvas.mpl_connect('close_event',f)
    ## main loop selecting spline points
    xbg,ybg = np.array([],dtype=float),np.array([],dtype=float) 
    while True:
        ## get a new background point selection, any funny business here
        ## indicates to exit the loop. For example "enter".
        try:
            xi,yi = plt.ginput(n=1,timeout=0)[0]
        except:
            break
        ## compute if close to an existing point
        distance = np.sqrt(((ybg-yi)/(ylim[-1]-ylim[0]))**2 + ((xbg-xi)/(xlim[-1]-xlim[0]))**2)
        i = distance<0.02
        ## if so remove the existing point, else add the new point
        if np.any(i):
            xbg,ybg = xbg[~i],ybg[~i]
        else:
            xbg = np.concatenate((xbg,[xi]))
            ybg = np.concatenate((ybg,[yi]))
        ## sort
        i = np.argsort(xbg)
        xbg,ybg = xbg[i],ybg[i]
        ## redraw curve
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.autoscale(False)
        ax.plot(x,y,color=newcolor(0))
        ## draw points
        ax.plot(xbg,ybg,ls='',marker='x',color=newcolor(1))
        ## draw spline background
        if len(xbg)>1:
            xspline = x[(x>=xbg[0])&(x<=xbg[-1])]
            yspline = spline(xbg,ybg,xspline)
            ax.plot(xspline,yspline,ls='-',color=newcolor(1))
        plt.draw()
    return(xbg,ybg)

def annotate_ginput(label='',n=1):
    """Get a point on figure with click, and save a line of code to
    clipboard which sets an annotation at this point."""
    d = plt.ginput(n=n)
    annotate_command = '\n'.join([
        'ax.annotate("{label:s}",({x:g},{y:g}))'.format(label=label,x=x,y=y)
        for x,y in d])
    cl(annotate_command)
    return(annotate_command)

def annotate_vline(label,xpos,ax=None,color='black',fontsize='medium',
                   label_ypos=0.98,labelpad=0,zorder=None,alpha=1,
                   annotate_kwargs=None,**axvline_kwargs):
    """Draw a vertical line at xpos, and label it."""
    if ax==None: ax = plt.gca()
    axvline_kwargs.setdefault('alpha',alpha)
    axvline_kwargs.setdefault('zorder',zorder)
    line_object = ax.axvline(xpos,color=color,**axvline_kwargs)
    transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ## align labels to top or bottom
    ## the label
    if annotate_kwargs is None: annotate_kwargs = {}
    annotate_kwargs.setdefault('color',color)
    annotate_kwargs.setdefault('alpha',alpha)
    annotate_kwargs.setdefault('zorder',zorder)
    if label_ypos>0.5:
        annotate_kwargs.setdefault('va','top')
    else:
        annotate_kwargs.setdefault('va','bottom')
    label_object = ax.annotate(label,
                               xy=(xpos+labelpad, label_ypos),
                               xycoords=transform,
                               rotation=90,
                               fontsize=fontsize,
                               # ha='right',
                               # color=color,
                               # backgroundcolor='white',
                               **annotate_kwargs
    )
    label_object.set_in_layout(False)
    return(line_object,label_object)

def annotate_hline(label,ypos,ax=None,color='black',fontsize='medium',va='bottom',
                   loc='right', # or 'left'
                   **axhline_kwargs):
    """Draw a vertical line at xpos, and label it."""
    if ax==None: ax = plt.gca()
    line_object = ax.axhline(ypos,color=color,**axhline_kwargs)
    transform = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData,)
    label_object = ax.annotate(
        label,
        xy=((0.98 if loc=='right' else 0.02), ypos),
        xycoords=transform,
        ha=('right' if loc=='right' else 'left'),
        va=va,color=color,fontsize=fontsize) 
    return(line_object,label_object)

def annotate_hspan(label,y0,y1,ax=None,color='black',fontsize='medium',label_ypos='bottom',alpha=0.5,**axhspan_kwargs):
    """Draw a vertical line at xpos, and label it. label_ypos in 'top','bottom','center'."""
    if ax==None: ax = plt.gca()
    line_object = ax.axhspan(y0,y1,color=color,alpha=alpha,**axhspan_kwargs)
    transform = matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData,)
    if label_ypos=='bottom':
        ylabel,va = min(y0,y1),'top'
    elif label_ypos=='top':
        ylabel,va = max(y0,y1),'bottom'
    elif label_ypos=='center':
        ylabel,va = 0.5*(y0+y1),'center'
    label_object = ax.annotate(label,xy=(0.98,ylabel),xycoords=transform,ha='right',va=va,color=color,fontsize=fontsize) 
    return(line_object,label_object)

def set_text_border(text_object,border_color='black',face_color='white',border_width=1):
    """Set text_object to have a border."""
    import matplotlib.patheffects as path_effects
    text_object.set_color(face_color)
    text_object.set_path_effects([
        path_effects.Stroke(linewidth=border_width,foreground=border_color), # draw border
         path_effects.Normal(), # draw text on top
    ])

def plot_and_label_points(x,y,labels,ax=None,fontsize='medium',**plot_kwargs):
    """Plot like normal but also label each data point."""
    if ax is None: ax = plt.gca()
    l = ax.plot(x,y,**plot_kwargs)
    a = [ax.annotate(label,(xx,yy),fontsize=fontsize) for (xx,yy,label) in zip(x,y,labels)]
    return(l,a)

def plot_lines_and_disjointed_points(x,y,max_separation=1,ax=None,**plot_kwargs):
    """Plot continguous data (x separation < max_separation) as line
    segments and disjointed points with markers."""
    if ax is None: ax = plt.gca()
    plot_kwargs.setdefault('marker','o')
    plot_kwargs.setdefault('linestyle','-')
    plot_kwargs.setdefault('color',newcolor(0))
    if 'ls' in plot_kwargs:
        plot_kwargs['linestyle'] = plot_kwargs.pop('ls')
    i = np.argsort(x)
    x,y = x[i],y[i]
    d = np.diff(x)
    ## plot line segments
    kwargs = copy(plot_kwargs)
    kwargs['marker']=''
    for i in find(d<=max_separation):
        ax.plot(x[[i,i+1]],y[[i,i+1]],**kwargs)
    ## plot markers
    kwargs = copy(plot_kwargs)
    kwargs['linestyle']=''
    ## one point only
    if len(x)==1:
        ax.plot(x,y,**kwargs)
    ## disjointed points
    for i in find(d>max_separation):
        ## first point along
        if i==0:
            ax.plot(x[i],y[i],**kwargs)
        ## second jump
        if d[i-1]>max_separation:
            ax.plot(x[i],y[i],**kwargs)
        ## last point alone
        if i==len(d)-1:
            ax.plot(x[-1],y[-1],**kwargs)

def axesText(x,y,s,**kwargs):
    """Just like matplotlib ax.text, except defaults to axes fraction
    coordinates, and centers text."""
    ax = plt.gca()
    kwargs.setdefault('transform',ax.transAxes)
    kwargs.setdefault('verticalalignment','center')
    kwargs.setdefault('horizontalalignment','center')
    return ax.text(x,y,s,**kwargs)

def set_ticks(
        x_or_y='x',
        locs=None,
        labels=None,
        spacing=None,
        divisions=None,
        ax=None,               # axes
        fontsize=None,
        rotation=None,
        **labels_kwargs
):
    ## get an axis
    if ax is None:
        ax = plt.gca()
    assert x_or_y in ('x','y')
    axis = (ax.xaxis if x_or_y=='x' else ax.yaxis)
    ## set major and minor locs
    if locs is not None:
        axis.set_ticks(locs)
    elif spacing is not None:
        beg,end = (ax.get_xlim() if x_or_y=='x' else ax.get_ylim())
        axis.set_ticks(np.arange(((beg+spacing*0.9999)//spacing)*spacing, end, spacing))
        if divisions is not None:
            minor_spacing = spacing/divisions
            axis.set_ticks(np.arange(((beg+minor_spacing*0.9999)//minor_spacing)*minor_spacing, end, minor_spacing),minor=True)
    ## set labels
    if labels is not None:
        axis.set_ticklabels(labels,**labels_kwargs)
    ## set rotation
    if rotation is not None:
        if x_or_y=='x':
            verticalalignment = 'top'
            horizontalalignment = 'right'
        else:
            verticalalignment = 'center'
            horizontalalignment = 'right'
        for label in axis.get_ticklabels():
            label.set_rotation(rotation)
            label.set_verticalalignment(verticalalignment)
            label.set_horizontalalignment(horizontalalignment)
   ## fontsize
    if fontsize is not None:
       for label in axis.get_ticklabels():
           label.set_size(fontsize)

def rotate_tick_labels(x_or_y,rotation=90,ax=None):
    if ax is None: ax=plt.gca()
    assert x_or_y in ('x','y')
    if x_or_y=='x':
        labels = ax.xaxis.get_ticklabels()
        verticalalignment = 'top'
        horizontalalignment = 'right'
    elif x_or_y=='y':
        labels = ax.xaxis.get_ticklabels()
        verticalalignment = 'center'
        horizontalalignment = 'right'
    else:
        raise Exception(f'Bad x_or_y: {repr(x_or_y)}')
    for t in labels:
        t.set_rotation(rotation)
        t.set_verticalalignment(verticalalignment)
        t.set_horizontalalignment(horizontalalignment)

def set_tick_labels_text(
        strings,
        locations=None,
        axis='x',
        ax=None,
        rotation=70,
        **set_ticklabels_kwargs,):
    """Set a list of strings as text labels."""
    if ax is None:
        ax = plt.gca()
    if axis=='x':
        axis = ax.xaxis
    elif axis=='y':
        axis = ax.yaxis
    if locations is None:
        locations = np.arange(len(strings))
    axis.set_ticks(locations)
    axis.set_ticklabels(strings,rotation=rotation,**set_ticklabels_kwargs)

def set_tick_spacing(
        axis='x',               # 'x' or 'y'
        major_spacing=1,        # absolute
        minor_divisions=None,       # number of minor tick intervals per major,None for default
        ax=None):
    """Simple method for adjusting major/minor tick mark spacing."""
    if ax == None: ax=plt.gca()
    assert axis in ('x','y'),'axis must be x or y'
    if axis=='x':
        axis,(beg,end) = ax.xaxis,ax.get_xlim()
    elif axis=='y':
        axis,beg,end = ax.yaxis,ax.get_ylim()
    axis.set_ticks(np.arange(
        ((beg+major_spacing*0.9999)//major_spacing)*major_spacing,
        end, major_spacing))
    if minor_divisions is not None:
        minor_spacing = major_spacing/minor_divisions
        axis.set_ticks(np.arange(
            ((beg+minor_spacing*0.9999)//minor_spacing)*minor_spacing,
            end, minor_spacing),minor=True)

def show(fmt='x',fig=None):
    """ Show current plot in some customised way. Also adds a red line as
    border.
    Inputs:
    fmt - How to show, png: save as /tmp/tmp.png
                       eps: save as /tmp/tmp.eps
                       x:   run plt.show()
    fig - figure object, default to to gcf().
    """
    if fig == None:
        fig = plt.gcf()
    # fig.patches.append(plt.Rectangle((0.,0.),1.,1.,edgecolor='red',
                                     # facecolor='none',linewidth=0.5,
                                     # transform=fig.transFigure, figure=fig))
    if fmt=='eps':
        fig.savefig('/tmp/tmp.eps')
    elif fmt=='png':
        fig.savefig('/tmp/tmp.png')
    elif fmt=='x':
        try:
            __IPYTHON__
        except NameError:
            for figure_number in plt.get_fignums():
                extra_interaction(fig=plt.figure(figure_number))    
            plt.show()
    else:
        raise Exception('fmt ``'+str(fmt)+"'' unknown.")

show_if_noninteractive = show

def qplot(xydata,**kwargs):
    """Issue a plot command and then output to file."""
    fig=plt.figure()
    ax=fig.gca()
    ax.plot(xydata[:,0],xydata[:,1:],**kwargs)
    legend()
    ax.grid(True)
    fig.savefig('/tmp/tmp.eps')

def qxlim(*args,**kwargs):
    """Issue a plot command and then output to file."""
    fig = plt.gcf()
    ax=fig.gca()
    ax.set_xlim(*args,**kwargs)
    fig.savefig('/tmp/tmp.eps')

def qylim(*args,**kwargs):
    """Issue a plot command and then output to file."""
    fig = plt.gcf()
    ax=fig.gca()
    ax.set_ylim(*args,**kwargs)
    fig.savefig('/tmp/tmp.eps')

def ylogerrorbar(x,y,dy,ylolimScale=0.9,*args,**kwargs):
    """Ensures lower bound of error bars doesn't go negative messing
    up drawing with log scale y axis. 
    
    If necessary setes lower error lim to ylolimScale*y.
    
    All args and kwargs passed to
    regular errorbars."""
    ax = plt.gcf().gca()
    y = np.array(y)
    dy = np.array(dy)
    dylower = copy(dy)
    i = dylower>=y
    dylower[i] = ylolimScale*y[i]
    ax.errorbar(x,y,yerr=np.row_stack((dylower,dy)),*args,**kwargs)
    

def qplotfile(filenames,showLegend=True):
    """
    Try to guess a good way to 2D plot filename.

    Probably needs more work.
    """

    ## ensure a list of names, not just one
    if isinstance(filenames,str):
        filenames = [filenames]

    ## begin figure
    f = plt.figure()
    a = f.gca()

    ## loop through all filenames
    for filename in filenames:

        ## get data
        filename = expand_path(filename)
        x = np.loadtxt(filename)

        ## plot data
        if x.ndim==1:
            a.plot(x,label=filename)
        else:
            for j in range(1,x.shape[1]):
                a.plot(x[:,0],x[:,j],label=filename+' '+str(j-1))

    ## show figure
    if showLegend:
        legend()
    f.show()

def savefig(path,fig=None,**kwargs):
    """Like fig.savefig except saves first to a temporary file in
    order to achieve close to instant file creation."""
    path = expand_path(path)
    name,ext = os.path.splitext(path)
    tmp = tempfile.NamedTemporaryFile(suffix=ext)
    kwargs.setdefault('dpi',300)
    if fig is None: fig = plt.gcf()
    # kwargs.setdefault('transparent',True)
    fig.savefig(tmp.name,**kwargs)
    shutil.copyfile(tmp.name,path)

def decode_format_string(s):
    """Get the different arts of a format string, return as dictionary."""
    g = re.match(r'([<>+-]*)([0-9]*).?([0-9]*)([fgsed])',s).groups()
    return(dict(prefix=g[0],length=g[1],precision=g[2],type=g[3]))


def parentheses_style_errors_format(
        x,s,
        error_significant_figures=2,
        tex=False,
        treat_zero_as_nan=False,
        default_sig_figs=3,     # if not error to indicate uncertainty use this many significant figures
        max_leading_zeros=3,    # before use scientific notation
        fmt='f',                # or 'e'
        # nan_data_as_blank=False, # do not print nans, does something else instead
        nan_substitute=None,
):
    """
    Convert a value and its error in to the form 1234.456(7) where the
    parantheses digit is the error in the least significant figure
    otherwise. If bigger than 1 gives 1200(400). If error_significant_figures>1
    print more digits on the error.
    \nIf tex=True make nicer still.
    """
    ## vectorise
    if np.iterable(x):
        return [
            format_parentheses_style_errors(
                xi,si,
                error_significant_figures,
                tex,
                treat_zero_as_nan,
                default_sig_figs,
                max_leading_zeros,
                fmt,
                # nan_data_as_blank=nan_data_as_blank,
                nan_substitute=nan_substitute,
            ) for (xi,si) in zip(x,s)]
    assert fmt in ('e','f'),'Only "e" and "f" formatting implemented.'
    ## deal with zeros
    if treat_zero_as_nan:
        if x==0.: x=np.nan
        if s==0.: s=np.nan
    ## data is nan
    if np.isnan(x):
        if nan_substitute is None:
            retval = 'nan(nan)'
        else:
            retval = nan_substitute
    ## data exists but uncertainty is nan, just return with default_sig_figs
    elif np.isnan(s):
        if fmt=='f':
            retval = format_float_with_sigfigs(x,default_sig_figs)
        elif fmt=='e':
            retval = format(x,f'0.{default_sig_figs-1}e')
    ## data and uncertainty -- computed parenthetical error
    else:
        if 'f' in fmt:
            ## format data string 'f'
            ## format error string
            t=format(s,'0.'+str(error_significant_figures-1)+'e') ## string rep in form +1.3e-11
            precision = int(re.sub('.*e','',t))-error_significant_figures+1
            s = t[0:1+error_significant_figures].replace('.','').replace('e','')
            x=(round(x/10**precision)+0.1)*10**precision
            x=format(x,('+0.30f' if fmt[0]=='+' else '0.30f'))
            i=x.find('.')
            if precision < 0:
                x=x[:i-precision+1]
            elif precision >0:
                x=x[:i]
                for j in range(precision): s=s+'0'
            elif precision==0:
                x=x[:i]
            retval = x+'('+s+')'
        elif 'e' in fmt:
            ## format data string 'e'
            r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(x,'0.50e'))
            value_precision = int(r.group(2))
            r = re.match(r'^([+-]?[0-9]\.[0-9]+)e([+-][0-9]+)$',format(s,'0.50e'))
            error_precision = int(r.group(2))
            error_string = r.group(1).replace('.','')[:error_significant_figures]
            value_string = str(int(np.round(x*10**(-error_precision+error_significant_figures-1))))
            ## add decimal dot
            if value_string[0] in '+-':
                if len(value_string)>2: value_string = value_string[0:2]+'.'+value_string[2:]
            else:
                if len(value_string)>1: value_string = value_string[0]+'.'+value_string[1:]
            error_string = str(int(np.round(s*10**(-error_precision+error_significant_figures-1))))
            retval = f'{value_string}({error_string})e{value_precision}'
        else:
            raise Exception('fmt must be "e" or "f"')
    if tex:
        ## separate thousands by \,, unless less than ten thousand
        # if not re.match(r'[0-9]{4}[.(,]',retval): 
        #     while re.match(r'[0-9]{4,}[.(,]',retval):
        #         retval = re.sub(r'([0-9]+)([0-9]{3}[,.(].*)',r'\1,\2',retval)
        #     retval = retval.replace(',','\,')
        while True:
            r = re.match(r'^([+-]?[0-9,]*[0-9])([0-9]{3}(?:$|[.(]).*)',retval)
            if not r:
                break
            retval = r.group(1)+','+r.group(2)
        retval = retval.replace(',','\,')
        ## separate decimal thousands by \,, unless less than ten thousandths
        r = re.match(r'(.*\.)([0-9]{5,})(.*)',retval)
        if r:
            beg,decimal_digits,end = r.groups()
            retval = beg+r'\,'.join([decimal_digits[i:i+3] for i in range(0,len(decimal_digits),3)])+end
        ## replace nan with --
        # retval = retval.replace('nan','--')
        # retval = retval.replace('--(--)','--')
        t = re.match(r'(.*)e([+-])([0-9]+)(.*)',retval)
        if t:
            beg,exp_sign,exp,end = t.groups()
            exp = str(int(exp)) # get to single digit int
            if exp==0:  exp_sign = '' # no e-0
            if exp_sign=='+': exp_sign = '' # no e+10
            retval = f'{beg}\\times10^{{{exp_sign}{exp}}}{end}'
        # retval = re.sub(r'e([+-]?)0?([0-9]+)',r'\\times10^{\1\2}',retval)
        # retval = re.sub(r'e([+-]?[0-9]+)',r'\\times10^{\1}',retval)
        retval = '$'+retval.replace('--','-')+'$' # encompass
    return(retval)
format_parentheses_style_errors = parentheses_style_errors_format # deprecated
parentheses_style_errors = parentheses_style_errors_format # deprecated name

def parentheses_style_errors_decode(string):
    """Convert string of format '##.##(##)' where parentheses contain
    an uncertainty esitmate of the least significant digit of the
    preceding value. Returns this value and an error as floats."""
    ## if vector of strings given return an array of decoded values
    if not np.isscalar(string):
        return(np.array([parentheses_style_errors_decode(t) for t in string]))
    ## 
    m = re.match(r'([0-9.]+)\(([0-9]+)\)',str(string))
    if m is None:
        warnings.warn('Could not decode `'+str(string)+"'")
        return np.nan,np.nan
    valstr,errstr = m.groups()
    tmpstr = list(re.sub('[0-9]','0',valstr,))
    i = len(tmpstr)
    for j in reversed(errstr):
        i=i-1
        while tmpstr[i]=='.': i=i-1
        tmpstr[i] = j
    return float(valstr),float(''.join(tmpstr))

def format_tex_scientific(
        x,
        sigfigs=2,
        include_math_environment=True,
        nan_behaviour='--',     # None for error on NaN, else a replacement string
):
    """Convert a string to scientific notation in tex code."""
    if np.isnan(x):
        if nan_behaviour is None:
            raise Exception("NaN not handled.")
        else:
            return('--')
    s = format(x,'0.'+str(sigfigs-1)+'e')
    m = re.match(r'(-?[0-9.]+)[eEdD]\+?(-)?0*([0-9]+)',s)
    digits,sign,exponent =  m.groups()
    if sign is None:
        sign=''
    if include_math_environment:
        delim = '$'
    else:
        delim = ''
    return r'{delim}{digits}\times10^{{{sign}{exponent}}}{delim}'.format(
        digits=digits,sign=sign,exponent=exponent,delim=delim)

def format_numprint(x,fmt='0.5g',nan_behaviour='--'):
    """Make an appropriate latex numprint formatting command. Fomra number
    with fmt first. nan_behaviour # None for error on NaN, else a
    replacement string """
    if np.isnan(x):
        if nan_behaviour is None:
            raise Exception("NaN not handled.")
        else:
            return('--')
    return(f'\\np{{{format(x,fmt)}}}')

def format_float_with_sigfigs(
        x,
        sigfigs,
        tex=False,
        fmt='f',                # or 'e'
):
    """Convert a float to a float format string with a certain number of
    significant figures. This is different to numpy float formatting
    which controls the number of decimal places."""
    assert sigfigs>0
    if tex:
        thousands_separator = r'\,'
    else:
        thousands_separator = ''
    ## get number of sigfigs rounded and printed into a string, special case for x=0
    if x!=0:
        x = float(x)
        sign_x = np.sign(x)
        x = np.abs(x)
        x = float(x)
        exponent = int(np.floor(np.log10(x)))
        decimal_part = max(0,sigfigs-exponent-1)
        s = format(sign_x*np.round(x/10**(exponent+1-sigfigs))*10**(exponent+1-sigfigs),'0.{0:d}f'.format(decimal_part))
    else:
        if sigfigs==1:
            s = '0'
        else:
            s = '0.'+''.join(['0' for t in range(sigfigs-1)])
    ## Split into >=1. <1 components
    if s.count('.')==1:
        greater,lesser = s.split('.')
        sep = '.'
    else:
        greater,sep,lesser = s,'',''
    ## add thousands separator, if number is bigger than 9999
    if len(greater)>4:
        indices_to_add_thousands_separator = list(range(len(greater)-3,0,-3))[-1::-1]
        greater = thousands_separator.join([greater[a:b] for (a,b) in zip(
            [0]+indices_to_add_thousands_separator,
            indices_to_add_thousands_separator+[len(greater)])])
    ## recomprise
    s = greater+sep+lesser
    return(s)
    
def format_fixed_width(x,sigfigs,width=None):
    """Creates a exponential form floating point number with given
    significant figures and total width. If this fails then return a
    str form of the given width. Default width is possible."""
    if width is None: 
        width = sigfigs+6
    try:
        return format(x,'>+{}.{}e'.format(width,sigfigs-1))
    except ValueError:
        return format(x,'>'+str(width))

def format_string_or_general_numeric(x):
    """Return a string which is the input formatted 'g' if it is numeric or else is str representation."""
    try:
        return format(x,'g')
    except ValueError:
        return(str(x))

def format_as_disjoint_ranges(x,separation=1,fmt='g'):
    """Identify ranges and gaps in ranges and print as e.g, 1-5,10,12-13,20"""
    x = np.sort(x)
    i = [0]+list(find(np.diff(x)>separation)+1)+[len(x)] # range edges
    ## Print each range as a-b if range is required or a if single
    ## value is disjoin. Separate with commas.
    return(','.join([
        format(x[a],fmt)+'-'+format(x[b-1],fmt) if (b-a)>1 else format(x[a],fmt)
                    for (a,b) in zip(
                            [0]+list(find(np.diff(x)>separation)+1),
                            list(find(np.diff(x)>separation)+1)+[len(x)])]))

def bibtex_file_to_dict(filename):
    """Returns a dictionary with a pybtex Fielddict, indexed by bibtex
    file keys."""
    from pybtex.database import parse_file
    database  = parse_file(filename)
    entries = database.entries
    retval_dict = collections.OrderedDict()
    for key in entries:
        fields = entries[key].rich_fields
        retval_dict[key] = {key:str(fields[key]) for key in fields}
    return(retval_dict)

def make_latex_table(columns,environment='tabular',preamble='',postscript='',):
    """Convenient method to create a simple LaTeX table from data.

    Columns is a list of Value_Arrays or dictionaries with the following keys:

    val -- an array or somesuch, not necessarily numerical
    fmt -- (optional?) format string, or an arbitrary function that
           converts elements of data to a string
    latex_table_alignment -- (optional?) latex table alignment code
    name -- (optional?) To go at top of table.

    Other options:

    environment -- enclosing latex environment, i.e., tabular or longtable
    preamble -- appears before environment
    postscript -- appears after environment """
    ## convert dictionary to Value_Array if necessary
    for i in range(len(columns)):
        if isinstance(columns[i],dict):
            columns[i] = Value_Array(**columns[i])
    string = '\n'.join([
        preamble,
        r'\begin{'+environment+'}{'+''.join([col.latex_table_alignment for col in columns])+'}',
        # r'\hline\hline',
        r'\toprule',
        ' & '.join([col.name for col in columns])+' \\\\',
        # r'\hline',
        r'\midrule',
        make_table(columns,field_separator=' & ',row_separator=' \\\\\n',comment='',print_description=False,print_keys=False)+' \\\\',
        # r'\hline\hline',
        r'\bottomrule',
        r'\end{'+environment+'}',
        postscript,
    ])
    return string

def make_table(columns,header=None,row_separator='\n',
               field_separator=' ', comment='#',
               print_description=True, print_keys=True):
    """Convenient method to create a simple table from data.\n
    Columns is a list of Value_Array objects\n
    Other options:\n
    header -- put at top of table
    """
    ## keys and data
    rows = list(zip(*[t.get_formatted_list(include_column_label=print_keys) for t in columns]))
    rows = [field_separator.join(row) for row in rows]
    rows[0] = comment+' '+rows[0] # comment keys
    ## add space to align commented keys with data
    comment_space = ''.join([' ' for t in comment])
    for i in range(len(comment)):
        comment_space = comment_space+' '
    rows[1:] = [comment_space+t for t in rows[1:]]
    ## header and description
    if print_description:
        for col in columns[-1::-1]:
            rows.insert(0,comment+' '+col.name.strip()+': '+col.description)
    if header!=None:
        for header_line in header.split('\n')[-1::-1]:
            rows.insert(0,comment+' '+header_line)
    return row_separator.join(rows)

def tabulate(tabular_data, headers=[], tablefmt="plain",
             floatfmt="g", numalign="decimal", stralign="left",
             missingval=""):
    """From tabulate.tabulate with "csv" tablefmt added."""
    import tabulate as _tabulate
    _tabulate._table_formats["csv"] = _tabulate.TableFormat(
        lineabove=None, linebelowheader=None,
        linebetweenrows=None, linebelow=None,
        headerrow=_tabulate.DataRow("", ", ", ""),
        datarow=_tabulate.DataRow("", ", ", ""),
        padding=0, with_header_hide=None)
    return(_tabulate.tabulate(tabular_data,headers,tablefmt,floatfmt,numalign,stralign,missingval))



subplot_append = subplot
isiterableNotString = isiterable_not_string
simpleTickLabels = simple_tick_labels
annotateArrow = annotate
dict_array_to_txt = dict_array_to_file
txt2dict = txt_to_dict
str2file = string_to_file
array2str = array_to_string
awkFilteredFile = pipe_through_awk
locatePeaks = locate_peaks
dateString = date_string
expandPath = expand_path
sheet2dict = sheet_to_dict
safeUpdateAttr = safe_update_attr
str2array = string_to_array

def add_ignore_nan(*args):
    """Add arrays in args, as if nan values are acutally zero."""
    retval = np.zeros(args[0].shape)
    for arg in args:
        i = ~np.isnan(arg)
        retval[i] += arg[i]
    return retval

def downSample(x,n):
    """Reduce the number of points in x by factor, summing
    n-neighbours. Any remaining data for len(x) not a multiple of n is
    discarded."""
    return np.array([np.sum(x[i*n:i*n+n]) for i in range(int(len(x)/n))])

def myAnnotate(text,x=0.1,y=0.9,fontsize=10,textcoords='axes fraction'):
    """
    A convenient alternative to annotate. Sets coordinates to axes
    fraction. Defaults to top left of axis. If usetex=True also
    enclose text with LaTeX minipage environment for more complex
    annotation.
    """
    if matplotlib.rcParams['text.usetex']==True:
        text=r'\begin{minipage}{\textwidth}'+text+r'\end{minipage}'
    return annotate(text,(x,y),textcoords=textcoords,
             fontsize=fontsize,horizontalalignment='left',
             verticalalignment='top',)


def array2txt(x):
    """Turn array into space and newline separated text, only works
    for 0, 1 or 2 dimensional arrays.
    
    LOOKS LIKE THIS MIGHT BE SUPPLANTED BY ARRAY2STR!!
    """
    if x.ndim==0:
        return str(x)
    elif x.ndim==1:
        return ' '.join([str(ix) for ix in x])
    elif x.ndim==2:
        ## recursively create each row and then join with \n
        return '\n'.join([array2txt(x[i,:]) for i in range(x.shape[0])])


def csv2dict(path=None,*args,**kwargs):
    """Deprecated, use sheet2dict."""
    return sheet2dict(path,*args,**kwargs)

@functools.lru_cache
def cached_pycode(*args,**kwargs):
    return(pycode(*args,**kwargs))

def lambdify_sympy_expression(
        expression,     # sympy expression
        variables=(),   # a list of argument variables for the final expression
        substitutions=None, # a dict with {a:b} leading to kwargs substitutions a=b
        parameter_set=None, # special case to handle optimise_model.Parameter_Set substitutions
): 
    """A slightly faster reduced lambdify for sympy functions, parameters
    is a Parameter_Set object to get variable names from. Function
    variable are defined in the final function."""
    ## make into a python string
    t =  cached_pycode(expression,fully_qualified_modules=False) 
    ## replace math functions
    for t0,t1 in (('sqrt','np.sqrt'),):
        t = t.replace(t0,t1)
    ## argument list
    arglist = list(variables)
    if substitutions is None: substitutions = {}
    arglist.extend([f'{key}=substitutions[{repr(key)}]' for key in substitutions])
    ## replace variable names with parameter_set['variable_name']
    if parameter_set is not None: 
        for t0 in parameter_set.keys() :
            t = re.sub(r"\b"+t0+r"\b",f"parameter_set[{repr(t0)}]",t)
        arglist.append('parameter_set=parameter_set')
    ## create function of variables with eval, also getting
    ## closure on parameter_set
    eval_expression = f'lambda {",".join(arglist)}: {t}'
    return(eval(eval_expression))

def latexify_sympy_expression_string(string_from_sympy_expression) :
    """Massive hack to get sympy matrix elements to be latex compatible."""
    warnings.warn("Has errors I believe.")
    string = string_from_sympy_expression
    for original,replacement in (
            (r'\.0\b',''), # change 1.0 4.0 etc to 1 or 4
            (r'\b1\*',''), # eliminate 1.0*
            (r'\*1\b',''), # eliminate *1.0
            (r'\*',''), # eliminate multiplication signs
            (r' ',''), # eliminate spaces              
            (r'0\.5([^0-9])',r'\\frac{1}{2}\1'),
            (r'0\.666666666[0-9]*([^0-9])',r'\\frac{2}{3}\1'),
            (r'0\.16666666[0-9]*([^0-9])',r'\\frac{1}{6}\1'),
            (r'0.08333333333[0-9]*([^0-9])',r'\\frac{1}{12}\1'),
            (r'0\.33333333[0-9]*([^0-9])',r'\\frac{1}{3}\1'),
            (r'1\.33333333[0-9]*([^0-9])',r'\\frac{4}{3}\1'),
            (r'0.353553390[0-9]+([^0-9])',r'2^\\frac{-3}{2}\1'),
            (r'0.11785113[0-9]+([^0-9])',r'\\frac{2^\\frac{-3}{2}}{3}\1'),
            (r'1\.4142135[0-9]+([^0-9])',r'2^\\frac{1}{2}\1'),
            (r'2\.82842712[0-9]*([^0-9])',r'2^\\frac{3}{2}\1'),
            (r'5\.65685424[0-9]*([^0-9])',r'2^\\frac{5}{2}\1'),
            (r'0\.94280904[0-9]*([^0-9])',r'\\frac{2^\\frac{3}{2}}{3}\1'),
            (r'0\.70710678[0-9]*([^0-9])',r'2^\\frac{-1}{2}\1'),
            (r'0\.4714045207[0-9]+([^0-9])',r'\\frac{2^\\frac{1}{2}}{3}\1'),
            (r'([Aλγgηξ])([DH])',r'\1_\2'),
            (r'J\(J\+1\)','x'),
            (r'sqrt\((.)\)',r'\1^\\frac{1}{2}'),
            (r'sqrt\(([^)]+)\)',r'(\1)^\\frac{1}{2}'),
            (r'\+',r' + '), # put some spaces back in
            (r'-',r' - '), # put some spaces back in
            (r'J ([+-]) ',r'J\1'), # not these ones
            (r'x ([+-]) ',r'x\1'), # not these ones
            (r'\( ([+-]) ',r'(\1'), # not these ones
            (r'\(x\)',r'x'), # not these ones
            (r'J([^+-]*)\(J\+1\)',r'\1x'), # not these ones
            (r'([xJ)])([0-9]+)',r'\1^{\2}'),                     # superscript powers
            ):
        string = re.sub(original,replacement,string)
    return(string)

