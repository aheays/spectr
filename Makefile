#####################
## f90 compilation ##
#####################

## compiler command with flags
FC := gfortran -fopenmp -lgomp -O3 -llapack -Wall -ffree-line-length-none # normal / optimised / parallel

## make rule
%.mod: %.f90 ; $(FC) -c $<


######################
## f2py compilation ##
######################

## example f2py3 command:
## cronic f2py3 --quiet --opt="-O3 -fcheck=bounds" -llapack -c -m spectra_fortran myf.f90 spectra_fortran.f90 

## Outputs a .so file with machine dependent suffix.
F2PY_SUFFIX := $(shell python3-config --extension-suffix)

## compiler command with flags built in
F2PY := f2py3 -c --quiet -lgomp -llapack --f90flags="-Wall -fopenmp -ffree-line-length-none" --opt="-O3"

## the generic make rule for a fortran file ending with the f2py
## suffix.  
%$(F2PY_SUFFIX): %.f ; $(F2PY)  -m $* $< 
%$(F2PY_SUFFIX): %.f90 ; $(F2PY)  -m $* $< 

## alterantive flags for debug
# FC := gfortran -llapack -fbounds-check -g -ffree-line-length-none # debug
# F2PY := f2py3  -c --quiet -lgomp -llapack --f90flags="-fbounds-check -g -ffree-line-length-none" --opt="-O3" # debug

## string used to identify something abtou f2py output shared objects
# F2PY := f2py3 -c --fcompiler=gfortran -llapack  #debug


###########
## rules ##
###########

all: fortran_tools line_profiles_tran2014

clean: ; trash *.so *.o *.mod  > /dev/null 2>&1 || true

fortran_tools: fortran_tools$(F2PY_SUFFIX) ;

line_profiles_tran2014: line_profiles_tran2014$(F2PY_SUFFIX) ;
