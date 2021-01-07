## example f2py3 command:
## 
## cronic f2py3 --quiet --opt="-O3 -fcheck=bounds" -llapack -c -m spectra_fortran myf.f90 spectra_fortran.f90 

F2PY_SUFFIX := $(shell python3-config --extension-suffix)

## gfortran flags and f2py flags
FC := gfortran -fopenmp  -lgomp -O3 -llapack -Wall -ffree-line-length-none # normal / optimised / parallel
F2PY := f2py3 -c --quiet -lgomp -llapack --f90flags="-Wall -fopenmp -ffree-line-length-none" --opt="-O3" # normal / optimised / parallel


## alterantive flags for debug
# FC := gfortran -llapack -fbounds-check -g -ffree-line-length-none # debug
# F2PY := f2py3  -c --quiet -lgomp -llapack --f90flags="-fbounds-check -g -ffree-line-length-none" --opt="-O3" # debug

## string used to identify something abtou f2py output shared objects
# F2PY := f2py3 -c --fcompiler=gfortran -llapack  #debug

# all:  myc myf fortran_tools lib_molecules_fortran
all:           fortran_tools
clean:         ; trash *.so *.o *.mod  > /dev/null 2>&1 || true
# myc:           myc.so ; 
# myc.so:        myc.c  ; gcc -lm -fPIC -shared -o myc.so myc.c
fortran_tools: fortran_tools$(F2PY_SUFFIX) ;
fortran_tools$(F2PY_SUFFIX): fortran_tools.f90  ; $(F2PY) -m fortran_tools fortran_tools.f90 
# lib_molecules_fortran: lib_molecules_fortran$(F2PY_SUFFIX) ;

%$(F2PY_SUFFIX): %.f90 ; $(F2PY) -m $* $< 
%.mod: %.f90 ; $(FC) -c $<


