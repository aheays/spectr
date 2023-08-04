# ######################
# ## f2py compilation ##
# ######################

# ## Outputs a .so file with machine dependent suffix.
# F2PY_SUFFIX := $(python <<< "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# ## compiler command with flags built in
# # F2PY := f2py3 -c --quiet -llapack --f90flags="-Wall -ffree-line-length-none" --opt="-O3"
# F2PY := f2py3 -c --quiet -llapack --f90flags="-Wall -ffree-line-length-none" --opt="-O3"

# ## the generic make rule for a fortran file ending with the f2py
# ## suffix.  
# %$(F2PY_SUFFIX): %.f ; $(F2PY)  -m $* $< 
# %$(F2PY_SUFFIX): %.f90 ; $(F2PY)  -m $* $< 

# ## alternative flags for debug
# # FC := gfortran -llapack -fbounds-check -g -ffree-line-length-none # debug
# # F2PY := f2py3  -c --quiet -lgomp -llapack --f90flags="-fbounds-check -g -ffree-line-length-none" --opt="-O3" # debug

# ## alternative flags including omp
# # F2PY := f2py3 -c --quiet -lgomp -llapack --f90flags="-Wall -fopenmp -ffree-line-length-none" --opt="-O3"

# ## string used to identify something abtou f2py output shared objects
# # F2PY := f2py3 -c --fcompiler=gfortran -llapack  #debug

# # #####################
# # ## f90 compilation ##
# # #####################

# # ## compiler command with flags
# # # FC := gfortran -fopenmp -lgomp -O3 -llapack -Wall -ffree-line-length-none # normal / optimised / parallel
# # FC := gfortran -fopenmp -O3 -llapack -Wall -ffree-line-length-none # normal / optimised / parallel

# # ## make rule
# # %.mod: %.f90 ; $(FC) -c $<



# ###########
# ## rules ##
# ###########

# # all: fortran_tools line_profiles_tran2014
# all: fortran_tools 

# clean: ; trash *.so *.o *.mod  > /dev/null 2>&1 || true

# fortran_tools: fortran_tools$(F2PY_SUFFIX) ;

# # line_profiles_tran2014: line_profiles_tran2014$(F2PY_SUFFIX) ;

fortran_tools.cpython-311-x86_64-linux-gnu.so: fortran_tools.f90
	python -m numpy.f2py -c fortran_tools.f90 -m fortran_tools  -llapack --f90flags="-Wall -ffree-line-length-none -static-libgfortran"

clean:
	rm *.so
