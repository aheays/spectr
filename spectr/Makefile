fortran_tools.cpython-311-x86_64-linux-gnu.so: fortran_tools.f90
	python -m numpy.f2py -c fortran_tools.f90 -m fortran_tools  -llapack --f90flags="-Wall -ffree-line-length-none -static-libgfortran"

clean:
	rm *.so
