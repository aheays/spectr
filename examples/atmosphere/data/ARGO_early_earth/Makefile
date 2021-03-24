all: argo

argo: argo.o dlsode.o
	gfortran -O2 -ffpe-summary='none' -o argo argo.o dlsode.o

argo.o: argo.f
	gfortran -O2 -ffpe-summary='none' -c argo.f

dlsode.o: dlsode.f
	gfortran -O2 -ffpe-summary='none' -c dlsode.f

clean:
	rm -rf *o argo *.pyc

