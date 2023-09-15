


all: fortran_extension python_package

python_package: 
	python -m build

fortran_extension: 
	make -C spectr

clean:
	rm spectr/*.so
