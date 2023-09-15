all: fortran_extension python_package

## build python package, sets platform compatibility tag manually (!)
python_package: 
	python -m build -w \
	-C="--global-option=--plat-name" -C="--global-option=linux_x86_64" .

fortran_extension: 
	make -C spectr

clean:
	rm spectr/*.so
