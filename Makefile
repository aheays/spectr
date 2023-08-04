all:
	make -C spectr
	python -m build

clean:
	rm spectr/*.so
