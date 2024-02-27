import os
from setuptools import setup
from setuptools.command.build import build


class custom_build(build):

    def run(self):
        os.system('python -m numpy.f2py -c fortran_tools.f90 -m fortran_tools  -llapack --f90flags="-Wall -ffree-line-length-none -static-libgfortran"')
        build.run(self)

setup(cmdclass=dict(build=custom_build),)

