import os
from setuptools import setup
from setuptools.command.build import build


class custom_build(build):

    def run(self):
        os.system('make -C spectr')
        build.run(self)

setup(cmdclass=dict(build=custom_build),)

