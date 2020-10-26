import re

import numpy as np


from . import tools
from . import plotting
from . import kinetics
from . import atmosphere
from .dataset import Dataset

def load_verif(output_directory):
    for filename in tools.glob(f'{output_directory}/Reactions/*_verif.dat'):
        print('DEBUG:', filename)
        data = Dataset()
        with open(filename,'r') as fid:
            previous_line = None
            for line in fid:

                ## beginning of new time period
                if r:=re.match(r'^ *T\(SEC\)= *(.*)',line):
                    time = float(r.group(1))

                ## new species found
                if r:=re.match(r'^ *REACTIONS OF PRODUTION  (.*)',line):
                    species = previous_line.strip()
                    print('DEBUG:', species)
                    ## get production rates
                    production_total = float(r.group(1))
                    line = fid.readline()
                    while r:=re.match(r'^ *([0-9]+) +(.*)',line):
                        data.append(
                            t=time,
                            species=species,
                            type='production',
                            reaction=r.group(1).strip(),
                            rate=float(r.group(2)),)
                        line = fid.readline()

                ## get destruction rates
                if r:=re.match(r'^ *REACTIONS OF DESTRUCTION  (.*)',line):
                    destruction_total = float(r.group(1))
                    line = fid.readline()
                    while r:=re.match(r'^ *([0-9]+) +(.*)',line):
                        data.append(
                            t=time,
                            species=species,
                            type='destruction',
                            reaction=r.group(1).strip(),
                            rate=float(r.group(2)),)
                        line = fid.readline()
                previous_line = line

        break
    print( len(data))
    print( data.unique('species'))
    # print( data.matches(species='C'))
