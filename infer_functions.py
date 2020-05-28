import numpy as np

from spectr import quantum_numbers
from spectr import tools

d = infer_functions = tools.AutoDict({})



d['class']['self'] = lambda self:self.__class__.__name__

def _f(encoded):
    d = quantum_numbers.decode_level(encoded)
    if 'species' not in d:
        raise InferException
    return(d['species'])
d['species']['encoded'] = _f


d['ν']['Ep','Epp'] = lambda Ep,Epp: Ep-Epp
d['Epp']['Ep','ν'] = lambda Ep,ν: Ep-ν
d['Ep']['Epp','ν'] = lambda Epp,ν: Epp+ν



        
