import numpy as np

from spectr import quantum_numbers

def species_from_encoded_level(encoded):
    # if np.isscalar(encoded):
    d = quantum_numbers.decode_level(encoded)
    if 'species' not in d:
        raise InferException
    return(d['species'])
        
