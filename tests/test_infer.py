import numpy as np

from spectr import infer

def test_species_from_encoded_level():
    t = infer.species_from_encoded_level('32S16O_A.3Π(v=0,Ω=1,J=5)')
    assert t == '32S16O'
    t = infer.species_from_encoded_level(['32S16O_A.3Π(v=0,Ω=1,J=5)', '33S16O_A.3Π(v=0,Ω=1,J=5)',])
    assert list(t) == ['32S16O','33S16O']
