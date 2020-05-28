from spectr import quantum_numbers

def test_decode_level():
    t = quantum_numbers.decode_level('32S16O_A.3Π(v=0,Ω=1,J=5)')
    assert t == {'species': '32S16O', 'label': 'A', 'S': 1.0, 's': 0,
                 'Λ': 1, 'v': 0, 'Ω': 1.0, 'J': 5.0}
