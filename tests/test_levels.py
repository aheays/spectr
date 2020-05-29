from pprint import pprint 
from spectr.levels import Levels

def test_construct():
    t = Levels()
    assert t.name == 'Levels'
    assert t['class'] == 'Levels'
    assert len(t._prototypes)>0 

def test_assignment():
    t = Levels(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']

def test_str():
    t = Levels(name='ddd',description='fff',notes=['a','b'])
    print(t)

# def test_decode():
    # t = Levels(encoded='32S16O_A.3Π(v=0,Ω=1,J=5)')
    # assert t['species'] == '32S16O'
    # t = Levels(encoded=['32S16O_A.3Π(v=0,Ω=1,J=5)', '33S16O_A.3Π(v=0,Ω=1,J=6)'])
    # assert list(t['species']) == ['32S16O', '33S16O']


    
