from spectr.levels import Level

def test_level_construct():
    t = Level()
    assert t.name == 'level'
    assert t['class'] == 'level'

def test_level_assignment():
    t = Level(name='ddd')
    assert t.name == 'ddd'
    t['description'] = 'fff'
    t['notes'] = ['a','b']
    assert list(t['notes']) == ['a','b']

def test_level_str():
    t = Level(name='ddd',description='fff',notes=['a','b'])
    print( t)

def test_level_decode():
    t = Level(encoded='32S16O_A.3Π(v=0,Ω=1,J=5)')
    assert t['species'] == '32S16O'
    t = Level(encoded=['32S16O_A.3Π(v=0,Ω=1,J=5)', '33S16O_A.3Π(v=0,Ω=1,J=6)'])
    assert list(t['species']) == ['32S16O', '33S16O']


    
# def test_level_prototypes():
    # t = Level()
    # print( t._infer('Tref'))
    # # assert t['Tref'] == 0
    # assert t['Tref'] == 0
