from spectr.levels import Base

def test_base_construct():
    t = Base()
    assert t.name == 'base'
    t = Base()
    assert t.name == 'base'
    assert t['class'] == 'base'

# def test_base_prototypes():
    # t = Base()
    # print( t._infer('Tref'))
    # # assert t['Tref'] == 0
    # assert t['Tref'] == 0
