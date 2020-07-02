import time

from spectr import optimise

def test_ParameterSet_instantiate():
    t = optimise.ParameterSet()

def test_ParameterSet_set_get():
    t = optimise.ParameterSet()
    t['x'] = 5
    assert t['x'] == 5
    assert t['x'] != 4
    t = optimise.ParameterSet()
    t['x'] = 5,False,1e-3
    x = t['x']
    assert x.value == 5
    assert x.vary == False 
    assert x.step == 1e-3
    t = optimise.ParameterSet()
    t['x'] = 5
    x = t['x']
    assert x == 5
    t = optimise.ParameterSet()
    t['x'] = (5,False,1e-3)
    x = t['x']
    assert x.value == 5
    assert x.vary == False 
    assert x.step == 1e-3
    t = optimise.ParameterSet(x=5,y=0.1)
    x = t['x']
    assert x == 5
    assert t['y'] == 0.1
    t = optimise.ParameterSet(x=(5,False,1e-3,'description'),y=(0.1,True,1e-10))
    x = t['x']
    assert x.value == 5
    assert x.vary == False 
    assert x.step == 1e-3
    print( x.description)
    assert x.description == 'x description'

def test_ParameterSet_set_print():
    t = optimise.ParameterSet(x=(5,False,1e-3,'description'),y=(0.1,True,1e-10))
    print(t)

def test_ParameterSet_set_save():
    t = optimise.ParameterSet(x=(5,False,1e-3,'description'),y=(0.1,True,1e-10))
    t.save('tmp/parameters.psv')

def test_timestamps():
    t0 = time.time()
    t = optimise.ParameterSet(x=1)
    assert t['x'].timestamp == t.timestamp
    t['y'] = 25
    assert t['x'].timestamp < t.timestamp
    assert t['y'].timestamp == t.timestamp

def test_instantiate_optimiser():
    t = optimise.Optimiser()

def test_optimiser_add_parameter():
    t = optimise.Optimiser()
    p = t.add_parameter(0.1,True,1e-5)
    assert len(t.parameters)==1
    p = t.add_parameter(0.1,True,1e-5)
    assert len(t.parameters)==2

def test_optimiser_add_parameter_set():
    t = optimise.Optimiser()
    p = t.add_parameter_set(x=1,y=(0.1,True,1e-5))
    assert len(t.parameters)==2
    assert t.parameters[0] is p['x']
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    assert len(t.parameters)==2

def test_optimiser_get_print_parameter_array():
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    d = t.get_parameter_array()
    print( d)

