import time

from matplotlib import pyplot as plt

from spectr import optimise
from spectr import tools

show_plots =False

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
    assert x.description == 'x description'
    t['x'] = 6
    assert x.value == 6
    assert x.vary == False 
    assert x.step == 1e-3
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

def test_suboptimisers_get_parameters():
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    u = optimise.Optimiser(z=1,w=(0.1,True,1e-5))
    t.add_suboptimiser(u)
    assert len(u.get_parameters()) == 2
    assert len(t.get_parameters()) == 4
    u = optimise.Optimiser(z=1,w=(0.1,True,1e-5))
    t = optimise.Optimiser('t',u,x=1,y=(0.1,True,1e-5))
    assert len(t._suboptimisers) == 1
    assert len(t.get_parameters()) == 4

def test_get_all_suboptimisers():
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    u = optimise.Optimiser(z=1,w=(0.1,True,1e-5))
    v = optimise.Optimiser(z=1,w=(0.1,True,1e-5))
    t.add_suboptimiser(u)
    u.add_suboptimiser(v)
    assert len(t._suboptimisers) == 1
    assert len(t._get_all_suboptimisers()) == 3
    assert len(t.get_parameters()) == 6
    t.add_suboptimiser(v)
    assert len(t._get_all_suboptimisers()) == 3
    assert len(t.get_parameters()) == 6

def test_optimiser_construct():
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    t.add_construct_function(lambda: [1,2,3])
    assert len(t._construct_functions) == 1
    assert list(t.construct()) == [1,2,3]
    t = optimise.Optimiser()
    x = t.add_parameter('x',0.1,False,1e-5)
    t.add_construct_function(lambda: x-1)
    assert list(t.construct()) == [-0.9]
    x.value = 0.2
    assert list(t.construct()) == [-0.8]

def test_optimiser_has_changed():
    t = optimise.Optimiser()
    pt = t.add_parameter_set(x=1,y=(0.1,True,1e-5))
    assert t.has_changed()
    t.construct()
    assert not t.has_changed()
    t = optimise.Optimiser()
    pt = t.add_parameter_set(x=1,y=(0.1,True,1e-5))
    u = optimise.Optimiser()
    pu = u.add_parameter_set(z=1,w=(0.1,True,1e-5))
    t.add_suboptimiser(u)
    assert t.has_changed()
    assert u.has_changed()
    t.construct()
    assert not t.has_changed()
    assert not u.has_changed()
    pu['z'] = 5
    assert t.has_changed()
    assert u.has_changed()
    t.construct()
    pt['x'] = 2.5
    assert t.has_changed()
    assert not u.has_changed()

def test_optimise():
    t = optimise.Optimiser()
    x = t.add_parameter('x',0.1, True,1e-5)
    t.add_construct_function(lambda: x-1)
    residual = t.optimise()
    assert tools.rms(residual) < 1e-5
    assert abs(x.value-1) < 1e-5

def test_optimiser_format_input():
    t = optimise.Optimiser()
    assert t.format_input() == "from spectr import *\n\no = Optimiser('o')"
    t.print_input()
    t.print_input('^from')
    assert t.format_input('^from.*') == "from spectr import *\n"

def test_optimiser_str():
    t = optimise.Optimiser()
    p = t.add_parameter_set(x=1,y=(0.1,True,1e-5))

def test_plot_residual():
    t = optimise.Optimiser('t')
    t.add_construct_function(lambda: tools.randn(30))
    t.add_construct_function(lambda: tools.randn(30)*2)
    u = optimise.Optimiser('u')
    u.add_construct_function(lambda: tools.randn(30)*3)
    t.add_suboptimiser(u)
    t.optimise()
    t.plot_residual()
    if show_plots:
        plt.show()

def test_plot():
    t = optimise.Optimiser('t')
    def f():
        plt.plot([1,2,5])
        plt.title('t')
    t.add_plot_function(f)
    def f():
        plt.plot([-3,-2,-1])
    t.add_plot_function(f)
    u = optimise.Optimiser('u')
    def f():
        plt.plot([10,20,30])
    t.add_plot_function(f)
    t.add_suboptimiser(u)
    t.plot()
    if show_plots:
        plt.show()

def test_save_to_directory():
    t = optimise.Optimiser('t')
    u = optimise.Optimiser('u')
    t.add_suboptimiser(u)
    u.description = 'A description.'
    t.add_construct_function(lambda: tools.randn(30))
    u.add_construct_function(lambda: tools.randn(30)*3)
    t.optimise()
    t.save_to_directory('tmp/test_save_to_directory')
    # def f():
        # plt.plot([1,2,5])
        # plt.title('t')
    # t.add_plot_function(f)
    # def f():
        # plt.plot([-3,-2,-1])
    # t.add_plot_function(f)
    # u = optimise.Optimiser('u')
    # def f():
        # plt.plot([10,20,30])
    # t.add_plot_function(f)
    # t.add_suboptimiser(u)
    # t.plot()
    # if show_plots:
        # plt.show()
