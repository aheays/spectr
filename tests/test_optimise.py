import time

from matplotlib import pyplot as plt
import numpy as np
from pytest import approx

from spectr import optimise
from spectr.optimise import *
from spectr import tools
from spectr.dataset import Dataset

show_plots =False

# def test_instantiate_optimiser():
    # t = optimise.Optimiser()

# def test_P():
    # x = P(25)
    # assert x.value == 25
    # x = P(25,True,1e-3)
    # assert x.value == 25
    # assert x.vary == True
    # assert x.step == 1e-3
    # assert x*5 == 125

# def test_optimiser_add_parameter():
    # t = optimise.Optimiser()
    # p = t.add_parameter(0.1,True,1e-5)
    # assert len(t.parameters)==1
    # p = t.add_parameter(0.1,True,1e-5)
    # assert len(t.parameters)==2

# def test_optimiser_named_parameter():
    # t = optimise.Optimiser(x=1)
    # assert len(t.parameters)==1
    # assert t['x'] == 1
    # t = optimise.Optimiser(x=1, y=P(25,True), z=(1,False,1e-3),)
    # assert len(t.parameters)==3
    # assert t['x'] == 1
    # assert t['y'] == 25
    # assert t['z'] == 1

# # def test_optimiser_get_print_parameter_dataset():
    # # t = optimise.Optimiser()
    # # t.add_parameter(x=1)
    # # t.add_parameter(y=(0.1,True,1e-5))
    # # d = t.get_parameter_dataset()
    # # print(d)

def test_suboptimisers_get_parameters():
    t = optimise.Optimiser()
    t.add_parameter(1,True,)
    t.add_parameter(0.1,True,1e-5)
    u = optimise.Optimiser()
    u.add_parameter(1,True )
    u.add_parameter(0.1,True,1e-5)
    t.add_suboptimiser(u)
    assert u._get_parameters()[0] == [1.0,0.1]
    assert u._get_parameters()[1] == [1e-4,1e-5]
    assert t._get_parameters()[0] == [1.0,0.1,1.0,0.1]
    assert t._get_parameters()[1] == [1e-4,1e-5,1e-4,1e-5,]
    u = optimise.Optimiser()
    u.add_parameter(1,True )
    u.add_parameter(0.1,True,1e-5)
    t = optimise.Optimiser('t',u)
    t.add_parameter(1,True )
    t.add_parameter(0.1,True,1e-5)
    assert len(t._suboptimisers) == 1
    assert len(t._get_parameters()[0]) == 4

def test_get_all_suboptimisers():
    t = optimise.Optimiser()
    t.add_parameter(1)
    u = optimise.Optimiser()
    u.add_parameter(1)
    v = optimise.Optimiser()
    v.add_parameter(1)
    t.add_suboptimiser(u)
    u.add_suboptimiser(v)
    assert len(t._suboptimisers) == 1
    assert len(t._get_all_suboptimisers()) == 3
    assert len(t._get_parameters()[0]) == 0
    t.add_suboptimiser(v)
    assert len(t._get_all_suboptimisers()) == 3
    assert len(t._get_parameters()[0]) == 0

def test_optimiser_construct():
    t = optimise.Optimiser()
    t.add_parameter(1)
    t.add_parameter(0.1,True,1e-5)
    t.add_construct_function(lambda: [1,2,3])
    assert len(t._construct_functions) == 1
    assert list(t.construct()) == [1,2,3]
    t = optimise.Optimiser()
    x = t.add_parameter(0.1,False,1e-5)
    t.add_construct_function(lambda: x-1)
    assert list(t.construct()) == [-0.9]
    x.value = 0.2
    assert x.value == 0.2
    assert list(t.construct()) == [-0.8]

def test_optimiser_has_changed():
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    assert t.has_changed()
    t.construct()
    assert not t.has_changed()
    t = optimise.Optimiser(x=1,y=(0.1,True,1e-5))
    u = optimise.Optimiser(z=1,w=(0.1,True,1e-5))
    t.add_suboptimiser(u)
    assert t.has_changed()
    assert u.has_changed()
    t.construct()
    assert not t.has_changed()
    assert not u.has_changed()
    u['z'] = 5
    assert u.has_changed()
    assert t.has_changed()
    t.construct()
    t['x'] = 2.5
    assert t.has_changed()
    assert not u.has_changed()

def test_optimise():
    t = optimise.Optimiser()
    x = t.add_parameter(0.1, True,1e-5)
    t.add_construct_function(lambda: x-1)
    t.optimise(verbose=True )
    assert tools.rms(t.residual) < 1e-2
    assert abs(x.value-1) < 1e-2

def test_optimiser_format_input():
    t = optimise.Optimiser()
    # assert t.format_input() == "from spectr import *\n\no = Optimiser(name='o')"
    t.print_input()
    t.print_input('^from')
    assert t.format_input('^from.*') == "from spectr import *\n"

def test_format_input_decorator():
    ## explicit crateion of method
    class c(Optimiser):
        def m(self,x):
            def f():
                return x-5
            self.add_construct_function(f)
            self.add_format_input_function(lambda: f'{self.name}.m(x={repr(x)})')
    optimiser = c()
    optimiser.m(x=25)
    assert list(optimiser.construct()) == [20]
    print(optimiser.format_input())
    ## decorate creation of method
    class c(Optimiser):
        @optimise_method()
        def m(self,x):
            return x-5
    optimiser = c()
    optimiser.m(x=P(25))
    assert len(optimiser._construct_functions)==1
    assert len(optimiser.parameters)==1
    assert len(optimiser._format_input_functions) == 2
    assert list(optimiser.construct()) == [20]
    ## also detect suboptimisers
    class c(Optimiser):
        @optimise_method()
        def m(self,p,o):
            return p-5
    optimiser = c()
    optimiser.m(p=P(25),o=Optimiser('hello'))
    assert len(optimiser._construct_functions)==1
    assert len(optimiser.parameters)==1
    assert len(optimiser.suboptimisers)==1
    assert len(optimiser._format_input_functions) == 3
    assert list(optimiser.construct()) == [20]

def test_optimise_method_decorator():
    ## an Optimiser subclass using optimise_method
    class C(Optimiser):
        @optimise_method('m')
        def m(self,x,_cache=None):
            _cache['x'] = float(x)
            return [float(x)]
        @optimise_method('n')
        def n(self,x,y=None):
            return [float(x+1)]
    ## an instance
    c = C()
    ## add some parmaeters
    pm = c.m(P(5, True))
    pn = c.n(P(5, True),y=5)
    ## fit them
    c.optimise()
    ## check the results are as expected
    assert pm['x'] == approx(0)
    assert pn['x'] == approx(-1,1e-3)

# def test_optimiser_str():
    # t = optimise.Optimiser()
    # p = t.add_parameter_dict(x=1,y=(0.1,True,1e-5))
    # print( t)

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

def test_dataset_parameters():
    o = Optimiser()
    d = Dataset(
        x=[1.0,2.0,3.0],
        x_vary=[True,True,False,],
    )
    d['x_unc'] = np.nan
    d['x_step'] = 1e-3
    o.add_suboptimiser(d)
    p,s,dp = o._get_parameters()
    assert p == [1,2]
    assert s == [1e-3,1e-3]
    assert np.isnan(dp[0]) and np.isnan(dp[1])
    o._set_parameters([4,5])
    assert all(d['x'] == [4,5,3])
    o._set_parameters([4.5,5.5],[1,2])
    assert all(d['x_unc'][:2] == [1,2])
    assert np.isnan(d['x_unc'][2])
    def _f():
        return d['x']-np.array([2,3,4])
    o.add_construct_function(_f)
    o.optimise()
    assert d['x'][0] == approx(2)
    assert d['x'][1] == approx(3)
    assert d['x'][2] == 3

# def test_save_to_directory():
    # t = optimise.Optimiser('t')
    # u = optimise.Optimiser('u')
    # t.add_suboptimiser(u)
    # u.description = 'A description.'
    # t.add_construct_function(lambda: tools.randn(30))
    # u.add_construct_function(lambda: tools.randn(30)*3)
    # t.optimise()
    # t.save_to_directory('tmp/test_save_to_directory')
