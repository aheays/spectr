from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import viblevel
# from spectr import spectrum
# from spectr import tools
# from spectr import lines
# from spectr import levels
# from spectr import plotting

make_plot = False 

def test_init():
    viblevel.VibLevel('test','14N2')

# def test_init():
    # t = viblevel.VibLevel('test','[14N]2')
    # t.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    # t.add_level(name='A.3Πu(v=0)',Tv=3000,Bv=1.2,Av=100)
    # t.construct()
    # # t.finalise()
    # # # print( t.rotational_level)
    # # # print( t.vibrational_level)
    # # print( t.vibrational_spin_level)
    # # from spectr import plotting
    # # fig = plotting.qfig(1)
    # # ax = fig.gca()
    # # t.rotational_level.plot('J','E',ax=ax)
    # # # for i in range(t.H.shape[1]):
        # # # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # # # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # # # plotting.legend()
    # # plotting.show()
    # # # assert False

def test_init():
    x = viblevel.VibLevel('x','[14N]2')
    x.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    y = viblevel.VibLevel('x','[14N]2')
    y.add_level(name='A.3Πu(v=0)',Tv=3000,Bv=1.2,Av=100)
    z = viblevel.VibLine('z',y,x)
    z.add_transition_moment(name='A.3Πu(v=0)-X.3Σ+u(v=0)')
    z.construct()
    # from spectr import plotting
    # fig = plotting.qfig(1)
    # ax = fig.gca()
    z.rotational_line['Teq'] = 300
    z.rotational_line['partition'] = 1
    z.rotational_line.verbose = True
    print( z.rotational_line['classname'])
    print( z.rotational_line['classname_l'])
    print( z.rotational_line['J_l'])
    print( z.rotational_line['isotopologue_l'])
    print( z.rotational_line['label_l'])
    print( z.rotational_line['species_l'])
    print( z.rotational_line['S_l'])
    print( z.rotational_line['σv_l'])
    print( z.rotational_line['sa_l'])
    print( z.rotational_line['g_l'])
    # # z.rotational_line['α_l'] 
    # x,y = z.rotational_line.calculate_spectrum(xkey='ν',ykey='σ',ΓG=1,ΓL=None)
    # ax.plot(x,y)
    # # for i in range(t.H.shape[1]):
        # # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # # plotting.legend()
    # plotting.show()
    # # assert False
   # #  
    # # print( t.rotational_level)
    # # # print( t.vibrational_level)
    # # from spectr import plotting
    # # fig = plotting.qfig(1)
    # # ax = fig.gca()
    # # t.rotational_level.plot('J','E',ax=ax)
    # # # for i in range(t.H.shape[1]):
        # # # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # # # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # # # plotting.legend()
    # # plotting.show()
    # # # assert False

test_init()
