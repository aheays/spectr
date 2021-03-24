from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import *

make_plot = False 

def test_level_init():
    viblevel.VibLevel('test','[14N]2')

def test_one_level():
    t = viblevel.VibLevel('test','[14N]2',J=range(11))
    t.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    t.construct()
    if make_plot:
        qfig()
        t.level.plot('J','E')
        show()

def test_multiple_levels():
    t = viblevel.VibLevel('test','[14N]2',J=range(11))
    for v in range(4):
        t.add_level(name=f'A.3Πu(v={v})',Tv=3000+500*v,Bv=1.2,Av=100)
    t.construct()
    if make_plot:
        qfig()
        t.level.plot('J','E')
        show()

def test_LS_interaction():
    t = viblevel.VibLevel('test','[14N]2',J=range(21))
    t.add_level(name=f'A.3Πu(v=0)',Tv=3000,Bv=2.5,Av=50)
    t.add_level(name=f'B.3Σ+u(v=0)',Tv=3080,Bv=1,λv=1)
    t.add_LS_coupling('A.3Πu(v=0)','B.3Σ+u(v=0)',ηv=100)
    t.construct()
    if make_plot:
        qfig()
        t.level.plot('J','E')
        show()

def test_JL_interaction():
    t = viblevel.VibLevel('test','[14N]2',J=range(21))
    t.add_level(name=f'A.3Πu(v=0)',Tv=3000,Bv=2.5,Av=50)
    t.add_level(name=f'B.3Σ+u(v=0)',Tv=3080,Bv=1,λv=1)
    t.add_JL_coupling('A.3Πu(v=0)','B.3Σ+u(v=0)',ξv=10)
    t.construct()
    if make_plot:
        qfig()
        t.level.plot('J','E')
        show()

def test_line():
    x = viblevel.VibLevel('x','[14N]2')
    x.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    y = viblevel.VibLevel('y','[14N]2')
    y.add_level(name=f'A.3Πu(v=0)',Tv=3000*1000,Bv=1.2,Av=100)
    z = viblevel.VibLine('z',y,x,J_l=range(0,30),)
    z.add_transition_moment(f'A.3Πu(v=0)','X.3Σ+u(v=0)',μv=1)
    z.construct()
    if make_plot:
        qfig()
        z.line['Tex'] = 100
        z.line.Zsource = 'self'
        z.line['Inuclear_l'] = 1
        z.line['ν']
        z.line.plot_stick_spectrum('ν','σ')
        show()


def test_multiple_lines():
    x = viblevel.VibLevel('x','[14N]2')
    y = viblevel.VibLevel('y','[14N]2')
    x.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    z = viblevel.VibLine('z',y,x,J_l=range(0,10),)
    for v in range(3):
        y.add_level(name=f'A.3Πu(v={v:d})',Tv=3000+v*1000,Bv=1.2,Av=100)
        z.add_transition_moment(f'A.3Πu(v={v:d})','X.3Σ+u(v=0)',μv=v)
    z.construct()
    if make_plot:
        qfig()
        x.level.plot('J','E')
        qfig()
        y.level.plot('J','E')
        qfig()
        z.line['Tex'] = 100
        z.line.Zsource = 'self'
        z.line['Inuclear_l'] = 1
        z.line['ν']
        z.line.plot_stick_spectrum('ν','σ')
        show()
