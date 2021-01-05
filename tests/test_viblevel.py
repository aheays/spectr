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

def test_viblevel_init():
    viblevel.VibLevel('test','[14N]2')

def test_viblevel():
    t = viblevel.VibLevel('test','[14N]2')
    t.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    t.add_level(name='A.3Πu(v=0)',Tv=3000,Bv=1.2,Av=100)
    t.construct()
    print( len(t.rotational_level))
    print( len(t.vibrational_spin_level))
        # print( t.vibrational_level)
    # print( t.vibrational_spin_level)
    # print( t.rotational_level[:5])
    from spectr import plotting
    fig = plotting.qfig(1)
    ax = fig.gca()
    ax.plot(t.E)
    # t.rotational_level.plot('J','E',ax=ax)
    # for i in range(t.H.shape[1]):
        # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # plotting.legend()
    plotting.show()
    # assert False
test_viblevel()

# def test_vibline():
    # import time ; timer = time.time() # DEBUG
    # x = viblevel.VibLevel('x','[14N]2')
    # y = viblevel.VibLevel('x','[14N]2')
    # # y.add_level(name='A.3Πu(v=0)',Tv=3000,Bv=1.2,Av=100)
    # vs = range(0,1)
    # for v in vs:
        # x.add_level(name=f'X.3Σ+u(v={v})',Tv=1000,Bv=1)
        # y.add_level(name=f'A.3Πu(v={v})',Tv=3000,Bv=1.2,Av=100)
    # z = viblevel.VibLine('z',y,x)
    # for v1 in vs:
        # for v2 in vs:
            # z.add_transition_moment(name=f'A.3Πu(v={v1})-X.3Σ+u(v={v2})')
    # print('Time elapsed:',format(time.time()-timer,'12.6f'),'line: 42 file: /home/heays/src/python/spectr/tests/test_viblevel.py'); timer = time.time() # DEBUG

    # import time ; timer = time.time() # DEBUG
    # z.construct()
    # print('Time elapsed:',format(time.time()-timer,'12.6f'),'line: 49 file: /home/heays/src/python/spectr/tests/test_viblevel.py'); timer = time.time() # DEBUG

    # x.timestamp = 0
    # y.timestamp = 0
    # z.timestamp = 0
    # import time ; timer = time.time() # DEBUG
    # z.construct()
    # print('Time elapsed:',format(time.time()-timer,'12.6f'),'line: 49 file: /home/heays/src/python/spectr/tests/test_viblevel.py'); timer = time.time() # DEBUG

    # print( len(z.vibrational_spin_line))
    # print( z.rotational_line[:5])
    # print( len(z.rotational_line))
    # z.rotational_line.remove_match(Sij=0)
    # print( len(z.rotational_line))

    # from spectr import plotting
    # fig = plotting.qfig(1)
    # ax = fig.gca()
    # # print( z.vibrational_line)
    # # print( z.vibrational_spin_line)
   #  
    # z.rotational_line['Teq'] = 300
    # z.rotational_line['partition'] = 1
    # z.rotational_line.verbose = True
    # for key in (
            # 'Sij',
            # ):
        # z.rotational_line.assert_known(key)
    # # print( z.rotational_line['classname'])
    # # print( z.rotational_line['classname_l'])
    # # print( z.rotational_line['J_l'])
    # # print( z.rotational_line['species'])
    # # print( z.rotational_line['label_l'])
    # # print( z.rotational_line['species_l'])
    # # print( z.rotational_line['S_l'])
    # # print( z.rotational_line['σv_l'])
    # # print( z.rotational_line['sa_l'])
    # # print( z.rotational_line['g_l'])
    # # # z.rotational_line['α_l'] 
    # # x,y = z.rotational_line.calculate_spectrum(xkey='ν',ykey='σ',ΓG=1,ΓL=None)
    # # ax.plot(x,y)
    # # # for i in range(t.H.shape[1]):
        # # # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # # # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # # # plotting.legend()
    # # plotting.show()
    # # # assert False
   # # #  
    # # # print( t.rotational_level)
    # # # # print( t.vibrational_level)
    # # # from spectr import plotting
    # # # fig = plotting.qfig(1)
    # # # ax = fig.gca()
    # # # t.rotational_level.plot('J','E',ax=ax)
    # # # # for i in range(t.H.shape[1]):
        # # # # # ax.plot(t.J,t.H[:,i,i],label=str(i))
        # # # # ax.plot(t.J,t.Hp[:,i].real,label=str(i))
    # # # # plotting.legend()
    # # # plotting.show()
    # # # # assert False

#  
