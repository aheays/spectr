from pprint import pprint
import pytest
from pytest import raises,approx
import numpy as np

from spectr import *

make_plot =  True 

def test_viblevel_init():
    viblevel.VibLevel('test','[14N]2')

def test_viblevel():
    t = viblevel.VibLevel('test','[14N]2',J=range(5))
    t.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    for v in range(1):
        t.add_level(name=f'A.3Πu(v={v})',Tv=3000,Bv=1.2,Av=100)
    t.construct()
    # assert len(t.rotational_level) == 271
    # assert len(t.vibrational_spin_level) == 9
    # assert len(t.vibrational_level) == 2
    # print( len(t.rotational_level))
    if make_plot:
        qfig()
        i = t.rotational_level['J'] < t.rotational_level['Ω']
        t.rotational_level[~i].plot('J','E')
    # t.rotational_level['Tex'] = 300
    # t.rotational_level.partition_source = 'self'
    # t.rotational_level['Inuclear'] = 1
    # t.rotational_level['species']
    # t.rotational_level['Tex']
    # t.rotational_level.verbose = True 
    # t.rotational_level['g']
    # t.rotational_level['partition']
    # t.rotational_level.sort('J')
    # print( t.rotational_level)
        
def test_vibline():
    x = viblevel.VibLevel('x','[14N]2')
    y = viblevel.VibLevel('x','[14N]2')
    x.add_level(name='X.3Σ+u(v=0)',Tv=1000,Bv=1)
    z = viblevel.VibLine('z',y,x,J_l=range(21))
    for v in range(1):
        y.add_level(name=f'A.3Πu(v={v:d})',Tv=3000+v*1000,Bv=1.2,Av=100)
        z.add_transition_moment(name=f'A.3Πu(v={v:d})-X.3Σ+u(v=0)')
    import time ; timer = time.time() # DEBUG
    z.construct()
    print('Time elapsed:',format(time.time()-timer,'12.6f'),'line: 42 file: /home/heays/src/python/spectr/tests/test_viblevel.py'); timer = time.time() # DEBUG
    # assert len(z.rotational_line) == 1674
    # assert len(z.vibrational_line) == 1
    # assert len(z.vibrational_spin_line) == 18
    # print( len(z.rotational_line))
    i = (z.rotational_line['J_l']<z.rotational_line['Ω_l'])|(z.rotational_line['J_u']<z.rotational_line['Ω_u'])
    z.rotational_line.index(~i)
    print( len(z.rotational_line))
    # print( z.rotational_line[:100])

    # # # z.rotational_line.verbose =False 
    # # # z.rotational_line['partition']
    # # # print( z.rotational_line)

    # # z.rotational_line.verbose = True
    # # for key in (
            # # # 'species',
            # # # 'Sij',
            # # # 'f',
            # # # # 'Inuclear_l',
            # # # # # 'partition_source_l',
            # # # # 'species_l',
            # # # 'E_l',
            # # # 'J_l',
            # # # 'sa_l',
            # # # 'point_group_l',
            # # # 'Inuclear_l',
            # # # 'gnuclear_l',
            # # # 'g_l',
            # # # # 'Tex_l',
            # # # # # 'partition_source_l',
            # # # 'Z',
            # # 'α_l',
            # # # # # # 'σ',
            # # ):
        # # print( key,z.rotational_line[key][:5])
        # # print( key,np.max(z.rotational_line[key]))
        # # print( key,np.sum(z.rotational_line[key]))


    if make_plot:
        qfig(1)
        z.rotational_line['Tex'] = 100
        z.rotational_line.Zsource = 'self'
        z.rotational_line['Inuclear_l'] = 1
        z.rotational_line.plot_stick_spectrum('ν','σ')
        # z.rotational_line.plot_stick_spectrum('ν','Sij')
        # # yscale('log')
    # # # # # print( z.rotational_line.matches(ν_max=0))
    # # # # # print( z.rotational_line.matches(ν_max=0))
    # # # # print( z.rotational_line)
    # # # # print(x.rotational_level)

# test_viblevel()        
# test_vibline()

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


if make_plot:
    show()
