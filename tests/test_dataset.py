import pytest
from pytest import raises,approx
import numpy as np

from spectr.dataset import *
from spectr.optimise import P
from spectr.exceptions import InferException

show_plots =False

def test_dataset_construct():
    t = Dataset()

def test_dataset_construct_get_set_data():
    t = Dataset()
    assert len(t) == 0
    t = Dataset()
    t.set('x',[1,2,3])
    assert len(t) == 3
    assert all(t.get('x')==[1,2,3])
    t.set('x',25,1)
    assert len(t) == 3
    assert all(t.get('x')==[1,25,3])
    t = Dataset()
    t['x'] = [1,2,3]
    assert len(t) == 3
    assert all(t['x']==[1,2,3])
    t = Dataset()
    t['x'] = []
    assert len(t) == 0
    t = Dataset(x=['a','b'])
    assert len(t) == 2
    assert list(t['x']) == ['a','b']
    t = Dataset(x=[1.,2.], d_x=[0.1,0.2],)
    assert len(t)==2
    assert list(t['x']) == [1.,2.]
    assert list(t['d_x']) == [0.1,0.2]

def test_dataset_set_defaults():
    ## set default
    t = Dataset()
    t['y'] = 'a'
    assert len(t) == 0
    assert set(t.keys()) == set(['y'])
    ## set vector then set default
    t = Dataset()
    t['x'] = [1,2]
    t['y'] = 'a'
    assert len(t) == 2
    assert all(t['x'] == [1,2])
    assert all(t['y'] == ['a','a'])
    assert set(t.keys()) == set(['x','y'])
    ## set default only, then set vector 
    t = Dataset()
    t['y'] = 'a'
    t['x'] = [1,2]
    assert len(t) == 2
    assert all(t['x'] == [1,2])
    assert all(t['y'] == ['a','a'])
    assert set(t.keys()) == set(['x','y'])
    ## set in init
    t = Dataset(x=[1,2],y='a')
    assert len(t) == 2
    assert all(t['x'] == [1,2])
    assert all(t['y'] == ['a','a'])
    assert set(t.keys()) == set(['x','y'])

def test_dataset_prototypes():
    t = Dataset()
    t.add_prototype('x',description="X is a thing.",kind='f')
    t['x'] = [5]
    assert isinstance(t['x'][0],float)
    assert t._data['x']['description'] == "X is a thing."

def test_dataset_permit_nonprototyped_data():
    t = Dataset()
    t.permit_nonprototyped_data = True
    t['x'] = [5]
    t.permit_nonprototyped_data = False
    with raises(AssertionError):
        t['y'] = [5]

def test_dataset_index():
    t = Dataset(x=[1,2,3,4,5])
    t.index([0,1])
    assert list(t['x']) == [1,2]
    t = Dataset(x=[1,2,3,4,5])
    t.index([True,True,True,False,False,])
    assert list(t['x']) == [1,2,3]

def test_dataset_get_copy():
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy()
    assert list(u['x']) == [1,2,3]
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',))
    assert 'y' not in u 
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',),[False,False,True])
    assert list(u['x']) == [3]

def test_dataset_extend():
    t = Dataset()
    t.extend()
    assert len(t)==0
    t = Dataset(x=[1])
    t.extend(x=[2])
    print( t['x'])
    assert list(t['x']) == [1,2]
    t = Dataset(x=[1,2])
    t.extend(x=[3])
    assert len(t) == 3
    assert list(t['x']) == [1,2,3]
    t = Dataset(x=[3])
    t.extend(x=[1,2])
    assert len(t) == 3
    assert list(t['x']) == [3,1,2]
    t = Dataset(x=[1,2])
    t.extend(x=[2,3])
    assert list(t['x']) == [1,2,2,3]
    assert len(t) == 4
    t = Dataset(x=[1,2],y=['a','a'])
    t.extend(x=[2,3],y=['a','b'])
    assert list(t['x']) == [1,2,2,3]
    assert list(t['y']) == ['a','a','a','b']
    assert len(t) == 4
    ## with defaults
    t = Dataset(x=[1], y='a',)
    assert all(t['x'] == [1])
    assert all(t['y'] == ['a'])
    t.extend(x=[2,3],)
    assert len(t) == 3
    assert all(t['x'] == [1,2,3])
    assert all(t['y'] == ['a','a','a'])
    t.extend(x=[3],y='b')
    assert all(t['x'] == [1,2,3,3])
    assert all(t['y'] == ['a','a','a','b'])
    t.extend(x=[4])
    assert all(t['x'] == [1,2,3,3,4])
    assert all(t['y'] == ['a','a','a','b','a'])

def test_dataset_append():
    t = Dataset(x=[])
    t.append(x=1)
    assert list(t['x']) == [1]
    assert len(t)==1
    t = Dataset(x=[0])
    t.append(x=1)
    assert list(t['x']) == [0,1]
    assert len(t)==2
    t = Dataset(x=[0,1,2])
    t.append(x=1)
    assert list(t['x']) == [0,1,2,1]
    assert len(t)==4
    t = Dataset(x=[0],y=['a'])
    t.append(x=1,y='b')
    assert list(t['x']) == [0,1]
    assert list(t['y']) == ['a','b']
    assert len(t)==2
    t = Dataset()
    t.append(x=1,)
    assert list(t['x']) == [1]
    ## t = Dataset()
    ## p = P(1,False,1e-3)
    ## t['x'] = p
    ## assert t['x'] == 1
    ## p.value = 5
    ## assert t['x'] == 5
    ## appending with defaults
    t = Dataset(x=[1],y='a')
    t.append(x=2)
    assert all(t['x'] == [1,2])
    assert all(t['y'] == ['a','a'])
    t.append(x=3,y='b')
    assert all(t['x'] == [1,2,3])
    assert all(t['y'] == ['a','a','b'])
    t.append(x=4)
    assert all(t['x'] == [1,2,3,4])
    assert all(t['y'] == ['a','a','b','a'])

def test_dataset_infer():
    t = Dataset(x=[1],y=[2])
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    t.add_infer_function('w',('y','z'),lambda self,y,z:y*z)
    assert t['w'] == [6]
    t = Dataset(x=[1,2,3],y=[2,2,2])
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    with raises(InferException):
        t['w']

def test_dataset_infer_autoremove_inferences():
    t = Dataset()
    t['x'] = [1.]
    t['y'] = [2.]
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    t['z']
    pprint( t._data['x'])
    pprint( t._data['y'])
    pprint( t._data['z'])
    assert 'z' in t
    t['x'] = [2]
    print( t)
    assert 'z' not in t
    t['z']
    t['z'] = [5]
    t['x'] = [2]
    assert 'z' in t

def test_dataset_infer_with_uncertainties():
    t = Dataset()
    t['x'],t['d_x']= [1.],[0.1]
    t['y'],t['d_y'] = [2.],[0.2]
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    assert t['d_z'] == approx(np.sqrt(0.1**2+0.2**2))
    t = Dataset()
    t['x'],t['d_x']= [1.],[0.1]
    t['y'] = [2.]
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    assert abs(t['d_z'] - np.sqrt(0.1**2))/t['d_z'] < 1e-5
    t = Dataset()
    t['y'],t['d_y'] = [2.],[0.2]
    t['p'],t['d_p'] = [3.],[0.3]
    t.add_infer_function('z',('y','p'),lambda self,y,p:y*p)
    assert t['z'] == [2*3]
    assert t['d_z'] == approx(np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)
    t = Dataset()
    t['x'],t['d_x'] = [1.],[0.1]
    t['y'],t['d_y'] = [2.],[0.2]
    t['p'],t['d_p'] = [3.],[0.3]
    t.add_infer_function('z',('x','y','p'),lambda self,x,y,p:x+y*p)
    assert t['z'] == 1+2*3
    assert abs(t['d_z'] - np.sqrt(0.1**2 + (np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)**2) )/t['d_z'] < 1e-5

def test_dataset_match_matches():
    t = Dataset(x=[1,2,2,3],y=[4,4,4,4])
    assert list(t.match(x=2)) == [False ,True,True ,False]
    assert list(t.match(x=2,y=4))== [False ,True,True ,False]
    assert list(t.match(x=2,y=5))== [False ,False,False,False]
    assert list(t.match(x=[2,3],y=4))== [False ,True,True , True]
    t = Dataset(x=[1,2,2,3],y=[4,4,4,4])
    u = t.matches(x=2)
    assert list(u['x']) == [2,2]

def test_dataset_unique_functions():
    t = Dataset(x=[1,2,2,2],y=['a','b','b','c'])
    assert list(t.unique('x')) == [1,2]
    assert set(t.unique_combinations('x','y')) == set([(1,'a'),(2,'b'),(2,'c')])
    u = t.unique_dicts('x','y')
    assert str(u) == "[{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}, {'x': 2, 'y': 'c'}]"
    u = t.unique_dicts_match('x','y')
    assert str(u) == "[({'x': 1, 'y': 'a'}, array([ True, False, False, False])), ({'x': 2, 'y': 'b'}, array([False,  True,  True, False])), ({'x': 2, 'y': 'c'}, array([False, False, False,  True]))]"
    u = t.unique_dicts_matches('x','y')

def test_dataset_sort():
    t = Dataset(x=[1,3,2])
    t.sort('x')
    assert list(t['x']) == [1,2,3]
    t = Dataset(x=[2,1,3,4],y=[3,3,1,2],z=['3','3','2','1'])
    t.sort('x','y','z')
    assert list(t['x']) == [4,3,1,2]
    assert list(t['y']) == [2,1,3,3]
    assert list(t['z']) == ['1','2','3','3']

def test_dataset_plotting():
    t = Dataset()
    t['x'] = [1,2,3]
    t['y'] = [1,2,3]
    t['δy'] = [0.1,0.2,0.3]
    t['z'] = [2,4,5]
    t.plot('x',('y','z'),show=show_plots)

def test_dataset_load_save_to_file():
    ## npz archive
    t = Dataset()
    t['x'] = [1,2,3]
    t['f'] = [1.29,1.29,3.342]
    t.save('tmp/t0.npz')
    u = Dataset()
    u.load('tmp/t0.npz')
    assert set(u.keys()) == {'x','f'}
    assert list(u['x']) == list(t['x'])
    assert list(u['f']) == list(t['f'])
    ## hdf5 archive
    t = Dataset()
    t['x'] = [1,2,3]
    t['f'] = [1.29,1.29,3.342]
    t['z'] = ['a','b','c']
    t.save('tmp/t0.h5')
    u = Dataset()
    u.load('tmp/t0.h5')
    assert set(u.keys()) == {'x','f','z'}
    assert all(u['x'] == t['x'])
    assert np.all(u['z'] == t['z'])
    assert all(u['f'] == t['f'])
    ## text file
    t = Dataset()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = [1.29,1.29,3.342]
    t.save('tmp/t0.psv')
    u = Dataset()
    u.load('tmp/t0.psv')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert all(u['f'] == t['f'])
    ## ␞-separated text file
    t = Dataset()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = [1.29,1.29,3.342]
    t.save('tmp/t0.rs')
    u = Dataset()
    u.load('tmp/t0.rs')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert all(u['f'] == t['f'])
    t.save('tmp/t0.psv',delimiter='|')
    u = Dataset()
    u.load('tmp/t0.psv',delimiter='|')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert all(u['f'] == t['f'])

def test_uncertainty():
    t = Dataset(x=[1,2,3],f=5)
    t.set_uncertainty('x',0.1)
    assert all(t.get_uncertainty('x') == [0.1,0.1,0.1])
    t = Dataset(x=[1,2,3],f=5)
    t['d_x'] = 0.1
    assert all(t['d_x'] == [0.1,0.1,0.1])
    t = Dataset(x=[1,2,3],f=5)
    t.set_uncertainty('x',0.1)
    t.set_uncertainty('x',0.2,[0,2])
    assert all(t.get_uncertainty('x') == [0.2,0.1,0.2 ])

def test_differentiation_step():
    t = Dataset(x=[1,2,3],f=5)
    t.set_differentiation_step('x',0.1)
    assert all(t.get_differentiation_step('x') == [0.1,0.1,0.1])
    t = Dataset(x=[1,2,3],f=5)
    t['s_x'] = 0.1
    assert all(t['s_x'] == [0.1,0.1,0.1])

def test_vary():
    t = Dataset(x=[1,2,3],f=5)
    t.set_vary('x',[True,True,False])
    assert all(t.get_vary('x') == [True,True,False])
    t = Dataset(x=[1,2,3],f=5)
    t['v_x'] = False 
    assert all(t['v_x'] == [False,False,False,])
    
