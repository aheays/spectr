import pytest
from pytest import raises
import numpy as np

from spectr.dataset import *
from spectr.exceptions import InferException

def test_construct():
    t = DataSet()

def test_get_key_without_uncertainty():
    t = DataSet()
    assert t._get_key_without_uncertainty('d_ddd') == 'ddd'
    assert t._get_key_without_uncertainty('dddd') is None
    assert t._get_key_without_uncertainty('d_') is None
    t = DataSet()
    t.uncertainty_prefix = 'σ'
    assert t._get_key_without_uncertainty('d_ddd') == None
    assert t._get_key_without_uncertainty('σf') == 'f'

def test_construct_get_set_data():
    t = DataSet()
    assert t.is_scalar()
    t = DataSet(x=5)
    assert t.is_scalar()
    assert t['x'] == 5
    t = DataSet(x=['a','b'])
    assert not t.is_scalar()
    assert list(t['x']) == ['a','b']
    t = DataSet(x=['a','b'],y=5)
    assert not t.is_scalar()
    assert t['y'] == 5
    assert list(t['x']) == ['a','b']
    t = DataSet(
        x=[1.,2.],
        d_x=[0.1,0.2],
    )
    assert not t.is_scalar()
    assert len(t)==2
    assert list(t['x']) == [1.,2.]
    assert list(t['d_x']) == [0.1,0.2]

def test_set_get_value():
    t = DataSet()
    t.set('x',5.,0.1)
    assert t.get_value('x') == 5.
    assert t.get_uncertainty('x') == 0.1
    assert t['x'] == 5.
    t = DataSet()
    t['y'] = ['a','b']
    assert list(t['y']) == ['a','b']
    t = DataSet(y=['a','b'])
    assert list(t['y']) == ['a','b']
    t = DataSet()
    t.set('x',5,kind=float)
    assert isinstance(t['x'],float)

def test_setitem_getitem():
    t = DataSet()
    t['x'] = 5
    assert t['x']==5
    t.set('y',5.)
    t.set_uncertainty('y',0.1)
    assert t['y']==5
    assert t['d_y']==0.1

def test_prototypes():
    t = DataSet()
    t.add_prototype('x',description="X is a thing.",kind=float)
    t['x'] = 5
    assert isinstance(t['x'],float)
    assert t._data['x'].description == "X is a thing."

def test_permit_nonprototyped_data():
    t = DataSet()
    t.permit_nonprototyped_data = True
    t['x'] = 5
    t.permit_nonprototyped_data = False
    with pytest.raises(AssertionError):
        t['y'] = 5

def test_len_is_scalar():
    t = DataSet()
    t['y'] = ['a','b']
    assert len(t) == 2
    t = DataSet(y='a')
    assert t.is_scalar()
    t = DataSet(y='a',x=[1,2])
    assert not t.is_scalar()
    assert not t.is_scalar('x')
    assert t.is_scalar('y')

def test_index():
    t = DataSet(x=[1,2,3,4,5])
    t.index([0,1])
    assert list(t['x']) == [1,2]
    t = DataSet(x=[1,2,3,4,5])
    t.index([True,True,True,False,False,])
    assert list(t['x']) == [1,2,3]

def test_make_array():
    t = DataSet(x=1,y=[1,2,3])
    assert(t['x']==1)
    t.make_array('x')
    assert(len(t['x'])==3)

def test_make_scalar():
    t = DataSet(x=1,y=[1,1,1])
    assert(len(t['y'])==3)
    t.make_scalar('y')
    assert np.isscalar(t['y'])
    
def test_get_copy():
    t = DataSet(x=[1,2,3],y=['a','b','c'])
    u = t.copy()
    assert list(u['x']) == [1,2,3]
    t = DataSet(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',))
    assert 'y' not in u 
    t = DataSet(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',),[False,False,True])
    assert list(u['x']) == [3]

def test_concatenate():
    t = DataSet()
    t.concatenate(DataSet())
    assert t.is_scalar()
    t = DataSet(x=1)
    t.concatenate(DataSet(x=1))
    assert t.is_scalar()
    assert t['x'] == 1
    t = DataSet(x=1)
    t.concatenate(DataSet(x=2))
    assert t.is_scalar('x')
    assert t['x'] == 1
    t = DataSet(x=[1,2])
    t.concatenate(DataSet(x=[3]))
    assert not t.is_scalar()
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = DataSet(x='a',y=[1,2],z=0.5)
    t.concatenate(DataSet(x='a',y=[3],z=0.6))
    assert t['x'] == 'a'
    assert list(t['y']) == [1,2,3]
    assert not t.is_scalar()
    assert all(np.abs(np.array(t['z'])-np.array([0.5,0.5,0.6]))<1e-5)
    assert len(t) == 3
    t = DataSet(x=[1])
    t.concatenate(DataSet(x=[2,3]))
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = DataSet(x=[1])
    t.concatenate(DataSet(x=[2,3]))
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = DataSet(x=[1],y='a')
    u = DataSet(x=[2,3],y=['a','b'])
    t.concatenate(u)
    assert list(t['x']) == [1,2,3]
    assert list(t['y']) == ['a','a','b']
    assert len(t) == 3
    t = DataSet()
    t.concatenate(DataSet(x=5,y=[1,2]))
    assert len(t) == 2
    assert t['x'] == 5

def test_extend():
    t = DataSet()
    t.extend()
    assert t.is_scalar()
    t = DataSet(x=1)
    t.extend(x=2)
    assert list(t['x']) == [1,2]
    t = DataSet(x=[1,2])
    t.extend(x=3)
    assert len(t) == 3
    assert list(t['x']) == [1,2,3]
    t = DataSet(x=3)
    t.extend(x=[1,2])
    assert len(t) == 3
    assert list(t['x']) == [3,1,2]
    t = DataSet(x=[1,2])
    t.extend(x=[2,3])
    assert list(t['x']) == [1,2,2,3]
    assert len(t) == 4
    t = DataSet(x=[1,2],y='a')
    t.extend(x=[2,3],y=['a','b'])
    assert list(t['x']) == [1,2,2,3]
    assert list(t['y']) == ['a','a','a','b']
    assert len(t) == 4

def test_append():
    t = DataSet(x=0)
    t.append(x=1)
    assert list(t['x']) == [0,1]
    assert len(t)==2
    t = DataSet(x=[0])
    t.append(x=1)
    assert list(t['x']) == [0,1]
    assert len(t)==2
    t = DataSet(x=[0,1,2])
    t.append(x=1)
    assert list(t['x']) == [0,1,2,1]
    assert len(t)==4
    t = DataSet(x=[0],y=['a'])
    t.append(x=1,y='b')
    assert list(t['x']) == [0,1]
    assert list(t['y']) == ['a','b']
    assert len(t)==2
    t = DataSet(x=[0],y='a')
    t.append(x=1,y='b')
    assert list(t['x']) == [0,1]
    assert list(t['y']) == ['a','b']
    assert len(t)==2

def test_infer():
    t = DataSet(x=1,y=2)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    t.add_infer_function('w',('y','z'),lambda y,z:y*z)
    assert t['w'] == 6
    t = DataSet(x=[1,2,3],y=2)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    with pytest.raises(InferException):
        t['w']

def test_infer_autoremove_inferences():
    t = DataSet()
    t['x'] = 1.
    t['y'] = 2.
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    t['z']
    assert 'z' in t
    t['x'] = 2
    assert 'z' not in t
    t['z']
    t['z'] = 5
    t['x'] = 2
    assert 'z' in t

def test_infer_with_uncertainties():
    t = DataSet()
    t['x'],t['d_x']= 1.,0.1
    t['y'],t['d_y'] = 2.,0.2
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    assert abs(t['d_z'] - np.sqrt(0.1**2+0.2**2))/t['d_z'] < 1e-5
    t = DataSet()
    t['x'],t['d_x']= 1.,0.1
    t['y'] = 2.
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    assert abs(t['d_z'] - np.sqrt(0.1**2))/t['d_z'] < 1e-5
    t = DataSet()
    t['y'],t['d_y'] = 2.,0.2
    t['p'],t['d_p'] = 3.,0.3
    t.add_infer_function('z',('y','p'),lambda y,p:y*p)
    assert t['z'] == 2*3
    assert abs(t['d_z'] - np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3 )/t['d_z'] < 1e-5
    t = DataSet()
    t['x'],t['d_x']= 1.,0.1
    t['y'],t['d_y'] = 2.,0.2
    t['p'],t['d_p'] = 3.,0.3
    t.add_infer_function('z',('x','y','p'),lambda x,y,p:x+y*p)
    assert t['z'] == 1+2*3
    assert abs(t['d_z'] - np.sqrt(0.1**2 + (np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)**2) )/t['d_z'] < 1e-5

def test_match_matches():
    t = DataSet(x=[1,2,2,3],y=4)
    assert list(t.match(x=2)) == [False ,True,True ,False]
    assert list(t.match(x=2,y=4))== [False ,True,True ,False]
    assert list(t.match(x=2,y=5))== [False ,False,False,False]
    assert list(t.match(x=[2,3],y=4))== [False ,True,True , True]
    t = DataSet(x=[1,2,2,3],y=4)
    u = t.matches(x=2)
    assert list(u['x']) == [2,2]

def test_unique_functions():
    t = DataSet(x=[1,2,2,2],y=['a','b','b','c'])
    assert list(t.unique('x')) == [1,2]
    assert set(t.unique_combinations('x','y')) == set([(1,'a'),(2,'b'),(2,'c')])
    u = t.unique_dicts('x','y')
    assert str(u) == "[{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}, {'x': 2, 'y': 'c'}]"
    u = t.unique_dicts_match('x','y')
    assert str(u) == "[({'x': 1, 'y': 'a'}, array([ True, False, False, False])), ({'x': 2, 'y': 'b'}, array([False,  True,  True, False])), ({'x': 2, 'y': 'c'}, array([False, False, False,  True]))]"
    u = t.unique_dicts_matches('x','y')

def test_sort():
    t = DataSet(x=[1,3,2])
    t.sort('x')
    assert list(t['x']) == [1,2,3]
    t = DataSet(x=[2,1,3,4],y=[3,3,1,2],z=['3','3','2','1'])
    t.sort('x','y','z')
    assert list(t['x']) == [4,3,1,2]
    assert list(t['y']) == [2,1,3,3]
    assert list(t['z']) == ['1','2','3','3']

def test_plotting():
    t = DataSet()
    t['x'] = [1,2,3]
    t['y'] = [1,2,3]
    t['δy'] = [0.1,0.2,0.3]
    t['z'] = [2,4,5]
    t.plot('x',('y','z'),show=False)

def test_load_save_to_file():
    ## npz archive
    t = DataSet()
    t['x'] = [1,2,3]
    t['f'] = 1.29
    t.save('tmp/t0.npz')
    u = DataSet()
    u.load('tmp/t0.npz')
    assert set(u.keys()) == {'x','f'}
    assert list(u['x']) == list(t['x'])
    assert u['f'] == t['f']
    ## hdf5 archive
    t = DataSet()
    t['x'] = [1,2,3]
    t['f'] = 1.29
    t['z'] = ['a','b','c']
    t.save('tmp/t0.h5')
    u = DataSet()
    u.load('tmp/t0.h5')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert np.all(u['z'] == t['z'])
    assert u['f'] == t['f']
    ## text file
    t = DataSet()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = 1.29
    t.save('tmp/t0.txt',comment='# ')
    u = DataSet()
    u.load('tmp/t0.txt')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']
    ## ␞-separated text file
    t = DataSet()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = 1.29
    t.save('tmp/t0.rs',delimiter='␞',comment='# ')
    u = DataSet()
    u.load('tmp/t0.rs',delimiter='␞')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']
    t.save('tmp/t0.psv',delimiter='|',comment='# ')
    u = DataSet()
    u.load('tmp/t0.psv',delimiter='|')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']


