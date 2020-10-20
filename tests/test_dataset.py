import pytest
from pytest import raises
import numpy as np

from spectr.dataset import *
from spectr.exceptions import InferException


def test_datum_construct():
    t = Datum(value=1)
    assert t._value == 1
    assert t.value == 1
    assert t.kind == 'i'
    assert not t.has_uncertainty()
    t = Datum(value='dd')
    assert t.kind == 'U'
    assert t.value == 'dd'
    t = Datum(value=1.5,uncertainty=0.3)
    assert t.has_uncertainty()
    assert t.kind == 'f'
    assert t.value == 1.5
    assert t.uncertainty == 0.3
    t = Datum(value=1,uncertainty=0.1)
    assert t.kind == 'f'
    assert t.has_uncertainty()
    t = Datum(value=-51,kind=float,cast=lambda x:float(abs(x)))
    assert t.value == 51
    t = Datum(value=[1,2,3])
    assert t.value == [1,2,3]
    assert t.kind == 'O'
    assert not t.has_uncertainty()
    with pytest.raises(ValueError):
        t = Datum(value='a',kind=float)
    with pytest.raises(AssertionError):
        t = Datum(value='a',uncertainty=1.)

def test_datum_has_uncertainty():
    t = Datum(value=1,uncertainty=1)
    assert t.has_uncertainty()
    t = Datum(value=1)
    assert not t.has_uncertainty()

def test_datum_str():
    t = Datum(value=1,uncertainty=0.1)
    assert str(t)=='+1.00000000e+00 ± 0.1'
    t = Datum(value=1,uncertainty=0.1,fmt='0.5f')
    assert str(t)=='1.00000 ± 0.1'
    t = Datum(value='a')
    assert str(t)=='a'
    t = Datum(value=1)
    assert str(t)=='1'

def test_datum_math_builtins():
    t = Datum(value=5)
    assert -t == -5
    assert float(t) == 5.0
    assert +t == 5
    assert abs(t) == 5
    assert t + 2 == 7
    assert 2 + t == 7
    assert t - 2 == 3
    assert 2 - t == -3
    assert t / 2 == 2.5
    assert 10 / t == 2
    assert t * 2 == 10
    assert 10 * t == 50
    assert t**2 == 25
    assert 2**t == 32
    
def test_datum_timestamp():
    t0 = time.time()
    d = Datum('x')
    t1 = d.timestamp
    assert t1>t0
    d.value = 'j'
    t2 = d.timestamp
    assert t2>t1




def test_data_construct():
    t = Data(value=[1,2])
    assert list(t.value) == [1,2]
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    assert list(t.value) == [1.5,2.5]
    assert list(t.uncertainty) == [0.3,0.2]
    assert len(t) == 2
    t = Data(value=[1.5,2.5],uncertainty=1)
    assert list(t.value) == [1.5,2.5]
    assert list(t.uncertainty) == [1,1]
    t = Data(value=[1,2],uncertainty=1)
    assert list(t.value) == [1,2]
    assert list(t.uncertainty) == [1,1]
    with pytest.raises(ValueError):
        t = Data(value=[1,2],uncertainty=[1,2,3])

def test_data_has_uncertainty():
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    assert t.has_uncertainty()
    t = Data(value=[1.5,2.5])
    assert not t.has_uncertainty()

def test_data_index():
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    t.index([True,False])
    assert list(t.value) == [1.5]
    assert list(t.uncertainty) == [0.3]
    assert len(t) == 1

def test_data_append():
    t = Data(value=[2,34])
    t.append(5)
    assert list(t.value) == [2,34,5]
    t = Data(value=[2.,34.],uncertainty=[0.2,0.3])
    t.append(5,0.5)
    assert list(t.value) == [2.,34.,5.]
    assert list(t.uncertainty) == [0.2,0.3,0.5]

def test_data_extend():
    t = Data(value=['a','b'])
    t.extend(['c','d'])
    assert list(t) == ['a','b','c','d']
    t = Data(value=[1.,2.],uncertainty=[0.1,0.1])
    t.extend([3,4],[1,2])
    assert list(t.value) == [1.,2.,3.,4.,]
    assert list(t.uncertainty) == [0.1,0.1,1.,2.]

def test_data_object():
    t = Data(value=[[1,2],None],kind=object)
    assert list(t.value) == [[1,2],None]
    t = Data(value=[{},np.array([1,2,3]),None],kind=object)
    assert t.value[0] == {}

def test_data_str():
    t = Data(value=[1,2,3])
    assert str(t)=='1\n2\n3'
    t = Data(value=[1,2,3],uncertainty=[0.1,5,6])
    assert str(t)=='+1.00000000e+00 ± 0.1\n+2.00000000e+00 ± 5\n+3.00000000e+00 ± 6'
    t = Data(value=['a','b','c'])
    assert str(t)=='a\nb\nc'



def test_dataset_construct():
    t = Dataset()

def test_dataset_get_key_without_uncertainty():
    t = Dataset()
    assert t._get_key_without_uncertainty('d_ddd') == 'ddd'
    assert t._get_key_without_uncertainty('dddd') is None
    assert t._get_key_without_uncertainty('d_') is None
    t = Dataset()
    t.uncertainty_prefix = 'σ'
    assert t._get_key_without_uncertainty('d_ddd') == None
    assert t._get_key_without_uncertainty('σf') == 'f'

def test_dataset_construct_get_set_data():
    t = Dataset()
    assert t.is_scalar()
    t = Dataset(x=5)
    assert t.is_scalar()
    assert t['x'] == 5
    t = Dataset(x=['a','b'])
    assert not t.is_scalar()
    assert list(t['x']) == ['a','b']
    t = Dataset(x=['a','b'],y=5)
    assert not t.is_scalar()
    assert t['y'] == 5
    assert list(t['x']) == ['a','b']
    t = Dataset(
        x=[1.,2.],
        d_x=[0.1,0.2],
    )
    assert not t.is_scalar()
    assert len(t)==2
    assert list(t['x']) == [1.,2.]
    assert list(t['d_x']) == [0.1,0.2]

def test_dataset_set_get_value():
    t = Dataset()
    t.set('x',5.,0.1)
    assert t.get_value('x') == 5.
    assert t.get_uncertainty('x') == 0.1
    assert t['x'] == 5.
    t = Dataset()
    t['y'] = ['a','b']
    assert list(t['y']) == ['a','b']
    t = Dataset(y=['a','b'])
    assert list(t['y']) == ['a','b']
    t = Dataset()
    t.set('x',5,kind=float)
    assert isinstance(t['x'],float)

def test_dataset_setitem_getitem():
    t = Dataset()
    t['x'] = 5
    assert t['x']==5
    t.set('y',5.)
    t.set_uncertainty('y',0.1)
    assert t['y']==5
    assert t['d_y']==0.1

def test_dataset_prototypes():
    t = Dataset()
    t.add_prototype('x',description="X is a thing.",kind=float)
    t['x'] = 5
    assert isinstance(t['x'],float)
    assert t._data['x'].description == "X is a thing."

def test_dataset_permit_nonprototyped_data():
    t = Dataset()
    t.permit_nonprototyped_data = True
    t['x'] = 5
    t.permit_nonprototyped_data = False
    with pytest.raises(AssertionError):
        t['y'] = 5

def test_dataset_len_is_scalar():
    t = Dataset()
    t['y'] = ['a','b']
    assert len(t) == 2
    t = Dataset(y='a')
    assert t.is_scalar()
    t = Dataset(y='a',x=[1,2])
    assert not t.is_scalar()
    assert not t.is_scalar('x')
    assert t.is_scalar('y')

def test_dataset_index():
    t = Dataset(x=[1,2,3,4,5])
    t.index([0,1])
    assert list(t['x']) == [1,2]
    t = Dataset(x=[1,2,3,4,5])
    t.index([True,True,True,False,False,])
    assert list(t['x']) == [1,2,3]

def test_dataset_make_vector():
    t = Dataset(x=1,y=[1,2,3])
    assert(t['x']==1)
    t.make_vector('x')
    assert(len(t['x'])==3)

def test_dataset_make_scalar():
    t = Dataset(x=1,y=[1,1,1])
    assert(len(t['y'])==3)
    t.make_scalar('y')
    assert np.isscalar(t['y'])
    
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

def test_dataset_concatenate():
    t = Dataset()
    t.concatenate(Dataset())
    assert t.is_scalar()
    t = Dataset(x=1)
    t.concatenate(Dataset(x=1))
    assert t.is_scalar()
    assert t['x'] == 1
    t = Dataset(x=1)
    t.concatenate(Dataset(x=2))
    assert t.is_scalar('x')
    assert t['x'] == 1
    t = Dataset(x=[1,2])
    t.concatenate(Dataset(x=[3]))
    assert not t.is_scalar()
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = Dataset(x='a',y=[1,2],z=0.5)
    t.concatenate(Dataset(x='a',y=[3],z=0.6))
    assert t['x'] == 'a'
    assert list(t['y']) == [1,2,3]
    assert not t.is_scalar()
    assert all(np.abs(np.array(t['z'])-np.array([0.5,0.5,0.6]))<1e-5)
    assert len(t) == 3
    t = Dataset(x=[1])
    t.concatenate(Dataset(x=[2,3]))
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = Dataset(x=[1])
    t.concatenate(Dataset(x=[2,3]))
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = Dataset(x=[1],y='a')
    u = Dataset(x=[2,3],y=['a','b'])
    t.concatenate(u)
    assert list(t['x']) == [1,2,3]
    assert list(t['y']) == ['a','a','b']
    assert len(t) == 3
    t = Dataset()
    t.concatenate(Dataset(x=5,y=[1,2]))
    assert len(t) == 2
    assert t['x'] == 5

def test_dataset_extend():
    t = Dataset()
    t.extend()
    assert t.is_scalar()
    t = Dataset(x=[1])
    t.extend(x=[2])
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
    t = Dataset(x=[1,2],y='a')
    t.extend(x=[2,3],y=['a','b'])
    assert list(t['x']) == [1,2,2,3]
    assert list(t['y']) == ['a','a','a','b']
    assert len(t) == 4

def test_dataset_append():
    t = Dataset(x=[0])
    t.append(x=1)
    assert list(t['x']) == [0,1]
    assert len(t)==2
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
    t = Dataset(x=[0],y='a')
    t.append(x=1,y='b')
    assert list(t['x']) == [0,1]
    assert list(t['y']) == ['a','b']
    assert len(t)==2

def test_dataset_infer():
    t = Dataset(x=1,y=2)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    t.add_infer_function('w',('y','z'),lambda y,z:y*z)
    assert t['w'] == 6
    t = Dataset(x=[1,2,3],y=2)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    with pytest.raises(InferException):
        t['w']

def test_dataset_infer_autoremove_inferences():
    t = Dataset()
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

def test_dataset_infer_with_uncertainties():
    t = Dataset()
    t['x'],t['d_x']= 1.,0.1
    t['y'],t['d_y'] = 2.,0.2
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    assert abs(t['d_z'] - np.sqrt(0.1**2+0.2**2))/t['d_z'] < 1e-5
    t = Dataset()
    t['x'],t['d_x']= 1.,0.1
    t['y'] = 2.
    t.add_infer_function('z',('x','y'),lambda x,y:x+y)
    assert t['z'] == 3
    assert abs(t['d_z'] - np.sqrt(0.1**2))/t['d_z'] < 1e-5
    t = Dataset()
    t['y'],t['d_y'] = 2.,0.2
    t['p'],t['d_p'] = 3.,0.3
    t.add_infer_function('z',('y','p'),lambda y,p:y*p)
    assert t['z'] == 2*3
    assert abs(t['d_z'] - np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3 )/t['d_z'] < 1e-5
    t = Dataset()
    t['x'],t['d_x']= 1.,0.1
    t['y'],t['d_y'] = 2.,0.2
    t['p'],t['d_p'] = 3.,0.3
    t.add_infer_function('z',('x','y','p'),lambda x,y,p:x+y*p)
    assert t['z'] == 1+2*3
    assert abs(t['d_z'] - np.sqrt(0.1**2 + (np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)**2) )/t['d_z'] < 1e-5

def test_dataset_match_matches():
    t = Dataset(x=[1,2,2,3],y=4)
    assert list(t.match(x=2)) == [False ,True,True ,False]
    assert list(t.match(x=2,y=4))== [False ,True,True ,False]
    assert list(t.match(x=2,y=5))== [False ,False,False,False]
    assert list(t.match(x=[2,3],y=4))== [False ,True,True , True]
    t = Dataset(x=[1,2,2,3],y=4)
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
    t.plot('x',('y','z'),show=False)

def test_dataset_load_save_to_file():
    ## npz archive
    t = Dataset()
    t['x'] = [1,2,3]
    t['f'] = 1.29
    t.save('tmp/t0.npz')
    u = Dataset()
    u.load('tmp/t0.npz')
    assert set(u.keys()) == {'x','f'}
    assert list(u['x']) == list(t['x'])
    assert u['f'] == t['f']
    ## hdf5 archive
    t = Dataset()
    t['x'] = [1,2,3]
    t['f'] = 1.29
    t['z'] = ['a','b','c']
    t.save('tmp/t0.h5')
    u = Dataset()
    u.load('tmp/t0.h5')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert np.all(u['z'] == t['z'])
    assert u['f'] == t['f']
    ## text file
    t = Dataset()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = 1.29
    t.save('tmp/t0.txt',comment='# ')
    u = Dataset()
    u.load('tmp/t0.txt')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']
    ## ␞-separated text file
    t = Dataset()
    t['x'] = [1,2,3]
    t['z'] = ['a','b','c']
    t['f'] = 1.29
    t.save('tmp/t0.rs',delimiter='␞',comment='# ')
    u = Dataset()
    u.load('tmp/t0.rs',delimiter='␞')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']
    t.save('tmp/t0.psv',delimiter='|',comment='# ')
    u = Dataset()
    u.load('tmp/t0.psv',delimiter='|')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']


