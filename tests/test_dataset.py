import pytest
from pytest import raises,approx
import numpy as np

from spectr import dataset
from spectr.dataset import *
from spectr.optimise import P
from spectr.exceptions import InferException

show_plots = False

def test_dataset_construct():
    t = Dataset()

def test_dataset_construct_get_set_data():
    t = Dataset()
    assert len(t) == 0
    t = Dataset()
    t.set('x',[1,2,3])
    assert len(t) == 3
    assert all(t.get('x')==[1,2,3])
    t.set('x',25,index=1)
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
    t = Dataset(x=[1.,2.], unc_x=[0.1,0.2],)
    assert len(t)==2
    assert list(t['x']) == [1.,2.]
    assert list(t['unc_x']) == [0.1,0.2]

def test_uncertainties():
    t = Dataset()
    t.set('x',[1.,2.,3])
    t.set('x',0.1,'unc')
    assert all(t.get('x','unc') == [0.1,0.1,0.1])
    t = Dataset(x=[1.,2,3],f=5)
    t.set('x',0.1,'unc')
    assert all(t.get('x','unc') == [0.1,0.1,0.1])
    t = Dataset(x=[1.1,2,3],f=5)
    t['x_unc'] = 0.1
    assert all(t['x_unc'] == [0.1,0.1,0.1])
    t = Dataset(x=[1.,2,3],f=5)
    t.set('x',0.1,'unc')
    t.set('x',0.2,'unc',[0,2])
    assert all(t.get('x','unc') == [0.2,0.1,0.2 ])

def test_get_set_uncertainties_as_key():
    t = Dataset()
    t['x'] = [1.,2.,3]
    t['x_unc'] = [0.1,0.2,0.3]
    assert all(t.get('x','unc') == [0.1,0.2,0.3])
    assert all(t['x_unc'] == [0.1,0.2,0.3])
    t = Dataset(x=[1.,2.,3],x_unc=[0.1,0.2,0.3])
    assert all(t.get('x','unc') == [0.1,0.2,0.3])

def test_dataset_prototypes():
    t = Dataset()
    assert t.get_prototype('x') is None
    t.set_prototype('x',kind='f',description="X is a thing.",)
    assert t.get_prototype('x') is not None
    t['x'] = [5]
    assert isinstance(t['x'][0],float)
    assert t._data['x']['description'] == "X is a thing."
    assert t.get_prototype('x')['description'] == "X is a thing."

def test_dataset_default_prototypes():
    ## without default prototype
    class t(Dataset):
        default_prototypes = {}
    x = t(permit_nonprototyped_data=False)
    with raises(Exception):
        x['x'] = [1,2,3]
    ## with default prototype
    class t(Dataset):
        default_prototypes = {'x':{'kind':'i',},}
    x = t(permit_nonprototyped_data=False)
    # assert 'x' in x.prototypes
    x['x'] = [1,2,3]
    assert np.all(x['x'] == [1,2,3])

def test_dataset_permit_nonprototyped_data():
    t = Dataset()
    t.permit_nonprototyped_data = True
    t['x'] = [5]
    t.permit_nonprototyped_data = False
    with raises(Exception):
        qt['y'] = [5]

def test_defaults():
    ## set and use a default
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    assert t.get_default('x') is None
    with raises(Exception):
        t.append(y='unc')
    t.set_default('x',5)
    assert t.get_default('x') is not None
    assert t.get_default('x') == 5
    t.append(y='unc')
    assert all(t['x'] == [1,2,3,5])
    t.extend(y=['e','f'])
    assert all(t['x'] == [1,2,3,5,5,5])
    ## set default at instantiation
    t = Dataset(y='a')
    assert t.get_default('y') is not None
    assert t.get_default('y') == 'a'
    t.extend(x=[1,2])
    assert t.get_default('y') == 'a'
    t = Dataset(x=[1,2],y='a')
    assert t.get_default('y') == 'a'
    assert len(t) == 2
    assert all(t['x'] == [1,2])
    assert all(t['y'] == ['a','a'])

def test_auto_defaults():
    t = Dataset(x=[1,2,3])
    t.set_prototype('x','f')
    t.set_prototype('y','f')
    assert np.isnan(t.get_prototype('x')['default'])
    t.auto_defaults = False
    t.extend(x=[1,3,4])
    with raises(Exception):
        t.append(x=5,y=2)
    t = Dataset(x=[1,2,3])
    t.set_prototype('x','f')
    t.set_prototype('y','f')
    t.auto_defaults = True
    t.extend(x=[1,3,4])
    t.append(x=5,y=2)
    assert all(t['x'] == [1,2,3,1,3,4,5])
    assert t['y'][-1] == 2
    assert all([np.isnan(tt) for tt in t['y'][:-1]])

def test_dataset_index():
    t = Dataset(x=[1,2,3,4,5])
    t.index([0,1])
    assert list(t['x']) == [1,2]
    t = Dataset(x=[1,2,3,4,5])
    t.index([True,True,True,False,False,])
    assert list(t['x']) == [1,2,3]
    t = Dataset()
    t['x']=[1,2,3,4,5.]
    t['x_unc']=[1,2,3,4,5]
    t['x_step']=[1,2,3,4,5]
    t['x_vary']=[False,True,True,True,True,]
    t.index([True,True,True,False,False,])
    assert list(t['x']) == [1,2,3]
    assert list(t['x_unc']) == [1,2,3]
    assert list(t['x_step']) == [1,2,3]
    assert list(t['x_vary']) == [False,True,True,]

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
    t.set_default('y','a')
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
    t.set_default('y','a')
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
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    t.set_prototype('w','f')
    t.add_infer_function('w',('y','z'),lambda self,y,z:y*z)
    assert t['w'] == [6]
    t = Dataset(x=[1,2,3],y=[2,2,2])
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    with raises(InferException):
        t['w']

def test_dataset_infer_autoremove_inferences():
    t = Dataset()
    t['x'] = [1.]
    t['y'] = [2.]
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    t['z']
    pprint( t._data['x'])
    pprint( t._data['y'])
    pprint( t._data['z'])
    assert 'z' in t
    t['x'] = [2]
    assert 'z' not in t
    t['z']
    t['z'] = [5]
    t['x'] = [2]
    assert 'z' in t

def test_dataset_infer_with_uncertainties():
    t = Dataset()
    t['x'] = [1.]
    t['y'] = [2.]
    t.set('x',[0.1],'unc')
    t.set('y',[0.2],'unc')
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    assert t.get('z','unc') == approx(np.sqrt(0.1**2+0.2**2))
    t = Dataset()
    t['x'] = [1.]
    t.set('x',[0.1],'unc')
    t['y'] = [2.]
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y'),lambda self,x,y:x+y)
    assert t['z'] == [3]
    dz = t.get('z','unc')
    assert abs(dz - np.sqrt(0.1**2))/dz < 1e-5
    t = Dataset()
    t['y'] = [2.]
    t['p'] = [3.]
    t.set('y',[0.2],'unc')
    t.set('p',[0.3],'unc')
    t.set_prototype('z','f')
    t.add_infer_function('z',('y','p'),lambda self,y,p:y*p)
    assert t['z'] == [2*3]
    assert t.get('z','unc') == approx(np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)
    t = Dataset()
    t['x'] = [1.]
    t['y'] = [2.]
    t['p'] = [3.]
    t.set('x',[0.1],'unc')
    t.set('y',[0.2],'unc')
    t.set('p',[0.3],'unc')
    t.set_prototype('z','f')
    t.add_infer_function('z',('x','y','p'),lambda self,x,y,p:x+y*p)
    assert t['z'] == 1+2*3
    dz = t.get('z','unc')
    assert abs(dz - np.sqrt(0.1**2 + (np.sqrt((0.2/2)**2+(0.3/3)**2)*2*3)**2) )/dz < 1e-5

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
    tools.mkdir('tmp')
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
test_dataset_load_save_to_file()
def test_load_from_string():
    t = Dataset()
    t.load_from_string('''
    x = 5
    y = 'ddd'
    a|  b   |  c | d
    1|  2   |  c | 1
    3|  -5.0|  d | 1
    1|  2   |  e | xxx
    ''')
    assert all(t['a'] == [1,3,1])
    assert all(t['c'] == ['c','d','e'])
    assert all(t['d'] == ['1','1','xxx'])
    assert all(t['x'] == [5,5,5])
    assert all(t['y'] == ['ddd','ddd','ddd'])
    t = Dataset(load_from_string='''
    x = 5
    y = 'ddd'
    a|  b   |  c | d
    1|  2   |  c | 1
    3|  -5.0|  d | 1
    1|  2   |  e | xxx
    ''')
    assert all(t['a'] == [1,3,1])

def test_step():
    t = Dataset(x=[1,2,3.0],f=5)
    t.set('x',0.1,'step')
    assert all(t.get('x','step') == [0.1,0.1,0.1])
    t = Dataset(x=[1.0,2,3],f=5)
    t['x_step'] = 0.1
    assert all(t.get('x','step') == [0.1,0.1,0.1])

def test_vary():
    t = Dataset(x=[1,2,3.0],f=5)
    t.set('x',[True,True,False],'vary')
    assert all(t.get('x','vary') == [True,True,False])
    t = Dataset(x=[1,2,3.0],f=5)
    t['x_vary'] = False
    assert all(t.get('x','vary') == [False,False,False,])
    assert all(t['x_vary'] == [False,False,False,])

def test_rows():
    x = Dataset(x=[1,2,3],y=[4,5,6])
    for i,d in enumerate(x.rows()):
        assert d['x'] == x['x'][i]
        assert d['y'] == x['y'][i]

def test_row_data():
    x = Dataset(x=[1,2,3],y=[4,5,6])
    for i,d in enumerate(x.row_data()):
        assert d[0] == x['x'][i]
        assert d[1] == x['y'][i]

def test_find_common():
    x = Dataset(x=[1,2,3],y=[1,2,3])
    y = Dataset(x=[3,1,2],y=[4,5,6])
    i,j = find_common(x,y,'x')
    assert all(i == [0,1,2])
    assert all(j == [1,2,0])
    x = Dataset(x=[1,2,3],z=['a','b','c'],y=[1,2,3],)
    y = Dataset(x=[3,1,2],z=['c','a','unc'],y=[4,5,6],)
    i,j = find_common(x,y,'x','z')
    assert all(i == [0,2])
    assert all(j == [1,0])

def test_class_and_description_attributes():
    x = Dataset(description='abc', x=[1,2,3],z=['a','b','c'],y=[3,3,3],)
    assert x.attributes['description'] == 'abc'
    assert x['description'] == 'abc'
    print(x.attributes['classname'])
    assert x.attributes['classname'] == 'dataset.Dataset'
    x = Dataset()
    x.description = 'abc'
    assert x.description == 'abc'
    x = Dataset()
    x['description'] = 'abc'
    assert x['description'] == 'abc'

def test_get_common():
    x = Dataset(x=[1,2,3],z=['a','b','c'],y=[1,2,3],)
    y = Dataset(x=[3,1,2],z=['c','a','d'],y=[4,5,6],)
    x,y = get_common(x,y,'x','z')
    assert all(x['y'] == [1,3])
    assert all(y['y'] == [5,4])

def test_load():
    x = dataset.load('data/test_load.psv')
    assert x['classname'] == 'dataset.Dataset'

def test_units():
    x = Dataset(x=[1,2,3])
    assert x.get_units('x') is None
    x.set_prototype('x','f', units='m')
    x = Dataset()
    x.set('x',[1,2,3],units='m')
    assert x.get_units('x') == 'm'
    x = Dataset()
    x.set_prototype('x','f', units='m')
    x['x'] = [1,2,3]
    assert x.get_units('x') == 'm'
    assert all(x.get('x') == [1,2,3])
    assert x.get('x',units='nm') == approx([1e9,2e9,3e9])
    x.set('x',[0.1,0.2,0.3],'unc')
    assert x.get('x','unc',units='nm') == approx([0.1e9,0.2e9,0.3e9])

def test_format_description():
    x = Dataset()
    x.set('x',[1,2,3],kind='f',units='m',description='This is x',fmt='0.3f')
    x.set('x_unc',[0.1,0.1,0.3])

