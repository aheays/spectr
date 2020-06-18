from spectr.dataset import Data,Dataset
import numpy as np

#######################
## Test Data object ##
#######################

def test_data_construct():
    t = Data(value=1)
    assert t._value == 1
    t = Data(value='dd')
    assert t._value == 'dd'
    t = Data(value=1.5,uncertainty=0.3)
    assert t._value == 1.5
    assert t._uncertainty == 0.3
    t = Data(value=[1.5,2.5],uncertainty=[0.3,0.2])
    assert list(t._value) == [1.5,2.5]
    assert abs(t._uncertainty[0]-0.3)<1e-5
    assert abs(t._uncertainty[1]-0.2)<1e-5
    t = Data(value=1,kind=float)
    assert type(t.value) == float

def test_data_set_and_get_value_and_uncertainty():
    t = Data()
    t.value = 5
    assert t.value == 5.
    t.uncertainty = 0.1
    assert t.uncertainty == 0.1
    t.value = [2,34]
    assert list(t.value) == [2.,34.]
    t.uncertainty = 0.5
    assert list(t.uncertainty) == [0.5,0.5]
    t.uncertainty = [0.2,0.2]
    assert list(t.uncertainty) == [0.2,0.2]

def test_data_index():
    t = Data(value=[1,2,3])
    t.index([True,True,False])
    assert list(t.value) == [1,2]
    assert len(t) == 2
    t = Data(value=[1,2,3],uncertainty=[0.1,0.2,0.3])
    t.index([True,True,False])
    assert list(t.value) == [1,2]
    assert list(t.uncertainty) == [0.1,0.2]

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
    t = Data(value=None,kind=object)
    assert t.value is None
    t = Data(value=[1,2,3],kind='S')
    assert t.value == [1,2,3]
    t = Data(value=['a',{},25],kind='O')
    assert list(t.value) == ['a',{},25]

######################
## Test Dataset object ##
######################

def test_dataset_construct():
    t = Dataset()

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

def test_construct_with_data_initialisation():
    t = Dataset(x=5)
    assert t['x'] == 5
    t = Dataset(x=['a','b'])
    assert list(t['x']) == ['a','b']
    t = Dataset(x=['a','b'],y=5)
    assert t['y'] == 5
    assert list(t['x']) == ['a','b']
    t = Dataset(x=[1.,2.],ux=[0.1,0.2])
    assert list(t['x']) == [1.,2.]
    assert list(t.get_uncertainty('x')) == [0.1,0.2]
    
def test_setitem_getitem():
    t = Dataset()
    t['x'] = 5
    assert t['x']==5
    t.set('y',5.,0.1)
    assert t['y']==5
    assert t['uy']==0.1

def test_dataset_prototypes():
    t = Dataset()
    t.add_prototype('x',description="X is a thing.",kind=float)
    t.set('x',5)
    assert isinstance(t['x'],float)
    assert t._data['x'].description == "X is a thing."

def test_dataset_permit_nonprototyped_data():
    t = Dataset()
    t.permit_nonprototyped_data = True
    t.set('x',5)
    # t.permit_nonprototyped_data = False
    # t.set('y',5)

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

def test_get_copy():
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy()
    assert list(u['x']) == [1,2,3]
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',))
    assert 'y' not in u 
    t = Dataset(x=[1,2,3],y=['a','b','c'])
    u = t.copy(('x',),[False,False,True])
    assert list(u['x']) == [3]

def test_dataset_construct_set_value():
    t = Dataset(x=5,y=['a','b'])
    assert t['x'] == 5
    assert all(t['y'] == ['a','b'])

def test_dataset_concatenate():
    t = Dataset()
    t.concatenate(Dataset())
    assert t.is_scalar()
    t = Dataset(x=1)
    t.concatenate(Dataset(x=2))
    print( t['x'])
    assert t['x'] == 1
    assert t.is_scalar()
    t = Dataset(x=[1,2])
    t.concatenate(Dataset(x=3))
    assert list(t['x']) == [1,2]
    assert len(t) == 2
    t = Dataset(x=1)
    t.concatenate(Dataset(x=[2,3]))
    assert list(t['x']) == [2,3]
    assert len(t) == 2
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

def test_dataset_extend():
    t = Dataset()
    t.extend()
    assert t.is_scalar()
    t = Dataset(x=1)
    t.extend(x=2)
    print( t['x'])
    assert t['x'] == 1
    assert t.is_scalar()
    t = Dataset(x=[1,2])
    t.extend(x=3)
    assert list(t['x']) == [1,2]
    assert len(t) == 2
    t = Dataset(x=1)
    t.extend(x=[2,3])
    assert list(t['x']) == [2,3]
    assert len(t) == 2
    t = Dataset(x=[1])
    t.extend(x=[2,3])
    assert list(t['x']) == [1,2,3]
    assert len(t) == 3
    t = Dataset(x=[1],y='a')
    t.extend(x=[2,3],y=['a','b'])
    assert list(t['x']) == [1,2,3]
    assert list(t['y']) == ['a','a','b']
    assert len(t) == 3

def test_dataset_append():
    t = Dataset(x=0)
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
    assert list(t['z']) == [3,4,5]
    t = Dataset()
    t.set('x',1.,0.1)
    t.set('y',2.,0.5)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y,lambda x,y,dx,dy:np.sqrt(dx**2+dy**2))
    assert t['z'] == 3
    assert t.get_uncertainty('z') == np.sqrt(0.1**2+0.5**2)
    t = Dataset()
    t.set('x',1.,0.1)
    t.set('y',2.,0.5)
    t.add_infer_function('z',('x','y'),lambda x,y:x+y,lambda x,y,dx,dy:np.sqrt(dx**2+dy**2))
    t['z']
    assert 'z' in t
    t.set('x',2.,0.2)
    assert 'z' not in t
    t['z']
    t['z'] = 5
    t.set('x',2.,0.2)
    assert 'z' in t

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
    t.set('x',[1,2,3])
    t.set('y',[1,2,3],[0.1,0.2,0.3])
    t.set('z',[2,4,5])
    t.plot('x',('y','z'),show=False)

def test_dataset_load_save_to_file():
    ## npz archive
    t = Dataset()
    t.set('x',[1,2,3],)
    t.set('f',1.29)
    t.save('tmp/t0.npz')
    u = Dataset()
    u.load('tmp/t0.npz')
    assert set(u.keys()) == {'x','f'}
    assert list(u['x']) == list(t['x'])
    assert u['f'] == t['f']
    ## hdf5 archive
    t = Dataset()
    t.set('x',[1,2,3],)
    t.set('f',1.29)
    t.save('tmp/t0.h5')
    u = Dataset()
    u.load('tmp/t0.h5')
    assert set(u.keys()) == {'x','f'}
    assert list(u['x']) == list(t['x'])
    assert u['f'] == t['f']
    ## text file
    t = Dataset()
    t.set('x',[1,2,3],)
    t.set('z',['a','b','c'],)
    t.set('f',1.29)
    t.save('tmp/t0.txt')
    u = Dataset()
    u.load('tmp/t0.txt')
    assert set(u.keys()) == {'x','f','z'}
    assert list(u['x']) == list(t['x'])
    assert list(u['z']) == list(t['z'])
    assert u['f'] == t['f']


