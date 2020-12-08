import numpy as np

from spectr import tools

def test_import():
    from spectr import tools
    pass

def test_AutoDict():
    t = tools.AutoDict([])
    t['x'] = [5,6]
    assert t['x'] == [5,6]
    assert t['y'] == []
    t = tools.AutoDict({})
    assert t['y'] == {}
    t['x']['a'] = 1
    assert t['x'] == {'a':1}
    ## two of them!
    t = tools.AutoDict(tools.AutoDict({}))
    assert t['x']['a'] == {}
    t['y']['b'][1] = 5
    assert t['y']['b'] == {1:5}

def test_vectorise():
    @tools.vectorise()
    def f(x):
        return x**2
    assert f(5) == 25
    assert list(f([1,2,2,3])) == [1,4,4,9]
    @tools.vectorise()
    def f(x,y):
        return x+y
    assert f(5,3) == 8
    assert list(f([1,3],[2,4])) == [3,7]
    assert list(f([1,3],2)) == [3,5]
    @tools.vectorise()
    def f(x,y):
        return x+y
    assert list(f([1,3],np.array([2,4]))) == [3,7]
    @tools.vectorise(vargs=(0,))
    def f(x,y):
        return x+y
    a,b = f([1,3],np.array([2,4]))
    assert all(a == np.array([3,5]))
    assert all(b == np.array([5,7]))
    @tools.vectorise(dtype=float)
    def f(x,y):
        return x+y
    retval = f([1,3],[2,4])
    assert isinstance(retval,np.ndarray)
    assert all(retval == [3,7])
    @tools.vectorise(cache=True)
    def f(x,y):
        return x+y
    retval = f([1,3,1],[2,4,2])
    assert retval == [3,7,3]
    @tools.vectorise(cache=True)
    def f(x,y):
        return x+y
    t = np.full(1000,100)
    retval = f([1,3,1],[2,4,2])
    assert retval == [3,7,3]

# def test_vectorise_function_in_chunks():
    # @tools.vectorise_function_in_chunks(float)
    # def f(x):
        # return x**2
    # assert f(5) == 25
    # assert list(f([1,2,2,3])) == [1,4,4,9]
    # @tools.vectorise_function_in_chunks(float)
    # def f(x,y):
        # return x+y
    # assert f(5,3) == 8
    # assert list(f([1,3],[2,4])) == [3,7]
    # assert list(f([1,3],2)) == [3,5]
    # @tools.vectorise_function_in_chunks(float)
    # def f(x,y):
        # return x+y
    # assert list(f([1,3],np.array([2,4]))) == [3,7]

# def test_vectorise_function():
    # @tools.vectorise_function
    # def f(x,y):
        # if not np.isscalar(x):
            # assert not np.isscalar(y)
        # return x + y
    # assert f(5,0) == 5
    # assert list(f(1,[1,2])) == [2,3]

def test_vectorise_arguments():
    @tools.vectorise_arguments
    def f(x,y):
        retval = np.full(np.shape(x),1)
        retval[(x==2)&(y==2)] = 2
        return retval
    assert list(f([1,2,3],[2,2,2])) == [1,2,1]
    assert list(f([1,2,3],2)) == [1,2,1]
    assert f(1,2) == 1
    
