from midas import utils
from queue import Queue

def gen():
    for i in range(100):
        yield i
gen.N_SAMPLES=100 

def dict_gen():
    for i in range(100):
        yield {"a":i, "b":i*2}
dict_gen.N_SAMPLES=100

def base_test_prefetch_w_size(size):
    
    blocking_queue = Queue(size)
    
    transform_f = lambda x: x*2
    
    array = []  
    
    for data in utils.prefetch(blocking_queue, gen(), transform_f):
        array.append(data)
        
    expected_array = list(map(transform_f,range(100)))
    
    assert len(expected_array) == len(array)
    assert all([expected_array[i]==array[i] for i in range(len(expected_array))])    

def test_prefetch():
    base_test_prefetch_w_size(1)
    base_test_prefetch_w_size(2)
    base_test_prefetch_w_size(10)
    base_test_prefetch_w_size(100)
    base_test_prefetch_w_size(101)
    base_test_prefetch_w_size(200)
    
    
    
    