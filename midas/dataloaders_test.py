from optax import lamb
from midas.utils_test import dict_gen
from midas import DataLoader

def test_counting_python_samples():
  
    dl = DataLoader(dict_gen)
    #dl = dl.map(lambda x:{**x, x["c"]:x["a"]+x["b"]})
    n_samples_python = dl.get_python_iterator_n_samples()
    n_samples_dataset = dl.get_n_samples()
    assert n_samples_python==dict_gen.N_SAMPLES
    assert n_samples_dataset==dict_gen.N_SAMPLES
    
def test_python_tfdataset_map():
    
    dl = DataLoader(dict_gen)
    dl = dl.map(lambda x:{**x, "c":x["a"]+x["b"]})
    n_samples_python = dl.get_python_iterator_n_samples()
    n_samples_dataset = dl.get_n_samples()
    
    assert n_samples_python==dict_gen.N_SAMPLES
    assert n_samples_dataset==dict_gen.N_SAMPLES
    
    for data in dl:
        assert data["c"]==data["a"]+data["b"]
        
def test_python_tfdataset_batch():
    
    dl = DataLoader(dict_gen)
    dl = dl.map(lambda x:{**x, "c":x["a"]+x["b"]})
    dl = dl.batch(20)
    
    n_samples_python = dl.get_python_iterator_n_samples()
    n_samples_dataset = dl.get_n_samples()
    
    assert n_samples_python==dict_gen.N_SAMPLES
    assert n_samples_dataset==dict_gen.N_SAMPLES/20
    
    for data in dl:
        for i in range(20):
            assert data["c"][i]==data["a"][i]+data["b"][i]