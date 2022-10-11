from optax import lamb
from midas.utils_test import dict_gen
from midas import DataLoader
import tensorflow_datasets as tfds
import tensorflow as tf

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
            
def test_DataLoader_from_tfdataset():
    
    mnist_data = tfds.load('mnist')

    BATCH_SIZE = 128
    
    def preprocess(batch):
        batch['image'] = tf.image.convert_image_dtype(batch['image'], tf.float32)
        batch['image'] = (batch['image'] - 0.5) / 0.5  # tanh range is -1, 1
        batch['label'] = tf.cast(batch['label'], tf.int32)
        return batch
    
    train_dl = DataLoader(mnist_data['train']) \
                     .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                     .cache() \
                     .shuffle(5000) \
                     .batch(BATCH_SIZE, drop_remainder=True)
                     
    for data in train_dl:
        assert data["image"].shape==(BATCH_SIZE,28,28,1)
        assert data["label"].shape==(BATCH_SIZE,)
