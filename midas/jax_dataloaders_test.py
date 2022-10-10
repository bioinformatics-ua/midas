# Unit test only for the jax_dataloaders
from midas.utils_test import dict_gen

# Unit tests that combine uses the Tf dataloaders converts
from midas import DataLoader
import jax.numpy as jnp
import jax
from jaxlib.xla_extension import DeviceArray
from jaxlib.xla_extension.pmap_lib import ShardedDeviceArray

def test_simple_conversion():
    
    dataloader = DataLoader(dict_gen)
    dataloader = dataloader.to_jax()
    
    mock_transformation = ["tf.data.Dataset.from_generator",
                           "midas.DataLoader.to_jax"]
    
    for data in dataloader:
        #verify every element
        assert isinstance(data["a"], DeviceArray)
        assert isinstance(data["b"], DeviceArray)
        
        assert data["a"].shape==()
        assert data["b"].shape==()
        
    # check last value
    assert int(jax.device_get(data["a"]))==99
    assert int(jax.device_get(data["b"]))==198
    
    for i,transformation in enumerate(dataloader.get_transformation_list()):
        assert transformation==mock_transformation[i]
        
def test_simple_batch_conversion():
    
    dataloader = DataLoader(dict_gen)
    dataloader = dataloader.batch(20)
    dataloader = dataloader.to_jax()
    
    mock_transformation = ["tf.data.Dataset.from_generator",
                           "tf.data.Dataset.batch",
                           "midas.DataLoader.to_jax"]
    
    for data in dataloader:
        #verify every element
        assert isinstance(data["a"], DeviceArray)
        assert isinstance(data["b"], DeviceArray)
        
        assert data["a"].shape[0]==20
        assert data["b"].shape[0]==20
        
    # check last value
    assert int(jax.device_get(data["a"][-1]))==99
    assert int(jax.device_get(data["b"][-1]))==198
    
    for i,transformation in enumerate(dataloader.get_transformation_list()):
        assert transformation==mock_transformation[i]
        
def test_simple_batch_conversion_w_lambda():

    def sum_on_b(data):
        print("test?", data["b"])
        data["b"] = jnp.sum(data["b"], axis=-1)
        return data
    
    def argmax_on_a_to_c(data):
        data["c"] = jnp.argmax(data["a"], axis=-1)
        return data
    
    dataloader = DataLoader(dict_gen)
    dataloader = dataloader.batch(20)
    dataloader = dataloader.to_jax()
    dataloader = dataloader.lambda_transformation(sum_on_b)
    dataloader = dataloader.lambda_transformation(argmax_on_a_to_c)
    
    mock_transformation = ["tf.data.Dataset.from_generator",
                           "tf.data.Dataset.batch",
                           "midas.DataLoader.to_jax",
                           "midas.JaxDataLoader.lambda_transformation",
                           "midas.JaxDataLoader.lambda_transformation"]
    
    for data in dataloader:
        #verify every element
        assert isinstance(data["a"], DeviceArray)
        assert isinstance(data["b"], DeviceArray)
        assert isinstance(data["c"], DeviceArray)
        
        assert data["a"].shape[0]==20
        assert data["b"].shape==()
        assert data["c"].shape==()
        
    # check last value
    assert int(jax.device_get(data["a"][-1]))==99
    assert int(jax.device_get(data["b"]))==3580
    assert int(jax.device_get(data["c"]))==19
    
    for i,transformation in enumerate(dataloader.get_transformation_list()):
        assert transformation==mock_transformation[i]
        
def test_simple_batch_conversion_w_shard():
    
    dataloader = DataLoader(dict_gen)
    dataloader = dataloader.batch(20)
    dataloader = dataloader.to_jax()
    dataloader = dataloader.shard()
    
    number_of_devices=jax.local_device_count()
    
    mock_transformation = ["tf.data.Dataset.from_generator",
                           "tf.data.Dataset.batch",
                           "midas.DataLoader.to_jax",
                           "midas.JaxDataLoader.shard"]
    
    for data in dataloader:
        #verify every element
        assert isinstance(data["a"], DeviceArray)
        assert isinstance(data["b"], DeviceArray)
        
        assert data["a"].shape==(number_of_devices,20/number_of_devices)
        assert data["b"].shape==(number_of_devices,20/number_of_devices)
        
    # check last value
    assert int(jax.device_get(data["a"][-1,-1]))==99
    assert int(jax.device_get(data["b"][-1,-1]))==198
    
    for i,transformation in enumerate(dataloader.get_transformation_list()):
        assert transformation==mock_transformation[i]
        
def test_simple_batch_conversion_w_shard_fixedvalue():
    
    number_of_devices=10
    
    dataloader = DataLoader(dict_gen)
    dataloader = dataloader.batch(20)
    dataloader = dataloader.to_jax()
    dataloader = dataloader.shard(devices=number_of_devices)
    
    mock_transformation = ["tf.data.Dataset.from_generator",
                           "tf.data.Dataset.batch",
                           "midas.DataLoader.to_jax",
                           "midas.JaxDataLoader.shard"]
    
    for data in dataloader:
        #verify every element
        assert isinstance(data["a"], DeviceArray)
        assert isinstance(data["b"], DeviceArray)
        
        assert data["a"].shape==(number_of_devices,20/number_of_devices)
        assert data["b"].shape==(number_of_devices,20/number_of_devices)
        
    # check last value
    assert int(jax.device_get(data["a"][-1,-1]))==99
    assert int(jax.device_get(data["b"][-1,-1]))==198
    
    for i,transformation in enumerate(dataloader.get_transformation_list()):
        assert transformation==mock_transformation[i]
        
        
def test_simple_batch_conversion_w_shard_prefetch():
    
    # this test should only run if more than 1 device is available
    number_of_devices=jax.local_device_count()
    
    if number_of_devices>1:
        dataloader = DataLoader(dict_gen)
        dataloader = dataloader.batch(20)
        dataloader = dataloader.to_jax()
        dataloader = dataloader.shard()
        dataloader = dataloader.prefetch_to_devices()
        
        mock_transformation = ["tf.data.Dataset.from_generator",
                                "tf.data.Dataset.batch",
                                "midas.DataLoader.to_jax",
                                "midas.JaxDataLoader.shard",
                                "midas.JaxDataLoader.prefetch_to_devices"]
        
        for data in dataloader:
            #verify every element
            assert isinstance(data["a"], ShardedDeviceArray)
            assert isinstance(data["b"], ShardedDeviceArray)
            
            a_placement=set()
            b_placement=set()
            
            for i in range(number_of_devices):
                assert isinstance(data["a"].device_buffers[i], DeviceArray)
                a_device_id = data["a"].device_buffers[i].device().id
                assert a_device_id not in a_placement
                a_placement.add(a_device_id)
                assert isinstance(data["b"].device_buffers[i], DeviceArray)
                b_device_id = data["b"].device_buffers[i].device().id
                assert b_device_id not in b_placement
                b_placement.add(b_device_id)
            
            assert data["a"].shape==(number_of_devices,20/number_of_devices)
            assert data["b"].shape==(number_of_devices,20/number_of_devices)
            
        # check last value
        assert int(jax.device_get(data["a"][-1,-1]))==99
        assert int(jax.device_get(data["b"][-1,-1]))==198
        
        for i,transformation in enumerate(dataloader.get_transformation_list()):
            assert transformation==mock_transformation[i]

    
