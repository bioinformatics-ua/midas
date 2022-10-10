from typing import Iterable, List
from queue import Queue
from midas.generic_dataloaders import LambdaDataLoader, _DataLoaderIterator
from midas.utils import prefetch, check_if_lib_is_installed

# tensorflow is required!!
from tensorflow.experimental.dlpack import to_dlpack

if check_if_lib_is_installed("jax"):
    # it may be not necessary if there is no intention
    # to convert the dataloader to a jax format
    import jax.dlpack

def from_tf_to_jax(tensor):
    return jax.dlpack.from_dlpack(to_dlpack(tensor))

class JaxDataLoader(LambdaDataLoader):
    
    def lambda_transformation(self, transform_f):
        return JaxDataLoader(input_dataset=self, 
                             transform_f=transform_f,
                             python_iterable=self.python_iterable,
                             python_iterable_n_samples=self.python_iterable_n_samples,
                             transformation_tracker=self._transformation_tracker[:] + ["polydl.JaxDataLoader.lambda_transformation"])
    
    def prefetch_to_devices(self, size=2):
        return JaxPrefetchToDevicesDataLoader(input_dataset=self, 
                                              size=size,
                                              python_iterable=self.python_iterable,
                                              python_iterable_n_samples=self.python_iterable_n_samples,
                                              transformation_tracker=self._transformation_tracker[:] + ["polydl.JaxDataLoader.prefetch_to_devices"])
    
    def shard(self, devices:int=None):
        return JaxShardDataLoaders(input_dataset=self, 
                                   devices=devices,
                                   python_iterable=self.python_iterable,
                                   python_iterable_n_samples=self.python_iterable_n_samples,
                                   transformation_tracker=self._transformation_tracker[:] + ["polydl.JaxDataLoader.shard"])

class TfJaxConverterDataLoader(JaxDataLoader):
    def __init__(self, 
                 input_dataset,
                 python_iterable: Iterable,
                 python_iterable_n_samples:int=None,
                 transformation_tracker:List[str] = None):
        
        super().__init__(input_dataset=input_dataset,
                         python_iterable=python_iterable,
                         python_iterable_n_samples=python_iterable_n_samples,
                         transformation_tracker=transformation_tracker)
    
    def _transform(self, data):
        return jax.tree_util.tree_map(from_tf_to_jax , data)

class JaxShardDataLoaders(JaxDataLoader):
    """
    Similar to shard method from flax
    https://flax.readthedocs.io/en/latest/_modules/flax/training/common_utils.html#shard
    """
    def __init__(self, 
                 input_dataset: Iterable,
                 devices:int=None,
                 python_iterable: Iterable=None,
                 python_iterable_n_samples:int=None,
                 transformation_tracker:List[str] = None) -> None:
        super().__init__(input_dataset, 
                         python_iterable=python_iterable,
                         python_iterable_n_samples=python_iterable_n_samples,
                         transformation_tracker=transformation_tracker)
        
        if devices is None:
            self.devices=jax.local_device_count()
        else:
            self.devices=devices
            
    def _transform(self, data):
        return jax.tree_util.tree_map(lambda x: x.reshape((self.devices, -1) + x.shape[1:]) , data)


class JaxPrefetchToDevicesDataLoader(JaxDataLoader):
    
    def __init__(self, 
                 input_dataset: Iterable, 
                 size:int=2,
                 python_iterable: Iterable=None,
                 python_iterable_n_samples:int=None,
                 transformation_tracker:List[str] = None) -> None:
        super().__init__(input_dataset, 
                         python_iterable=python_iterable,
                         python_iterable_n_samples=python_iterable_n_samples,
                         transformation_tracker=transformation_tracker)
        self.max_size = size
        
    class JaxPrefetchToDevicesDataLoaderIterator(_DataLoaderIterator):
        def __init__(self, 
                     input_dataset,
                     max_size:int,
                     transform_f) -> None:
            super().__init__(input_dataset=input_dataset, transform_f=transform_f)
            self.queue = Queue(max_size)
            
        def _next_gen(self):
            for data in prefetch(self.queue, self.input_dataset, self.transform_f):
                yield data
            
        def __next__(self):
            return next(self.iterator)
    
    def __iter__(self):
        return self.JaxPrefetchToDevicesDataLoaderIterator(input_dataset=self.input_dataset,
                                                           max_size=self.max_size,
                                                           transform_f=self._transform)    
    def _transform(self, data):
        return jax.tree_util.tree_map(lambda x: jax.device_put_sharded(list(x),devices=jax.local_devices()), data)