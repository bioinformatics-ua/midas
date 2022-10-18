from typing import Iterable, List
from queue import Queue
from midas.generic_dataloaders import LambdaDataLoader, _DataLoaderIterator
from midas.utils import check_if_lib_is_installed

# tensorflow is required!!
from tensorflow.experimental.dlpack import to_dlpack

if check_if_lib_is_installed("torch"):
    # it may be not necessary if there is no intention
    # to convert the dataloader to a numpy format
    import torch

def from_tf_to_torch(tensor):
    return torch.from_dlpack(to_dlpack(tensor))

class TorchDataLoader(LambdaDataLoader):
    # add here transformation that should be visible to a 
    # NumpyDataLoader
    def lambda_transformation(self, transform_f):
        return TorchDataLoader(input_dataset=self, 
                                     transform_f=transform_f,
                                     python_iterable=self.python_iterable,
                                     skip_keys = self.skip_keys,
                                     python_iterable_n_samples=self.python_iterable_n_samples,
                                     transformation_tracker=self._transformation_tracker[:] + ["midas.NumpyDataLoader.lambda_transformation"])
    
    
class TfTorchConverterDataLoader(TorchDataLoader):
    def __init__(self, 
                 input_dataset,
                 **kwargs):
        
        super().__init__(input_dataset=input_dataset,
                         **kwargs)
    
    def _transform(self, data):
        return { k:from_tf_to_torch(v) for k,v in data.items() }
