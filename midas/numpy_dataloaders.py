from typing import Iterable, List
from queue import Queue
from midas.generic_dataloaders import LambdaDataLoader, _DataLoaderIterator
from midas.utils import check_if_lib_is_installed

# tensorflow is required!!
from tensorflow.experimental.dlpack import to_dlpack

if check_if_lib_is_installed("numpy"):
    # it may be not necessary if there is no intention
    # to convert the dataloader to a numpy format
    import numpy as np

def from_tf_to_np(tensor):
    return np.from_dlpack(to_dlpack(tensor))

class NumpyDataLoader(LambdaDataLoader):
    # add here transformation that should be visible to a 
    # NumpyDataLoader
    def lambda_transformation(self, transform_f):
        return NumpyDataLoader(input_dataset=self, 
                                     transform_f=transform_f,
                                     python_iterable=self.python_iterable,
                                     python_iterable_n_samples=self.python_iterable_n_samples,
                                     transformation_tracker=self._transformation_tracker[:] + ["midas.NumpyDataLoader.lambda_transformation"])
    

class TfNumpyConverterDataLoader(NumpyDataLoader):
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
        return { k:from_tf_to_np(v) for k,v in data.items() }
