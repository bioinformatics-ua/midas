
import tensorflow as tf
# disable tf GPUs? check if it is needed
#tf.config.set_visible_devices([], 'GPU')
from typing import List, Iterable
import inspect

from midas.generic_dataloaders import GenericDataLoader
from midas.jax_dataloaders import TfJaxConverterDataLoader
from midas.numpy_dataloaders import TfNumpyConverterDataLoader
from midas.torch_dataloaders import TfTorchConverterDataLoader

def find_dtype_and_shapes(data_generator, k=10):
    """
    Automatically infer the data type and shapes of the samples from 
    the *data_generator*.
    
    Args:
      data_generator (python generator): python generator that output
        samples in a **dictionary** format.
        
      k (int): Number of samples that are output from *data_generator*
        in order to correctly infer the shape of each dictionary value.
        For instance, if a sample contains a dynamic shape, this function
        will return None in the place of the dimension that as a dynamic 
        shape. If set to -1 it will read the entire generator
    """

    if k==-1:
        samples = [ sample for sample in data_generator ]
    elif k>0:
        generator = iter(data_generator)
        samples = [ next(generator) for i in range(k) ]

    if isinstance(samples[0], dict):
        
        dtypes = {}
        shapes = {}

        for key in samples[0].keys():
            tf_value = tf.constant(samples[0][key])
            dtypes[key] = tf_value.dtype
            shapes[key] = tf_value.shape

        # infer the shape since it can be None if some dim is not in agreement 
        for i in range(len(samples)-1):
            assert len(set(samples[i].keys()) - set(samples[i+1].keys())) == 0
            for key in samples[i+1].keys():
                tf_value = tf.constant(samples[i+1][key])

                # they must have the same dimensionality, but can have diff values per dimension
                assert len(tf_value.shape) == len(shapes[key])

                # sample with diff value in one of the dimensions
                if tf_value.shape != shapes[key]:
                    new_shape = list(shapes[key])
                    for j in range(len(shapes[key])):
                        if shapes[key][j] is not None and shapes[key][j]!=tf_value.shape[j]:
                            new_shape[j] = None
                    shapes[key] = tf.TensorShape(new_shape)

    else:
        raise ValueError(f"The find_dtype_and_shapes only supports samples in the dict format. Expected dict but found {type(samples[0])}")
        
    return shapes, dtypes

def convert_in_Tensor_spec(shapes, dtypes):
    return { key: tf.TensorSpec(shape=shape, dtype=dtypes[key], name=f"{key}_input")for key, shape in shapes.items()}

class DataLoader(GenericDataLoader):
    
    def __init__(self, 
                 python_iterable,
                 python_iterable_n_samples=None,
                 output_signature=None,
                 infer_k=3,
                 transformation_tracker:List[str] = None,
                 input_dataset=None):

        super().__init__(python_iterable=python_iterable, 
                         python_iterable_n_samples=python_iterable_n_samples,
                         transformation_tracker=transformation_tracker)
        
        self.output_signature = output_signature
        
        if input_dataset is None:
            # if not available, try to automaticly infer the shape of the generator by looking at K samples
            if self.output_signature is None:
                self.output_signature = convert_in_Tensor_spec(*find_dtype_and_shapes(self.python_iterable, k=infer_k))
                
            _tf_ds = tf.data.Dataset.from_generator(self.python_iterable, 
                                                    output_signature=self.output_signature)
            # add this transformation
            self._transformation_tracker.append("tf.data.Dataset.from_generator")
            
            # this is equivalent to self.input_dataset=_tf_ds
            self._set_input_dataset(_tf_ds)
        else:
            self._set_input_dataset(input_dataset)
        
        for method in filter(lambda x: not x[0].startswith("_"), inspect.getmembers(tf.data.Dataset, predicate=inspect.isfunction)):
            # all known public tf.data.Dataset API transformation functions will be automaticly added to the DataLoader class
            setattr(self, method[0], self.__add_tf_dataset_f(method[0]))
                    
    
    # a bit of magic
    def __add_tf_dataset_f(self, name):
        """
        Builds a function
        """
        def f(*args, **kwargs):
            new_input_dataset = getattr(self.input_dataset, name)(*args, **kwargs)
            new_dl = DataLoader(python_iterable=self.python_iterable,
                                python_iterable_n_samples=self.python_iterable_n_samples,
                                output_signature=self.output_signature,
                                transformation_tracker = self._transformation_tracker[:] + [f"tf.data.Dataset.{name}"],
                                input_dataset=new_input_dataset)
            return new_dl
        return f
    
    def __iter__(self):
        # returns the current tf dataset iterator
        return self.input_dataset.__iter__()
    
    def get_transformation_list(self):
        return self._transformation_tracker
        
    def to_jax(self):

        return TfJaxConverterDataLoader(input_dataset=self.input_dataset,
                             python_iterable=self.python_iterable,
                             python_iterable_n_samples=self.python_iterable_n_samples,
                             transformation_tracker=self._transformation_tracker[:] + ["midas.DataLoader.to_jax"])
        
    def to_numpy(self):

        return TfNumpyConverterDataLoader(input_dataset=self.input_dataset,
                             python_iterable=self.python_iterable,
                             python_iterable_n_samples=self.python_iterable_n_samples,
                             transformation_tracker=self._transformation_tracker[:] + ["midas.DataLoader.to_numpy"])
        
    def to_torch(self):
        
        return TfTorchConverterDataLoader(input_dataset=self.input_dataset,
                             python_iterable=self.python_iterable,
                             python_iterable_n_samples=self.python_iterable_n_samples,
                             transformation_tracker=self._transformation_tracker[:] + ["midas.DataLoader.to_torch"])
        
    def to_lambdaDataLoader(self, new_dl_class):
        return new_dl_class(input_dataset=self.input_dataset,
                             python_iterable=self.python_iterable,
                             python_iterable_n_samples=self.python_iterable_n_samples,
                             transformation_tracker=self._transformation_tracker[:] + [new_dl_class.__name__])


    