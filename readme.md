# MIDAS: **M**ult**I**-framework **D**atalo**A**der**S**

MIDAS leverages the power of the [DLPack](https://github.com/dmlc/dlpack) protocol to export highly efficient [_tf.data.Dataset_](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to other Deep Learning frameworks, like JAX or PyTorch. Hence, **MIDAS** is a multi-framework DataLoader that can be leveraged to produce highly efficient data pipelines.

## How to use it?
One of the selling points of MIDAS is its simplicity and ease of use, consider the below example, where we build a python generator as our source data that is automatically converted to an efficient DataLoader that distributes the data between two GPUs in jax.

```python
from midas import DataLoader

def dummy_data_generator():
    for i in range(1000):
        yield {"x": i, "y": i*3}

dl = DataLoader(dummy_data_generator)
# at this point dl wraps the tf.data.Dataset, so every tf.data.Dataset can be used over the 
# dl object as shown bellow
dl = dl.map(lambda x: {**x, "is_y_even": x["y"]%2==0}) # tf.data.Dataset.map
dl = dl.cache() # tf.data.Dataset.cache
dl = dl.shuffle(1000) # tf.data.Dataset.shuffle
dl = dl.batch(10) # tf.data.Dataset.batch
# the .to_jax is a conversion method that will convert the Tensors from tf.data.Dataset to JAX tensors
jdl = dl.to_jax()
# Now jdl is a JaxDataLoader that supports jax-specific types of transformations.
jdl = jdl.shard() # shard the data between two GPUs (Number of GPUs available on this machine)
jdl = jdl.prefetch_to_devices() # send the data to the GPUs

for data in jdl:
    # iterating over the dataloader
    pass

print(data)
```
Output:
```python
{'is_y_even': ShardedDeviceArray([[0, 0, 0, 0, 0],
                     [1, 0, 1, 0, 1]], dtype=uint8),
 'x': ShardedDeviceArray([[689,  97,   9, 129, 945],
                     [274, 761,  42, 937, 470]], dtype=int32),
 'y': ShardedDeviceArray([[2067,  291,   27,  387, 2835],
                     [ 822, 2283,  126, 2811, 1410]], dtype=int32)}
```

### Additional utility provided by MIDAS
Besides handling the DataLoaders conversions, MIDAS also adds some utility functions like:

`get_python_iterator_n_samples` will return the number of samples output by the python iterable, if this value is not specified during the initialization of a _midas.DataLoader_ would be inferred by automatically transversing the python iterable. Therefore, do not call `get_python_iterator_n_samples` if your python iterable does not stop.
```
# continuation with the jdl created in the previous example
>>> jdl.get_python_iterator_n_samples()
1000
```

`get_n_samples` will return the number of samples output by the DataLoader. Note that this number will differ from the get_python_iterator_n_samples if any data aggregation was performed. For instance, in this case, we are using the `.batch(10)` transformation that aggregates 10 sequential samples into one. Therefore, the current value for get_n_samples would be `jdl.get_python_iterator_n_samples()/10`.
```
# continuation with the jdl created in the previous example
>>> jdl.get_n_samples()
100
```

`get_transformation_list` returns a list with all the transformations that were applied to the current DataLoader. For now _midas.DataLoder_ does not take advantage of this information, but in a future version, this can be the starting point to implement a DataLoader chain optimizer that rearranges the specified transformations into a more suitable order of execution that maximizes performance.
```
# continuation with the jdl created in the previous example
>>> jdl.get_transformation_list()
['tf.data.Dataset.from_generator', 
 'tf.data.Dataset.map', 
 'tf.data.Dataset.cache', 
 'tf.data.Dataset.shuffle', 
 'tf.data.Dataset.batch', 
 'midas.DataLoader.to_jax', 
 'midas.JaxDataLoader.shard', 
 'midas.JaxDataLoader.prefetch_to_devices']
```

## How to install

From pypi
```
TODO
```

From this repository (last commit in the master branch)
```
pip install git+https://github.com/bioinformatics-ua/midas.git
```

## How this works?

**TL;DR:** Python iterable -> _tf.data.Dataset_ -> _tf.data.Dataset_ data transformations -> DLPack -> Another DL framework (like JAX or PyTorch)

There are three basic steps to build DataLoaders with MIDAS:
- **Definition of python iterable**: This corresponds to the source of your data. Here, the _midas.DataLoader_ will fetch your data in order to build the associated _tf.data.Dataset_. The only restriction here is that the iterable (python generator or class iterable) must yield a python dictionary (or similar structure) with your data. We decided that the data flow when using MIDAS should always be represented as a python dictionary given its high interpretability. Therefore, _midas.DataLoader_ will automatically raise an exception if any sample is not formatted as a dictionary.
- **Using fast tf.data.Dataset transformations**: After creating the _midas.DataLoader_ the further preprocessing should be done by leveraging the _tf.data.Dataset_ API. For instance, to transform the data call `.map`, to shuffle use `.shuffle`, to batch use `.batch` as you would normally do as a normal _tf.data.Dataset_. (To check the complete list of transformations [click here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#methods_2))
- **Tensor conversion for the desired DL framework**: At the last step, after we have defined all of the preprocessing steps, we can now convert the resulting _tf.Tensor_ to our framework tensors. For this, we have exposed the following methods (`.to_jax`, `.to_torch`, `.to_numpy` and `.to_lambdaDataLoader`, where this last one is for functional DataLoader conversion.) 

## Why should you use this?

**First reason, It is really simple to use**, you just need to specify your data as a python iterable (generation function or a class) and that's it the **midas.DataLoader** will take care of the conversions to _tf.data.Dataset_ and will expose transformation methods to change the data between frameworks.

**Second reason, it is FAST!**, MIDAS is a direct wrapper of the _tf.data.Dataset_ class and its transformation, which converts all of the defined processing steps into a computation graph on CPU, then thanks to the _DLPack_ the conversion to other DL frameworks is achieved with almost no overhead.

**Third reason, it is highly extensible**, thanks to a modular coding approach new Deep Learning frameworks can be added following a functional programming style. For that, the user just needs to call `.to_lambdaDataLoader(NEW_CLASS_CONVERTER_DATALOADER)`, and implement a new class that extends `LambdaDataLoader` (Note: the DLPack should be used for fast conversion between different frameworks Tensors, however, this is not required).

**Additional functionality for other DL frameworks**, besides the framework conversion, MIDAS also expose framework-specific transformations. For instance, when converting to a `JaxDataLoader` it is also possible to follow the dataloader with the `shard` and `prefetch_to_devices` transformations, which automatically shards and distributes the data to the best accelerators devices found by _jax_.

**But I am already a TF user, why should I care?**. Well, MIDAS also offers the additional functionality of automatically converting python iterable (even for data with variable lengths, like text!) to _tf.data.Dataset_. For that it automatically infers the _output_signature_ produced by your iterable, making it easier to build DataLoaders from generators.

# Thecnical stuff

## Built-in tensor conversions

Currently, MIDAS supports the following tensor conversions:

    - tensorflow -> jax
    - tensorflow -> torch
    - tensorflow -> numpy

Additionally, dynamic new conversion can be added by extending `LambdaDataLoader` and passing the resulting class to the `.to_lambdaDataLoader`method.

## How MIDAS automatically infer the output_signature of a python_iterable?

When no `output_signature` is specified, the _midas.DataLoader_ will automatically infer the correct `output_signature` from the samples of the python iterable. For that, a specific number of samples are gathered from the python iterable, which is specified by the `infer_k=3` argument. So, by default, the DataLoader samples 3 instances of the python iterable and then each value of its dictionary is automatically converted to a tf.Tensor. From here we catch the shape and the data type of each dictionary value. Then we perform the same procedure on the remainder of the samples and at each step, we verify its consistency with the previous discover data type and shape. If the data type changes, then it's an Error. On the other hand, if the shape changes it means that the data has a variable length, which is automatically handled by assigning the None shape.

Consider this fixed size example first:
```
from midas import DataLoader

def dummy_data_generator():
    for i in range(1000):
        yield {"x": i, "y": i*3}

dl = DataLoader(dummy_data_generator)
dl.output_signature
>>> {'x': TensorSpec(shape=(), dtype=tf.int32, name='x_input'), 'y': TensorSpec(shape=(), dtype=tf.int32, name='y_input')}
```
Then consider this variable size example:
```
from midas import DataLoader
import random

def dummy_data_generator():
    for i in range(1000):
        yield {"x": [i for _ in range(random.randint(1,5))], "y": i*3}

dl = DataLoader(dummy_data_generator)
dl.output_signature
>>> {'x': TensorSpec(shape=(None,), dtype=tf.int32, name='x_input'), 'y': TensorSpec(shape=(), dtype=tf.int32, name='y_input')}
```

Here, since `x` is represented as a list that ranges from 1 element to 5 elements, the _midas.DataLoader_ detected an inconsistency in the shape sizes, hence making the assumption that must be a Tensor with variable length. Therefore, it has represented with an unknown shape (None,).

Furthermore, `infer_k` controls how many samples are consumed in order to infer the correct shape. For performance reasons the default value is low (3), which may produce some errors on datasets where data with variable length is rare since the DataLoader will only detect this if one of the first three samples has a different shape. So during this case, consider increasing the value of `infer_k`. As an extreme resource setting, `infer_k=-1` will force the DataLoader to check the shapes of every sample on the python iterable.