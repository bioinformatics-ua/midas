# MIDAS: **M**ult**I**-framework **D**atalo**A**der**S**

MIDAS leverages the power of the [DLPack](https://github.com/dmlc/dlpack) protocol to export highly efficient [_tf.data.Dataset_](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) to other Deep Learning frameworks, like JAX or PyTorch. Hence, **MIDAS** is a multi-framework DataLoader that can be leverage to produce highly efficient data pipelines.

## How to use?
One of the selling points of MIDAS is it simplicity and easy of use, consider the bellow example, where we build a python generator as our source data and then build an efficient DataLoader that distributes the data between two GPU's .

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
# the .to_jax is a convertion method that will convert the Tensors from tf.data.Dataset to JAX tensors
jdl = dl.to_jax()
# Now jdl is a JaxDataLoader that supports jax specific types of transformations.
jdl = jdl.shard() # shard the data between two GPUs (Number of GPUs available on this machine)
jdl = jdl.prefetch_to_devices() # send the data to the GPUs

for data in jdl:
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

## How this works?

**TL;DR:** Python iterable -> _tf.data.Dataset_ -> _tf.data.Dataset_ data transformations -> DLPack -> Another framework (like JAX or PyTorch)

There a three basic steps to build DataLoaders with MIDAS:
- **Definition of python iterable**: This corresponds to the source of your data. Here the _midas.DataLoader_ will fetch your data in order to build the associated _tf.data.Dataset_. The only restriction here is that the Here a iterable (python generator or class iterable) 
- Second item
- Third item


## Why should you use this?

**First reason, It is realy simple to use**, you just need to specify your data as a python iterable (generation function or a class) and that's it the **midas.DataLoader** will take care of the convertions to _tf.data.Dataset_ and will expose transformation methods to change the data between frameworks.

Take this as an example:
<details>
  <summary>**Click here to show**</summary>
```python
from midas import DataLoader

def dummy_data_generator():
    for i in range(1000):
        yield {"x": i, "y": i*3}

dl = DataLoader(dummy_data_generator)
dl = dl.map(lambda x: {**x, "is_y_even": x["y"]%2==0}) # tf.data.Dataset.map
dl = dl.cache() # tf.data.Dataset.cache
dl = dl.shuffle(1000) # tf.data.Dataset.shuffle
dl = dl.batch(50) # tf.data.Dataset.batch
dl = dl.to_jax()

for data in dl:
    pass

print(data)
```
Output:
```python
{'is_y_even': DeviceArray([0, 1, 1, 0, 0], dtype=uint8),
 'x': DeviceArray([195, 144, 484, 651, 647], dtype=int32),
 'y': DeviceArray([ 585,  432, 1452, 1953, 1941], dtype=int32)}
```
</details>

**Second reason, it is FAST!**, MIDAS is a direct wrapper of the _tf.data.Dataset_ class and its transformation, which converts all of the defined processing steps into a computation graph on CPU, then thanks to the _DLPack_ the convertion to other DL frameworks is achieved with almost no overhead.

**Third reason, it is highly extensible**, thanks to a modular coding approach new Deep Learning frameworks can be added following a functional programming style. For that, the user just needs to call `.to_lambdaDataLoader(NEW_CLASS_CONVERTER_DATALOADER)`, and implement a new class that extends `LambdaDataLoader` (Note: the DLPack should be use for fast convertion between different frameworks Tensors, however, this is not required).

**Additional functionality for other DL frameworks**, besides the framework convertion, MIDAS also expose framework specific transformations. For instance, when converting to a `JaxDataLoader` it is also possible to follow the dataloader with the `shard` and `prefetch_to_devices` transformations, which automaticly shards and distributes the data to the best accelerators devices found by _jax_.

**But I am already a TF user, why should I care?**. Well, MIDAS also offers the additional functionallity of automaticly converting python iterables to _tf.data.Dataset_. For that it automaticly infers the _output_signature_ produced by your iterable, making more easy to build DataLoaders from generators.

# Thecnical stuff

## How MIDAS automaticly infer the output_signature of a python_iterable?

