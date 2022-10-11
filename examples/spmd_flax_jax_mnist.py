import jax
import jax.numpy as jnp
import os
from functools import partial
from typing import Any, Tuple
from jaxlib.xla_client import SingleDeviceSharding


import tensorflow as tf
# do not allow tensorflow to grab the gpus
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

from jax.nn.initializers import normal as normal_init
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key

from flax import linen as nn
from flax import jax_utils
from tqdm import tqdm

import optax

from midas import DataLoader

class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

if __name__ == '__main__':
    GPUS = jax.local_devices()
    DEVICE_COUNT = len(GPUS)
    print(f"Jax found {GPUS}: {DEVICE_COUNT}")

    mnist_data = tfds.load('mnist')

    mnist_train, mnist_test = mnist_data['train'], mnist_data['test']

    SEED = 42
    PER_DEVICE_BATCH_SIZE = 64
    BATCH_SIZE = DEVICE_COUNT * PER_DEVICE_BATCH_SIZE
    EPOCHS = 20
    LEARNING_RATE = 0.0002
    
    def preprocess(batch):
        batch['image'] = tf.image.convert_image_dtype(batch['image'], tf.float32)
        batch['image'] = (batch['image'] - 0.5) / 0.5  # tanh range is -1, 1
        batch['label'] = tf.cast(batch['label'], tf.int32)
        return batch
    
    
    STEPS_PER_EPOCH = len(mnist_data["train"]) // BATCH_SIZE
    
    train_dl = DataLoader(mnist_train) \
                     .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                     .cache() \
                     .shuffle(5000) \
                     .batch(BATCH_SIZE, drop_remainder=True) \
                     .to_jax() \
                     .shard() \
                     .prefetch_to_devices()    
    
    STEPS_PER_EPOCH_TEST = len(mnist_data["test"]) // BATCH_SIZE
    
    test_dl = DataLoader(mnist_data["test"]) \
                     .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                     .cache() \
                     .batch(BATCH_SIZE) \
                     .to_jax() \
                     .shard() \
                     .prefetch_to_devices()

    PRNGKey = jnp.ndarray
    
    @partial(jax.pmap, static_broadcasted_argnums=(1, 2))
    def create_state(rng, model_cls, input_shape): 
        r"""Create the training state given a model class. """ 

        model = model_cls()   

        tx = optax.adam(LEARNING_RATE, b1=0.5, b2=0.999) # defining the optimizer
        variables = model.init(rng, jnp.ones(input_shape))   # initializing the model

        state = TrainState.create(apply_fn=model.apply, 
                                  tx=tx, 
                                  params=variables['params'])
        
        return state    
    
    @partial(jax.pmap, axis_name='num_devices')
    def training_step(model_state: TrainState,
                      images: jnp.ndarray,
                      labels: jnp.ndarray):

        def loss_fn(params):
            logits = model_state.apply_fn({'params': params}, images)
            labels_one_hot = jax.nn.one_hot(labels, 10)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot))

            return loss
    
        # Get grad function
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(model_state.params)

        # Average across the devices.
        grads = jax.lax.pmean(grads, axis_name='num_devices')
        loss = jax.lax.pmean(loss, axis_name='num_devices')

        # Update the Generator through gradient descent.
        new_model_state = model_state.apply_gradients(grads=grads)

        return new_model_state, loss
    
    @partial(jax.pmap, axis_name='num_devices')
    def evaluation_step(model_state: TrainState,
                        images: jnp.ndarray,
                        labels: jnp.ndarray):
        
        logits = model_state.apply_fn({'params': model_state.params}, images)
        prediction = jnp.argmax(logits, axis=-1)
        accuracy = jax.lax.pmean(jnp.mean(prediction == labels), axis_name='num_devices')
        
        return accuracy
    
    key = jax.random.PRNGKey(seed=SEED)
    key, sub_key = jax.random.split(key)
    replicated_key = jax_utils.replicate(sub_key)

    model_state = create_state(replicated_key, 
                               CNN,
                               (PER_DEVICE_BATCH_SIZE, *next(iter(train_dl))["image"].shape[2:]))
    
    
    
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epoch...",
                        position=0, leave=True):
        accuracy_epoch = []
        loss_epoch = []
        with tqdm(total=STEPS_PER_EPOCH, desc="Training...",
                    leave=False) as progress_bar_train:                
            for step, batch_train in enumerate(train_dl): 

                model_state, loss = training_step(model_state,
                                                batch_train["image"],
                                                batch_train["label"])

                loss_epoch.append(jax_utils.unreplicate(loss))
                progress_bar_train.update(1)

        with tqdm(total=STEPS_PER_EPOCH_TEST, desc="Evaluation...",
                leave=False) as progress_bar_evaluation:
            for step, batch_test in enumerate(test_dl):
                # get pred
                accuracy = evaluation_step(model_state,
                                            batch_test["image"],
                                            batch_test["label"])

                accuracy_epoch.append(jax_utils.unreplicate(accuracy))
                progress_bar_evaluation.update(1)

        
        message = f"Epoch: {epoch: <2} | "
        message += f"Loss: {sum(loss_epoch)/len(loss_epoch):.4f} | "
        message += f"Accuracy: {sum(accuracy_epoch)/len(accuracy_epoch):.4f} | "
        progress_bar_train.write(message)
