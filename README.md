# Neural Network

Neural Network is an mathematical function which takes input and returns output

$$
out = NN(input)
$$

MNIST is an dataset of 1-9 digits handwritten images.

We will make a neural network to solve and predict these digits from images.

For that, we will create 2 layers which will contain Weights and Biases. 


We will use ReLU activation function, it turns all negative numbers to 0 and positive number to as it is.
This will help us filter out the Neurons as possibly those with negative value are not contributing to recognize
the patterns. And gradients of 0 will be 0 so no updates on those neurons. 


Let's define Layers

Let W1,b1, W2, b2 are Matrices 

$$
input_dim = 784
hidden_dim = 784
output_dim = 10
$$


Layer 1:
$$
W1 belongs to R^(input_dim x hidden_dim)
b1 belongs to R^(hidden_dim)
$$

$$
W2 belongs to R^(hidden_dim x hidden_dim)
b2 belongs to R^(output_dim)
$$


Forward paas computation:

Layer 1:
$$
Z1 = inputs @ W1 + b1
Z1 = ReLU(Z1) = Max(0, Z1)
$$

Layer 2:
$$
Z2 = Z1 @ W2 + b2
Z2 = ReLU(Z2) = Max(0, Z2)
$$


This is How we do forward paas, to train model, we need to compute gradients through backward paas

you must have studied about Chain Rule in High School.

$$
dx/ds = dx/du * du/ds
$$


Similarly to compute gradient from dLoss/dW1, we can use chain rule by tracing in backward direction.


Loss Function:

Here we are using cross entropy loss, which is just lagorithm version of Mean squared error




# Build a Neural Network from Scratch with JAX

JAX brings NumPy-style array programming together with automatic
differentiation, just-in-time (JIT) compilation, and hardware acceleration.
It runs on CPUs, GPUs, and TPUs.

Unlike PyTorch, JAX does not provide a high-level neural-network API in its
core package. Instead, it gives us composable transformations such as
`jax.jit` and `jax.value_and_grad`. This lower-level approach makes JAX a
great way to understand what happens inside a training loop.

JIT stands for **just-in-time compilation**. Decorating a function with
`jax.jit` tells JAX to compile it into optimized machine code. The first call
performs the tracing and compilation; later calls with compatible input shapes
and data types can reuse the compiled program.

## A few important things to know about JAX

### 1. JAX arrays are immutable

Unlike NumPy arrays, JAX arrays cannot be modified in place. Direct item
assignment therefore raises an error:

```python
key = jax.random.key(0)
a = jax.random.normal(key, shape=(2, 25))

a[0, 1] = 30.0  # TypeError: JAX arrays are immutable
```

Use JAX's indexed-update syntax to create an updated array:

```python
updated_a = a.at[0, 1].set(30.0)
```

The original array `a` remains unchanged. JAX also provides indexed
operations such as `.add()`:

```python
updated_a = a.at[0, 1].add(5.0)
```

This functional update style works naturally with transformations such as
`jax.jit`.

### 2. Conditions are not compiled in JAX function.

During JIT compilation, JAX traces the operations performed by a function.
A regular Python `if` statement fails when its condition depends on a runtime
JAX value:

```python
@jax.jit
def choose_value(x):
    if x == 1:  # Error: x is a traced JAX value
        return x + 10
    return x - 10
```

Use `jax.lax.cond` when only one of two branches should run:

```python
@jax.jit
def choose_value(x):
    return jax.lax.cond(
        x == 1,
        lambda value: value + 10,
        lambda value: value - 10,
        x,
    )
```

Both branches must return values with compatible shapes and data types. For
simple element-wise selection, `jnp.where` is often more concise:

```python
@jax.jit
def choose_values(x):
    return jnp.where(x == 1, x + 10, x - 10)
```

## Tutorial 

In this tutorial, we will build and train a two-layer neural network on MNIST
without using a neural-network framework. Familiarity with Python, NumPy, and
basic PyTorch concepts will be helpful.

Our model has the following architecture:

```text
784 input pixels → 256 hidden units → ReLU → 10 output logits
```

## Imports

```python
from pathlib import Path
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
```

## 1. Download and prepare MNIST

MNIST contains 28 × 28 grayscale images of handwritten digits. We flatten
each image into a vector of 784 values, normalize its pixels from `[0, 255]`
to `[0, 1]`, and convert the NumPy arrays to JAX arrays.

```python
MNIST_URL = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/mnist.npz"
)


def load_mnist():
    cache_directory = Path.home() / ".cache" / "mnist"
    cache_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = cache_directory / "mnist.npz"

    if not dataset_path.exists():
        print("Downloading MNIST...")
        urlretrieve(MNIST_URL, dataset_path)
        print("Download complete.")

    with np.load(dataset_path) as dataset:
        x_train = dataset["x_train"]
        y_train = dataset["y_train"]
        x_test = dataset["x_test"]
        y_test = dataset["y_test"]

    # Convert (number_of_images, 28, 28) into
    # (number_of_images, 784).
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Normalize pixel values to [0, 1].
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return (
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        jnp.asarray(x_test),
        jnp.asarray(y_test),
    )
```

The resulting shapes are:

```text
x_train: (60000, 784)
y_train: (60000,)
x_test:  (10000, 784)
y_test:  (10000,)
```

## 2. Initialize a layer

Each dense layer contains a weight matrix and a bias vector. We initialize
the weights with He initialization, which works well with ReLU activations,
and initialize the biases to zero.

JAX handles randomness explicitly. A random key represents the state of the
random-number generator, and independent random operations should receive
independent keys.

```python
def initialize_layer(key, input_size, output_size):
    weights = (
        jax.random.normal(
            key,
            shape=(input_size, output_size),
        )
        * jnp.sqrt(2.0 / input_size)
    )

    biases = jnp.zeros(
        shape=(output_size,),
        dtype=jnp.float32,
    )

    return {
        "weights": weights,
        "biases": biases,
    }
```

## 3. Initialize the network

Our network contains two dense layers:

- The first maps 784 input pixels to 256 hidden features.
- The second maps those 256 features to 10 class logits.

### The mathematics

For a batch of `B` images, let:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `X` | `B × 784` | Input batch |
| `W₁` | `784 × 256` | First-layer weights |
| `b₁` | `256` | First-layer biases |
| `W₂` | `256 × 10` | Second-layer weights |
| `b₂` | `10` | Second-layer biases |

The first dense layer computes:

```text
Z₁ = XW₁ + b₁
```

We then apply ReLU element by element:

```text
H = ReLU(Z₁) = max(0, Z₁)
```

The second dense layer converts the hidden features into class logits:

```text
Z₂ = HW₂ + b₂
```

Therefore, the complete network is:

```text
Z₂ = ReLU(XW₁ + b₁)W₂ + b₂
```

Here, `Z₁` has shape `B × 256`, while `Z₂` has shape `B × 10`. JAX
broadcasts each bias vector across the batch, so `b₁` is added to every row
of `XW₁`, and `b₂` is added to every row of `HW₂`. In the code, `@` performs
matrix multiplication and `+` performs the broadcasted bias addition.

We split the key so that each layer receives its own random key.

```python
def initialize_network(key):
    key1, key2 = jax.random.split(key)

    return {
        "layer1": initialize_layer(
            key1,
            input_size=784,
            output_size=256,
        ),
        "layer2": initialize_layer(
            key2,
            input_size=256,
            output_size=10,
        ),
    }
```

The returned nested dictionary is a JAX **PyTree**. JAX transformations can
operate on every array contained in it while preserving the nested
structure.

## 4. Define the forward pass

The first layer is followed by a ReLU activation. The second layer produces
unnormalized class scores called logits.

```python
def relu(x):
    return jnp.maximum(x, 0.0)


def forward(params, x):
    layer1 = params["layer1"]
    layer2 = params["layer2"]

    z1 = (
        x @ layer1["weights"]
        + layer1["biases"]
    )

    hidden = relu(z1)

    logits = (
        hidden @ layer2["weights"]
        + layer2["biases"]
    )

    return logits
```

For a batch containing `B` images, the input has shape `(B, 784)` and the
returned logits have shape `(B, 10)`.

## 5. Compute cross-entropy loss

For each sample, the log-probability of class `i` is:

```text
log p(i) = z_i - log(sum(exp(z_j)))
```

`jax.nn.logsumexp` performs this calculation in a numerically stable way.
`jnp.take_along_axis` then selects the log-probability of the correct class
from each row.

```python
def cross_entropy_loss(params, x, targets):
    logits = forward(params, x)

    log_probabilities = (
        logits
        - jax.nn.logsumexp(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    correct_log_probabilities = jnp.take_along_axis(
        log_probabilities,
        targets[:, None],
        axis=1,
    ).squeeze(axis=1)

    return -jnp.mean(correct_log_probabilities)
```

If `targets` has shape `(B,)`, then `targets[:, None]` has shape `(B, 1)`.
This lets `take_along_axis` choose one class from each of the `B` rows.
Negating and averaging those selected log-probabilities gives the batch
cross-entropy loss.

## 6. Make predictions and measure accuracy

The predicted class is the index of the largest logit. We compile `predict`
with `jax.jit` so repeated calls with the same input shapes can use optimized
machine code.

Accuracy is computed in batches to avoid evaluating the entire dataset at
once.

```python
@jax.jit
def predict(params, x):
    logits = forward(params, x)
    return jnp.argmax(logits, axis=1)


def calculate_accuracy(
    params,
    x,
    targets,
    batch_size=1000,
):
    if x.shape[0] != targets.shape[0]:
        raise ValueError(
            "Images and targets must contain the same number of samples, "
            f"but got {x.shape[0]} images and {targets.shape[0]} targets."
        )

    correct_predictions = 0
    number_of_samples = x.shape[0]

    for start in range(0, number_of_samples, batch_size):
        end = min(start + batch_size, number_of_samples)

        predictions = predict(
            params,
            x[start:end],
        )

        correct_predictions += int(
            jnp.sum(predictions == targets[start:end])
        )

    return correct_predictions / number_of_samples
```

## 7. Define one training step

`jax.value_and_grad` evaluates the loss and computes its gradient with
respect to the first argument, `params`. Because `params` is a PyTree, the
returned gradients have the same nested structure.

We apply stochastic gradient descent (SGD) to every parameter with
`tree_map`, then return both the updated parameters and the loss.

```python
@jax.jit
def train_step(
    params,
    x_batch,
    y_batch,
    learning_rate,
):
    loss, gradients = jax.value_and_grad(
        cross_entropy_loss
    )(
        params,
        x_batch,
        y_batch,
    )

    updated_params = tree_map(
        lambda param, gradient: (
            param - learning_rate * gradient
        ),
        params,
        gradients,
    )

    return updated_params, loss
```

The function is pure: instead of modifying `params` in place, it returns a
new PyTree containing the updated arrays. This functional style is important
for JAX transformations such as `jit`.

## 8. Build the training loop

At the beginning of each epoch, we generate a random permutation of the
training indices. We keep only complete batches so that every call to the
JIT-compiled `train_step` receives arrays with the same shape.

```python
def train(
    params,
    x_train,
    y_train,
    x_test,
    y_test,
    key,
    epochs=10,
    batch_size=128,
    learning_rate=0.1,
):
    number_of_samples = x_train.shape[0]

    number_of_batches = number_of_samples // batch_size
    samples_used = number_of_batches * batch_size

    for epoch in range(1, epochs + 1):
        key, shuffle_key = jax.random.split(key)

        shuffled_indices = jax.random.permutation(
            shuffle_key,
            number_of_samples,
        )
        shuffled_indices = shuffled_indices[:samples_used]

        epoch_losses = []

        for batch_index in range(number_of_batches):
            start = batch_index * batch_size
            end = start + batch_size

            batch_indices = shuffled_indices[start:end]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            params, loss = train_step(
                params,
                x_batch,
                y_batch,
                learning_rate,
            )

            epoch_losses.append(loss)

        average_loss = jnp.mean(
            jnp.stack(epoch_losses)
        )

        train_accuracy = calculate_accuracy(
            params,
            x_train,
            y_train,
        )

        test_accuracy = calculate_accuracy(
            params,
            x_test,
            y_test,
        )

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Loss: {float(average_loss):.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test accuracy: {test_accuracy * 100:.2f}%"
        )

    return params
```

## 9. Run the program

Finally, load the dataset, create independent keys for initialization and
training, measure the untrained model's accuracy, and start training.

```python
def main():
    x_train, y_train, x_test, y_test = load_mnist()

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    key = jax.random.key(42)
    parameter_key, training_key = jax.random.split(key)

    params = initialize_network(parameter_key)

    initial_accuracy = calculate_accuracy(
        params,
        x_test,
        y_test,
    )

    print(
        f"\nInitial test accuracy: "
        f"{initial_accuracy * 100:.2f}%\n"
    )

    trained_params = train(
        params=params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        key=training_key,
        epochs=10,
        batch_size=128,
        learning_rate=0.1,
    )

    return trained_params


if __name__ == "__main__":
    main()
```


## Result

```text
Initial test accuracy: 9.22%

Epoch  1/10 | Loss: 0.4524 | Train accuracy: 92.10% | Test accuracy: 92.36%
Epoch  2/10 | Loss: 0.2488 | Train accuracy: 94.00% | Test accuracy: 94.04%
Epoch  3/10 | Loss: 0.1993 | Train accuracy: 95.17% | Test accuracy: 95.02%
Epoch  4/10 | Loss: 0.1671 | Train accuracy: 95.80% | Test accuracy: 95.53%
Epoch  5/10 | Loss: 0.1438 | Train accuracy: 96.45% | Test accuracy: 96.03%
Epoch  6/10 | Loss: 0.1261 | Train accuracy: 96.85% | Test accuracy: 96.45%
Epoch  7/10 | Loss: 0.1120 | Train accuracy: 97.14% | Test accuracy: 96.73%
Epoch  8/10 | Loss: 0.1011 | Train accuracy: 97.47% | Test accuracy: 96.92%
Epoch  9/10 | Loss: 0.0923 | Train accuracy: 97.66% | Test accuracy: 96.89%
Epoch 10/10 | Loss: 0.0846 | Train accuracy: 97.98% | Test accuracy: 97.11%
```
