from pathlib import Path
import random
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import numpy as np


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


    # original image shape -- (num_of_img, 28, 28)

    # flatten the image
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Convert pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Convert numpy arrays into JAX arrays
    return (
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        jnp.asarray(x_test),
        jnp.asarray(y_test)
    )



def initalize_layer(key, input_size, output_size):
    weights = (
        jax.random.normal(
            key,
            shape=(input_size, output_size)
        )
        * jnp.sqrt(2.0 / input_size)
    )

    biases = jnp.zeros(
        shape=(output_size,),
        dtype=jnp.float32,
    )

    return {
        "weights": weights,
        "biases": biases
    }


def initalize_network(key):
    key1, key2 = jax.random.split(key)

    return {
        "layer1": initalize_layer(
            key1,
            input_size=784,
            output_size=256
        ),
        "layer2": initalize_layer(
            key2,
            input_size=256,
            output_size=10,
        ),
    }


def relu(x):
    return jnp.maximum(x, 0.0)

def forward(params, x):
    layer1 = params["layer1"]
    layer2 = params["layer2"]

    # First linear layer
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



def cross_entropy_loss(params, x, targets):
    logits = forward(params, x)

    # z_i = -log(sum(exp(z_j)))

    log_probabilities =  logits - jax.nn.logsumexp(logits, axis=1, keepdims=True)

    # Select the log probabilties corresponding 
    # to the correct class for every sample
    correct_log_probabilites = jnp.take_along_axis(
        log_probabilities,
        targets[:, None],
        axis=1,
    ).squeeze(axis=1)

    return -jnp.mean(correct_log_probabilites)



@jax.jit
def predict(params, x):
    logits = forward(params, x)

    return jnp.argmax(logits, axis=1)


def calcualte_accuracy(
    params,
    x,
    targets,
    batch_size=1000
):
    if x.shape[0] != targets.shape[0]:
        raise ValueError(
            "Images and targets must contain the same number of samples, "
            f"but got {x.shape[0]} images and {targets.shape[0]} targets."
        )

    correct_predictions = 0
    num_of_samples = x.shape[0]

    for start in range(0, num_of_samples, batch_size):

        end = min(start + batch_size, num_of_samples)

        predictions = predict(
            params,
            x[start:end]
        )
        
        correct_predictions += int(
            jnp.sum(predictions == targets[start:end])
        )

    return correct_predictions / num_of_samples


@jax.jit
def train_step(
    params,
    x_batch,
    y_batch,
    learning_rate
):
    loss, grad = jax.value_and_grad(
        cross_entropy_loss
    )(
        params,
        x_batch,
        y_batch
    )

    updated_params = tree_map(
        lambda param, grad: param - learning_rate * grad,
        params,
        grad
    )

    return updated_params, loss



def train(
    params,
    x_train,
    y_train,
    x_test,
    y_test,
    key,
    epochs=10,
    batch_size=128,
    learning_rate=0.1
):
    number_of_samples = x_train.shape[0]

    number_of_batches = number_of_samples // batch_size
    samples_used = number_of_batches * batch_size

    for epoch in range(1, epochs+1):
        key, shuffle_key = jax.random.split(key)

        shuffled_indices = jax.random.permutation(
            shuffle_key,
            number_of_samples
        )

        shuffled_indices = shuffled_indices[:samples_used]
        
        epoch_losses = []

        for batch_index in range(number_of_batches):
            start = batch_index * batch_size
            end = start + batch_size

            batch_indices = shuffled_indices[start: end]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            params, loss = train_step(
                params,
                x_batch,
                y_batch,
                learning_rate
            )

            epoch_losses.append(loss)

        average_loss = jnp.mean(
            jnp.stack(epoch_losses)
        )

        train_accuracy = calcualte_accuracy(
            params,
            x_train,
            y_train
        )

        test_accuracy = calcualte_accuracy(
            params,
            x_test,
            y_test
        )

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Loss: {float(average_loss):.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test accuracy: {test_accuracy * 100:.2f}%"
        )

    return params

    

def main():
    x_train, y_train, x_test, y_test = load_mnist()

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    key = jax.random.key(42)

    parameter_key, training_key = jax.random.split(key)

    params = initalize_network(parameter_key)

    inital_acc = calcualte_accuracy(
        params,
        x_test,
        y_test
    )

    print(
        f"\nInitial test accuracy: "
        f"{inital_acc * 100:.2f}%\n"
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
        learning_rate=0.1
    )



if __name__ == "__main__":
    main()




