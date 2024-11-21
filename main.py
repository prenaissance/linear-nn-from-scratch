import gzip
import os
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

import lib.nn as nn
import lib.nn.functional as F
from lib.nn.loss import jacobian_loss
from lib.nn.optim import GDOptimizer


def download_mnist_dataset():
    if os.path.exists("data"):
        print("Dataset already downloaded")
        return

    os.makedirs("data")

    train_images_url = "https://github.com/mkolod/MNIST/raw/refs/heads/master/train-images-idx3-ubyte.gz"
    train_labels_url = "https://github.com/mkolod/MNIST/raw/refs/heads/master/train-labels-idx1-ubyte.gz"
    test_images_url = "https://github.com/mkolod/MNIST/raw/refs/heads/master/t10k-images-idx3-ubyte.gz"
    test_labels_url = "https://github.com/mkolod/MNIST/raw/refs/heads/master/t10k-labels-idx1-ubyte.gz"

    output_dir = "data"

    for url in [train_images_url, train_labels_url, test_images_url, test_labels_url]:
        filename = url.split("/")[-1]
        output_path = os.path.join(output_dir, filename)
        urllib.request.urlretrieve(url, output_path)

        with gzip.open(output_path, "rb") as f:
            with open(output_path.replace(".gz", ""), "wb") as out:
                out.write(f.read())

        os.remove(output_path)


def get_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    download_mnist_dataset()
    mndata = MNIST("data")
    _images, _labels = mndata.load_training()
    images = np.array(_images) / 255.0
    labels = np.array(_labels)

    _test_images, _test_labels = mndata.load_testing()
    test_images = np.array(_test_images) / 255.0

    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.Sigmoid(),
        nn.Linear(128, 10),
        nn.Softmax(),
    )
    print(list(model.parameters()))

    loss_fn = jacobian_loss
    optimizer = GDOptimizer(model.parameters(), lr=0.01)
    # Initial pass
    outputs = model(test_images.T)
    predictions = np.argmax(outputs, axis=0)
    accuracy = get_accuracy(_test_labels, predictions)
    print(f"Initial Accuracy: {accuracy}")

    for epoch in range(40):
        batch_size = 10_000
        for i in range(0, len(images), batch_size):
            images_batch = images[i : i + batch_size, :]
            labels_batch = labels[i : i + batch_size]

            images_batch = images_batch.T
            labels_batch = F.one_hot_encode(labels_batch, 10)

            optimizer.zero_grad()

            outputs = model(images_batch)
            loss_gradient = loss_fn(labels_batch, outputs)
            model.backward(outputs, loss_gradient)
            # print(model.layers[0].bias.grad)

            optimizer.step()

        outputs = model(test_images.T)
        predictions = np.argmax(outputs, axis=0)
        accuracy = get_accuracy(_test_labels, predictions)
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")
    print("Training complete")

    # Plotting
    random_samples = np.random.randint(0, len(_test_labels), 9)
    fig, axs = plt.subplots(3, 3)
    for i, sample_idx in enumerate(random_samples):
        ax = axs[i // 3, i % 3]
        ax.imshow(np.array(_test_images[sample_idx]).reshape(28, 28), cmap="gray")
        ax.set_title(f"Predicted: {predictions[sample_idx]}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
