import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # create datastructure for storing images and corresponding labels
    images = []
    labels = []

    # loop through each category as a subdirectory
    for idx in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, str(idx))
        # generate a list of all images in a category and loop through
        for img in os.listdir(path):
            image = cv2.imread(os.path.join(data_dir, str(idx), img))
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            labels.append(idx)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # convolutional layer learning 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # max-pooling layer with 2 x 2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # second convolutional layer learning 32 filters using a 5x5 kernel
        tf.keras.layers.Conv2D(32, (2, 2), activation="relu"),

        # second pooling layer with 2 x 2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # flatten
        tf.keras.layers.Flatten(),

        # hidden layer + 20% dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")


    ])

    # Train neural network
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    main()

# ### Results ### #

# 5x5 kernel, .2 dropout, 3x3 max pooling
    # 333/333 - 2s - loss: 0.0130 - accuracy: 0.9193 - 2s/epoch - 5ms/step

# 5x5 kernel, .5 dropout, 3x3 max pooling
    # 333/333 - 2s - loss: 0.0256 - accuracy: 0.7869 - 2s/epoch - 5ms/step

# 3x3 kernel, .5 dropout, 3x3 max pooling
    # 333/333 - 2s - loss: 0.0282 - accuracy: 0.8045 - 2s/epoch - 5ms/step

# 3x3 kernel, .2 dropout, 3x3 max pooling
    # 333/333 - 2s - loss: 0.0168 - accuracy: 0.8956 - 2s/epoch - 5ms/step

# 3x3 kernel, .2 dropout, 2x2 max pooling
    # 333/333 - 2s - loss: 0.0079 - accuracy: 0.9713 - 2s/epoch - 7ms/step

