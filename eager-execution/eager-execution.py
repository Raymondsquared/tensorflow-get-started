from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe


# Configure imports and eager execution
tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Download the dataset

# From this view of the dataset, we see the following:

# 1. The first line is a header containing information about the dataset:
# 2. There are 120 total examples. Each example has four features and one of
# three possible label names.
# 3. Subsequent rows are data records, one example per line, where:
# 4. The first four fields are features: these are characteristics of an example.
# Here, the fields hold float numbers representing flower measurements.
# 5. The last column is the label: this is the value we want to predict.
# For this dataset, it's an integer value of 0, 1, or 2 that corresponds to a flower name.

# Each label is associated with string name (for example, "setosa"),
# but machine learning typically relies on numeric values.
# The label numbers are mapped to a named representation, such as:

# 0: Iris setosa
# 1: Iris versicolor
# 2: Iris virginica

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


# Parse the dataset
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


# Create the training tf.data.Dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])


# Create a model using Keras

# The ideal number of hidden layers and neurons depends on the problem and the dataset.
# Like many aspects of machine learning, picking the best shape of the neural network
# requires a mixture of knowledge and experimentation.
#
# As a rule of thumb, increasing the number of hidden layers and neurons typically
# creates a more powerful model, which requires more data to train effectively.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            10,
            activation="relu",
            input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ])


# Define the loss and gradient function
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


# Create an optimizer

# This is a hyperparameter that you'll commonly adjust to achieve better results.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
