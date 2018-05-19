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
