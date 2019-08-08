import numpy as np

import tensorflow as tf
import scipy
import scipy.io

import os, subprocess
import timeit

from collections import deque
from sklearn.metrics import confusion_matrix

Src = os.path.dirname(os.path.abspath(__file__))  # src directory
Root = os.path.dirname(Src) + "/"  # root directory
Src = Src + "/"
Data = os.path.join(Root, "data") + "/"
Models = os.path.join(Root, "models") + "/"
Results = os.path.join(Root, "results") + "/"

############################
# Neural Network Functions #
############################

# Convolution Layer
def conv(
    x, filter_size, num_filters, stride, name, padding="SAME", trainable=True, relu=True
):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(
        x, W, strides=[1, stride, stride, 1], padding=padding
    )

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable(
            "W",
            shape=[filter_size, filter_size, input_channels, num_filters],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            collections=["variables"],
        )
        biases = tf.get_variable(
            "b",
            shape=[num_filters],
            trainable=trainable,
            initializer=tf.zeros_initializer(),
        )
        conv = convolve(x, weights)

        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


# Rectangular Convolution Layer
def conv_rect(
    x, filter_size, num_filters, stride, name, padding="SAME", trainable=True, relu=True
):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(
        x, W, strides=[1, stride, stride, 1], padding=padding
    )

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable(
            "W",
            shape=[filter_size[0], filter_size[1], input_channels, num_filters],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            collections=["variables"],
        )
        biases = tf.get_variable(
            "b",
            shape=[num_filters],
            trainable=trainable,
            initializer=tf.zeros_initializer(),
        )
        conv = convolve(x, weights)

        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


# Deconvolution layer
def deconv(
    x, filter_size, num_filters, stride, name, padding="SAME", relu=True, trainable=True
):
    activation = None
    if relu:
        activation = tf.nn.relu
    return tf.layers.conv2d_transpose(
        x,
        num_filters,
        filter_size,
        stride,
        padding=padding,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        activation=activation,
        name=name,
        trainable=trainable,
    )


# Fully Connected Layer
def fc(x, num_out, name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        weights = tf.get_variable(
            "W",
            shape=[num_in, num_out],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            collections=["variables"],
        )
        biases = tf.get_variable(
            "b", [num_out], initializer=tf.zeros_initializer(), trainable=trainable
        )
        x = tf.matmul(x, weights) + biases
        if relu:
            x = tf.nn.relu(x)
    return x


# Local Response Normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(
        x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name
    )


def max_pool(x, filter_size, stride, name=None, padding="SAME"):
    return tf.nn.max_pool(
        x,
        ksize=[1, filter_size, filter_size, 1],
        strides=[1, stride, stride, 1],
        padding=padding,
        name=name,
    )


def max_pool_rect(x, filter_size, stride, name=None, padding="SAME"):
    return tf.nn.max_pool(
        x,
        ksize=[1, filter_size[0], filter_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        padding=padding,
        name=name,
    )


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError(
            "number of features({}) is not "
            "a multiple of num_units({})".format(num_channels, num_units)
        )
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output


def iou(gt, pred, seg):
    gt_seg = create_seg(gt, seg)
    pred_seg = create_seg(pred, seg)
    overlap = np.minimum(gt_seg, pred_seg)
    return 2 * np.sum(overlap) / (np.sum(gt_seg) + np.sum(pred_seg))


############################
# OS Functions #
############################


def list_dir(dir, ext, return_name=False):
    ext = "." + ext.lower()
    if return_name:
        return sorted(
            [
                (file[: -len(ext)], dir + file)
                for file in os.listdir(dir)
                if file[-len(ext) :].lower() == ext
            ]
        )
    else:
        return sorted(
            [dir + file for file in os.listdir(dir) if file[-len(ext) :].lower() == ext]
        )


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_name(path):
    return ".".join(path.split("/")[-1].split(".")[:-1])


def remove(path):
    if not os.path.exists(path):
        return
    elif os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)


def shell(cmd, wait=True, ignore_error=True):
    if type(cmd) != str:
        cmd = " ".join(cmd)
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if not wait:
        return process
    out, err = process.communicate()
    if err and not ignore_error:
        print(err.decode("UTF-8"))
        raise RuntimeError("Error in command line call")
    return out.decode("UTF-8"), err.decode("UTF-8") if err else None
