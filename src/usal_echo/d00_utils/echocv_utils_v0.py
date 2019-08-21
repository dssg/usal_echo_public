#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import subprocess

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
    x,
    filter_size,
    num_filters,
    stride,
    weight_decay,
    name,
    padding="SAME",
    groups=1,
    trainable=True,
    relu=True,
):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(
        x, W, strides=[1, stride, stride, 1], padding=padding
    )

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable(
            "W",
            shape=[filter_size, filter_size, input_channels // groups, num_filters],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            # regularizer=regularizer,
            collections=["variables"],
        )
        biases = tf.get_variable(
            "b",
            shape=[num_filters],
            trainable=trainable,
            initializer=tf.zeros_initializer(),
        )

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [
                convolve(i, k) for i, k in zip(input_groups, weight_groups)
            ]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


def conv_rect(
    x,
    filter_size,
    num_filters,
    stride,
    weight_decay,
    name,
    padding="SAME",
    groups=1,
    trainable=True,
    relu=True,
):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(
        x, W, strides=[1, stride, stride, 1], padding=padding
    )

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable(
            "W",
            shape=[
                filter_size[0],
                filter_size[1],
                input_channels // groups,
                num_filters,
            ],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            # regularizer=regularizer,
            collections=["variables"],
        )
        biases = tf.get_variable(
            "b",
            shape=[num_filters],
            trainable=trainable,
            initializer=tf.zeros_initializer(),
        )

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [
                convolve(i, k) for i, k in zip(input_groups, weight_groups)
            ]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


def deconv(
    x, filter_size, num_filters, stride, weight_decay, name, padding="SAME", relu=True
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
    )


# Fully Connected Layer
def fc(x, num_out, weight_decay, name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable(
            "W",
            shape=[num_in, num_out],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable,
            # regularizer=regularizer,
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
# Resnet Functions #
############################

_BATCH_NORM_DECAY = 0.95
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == "channels_first" else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=is_training,
    )
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == "channels_first":
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        )
    else:
        padded_inputs = tf.pad(
            inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        )
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=("SAME" if strides == 1 else "VALID"),
        use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0),
        data_format=data_format,
    )


def dilated_conv2d_fixed_padding(
    inputs, num_filters, kernel_size, rate, data_format, name
):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    with tf.variable_scope(name):
        input_channels = int(inputs.get_shape()[-1])
        filter_size = kernel_size
        weights = tf.get_variable(
            "W",
            shape=[filter_size, filter_size, input_channels, num_filters],
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=["variables"],
        )
        conv_out = tf.nn.atrous_conv2d(
            value=inputs, filters=weights, rate=rate, padding="SAME"
        )
    return conv_out


def building_block(
    inputs, filters, is_training, projection_shortcut, strides, data_format
):
    """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format,
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        data_format=data_format,
    )

    return inputs + shortcut


def building_block_dilated(
    inputs,
    num_filters,
    is_training,
    projection_shortcut,
    rate,
    data_format,
    name,
    skip=1,
):
    """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = dilated_conv2d_fixed_padding(
        inputs=inputs,
        num_filters=num_filters,
        kernel_size=3,
        rate=rate,
        data_format=data_format,
        name=name + "_1",
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = dilated_conv2d_fixed_padding(
        inputs=inputs,
        num_filters=num_filters,
        kernel_size=3,
        rate=rate,
        data_format=data_format,
        name=name + "_2",
    )
    if skip:
        return inputs + shortcut
    else:
        return inputs


def bottleneck_block(
    inputs, filters, is_training, projection_shortcut, strides, data_format
):
    """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        data_format=data_format,
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        data_format=data_format,
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        data_format=data_format,
    )

    return inputs + shortcut


def bottleneck_block_dilated(
    inputs, num_filters, is_training, projection_shortcut, rate, data_format, skip=1
):
    """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = dilated_conv2d_fixed_padding(
        inputs=inputs,
        num_filters=num_filters,
        kernel_size=1,
        rate=rate,
        data_format=data_format,
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = dilated_conv2d_fixed_padding(
        inputs=inputs,
        num_filters=num_filters,
        kernel_size=3,
        rate=rate,
        data_format=data_format,
    )

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = dilated_conv2d_fixed_padding(
        inputs=inputs,
        num_filters=4 * num_filters,
        kernel_size=1,
        rate=rate,
        data_format=data_format,
    )
    if skip:
        return inputs + shortcut
    else:
        return inputs


def block_layer(
    inputs, filters, block_fn, blocks, strides, is_training, name, data_format
):
    """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs,
            filters=filters_out,
            kernel_size=1,
            strides=strides,
            data_format=data_format,
        )

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(
        inputs, filters, is_training, projection_shortcut, strides, data_format
    )

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)


def block_layer_dilated(
    inputs, num_filters, block_fn, blocks, rate, is_training, name, data_format, skip=1
):
    """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * num_filters if block_fn is bottleneck_block else num_filters
    inputs = tf.transpose(inputs, [0, 2, 3, 1])

    def projection_shortcut(inputs):
        return dilated_conv2d_fixed_padding(
            inputs=inputs,
            num_filters=filters_out,
            kernel_size=1,
            rate=rate,
            data_format=data_format,
            name=name,
        )

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(
        inputs,
        num_filters,
        is_training,
        projection_shortcut,
        rate,
        data_format,
        name,
        skip,
    )

    for _ in range(1, blocks):
        inputs = block_fn(
            inputs,
            num_filters,
            is_training,
            None,
            rate,
            data_format,
            name + "_" + str(_),
            skip,
        )

    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    return tf.identity(inputs, name)


def imagenet_resnet_v2_generator(block_fn, layers, num_classes, data_format=None):
    """Generator for ImageNet ResNet v2 models.
  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
    if data_format is None:
        data_format = (
            "channels_first" if tf.test.is_built_with_cuda() else "channels_last"
        )

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == "channels_first":
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format
        )
        inputs = tf.identity(inputs, "initial_conv")
        inputs = tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=3,
            strides=2,
            padding="SAME",
            data_format=data_format,
        )
        inputs = tf.identity(inputs, "initial_max_pool")

        inputs = block_layer(
            inputs=inputs,
            filters=64,
            block_fn=block_fn,
            blocks=layers[0],
            strides=1,
            is_training=is_training,
            name="block_layer1",
            data_format=data_format,
        )
        inputs = block_layer(
            inputs=inputs,
            filters=128,
            block_fn=block_fn,
            blocks=layers[1],
            strides=2,
            is_training=is_training,
            name="block_layer2",
            data_format=data_format,
        )
        inputs = block_layer(
            inputs=inputs,
            filters=256,
            block_fn=block_fn,
            blocks=layers[2],
            strides=2,
            is_training=is_training,
            name="block_layer3",
            data_format=data_format,
        )
        inputs = block_layer(
            inputs=inputs,
            filters=512,
            block_fn=block_fn,
            blocks=layers[3],
            strides=2,
            is_training=is_training,
            name="block_layer4",
            data_format=data_format,
        )

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs,
            pool_size=7,
            strides=1,
            padding="VALID",
            data_format=data_format,
        )
        inputs = tf.identity(inputs, "final_avg_pool")
        inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, "final_dense")
        return inputs

    return model


def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {"block": building_block, "layers": [2, 2, 2, 2]},
        34: {"block": building_block, "layers": [3, 4, 6, 3]},
        50: {"block": bottleneck_block, "layers": [3, 4, 6, 3]},
        101: {"block": bottleneck_block, "layers": [3, 4, 23, 3]},
        152: {"block": bottleneck_block, "layers": [3, 8, 36, 3]},
        200: {"block": bottleneck_block, "layers": [3, 24, 36, 3]},
    }

    if resnet_size not in model_params:
        raise ValueError("Not a valid resnet_size:", resnet_size)

    params = model_params[resnet_size]
    return imagenet_resnet_v2_generator(
        params["block"], params["layers"], num_classes, data_format
    )