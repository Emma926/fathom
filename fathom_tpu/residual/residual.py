from math import sqrt
import numpy as np
import tensorflow as tf
from collections import namedtuple
from fathom.imagenet.imagenet_cp import *
from fathom.nn import *
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost.
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1, momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
      training=True, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs

def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.

  Returns:
    A tensor of size [batch, channels, height_out, width_out] with the
      input either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                  [pad_beg, pad_end], [pad_beg, pad_end]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format='channels_first')

def model_fn(features, labels, mode, params):
      del params  # Unused.

      if mode != tf.estimator.ModeKeys.TRAIN:
        raise RuntimeError("mode {} is not supported yet".format(mode))

      LayerBlock = namedtuple(
        'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
      blocks = [
        LayerBlock(3, 256, 64),
        LayerBlock(4, 512, 128),
        LayerBlock(6, 1024, 256),
        LayerBlock(3, 2048, 512)
      ]

      features = tf.transpose(features, [0, 3, 1, 2])
      # %%
      # First convolution expands to 64 channels and downsamples

      net = conv2d_fixed_padding(inputs=features, filters=64, kernel_size=7, strides = 2)
      #net = tf.layers.conv2d(inputs=features, filters=64, kernel_size=7, strides = 2, padding='VALID')
      net = tf.identity(net, 'initial_conv')
      # %%
      # Max pool and downsampling
      net = tf.layers.max_pooling2d(
        inputs=net, pool_size=3, strides=2, padding='SAME')
      net = tf.identity(net, 'initial_max_pool')

      # %%
      for block_i, block in enumerate(blocks):
        filters_out = block.num_filters

        net = batch_norm_relu(net)
        shortcut = conv2d_fixed_padding(inputs=net, filters=filters_out, kernel_size=1, strides=2)
        #shortcut = tf.layers.conv2d(
        #  inputs=net, filters=block.num_filters, kernel_size=1, strides=2,
        #  padding='VALID', data_format='channels_first')
        net = conv2d_fixed_padding(
            inputs=net, filters=block.bottleneck_size, kernel_size=1, strides=1)
        #net = tf.layers.conv2d(
        #  inputs=net, filters=block.bottleneck_size, kernel_size=1, strides=1,
        #  padding='VALID', data_format='channels_first')
        net = batch_norm_relu(net)
        net = conv2d_fixed_padding(
          inputs=net, filters=block.bottleneck_size, kernel_size=3, strides=2)
        #net = fixed_padding(net, 3)
        #net = tf.layers.conv2d(
        #  inputs=net, filters=block.bottleneck_size, kernel_size=3, strides=2,
        #  padding='SAME', data_format='channels_first')

        net = batch_norm_relu(net)
        net = conv2d_fixed_padding(
          inputs=net, filters=4 * block.bottleneck_size, kernel_size=1, strides=1)
        #net = tf.layers.conv2d(
        #  inputs=net, filters=block.num_filters, kernel_size=1, strides=1,
        #  padding='VALID', data_format='channels_first')

        net = tf.identity(net)

        for repeat_i in range(1,block.num_repeats):
          shortcut = net
          net = batch_norm_relu(net)
          net = conv2d_fixed_padding(
            inputs=net, filters=block.bottleneck_size, kernel_size=1, strides=1)
          net = batch_norm_relu(net)
          net = conv2d_fixed_padding(
            inputs=net, filters=block.bottleneck_size, kernel_size=3, strides=1)

          net = batch_norm_relu(net)
          net = conv2d_fixed_padding(
            inputs=net, filters=4 * block.bottleneck_size, kernel_size=1, strides=1)

          net = net + shortcut
          net = tf.identity(net)

      # %%
      net = batch_norm_relu(net)
      net = tf.layers.average_pooling2d(net,
          pool_size=net.get_shape().as_list()[2],
          strides=1, padding='VALID')
      net = tf.identity(net, 'final_avg_pool')
      net = tf.reshape(
          net,
          [-1, net.get_shape().as_list()[1] *
            net.get_shape().as_list()[2] *
            net.get_shape().as_list()[3]])

      net = tf.layers.dense(inputs=net, units=FLAGS.num_classes)
      logits = tf.identity(net, 'final_dense')


      # Calculating the loss.
      loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)

      # Configuring the optimization algorithm.
      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate, tf.train.get_global_step(), 25000, 0.97)
      if FLAGS.use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(
          tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
      else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op)


  
def main(unused_argv):
  assert len(unused_argv) == 1, (
      "Unrecognized command line arguments: %s" % unused_argv[1:])

  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=FLAGS.save_checkpoints_secs,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      #tpu_config=tpu_config.TPUConfig(5, FLAGS.num_shards, per_host_input_for_training = True),
      tpu_config=tpu_config.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      config=run_config)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

  total = time.time() - start
  print("Total time: " + str(total))

if __name__ == "__main__":
  tf.app.run()
