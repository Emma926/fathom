from math import sqrt
import numpy as np
from fathom.imagenet.imagenet_cp import *
from fathom.nn import *
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time

def model_fn(features, labels, mode, params):
  del params  # Unused.

  if mode != tf.estimator.ModeKeys.TRAIN:
    raise RuntimeError("mode {} is not supported yet".format(mode))

  # Convolution and pooling layers.
  input_layer = features
      # block 1 -- outputs 112x112x64
  conv1_1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=3,
      activation=tf.nn.relu,
      strides=1)
  conv1_2 = tf.layers.conv2d(
      inputs=conv1_1,
      filters=64,
      kernel_size=3,
      activation=tf.nn.relu,
      strides=1)
  pool1 = tf.layers.max_pooling2d(
      inputs=conv1_2,
      pool_size=2,
      strides=2)

  conv2_1 = tf.layers.conv2d( 
      inputs=pool1,
      filters=128,
      activation=tf.nn.relu,
      kernel_size=3)
  conv2_2 = tf.layers.conv2d(
      inputs=conv2_1,
      filters=128,
      activation=tf.nn.relu,
      kernel_size=3)
  pool2 = tf.layers.max_pooling2d(
      inputs=conv2_2,
      pool_size=2,
      strides=2)

  conv3_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      activation=tf.nn.relu,
      kernel_size=3)
  conv3_2 = tf.layers.conv2d(
      inputs=conv3_1, 
      filters=256,
      activation=tf.nn.relu,
      kernel_size=3)
  pool3 = tf.layers.max_pooling2d(
      inputs=conv3_2,
      pool_size=2,
      strides=2)

  conv4_1 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      activation=tf.nn.relu,
      kernel_size=3)
  conv4_2 = tf.layers.conv2d(
      inputs=conv4_1,
      filters=512,
      activation=tf.nn.relu,
      kernel_size=3)
  pool4 = tf.layers.max_pooling2d(
      inputs=conv4_2,
      pool_size=2,
      strides=2)

  conv5_1 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      activation=tf.nn.relu,
      kernel_size=3)
  conv5_2 = tf.layers.conv2d(
      inputs=conv5_1,
      filters=512,
      activation=tf.nn.relu,
      kernel_size=3)
  pool5 = tf.layers.max_pooling2d(
      inputs=conv5_2,
      pool_size=2,
      strides=2)


  shp = pool5.get_shape().as_list() # pool2 if shrunk
  flattened_shape = shp[1] * shp[2] * shp[3]
  resh1 = tf.reshape(pool5, [shp[0], flattened_shape], name="resh1")

  # Fully connected layers with dropout.
  fc6 = tf.layers.dense(
      inputs=resh1,
      units=4096,
      activation=tf.nn.relu)
  drp6 = tf.layers.dropout(
      inputs=fc6,
      rate=(1-FLAGS.dropout_keep_prob))
  fc7 = tf.layers.dense(
      inputs=drp6,
      units=4096,
      activation=tf.nn.relu)
  drp7 = tf.layers.dropout(
      inputs=fc7,
      rate=(1-FLAGS.dropout_keep_prob))
  fc8 = tf.layers.dense(
      inputs=drp7,
      units=FLAGS.num_classes)


  # Calculating the loss.
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=fc8)

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
