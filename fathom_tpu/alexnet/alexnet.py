#!/usr/bin/env python
import tensorflow as tf

from fathom.imagenet.imagenet_cp import *
from fathom.nn import *
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
  conv1 = tf.contrib.layers.conv2d(
      inputs=input_layer,
      num_outputs=64,
      kernel_size=[11, 11],
      stride=4,
      padding="VALID")
  pool1 = tf.contrib.layers.max_pool2d(
      inputs=conv1,
      kernel_size=[3, 3],
      stride=2)
  conv2 = tf.contrib.layers.conv2d(
      inputs=pool1,
      num_outputs=192,
      kernel_size=[5, 5])
  pool2 = tf.contrib.layers.max_pool2d(
      inputs=conv2,
      kernel_size=[3, 3],
      stride=2)
  conv3 = tf.contrib.layers.conv2d(
      inputs=pool2,
      num_outputs=384,
      kernel_size=[3, 3])
  conv4 = tf.contrib.layers.conv2d(
      inputs=conv3,
      num_outputs=384,
      kernel_size=[3, 3])
  conv5 = tf.contrib.layers.conv2d(
      inputs=conv4,
      num_outputs=256,
      kernel_size=[3, 3])
  pool5 = tf.contrib.layers.max_pool2d(
      inputs=conv5,
      kernel_size=[3, 3],
      stride=2)
  reshaped_pool5 = tf.reshape(pool5, [-1, 5 * 5 * 256])

  # Fully connected layers with dropout.
  fc6 = tf.contrib.layers.fully_connected(
      inputs=reshaped_pool5,
      num_outputs=4096)
  drp6 = tf.contrib.layers.dropout(
      inputs=fc6,
      keep_prob=FLAGS.dropout_keep_prob)
  fc7 = tf.contrib.layers.fully_connected(
      inputs=drp6,
      num_outputs=4096)
  drp7 = tf.contrib.layers.dropout(
      inputs=fc7,
      keep_prob=FLAGS.dropout_keep_prob)
  fc8 = tf.contrib.layers.fully_connected(
      inputs=drp7,
      num_outputs=FLAGS.num_classes,
      activation_fn=None)

  # Calculating the loss.
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=labels, logits=fc8)

  # Configuring the optimization algorithm.
  learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate, tf.train.get_global_step(), 25000, 0.97)
#  learning_rate = FLAGS.learning_rate
  if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  total = 0
  for i in tf.global_variables():
    t = 1
    for j in i.get_shape():
        t *= int(j)
    total += t
  print total
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
