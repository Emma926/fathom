from math import sqrt
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import fathom.imagenet.mnist as input_data
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time

tf.flags.DEFINE_string("eval_file", "", "Path to mnist evaluation data.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_string("train_file", "", "Path to mnist training data.")
tf.flags.DEFINE_integer("dataset_reader_buffer_size", None,
                        "The size of the buffer for dataset read operations.")
FLAGS = tf.flags.FLAGS

def model_fn(features, labels, mode, params):
      del params

    #with G.as_default():
      transfer = tf.nn.softplus

      hidden = tf.layers.dense(inputs = features, units = 200, activation=transfer)
      reconstruction = tf.layers.dense(inputs = hidden, units = 784)

      # for an autoencoder, the cost/loss is not just part of training
      loss_op = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, labels), 2.0))

      learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                             tf.train.get_global_step(), 100000,
                                             0.96)
      if FLAGS.use_tpu:
        opt = tpu_optimizer.CrossShardOptimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
      else:
        opt = tf.train.AdamOptimizer()
      train_op = opt.minimize(loss_op, global_step=tf.train.get_global_step())
      #return tf.estimator.EstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)
      return tpu_estimator.TPUEstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)

def get_input_fn(filename):

  def input_fn(params):
    """A simple input_fn using the experimental input pipeline."""
    batch_size = params["batch_size"]
    print("batch size: " + str(batch_size))
  
    def parser(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      features = tf.parse_single_example(
          serialized_example,
          features={
              "image_raw": tf.FixedLenFeature([], tf.string),
              "label": tf.FixedLenFeature([], tf.int64),
          })
      image = tf.decode_raw(features["image_raw"], tf.uint8)
      image.set_shape([28*28])
      # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
      image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
      label = tf.cast(features["label"], tf.int32)
      return image, label
  
    dataset = tf.contrib.data.TFRecordDataset(
        filename, buffer_size=FLAGS.dataset_reader_buffer_size)
    dataset = dataset.map(parser).cache().repeat().batch(batch_size)
    images, labels = dataset.make_one_shot_iterator().get_next()
    # Give inputs statically known shapes.
    images.set_shape([batch_size, 28*28])
    labels.set_shape([batch_size])
    print("image size: " + str(images.shape))
    return images, images
  return input_fn
  
def main(unused_argv):
  del unused_argv

  start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  run_config = tpu_config.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
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
      train_batch_size=128,
      eval_batch_size=128,
      config=run_config)
  estimator.train(input_fn=get_input_fn(FLAGS.train_file), max_steps=FLAGS.train_steps)
  estimator.evaluate(input_fn=get_input_fn(FLAGS.eval_file), steps=100)

  total = time.time() - start
  print("Total time: " + str(total))

if __name__ == "__main__":
  tf.app.run()
