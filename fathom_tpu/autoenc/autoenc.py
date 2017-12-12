import tensorflow as tf
from fathom.imagenet.mnist import *
from fathom.nn import *
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
import time

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
      train_batch_size=FLAGS.batch_size,
  #    eval_batch_size=128,
      config=run_config)
  estimator.train(input_fn=get_input_fn(FLAGS.train_file), max_steps=FLAGS.train_steps)
  #estimator.evaluate(input_fn=get_input_fn(FLAGS.eval_file), steps=100)

  total = time.time() - start
  print("Total time: " + str(total))

if __name__ == "__main__":
  tf.app.run()
