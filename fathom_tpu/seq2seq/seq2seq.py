#!/usr/bin/env python
import tensorflow as tf
import numpy as np

import math
import random
import sys
import time

from fathom.nn import *
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer

import data_utils

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
target_vocab_size = 40000
source_vocab_size = 40000
en_vocab_size = 40000
fr_vocab_size = 40000

def model_fn(features, labels, mode, params):
  del params  # Unused.

  # If we use sampled softmax, we need an output projection.
  output_projection = None
  softmax_loss_function = None
  # Sampled softmax only makes sense if we sample less than vocabulary size.
  num_samples = 512
  num_layers = 3
  size = 256
  num_unrolled_steps = 35

  if num_samples > 0 and num_samples < target_vocab_size:
    w = tf.get_variable("proj_w", [size, target_vocab_size])
    w_t = tf.transpose(w)
    b = tf.get_variable("proj_b", [target_vocab_size])
    output_projection = (w, b)

    def sampled_loss(labels, logits):
      labels = tf.reshape(labels, [-1, 1])
      # We need to compute the sampled_softmax_loss using 32bit floats to
      # avoid numerical instabilities.
      local_w_t = tf.cast(w_t, tf.float32)
      local_b = tf.cast(b, tf.float32)
      local_inputs = tf.cast(logits, tf.float32)
      return tf.nn.sampled_softmax_loss(
              weights=local_w_t,
              biases=local_b,
              labels=labels,
              inputs=local_inputs,
              num_sampled=num_samples,
              num_classes=target_vocab_size)
    softmax_loss_function = sampled_loss

  # Create the internal multi-layer cell for our RNN.
  for l in range(num_layers):
    with tf.variable_scope("rnn_%d" % l):
      unstacked_inputs = tf.unstack(
          inputs, num=num_unrolled_steps, axis=0)
      cell = tf.nn.rnn_cell.BasicLSTMCell(size)
      outputs, _ = tf.nn.static_rnn(cell,
                                    unstacked_inputs,
                                    dtype=tf.float32)
      cell = tf.stack(outputs, axis=0)
      cell = tf.nn.dropout(cell, 1 - FLAGS.dropout_prob)


  # The seq2seq function: we use embedding for the input and attention.
  def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
    return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
        encoder_inputs, decoder_inputs, cell,
        num_encoder_symbols=source_vocab_size,
        num_decoder_symbols=target_vocab_size,
        embedding_size=size,
        output_projection=output_projection,
        feed_previous=do_decode)

  _outputs, losses = tf.contrib.legacy_seq2seq.model_with_buckets(
      encoder_inputs, decoder_inputs, targets,
      target_weights, buckets,
      lambda x, y: seq2seq_f(x, y, False),
      softmax_loss_function=softmax_loss_function)

  updates = None
  # Gradients and SGD update operation for training the model.
  params = tf.trainable_variables()
  gradient_norms = []
  updates = []
  opt = tpu_optimizer.CrossShardOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate))
  for b in xrange(len(buckets)):
    gradients = opt.compute_gradients(losses[b], params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     max_gradient_norm)
    gradient_norms.append(norm)
    updates.append(opt.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step))

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=updates)


train_set = []

def read_data( source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids])
              break
          source, target = source_file.readline(), target_file.readline()
    return data_set

def load_data():
    data_dir = "data/"

    print("Preparing WMT data in %s" % data_dir)
    en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
        data_dir, en_vocab_size, fr_vocab_size)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % max_train_data_size)
    dev_set = read_data(en_dev, fr_dev)
    train_set = read_data(en_train, fr_train, max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]


def input_fn(params):
  batch_size = params["batch_size"]

  encoder_inputs, decoder_inputs, target_weights = get_batch(
          train_set, bucket_id)
  output_feeds, input_feeds = self.step_feeds(encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
  return input_feeds, output_feeds


def step_feeds( encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Construct feeds for given inputs.

    Args:
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[encoder_inputs[l].name] = encoder_inputs[l]
      #print("encoder", len(encoder_inputs[l]), encoder_inputs[l].get_shape())
    for l in xrange(decoder_size):
      input_feed[decoder_inputs[l].name] = decoder_inputs[l]
      #print("decoder", len(decoder_inputs[l]), decoder_inputs[l].get_shape())
      input_feed[target_weights[l].name] = target_weights[l]
      #print("target", len(target_weights[l]), target_weights[l].get_shape())

    # Since our targets are decoder inputs shifted by one, we need one more.
    #last_target = decoder_inputs[decoder_size].name
    last_target = decoder_inputs[decoder_size]
    input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [updates[bucket_id],  # Update Op that does SGD.
                     gradient_norms[bucket_id],  # Gradient norm.
                     losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(_outputs[bucket_id][l])

    return output_feed, input_feed

def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights



def main(unused_argv):
  assert len(unused_argv) == 1, (
      "Unrecognized command line arguments: %s" % unused_argv[1:])
  load_data()

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
