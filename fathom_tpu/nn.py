#!/usr/bin/env python

import tensorflow as tf
import time

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("use_sloppy", True, "Use sloppy=True in parallel_intereleave")
tf.flags.DEFINE_bool("use_sloppy_interleave", True, "Use sloppy_interleave rather than interleave")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Total number of steps. Note that the actual number of "
                        "steps is the next multiple of --iterations greater "
                        "than this value.")
tf.flags.DEFINE_integer("save_checkpoints_secs", None,
                        "Seconds between checkpoint saves. If None, "
                                                "checkpoint will not be saved.")
tf.flags.DEFINE_string("master", "",
                       "GRPC URL of the Cloud TPU instance.")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_string("machine", 'tpu/2x2', "machine")
tf.flags.DEFINE_string("out_dir", '', "output directory")

tf.flags.DEFINE_integer("num_classes", 1000,
                        "Number of distinct labels in the data.")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8,
                      "Keep probability of the dropout layers.")
tf.flags.DEFINE_string("data_dir", "",
                       "Path to the directory that contains the 1024 TFRecord "
                                              "Imagenet training data files.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_string("train_file", "", "Path to training data.")
tf.flags.DEFINE_integer("dataset_reader_buffer_size", None,
                        "The size of the buffer for dataset read operations.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training. Note that this "
                                                "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer('shuffle_buffer_size', 1000,
                        'Size of the shuffle buffer used to randomize ordering')
tf.flags.DEFINE_integer('cycle_length', 32,
                        'The number of threads to interleave from in parallel.')
tf.flags.DEFINE_integer('num_parallel_calls', 32,
                        'The number of threads to parse the data in parallel.')

FLAGS = tf.flags.FLAGS

def dump(name,total):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = FLAGS.out_dir + "/" + FLAGS.machine + '/' + name + '-' + str(FLAGS.batch_size) + '-' + str(FLAGS.train_steps) + '-' + timestr
    model = name + '\n'
    model += "Machine: " + FLAGS.machine + '\n'
    model += "Batch size: " + str(FLAGS.batch_size) + '\n'
    model += "Steps: " + str(FLAGS.train_steps) + '\n'
    model += "Cycle length: " + str(FLAGS.cycle_length) + '\n'
    model += "Num parallel calls: " + str(FLAGS.num_parallel_calls) + '\n'
    with gfile.FastGFile(path, 'w') as f:
      f.write(model)
      f.write("Total runtime: " + str(total) + ' seconds.\n')

