#!/usr/bin/env python

import tensorflow as tf
import os
from fathom.nn import *
#from vgg_preprocessing import *

#image_preprocessing_fn = preprocess_image

def input_fn(params):
    """Passes data to the estimator as required."""
    batch_size = params["batch_size"]

    def parser(serialized_example):
      """Parses a single tf.Example into a 224x224 image and label tensors."""

      final_image = None
      final_label = None
      features = tf.parse_single_example(
            serialized_example,
            features={
                "image/encoded": tf.FixedLenFeature([], tf.string),
                "image/class/label": tf.FixedLenFeature([], tf.int64),
            })
      image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.image.resize_images(
            image,
            size=[224, 224])
      final_label = tf.cast(features["image/class/label"], tf.int32)


      final_image = (tf.cast(image, tf.float32) * (1. / 255)) - 0.5

      #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      #final_image = image_preprocessing_fn(
      #  image=image,
      #  output_height=224,
      #  output_width=224,
      #  is_training=True)
      return final_image, tf.one_hot(final_label, FLAGS.num_classes)

    file_pattern = os.path.join(FLAGS.data_dir, 'train-*')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat()

    def prefetch_dataset(filename):
      buffer_size = 256 * 1024 * 1024  # 256 MB
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    if FLAGS.use_sloppy_interleave:
      dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(prefetch_dataset, cycle_length=FLAGS.cycle_length, sloppy=FLAGS.use_sloppy))
    else:
        dataset = dataset.interleave(
          prefetch_dataset, cycle_length=FLAGS.cycle_length)

    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

    dataset = dataset.map(parser,num_parallel_calls=FLAGS.num_parallel_calls)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)

    images, labels = dataset.make_one_shot_iterator().get_next()
    return (
        tf.reshape(images, [batch_size, 224, 224, 3]),
        tf.reshape(labels, [batch_size, FLAGS.num_classes])
    )
