import tensorflow as tf
from fathom.nn import *

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
    return images, images
  return input_fn
  
