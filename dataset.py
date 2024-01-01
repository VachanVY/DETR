import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv as nn_cv

# get dataset
def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
  image = inputs["image"]
  boxes = nn_cv.bounding_box.convert_format(
      inputs["objects"]["bbox"],
      images=image,
      source="rel_yxyx",
      target=bounding_box_format,
  )
  bounding_boxes = {
      "classes": tf.cast(inputs["objects"]["label"] + 1, dtype=tf.float32), # + 1 to incorparate no_object class
      "boxes": tf.cast(boxes, dtype=tf.float32),
  }
  return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

def load_pascal_voc(split, dataset, bounding_box_format):
  ds = tfds.load(dataset, split=split, with_info=False, shuffle_files=True)
  ds = ds.map(lambda x: unpackage_raw_tfds_inputs(x, bounding_box_format=bounding_box_format), num_parallel_calls=tf.data.AUTOTUNE)
  return ds
