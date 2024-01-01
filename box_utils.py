import tensorflow as tf

# Box functions
def swap_xy(boxes, batched:bool = True):
    """`(xmin, ymin, xmax, ymax)` => `(ymin, xmin, ymax, xmax)`"""
    if batched:
        return tf.vectorized_map(lambda boxes: swap_xy(boxes, batched=False), boxes)
    boxes = tf.convert_to_tensor([boxes]) if tf.shape(boxes).shape[0]==1 else tf.convert_to_tensor(boxes)
    boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
    return boxes

def xyxy_to_xywh(boxes, batched:bool = True):
    """`(xmin, ymin, xmax, ymax)` => `(xcenter, ycenter, width_box, height_box)`"""
    if batched:
        return tf.vectorized_map(lambda boxes: xyxy_to_xywh(boxes, batched=False), boxes)
    boxes = tf.convert_to_tensor([boxes]) if tf.shape(boxes).shape[0]==1 else tf.convert_to_tensor(boxes)
    boxes = tf.concat([(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]], axis=-1)
    return boxes
    
def xywh_to_xyxy(boxes, batched:bool = True):
    """`(xcenter, ycenter, width_box, height_box)` => `(xmin, ymin, xmax, ymax)`"""
    if batched:
        return tf.vectorized_map(lambda boxes: xywh_to_xyxy(boxes, batched=False), boxes)
    boxes = tf.convert_to_tensor([boxes]) if tf.shape(boxes).shape[0]==1 else tf.convert_to_tensor(boxes)
    boxes =  tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)
    return boxes
