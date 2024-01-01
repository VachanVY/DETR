from box_utils import swap_xy, xywh_to_xyxy

import tensorflow as tf
from tensorflow_addons.losses import GIoULoss
tnp = tf.experimental.numpy
nn = tf.keras

# Hungarian Loss
class HungarianLoss(nn.losses.Loss):
    def __init__(self, lambda_giou=2, lambda_l1=5, **kwargs):
        super().__init__(**kwargs)
        self.lambda_giou = lambda_giou
        self.lambda_l1 = lambda_l1
        self.cross_entropy = tf.losses.SparseCategoricalCrossentropy()
        self.giou = GIoULoss()
        self.mae = lambda bbox_true, bbox_pred: tf.reduce_mean(tf.abs(bbox_true-bbox_pred))

    def BoxLoss(self, bbox_true, bbox_pred):
        bbox_true = swap_xy(xywh_to_xyxy(bbox_true))
        bbox_pred = swap_xy(xywh_to_xyxy(bbox_pred))
        
        mask = tf.reduce_all(tf.not_equal(bbox_true, 0.0), axis=-1)
        bbox_true = tf.boolean_mask(bbox_true, mask) # masked, no batch dim
        bbox_pred = tf.boolean_mask(bbox_pred, mask) # masked, no batch dim
        return self.lambda_giou*self.giou(bbox_true, bbox_pred) + self.lambda_l1*self.mae(bbox_true, bbox_pred)

    def call(self, y_true, y_hat):
        class_true, bbox_true = y_true # (B, N), (B, N, 4)
        class_prob, bbox_pred = y_hat # (B, N, n_classes), (B, N, 4)

        class_true = tf.reshape(class_true, [-1]) # (_) flattened
        class_prob = tf.reshape(class_prob, (-1, class_prob.shape[-1])) # (_, n_classes)

        masked_bbox_cost = self.BoxLoss(bbox_true, bbox_pred)
        class_cost = self.cross_entropy(class_true, class_prob)
        return class_cost + masked_bbox_cost
    