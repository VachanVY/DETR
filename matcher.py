from box_utils import swap_xy, xywh_to_xyxy

import tensorflow as tf
from tensorflow_addons.losses import GIoULoss

from scipy.optimize import linear_sum_assignment

# Matcher
class Matcher:
    """
    Args:
        y     {"class_true": shape(m, n_objects),    "bbox_true": shape(m, n_objects, 4)},\n
        y_hat {"class_prob": shape(m, N, n_classes), "bbox_pred": shape(m, N, 4)}
    output:
        (matched_class_prob: (m, N, n_classes), {matched_bbox_pred: (m, N, 4))
    """
    @staticmethod
    def L_box(bbox_true, bbox_pred, lambda_giou=2, lambda_l1=5):
        return lambda_giou*GIoULoss()(bbox_true, bbox_pred) + lambda_l1*tf.norm(bbox_true - bbox_pred)

    @staticmethod
    def L_match(class_true, prob_pred_of_class_true, bbox_true, bbox_pred):
        class_bool = tf.constant(0.) if class_true == 0 else tf.constant(1.)
        return -class_bool * prob_pred_of_class_true + class_bool * Matcher.L_box(bbox_true, bbox_pred)

    @staticmethod
    def compute_cost_matrix(class_true, class_prob, bbox_true, bbox_pred):
        """(N), (N, n_classes), (N, 4), (N, 4)"""
        N = tf.shape(class_true)[0]
        cost_i = lambda i: tf.map_fn(
            lambda j: Matcher.L_match(
                class_true[i], 
                class_prob[j, int(class_true[i])], 
                bbox_true[i], 
                bbox_pred[j]
                ), 
            tf.range(N), fn_output_signature=tf.float32
            )
        return tf.map_fn(lambda i: cost_i(i), tf.range(N), fn_output_signature=tf.float32)

    @staticmethod
    def batched_cost_matrix(class_true, bbox_true, class_prob, bbox_pred):
        """
        For Multiple Batches:
            class_true: (m, N),
            class_prob: (m, N, n_classes),
            bbox_true: (m, N, 4),
            bbox_pred: (m, N, 4)
        """
        print(tf.shape(class_true))
        return tf.map_fn(
            lambda B: Matcher.compute_cost_matrix(class_true[B], class_prob[B], bbox_true[B], bbox_pred[B]), 
            tf.range(tf.shape(class_true)[0]), fn_output_signature=tf.float32
            )

    @staticmethod
    def match(class_true, bbox_true, class_prob, bbox_pred):
        bbox_true, bbox_pred = swap_xy(xywh_to_xyxy(bbox_true)), swap_xy(xywh_to_xyxy(bbox_pred))

        C = Matcher.batched_cost_matrix(class_true, bbox_true, class_prob, bbox_pred)
        idx = tf.stack([linear_sum_assignment(C[i])[1] for i in range(C.shape[0])])

        class_prob = tf.gather(class_prob, idx, batch_dims=1)
        bbox_pred = tf.gather(bbox_pred, idx, batch_dims=1)
        return class_prob, bbox_pred

    def __call__(self, y, y_hat):
        class_true, bbox_true = y
        class_prob, bbox_pred = y_hat

        class_prob, bbox_pred = Matcher.match(class_true, bbox_true, class_prob, bbox_pred)
        return tf.stop_gradient(class_prob), tf.stop_gradient(bbox_pred)
    