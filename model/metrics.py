import tensorflow as tf
from tensorflow import keras
from keras import backend as K


def recall(y_gt, y_pred):
    y_pred_true = tf.cast(y_pred > 0.5, tf.float32)
    y_true_gt = y_gt
    true_positives = K.sum(K.round(K.clip(y_true_gt * y_pred_true, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true_gt, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall


def fp(y_gt, y_pred):
    y_pred_true = tf.cast(y_pred > 0.5, tf.float32)
    y_false_gt = tf.cast(y_gt == 0, tf.float32)

    false_positives = K.sum(K.round(K.clip(y_false_gt * y_pred_true, 0, 1)))
    all_pixel = K.sum(tf.ones_like(y_pred))
    return false_positives / all_pixel


def prec(y_gt, y_pred):
    y_pred_true = tf.cast(y_pred > 0.5, tf.float32)
    y_true_gt = y_gt

    true_positives = K.sum(K.round(K.clip(y_true_gt * y_pred_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def dice_coefficient_m(y_true, y_pred):
    precision = prec(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
