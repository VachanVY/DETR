import tensorflow as tf
nn = tf.keras
import keras_cv as nn_cv

def BackBone(d_model: int):
    """Pre-trained Conv-Net + Conv-DownSample"""
    AugmentData = lambda : nn.Sequential([
            nn_cv.layers.RandomHue(factor=0.4, value_range=[0, 255]),
            nn_cv.layers.RandomChannelShift(value_range=[0, 255], factor=0.4),
            nn_cv.layers.GridMask(),
            nn.layers.RandomBrightness(factor=0.2, value_range=[0, 255])
        ])
    inputs = nn.layers.Input(shape=(None, None, 3))
    x = AugmentData()(inputs)
    x = nn.applications.EfficientNetV2B0(include_preprocessing=True, include_top=False, weights="imagenet")(x) # (B, H, W, 1024)
    x = nn.layers.Conv2D(filters=d_model, kernel_size=1)(x) # (B, H, W, d)
    outputs = nn.layers.Reshape((-1, d_model))(x) # (B, H*W, d)
    return nn.Model(inputs=inputs, outputs=outputs)
