import tensorflow as tf
import keras
from keras.layers import *


# Source: https://github.com/taki0112/SENet-Tensorflow
# Paper: https://arxiv.org/pdf/1709.01507.pdf


def SE_block(input_x, r):
    out_dim = input_x.shape[-1]
    # Squeeze, but keep dims/filters
    squeeze = GlobalAveragePooling2D(keepdims=True)(input_x)

    # Learning Weights
    excitation = Dense(out_dim / r, activation='relu', name='fully_connected1')(squeeze)
    excitation = Dense(out_dim, activation='sigmoid', name='fully_connected2')(excitation)

    # Reweight feature maps
    scale = Multiply()([input_x, excitation])
    return scale


def build_model(size=512, pad='same', n_channels=1, n_classes=2, bn=True, r=16):
    """
    :param size: Size of input image
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param n_channels: Number of channels of the input image
    :param n_classes: Number of classes to predict
    :param bn: Whether to perform batch normalization after each convolution
    :param r: Reduction ratio in SE block. A high reduction ratio reduces computation cost,
              but might affect performance. (default: 16)

    Builds the Unet model

    :return: model
    """
    # Input might not fit depending on size due to cropping
    inputs = Input(shape=(size, size, n_channels), name='input')

    conv = Conv2D(filters=64, kernel_size=(3, 3))(inputs)

    # Output
    outputs = SE_block(conv, r)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
