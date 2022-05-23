from math import ceil
import copy
import keras
from keras.layers import *
# from tensorflow.keras.layers import *

# Paper: https://arxiv.org/pdf/1512.03385.pdf


def add_block(block_num, input, filters, layers, pad):
    for l in range(layers):
        # Conv2D
        input = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                       name='{}.{}_3x3_conv_{}'.format(block_num, l, filters))(input)
    return MaxPool2D(pool_size=(3, 3), strides=2, padding=pad, name='{}.{}_3x3_MaxPool'.format(block_num, l))(input)


def build_model(size=512, pad='same', n_channels=1, n_classes=2, bn=True):
    """
    :param size: Size of input image
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param n_channels: Number of channels of the input image
    :param n_classes: Number of classes to predict
    :param bn: Whether to perform batch normalization after each convolution

    Builds the Unet model

    :return: model
    """
    # Input might not fit depending on size due to cropping
    inputs = Input(shape=(size, size, n_channels), name='input')
    block = inputs
    f = 64
    for num, l in enumerate([2, 2, 4, 4, 4]):
        block = add_block(num, block, filters=f, layers=l, pad=pad)
        if f < 512:
            f *= 2
    flat = Flatten()(block)
    fc1 = Dense(4096, activation='relu')(flat)
    fc2 = Dense(4096, activation='relu')(fc1)

    # Output
    outputs = Dense(n_classes, activation='softmax', name='output')(fc2)
    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = build_model(size=224)
    print(model.summary())
