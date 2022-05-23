from timeit import repeat
import keras
from keras.layers import *


# Paper: https://arxiv.org/pdf/1512.03385.pdf



def add_block(layer, num, num_blocks, filters, pad, plain=False):
    connect = layer
    for b in range(num_blocks):
        s = 2 if (b == 0 and num != 0) else 1
        layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=s, padding=pad, activation='relu',
                       name='{}.{}.{}_3x3_Conv2D_{}_s{}'.format(num+1, b+1, 1, filters, s))(layer)
        # if b != 0 or num == 0:
        #     layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
        #                    name='{}.{}.{}_3x3_Conv2D_{}'.format(num+1, b+1, 1, filters))(layer)
        layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                       name='{}.{}.{}_3x3_Conv2D_{}_s1'.format(num+1, b+1, 2, filters))(layer)
        if plain:
            tmp = layer
            # https://github.com/keras-team/keras/issues/2608
            layer = ZeroPadding2D(((connect.shape[1] - layer.shape[1])//2, (connect.shape[2] - layer.shape[2])//2 ))(layer)
            layer = Add(name='{}.{}.{}_shortcut'.format(num+1, b+1, 3, filters))([connect, layer])
            connect = tmp
    return layer


def build_model(size=224, pad='same', n_channels=1, n_classes=2, plain=False):
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

    f = 64
    # Initial Convolution
    layer = Conv2D(filters=f, kernel_size=(7, 7), strides=2, padding=pad, activation='relu',
                   name='0.1_7x7_Conv2D_64_s2')(inputs)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding=pad, name='0.2_3x3_MaxPool_s2')(layer)
    for i, block_size in enumerate([3, 4, 6, 3]):
        layer = add_block(layer, num=i, num_blocks=block_size, filters=f, pad=pad, plain=plain)
        f *= 2
        # if i != 3:
        #     layer = Conv2D(filters=f, kernel_size=(3, 3), strides=2, padding=pad, activation='relu',
        #                    name='{}.1_3x3_Conv2D_{}_s2'.format(i+2, f))(layer)
    # Final Average Pool
    avg_pool = GlobalAveragePooling2D()(layer)
    # Output
    outputs = Dense(1000, activation='softmax', name='output')(avg_pool)
    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = build_model(plain=True)
    print(model.summary())
