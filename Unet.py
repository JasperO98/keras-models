import keras
from keras.layers import *


# Paper: https://link.springer.com/content/pdf/10.1007%2F978-3-319-24574-4_28.pdf


def down_sample_block(inputs, pad, bn, i):
    """
    :param inputs: Previous output layer
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param bn: Whether to perform batch normalization after each convolution
    :param i: Number of down sample block, count ascending

    Creates a down sample block of Unet, which is a component of the encoder consisting of convolution and max pooling

    :return: last convolutional layer and max pool layer
    """
    # Define filters, start with 64 else double previous filters
    if i == 1:
        f = 64
    else:
        f = inputs.shape[-1]*2
    # Convolution
    conv1 = Conv2D(filters=f, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                   name='down{}_conv_a'.format(i))(inputs)
    if bn:
        conv1 = BatchNormalization(name='down{}_bn_a'.format(i))(conv1)
    conv2 = Conv2D(filters=f, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                   name='down{}_conv_b'.format(i))(conv1)
    if bn:
        conv2 = BatchNormalization(name='down{}_bn_b'.format(i))(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, name='down{}_maxpool'.format(i))(conv2)
    return conv2, max_pool


def up_sample_block(inputs, connect, pad, bn, i):
    """
    :param inputs: Previous output layer
    :param connect: The layer to concat with the input layer
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param bn: Whether to perform batch normalization after each convolution
    :param i: Number of up sample block, count descending

    Creates an up sample block of Unet, which is a component of the decoder consisting of:
    - Deconvolution
    - Making a skip connection between the deconvoluted image and output of associated encoder block
    - Convolution

    :return: last layer of block
    """
    # New filters are half of the input
    f = inputs.shape[-1]//2

    # Up sample
    up_sample = Conv2DTranspose(filters=f, kernel_size=(2, 2), strides=2, padding=pad, activation=None,
                                name='up{}_deconv'.format(i))(inputs)

    # Crop to fit up sample
    top = (connect.shape[1] - up_sample.shape[1]) // 2
    bot = top
    # bot = int(ceil((connect.shape[1] - upsample.shape[1]) / 2))
    crop = Cropping2D(((top, bot), (top, bot)), name='up{}_crop'.format(i))(connect)

    # Skip connect
    concat = Concatenate(name='up{}_concat'.format(i))([up_sample, crop])
    if bn:
        concat = BatchNormalization(name='up{}_bn'.format(i))(concat)

    # Conv
    conv1 = Conv2D(filters=f, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                   name='up{}_conv_a'.format(i))(concat)
    if bn:
        conv1 = BatchNormalization(name='up{}_bn_a'.format(i))(conv1)
    conv2 = Conv2D(filters=f, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                   name='up{}_conv_b'.format(i))(conv1)
    if bn:
        conv2 = BatchNormalization(name='up{}_bn_b'.format(i))(conv2)
    return conv2


def build_model(size=512, pad='same', n_channels=1, n_classes=2):
    """
    :param size: Size of input image
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param n_channels: Number of channels of the input image
    :param n_classes: Number of classes to predict

    Builds the Unet model

    :return: model
    """
    # Input might not fit depending on size due to cropping
    inputs = Input(shape=(size, size, n_channels), name='input')

    # Encoder
    down_conv1, max_pool1 = down_sample_block(inputs, pad, bn, 1)
    down_conv2, max_pool2 = down_sample_block(max_pool1, pad, bn, 2)
    down_conv3, max_pool3 = down_sample_block(max_pool2, pad, bn, 3)
    down_conv4, max_pool4 = down_sample_block(max_pool3, pad, bn, 4)

    # Final convolution block
    down_conv5 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                        name='final_conv_a')(max_pool4)
    if bn:
        down_conv5 = BatchNormalization(name='final_bn_a')(down_conv5)
    down_conv6 = Conv2D(filters=1024, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                        name='final_conv_b')(down_conv5)
    if bn:
        down_conv6 = BatchNormalization(name='final_bn_b')(down_conv6)

    # Decoder
    up_conv1 = up_sample_block(down_conv6, down_conv4, pad, bn, 4)
    up_conv2 = up_sample_block(up_conv1, down_conv3, pad, bn, 3)
    up_conv3 = up_sample_block(up_conv2, down_conv2, pad, bn, 2)
    up_conv4 = up_sample_block(up_conv3, down_conv1, pad, bn, 1)

    # Output
    outputs = Conv2D(filters=n_classes, kernel_size=(1, 1), strides=1, activation='softmax', name='output')(up_conv4)
    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
