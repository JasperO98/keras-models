import keras
from keras.layers import *


# Paper: https://arxiv.org/pdf/1512.03385.pdf


def SE_block(input_x, r):
    out_dim = input_x.shape[-1]
    name = input_x.name.split('/')[0]
    
    # Squeeze, but keep dims/filters
    squeeze = GlobalAveragePooling2D(keepdims=True, name=name + '_Squeeze')(input_x)

    # Learning Weights
    excitation = Dense(out_dim / r, activation='relu', name=name + '_ExciteReduction')(squeeze)
    excitation = Dense(out_dim, activation='sigmoid', name=name + '_Excite')(excitation)

    # Reweight feature maps
    scale = Multiply(name=name + '_Scale')([input_x, excitation])
    return scale


def add_block(layer, num, num_blocks, filters, pad, plain, SE, r, version):
    """
    :param layer: Input layer
    :param num: Section number of the network
    :param num_blocks: Number of blocks to add to the section
    :param filters: The number of filters of the current section
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param plain: Whether or not to add the residual connections (False) or keep the network plain (True)

    Creates plain or residual blocks.

    :return: Full section of plain or residual blocks.
    """
    # Define layer to connect to
    connect = layer
    for b in range(num_blocks):
        # First layer of block, stride is 2 if it is the first layer of all blocks
        s = 2 if (b == 0 and num != 0) else 1
        # Basic block
        if version in [18, 34]:
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=s, padding=pad, activation='relu',
                        name='{}.{}.{}_3x3_Conv2D_{}_s{}'.format(num+1, b+1, 1, filters, s))(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                        name='{}.{}.{}_3x3_Conv2D_{}_s1'.format(num+1, b+1, 2, filters))(layer)
        # So called "Bottleneck"
        elif version in [50, 101, 152]:
            layer = Conv2D(filters=filters, kernel_size=(1, 1), strides=s, padding=pad, activation='relu',
                        name='{}.{}.{}_1x1_Conv2D_{}_s{}'.format(num+1, b+1, 1, filters, s))(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding=pad, activation='relu',
                        name='{}.{}.{}_3x3_Conv2D_{}_s1'.format(num+1, b+1, 2, filters))(layer)
            layer = Conv2D(filters=filters*4, kernel_size=(1, 1), strides=1, padding=pad, activation='relu',
                        name='{}.{}.{}_1x1_Conv2D_{}_s1'.format(num+1, b+1, 3, filters*4))(layer)
        else:
            exit(1)
        # Add Squeeze and Excitation if needed
        if SE:
            layer = SE_block(layer, r=r)
        # Add skip connect if it is not a plain ResNet
        if not plain:
            tmp = layer
            # Reduce dimension if stride was 2 or filters do not overlap
            if s == 2 or layer.shape[-1] != connect.shape[-1]:
                connect = Conv2D(filters=filters if version in [18, 34] else filters * 4, kernel_size=1, strides=s, padding="valid")(connect)
            # Add connection layer to current layer
            layer = Add(name='{}.{}.{}_shortcut'.format(num+1, b+1, 3, filters))([connect, layer])
            # Save new connect layer
            connect = tmp
    # Return block
    return layer


def build_model(size=224, pad='same', n_channels=1, n_classes=2, version=34, plain=False, SE=False, r=16):
    """
    :param size: Size of input image
    :param pad: Padding of the convolution operation (same=padding, valid=no padding)
    :param n_channels: Number of channels of the input image
    :param n_classes: Number of classes to predict
    :param bn: Whether to perform batch normalization after each convolution

    Builds the ResNet model

    :return: model
    """
    sections = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[version]
    # Input might not fit depending on size due to cropping
    inputs = Input(shape=(size, size, n_channels), name='input')

    f = 64
    # Initial Convolution
    layer = Conv2D(filters=f, kernel_size=(7, 7), strides=2, padding=pad, activation='relu',
                   name='0.1_7x7_Conv2D_64_s2')(inputs)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding=pad, name='0.2_3x3_MaxPool_s2')(layer)
    # Add sections, depending on version
    for i, section_size in enumerate(sections):
        layer = add_block(layer, num=i, num_blocks=section_size, filters=f, pad=pad, plain=plain, SE=SE, r=r, version=version)
        f *= 2
    # Final Average Pool
    avg_pool = GlobalAveragePooling2D()(layer)
    # Output
    outputs = Dense(n_classes, activation='softmax', name='output')(avg_pool)
    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = build_model(version=34, plain=False, SE=True)
    print(model.summary())
