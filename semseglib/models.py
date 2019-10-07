from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout, \
    Activation, Add, Lambda
from keras.models import Model
from keras import initializers
import keras
import numpy as np

######## Blocks
### Vanilla
def vanilla_block(X, f, level_number, direction, batchnorm=0, dilations=None):
    suffix = "_" + direction + "_" + str(level_number)

    if batchnorm == 2:
        X = BatchNormalization(name="batchnorm" + suffix + "a")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "a")(X)
    X = Activation("relu", name="relu" + suffix + "a")(X)

    if batchnorm:
        X = BatchNormalization(name="batchnorm" + suffix + "b")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "b")(X)
    X = Activation("relu", name="relu" + suffix + "b")(X)

    return X

### Residual
def residual_conv_block(X, f, level_number, direction, batchnorm=0, dilations=None):
    suffix = "_" + direction + "_" + str(level_number)
    shortcut = X

    if batchnorm == 2:
        X = BatchNormalization(name="batchnorm" + suffix + "a")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "a")(X)
    X = Activation("relu", name="relu" + suffix + "a")(X)

    if batchnorm:
        X = BatchNormalization(name="batchnorm" + suffix + "b")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "b")(X)
    X = Activation("relu", name="relu" + suffix + "b")(X)

    shortcut = Conv2D(f, 1, kernel_initializer="he_normal", name="conv" + suffix + "_short")(shortcut)
    X = Add(name="add" + suffix)([X, shortcut])

    return X

def residual_concat_block(X, f, level_number, direction, batchnorm=0, dilations=None):
    suffix = "_" + direction + "_" + str(level_number)
    shortcut = X

    if batchnorm == 2:
        X = BatchNormalization(name="batchnorm" + suffix + "a")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "a")(X)
    X = Activation("relu", name="relu" + suffix + "a")(X)

    if batchnorm:
        X = BatchNormalization(name="batchnorm" + suffix + "b")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "b")(X)
    X = Activation("relu", name="relu" + suffix + "b")(X)

    X = Concatenate(axis=3, name="merge" + suffix + "_short")([X, shortcut])

    return X

def residual_zeropad_block(X, f, level_number, direction, batchnorm=0, dilations=None):
    suffix = "_" + direction + "_" + str(level_number)
    shortcut = X

    if batchnorm == 2:
        X = BatchNormalization(name="batchnorm" + suffix + "a")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "a")(X)
    X = Activation("relu", name="relu" + suffix + "a")(X)

    if batchnorm:
        X = BatchNormalization(name="batchnorm" + suffix + "b")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="he_normal",
               name="conv" + suffix + "b")(X)
    X = Activation("relu", name="relu" + suffix + "b")(X)

    X_channels = X.shape.as_list()[-1]
    shortcut_channels = shortcut.shape.as_list()[-1]
    if X_channels >= shortcut_channels:
        identity_weights = np.eye(shortcut_channels, X_channels, dtype=np.float32)
        shortcut = Conv2D(X_channels, kernel_size=1, strides=1, use_bias=False, trainable=False,
                          kernel_initializer=initializers.Constant(value=identity_weights),
                          name="zeropad" + suffix)(shortcut)
    else:
        identity_weights = np.eye(X_channels, shortcut_channels, dtype=np.float32)
        X = Conv2D(shortcut_channels, kernel_size=1, strides=1, use_bias=False, trainable=False,
                   kernel_initializer=initializers.Constant(value=identity_weights),
                   name="zeropad" + suffix)(X)
    X = Add(name="add" + suffix)([X, shortcut])

    return X

### Atrous
def atrous_single_cell(X, f, level_number, direction, batchnorm=0, d=5):
    suffix = "_" + direction + "_" + str(level_number) + "_d" + str(d)

    if batchnorm == 2:
        X = BatchNormalization(name="batchnorm" + suffix + "a")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="one", dilation_rate=(3, 3),
               name="conv" + suffix + "a")(X)
    X = Activation("relu", name="relu" + suffix + "a")(X)

    if batchnorm:
        X = BatchNormalization(name="batchnorm" + suffix + "b")(X)
    X = Conv2D(f, 3, padding="same", kernel_initializer="one", dilation_rate=(3, 3),
               name="conv" + suffix + "b")(X)
    X = Activation("relu", name="relu" + suffix + "b")(X)

    return X

def atrous_block_residual_conv1x1(X, f, level_number, direction, batchnorm=0, dilations=(1,)):
    cell_outputs = []
    suffix = "_" + direction + "_" + str(level_number)

    # Atrous convolutions
    for d in dilations:
        cell_outputs.append(atrous_single_cell(X, f, level_number, direction, batchnorm, d))

    # Shortcut
    cell_outputs.append(Conv2D(f, 1, kernel_initializer="he_normal", name="conv" + suffix + "_short")(X))
    X = Add(name="add" + suffix)(cell_outputs)

    return X

def atrous_block_residual_zeropad(X, f, level_number, direction, batchnorm=0, dilations=(1,)):
    cell_outputs = []
    suffix = "_" + direction + "_" + str(level_number)

    # Atrous convolutions
    for d in dilations:
        cell_outputs.append(atrous_single_cell(X, f, level_number, direction, batchnorm, d))

    # Shortcut
    shortcut_channels = X.shape.as_list()[-1]
    if f >= shortcut_channels:
        identity_weights = np.eye(shortcut_channels, f, dtype=np.float32)
        X = Conv2D(f, kernel_size=1, strides=1, use_bias=False, trainable=False,
                          kernel_initializer=initializers.Constant(value=identity_weights),
                          name="zeropad" + suffix)(X)
    else:
        identity_weights = np.eye(f, shortcut_channels, dtype=np.float32)
        for i in range(len(cell_outputs)):
            cell_outputs[i] = Conv2D(shortcut_channels, kernel_size=1, strides=1, use_bias=False, trainable=False,
                                     kernel_initializer=initializers.Constant(value=identity_weights),
                                     name="zeropad" + suffix + "_" + str(i))(cell_outputs[i])

    cell_outputs.append(X)
    X = Add(name="add" + suffix)(cell_outputs)

    return X

######## U-Net
def unet(input_shape, levels, f1, block_type="vanilla", batchnorm=False, downsampling="maxpool", classes=1,
         dilation_scheme=None):
    inputs = Input(input_shape)
    block_dict = {
        "vanilla": vanilla_block,
        "residual_conv": residual_conv_block,
        "residual_concat": residual_concat_block,
        "residual_zeropad": residual_zeropad_block,
        "atrous_residual_conv1x1": atrous_block_residual_conv1x1,
        "atrous_residual_zeropad": atrous_block_residual_zeropad
    }
    copy = []
    if not(block_type.startswith("atrous")):
        dilation_scheme = [[1] for _ in range(levels)]

    # ENCODER
    X = block_dict[block_type](inputs, f1, 1, "down", batchnorm * 1, dilation_scheme[0])
    copy.append(X)
    if downsampling == "maxpool":
        X = MaxPooling2D((2,2), name="pool_1")(X)
    elif downsampling.startswith("conv"):
        X = Conv2D(f1, kernel_size=int(downsampling[4]), strides=2, padding="same",
                   kernel_initializer="he_normal", name="pool_1")(X)
    for level_number in range(2, levels):
        X = block_dict[block_type](X, f1 * 2**(level_number-1), level_number, "down", batchnorm * 2, dilation_scheme[level_number-1])
        copy.append(X)
        if downsampling == "maxpool":
            X = MaxPooling2D((2,2), name="pool_" + str(level_number))(X)
        elif downsampling.startswith("conv"):
            X = Conv2D(f1 * 2**(level_number-1), kernel_size=int(downsampling[4]), strides=2, padding="same",
                       kernel_initializer="he_normal", name="pool_" + str(level_number))(X)

    # BRIDGE
    X = block_dict[block_type](X, f1 * 2**(levels-1), levels, "bridge", batchnorm * 2, dilation_scheme[levels-1])

    # DECODER
    for level_number in reversed(range(1, levels)):
        X = UpSampling2D((2,2), name="up_" + str(level_number))(X)
        X = Concatenate(axis=3, name="merge_" + str(level_number))([X, copy[level_number-1]])
        X = block_dict[block_type](X, f1 * 2**(level_number-1), level_number, "up", batchnorm * 2, dilation_scheme[level_number-1])

    X = Conv2D(classes+1, 1, name="conv_out")(X)

    if classes == 1:
        outputs = Activation("sigmoid", name="sigmoid")(X)
    else:
        outputs = Lambda(lambda x: keras.activations.softmax(x, axis=-1))(X)

    model = Model(inputs, outputs)
    return model