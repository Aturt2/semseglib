from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate, Dropout, Activation, Add, ZeroPadding1D
from keras.models import Model
from keras import backend as K

######## Blocks
def vanilla_block(X, f, level_number, direction, batchnorm=0):
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

def residual_conv_block(X, f, level_number, direction, batchnorm=0):
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

def residual_concat_block(X, f, level_number, direction, batchnorm=0):
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

def zeropad(X, ):
    X_shape = X.shape.as_list()[-1]
    shortcut_shape = shortcut.shape.as_list()[-1]
    zeros = K.zeros_like(X)
    zeros = zeros[:, :, :, :(X_shape - shortcut_shape)]

def residual_zeropad_block(X, f, level_number, direction, batchnorm=0):
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

    print(X.shape.as_list()[-1], shortcut.shape.as_list()[-1])

    X_shape = X.shape.as_list()[-1]
    shortcut_shape = shortcut.shape.as_list()[-1]
    if X_shape >= shortcut_shape:
        zeros = K.zeros_like(X)
        zeros = zeros[:, :, :, :(X_shape - shortcut_shape)]
        shortcut = Concatenate(axis=3, name="merge_zeros" + suffix)([zeros, shortcut])
    else:
        zeros = K.zeros_like(shortcut)
        zeros = zeros[:, :, :, :(shortcut_shape - X_shape)]
        X = Concatenate(axis=3, name="merge_zeros" + suffix)([zeros, X])
    X = Add(name="add" + suffix)([X, shortcut])

    return X

######## U-Net
def unet(input_shape, levels, f1, block_type="vanilla", batchnorm=False):
    inputs = Input(input_shape)
    block_dict = {
        "vanilla": vanilla_block,
        "residual_conv": residual_conv_block,
        "residual_concat": residual_concat_block,
        "residual_zeropad": residual_zeropad_block
    }
    copy = []

    # ENCODER
    X = block_dict[block_type](inputs, f1, 1, "down", batchnorm * 1)
    copy.append(X)
    X = MaxPooling2D((2,2), name="pool_1")(X)
    for level_number in range(2, levels):
        X = block_dict[block_type](X, f1 * 2**(level_number-1), level_number, "down", batchnorm * 2)
        copy.append(X)
        X = MaxPooling2D((2,2), name="pool_" + str(level_number))(X)

    # BRIDGE
    X = block_dict[block_type](X, f1 * 2**(levels-1), levels, "bridge", batchnorm * 2)

    # DECODER
    for level_number in reversed(range(1, levels)):
        X = UpSampling2D((2,2), name="up_" + str(level_number))(X)
        X = Concatenate(axis=3, name="merge_" + str(level_number))([X, copy[level_number-1]])
        X = block_dict[block_type](X, f1 * 2**(level_number-1), level_number, "up", batchnorm * 2)

    X = Conv2D(1, 1, name="conv_out")(X)
    outputs = Activation("sigmoid", name="sigmoid")(X)

    model = Model(inputs, outputs)
    return model