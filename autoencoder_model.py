import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate, UpSampling2D, BatchNormalization, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

USE_POOLING = True

def max_abs_error(y_true, y_pred):
    return K.max(K.abs( y_true - y_pred ))

def custom_loss(y_true, y_pred):
    a = 0.1
    b = 1.0
    return a*max_abs_error(y_true, y_pred) + b*keras.losses.mean_absolute_error(y_true, y_pred)

def make_autoencoder(size,lr=0.02,bn=False):
    initializer = keras.initializers.glorot_uniform()
    # loss_fcn = custom_loss
    loss_fcn = keras.losses.mean_squared_error#keras.losses.mean_absolute_error

    optimizer = Adam(lr=lr)
    # optimizer = SGD(lr=lr)
    # optimizer = keras.optimizers.Adamax(lr=lr)

    conv_depth_1 = 32
    conv_depth_2 = 32
    conv_depth_3 = 16

    size_1 = (5,5)#(8,8)
    size_2 = (5,5)
    size_3 = (5,5)
    stride_1 = 1
    stride_2 = 1
    stride_3 = 1

    bottleneck_size = 128
    bottleneck_activity_regularizer = keras.regularizers.l1(0.0)

    (fy,fx) = (2,2) if USE_POOLING else (1,1)

    input = Input(shape=size)
    x = Convolution2D(conv_depth_1,
                      size_1,
                      use_bias=False,
                      name='conv_1',
                      strides=(stride_1,stride_1),
                      padding='same',
                      kernel_initializer=initializer) (input)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(conv_depth_2,
                      size_2,
                      use_bias=False,
                      name='conv_2',
                      strides=(stride_2,stride_2),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(conv_depth_3,
                      size_3,
                      use_bias=False,
                      name='conv_3',
                      strides=(stride_3,stride_3),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    snoop = x
    x = Dropout(0.2)(x)

    # x = MaxPooling2D((stride_3,stride_3))(x)
    x = Flatten()(x)

    encoded = Dense(bottleneck_size,
                    name='bottleneck',
                    use_bias=False,
                    activity_regularizer=bottleneck_activity_regularizer,
                    kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)
    x = Dense(int(size[0]/stride_3)*int(size[1]/stride_3)*3,
                  use_bias=False,
                  name='dense_1',
                  kernel_initializer=initializer)(encoded)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)
    x = keras.layers.Reshape((int(size[0]/stride_3), int(size[1]/stride_3), 3))(x)
    x = UpSampling2D((stride_3,stride_3))(x)

    x = Convolution2D(conv_depth_3,
                      size_3,
                      use_bias=False,
                      name='deconv_1',
                      strides=(stride_3,stride_3),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)


    x = Convolution2D(conv_depth_2,
                      size_2,
                      use_bias=False,
                      name='deconv_2',
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)
    x = UpSampling2D((stride_1,stride_1))(x)

    x = Convolution2D(conv_depth_1,
                      size_1,
                      use_bias=False,
                      name='deconv_3',
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    output = Convolution2D(size[2],
                           (1,1),
                           use_bias=False,
                           name='output',
                           padding='same',
                           activation='softsign',
                           kernel_initializer=initializer)(x)
    model = Model(input,output)
    peephole_model = Model(input,snoop)
    peephole_model.compile(optimizer=optimizer, loss=loss_fcn)
    model.compile(optimizer=optimizer, loss=loss_fcn)
    model.summary()
    return model, peephole_model
