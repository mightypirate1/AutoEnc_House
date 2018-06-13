import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate, UpSampling2D
from keras import backend as K

USE_POOLING = True

def max_abs_error(y_true, y_pred):
    return K.max(K.abs( y_true - y_pred ))
def custom_loss(y_true, y_pred):
    a = 1.0
    b = 1.0
    return a*max_abs_error(y_true, y_pred) + b*keras.losses.mean_absolute_error(y_true, y_pred)

def make_autoencoder(size,lr=0.02):
    initializer = keras.initializers.glorot_uniform()

    # loss_fcn = custom_loss
    loss_fcn = keras.losses.mean_absolute_error

    optimizer = Adam(lr=lr)
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

    bottleneck_size = 256
    bottleneck_activity_regularizer = keras.regularizers.l1(0.0)

    (fy,fx) = (2,2) if USE_POOLING else (1,1)

    input = Input(shape=size)
    x = Convolution2D(conv_depth_1, size_1, strides=(stride_1,stride_1), padding='same', kernel_initializer=initializer)(input)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(conv_depth_2, size_2, strides=(stride_2,stride_2), padding='same', kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(conv_depth_3, size_3, strides=(stride_3,stride_3), padding='same', kernel_initializer=initializer)(input)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)

    # x = MaxPooling2D((stride_3,stride_3))(x)
    x = Flatten()(x)

    encoded = Dense(bottleneck_size, use_bias=False, activity_regularizer=bottleneck_activity_regularizer, kernel_initializer=initializer)(x)

    x = Dense(int(96/stride_3)*int(96/stride_3)*3, kernel_initializer=initializer)(encoded)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)
    x = keras.layers.Reshape((int(96/stride_3), int(96/stride_3), 3))(x)
    x = UpSampling2D((stride_3,stride_3))(x)

    x = Convolution2D(conv_depth_3, size_3, strides=(stride_3,stride_3), padding='same', kernel_initializer=initializer)(input)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)


    x = Convolution2D(conv_depth_2, size_2, padding='same', kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)
    x = UpSampling2D((stride_1,stride_1))(x)

    x = Convolution2D(conv_depth_1, size_1, padding='same', kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    x = Dropout(0.2)(x)

    output = Convolution2D(size[2], (1,1), padding='same', activation='sigmoid', kernel_initializer=initializer)(x)
    model = Model(input,output)

    model.compile(optimizer=optimizer, loss=loss_fcn)
    return model
