import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, UpSampling2D, BatchNormalization, Activation
from keras import backend as K
import numpy as np

USE_POOLING = True

def spatial_soft_argmax(z,size, alpha=1.0, beta=0.0):
    pos_x, pos_y = space_blocks(size)
    x = Lambda( lambda x : x-K.max(x,axis=(1,2), keepdims=True ))(z) #For numerical stability
    exp_z = Lambda( lambda a : K.exp(alpha*(a+beta)) )(x)
    softmax = Lambda( lambda a : a / K.sum(K.sum(a,axis=1,keepdims=True),axis=2,keepdims=True) )(exp_z)
    x = Lambda(lambda a : K.sum(K.sum(a*pos_x,axis=1,keepdims=True),axis=2,keepdims=False))(softmax)
    y = Lambda(lambda a : K.sum(K.sum(a*pos_y,axis=1,keepdims=True),axis=2,keepdims=False))(softmax)
    X = Concatenate(axis=1)([x,y])
    return X

def position_decoder(z,size, alpha=0.2):
    ''' Takes a tensor z of shape (samples, 2 , c) where
        the 2nd dimention refers to x,y coordintates. I.e. x,y coordintates for
        c number of features. Returns feature maps of desired size
        (but the number of channels need still be c, so size=(i,j,c)), where the
        point x,y gets value close to 1, and further away less and less.
        '''
    pos_x, pos_y = space_blocks(size)
    x = Lambda(lambda x : x[:,0])(z)
    y = Lambda(lambda x : x[:,1])(z)
    x_coords = Lambda( lambda x : K.constant(np.ones(size))*K.reshape(x,(-1,1,1,size[2])) ) (x)
    y_coords = Lambda( lambda x : K.constant(np.ones(size))*K.reshape(x,(-1,1,1,size[2])) ) (y)
    pos_x, pos_y = space_blocks(size)
    delta_x_squared = Lambda( lambda x : K.square(pos_x-x) )(x_coords)
    delta_y_squared = Lambda( lambda y : K.square(pos_y-y) )(y_coords)
    tmp = Lambda ( lambda x : K.concatenate( (K.expand_dims(x,-1),K.expand_dims(delta_y_squared,-1)) ) )(delta_x_squared)
    delta_sq_tot = Lambda( lambda x : K.mean(x,axis=-1) )(tmp)
    dist = Lambda( lambda d : K.sqrt(d) )(delta_sq_tot)
    decoded = Lambda( lambda d : alpha-d )(dist)
    return decoded

def space_blocks(size):
    a = np.tile( (np.arange(size[0])/size[0]).reshape((size[0],1,1)), (1,size[1],size[2]) )
    b = np.tile( (np.arange(size[1])/size[1]).reshape((1,size[1],1)), (size[0],1,size[2]) )
    return K.constant(a),K.constant(b)

def max_abs_error(y_true, y_pred):
    return K.max(K.abs( y_true - y_pred ))

def custom_loss(y_true, y_pred):
    a = 0.1
    b = 1.0
    return a*max_abs_error(y_true, y_pred) + b*keras.losses.mean_absolute_error(y_true, y_pred)

def make_autoencoder(size,lr=0.02,bn=False):

    allow_bias = True

    initializer = keras.initializers.glorot_uniform()
    keras.layers.ELU(alpha=1.0)

    # loss_fcn = custom_loss
    loss_fcn = keras.losses.mean_squared_error#keras.losses.mean_absolute_error

    optimizer = Adam(lr=lr)

    conv_depth_1 = 32
    conv_depth_2 = 32
    conv_depth_3 = 64

    size_1 = (5,5)#(8,8)
    size_2 = (5,5)
    size_3 = (5,5)
    stride_1 = 1
    stride_2 = 1
    stride_3 = 1

    dense_size_1 = 512
    bottleneck_activity_regularizer = keras.regularizers.l1(0.0)
    default_regularizer = keras.regularizers.l2(0.0)

    (fy,fx) = (2,2) if USE_POOLING else (1,1)

    input = Input(shape=size)
    x = Convolution2D(conv_depth_1,
                      size_1,
                      use_bias=allow_bias,
                      kernel_regularizer=default_regularizer,
                      bias_regularizer=default_regularizer,
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
                      use_bias=allow_bias,
                      kernel_regularizer=default_regularizer,
                      bias_regularizer=default_regularizer,
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
                      use_bias=allow_bias,
                      kernel_regularizer=default_regularizer,
                      bias_regularizer=default_regularizer,
                      name='conv_3',
                      strides=(stride_3,stride_3),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)

    ''' He we smuggle out some information... '''
    snoop = x
    encoded = spatial_soft_argmax(x,(size[0],size[1],conv_depth_3))
    positions = encoded
    ''' ------------------------------------- '''

    x = position_decoder(encoded,(size[0],size[1],conv_depth_3))
    x = Dropout(0.2)(x)
    x = Convolution2D(conv_depth_3,
                      size_3,
                      use_bias=allow_bias,
                      kernel_regularizer=default_regularizer,
                      bias_regularizer=default_regularizer,
                      name='deconv_1',
                      strides=(stride_3,stride_3),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)

    x = Convolution2D(3,
                      size_3,
                      use_bias=allow_bias,
                      kernel_regularizer=default_regularizer,
                      bias_regularizer=default_regularizer,
                      name='deconv_2',
                      strides=(stride_3,stride_3),
                      padding='same',
                      kernel_initializer=initializer)(x)
    x = keras.layers.ELU(alpha=1.0)(x)
    if bn:
        x = BatchNormalization(axis=-1)(x)


    output = x

    model = Model(input,output)
    model.compile(optimizer=optimizer, loss=loss_fcn)
    model.summary()

    peephole_model = Model(input,snoop)
    peephole_model.compile(optimizer=optimizer, loss=loss_fcn)

    position_model = Model(input,positions)
    position_model.compile(optimizer=optimizer, loss=loss_fcn)

    return model, peephole_model, position_model
