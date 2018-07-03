import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, UpSampling2D, BatchNormalization, Activation
from keras import backend as K
import numpy as np
import tensorflow as tf

USE_POOLING = True

def spatial_soft_argmax(z,size, alpha=1.0):
    pos_x, pos_y = space_blocks(size)
    mean, var = tf.nn.moments( z, (1,2), shift=None, name=None, keep_dims=True)
    var = var[:,0,:,:]
    tmp = z-mean
    alpha_tf = tf.get_variable('alpha_tensor', trainable=True, shape=(1,1,z.shape[3]), initializer=tf.constant_initializer(alpha*np.ones((1,1,z.shape[3]))) )
    exp_z = tf.exp(alpha_tf*z)
    weights = tf.reduce_sum(exp_z, axis=1, keep_dims=True)
    weights = tf.reduce_sum(weights, axis=2, keep_dims=True)
    softmax = tf.truediv(exp_z, weights)
    map_x = pos_x * softmax
    map_y = pos_y * softmax
    x = tf.reduce_sum( tf.reduce_sum(map_x, axis=1, keep_dims=True), axis=2, keep_dims=False )
    y = tf.reduce_sum( tf.reduce_sum(map_y, axis=1, keep_dims=True), axis=2, keep_dims=False )
    return tf.concat([x,y,var], axis=1), alpha_tf

def position_decoder(z,size):
    ''' Takes a tensor z of shape (samples, 2 , c) where
        the 2nd dimention refers to x,y coordintates. I.e. x,y coordintates for
        c number of features. Returns feature maps of desired size
        (but the number of channels need still be c, so size=(i,j,c)), where the
        point x,y gets value close to 1, and further away less and less.
        '''
    pos_x, pos_y = space_blocks(size)
    x = tf.reshape(z[:,0,:], (-1,1,1,size[2]))
    y = tf.reshape(z[:,1,:], (-1,1,1,size[2]))
    spread = tf.reshape(z[:,2,:], (-1,1,1,size[2]))
    x_coords = tf.multiply(x, tf.ones(size))
    y_coords = tf.multiply(y, tf.ones(size))
    pos_x, pos_y = space_blocks(size)
    delta_x_squared = tf.square(pos_x-x_coords)
    delta_y_squared = tf.square(pos_y-y_coords)
    distance = tf.sqrt(delta_x_squared+delta_y_squared)
    feature_map = spread-spread*distance
    return feature_map

def space_blocks(size):
    a = np.tile( (np.arange(size[0], dtype=np.float32)/size[0]).reshape((size[0],1,1)), (1,size[1],size[2]) )
    b = np.tile( (np.arange(size[1], dtype=np.float32)/size[1]).reshape((1,size[1],1)), (size[0],1,size[2]) )
    return tf.convert_to_tensor(a),tf.convert_to_tensor(b)

def max_abs_error(y_true, y_pred):
    return K.max(K.abs( y_true - y_pred ))

def custom_loss(y_true, y_pred):
    a = 0.1
    b = 1.0
    return a*max_abs_error(y_true, y_pred) + b*keras.losses.mean_absolute_error(y_true, y_pred)

def make_autoencoder(input_tensor, size, alpha=1.0, lr=0.02,bn=False, sess=None, use_dense_decoder=False):
    allow_bias = True
    if sess is not None:
        keras.backend.set_session(sess)
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)

    conv_depth_1 = 32
    conv_depth_2 = 32
    conv_depth_3 = 64

    size_1 = (5,5)#(8,8)
    size_2 = (5,5)
    size_3 = (5,5)
    stride_1 = 1
    stride_2 = 1
    stride_3 = 1

    training = tf.placeholder(dtype=tf.bool, name='training')

    x = tf.layers.conv2d(
                     input_tensor,
                     conv_depth_1,
                     name='conv1',
                     padding='same',
                     kernel_size=size_1,
                     strides=stride_1,
                     activation=tf.nn.elu,
                     kernel_initializer=initializer,
                     bias_initializer=initializer
                    )
    x = tf.layers.dropout(x, rate=0.2, training=training)

    x = tf.layers.conv2d(
                     x,
                     conv_depth_2,
                     name='conv2',
                     padding='same',
                     kernel_size=size_2,
                     strides=stride_2,
                     activation=tf.nn.elu,
                     kernel_initializer=initializer,
                     bias_initializer=initializer
                    )
    x = tf.layers.dropout(x, rate=0.2, training=training)

    x = tf.layers.conv2d(
                     x,
                     conv_depth_3,
                     name='conv3',
                     padding='same',
                     kernel_size=size_3,
                     strides=stride_3,
                     activation=tf.nn.elu,
                     kernel_initializer=initializer,
                     bias_initializer=initializer
                    )
    x = tf.layers.dropout(x, rate=0.2, training=training)


    ''' He we smuggle out some information... '''
    snoop = x
    encoded, alpha_tf = spatial_soft_argmax(x,(size[0],size[1],conv_depth_3), alpha=alpha)
    positions = encoded

    ''' Decoder starts here... '''
    if use_dense_decoder:
        x = tf.layers.dense(encoded, 256, activation=tf.nn.elu, kernel_initializer=initializer, bias_initializer=initializer)
        x = tf.layers.dense(x, (size[0]//2)*(size[1]//2)*size[2], activation=tf.nn.sigmoid, kernel_initializer=initializer, bias_initializer=initializer)
        x = tf.reshape(x, (-1, size[0]//2, size[1]//2,size[2]))
        output = tf.image.resize_nearest_neighbor(x, (size[0],size[1]))
        return output, snoop, positions, alpha_tf, training
    ''' ------------------------------------- '''
    else:
        x = position_decoder(encoded,(size[0],size[1],conv_depth_3))

        x = tf.layers.dropout(x, rate=0.2, training=training)


        x = tf.layers.conv2d(
                         x,
                         conv_depth_3,
                         name='deconv1',
                         padding='same',
                         kernel_size=size_3,
                         strides=stride_3,
                         activation=tf.nn.elu,
                         kernel_initializer=initializer,
                         bias_initializer=initializer
                        )
        x = tf.layers.dropout(x, rate=0.2, training=training)

        x = tf.layers.conv2d(
                         x,
                         conv_depth_2,
                         name='deconv2',
                         padding='same',
                         kernel_size=size_2,
                         strides=stride_2,
                         activation=tf.nn.elu,
                         kernel_initializer=initializer,
                         bias_initializer=initializer
                        )
        x = tf.layers.dropout(x, rate=0.2, training=training)

        x = tf.layers.conv2d(
                         x,
                         size[2],
                         name='deconv3',
                         padding='same',
                         kernel_size=size_1,
                         strides=stride_1,
                         activation=tf.nn.elu,
                         kernel_initializer=initializer,
                         bias_initializer=initializer
                        )
        x = tf.layers.dropout(x, rate=0.2, training=training)

        output = x

        return output, snoop, positions, alpha_tf, training
