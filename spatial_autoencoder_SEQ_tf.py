import keras
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, UpSampling2D, BatchNormalization, Activation
from keras import backend as K
import numpy as np
import tensorflow as tf
import spatial_softmax
USE_POOLING = True

def grey_downsample(x, in_size, down_factor=2):
    z = tf.reduce_mean(x, axis=-1, keep_dims=True)
    z = tf.image.resize_images(z, (in_size[0]/down_factor, in_size[1]/down_factor))
    return z

def preprocess_sequence(x,size):
    n = x.shape[0]
    ret_x = np.empty((n-2,3)+size)
    ret_x[:,0,:,:,:] = x[:-2 ,:,:,:] #for t-1
    ret_x[:,1,:,:,:] = x[1:-1,:,:,:] #for t
    ret_x[:,2,:,:,:] = x[2:  ,:,:,:] #for t+1
    return ret_x

def spatial_soft_argmax(z,size, alpha=1.0):
    softmax = spatial_softmax.spatial_softmax(z, temperature=alpha, trainable=True)
    x = tf.reshape(softmax[:,::2], (-1,1,size[2]))
    y = tf.reshape(softmax[:,1::2], (-1,1,size[2]))
    alpha_tf = None
    _, var = tf.nn.moments( z, (1,2), shift=None, name=None, keep_dims=True)
    var = tf.reshape(var, (-1,1,size[2]))
    return tf.concat([x,y,var], axis=1), alpha_tf

def smooth_loss(x):
    a, b, c = x
    x_t_minus1 = a[:,:2,:]
    x_t        = b[:,:2,:]
    x_t_plus1  = c[:,:2,:]
    smooth_term = tf.reduce_sum( tf.square((x_t - x_t_minus1) - (x_t_plus1 - x_t)), axis=1 )
    return tf.reduce_mean(smooth_term, axis=0)



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
    feature_map = 0.7 - distance#spread-spread*distance
    return feature_map

def space_blocks(size):
    a = np.tile( 2*(np.arange(size[0], dtype=np.float32)/size[0]-1).reshape((size[0],1,1)), (1,size[1],size[2]) )
    b = np.tile( 2*(np.arange(size[1], dtype=np.float32)/size[1]-1).reshape((1,size[1],1)), (size[0],1,size[2]) )
    return tf.convert_to_tensor(a),tf.convert_to_tensor(b)

def max_abs_error(y_true, y_pred):
    return K.max(K.abs( y_true - y_pred ))

def custom_loss(y_true, y_pred):
    a = 0.1
    b = 1.0
    return a*max_abs_error(y_true, y_pred) + b*keras.losses.mean_absolute_error(y_true, y_pred)

def make_autoencoder(input_tensor, size, alpha=1.0, lr=0.02,bn=False, sess=None, use_dense_decoder=False):
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    training = tf.placeholder(dtype=tf.bool, name='training')
    conv_depth_1 = 32
    conv_depth_2 = 32
    conv_depth_3 = 64
    size_1 = (5,5)#(8,8)
    size_2 = (5,5)
    size_3 = (5,5)
    stride_1 = 1
    stride_2 = 1
    stride_3 = 1
    def make_convs(in_tensor, reuse=None):
            x = tf.layers.conv2d(
                             in_tensor,
                             conv_depth_1,
                             name='conv1',
                             padding='same',
                             kernel_size=size_1,
                             strides=stride_1,
                             activation=tf.nn.elu,
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             reuse=reuse
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
                             bias_initializer=initializer,
                             reuse=reuse
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
                             bias_initializer=initializer,
                             reuse=reuse
                            )
            x = tf.layers.dropout(x, rate=0.2, training=training)
            return x

    ''' These are the inputs: 3 consecutive time frames... '''
    x_t_minus1 = input_tensor[:,0,:,:,:]
    x_t        = input_tensor[:,1,:,:,:]
    x_t_plus1  = input_tensor[:,2,:,:,:]
    ''' Visual features of them all '''
    fx_t_minus1 = make_convs(x_t_minus1, reuse=None)
    fx_t        = make_convs(x_t       , reuse=True)
    fx_t_plus1   = make_convs(x_t_plus1  , reuse=True)

    ''' He we smuggle out some information... '''
    snoop = fx_t
    encoded_x_t_minus1, _ = spatial_soft_argmax(fx_t_minus1,(size[0],size[1],conv_depth_3), alpha=alpha)
    encoded_x_t, alpha_tf = spatial_soft_argmax(fx_t,       (size[0],size[1],conv_depth_3), alpha=alpha)
    encoded_x_t_plus1, _  = spatial_soft_argmax(fx_t_plus1, (size[0],size[1],conv_depth_3), alpha=alpha)
    positions = (encoded_x_t_minus1, encoded_x_t, encoded_x_t_plus1)

    if use_dense_decoder:
        ''' As in paper... '''
        x = tf.contrib.layers.flatten(encoded_x_t)
        x = tf.layers.dense(x, 1024, activation=tf.nn.elu, kernel_initializer=initializer, bias_initializer=initializer)
        x = tf.layers.dense(x, size[0]*size[1]*size[2], activation=tf.nn.tanh, kernel_initializer=initializer, bias_initializer=initializer)
        output = tf.reshape(x, (-1, size[0], size[1],size[2]))
        return output, snoop, positions, alpha_tf, training
    else:
        ''' Another way of doing it... '''
        x = position_decoder(encoded_x_t,(size[0],size[1],conv_depth_3))

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
                         activation=tf.nn.tanh,
                         kernel_initializer=initializer,
                         bias_initializer=initializer
                        )
        x = tf.layers.dropout(x, rate=0.2, training=training)

        output = x

        return output, snoop, positions, alpha_tf, training
