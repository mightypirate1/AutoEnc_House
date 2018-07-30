import numpy as np
import tensorflow as tf

'''
#######################
First some misc. stuff:
#######################
'''
def grey_downsample(x, in_size, down_factor=4, gray=True):
    z = x
    if gray:
        z = tf.reduce_mean(z, axis=-1, keep_dims=True)
    z = tf.image.resize_images(z, (int(in_size[0]/down_factor), int(in_size[1]/down_factor)))
    return z
def preprocess_sequence(x,size):
    n = x.shape[0]
    ret_x = np.empty((n-2,3)+size)
    ret_x[:,0,:,:,:] = x[:-2 ,:,:,:] #for t-1
    ret_x[:,1,:,:,:] = x[1:-1,:,:,:] #for t
    ret_x[:,2,:,:,:] = x[2:  ,:,:,:] #for t+1
    return ret_x
def smooth_loss(x):
    a, b, c = x
    n_dims = len(a.get_shape().as_list())
    x_t_minus1 = a[:,:2,:] if n_dims == 3 else a
    x_t        = b[:,:2,:] if n_dims == 3 else b
    x_t_plus1  = c[:,:2,:] if n_dims == 3 else c
    smooth_term = tf.reduce_sum( tf.square((x_t - x_t_minus1) - (x_t_plus1 - x_t)), axis=1 )
    return tf.reduce_mean(smooth_term)
def space_blocks(size):
    x = 2*np.arange(size[0], dtype=np.float32).reshape((size[0],1,1))/(size[0]-1)-1
    y = 2*np.arange(size[1], dtype=np.float32).reshape((1,size[1],1))/(size[1]-1)-1
    X = np.tile(x, (1,size[1],1))
    Y = np.tile(y, (size[0],1,1))
    return tf.convert_to_tensor(X),tf.convert_to_tensor(Y)
'''
###################
Encoder transofrms:
###################
'''
def spatial_soft_argmax(z,size, reuse=None, alpha=1.0, sigma=8.0):
    ''' First we compute an actual spatial softmax '''
    with tf.variable_scope("spatial_soft_argmax", reuse=reuse):
        alpha_tf = tf.get_variable("alpha", [1], initializer=tf.constant_initializer(alpha), trainable=True)
    Z = alpha_tf*z
    max_z = tf.reduce_max(Z,     axis=1, keep_dims=True)
    max_z = tf.reduce_max(max_z, axis=2, keep_dims=True)
    exp_z = tf.exp( Z-max_z )
    w = tf.reduce_sum(exp_z, axis=1, keep_dims=True)
    w = tf.reduce_sum(w,     axis=2, keep_dims=True)
    softmax = exp_z / w
    ''' Then we use that softmax to compute expected positions of the features '''
    px, py = space_blocks(Z.get_shape().as_list()[1:])
    x = tf.reduce_sum(px*softmax, axis=2)
    x = tf.reduce_sum(x, axis=1, keep_dims=True)
    y = tf.reduce_sum(py*softmax, axis=2)
    y = tf.reduce_sum(y, axis=1, keep_dims=True)
    alpha_tf = None
    ''' Now we look at the feature maps and determine how much the feature is where its position was esimated. we call this presence. '''
    d_map = dist_map(x,y,Z.get_shape().as_list()[1:])
    gauss = tf.exp( -tf.square(sigma*d_map) )
    gauss_w = tf.reduce_sum(gauss, axis=1, keep_dims=True)
    gauss_w = tf.reduce_sum(gauss_w, axis=2, keep_dims=True)
    gauss_s = (gauss/gauss_w) * softmax
    tmp = tf.reduce_sum(gauss_s, axis=1, keep_dims=True)
    presence = tf.reduce_sum(tmp, axis=2)
    ''' We also want a way of identifying features which are not present at all in the current image. That is, their feature map is flat. '''
    f = softmax
    f_min = tf.reduce_min(f, axis=2)
    f_min = tf.reduce_min(f_min, axis=1, keep_dims=True)
    f_max = tf.reduce_max(f, axis=2)
    f_max = tf.reduce_max(f_max, axis=1, keep_dims=True)
    F = tf.abs(f_max - f_min)
    #f = softmax * (1 - gauss*(2*np.pi/sigma**2))
    #f1 = tf.reduce_sum(f,axis=2)
    #F = tf.reduce_sum(f1,axis=1, keep_dims=True)
    return tf.concat([x, y, presence, F], axis=1)

def dense_spatial_enc(z, size, reuse=None):
    def encode_layer(z, r, sigma=8, alpha=1):
        ''' First we compute an actual spatial softmax (Just to compute presence...)'''
        Z = alpha*tf.expand_dims(z,3)
        max_z = tf.reduce_max(Z,     axis=1, keep_dims=True)
        max_z = tf.reduce_max(max_z, axis=2, keep_dims=True)
        exp_z = tf.exp( Z-max_z )
        w = tf.reduce_sum(exp_z, axis=1, keep_dims=True)
        w = tf.reduce_sum(w,     axis=2, keep_dims=True)
        softmax = exp_z / w

        ''' Then do a mapping from the feature map to x- & y-positions '''
        x = tf.nn.elu(z)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,
                            300,
                            activation=tf.nn.elu,
                            name="hidden_encoder_layer",
                            reuse=r,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True)
                           )
        x_pos = tf.layers.dense(x,
                                1,
                                activation=tf.nn.tanh,
                                activity_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
                                name="encoder_x",
                                reuse=r,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True)
                               )
        y_pos = tf.layers.dense(x,
                                1,
                                activation=tf.nn.tanh,
                                activity_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
                                name="encoder_y",
                                reuse=r,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True)
                               )

        ''' Now we look at the feature maps and determine how much the feature is where its position was esimated. we call this presence. '''
        d_map = dist_map(x_pos,y_pos,Z.get_shape().as_list()[1:])
        gauss = tf.exp( -tf.square(sigma*d_map) )
        gauss_w = tf.reduce_sum(gauss, axis=1, keep_dims=True)
        gauss_w = tf.reduce_sum(gauss_w, axis=2, keep_dims=True)
        gauss_s = (gauss/gauss_w) * softmax
        tmp = tf.reduce_sum(gauss_s, axis=1, keep_dims=True)
        presence = tf.reduce_sum(tmp, axis=2)
        ret = tf.concat([x_pos, y_pos, presence[:,0,:]], axis=1)
        return tf.expand_dims(ret, 2)
    encoded = []
    n_channels = z.get_shape().as_list()[-1]
    r = reuse
    for y in range(n_channels):
        encoded.append(encode_layer(z[:,:,:,y], r))
        r = True
    return tf.concat(encoded, axis=2)

def dense_enc(z, n, size, reuse=None):
    x = tf.nn.elu(z)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x,
                        n,
                        name='encoder_layer',
                        activation=tf.nn.elu,
                        reuse=reuse)
    return x
'''
###################
Decoder transforms:
###################
'''
def dense_decoder(z,size):
    x = tf.layers.dense(z, size[0]*size[1]*size[2])
    return tf.reshape(x,(-1,)+size)
def none_enc(z, size, reuse=None):
    return tf.contrib.layers.flatten(z)
def dist_map(x,y, size): #size should be (w,h,c). x and y should be of size (n,c). output is of size (n,w,h,c)
    x = tf.reshape(x, (-1,1,1,size[2]))
    y = tf.reshape(y, (-1,1,1,size[2]))
    x_coords = tf.multiply(x, tf.ones(size))
    y_coords = tf.multiply(y, tf.ones(size))
    pos_x, pos_y = space_blocks(size)
    delta_x_squared = tf.square(pos_x-x_coords)
    delta_y_squared = tf.square(pos_y-y_coords)
    distance = tf.sqrt(delta_x_squared+delta_y_squared+0.000001)
    return distance
def position_decoder(z,size):
    x = z[:,0,:]
    y = z[:,1,:]
    presence = tf.expand_dims(tf.expand_dims(z[:,2,:], 1), 1)
    distance = dist_map(x,y,size)
    #feature_map = tf.nn.relu(amp-distance)
    #feature_map = tf.exp( -tf.square(distance*presence/8) )
    #feature_map = 0.7 - dist_map(x,y,size)
    feature_map = tf.nn.elu(presence - dist_map(x,y,size))
    return feature_map


'''
#####################
Architecture builder:
#####################
'''
def make_autoencoder(input_tensor, size, settings):
    ''' Parse the settings! '''
    if settings['encoder_transform'] == 'softargmax':
        assert settings['encoder_transf_size'] is None, "ERROR: when using the softargmax, you can not specify its size!"
        encoder_transform_fcn = spatial_soft_argmax
        print("Using encoder_transf: softargmax")
    elif settings['encoder_transform'] == 'dense_spatial':
        print("Using encoder_transf: dense_spatial")
        encoder_transform_fcn = dense_spatial_enc
    elif settings['encoder_transform'] == 'dense':
        print("Using encoder_transf: dense")
        assert settings['encoder_transf_size'] is not None, "ERROR: when using the dense encoder, you must specify its size with 'encoder_transf_size'"
        def dense_enc_wrapper(x, size, reuse=None):
            return dense_enc(x,settings['encoder_transf_size'], size, reuse=reuse)
        encoder_transform_fcn = dense_enc_wrapper
    else:
        raise Exception("Invalid encoder stage specified: {}".format(settings['encoder_transform']))
    if settings['decoder'] not in ['conv', 'dense']:
        raise Exception("Invalid decoder stage specified: {}".format(settings['decoder_transform']))
    if settings['decoder_transform'] is 'distmap':
        decoder_transform_fcn = position_decoder
        print("Using decoder_transf: distmap")
    elif settings['decoder_transform'] is 'dense':
        assert settings['encoder_transform'] is 'dense' and settings['decoder'] is 'conv', "ERROR: using the dense decoder requires dense encoder and convolutional decoder"
        print("Using decoder_transf: dense_decoder")
        decoder_transform_fcn = dense_decoder
    conv_depth_1 = settings['encoder_conv_depth'][0]
    conv_depth_2 = settings['encoder_conv_depth'][1]
    conv_depth_3 = settings['encoder_conv_depth'][2]
    size_1 = settings['encoder_conv_size'][0]
    size_2 = settings['encoder_conv_size'][1]
    size_3 = settings['encoder_conv_size'][2]
    stride_1 = 1
    stride_2 = 1
    stride_3 = 1
    bias_deconv = settings['decoder_use_bias']
    down_factor = settings['down_factor']


    ''' Prepare some functions! '''
    def batch_norm(x):
        return tf.layers.batch_normalization(x, center=True, scale=True, training=training)
    def make_convs(in_tensor, reuse=None):
            x = tf.layers.conv2d(
                             in_tensor,
                             conv_depth_1,
                             name='conv1',
                             padding='valid',
                             kernel_size=size_1,
                             strides=stride_1,
                             activation=tf.nn.elu,
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             reuse=reuse
                            )
            snoop1 = x
            if settings['bn']:
                x = batch_norm(x)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = tf.layers.conv2d(
                             x,
                             conv_depth_2,
                             name='conv2',
                             padding='valid',
                             kernel_size=size_2,
                             strides=stride_2,
                             activation=tf.nn.elu,
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             reuse=reuse
                            )
            snoop2 = x
            if settings['bn']:
                x = batch_norm(x)
            x = tf.layers.dropout(x, rate=0.2, training=training)

            x = tf.layers.conv2d(
                             x,
                             conv_depth_3,
                             name='conv3',
                             padding='valid',
                             kernel_size=size_3,
                             strides=stride_3,
                             activation=None, #tf.nn.tanh, #tf.nn.elu,
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             reuse=reuse
                            )
            snoop3 = x
            if settings['bn']:
                x = batch_norm(x)
            # x = tf.concat([x, -x], axis=3)
            x = tf.layers.dropout(x, rate=0.2, training=training)
            return x, (snoop1,snoop2,snoop3)
    print("building......")
    ''' Build the main architecture! '''
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    training = tf.placeholder(dtype=tf.bool, name='training')

    print("Adding 3 convolutional layers...")
    ''' These are the inputs: 3 consecutive time frames... '''
    x_t_minus1 = input_tensor[:,0,:,:,:]
    x_t        = input_tensor[:,1,:,:,:]
    x_t_plus1  = input_tensor[:,2,:,:,:]
    ''' Visual features of them all '''
    fx_t_minus1, _  = make_convs(x_t_minus1, reuse=None)
    fx_t,    _snoop  = make_convs(x_t       , reuse=True)
    fx_t_plus1 , _  = make_convs(x_t_plus1  , reuse=True)

    ''' He we smuggle out some information... '''
    print("Adding encoder transoform...")
    encoded_x_t_minus1 =  encoder_transform_fcn(fx_t_minus1,(size[0],size[1],conv_depth_3), reuse=None)
    encoded_x_t        =  encoder_transform_fcn(fx_t,       (size[0],size[1],conv_depth_3), reuse=True)
    encoded_x_t_plus1  =  encoder_transform_fcn(fx_t_plus1, (size[0],size[1],conv_depth_3), reuse=True)

    snoop = (x_t,) + _snoop
    positions = (encoded_x_t_minus1, encoded_x_t, encoded_x_t_plus1)

    if settings['decoder'] == 'dense':
        ''' As in paper... '''
        print("Adding DENSE decoder...")
        if settings['decoder_transform'] is not 'none':
            print("ERROR: DENSE decoder requires decoder_transform to be 'none'")
            exit()
        out_activation = tf.nn.tanh if settings['avg_subtraction'] else tf.nn.sigmoid
        out_size = ( -1, int(size[0]/down_factor), int(size[1]/down_factor), 1 if settings['gray'] else 3)

        x = tf.contrib.layers.flatten(encoded_x_t)
        #x = tf.layers.dense(
        #                    x,
        #                    1024,
        #                    activation=tf.nn.elu,
        #                    kernel_initializer=initializer,
        #                    bias_initializer=initializer
        #                   )
        x = tf.layers.dense(
                            x,
                            out_size[1]*out_size[2]*out_size[3],
                            activation=out_activation,
                            kernel_initializer=initializer,
                            bias_initializer=initializer
                           )
        output = tf.reshape(x, out_size)
        return output, snoop, positions, None, training
    elif settings['decoder'] == 'conv':
        ''' Another way of doing it... '''

        if settings['decoder_transform'] not in ['distmap', 'dense']:
            print("ERROR: Convolutional decoder requires decoder_transform to be 'distmap' or 'dense'")
            exit()
        print("Adding decoder transformation...")
        x = decoder_transform_fcn(encoded_x_t,(size[0],size[1],conv_depth_3))
        snoop += (x,)
        print("Adding convolutional decoder...")

        x = tf.layers.dropout(x, rate=0.2, training=training)
        if settings['decoder_transform'] is not 'dense': #The dense decoder has the equivaltence of this layer built into it...
            print("Adding 3 conv-layer decoder....")
            x = tf.layers.conv2d(
                             x,
                             conv_depth_3,
                             name='deconv1',
                             padding='same',
                             kernel_size=size_3,
                             strides=stride_3,
                             activation=tf.nn.elu,
                             kernel_initializer=initializer,
                             bias_initializer=initializer,
                             use_bias=bias_deconv
                            )
            snoop += (x,)
        else:
            print("Adding 2 conv-layers (1 is built into decoder transform)...")
            snoop4 = ()
        if settings['bn']:
            x = batch_norm(x)
        x = tf.layers.dropout(x, rate=0.2, training=training)

        x = tf.layers.conv2d(
                         x,
                         conv_depth_2,
                         name='deconv2',
                         padding='same',
                         kernel_size=size_2,
                         strides=int(stride_2),
                         activation=tf.nn.elu,
                         kernel_initializer=initializer,
                         bias_initializer=initializer,
                         use_bias=bias_deconv
                        )
        snoop += (x,)
        if settings['bn']:
            x = batch_norm(x)
        x = tf.layers.dropout(x, rate=0.2, training=training)

        x = tf.layers.conv2d(
                         x,
                         1 if settings['gray'] else 3,
                         name='deconv3',
                         padding='same',
                         kernel_size=size_1,
                         strides=int(stride_1*(down_factor)),
                         activation=tf.nn.tanh,
                         kernel_initializer=initializer,
                         bias_initializer=initializer,
                         use_bias=bias_deconv
                        )
        snoop += (x,)
        x = tf.layers.dropout(x, rate=0.2, training=training)

        if settings['output_bias']:
            b = tf.get_variable("output_bias", [1, 96, 96, 3], trainable=True)
            snoop += (b,)
            x = x + b
        output = x
        snoop += (output,)
        return output, snoop, positions, None, training
