import pickle
import numpy as np
import sys
import h5py
import os
import matplotlib.pyplot as plt
import cv2
import keras
import tensorflow as tf
from matplotlib.patches import Circle

from spatial_autoencoder_SEQ_tf import make_autoencoder, preprocess_sequence, smooth_loss, grey_downsample

MAKE_GRAYSCALE = False
work_dir = "knut/"
project = "pepperBig_spatialTest/"#"dev_env"
save_every_t = 10
display_result = not False
visualize_convs = True
weight_file = "weights_tf" #for outputing weights of the net in a file....


lr = 0.0001
initial_alpha = 7.0
down_factor = 4
minibatch_size = 32
n = 1000 #numbre of data vectors per file
n_epochs = 4100
batch_normalization = not True
disable_avg = False
weighted_loss = True

def load_file(file, make_gray=False, resize=None):
    with open(file.name,'rb') as f:
        raw_data = pickle.load(f)
    if make_gray:
        print("Graying not implemented...")
        exit()
        raw_data = np.mean(raw_data, axis=3)
        raw_data = raw_data.reshape(raw_data.shape+(1,))
    if resize is not None:
        print("Resizing not implemented yet...")
        exit()
        data = np.empty((n,)+size)
        for i in range(n):
            small = cv2.resize(raw_data[i,:,:,:3], None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA) / 255.0
            if MAKE_GRAYSCALE:
                small = small[:,:,np.newaxis]
            data[i,:,:,:] = small
    else:
        data = raw_data
    data = data.astype(np.float32)/255.0
    n = data.shape[0]
    avg = np.mean(data,axis=0)[np.newaxis,:]
    if batch_normalization:
        avg *= 0.0
    avg_block = np.concatenate((avg,)*n,axis=0)
    return data, n, avg, avg_block

############
############
############
############


files = sys.argv[2:]
n_files = len(sys.argv[2:])
mode = sys.argv[1]
assert mode in ["--train", "--test"], "Invalid mode..."
training = True if mode=="--train" else False
testing = not training
file_idx = 0

size = (96,96,3)

# size = (28,28,3)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist = mnist.train.images.reshape((-1,28,28))[:1000,:,:,np.newaxis]
# mnist = np.concatenate((mnist,mnist,mnist), axis=3)
# avg = np.mean(mnist, axis=0)[np.newaxis,:,:,:]
# avg_block = np.concatenate((avg,)*1000,axis=0)

with tf.Session() as session:
    ''' Inputs '''
    input_tf = tf.placeholder(shape=(None,3)+size, dtype=tf.float32)
    avg_tf = tf.placeholder(shape=(None,)+size, dtype=tf.float32)

    ''' Dataflow '''
    autoencoder_input_tf = input_tf if disable_avg else input_tf-tf.expand_dims( avg_tf, 1 )
    return_tensors = make_autoencoder(
                                        autoencoder_input_tf,
                                        alpha=initial_alpha,
                                        down_factor=down_factor,
                                        size=size,lr=lr,
                                        bn=batch_normalization,
                                        sess=session,
                                        use_dense_decoder=False
                                      )
    decoded_tf, snoop_tf, positions_tf, alpha_tf, train_mode_tf = return_tensors
    position_t_minus1_tf, position_t_tf, position_t_plus1_tf = positions_tf

    _, encoder_variance_tf = tf.nn.moments(snoop_tf, (1,2))
    output_tf = decoded_tf if disable_avg else decoded_tf + grey_downsample(avg_tf, size)
    ''' Loss '''
    if weighted_loss:
        w = tf.abs(avg_tf-input_tf[:,1,:,:,:])
        w = tf.layers.max_pooling2d(w, down_factor, down_factor, padding='same')
        w = tf.reduce_mean(w, axis=-1, keep_dims=True)
        mean_w, _ = tf.nn.moments( w, (1,2,3), shift=None, keep_dims=True)
        loss_weights_tf = 0.5*( 1+w/(mean_w+10**-3) )
    else:
        loss_weights_tf = 1.0
    error_loss_tf = tf.losses.mean_squared_error(output_tf, grey_downsample(input_tf[:,1,:,:,:], size), weights=loss_weights_tf)
    smooth_loss_tf = smooth_loss( positions_tf )
    variance_loss_tf = tf.reduce_mean(tf.clip_by_value(encoder_variance_tf ,0,0.01))
    c1,c2,c3 = 1.0, 0.1, 0.0
    loss_tf = c1*error_loss_tf + c2*smooth_loss_tf + c3*variance_loss_tf

    ''' Training/Saver/Init ops '''
    training_ops = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_tf)
    saver = tf.train.Saver()
    # train_writer = tf.summary.FileWriter( './logs/1/train ', session.graph)
    init_ops = tf.global_variables_initializer()

    if training:
        session.run(init_ops)
        # tf.summary.histogram('loss', loss_tf)
        # tf.summary.image('snoop', snoop_tf[0,:,:,:3])

        T=-1
        total_samples = 0
        for t in range(n_epochs):
            for infile in os.listdir(work_dir+project+"/data"):
                T+=1
                with open(work_dir+project+"/data/"+infile,'rb') as file:
                    raw_data, n, avg, avg_block = load_file(file)
                data = preprocess_sequence(raw_data,size)
                idx=0
                tot_loss = 0
                merge = tf.summary.merge_all()
                print("[",end='',flush=True)
                while idx<n-1:
                    feed_dict={
                                input_tf : data[idx:min(n,idx+minibatch_size),:,:,:],
                                avg_tf : avg_block[idx:min(n,idx+minibatch_size),:,:,:],
                                train_mode_tf : True,
                               }
                    output, loss_weights, _, snoop,loss = session.run([output_tf, loss_weights_tf, training_ops, snoop_tf, loss_tf], feed_dict=feed_dict)

                    # print("DebuG :: {} || {} || {} || {}".format(np.amax(output), np.amax(loss_weights), np.amax(snoop), np.amax(data[idx:min(n,idx+minibatch_size),:,:,:]) ))

                    tot_loss += minibatch_size * loss
                    idx += minibatch_size
                    print("-",end='',flush=True)
                print("] :: ",end='',flush=True)
                # train_writer.add_summary(summary, T)
                total_samples += n
                print("t={} -> n={} | loss={}".format(T,total_samples,tot_loss/n) )

                if T%save_every_t == 0:
                    path = work_dir+project+"/nets_tf/AE_tf_{}".format(10000+T)
                    print("Saving net ({})...".format(path),end='',flush=True)
                    save_path = saver.save(session, path)
                    print("[x]")

    ''' Save weights to a file '''
    if testing:
        idx = 0
        print("Layers loaded:")
        for net in files:
            print("net: {}".format(net))
            saver.restore(session,net)
            tensor_names = tf.trainable_variables()
            weights = session.run(tensor_names, feed_dict=None)
            layer_names = [x.name.split("/")[0] for x in tensor_names]
            weight_dict={}
            for i,x in enumerate(layer_names):
                weight_dict[x] = { 'layer' : x, 'weights' : [] }
            for i,x in enumerate(layer_names):
                weight_dict[x]['weights'].append( weights[i] )
            for x in weight_dict:
                x['weights'] = x.sort(key=lambda x:x.size)
                print(x, weight_dict[x])
            file_output = list(weight_dict.values())
            print("Saving weights to {}".format(weight_file+"{}.pkl".format(idx)))
            with open(weight_file+".pkl",'wb') as out_file:
                pickle.dump(file_output, out_file, pickle.HIGHEST_PROTOCOL)

    ''' Compare inputs and outputs '''
    if testing:
        print("Comparing input w. output...")
        for net in files:
            print(net)
            saver.restore(session,net)
            for infile in os.listdir(work_dir+project+"/data"):
                with open(work_dir+project+"/data/"+infile,'rb') as file:
                    data, n, avg, _ = load_file(file)
                for i in range(n):
                    feed_dict={
                                input_tf : data[i,:,:,:][np.newaxis,:],
                                avg_tf : avg,
                                train_mode_tf : False,
                               }
                    output,snoop, positions, loss = session.run([output_tf, snoop_tf, position_tf, loss_tf], feed_dict=feed_dict)
                    org = data[i,:,:,:]
                    clone = output[0]
                    snoop_layers = snoop[0]
                    positions = positions[0]
                    e = loss #np.sqrt(np.sum(np.square(np.abs(org-clone)),axis=2)).reshape(-1)
                    print("Mean error: {}    Max error: {}".format(e.mean(),e.max()))
                    if display_result:
                        for j in range(positions.shape[1]):
                            print("Feature {} pos: ({},{}) spread: {}".format(j,positions[0,j], positions[1,j], positions[2,j]))
                        snoop_destack_tuple = (org,clone)
                        if snoop_layers.shape[-1]%3 != 0:
                            snoop_layers = np.concatenate((snoop_layers, np.zeros( (size[0], size[1],3-snoop_layers.shape[-1]%3)) ), axis=-1)
                        limit = int(snoop_layers.shape[-1]/3)
                        snoop_destack_tuple += tuple([ (0.2**(i%2))+(-1)**(i%2)*snoop_layers[:,:,3*i:3*(i+1)] for i in range(limit)])
                        n = len(snoop_destack_tuple)
                        h = min(4,int(np.sqrt(n)))
                        if n%h != 0:
                            snoop_destack_tuple += (np.zeros( (size[0], size[1],3) ) )*(h-n%h)
                            n+=h-n%h
                        img = np.concatenate(snoop_destack_tuple[0:int(n/h)],axis=1)
                        idx = int(n/h)
                        for i in range(1,h):
                            x = np.concatenate( snoop_destack_tuple[idx:min(idx+int(n/h),len(snoop_destack_tuple))],axis=1)
                            img = np.concatenate(  (img,x) , axis=0 )
                            idx += int(n/4)
                        if not visualize_convs:
                            img = np.concatenate( (org,clone), axis=1 )

                        fig,ax = plt.subplots(1)
                        ax.set_aspect('equal')
                        ax.imshow(img)
                        # ''' Visualize the features detected! '''
                        for x,y,r in zip(positions[0,:],positions[1,:],positions[2,:]):
                             X = 0.5*(1+x)
                             Y = 0.5*(1+y)
                             radius = 7#0.5*r
                             transparency = 0.3#min(1, r*0.02)

                             c = Circle((size[0]*X,size[1]*Y), radius=radius, fill=False )
                             c.set_alpha(transparency)
                             c.set_antialiased(True)
                             c.set_ec(np.random.rand(3))
                             ax.add_patch(c)
                        plt.show()
