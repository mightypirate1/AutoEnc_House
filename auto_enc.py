import pickle
import numpy as np
import sys
import h5py
import os
import matplotlib.pyplot as plt
import cv2
import scipy
import keras
import tensorflow as tf
from matplotlib.patches import Circle
import code
import math
import scipy
import settings as s
import collections
from docopt import docopt
from autoencoder_modules import make_autoencoder, preprocess_sequence, smooth_loss, grey_downsample, space_blocks
work_dir = "knut/"


docstring = '''AutoEnc_House.
Usage:
  auto_enc.py --train [--settings=<settings>]
  auto_enc.py --test [--settings=<settings>] <nn>
'''
arguments = docopt(docstring)
print(arguments)

if not arguments['--settings']:
    #Default settings if none were specified!
    settings = s.parse_conf( s.spatial_ae_conv )
else:
    settings = s.parse_conf( s.get_conf( arguments['--settings'] ) )

'''  <THESE FUNCTIONS ARE JUST FOR DEBUGGING....> '''
def sb(size):
    x = 2*np.arange(size[0]).reshape((size[0],1,1))/(size[0]-1)-1
    y = 2*np.arange(size[1]).reshape((1,size[1],1))/(size[1]-1)-1
    X = np.tile(x, (1,size[1],1))
    Y = np.tile(y, (size[0],1,1))
    return X,Y
def sf(x, alpha=1.0):
    if len(x.shape) == 4:
        if x.shape[0] != 1:
            print("!")
            exit()
        else:
            x = x.reshape(x.shape[1:])
    x = alpha * x
    px,py = sb(x.shape)
    x = x - x.max()
    exp_x = np.exp(x)
    w = np.sum(exp_x, axis=(0,1))
    softmax = exp_x / w
    x = np.sum(softmax*px,axis=(0,1))
    y = np.sum(softmax*py,axis=(0,1))
    return softmax, (x,y)
''' </THESE FUNCTIONS ARE JUST FOR DEBUGGING....> '''


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
    avg_block = np.concatenate((avg,)*n,axis=0)
    return data, n, avg, avg_block

class DoneSignal(Exception):
    def __init__(self,message, data=None):
        self.message = message
        self.data = data
    def debug_print(self):
        if self.data is not None:
            for d in data:
                print(d)

def create_folders():
    paths = [
            work_dir,
            work_dir+project,
            work_dir+project+"nets_tf",
            work_dir+project+"init"
            ]
    print("---", flush=True)
    print("Checking directory structure...")
    for p in paths:
        if not os.path.isdir(p):
            os.makedirs(p)
            print("Created path: {}".format(p))
        else:
            print("Reusing existing: {}".format(p))
    if not os.path.isdir(work_dir+project+"data"):
        print("WARNING: no data folder exists for the current project. Did you run the data-gatherer? ([enter] to ignore)")
        input()

def save_weights_to_file(file_path):
    tensor_names = tf.trainable_variables()
    weights = session.run(tensor_names, feed_dict=None)
    layer_names = [x.name.split("/") for x in tensor_names]
    weight_dict={}
    for i,x in enumerate(layer_names):
        weight_dict[x[0]] = { 'layer' : x[0], 'weights' : {} }
    for i,x in enumerate(layer_names):
        if 'batch' in x[0] or 'conv' in x[0]:
            weight_dict[x[0]]['weights'][ x[1].split(":")[0] ] = weights[i]
    file_output = list(weight_dict.values())
    for x in file_output:
        print("Saving layer: {} ({})".format(x['layer'], [x['weights'][w].shape for w in x['weights']]))
    print("Saving weights to {}".format(file_path))
    with open(file_path,'wb') as out_file:
        pickle.dump(file_output, out_file, pickle.HIGHEST_PROTOCOL)

def get_data_from_files(files, start_idx, n_samples):
    data_list = []
    avg = None
    total_samples = 0
    stop_idx = None
    for i in range(len(files)):
        idx = (i+start_idx)%len(files)
        with open(files[idx],'rb') as file:
            raw_data, _n, new_avg, _ = load_file(file)
            if _n<3:
                continue
            avg = (total_samples*avg+_n*new_avg)/(total_samples+_n) if avg is not None else new_avg
            total_samples += _n
            _data = preprocess_sequence(raw_data,size)
            data_list.append(_data)
            if total_samples > n_samples:
                stop_idx = idx+1
                break
    data = np.concatenate(data_list, axis=0)
    n = data.shape[0]
    data = data[np.random.permutation(n),:,:,:,:]
    return data, avg, n, stop_idx

############
############
############
############

files = sys.argv[2:]
files = arguments['<nn>']
training = arguments['--train']
testing = arguments['--test']
project = settings['project_folder']
save_every_t = 10
weight_file = "weights_tf" #for outputing weights of the net in a file....
avg_file = work_dir + project + "nets_tf/" + "avgfile_tf"
size = (96,96,3)
create_folders()

with tf.Session() as session:
    ''' Inputs '''
    input_tf = tf.placeholder(shape=(None,3)+size, dtype=tf.float32)
    avg_tf = tf.placeholder(shape=(None,)+size, dtype=tf.float32)

    ''' Dataflow '''
    autoencoder_input_tf = input_tf if not settings['avg_subtraction'] else input_tf - tf.expand_dims( avg_tf, 1 )
    return_tensors = make_autoencoder(
                                        autoencoder_input_tf,
                                        size,
                                        settings
                                     )
    decoded_tf, snoop_tf, positions_tf, alpha_tf, train_mode_tf = return_tensors
    # snoop1_tf, snoop2_tf, snoop3_tf = snoop_tf
    position_t_minus1_tf, position_t_tf, position_t_plus1_tf = positions_tf

    output_tf = decoded_tf if not settings['avg_subtraction'] else decoded_tf + grey_downsample(avg_tf, size, down_factor=settings['down_factor'], gray=settings['gray'])
    ''' Loss '''
    if settings['weighted_loss']:
        w = tf.abs(avg_tf-input_tf[:,1,:,:,:])
        w = tf.layers.max_pooling2d(w, settings['down_factor'], settings['down_factor'], padding='same')
        w = tf.reduce_mean(w, axis=-1, keep_dims=True)
        mean_w, _ = tf.nn.moments( w, (1,2,3), shift=None, keep_dims=True)
        loss_weights_tf = ( 1+w/(mean_w+10**-3) )
    else:
        loss_weights_tf = tf.constant(1.0)

    error_loss_tf = tf.losses.mean_squared_error(output_tf, grey_downsample(input_tf[:,1,:,:,:], size, down_factor=settings['down_factor'], gray=settings['gray']), weights=loss_weights_tf)
    smooth_loss_tf = smooth_loss( positions_tf )

    presence_tf = position_t_tf[:,2,:] if settings['encoder_transform'] in ['softargmax', 'dense_spatial'] else tf.constant(1)
    presence_loss_tf = tf.losses.mean_squared_error( presence_tf,
                                                      tf.fill(tf.shape(presence_tf) ,1)                                                 )

    c1,c2,c3 = settings['loss_weights']
    loss_tf = c1*error_loss_tf + c2*smooth_loss_tf + c3*presence_loss_tf

    ''' Training/Saver/Init ops '''
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ##This is to make bn work
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        training_ops = tf.train.AdamOptimizer(learning_rate=settings['lr']).minimize(loss_tf)
    saver = tf.train.Saver()
    # train_writer = tf.summary.FileWriter( './logs/1/train ', session.graph)
    init_ops = tf.global_variables_initializer()

    #Data!
    train_data_files = [work_dir+project+"/data/"+x for x in os.listdir(work_dir+project+"/data")]
    test_data_files = [work_dir+project+"/testdata/"+x for x in os.listdir(work_dir+project+"/testdata")]
    train_file_idx = 0
    test_file_idx = 0

    if training:
        session.run(init_ops)
        # tf.summary.histogram('loss', loss_tf)
        # tf.summary.image('snoop', snoop_tf[0,:,:,:3])
        minibatch_size = settings['minibatch_size']
        total_samples = 0
        avg = np.zeros((1,)+size)

        loss_history = collections.deque(maxlen=40)

        try:
            print("=========================")
            for t in range(settings['n_epochs']):
                #Load data for training & testing
                data, new_avg, n, train_file_idx = get_data_from_files( train_data_files, train_file_idx, 3000 )
                test_data, _, n_train, train_file_idx = get_data_from_files( test_data_files, test_file_idx, 300 )
                avg = (total_samples*avg + n*new_avg) / (total_samples + n)

                #Train on data...
                idx=0
                tot_loss = 0
                last_print = 0
                merge = tf.summary.merge_all()
                print("[",end='',flush=True)
                while idx<n:
                    feed_dict={
                                input_tf : data[idx:min(data.shape[0],idx+minibatch_size),:,:,:],
                                avg_tf : avg,
                                train_mode_tf : True,
                               }
                    output, loss_weights, _,loss, presence_loss = session.run([output_tf, loss_weights_tf, training_ops, loss_tf, presence_loss_tf], feed_dict=feed_dict)
                    tot_loss += (min(n,idx+minibatch_size) - idx ) * loss
                    idx += minibatch_size
                    if idx/n - last_print > 0.05:
                        print("=",end='',flush=True)
                        last_print += 0.05
                print("] :: ",end='',flush=True)
                # train_writer.add_summary(summary, t)
                total_samples += n
                print("trainloss={} | t={} -> n={} | {}".format(tot_loss/n, t, total_samples) )

                #Test-set stuff:
                idx = 0
                test_loss = 0
                last_print = 0
                print("[",end='',flush=True)
                while idx<n_train:
                    feed_dict={
                                input_tf : test_data[idx:min(test_data.shape[0],idx+minibatch_size),:,:,:],
                                avg_tf : avg,
                                train_mode_tf : True,
                               }
                    loss = session.run([loss_tf], feed_dict=feed_dict)
                    test_loss += (min(n_train,idx+minibatch_size) - idx ) * loss[0]
                    idx += minibatch_size
                    if idx/n_train - last_print > 0.05:
                        print("-",end='',flush=True)
                        last_print += 0.05
                print("] :: =",end='',flush=True)
                print("testloss={}".format(test_loss / n_train))
                loss_history.append(test_loss)

                #Save if it's time to save!
                if t%save_every_t == 0:
                    with open(avg_file,'wb') as out_file:
                        pickle.dump(avg, out_file, pickle.HIGHEST_PROTOCOL)
                    path = work_dir+project+"/nets_tf/AE_tf_{}".format(10000+t)
                    print("Saving net ({})...".format(path),end='',flush=True)
                    save_path = saver.save(session, path)
                    print("[x]")

                #If the average test-loss of the last n/2 time steps is NOT lower than the average over the last n timesteps, we think loss is not decreasing!
                if len(loss_history) == loss_history.maxlen and (sum(loss_history) < 2*sum(list(loss_history)[:loss_history.maxlen//2])):
                    input("THIS IS A HYPOTHETICAL STOP SIGNAL DUE TO NON-DECREASING TEST-LOSS. [Ctrl-C] to stop, [Enter] to ignore.")
                    # raise DoneSignal("Training done: test-loss not decreasing after {} epochs.", data=loss_history)

            raise DoneSignal("Training done: epoch-limit {} reached.".format(settings['n_epochs']))
        except (KeyboardInterrupt, DoneSignal) as e:
            print("---------------------------------------")
            if type(e) is DoneSignal:
                print(e.message)
            else:
                print("AutoEncoder training cancelled by user!")
            print("---------------------------------------")
            input("Do you want to export these wights to use for initialization? ([enter] to continue, [ctrl-C] to abort)")
            save_weights_to_file(work_dir+project+"init/weights")
            if settings['avg_subtraction']:
                avg_out_file = work_dir+project+"init/avg_img"
                print("Saving average-file to: {}".format(avg_out_file))
                with open(avg_out_file, 'wb') as out_file:
                    pickle.dump(avg, out_file, pickle.HIGHEST_PROTOCOL)

#
#
#
#
#

    ''' Save weights to a file '''
    if testing:
        print("Layers loaded:")
        for idx, net in enumerate(files):
            print("net: {}".format(net))
            saver.restore(session,net)
            save_weights_to_file(weight_file+"{}.pkl".format(idx))

    ''' Compare inputs and outputs '''
    if testing:
        print("Comparing input w. output...")
        for net in files:
            #Load net!
            color_dict = {}
            print(net)
            saver.restore(session,net)
            #Get som data!
            test_data, avg, n_train, train_file_idx = get_data_from_files( test_data_files, test_file_idx, 1000 )
            #For each sample: run through autoencoder, and visualize the result!
            for i in range(n_train):
                feed_dict={
                            input_tf : test_data[i,:,:,:][np.newaxis,:],
                            avg_tf : avg,
                            train_mode_tf : False,
                           }
                ret = session.run([output_tf, position_t_tf, error_loss_tf, smooth_loss_tf, presence_loss_tf]+list(snoop_tf), feed_dict=feed_dict)
                output, positions, error_loss, smooth_loss, presence_loss = ret[:5]
                snoops = ret[5:]
                org = test_data[i,1,:,:,:]
                out = output[0,:,:,:].repeat(settings['down_factor'],axis=0).repeat(settings['down_factor'],axis=1)
                if settings['gray']:
                    out = out.repeat(3,axis=2)
                clone = np.zeros(org.shape)
                clone[:out.shape[0], :out.shape[1], :out.shape[2]] = out
                snoop_layers = snoops[1][0]
                positions = positions[0]

                #Below is tons of poorly structured code that gives you the figure with stuff in it.
                scale = 5.0
                org = scipy.misc.imresize(org, scale)
                clone = scipy.misc.imresize(clone, scale)
                c_shape = snoop_layers.shape
                snoop_destack_tuple = () #(org[:c_shape[0],:c_shape[1],:],clone[:c_shape[0],:c_shape[1],:])
                if snoop_layers.shape[-1]%3 != 0:
                    snoop_layers = np.concatenate((snoop_layers, np.zeros( (c_shape[0], c_shape[1],3-snoop_layers.shape[-1]%3)) ), axis=-1)
                limit = int(snoop_layers.shape[-1]/3)
                snoop_destack_tuple += tuple([ snoop_layers[:,:,3*i:3*(i+1)] for i in range(limit)])
                n = len(snoop_destack_tuple)
                h = min(4,int(np.sqrt(n)))
                if n%h != 0:
                    snoop_destack_tuple += (np.zeros( (c_shape[0], c_shape[1],3) ), )*(h-n%h)
                    n+=h-n%h
                img = np.concatenate(snoop_destack_tuple[0:int(n/h)],axis=1)
                idx = int(n/h)
                for i in range(1,h):
                    x = np.concatenate( snoop_destack_tuple[idx:min(idx+int(n/h),len(snoop_destack_tuple))],axis=1)
                    img = np.concatenate(  (img,x) , axis=0 )
                    idx += int(n/4)

                # code.interact(local=locals())
                fig,(ax1,ax2,ax3) = plt.subplots(3,1, figsize=(1,3))
                fig.figsize = (4.0,1.0)
                ax1.set_aspect('equal')
                ax1.imshow(img)
                ax2.imshow(np.concatenate((org,clone), axis=1))
                activation_img_tuple = []
                for n, snoop in enumerate(snoops):
                    s = snoop[0,4:-4,4:-4,:] if n != 3 else sf(snoop[0,4:-4,4:-4,:])[0]
                    M = np.zeros((98,98,3))
                    _x, _y, z = s.shape
                    x = (98-_x) / 2
                    y = (98-_y) / 2
                    random_layer_choice = np.random.permutation(np.arange(z))[:3] if z > 3 else np.arange(z)
                    if z == 16:
                        random_layer_choice = np.array([5,11,1])
                    s = s[:,:, random_layer_choice]
                    s -= np.amin(s)
                    s /= np.amax(s)
                    M[math.ceil(x):-math.floor(x), math.ceil(y):-math.floor(y),:] = s
                    scipy.misc.imsave('snoop_img/{}.png'.format(n), s)
                    activation_img_tuple.append(M)
                activation_img = np.concatenate(activation_img_tuple, axis=1)
                ax3.imshow(activation_img)
                # ''' Visualize the features detected! '''
                if settings['encoder_transform'] in ['softargmax', 'dense_spatial']:
                    for i, (x,y,r) in enumerate(zip(positions[0,:],positions[1,:],positions[2,:]) ):
                        feature_range = np.ptp(snoops[1][0,:,:,i])
                        print("Feature{}: x={}, y={}, p={}, m={}".format(i,x,y,r,feature_range))
                        X = 0.5*(1+y)*scale
                        Y = 0.5*(1+x)*scale
                        radius = [1.5, 7]
                        transparency = 1 #min(1,max(0,r))
                        if i not in color_dict:
                            color_dict[i] = np.random.rand(3)
                            #color_dict[i] = np.array([1,0,0])
                        #if i in [11, 5]:
                        #    #color_dict[i] = np.random.rand(3)
                        #    color_dict[i] = np.array([1,0,0]) if i==5 else np.array([1,1,0])
                        #    c = Circle((size[0]*X,size[1]*Y), radius=radius[1]*scale, fill=False, linewidth=3.0 )
                        #else:
                        #    continue
                        c = Circle((size[0]*X,size[1]*Y), radius=radius[0]*scale, fill=True )
                        c.set_alpha(transparency)
                        c.set_antialiased(True)
                        c.set_ec(color_dict[i])
                        c.set_fc(color_dict[i])
                        ax2.add_patch(c)
                print("Loss (error,smooth,presence): {} + {} + {}".format(error_loss, smooth_loss, presence_loss))
                plt.show()
