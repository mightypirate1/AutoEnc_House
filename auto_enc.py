import pickle
import numpy as np
import sys
import h5py
import os
import matplotlib.pyplot as plt
import cv2
import keras

from lsuv_init import LSUVinit
from autoencoder_model import make_autoencoder

MAKE_GRAYSCALE = False
work_dir = "knut/"
project = "pepperBig_trial2/"#"dev_env"
save_every_t = 100
display_result = not False
visualize_convs = True
weight_file = "weights" #for outputing weights of the net in a file....
lr = 0.00005
n = 1000 #numbre of data vectors per file
n_epochs = 4100
batch_normalization = not True
disable_avg = False
lsuv_init = not True
first_batch = True

def conv_weights_from_file(size,file):
    model,_,_ = make_autoencoder(size)
    model.load_weights(file)
    weights = []
    for layer in model.layers:
        print(layer, len(layer.get_weights()))
        if isinstance(layer, keras.layers.Convolution2D):
            w = layer.get_weights()
            weights.append(w)
    return weights


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
    if batch_normalization or disable_avg:
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

model, snoop, position = make_autoencoder(size=size,lr=lr,bn=batch_normalization)

if training:
    tb = keras.callbacks.TensorBoard(log_dir='./tensorboard/'+project, histogram_freq=0,
              write_graph=True, write_images=True)
    T=-1
    for t in range(n_epochs):
        for infile in os.listdir(work_dir+project+"/data"):
            T+=1
            with open(work_dir+project+"/data/"+infile,'rb') as file:
                data, _, avg, avg_block = load_file(file)
            if lsuv_init and first_batch:
                first_batch = False
                model = LSUVinit(model, data[:100,:,:,:])

            model.fit(data-avg_block,data-avg_block, batch_size=32, callbacks=[tb])
            print("t={} -> {} samples seen...".format(T,(T+1)*1000))
            if T%save_every_t == 0:
                print("Saving net...",end='',flush=True)
                model.save_weights(work_dir+project+"/nets/AE_net_{}".format(10000+T))
                with open(work_dir+project+"/nets/avg_img_{}".format(10000+T),'wb') as f:
                    pickle.dump(avg, f, pickle.HIGHEST_PROTOCOL)
                print("[x]")

if testing:
    idx = 0
    print("Layers loaded:")
    for net in files:
        print("net: {}".format(net))
        print("saving...")
        model.load_weights(net)
        layer_names = [ l.name for l in model.layers ]
        weights = [ l.get_weights() for l in model.layers ]
        weight_dict={}
        for i,x in enumerate(layer_names):
            if "conv" in x:
                weight_dict[x] = { 'layer' : x, 'weights' : [] }
        for i,x in enumerate(layer_names):
            if "conv" in x:
                weight_dict[x]['weights'].append( weights[i][0] )
        for x in weight_dict:
            print(x)
            for w in weight_dict[x]['weights']:
                print(w.shape)

        file_output = list(weight_dict.values())
        print("Saving weights to {}".format(weight_file+"{}.pkl".format(idx)))
        with open(weight_file+".pkl",'wb') as out_file:
            pickle.dump(file_output, out_file, pickle.HIGHEST_PROTOCOL)
        exit()

        print("Saving weights to {}".format(weight_file+"{}.pkl".format(idx)))
        with open(weight_file+"{}.pkl".format(idx),'wb') as out_file:
            pickle.dump(conv_weights_from_file(size,net), out_file, pickle.HIGHEST_PROTOCOL)
        idx += 1

''' Compare inputs and outputs '''
if testing:
    print("Comparing input w. output...")
    for net in files:
        model.load_weights(net)
        for infile in os.listdir(work_dir+project+"/data"):
            with open(work_dir+project+"/data/"+infile,'rb') as file:
                data, n, _, _ = load_file(file)
                avg_file = work_dir+project+"/nets/avg_img_"+net.split("_")[-1]
                with open(avg_file, 'rb') as f:
                    avg = pickle.load(f) if not batch_normalization else 0
            for i in range(n):
                org = data[i,:,:,:]
                clone = (model.predict(org[np.newaxis,:]-avg)+avg)[0]
                snoop_layers = snoop.predict(org[np.newaxis,:]-avg)[0]
                positions = position.predict(org[np.newaxis,:]-avg)[0]
                e = np.sqrt(np.sum(np.square(np.abs(org-clone)),axis=2)).reshape(-1)
                print("Mean error: {}    Max error: {}".format(e.mean(),e.max()))
                if display_result:
                    for j in range(positions.shape[1]):
                        print("Feature {} pos: ({},{})".format(j,positions[0,j], positions[1,j]))
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
                    plt.imshow(img)
                    plt.show()
