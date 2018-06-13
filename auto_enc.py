import pickle
import numpy as np
import sys
import h5py
import os
import matplotlib.pyplot as plt
import cv2
import keras

from autoencoder_model import make_autoencoder

MAKE_GRAYSCALE = False
work_dir = "/knut/"
project = "dev_env"
save_every_t = 1
weight_file = "weights" #for outputing weights of the net in a file....
lr = 0.0005
n = 1000 #numbre of data vectors per file
n_epochs = 10000


def conv_weights_from_file(size,file):
    model = make_autoencoder(size)
    model.load_weights(file)
    weights = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.convolutional.Conv2D):
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

    n = raw_data.shape[0]

    if resize is None:
        return raw_data.astype(np.float32)/255.0, raw_data.shape[0]
    else:
        print("Resize not implemented yet...")
        data = np.empty((n,)+size)
        for i in range(n):
            small = cv2.resize(raw_data[i,:,:,:3], None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA) / 255.0
            if MAKE_GRAYSCALE:
                small = small[:,:,np.newaxis]
            data[i,:,:,:] = small
    return data, n

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
model = make_autoencoder(size=size,lr=lr)

if training :
    T=-1
    for t in range(n_epochs):
        for infile in os.listdir(work_dir+project+"/data"):
            T+=1
            with open(work_dir+project+"/data/"+infile,'rb') as file:
                data, _ = load_file(file)
            history = model.fit(data,data, batch_size=32)
            print("t={} -> {} samples seen...".format(T,(T+1)*1000))
            if t%save_every_t == 0:
                print("Saving net...",end='',flush=True)
                model.save_weights(work_dir+project+"/nets/AE_net_{}".format(n_epochs+T))
                print("[x]")

if testing:
    idx = 0
    print("Layers loaded:")
    for net in files:
        model.load_weights(net)
        for layer in model.layers:
            n = 0
            print("---")
            print(layer)
            weights = layer.get_weights()
            print( [w.shape for w in weights] )
            for w in weights:
                ape = 1
                for d in w.shape:
                    ape *= d
                n += ape
            print("n={}".format(n))

        print("Saving weights to {}".format(weight_file+"{}.pkl".format(idx)))
        with open(weight_file+"{}.pkl".format(idx),'wb') as out_file:
            pickle.dump(conv_weights_from_file(size,net), out_file, pickle.HIGHEST_PROTOCOL)
        idx += 1

''' Get some stats '''
if testing and False:

    for net in files:
        model.load_weights(net)
        result = np.empty((18000))
        for i, infile in enumerate(os.listdir(work_dir)):
            with open(work_dir+"/"+infile,'rb') as file:
                data, _ = load_file(file)
            predictions = model.predict(data)
            true_answer = data
            result[i*1000:(i+1)*1000] = np.mean( (predictions-true_answer)**2, axis=(1,2,3) )
        print("[{}] ::  Mean(MSE): {} \t Var(MSE): {}".format(net, np.mean(result),np.var(result) ))

''' Compare inputs and outputs '''
if testing:
    print("Comparing input w. output...")
    for net in files:
        model.load_weights(net)
        for infile in os.listdir(work_dir+project+"/data"):
            with open(work_dir+project+"/data/"+infile,'rb') as file:
                data, n = load_file(file)
            for i in range(n):
                org = data[i]
                clone = model.predict(org[np.newaxis,:])[0]
                img = np.concatenate((org,clone),axis=1)
                plt.imshow(img)
                plt.show()
