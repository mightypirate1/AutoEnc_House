# AutoEnc_House
This project is designed to pre-train weights to be used in the PepperSocial-project.

## Installation:
Just pull the repository, install the dependencies and run (see "Usage").

#### Dependencies:

- [pip] (If you for some reason did not get pip with your python-installation)
- [Python3]
- [OpenCV]
- [tensorflow]

> If you have another preferred way of installing python packages, you do not need pip.

Additional python-packages:
- docopt
- scipy
- matplotlib
- numpy

```
pip install docopt scipy matplotlib numpy
```

> Or use your preferred way of installing python packages.

## Usage:

The expected use-case is that a data-set has been generated by the PepperSocial-project, and AutoEnc_House is used on that data. The following steps explain how this can be done.

#### Step 1 - Defining your project.

Create a symbolic link called "projects" to the folder where you store your projects:

```
cd /path/to/AutoEnc_House
ln -sf /path/to/where/projects/are/stored projects
```

The folder we link to is expected to contain projects. A project is a folder, which has sub-folders "data" and "testdata", both containing data generated by the PepperSocial-project's data-gatherer.

(If you want to use AutoEnc_House for other types of data, that works as well. Read "Custom data" below for how to do this)

#### Step 2 - Edit the settings.py-file.

The settings.py-file contains dictionaries with settings.

At the top of the file, are all the available options listed, and a brief explanation of what they do.
Below are the following pre-defined setting-dictionaries:
- spatial_ae_conv
- spatial_ae_dense
- spatial_ae_conv_learned_encoder
- regular_conv_ae_big_bottleneck
- regular_conv_ae_small_bottleneck

You can modify one of these, or define a new dictionary to suit your needs. The most important thing is that you set the 'project_folder' entry to match the project you want to use it for.

##### Example:
If
```
$ cd /path/to/AutoEnc_House
$ ls projects

foo

$ ls projects/foo

data
testdata
```

then the most simple setting for this would be:

```
my_autoencoder =  {
                    'project_folder' : 'foo',
                  }
```

You can of course add more options to change what type of autoencoder you build (take inspiration from the pre-defined ones to get started!).

All entries in the dictionary take the form 'option' : value.
If you for instance want to set the learning-rate to be 0.00005, you add it as follows:

```
my_autoencoder =  {
                    ...
                    'lr' : 0.00005,
                    ...
                  }
```

#### Step 3 - Train your autoencoder!

If everything was set up correctly, you can now train your autoencoder as follows:

```
$ python3 auto_enc.py --train --settings my_autoencoder
```
where 'my_autoencoder' can be replaced with any other dictionary defined in settings.py, including of course, the predefined ones listed in "Step 2".

It will train until test-loss no longer decreases (or until you press crtl-c), and then save the weights to a file 'weights' in the folder 'AutoEnc_House/projects/foo/init'.

#### Step 4 - (Optional and still under construction) Visualize the resulting model.
> EXPERIMENTAL!!!

```
$ python3 auto_enc.py --test --settings my_autoencoder path/to/AutoEnc_House/projects/foo/nets_tf/AE_tf_NNNNN
```

> It's important that the file-ending is omitted, as otherwise loading will fail.

This will load the trained model and visualize its input and the activations of the encoder convolutions. For spatial models it might also put circles at the positions of the detected features.

> NOTE: This feature is not stable, and for most configurations it will crash. You will have to dig into the code to make this work properly. (Contact me if you have questions)


#### Step 5 - Load your weights into the PepperSocial-project.

See the documentation for that project for details.


## Custom data:

To use this project for data not generated by the PepperSocial-project, follow these steps:

#### Create a project for the data
```
$ cd /path/to/AutoEnc_House/projects
$ mkdir -p my_project/data
$ mkdir -p my_project/testdata
```

#### Store data
Generate the data as you please and divide it into chunks. Store each chunk as a numpy-arrays of shape (N,W,H,C) (where W,H,C are the width, height and number of channels of the images, and N is the number of samples) in files using [pickle.dump].

Example:
```
...
data = sample_data() #returns an numpy array of shape (100000,128,96,3), containing 100000 images of size (128,96,3).
for i,idx in enumerate(range(0,100000,1000)):
  folder = 'data/' if i%5!=0 else 'testdata/'
  with open("my_project/"+folder+"chunk{:05}".format(idx), 'wb') as file:
    pickle.dump(data[idx:idx+1000,:,:,:], file, protocol=pickle.HIGHEST_PROTOCOL)
```
A simple setting for this data could be:
```
my_project_settings = {
                        'project_folder' : 'my_project',
                        'size' : (128,96,3), #Default size is (96,96,3), so we need to change it...
                        'description' : 'My custom project. Most settings are default, which makes this a type of spatial autoencoder.'
                      }
```





[Python3]: <https://www.python.org/downloads/>
[OpenCV]: <https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html>
[tensorflow]: <https://www.tensorflow.org/install/>
[pickle.dump]: <https://docs.python.org/2/library/pickle.html>
[pip]: <https://stackoverflow.com/a/6587528>
