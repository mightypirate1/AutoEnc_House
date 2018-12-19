'''
USAGE: set up a dict with your architecture settings, and pass it to make_autoencoder.
NOTE: the ones you do not specify are left as default!

available options are:  { key : [alternative1, alternative2...] (default) comment }

ARCHITECTURE:
data_size           : [(int, int, int)]                         Size of the input data (W,H,C)
encoder_transform   : ['softargmax', 'dense', 'dense_spatial'] ('softargmax')    applied after encoding convs
encoder_transf_size : [int]                   (None)            this is used to control the size of the encoded state. set to None it uses default values
decoder_transform   : ['distmap', 'dense', 'none'] ('distmap')  first transformation of the decoder
decoder             : ['conv', 'dense']       ('conv')          main decoder-stage (if you want to use 'conv' here, you need decoder_tranform=distmap)
bn                  : [bool]                  (False)           batch normalization?
encoder_conv_depth  : [(int,int,int)]         (64,32,16)        number of channels in the encoding convolutions
encoder_conv_size   : [((int,int),(int,int),(int,int))] ((7,7), (5,5), (5,5)) conv sizes
decoder_conv_depth  : [(int,int,int)]         (64,32,16)        number of channels in the decoding convolutions
decoder_conv_size   : [((int,int),(int,int),(int,int))] ((7,7), (5,5), (5,5)) conv sizes
output_bias         : [Bool]                  (False)           If enabled, before the actual output, an out-put sized trainable weight is added.

HYPERPARAMS:
lr                  : [float]                 (0.00001)         learning rate
init_alpha          : [float]                 (1.0)             The initial value of the temperature parameter alpha, used in the softmax
minibatch_size      : [int]                   (32)              minibatch size!
down_factor         : [int(any_power_of_2)]   (1)               how much the image is downsampled during reconstruction (set this to 2 and 96x96 -> 48x48)
grey                : [bool]                  (False)           do we make the image grey before we put it through the AE?
n_epochs            : [int]                   (4100)            n_epochs (1 epoch is ~3000 samples)
avg_subtraction     : [bool]                  (False)           do average subtraction instead as a type of proto-bn in the first layer only.

MISC:
weighted_loss       : [bool]                  (False)           weight data extra the more they differ from average?
project_folder      : [string]                ('default_project')  give your project a folder and name it!)

DESCRIPTION:
description         : [string]                ("")              verbal description of the configuration
'''

default_settings =  {
                      'data_size' : (96,96,3),
                      'encoder_transform' : 'softargmax',
                      'encoder_conv_depth' : (64,32,16),
                      'encoder_conv_size' : ((7,7), (5,5), (5,5)),
                      'decoder_conv_depth' : (64,32,16),
                      'decoder_conv_size' : ((7,7), (5,5), (5,5)),
                      'encoder_transf_size' : None,
                      'decoder_transform' : 'distmap',
                      'decoder' : 'conv',
                      'decoder_use_bias' : True,
                      'output_bias' : False,
                      'down_factor' : 1,
                      'gray' : False,
                      'lr' : 0.00001,
                      'init_alpha' : 1.0,
                      'loss_weights' : (1.0, 0.0, 0.0),
                      'bn' : False,
                      'minibatch_size' : 32,
                      'weighted_loss' : False,
                      'n_epochs' : 1500,
                      'avg_subtraction' : False,
                      'project_folder' : 'default_project',
                      'description' : "This configuration has no description. It is good practice to specify one in settings.py"
                    }

#########
#########
#########
#########
#########

''' Original architecture! '''
spatial_ae_conv =   {
                        'name' : "SAEV",
                        'encoder_transform' : 'softargmax',
                        'decoder_transform' : 'distmap',
                        'decoder' : 'conv',
                        'bn' : True,
                        'output_bias' : True,
                        'loss_weights' : (1.0, 0.001, 0.1),
                        'project_folder' : "AE_comp_1/",
                        'description' : "This architecture is: x->[3xConv]->softargmax->encoded->distmap->[3xConv]->y  with bn after each conv!"
                    }

''' Dr. Finn's spatial AE (approx) '''
spatial_ae_dense =  {
                        'name' : "Finn",
                        'encoder_transform' : 'softargmax',
                        'encoder_transf_size' : None,
                        'decoder_transform' : 'none',
                        'decoder' : 'dense',
                        'init_alpha' : 7.0,
                        'bn' : True,
                        'output_bias' : True,
                        'loss_weights' : (1.0, 0.01, 0.1),
                        'project_folder' : "AE_comp_2/",
                        'description' : "This architecture is: x->[3xConv]->softargmax->encoded->flatten->[DENSE]->y  with bn after each layer!"
                    }

''' Like the "original", but the softargmax is replaced with a trained mapping [feature-map -> (x,y, rho)] '''
spatial_ae_conv_learned_encoder =   {
                                        'name' : "experimental",
                                        'encoder_transform' : 'dense_spatial',
                                        'decoder_transform' : 'distmap',
                                        'decoder' : 'conv',
                                        'bn' : True,
                                        'output_bias' : True,
                                        'loss_weights' : (1.0, 0.001, 0.2),
                                        'lr' : 0.00001,
                                        'weighted_loss' : True,
                                        'project_folder' : "AE_comp_3/",
                                        'description' : "This architecture is: x->[3xConv]->[TRAINED encoder-fcn]->encoded->distmap->[3xConv]->y  with bn after each layer!"
                                    }

''' This is a standard conv-ae design. '''
regular_conv_ae_big_bottleneck  =   {
                                        'name' : "ConvAE",
                                        'encoder_transform' : 'dense',
                                        'encoder_transf_size' : 512,
                                        'decoder_transform' : 'dense',
                                        'decoder' : 'conv',
                                        'bn' : True,
                                        'loss_weights' : (1.0, 0.001, 0.0),
                                        'project_folder' : "AE_comp_4/",
                                        'description' : "This is a regular convolutional AE. architecture is: x->[3xConv]->[flatten+dense(512)]->encoded->[dense+reshape]->[3xConv]->y  with bn after each layer!"
                                    }

''' This is a standard conv-ae design. '''
regular_conv_ae_small_bottleneck  =     {
                                            'name' : "ConvAE (small)",
                                            'encoder_transform' : 'dense',
                                            'encoder_transf_size' : 64,
                                            'decoder_transform' : 'dense',
                                            'decoder' : 'conv',
                                            'bn' : True,
                                            'loss_weights' : (1.0, 0.001, 0.0),
                                            'project_folder' : "AE_comp_5/",
                                            'description' : "This is a regular convolutional AE. architecture is: x->[3xConv]->[flatten+dense(32)]->encoded->[dense+reshape]->[3xConv]->y  with bn after each layer!"
                                        }


#To make your conf accessible through lazy referencing, add it here!
quick_list = [x for x in dir() if "_ae_" in x]

###############
# Dont touch! #
###############

__dir__ = dir()
def parse_conf(setting):
    for x in default_settings:
        if x not in setting:
            setting[x] = default_settings[x]
    print("---------------------------------------")
    print("AutoEncoder settings:")
    print("---------------------------------------")
    for x in sorted(setting.items(), key=lambda x:x[0]):
        if x[0] is not 'description':
            print("{}".format(x[0]).rjust(25,' '), end=' : ')
            print(x[1])
    print("---------------------------------------")
    print("Description:")
    print(setting['description'])
    print("---------------------------------------")
    return setting

def get_conf(arg):
    if arg in __dir__:
        return eval(arg)
    elif arg in [str(x) for x in range(len(quick_list))]:
        return eval(quick_list[int(arg)])
    else:
        print("---------------------")
        print("Valid configurations:")
        for d in __dir__:
            if not ("__" in d or d in ['get_conf', 'parse_conf', 'quick_list']):
                print(d)
        print("---------------------")
        print(quick_list)
        raise ValueError("Invalid settings request!")
