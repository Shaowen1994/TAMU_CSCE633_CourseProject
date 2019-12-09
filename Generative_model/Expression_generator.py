################################# Import Packages ####################################

import os,sys
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
#import tflib.inception_score
import tflib.plot

import pickle

####################### Set Paths and Global Parameters ##############################

Check_point = sys.argv[1]
Expression = sys.argv[2]
Amount = sys.argv[3]
Result_path = sys.argv[4]

while Check_point.endswith('.'):
    Check_point = Check_point[:-1]

Epoch = Check_point.split('/')[-1].split('_')[1]

if not Result_path.endswith('/'):
    Result_path += '/'

if not Expression in ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']:
    print 'Error! No expression named "%s"!'%Expression

KIND = 'concat'

DATA_DIR = '' # Path of the real data

DIM = 64 # Channel dimension 
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
ITERS = 20000 # How many generator iterations to train for
OUTPUT_DIM = 30000 # Number of pixels (3*100*100)
IMAGE_DIM = 100
label_len = 20 

if not os.path.exists(Result_path):
    os.mkdir(Result_path)

################################## Load Data ########################################

expr_index_dict = {
                'Surprise':1,
                'Fear':2,
                'Disgust':3,
                'Happiness':4,
                'Sadness':5,
                'Anger':6,
                'Neutral':7
              }

im_list = []
label_list = []

with open('../Data/label_representation.pickle','rb') as la_d:
    label_dict = pickle.load(la_d)

Repre = label_dict[expr_index_dict[Expression]]
f_batch = [Repre] * int(Amount)

print 'Data load successfully.'

########################## Structure of the model ###################################

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, condition, kind='concat', noise=None): # Modified
    #noise_len = 20 # Modified

    if noise is None:
        if kind == 'add': # Modified
            noise = tf.random_normal([n_samples, 20]) # Modified
            noise_len = 20
        elif kind == 'concat': # Modified
            noise = tf.random_normal([n_samples, 128]) # Modified
            noise_len = 128

    if kind == 'add':  # Modified
        output = noise + condition  # Modified
        in_len = noise_len # Modified
    elif kind == 'concat': # Modified
        output = tf.concat([noise,condition],axis=1) # Modified
        in_len = noise_len + 20 # Modified

    output = lib.ops.linear.Linear('Generator.Input', in_len, 2*25*25*DIM, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 2*DIM, 25, 25])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

################################# Loss Functions ####################################

real_inputs_label = tf.placeholder(tf.float32, shape=[int(Amount), label_len]) # Modified
fake_data = Generator(int(Amount),real_inputs_label) # Modified

################### Training and evaluation functions ###############################

# For generating samples

def generate_image(label): # Modified
    samples = session.run(fake_data,feed_dict={real_inputs_label:label}) # Modified
    samples_r = ((samples+1.)*(255./2)).astype('int32') # Modified
    return samples_r.reshape((int(Amount), IMAGE_DIM, IMAGE_DIM, 3)) # Modified    

################################# Training Process ##################################

saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as session:

    saver.restore(session,Check_point)

    samples = generate_image(f_batch)

    np.save(Result_path + Expression + '_' + Epoch + '_' + Amount,samples)
