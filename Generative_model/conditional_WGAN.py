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

#KIND = sys.argv[1] # Two methods of combining the condition and the noise
                   # 'add' or 'concat'

#if KIND != 'add' and KIND != 'concat':
#    print 'Error! No kind named %s'%KIND
#    quit()

KIND = 'concat'

DATA_DIR = '' # Path of the real data

DIM = 64 # Channel dimension 
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 20000 # How many generator iterations to train for
OUTPUT_DIM = 30000 # Number of pixels (3*100*100)
IMAGE_DIM = 100
label_len = 20 

check_path = 'Check_points/'
if not os.path.exists(check_path):
    os.mkdir(check_path)
sample_path = 'Generated_Samples/'
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
g_im_path = 'Generated_Images/'
if not os.path.exists(g_im_path):
    os.mkdir(g_im_path)

#IS_file = open('Inception_scores','w')
#IS_file.close()
glabel_file = open('Labels_for_generation','w')
glabel_file.close()
g_loss_file = open('Generator_loss','w')
g_loss_file.close()
c_loss_file = open('Critic_loss','w')
c_loss_file.close()

################################## Load Data ########################################

label_name_dict = {
                1: 'Surprise',
                2: 'Fear',
                3: 'Disgust',
                4: 'Happiness',
                5: 'Sadness',
                6: 'Anger',
                7: 'Neutral'
              }

with open('../Data/im_dict.pickle','rb') as im_d:
    im_dict = pickle.load(im_d)

im_list = []
label_list = []

for label in im_dict['train']:
    array_path = '../Data/Data_Arrays/train/' + label_name_dict[label] + '/'
    for im_name in im_dict['train'][label]:
        im_name = array_path + im_name.strip('.jpg') + '_aligned.npy'
        im_array = np.load(im_name)
        im_list.append(im_array)
        label_list.append(label)

image_arrays = np.array(im_list)
label_arrays = np.array(label_list)

#image_arrays = np.ones((128,100,100,3))
#label_arrays = np.ones(128)

with open('../Data/label_representation.pickle','rb') as la_d:
    label_dict = pickle.load(la_d)

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

    #output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*DIM, 2*DIM, 5, output)
    #output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    #output = tf.nn.relu(output)

    #output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*DIM, DIM, 5, output)
    #output = lib.ops.batchnorm.Batchnorm('Generator.BN4', [0,2,3], output)
    #output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs,condition): # Modified
    output = tf.reshape(inputs, [-1, 3, IMAGE_DIM, IMAGE_DIM]) # Modified

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=3) 
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=3)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    #output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = tf.reshape(output, [BATCH_SIZE, -1])

    #output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    size = 100 # Modified
    output = lib.ops.linear.Linear('Discriminator.reduction',4608, size, output) # Modified
    output = tf.concat([output,condition],axis=1) # Modified
    output = tf.contrib.layers.fully_connected(output,300,scope='Discriminator.fully',reuse=tf.AUTO_REUSE) # Modified
    output = lib.ops.linear.Linear('Discriminator.output',300 , 1, output) # Modified

    #return tf.reshape(output, [-1])
    return output # Modified

################################# Loss Functions ####################################

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
real_inputs_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, label_len]) # Modified
fake_data = Generator(BATCH_SIZE,real_inputs_label) # Modified

disc_real = Discriminator(real_data, real_inputs_label) # Modified
disc_fake = Discriminator(fake_data, real_inputs_label) # Modified

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# Standard WGAN loss
gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

# Gradient penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates, real_inputs_label), [interpolates])[0] # Modified
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

# Optimizer
gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

################### Training and evaluation functions ###############################

# For generating samples

def generate_image(epoch,label): # Modified
    samples = session.run(fake_data,feed_dict={real_inputs_label:label}) # Modified
    samples_r = ((samples+1.)*(255./2)).astype('int32') # Modified
    lib.save_images.save_images(samples_r.reshape((BATCH_SIZE, 3, IMAGE_DIM, IMAGE_DIM)),g_im_path + 'samples_{}.jpg'.format(epoch))
    #return samples.reshape((BATCH_SIZE, 3, IMAGE_DIM, IMAGE_DIM)) # Modified
    return samples,samples_r.reshape((BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3)) # Modified    

# For calculating inception score

#def get_inception_score(samples_100): # Modified
#    all_samples = []
#    for i in xrange(10):
#        #all_samples.append(session.run(samples_100))
#        all_samples.append(samples_100)
#    all_samples = np.concatenate(all_samples, axis=0)
#    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
#    all_samples = all_samples.reshape((-1, 3, IMAGE_DIM, IMAGE_DIM)).transpose(0,2,3,1) # Modified
#    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators

def inf_train_gen(images,labels): # Modified
    epoch = 0
    while True:
        indices = np.arange(len(images),dtype=np.int) 
        np.random.shuffle(indices) 
        shuffle_im = [images[i] for i in indices] 
        shuffle_label =  [labels[i] for i in indices]
        length = len(images) - BATCH_SIZE + 1
        for i in xrange(0, length, BATCH_SIZE):
            if i + BATCH_SIZE >= length: 
                epoch += 1
            yield np.array(    
                [im.reshape(OUTPUT_DIM) for im in shuffle_im[i:i+BATCH_SIZE]], 
                dtype='int32'
            ),np.array(
                [label_dict[l] for l in shuffle_label[i:i+BATCH_SIZE]], 
                dtype='float32'
            ),epoch 

################################# Training Process ##################################

saver = tf.train.Saver(max_to_keep=None)

# Train loop

with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    gen = inf_train_gen(image_arrays,label_arrays)
    epoch_before = 0

    for iteration in xrange(ITERS):
         
        print 'Iteration: %d'%iteration

        # Train generator
        
        if iteration > 0:
            im_batch,l_batch,epoch = gen.next() # Modified
            gen_loss,_ = session.run([gen_cost,gen_train_op],feed_dict={real_inputs_label:l_batch})
            
            with open('Generator_loss','a') as g_loss_file:
                g_loss_file.write(str(gen_loss) + '\n')

        # Train critic
                    
        for i in xrange(CRITIC_ITERS):
            im_batch,l_batch,epoch = gen.next() # Modified
            disc_loss, _ = session.run([disc_cost, disc_train_op], 
                                       feed_dict={real_data_int:im_batch,real_inputs_label:l_batch})
            with open('Critic_loss','a') as c_loss_file:
                c_loss_file.write(str(disc_loss) + '\n')

        # Calculate inception score every 100 iters
        if epoch != epoch_before:
            print 'Epoch:',epoch
            epoch_before = epoch
            
            saver.save(session,check_path + 'model_{}'.format(epoch) + '_{}.ckpt'.format(iteration))

            im_labels = np.array([label_dict[3]]*BATCH_SIZE,dtype=np.float32)
            with open('Labels_for_generation','a') as glabel_file:
                glabel_file.write('Epoch ' + str(epoch) + ': ' + str(im_labels) + '\n')

            samples,samples_r = generate_image(epoch,im_labels)
            np.save(sample_path + 'samples_{}'.format(epoch),samples_r)
                 
 
            #inception_score = get_inception_score(samples) # Modified

        if epoch >= 300:
            quit() 

