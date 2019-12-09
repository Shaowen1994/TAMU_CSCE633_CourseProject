from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from numpy.random import randn
 
import pathlib
import random
import matplotlib.pyplot as plt
 
import tensorflow as tf
import numpy as np
 
from matplotlib.image import imread
from keras.preprocessing import image

tf.enable_eager_execution()
 
AUTOTUNE = tf.data.experimental.AUTOTUNE
all_images = list()
all_labels = list()
label_names = {"1":"Surprise", "2":"Fear", "3":"Disgust", "4":"Happiness", "5":"Sadness", "6":"Anger", "7":"Neutral"}
with open(r"/content/drive/My Drive/EmoLabel/list_patition_label.txt",'r') as f:
  temp = list()
  while True:
    line = f.readline();
    if line:
      temp.append(line)
    else:
      break
random.shuffle(temp)
for item in temp:
  group1 = item.split()
  group2 = group1[0].split(".")
  new_path = "/content/drive/My Drive/my_image/aligned/"+ group2[0]+"_aligned"+".jpg"
  # new_path = "/content/drive/My Drive/my_image/original/"+ group2[0]+".jpg"
  all_images.append(new_path)
  all_labels.append(int(group1[1])-1)
train_test_split = int(len(all_images)*0.2)
x_train = all_images[train_test_split:]
x_test = all_images[:train_test_split]
print(all_labels)
y_train = all_labels[train_test_split:]
y_test = all_labels[:train_test_split]

data_size = len(all_images)
IMG_SIZE=160
 
BATCH_SIZE = 16
 
def _parse_data(x,y):
  image = tf.read_file(x)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image,y
 
def _input_fn(x,y):
  ds=tf.data.Dataset.from_tensor_slices((x,y))
  ds=ds.map(_parse_data)
  ds=ds.shuffle(buffer_size=data_size)
  
  
  ds = ds.repeat()
  
  ds = ds.batch(BATCH_SIZE)
  
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  
  return ds
train_ds=_input_fn(x_train,y_train)
validation_ds=_input_fn(x_test,y_test)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(label_names),activation='softmax')

model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

history = model.fit(train_ds,epochs=100, steps_per_epoch=2,validation_steps=2,validation_data=validation_ds)
validation_steps = 20
loss0,accuracy0 = model.evaluate(validation_ds, steps = validation_steps)
 
print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()