# [CSCE_633 Machine Learning](http://people.tamu.edu/~atlaswang/19CSCE633.html) Course Project

## Abstract

While image recognition and generation have been developed for several years and gotten great breakthrough in many areas, facial expression related models have been less touched due to the data complexity and limited benchmark datasets. Based on a novel database RAF-DB, we applied a popular discriminative model, the VGG-net to recognize the expressions so as to see the discrimination of different expressions. We then learned a low dimensional representation for each facial expression and developed a conditional architecture for generative adversarial networks. Our generative model can learn the conditional distribution of the expressions by taking the representations as the input and successfully generate fake images for each condition.

***

## Pre-requisite 

* **NumPy**
* **SciPy**
* **Matplotlib**
* **Keras**
* **Tensorflow**

***

## Data

For this project we applied the RAF-DB. To download the dataset, please follow the link http://whdeng.cn/RAF/model1.html and contact the group of reference 1. After downloading the dataset, move the folder "basic/" inside the folder "Data/" in our repository to run our code.

***

## Discriminative Model
We choose VGG16 as our discriminative model. To instanstiate the VGG16 architecture.
The input data is in folder `/images/aligned` and resize them to 160\*160 as input data

To provide a pre-trained VGG16 model, `VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')`

Parameters:<br> 
1. `include_top`: whether to include the 3 fully-connected layers at the top of the network.<br>
2. `weights`: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet)<br>
3. `pooling`: Optional pooling mode for feature extraction<br>

Compile the model and train it 
```python
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
history = model.fit(train_ds,
                    epochs=100, 
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=validation_ds)
```
Parameters:<br> 
1. `optimizer`: String (name of optimizer) or optimizer instance<br>
2. `loss`: String (name of objective function) or objective function or Loss instance<br>
3. `metrics`:  List of metrics to be evaluated by the model during training and testing<br>         
***

## Representation learning

Go to the "Data/" folder. Open and follow the "Data_PreProcess.ipynb" file for data process. Open and follow the "Cov_AutoEncoder.ipynb" file to train the convolutional Autoencoder and applied PCA to get the expression represenations.

***

## Generative Model 

Go to the "Generative_model" foler. To train the cWGAN, run 

```
python conditional_WGAN.py
```

After training process (have already recorded some checkpoints), to generate samples for the certain expression, run

```
python Expression_generator.py  <checkpoint path>  <expression>  <sample amount> <sample path>
```
* **checkpoint path**: the relative path of the check point files on which you want to generate samples.
* **expression**: "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger" or "Neutral".
* **sample amonut**: the number of samples you want generate.
* **sample path**: the path of the folder where you want to record the samples.

***

## Generated Examples

Examples of generated samples:

![samples](/g_samples.png)

## Main References
\[1\] [Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
Recognition in the Wild](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf)

\[2\] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

\[3\] [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

\[4\] [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

\[5\] [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

\[6\] [De Novo Protein Design for Novel Folds using Guided Conditional Wasserstein Generative Adversarial Networks (gcWGAN)](https://www.biorxiv.org/content/biorxiv/early/2019/09/14/769919.full.pdf)
