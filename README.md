# [CSCE_633 Machine Learning](http://people.tamu.edu/~atlaswang/19CSCE633.html) Course Project

## Abstract

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
* checkpoint path: the relative path of the check point files on which you want to generate samples.
* expression: "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger" or "Neutral".
* sample amonut: the number of samples you want generate.
* sample path: the path of the folder where you want to record the samples.

***

## Generated Examples

## Main References
\[1\] [Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
Recognition in the Wild](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf)

\[2\] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

\[3\] [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

\[4\] [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf)

\[5\] [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

\[6\] [De Novo Protein Design for Novel Folds using Guided Conditional Wasserstein Generative Adversarial Networks (gcWGAN)](https://www.biorxiv.org/content/biorxiv/early/2019/09/14/769919.full.pdf)
