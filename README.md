# Awesome Deep Phenomena [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
Our understanding of modern neural networks lags behind their practical successes. This growing gap poses a challenge to the pace of progress in machine learning because fewer pillars of knowledge are available to designers of models and algorithms. Inspired by the [ICML 2019 workshop Identifying and Understanding Deep Learning Phenomena](http://deep-phenomena.org/), I collect papers which present interesting empirical study or insight into the nature of deep learning.

## Table of Contents
- [Deep Learning Theory](#deep-learning-theory)
- [Empirical Study](#empirical-study)
- [Understanding Robustness](#understanding-robustness)
- [Representation Learning](#representation-learning)
- [Interpretability](#interpretability)

## Deep Learning Theory

### 2020
- Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime
  [[paper]](https://arxiv.org/abs/2003.01054)
  [[code]](https://github.com/mariaref/Random_Features)
  - Stéphane d'Ascoli, Maria Refinetti, Giulio Biroli, Florent Krzakala. *ICML 2020*
  - Digest:  In this work, we develop a quantitative theory for this phenomenon in the so-called lazy learning regime of neural networks, by considering the problem of learning a high-dimensional function with random features regression. We obtain a precise asymptotic expression for the bias-variance decomposition of the test error, and show that the bias displays a phase transition at the interpolation threshold, beyond which it remains constant. 


### 2019
- On Exact Computation with an Infinitely Wide Neural Net.
  [[paper]](https://arxiv.org/abs/1904.11955)
  [[code]](https://github.com/ruosongwang/cntk)
  - Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, Ruosong Wang. *NeurIPS 2019*
  - Digest: The current paper gives the first efficient exact algorithm for computing the extension of NTK to convolutional neural nets, which we call Convolutional NTK (CNTK), as well as an efficient GPU implementation of this algorithm.

### 2018
- Neural Tangent Kernel: Convergence and Generalization in Neural Networks.
  [[paper]](https://arxiv.org/abs/1806.07572)
  - Arthur Jacot, Franck Gabriel, Clément Hongler. *NeurIPS 2018*
  - Digest: We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK).

## Empirical Study

### 2020
- Bayesian Deep Learning and a Probabilistic Perspective of Generalization.
  [[paper]](https://arxiv.org/abs/2002.08791)
  [[code]](https://github.com/izmailovpavel/understandingbdl)
  - Andrew Gordon Wilson, Pavel Izmailov.
  - Digest: We show that deep ensembles provide an effective mechanism for approximate Bayesian marginalization, and propose a related approach that further improves the predictive distribution by marginalizing within basins of attraction, without significant overhead. We also show that Bayesian model averaging alleviates double descent, resulting in monotonic performance improvements with increased flexibility. 


### 2019
- Deep Double Descent: Where Bigger Models and More Data Hurt.
  [[paper]](https://arxiv.org/abs/1912.02292)  
  - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever. *ICLR 2020*
  - Digest: We show that a variety of modern deep learning tasks exhibit a "double-descent" phenomenon where, as we increase model size, performance first gets worse and then gets better.

- Are All Layers Created Equal?
  [[paper]](https://arxiv.org/abs/1902.01996)  
  - Chiyuan Zhang, Samy Bengio, Yoram Singer. *ICML 2019 Workshop*
  - Digest: We show that the layers can be categorized as either "ambient" or "critical". Resetting the ambient layers to their initial values has no negative consequence, and in many cases they barely change throughout training. On the contrary, resetting the critical layers completely destroys the predictor and the performance drops to chanceh. 

### 2017
- Critical Learning Periods in Deep Neural Networks.
  [[paper]](https://arxiv.org/abs/1711.08856)
  - Alessandro Achille, Matteo Rovere, Stefano Soatto. *ICLR 2019*
  - Digest: Our findings indicate that the early transient is critical in determining the final solution of the optimization associated with training an artificial neural network. In particular, the effects of sensory deficits during a critical period cannot be overcome, no matter how much additional training is performed.


## Understanding Robustness

### 2019
- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
  [[paper]](https://arxiv.org/abs/1903.12261)
  [[dataset]](https://github.com/hendrycks/robustness)
  - Dan Hendrycks, Thomas Dietterich. *ICLR 2019*
  - Digest: In this paper we establish rigorous benchmarks for image classifier robustness. Our first benchmark, ImageNet-C, standardizes and expands the corruption robustness topic, while showing which classifiers are preferable in safety-critical applications. Then we propose a new dataset called ImageNet-P which enables researchers to benchmark a classifier's robustness to common perturbations. 

## Representation Learning

### 2019
- A critical analysis of self-supervision, or what we can learn from a single image.
  [[paper]](https://arxiv.org/abs/1904.13132)
  [[code]](https://github.com/yukimasano/linear-probes)
  - Yuki M. Asano, Christian Rupprecht, Andrea Vedaldi. *ICLR 2020*
  - Digest: We show that three different and representative methods, BiGAN, RotNet and DeepCluster, can learn the first few layers of a convolutional network from a single image as well as using millions of images and manual labels, provided that strong data augmentation is used. However, for deeper layers the gap with manual supervision cannot be closed even if millions of unlabelled images are used for training.



## Interpretability

### 2019
- Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley Values Approximation.
  [[paper]](https://arxiv.org/abs/1903.10992)
  - Marco Ancona, Cengiz Öztireli, Markus Gross. *ICML 2019*
  - Digest: In this work, by leveraging recent results on uncertainty propagation, we propose a novel, polynomial-time approximation of Shapley values in deep neural networks. We show that our method produces significantly better approximations of Shapley values than existing state-of-the-art attribution methods.

