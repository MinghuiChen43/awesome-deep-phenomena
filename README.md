# Awesome Deep Phenomena [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Our understanding of modern neural networks lags behind their practical successes. This growing gap poses a challenge to the pace of progress in machine learning because fewer pillars of knowledge are available to designers of models and algorithms. Inspired by the [ICML 2019 workshop Identifying and Understanding Deep Learning Phenomena](http://deep-phenomena.org/), I collect papers which present interesting empirical study and insight into the nature of deep learning.  

## Table of Contents

- [Empirical Study](#empirical-study)
- [Deep Double Descent](#deep-double-descent)
- [Lottery Ticket Hypothesis](#lottery-ticket-hypothesis)
- [Interactions with Neuroscience](#interactions-with-neuroscience)
- [Information Bottleneck](#information-bottleneck)
- [Neural Tangent Kernel](#neural-tangent-kernel)
- [Others](#others)

## Empirical Study

### Empirical Study: 2022

- Limitations of Neural Collapse for Understanding Generalization in Deep Learning. [[paper]](https://arxiv.org/abs/2202.08384)
  - Like Hui, Mikhail Belkin, Preetum Nakkiran.
  - Key Word: Neural Collapse.
  - Digest: We point out that Neural Collapse is primarily an optimization phenomenon, not a generalization one, by investigating the train collapse and test collapse on various dataset and architecture combinations. We propose more precise definitions — "strong" and "weak" Neural Collapse for both the train set and the test set — and discuss their theoretical feasibility.

### Empirical Study: 2021

- Masked Autoencoders Are Scalable Vision Learners. [[paper]](https://arxiv.org/abs/2111.06377) [[code]](https://github.com/facebookresearch/mae)
  - Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. *CVPR 2022*
  - Key Word: Self-Supervision; Autoencoders.
  - Digest: This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task.

- Exploring the Limits of Large Scale Pre-training. [[paper]](https://arxiv.org/abs/2110.02095)
  - Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, Hanie Sedghi. *ICLR 2022*
  - Key Word: Pre-training.
  - Digest: We investigate more than 4800 experiments on Vision Transformers, MLP-Mixers and ResNets with number of parameters ranging from ten million to ten billion, trained on the largest scale of available image data (JFT, ImageNet21K) and evaluated on more than 20 downstream image recognition tasks. We propose a model for downstream performance that reflects the saturation phenomena and captures the nonlinear relationship in performance of upstream and downstream tasks.

- Stochastic Training is Not Necessary for Generalization. [[paper]](https://arxiv.org/abs/2109.14119) [[code]](https://github.com/JonasGeiping/fullbatchtraining)
  - Jonas Geiping, Micah Goldblum, Phillip E. Pope, Michael Moeller, Tom Goldstein. *ICLR 2022*
  - Key Word: Stochastic Gradient Descent; Regularization.
  - Digest: It is widely believed that the implicit regularization of SGD is fundamental to the impressive generalization behavior we observe in neural networks. In this work, we demonstrate that non-stochastic full-batch training can achieve comparably strong performance to SGD on CIFAR-10 using modern architectures. To this end, we show that the implicit regularization of SGD can be completely replaced with explicit regularization even when comparing against a strong and well-researched baseline.

- Pointer Value Retrieval: A new benchmark for understanding the limits of neural network generalization. [[paper]](https://arxiv.org/abs/2107.12580)
  - Chiyuan Zhang, Maithra Raghu, Jon Kleinberg, Samy Bengio.
  - Key Word: Out-of-Distribution Generalization.
  - Digest: In this paper we introduce a novel benchmark, Pointer Value Retrieval (PVR) tasks, that explore the limits of neural network generalization. We demonstrate that this task structure provides a rich testbed for understanding generalization, with our empirical study showing large variations in neural network performance based on dataset size, task complexity and model architecture.

### Empirical Study: 2020

- When Do Curricula Work? [[paper]](https://arxiv.org/abs/2012.03107) [[code]](https://github.com/google-research/understanding-curricula)
  - Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Curriculum Learning.
  - Digest: We set out to investigate the relative benefits of ordered learning. We first investigate the implicit curricula resulting from architectural and optimization bias and find that samples are learned in a highly consistent order. Next, to quantify the benefit of explicit curricula, we conduct extensive experiments over thousands of orderings spanning three kinds of learning: curriculum, anti-curriculum, and random-curriculum -- in which the size of the training dataset is dynamically increased over time, but the examples are randomly ordered.

- Characterising Bias in Compressed Models. [[paper]](https://arxiv.org/abs/2010.03058)
  - Sara Hooker, Nyalleng Moorosi, Gregory Clark, Samy Bengio, Emily Denton.
  - Key Word: Pruning; Fairness.
  - Digest: The popularity and widespread use of pruning and quantization is driven by the severe resource constraints of deploying deep neural networks to environments with strict latency, memory and energy requirements. These techniques achieve high levels of compression with negligible impact on top-line metrics (top-1 and top-5 accuracy). However, overall accuracy hides disproportionately high errors on a small subset of examples; we call this subset Compression Identified Exemplars (CIE).

- What is being transferred in transfer learning? [[paper]](https://arxiv.org/abs/2008.11687) [[code]](https://github.com/google-research/understanding-transfer-learning)
  - Behnam Neyshabur, Hanie Sedghi, Chiyuan Zhang. *NeurIPS 2020*
  - Key Word: Transfer Learning.
  - Digest: We provide new tools and analyses to address these fundamental questions. Through a series of analyses on transferring to block-shuffled images, we separate the effect of feature reuse from learning low-level statistics of data and show that some benefit of transfer learning comes from the latter. We present that when training from pre-trained weights, the model stays in the same basin in the loss landscape and different instances of such model are similar in feature space and close in parameter space.

- Deep Isometric Learning for Visual Recognition. [[paper]](https://arxiv.org/abs/2006.16992) [[code]](https://github.com/HaozhiQi/ISONet)
  - Haozhi Qi, Chong You, Xiaolong Wang, Yi Ma, Jitendra Malik. *ICML 2020*
  - Key Word: Isometric Networks.
  - Digest: This paper shows that deep vanilla ConvNets without normalization nor skip connections can also be trained to achieve surprisingly good performance on standard image recognition benchmarks. This is achieved by enforcing the convolution kernels to be near isometric during initialization and training, as well as by using a variant of ReLU that is shifted towards being isometric.  

- On the Generalization Benefit of Noise in Stochastic Gradient Descent. [[paper]](https://arxiv.org/abs/2006.15081)
  - Samuel L. Smith, Erich Elsen, Soham De. *ICML 2020*
  - Key Word: Stochastic Gradient Descent.
  - Digest: In this paper, we perform carefully designed experiments and rigorous hyperparameter sweeps on a range of popular models, which verify that small or moderately large batch sizes can substantially outperform very large batches on the test set. This occurs even when both models are trained for the same number of iterations and large batches achieve smaller training losses.  

- Do CNNs Encode Data Augmentations? [[paper]](https://arxiv.org/abs/2003.08773)
  - Eddie Yan, Yanping Huang.
  - Key Word: Data Augmentations.
  - Digest: Surprisingly, neural network features not only predict data augmentation transformations, but they predict many transformations with high accuracy. After validating that neural networks encode features corresponding to augmentation transformations, we show that these features are primarily encoded in the early layers of modern CNNs.  

- Do We Need Zero Training Loss After Achieving Zero Training Error? [[paper]](https://arxiv.org/abs/2002.08709) [[code]](https://github.com/takashiishida/flooding)
  - Takashi Ishida, Ikko Yamane, Tomoya Sakai, Gang Niu, Masashi Sugiyama. *ICML 2020*
  - Key Word: Regularization.
  - Digest:  Our approach makes the loss float around the flooding level by doing mini-batched gradient descent as usual but gradient ascent if the training loss is below the flooding level. This can be implemented with one line of code, and is compatible with any stochastic optimizer and other regularizers. We experimentally show that flooding improves performance and as a byproduct, induces a double descent curve of the test loss.  

- Understanding Why Neural Networks Generalize Well Through GSNR of Parameters. [[paper]](https://arxiv.org/abs/2001.07384)
  - Jinlong Liu, Guoqing Jiang, Yunzhi Bai, Ting Chen, Huayan Wang. *ICLR 2020*
  - Key Word: Generalization Indicators.
  - Digest: In this paper, we provide a novel perspective on these issues using the gradient signal to noise ratio (GSNR) of parameters during training process of DNNs. The GSNR of a parameter is defined as the ratio between its gradient's squared mean and variance, over the data distribution.  

### Empirical Study: 2019

- Angular Visual Hardness. [[paper]](https://arxiv.org/abs/1912.02279)
  - Beidi Chen, Weiyang Liu, Zhiding Yu, Jan Kautz, Anshumali Shrivastava, Animesh Garg, Anima Anandkumar. *ICML 2020*
  - Key Word: Calibration; Example Hardness Measures.
  - Digest: We propose a novel measure for CNN models known as Angular Visual Hardness. Our comprehensive empirical studies show that AVH can serve as an indicator of generalization abilities of neural networks, and improving SOTA accuracy entails improving accuracy on hard example

- Fantastic Generalization Measures and Where to Find Them. [[paper]](https://arxiv.org/abs/1912.02178) [[code]](https://github.com/avakanski/Evaluation-of-Complexity-Measures-for-Deep-Learning-Generalization-in-Medical-Image-Analysis)
  - Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, Samy Bengio. *ICLR 2020*
  - Key Word: Complexity Measures; Spurious Correlations.
  - Digest: We present the first large scale study of generalization in deep networks. We investigate more then 40 complexity measures taken from both theoretical bounds and empirical studies. We train over 10,000 convolutional networks by systematically varying commonly used hyperparameters. Hoping to uncover potentially causal relationships between each measure and generalization, we analyze carefully controlled experiments and show surprising failures of some measures as well as promising measures for further research.

- Truth or Backpropaganda? An Empirical Investigation of Deep Learning Theory. [[paper]](https://arxiv.org/abs/1910.00359) [[code]](https://github.com/goldblum/TruthOrBackpropaganda)
  - Micah Goldblum, Jonas Geiping, Avi Schwarzschild, Michael Moeller, Tom Goldstein. *ICLR 2020*
  - Key Word: Local Minima.
  - Digest: The authors take a closer look at widely held beliefs about neural networks. Using a mix of analysis and experiment, they shed some light on the ways these assumptions break down.  

- Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML. [[paper]](https://arxiv.org/abs/1909.09157) [[code]](https://github.com/fmu2/PyTorch-MAML)
  - Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals. *ICLR 2020*
  - Key Word: Meta Learning.
  - Digest: Despite MAML's popularity, a fundamental open question remains -- is the effectiveness of MAML due to the meta-initialization being primed for rapid learning (large, efficient changes in the representations) or due to feature reuse, with the meta initialization already containing high quality features? We investigate this question, via ablation studies and analysis of the latent representations, finding that feature reuse is the dominant factor.

- Finding the Needle in the Haystack with Convolutions: on the benefits of architectural bias. [[paper]](https://arxiv.org/abs/1906.06766) [[code]](https://github.com/sdascoli/anarchitectural-search)
  - Stéphane d'Ascoli, Levent Sagun, Joan Bruna, Giulio Biroli. *NeurIPS 2019*
  - Key Word: Architectural Bias.
  - Digest:  In particular, Convolutional Neural Networks (CNNs) are known to perform much better than Fully-Connected Networks (FCNs) on spatially structured data: the architectural structure of CNNs benefits from prior knowledge on the features of the data, for instance their translation invariance. The aim of this work is to understand this fact through the lens of dynamics in the loss landscape.  

- Adversarial Training Can Hurt Generalization. [[paper]](https://arxiv.org/abs/1906.06032)
  - Aditi Raghunathan, Sang Michael Xie, Fanny Yang, John C. Duchi, Percy Liang.
  - Key Word: Adversarial Examples.
  - Digest: While adversarial training can improve robust accuracy (against an adversary), it sometimes hurts standard accuracy (when there is no adversary). Previous work has studied this tradeoff between standard and robust accuracy, but only in the setting where no predictor performs well on both objectives in the infinite data limit. In this paper, we show that even when the optimal predictor with infinite data performs well on both objectives, a tradeoff can still manifest itself with finite data.

- Bad Global Minima Exist and SGD Can Reach Them. [[paper]](https://arxiv.org/abs/1906.02613) [[code]](https://github.com/chao1224/BadGlobalMinima)
  - Shengchao Liu, Dimitris Papailiopoulos, Dimitris Achlioptas. *NeurIPS 2020*
  - Key Word: Stochastic Gradient Descent.
  - Digest: Several works have aimed to explain why overparameterized neural networks generalize well when trained by Stochastic Gradient Descent (SGD). The consensus explanation that has emerged credits the randomized nature of SGD for the bias of the training process towards low-complexity models and, thus, for implicit regularization. We take a careful look at this explanation in the context of image classification with common deep neural network architectures. We find that if we do not regularize explicitly, then SGD can be easily made to converge to poorly-generalizing, high-complexity models: all it takes is to first train on a random labeling on the data, before switching to properly training with the correct labels.

- Deep ReLU Networks Have Surprisingly Few Activation Patterns. [[paper]](https://arxiv.org/abs/1906.00904)
  - Boris Hanin, David Rolnick. *NeurIPS 2019*
  - Digest: In this paper, we show that the average number of activation patterns for ReLU networks at initialization is bounded by the total number of neurons raised to the input dimension. We show empirically that this bound, which is independent of the depth, is tight both at initialization and during training, even on memorization tasks that should maximize the number of activation patterns.  

- Sensitivity of Deep Convolutional Networks to Gabor Noise. [[paper]](https://arxiv.org/abs/1906.03455) [[code]](https://github.com/kenny-co/procedural-advml)
  - Kenneth T. Co, Luis Muñoz-González, Emil C. Lupu.
  - Key Word: Robustness.
  - Digest: Deep Convolutional Networks (DCNs) have been shown to be sensitive to Universal Adversarial Perturbations (UAPs): input-agnostic perturbations that fool a model on large portions of a dataset. These UAPs exhibit interesting visual patterns, but this phenomena is, as yet, poorly understood. Our work shows that visually similar procedural noise patterns also act as UAPs. In particular, we demonstrate that different DCN architectures are sensitive to Gabor noise patterns. This behaviour, its causes, and implications deserve further in-depth study.

- Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1905.05928)
  - Guangyong Chen, Pengfei Chen, Yujun Shi, Chang-Yu Hsieh, Benben Liao, Shengyu Zhang.
  - Key Word: Batch Normalization; Dropout.
  - Digest: Our work is based on an excellent idea that whitening the inputs of neural networks can achieve a fast convergence speed. Given the well-known fact that independent components must be whitened, we introduce a novel Independent-Component (IC) layer before each weight layer, whose inputs would be made more independent.  

- A critical analysis of self-supervision, or what we can learn from a single image. [[paper]](https://arxiv.org/abs/1904.13132) [[code]](https://github.com/yukimasano/linear-probes)
  - Yuki M. Asano, Christian Rupprecht, Andrea Vedaldi. *ICLR 2020*
  - Key Word: Self-Supervision.
  - Digest: We show that three different and representative methods, BiGAN, RotNet and DeepCluster, can learn the first few layers of a convolutional network from a single image as well as using millions of images and manual labels, provided that strong data augmentation is used. However, for deeper layers the gap with manual supervision cannot be closed even if millions of unlabelled images are used for training.  

- Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet. [[paper]](https://arxiv.org/abs/1904.00760) [[code]](https://github.com/wielandbrendel/bag-of-local-features-models)
  - Wieland Brendel, Matthias Bethge. *ICLR 2019*
  - Key Word: Bag-of-Features.
  - Digest: Our model, a simple variant of the ResNet-50 architecture called BagNet, classifies an image based on the occurrences of small local image features without taking into account their spatial ordering. This strategy is closely related to the bag-of-feature (BoF) models popular before the onset of deep learning and reaches a surprisingly high accuracy on ImageNet.  

- Transfusion: Understanding Transfer Learning for Medical Imaging. [[paper]](https://arxiv.org/abs/1902.07208) [[code]](https://github.com/PasqualeZingo/TransfusionReproducibilityChallenge)
  - Maithra Raghu, Chiyuan Zhang, Jon Kleinberg, Samy Bengio. *NeurIPS 2019*
  - Key Word: Transfer Learning; Medical Imaging.
  - Digest: we explore properties of transfer learning for medical imaging. A performance evaluation on two large scale medical imaging tasks shows that surprisingly, transfer offers little benefit to performance, and simple, lightweight models can perform comparably to ImageNet architectures.

- Identity Crisis: Memorization and Generalization under Extreme Overparameterization. [[paper]](https://arxiv.org/abs/1902.04698)
  - Chiyuan Zhang, Samy Bengio, Moritz Hardt, Michael C. Mozer, Yoram Singer. *ICLR 2020*
  - Key Word: Memorization.
  - Digest: We study the interplay between memorization and generalization of overparameterized networks in the extreme case of a single training example and an identity-mapping task.  

- Are All Layers Created Equal? [[paper]](https://arxiv.org/abs/1902.01996)  
  - Chiyuan Zhang, Samy Bengio, Yoram Singer. *ICML 2019 Workshop*
  - Key Word: Robustness.
  - Digest: We show that the layers can be categorized as either "ambient" or "critical". Resetting the ambient layers to their initial values has no negative consequence, and in many cases they barely change throughout training. On the contrary, resetting the critical layers completely destroys the predictor and the performance drops to chance.  

### Empirical Study: 2018

- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. [[paper]](https://arxiv.org/abs/1812.05720) [[code]](https://github.com/max-andr/relu_networks_overconfident)
  - Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf. *CVPR 2019*
  - Key Word: ReLU.
  - Digest: Classifiers used in the wild, in particular for safety-critical systems, should not only have good generalization properties but also should know when they don't know, in particular make low confidence predictions far away from the training data. We show that ReLU type neural networks which yield a piecewise linear classifier function fail in this regard as they produce almost always high confidence predictions far away from the training data.  

- On Implicit Filter Level Sparsity in Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1811.12495)
  - Dushyant Mehta, Kwang In Kim, Christian Theobalt. *CVPR 2019*
  - Key Word: Regularization; Sparsification.
  - Digest: We investigate filter level sparsity that emerges in convolutional neural networks (CNNs) which employ Batch Normalization and ReLU activation, and are trained with adaptive gradient descent techniques and L2 regularization or weight decay. We conduct an extensive experimental study casting our initial findings into hypotheses and conclusions about the mechanisms underlying the emergent filter level sparsity. This study allows new insight into the performance gap obeserved between adapative and non-adaptive gradient descent methods in practice.

- Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. [[paper]](https://arxiv.org/abs/1811.12359) [[code]](https://github.com/google-research/disentanglement_lib)
  - Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem. *ICML 2019*
  - Key Word: Disentanglement.
  - Digest: Our results suggest that future work on disentanglement learning should be explicit about the role of inductive biases and (implicit) supervision, investigate concrete benefits of enforcing disentanglement of the learned representations, and consider a reproducible experimental setup covering several data sets.  

- Insights on representational similarity in neural networks with canonical correlation. [[paper]](https://arxiv.org/abs/1806.05759) [[code]](https://github.com/google/svcca)
  - Ari S. Morcos, Maithra Raghu, Samy Bengio. *NeurIPS 2018*
  - Key Word: Representational Similarity.
  - Digest: Comparing representations in neural networks is fundamentally difficult as the structure of representations varies greatly, even across groups of networks trained on identical tasks, and over the course of training. Here, we develop projection weighted CCA (Canonical Correlation Analysis) as a tool for understanding neural networks, building off of SVCCA.

- Layer rotation: a surprisingly powerful indicator of generalization in deep networks? [[paper]](https://arxiv.org/abs/1806.01603) [[code]](https://github.com/ispgroupucl/layer-rotation-paper-experiments)
  - Simon Carbonnelle, Christophe De Vleeschouwer.
  - Key Word: Weight Evolution.
  - Digest: Our work presents extensive empirical evidence that layer rotation, i.e. the evolution across training of the cosine distance between each layer's weight vector and its initialization, constitutes an impressively consistent indicator of generalization performance. In particular, larger cosine distances between final and initial weights of each layer consistently translate into better generalization performance of the final model.

- Sensitivity and Generalization in Neural Networks: an Empirical Study. [[paper]](https://arxiv.org/abs/1802.08760)
  - Roman Novak, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein. *ICLR 2018*
  - Key Word: Sensitivity.
  - Digest: In this work, we investigate this tension between complexity and generalization through an extensive empirical exploration of two natural metrics of complexity related to sensitivity to input perturbations. We find that trained neural networks are more robust to input perturbations in the vicinity of the training data manifold, as measured by the norm of the input-output Jacobian of the network, and that it correlates well with generalization.  

### Empirical Study: 2017

- Deep Image Prior. [[paper]](https://arxiv.org/abs/1711.10925) [[code]](https://dmitryulyanov.github.io/deep_image_prior)
  - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky.
  - Key Word: Low-Level Vision.
  - Digest: In this paper, we show that, on the contrary, the structure of a generator network is sufficient to capture a great deal of low-level image statistics prior to any learning. In order to do so, we show that a randomly-initialized neural network can be used as a handcrafted prior with excellent results in standard inverse problems such as denoising, super-resolution, and inpainting.  

- Critical Learning Periods in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1711.08856)
  - Alessandro Achille, Matteo Rovere, Stefano Soatto. *ICLR 2019*
  - Key Word: Memorization.
  - Digest: Our findings indicate that the early transient is critical in determining the final solution of the optimization associated with training an artificial neural network. In particular, the effects of sensory deficits during a critical period cannot be overcome, no matter how much additional training is performed.  

- A Closer Look at Memorization in Deep Networks. [[paper]](https://arxiv.org/abs/1706.05394)
  - Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, Simon Lacoste-Julien. *ICML 2017*
  - Key Word: Memorization.
  - Digest: In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks (DNNs) on noise vs. real data. We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data.  

### Empirical Study: 2016

- Understanding deep learning requires rethinking generalization. [[paper]](https://arxiv.org/abs/1611.03530)
  - Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *ICLR 2017*
  - Key Word: Memorization.
  - Digest: Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data.  

## Deep Double Descent

### Deep Double Descent 2022

- Phenomenology of Double Descent in Finite-Width Neural Networks. [[paper]](https://arxiv.org/abs/2203.07337) [[code]](https://github.com/sidak/double-descent)
  - Sidak Pal Singh, Aurelien Lucchi, Thomas Hofmann, Bernhard Schölkopf. *ICLR 2022*
  - Key Word: Deep Double Descent.
  - Digest: 'Double descent' delineates the generalization behaviour of models depending on the regime they belong to: under- or over-parameterized. The current theoretical understanding behind the occurrence of this phenomenon is primarily based on linear and kernel regression models -- with informal parallels to neural networks via the Neural Tangent Kernel. Therefore such analyses do not adequately capture the mechanisms behind double descent in finite-width neural networks, as well as, disregard crucial components -- such as the choice of the loss function. We address these shortcomings by leveraging influence functions in order to derive suitable expressions of the population loss and its lower bound, while imposing minimal assumptions on the form of the parametric model.

### Deep Double Descent 2021

- Asymptotic Risk of Overparameterized Likelihood Models: Double Descent Theory for Deep Neural Networks. [[paper]](https://arxiv.org/abs/2103.00500)
  - Ryumei Nakada, Masaaki Imaizumi.
  - Key Word: Deep Double Descent.
  - Digest: We consider a likelihood maximization problem without the model constraints and analyze the upper bound of an asymptotic risk of an estimator with penalization. Technically, we combine a property of the Fisher information matrix with an extended Marchenko-Pastur law and associate the combination with empirical process techniques. The derived bound is general, as it describes both the double descent and the regularized risk curves, depending on the penalization.

- Distilling Double Descent. [[paper]](https://arxiv.org/abs/2102.06849)
  - Andrew Cotter, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, Sashank J. Reddi, Yichen Zhou.
  - Key Word: Deep Double Descent; Distillation.
  - Digest: Distillation is the technique of training a "student" model based on examples that are labeled by a separate "teacher" model, which itself is trained on a labeled dataset. The most common explanations for why distillation "works" are predicated on the assumption that student is provided with soft labels, e.g. probabilities or confidences, from the teacher model. In this work, we show, that, even when the teacher model is highly overparameterized, and provides hard labels, using a very large held-out unlabeled dataset to train the student model can result in a model that outperforms more "traditional" approaches.

### Deep Double Descent: 2020

- Understanding Double Descent Requires a Fine-Grained Bias-Variance Decomposition. [[paper]](https://arxiv.org/abs/2011.03321)
  - Ben Adlam, Jeffrey Pennington. *NeurIPS 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - Digest: Classical learning theory suggests that the optimal generalization performance of a machine learning model should occur at an intermediate model complexity, with simpler models exhibiting high bias and more complex models exhibiting high variance of the predictive function. However, such a simple trade-off does not adequately describe deep learning models that simultaneously attain low bias and variance in the heavily overparameterized regime. A primary obstacle in explaining this behavior is that deep learning algorithms typically involve multiple sources of randomness whose individual contributions are not visible in the total variance. To enable fine-grained analysis, we describe an interpretable, symmetric decomposition of the variance into terms associated with the randomness from sampling, initialization, and the labels.

- Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win. [[paper]](https://arxiv.org/abs/2010.03533) [[code]](https://github.com/google-research/rigl)
  - Utku Evci, Yani A. Ioannou, Cem Keskin, Yann Dauphin. *AAAI 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Sparse Neural Networks (NNs) can match the generalization of dense NNs using a fraction of the compute/storage for inference, and also have the potential to enable efficient training. However, naively training unstructured sparse NNs from random initialization results in significantly worse generalization, with the notable exceptions of Lottery Tickets (LTs) and Dynamic Sparse Training (DST). Through our analysis of gradient flow during training we attempt to answer: (1) why training unstructured sparse networks from random initialization performs poorly and; (2) what makes LTs and DST the exceptions?

- Multiple Descent: Design Your Own Generalization Curve. [[paper]](https://arxiv.org/abs/2008.01036)
  - Lin Chen, Yifei Min, Mikhail Belkin, Amin Karbasi. *NeurIPS 2021*
  - Key Word: Deep Double Descent.
  - Digest: This paper explores the generalization loss of linear regression in variably parameterized families of models, both under-parameterized and over-parameterized. We show that the generalization curve can have an arbitrary number of peaks, and moreover, locations of those peaks can be explicitly controlled. Our results highlight the fact that both classical U-shaped generalization curve and the recently observed double descent curve are not intrinsic properties of the model family. Instead, their emergence is due to the interaction between the properties of the data and the inductive biases of learning algorithms.

- Triple descent and the two kinds of overfitting: Where & why do they appear? [[paper]](https://arxiv.org/abs/2006.03509) [[code]](https://github.com/sdascoli/triple-descent-paper)
  - Stéphane d'Ascoli, Levent Sagun, Giulio Biroli.
  - Digest: Deep Double Descent.
  - Digest: In this paper, we show that despite their apparent similarity, these two scenarios are inherently different. In fact, both peaks can co-exist when neural networks are applied to noisy regression tasks. The relative size of the peaks is governed by the degree of nonlinearity of the activation function. Building on recent developments in the analysis of random feature models, we provide a theoretical ground for this sample-wise triple descent.  

- Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime. [[paper]](https://arxiv.org/abs/2003.01054) [[code]](https://github.com/lightonai/double-trouble-in-double-descent)
  - Stéphane d'Ascoli, Maria Refinetti, Giulio Biroli, Florent Krzakala. *ICML 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - Digest: Deep neural networks can achieve remarkable generalization performances while interpolating the training data perfectly. Rather than the U-curve emblematic of the bias-variance trade-off, their test error often follows a "double descent" - a mark of the beneficial role of overparametrization. In this work, we develop a quantitative theory for this phenomenon in the so-called lazy learning regime of neural networks, by considering the problem of learning a high-dimensional function with random features regression. We obtain a precise asymptotic expression for the bias-variance decomposition of the test error, and show that the bias displays a phase transition at the interpolation threshold, beyond which it remains constant.

- Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. [[paper]](https://arxiv.org/abs/2002.11328) [[code]](https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff)
  - Zitong Yang, Yaodong Yu, Chong You, Jacob Steinhardt, Yi Ma. *ICML 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - Digest: The classical bias-variance trade-off predicts that bias decreases and variance increase with model complexity, leading to a U-shaped risk curve. Recent work calls this into question for neural networks and other over-parameterized models, for which it is often observed that larger models generalize better. We provide a simple explanation for this by measuring the bias and variance of neural networks: while the bias is monotonically decreasing as in the classical theory, the variance is unimodal or bell-shaped: it increases then decreases with the width of the network.

- The Curious Case of Adversarially Robust Models: More Data Can Help, Double Descend, or Hurt Generalization. [[paper]](https://arxiv.org/abs/2002.11080)
  - Yifei Min, Lin Chen, Amin Karbasi. *UAI 2021*
  - Key Word: Deep Double Descent.
  - Digest: We challenge this conventional belief and show that more training data can hurt the generalization of adversarially robust models in the classification problems. We first investigate the Gaussian mixture classification with a linear loss and identify three regimes based on the strength of the adversary. In the weak adversary regime, more data improves the generalization of adversarially robust models. In the medium adversary regime, with more training data, the generalization loss exhibits a double descent curve, which implies the existence of an intermediate stage where more training data hurts the generalization. In the strong adversary regime, more data almost immediately causes the generalization error to increase.

### Deep Double Descent: 2019

- Deep Double Descent: Where Bigger Models and More Data Hurt. [[paper]](https://arxiv.org/abs/1912.02292)  
  - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever. *ICLR 2020*
  - Key Word: Deep Double Descent.
  - Digest: We show that a variety of modern deep learning tasks exhibit a "double-descent" phenomenon where, as we increase model size, performance first gets worse and then gets better.  

### Deep Double Descent: 2018

- Reconciling modern machine learning practice and the bias-variance trade-off. [[paper]](https://arxiv.org/abs/1812.11118)
  - Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal. *PNAS*
  - Key Word: Bias-Variance; Over-Parameterization.
  - Digest: In this paper, we reconcile the classical understanding and the modern practice within a unified performance curve. This "double descent" curve subsumes the textbook U-shaped bias-variance trade-off curve by showing how increasing model capacity beyond the point of interpolation results in improved performance.  

- A Modern Take on the Bias-Variance Tradeoff in Neural Networks. [[paper]](https://arxiv.org/abs/1810.08591)
  - Brady Neal, Sarthak Mittal, Aristide Baratin, Vinayak Tantia, Matthew Scicluna, Simon Lacoste-Julien, Ioannis Mitliagkas.
  - Key Word: Bias-Variance; Over-Parameterization.
  - Digest: The bias-variance tradeoff tells us that as model complexity increases, bias falls and variances increases, leading to a U-shaped test error curve. However, recent empirical results with over-parameterized neural networks are marked by a striking absence of the classic U-shaped test error curve: test error keeps decreasing in wider networks. Motivated by the shaky evidence used to support this claim in neural networks, we measure bias and variance in the modern setting. We find that both bias and variance can decrease as the number of parameters grows. To better understand this, we introduce a new decomposition of the variance to disentangle the effects of optimization and data sampling.

## Lottery Ticket Hypothesis

### Lottery Ticket Hypothesis: 2022

- Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions. [[paper]](https://openreview.net/forum?id=LdEhiMG9WLO) [[code]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning)
  - Shaochen Zhong, Guanqun Zhang, Ningjia Huang, Shuai Xu. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We revisit the idea of kernel pruning, a heavily overlooked approach under the context of structured pruning. This is because kernel pruning will naturally introduce sparsity to filters within the same convolutional layer — thus, making the remaining network no longer dense. We address this problem by proposing a versatile grouped pruning framework where we first cluster filters from each convolutional layer into equal-sized groups, prune the grouped kernels we deem unimportant from each filter group, then permute the remaining filters to form a densely grouped convolutional architecture (which also enables the parallel computing capability) for fine-tuning.

- Proving the Lottery Ticket Hypothesis for Convolutional Neural Networks. [[paper]](https://openreview.net/forum?id=Vjki79-619-)
  - Arthur da Cunha, Emanuele Natale, Laurent Viennot, Laurent_Viennot. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent theoretical works proved an even stronger version: every sufficiently overparameterized (dense) neural network contains a subnetwork that, even without training, achieves accuracy comparable to that of the trained large network. These works left as an open problem to extend the result to convolutional neural networks (CNNs). In this work we provide such generalization by showing that, with high probability, it is possible to approximate any CNN by pruning a random CNN whose size is larger by a logarithmic factor.

- Audio Lottery: Speech Recognition Made Ultra-Lightweight, Noise-Robust, and Transferable. [[paper]](https://openreview.net/forum?id=9Nk6AJkVYB) [[code]](https://github.com/VITA-Group/Audio-Lottery)
  - Shaojin Ding, Tianlong Chen, Zhangyang Wang. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis; Speech Recognition.
  - Digest: We investigate the tantalizing possibility of using lottery ticket hypothesis to discover lightweight speech recognition models, that are (1) robust to various noise existing in speech; (2) transferable to fit the open-world personalization; and 3) compatible with structured sparsity.

- Analyzing Lottery Ticket Hypothesis from PAC-Bayesian Theory Perspective. [[paper]](https://arxiv.org/abs/2205.07320)
  - Keitaro Sakamoto, Issei Sato.
  - Key Word: Lottery Ticket Hypothesis; PAC-Bayes.
  - Digest: We confirm this hypothesis and show that the PAC-Bayesian theory can provide an explicit understanding of the relationship between LTH and generalization behavior. On the basis of our experimental findings that flatness is useful for improving accuracy and robustness to label noise and that the distance from the initial weights is deeply involved in winning tickets, we offer the PAC-Bayes bound using a spike-and-slab distribution to analyze winning tickets.

- Dual Lottery Ticket Hypothesis. [[paper]](https://arxiv.org/abs/2203.04248) [[code]](https://github.com/yueb17/dlth)
  - Yue Bai, Huan Wang, Zhiqiang Tao, Kunpeng Li, Yun Fu. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: This paper articulates a Dual Lottery Ticket Hypothesis (DLTH) as a dual format of original Lottery Ticket Hypothesis (LTH). Correspondingly, a simple regularization based sparse network training strategy, Random Sparse Network Transformation (RST), is proposed to validate DLTH and enhance sparse network training.

- Reconstruction Task Finds Universal Winning Tickets. [[paper]](https://arxiv.org/abs/2202.11484)
  - Ruichen Li, Binghui Li, Qi Qian, Liwei Wang.
  - Key Word: Lottery Ticket Hypothesis; Self-Supervision.
  - Digest: We show that the image-level pretrain task is not capable of pruning models for diverse downstream tasks. To mitigate this problem, we introduce image reconstruction, a pixel-level task, into the traditional pruning framework. Concretely, an autoencoder is trained based on the original model, and then the pruning process is optimized with both autoencoder and classification losses.

- Finding Dynamics Preserving Adversarial Winning Tickets. [[paper]](https://arxiv.org/abs/2202.06488) [[code]](https://github.com/google/neural-tangents)
  - Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang. *AISTATS 2022*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - Digest: Based on recent works of Neural Tangent Kernel (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the lottery ticket hypothesis in adversarial context and we refer such sub-network structure as Adversarial Winning Ticket (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

### Lottery Ticket Hypothesis: 2021

- Plant 'n' Seek: Can You Find the Winning Ticket? [[paper]](https://arxiv.org/abs/2111.11153) [[code]](https://github.com/RelationalML/PlantNSeek)
  - Jonas Fischer, Rebekka Burkholz. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Currently, such algorithms are primarily evaluated on imaging data, for which we lack ground truth information and thus the understanding of how sparse lottery tickets could be. To fill this gap, we develop a framework that allows us to plant and hide winning tickets with desirable properties in randomly initialized neural networks. To analyze the ability of state-of-the-art pruning to identify tickets of extreme sparsity, we design and hide such tickets solving four challenging tasks.

- On the Existence of Universal Lottery Tickets. [[paper]](https://arxiv.org/abs/2111.11146) [[code]](https://github.com/relationalml/universallt)
  - Rebekka Burkholz, Nilanjana Laha, Rajarshi Mukherjee, Alkis Gotovos. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: The lottery ticket hypothesis conjectures the existence of sparse subnetworks of large randomly initialized deep neural networks that can be successfully trained in isolation. Recent work has experimentally observed that some of these tickets can be practically reused across a variety of tasks, hinting at some form of universality. We formalize this concept and theoretically prove that not only do such universal tickets exist but they also do not require further training.

- How many degrees of freedom do we need to train deep networks: a loss landscape perspective. [[paper]](https://arxiv.org/abs/2107.05802) [[code]](https://github.com/ganguli-lab/degrees-of-freedom)
  - Brett W. Larsen, Stanislav Fort, Nic Becker, Surya Ganguli. *ICLR 2022*
  - Key Word: Loss Landscape; Lottery Ticket Hypothesis.
  - Digest: A variety of recent works, spanning pruning, lottery tickets, and training within random subspaces, have shown that deep neural networks can be trained using far fewer degrees of freedom than the total number of parameters. We analyze this phenomenon for random subspaces by first examining the success probability of hitting a training loss sublevel set when training within a random subspace of a given training dimensionality.  

- A Winning Hand: Compressing Deep Networks Can Improve Out-Of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2106.09129)
  - James Diffenderfer, Brian R. Bartoldson, Shreya Chaganti, Jize Zhang, Bhavya Kailkhura. *NeurIPS 2021*
  - Key Word: Lottery Ticket Hypothesis; Out-of-Distribution Generalization.
  - Digest: We perform a large-scale analysis of popular model compression techniques which uncovers several intriguing patterns. Notably, in contrast to traditional pruning approaches (e.g., fine tuning and gradual magnitude pruning), we find that "lottery ticket-style" approaches can surprisingly be used to produce CARDs, including binary-weight CARDs. Specifically, we are able to create extremely compact CARDs that, compared to their larger counterparts, have similar test accuracy and matching (or better) robustness -- simply by pruning and (optionally) quantizing.

- Efficient Lottery Ticket Finding: Less Data is More. [[paper]](https://arxiv.org/abs/2106.03225) [[code]](https://github.com/VITA-Group/PrAC-LTH)
  - Zhenyu Zhang, Xuxi Chen, Tianlong Chen, Zhangyang Wang. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: This paper explores a new perspective on finding lottery tickets more efficiently, by doing so only with a specially selected subset of data, called Pruning-Aware Critical set (PrAC set), rather than using the full training set. The concept of PrAC set was inspired by the recent observation, that deep networks have samples that are either hard to memorize during training, or easy to forget during pruning.

- A Probabilistic Approach to Neural Network Pruning. [[paper]](https://arxiv.org/abs/2105.10065)
  - Xin Qian, Diego Klabjan. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We theoretically study the performance of two pruning techniques (random and magnitude-based) on FCNs and CNNs. Given a target network whose weights are independently sampled from appropriate distributions, we provide a universal approach to bound the gap between a pruned and the target network in a probabilistic sense. The results establish that there exist pruned networks with expressive power within any specified bound from the target network.

- On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning. [[paper]](https://arxiv.org/abs/2105.01648)
  - Marc Aurel Vischer, Robert Tjarko Lange, Henning Sprekeler. *ICLR 2022*
  - Key Word: Reinforcement Learning; Lottery Ticket Hypothesis.
  - Digest: The lottery ticket hypothesis questions the role of overparameterization in supervised deep learning. But how is the performance of winning lottery tickets affected by the distributional shift inherent to reinforcement learning problems? In this work, we address this question by comparing sparse agents who have to address the non-stationarity of the exploration-exploitation problem with supervised agents trained to imitate an expert. We show that feed-forward networks trained with behavioural cloning compared to reinforcement learning can be pruned to higher levels of sparsity without performance degradation.

- Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network. [[paper]](https://arxiv.org/abs/2103.09377) [[code]](https://github.com/chrundle/biprop)
  - James Diffenderfer, Bhavya Kailkhura. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis; Binary Neural Networks.
  - Digest: This provides a new paradigm for learning compact yet highly accurate binary neural networks simply by pruning and quantizing randomly weighted full precision neural networks. We also propose an algorithm for finding multi-prize tickets (MPTs) and test it by performing a series of experiments on CIFAR-10 and ImageNet datasets. Empirical results indicate that as models grow deeper and wider, multi-prize tickets start to reach similar (and sometimes even higher) test accuracy compared to their significantly larger and full-precision counterparts that have been weight-trained.

- Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training. [[paper]](https://arxiv.org/abs/2102.02887) [[code]](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization)
  - Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, Mykola Pechenizkiy. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: In this paper, we introduce a new perspective on training deep neural networks capable of state-of-the-art performance without the need for the expensive over-parameterization by proposing the concept of In-Time Over-Parameterization (ITOP) in sparse training. By starting from a random sparse network and continuously exploring sparse connectivities during training, we can perform an Over-Parameterization in the space-time manifold, closing the gap in the expressibility between sparse training and dense training.

- A Unified Paths Perspective for Pruning at Initialization. [[paper]](https://arxiv.org/abs/2101.10552)
  - Thomas Gebhart, Udit Saxena, Paul Schrater.
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - Digest: Leveraging recent theoretical approximations provided by the Neural Tangent Kernel, we unify a number of popular approaches for pruning at initialization under a single path-centric framework. We introduce the Path Kernel as the data-independent factor in a decomposition of the Neural Tangent Kernel and show the global structure of the Path Kernel can be computed efficiently. This Path Kernel decomposition separates the architectural effects from the data-dependent effects within the Neural Tangent Kernel, providing a means to predict the convergence dynamics of a network from its architecture alone.

### Lottery Ticket Hypothesis: 2020

- PHEW: Constructing Sparse Networks that Learn Fast and Generalize Well without Training Data. [[paper]](https://arxiv.org/abs/2010.11354) [[code]](https://github.com/ShreyasMalakarjunPatil/PHEW)
  - Shreyas Malakarjun Patil, Constantine Dovrolis. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - Digest:  Our work is based on a recently proposed decomposition of the Neural Tangent Kernel (NTK) that has decoupled the dynamics of the training process into a data-dependent component and an architecture-dependent kernel - the latter referred to as Path Kernel. That work has shown how to design sparse neural networks for faster convergence, without any training data, using the Synflow-L2 algorithm. We first show that even though Synflow-L2 is optimal in terms of convergence, for a given network density, it results in sub-networks with "bottleneck" (narrow) layers - leading to poor performance as compared to other data-agnostic methods that use the same number of parameters.

- A Gradient Flow Framework For Analyzing Network Pruning. [[paper]](https://arxiv.org/abs/2009.11839) [[code]](https://github.com/EkdeepSLubana/flowandprune)
  - Ekdeep Singh Lubana, Robert P. Dick. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent network pruning methods focus on pruning models early-on in training. To estimate the impact of removing a parameter, these methods use importance measures that were originally designed to prune trained models. Despite lacking justification for their use early-on in training, such measures result in surprisingly low accuracy loss. To better explain this behavior, we develop a general framework that uses gradient flow to unify state-of-the-art importance measures through the norm of model parameters.

- Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot. [[paper]](https://arxiv.org/abs/2009.11094) [[code]](https://github.com/JingtongSu/sanity-checking-pruning)
  - Jingtong Su, Yihang Chen, Tianle Cai, Tianhao Wu, Ruiqi Gao, Liwei Wang, Jason D. Lee. *NeurIPS 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We conduct sanity checks for the above beliefs on several recent unstructured pruning methods and surprisingly find that: (1) A set of methods which aims to find good subnetworks of the randomly-initialized network (which we call "initial tickets"), hardly exploits any information from the training data; (2) For the pruned networks obtained by these methods, randomly changing the preserved weights in each layer, while keeping the total number of preserved weights unchanged per layer, does not affect the final performance.

- Pruning Neural Networks at Initialization: Why are We Missing the Mark? [[paper]](https://arxiv.org/abs/2009.08576)
  - Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent work has explored the possibility of pruning neural networks at initialization. We assess proposals for doing so: SNIP (Lee et al., 2019), GraSP (Wang et al., 2020), SynFlow (Tanaka et al., 2020), and magnitude pruning. Although these methods surpass the trivial baseline of random pruning, they remain below the accuracy of magnitude pruning after training, and we endeavor to understand why. We show that, unlike pruning after training, randomly shuffling the weights these methods prune within each layer or sampling new initial values preserves or improves accuracy. As such, the per-weight pruning decisions made by these methods can be replaced by a per-layer choice of the fraction of weights to prune. This property suggests broader challenges with the underlying pruning heuristics, the desire to prune at initialization, or both.

- ESPN: Extremely Sparse Pruned Networks. [[paper]](https://arxiv.org/abs/2006.15741) [[code]](https://github.com/chomd90/extreme_sparse)
  - Minsu Cho, Ameya Joshi, Chinmay Hegde.
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Deep neural networks are often highly overparameterized, prohibiting their use in compute-limited systems. However, a line of recent works has shown that the size of deep networks can be considerably reduced by identifying a subset of neuron indicators (or mask) that correspond to significant weights prior to training. We demonstrate that an simple iterative mask discovery method can achieve state-of-the-art compression of very deep networks. Our algorithm represents a hybrid approach between single shot network pruning methods (such as SNIP) with Lottery-Ticket type approaches. We validate our approach on several datasets and outperform several existing pruning approaches in both test accuracy and compression ratio.

- Logarithmic Pruning is All You Need. [[paper]](https://arxiv.org/abs/2006.12156)
  - Laurent Orseau, Marcus Hutter, Omar Rivasplata. *NeurIPS 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: The Lottery Ticket Hypothesis is a conjecture that every large neural network contains a subnetwork that, when trained in isolation, achieves comparable performance to the large network. An even stronger conjecture has been proven recently: Every sufficiently overparameterized network contains a subnetwork that, at random initialization, but without training, achieves comparable accuracy to the trained large network. This latter result, however, relies on a number of strong assumptions and guarantees a polynomial factor on the size of the large network compared to the target function. In this work, we remove the most limiting assumptions of this previous work while providing significantly tighter bounds:the overparameterized network only needs a logarithmic factor (in all variables but depth) number of neurons per weight of the target subnetwork.

- Exploring Weight Importance and Hessian Bias in Model Pruning. [[paper]](https://arxiv.org/abs/2006.10903)
  - Mingchen Li, Yahya Sattar, Christos Thrampoulidis, Samet Oymak.
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Model pruning is an essential procedure for building compact and computationally-efficient machine learning models. A key feature of a good pruning algorithm is that it accurately quantifies the relative importance of the model weights. While model pruning has a rich history, we still don't have a full grasp of the pruning mechanics even for relatively simple problems involving linear models or shallow neural nets. In this work, we provide a principled exploration of pruning by building on a natural notion of importance.

- Progressive Skeletonization: Trimming more fat from a network at initialization. [[paper]](https://arxiv.org/abs/2006.09081) [[code]](https://github.com/naver/force)
  - Pau de Jorge, Amartya Sanyal, Harkirat S. Behl, Philip H.S. Torr, Gregory Rogez, Puneet K. Dokania. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent studies have shown that skeletonization (pruning parameters) of networks at initialization provides all the practical benefits of sparsity both at inference and training time, while only marginally degrading their performance. However, we observe that beyond a certain level of sparsity (approx 95%), these approaches fail to preserve the network performance, and to our surprise, in many cases perform even worse than trivial random pruning. To this end, we propose an objective to find a skeletonized network with maximum foresight connection sensitivity (FORCE) whereby the trainability, in terms of connection sensitivity, of a pruned network is taken into consideration.

- Pruning neural networks without any data by iteratively conserving synaptic flow. [[paper]](https://arxiv.org/abs/2006.05467) [[code]](https://github.com/ganguli-lab/Synaptic-Flow)
  - Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, Surya Ganguli.
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent works have identified, through an expensive sequence of training and pruning cycles, the existence of winning lottery tickets or sparse trainable subnetworks at initialization. This raises a foundational question: can we identify highly sparse trainable subnetworks at initialization, without ever training, or indeed without ever looking at the data? We provide an affirmative answer to this question through theory driven algorithm design.  

- Finding trainable sparse networks through Neural Tangent Transfer. [[paper]](https://arxiv.org/abs/2006.08228) [[code]](https://github.com/fmi-basel/neural-tangent-transfer)
  - Tianlin Liu, Friedemann Zenke. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - Digest: We introduce Neural Tangent Transfer, a method that instead finds trainable sparse networks in a label-free manner. Specifically, we find sparse networks whose training dynamics, as characterized by the neural tangent kernel, mimic those of dense networks in function space. Finally, we evaluate our label-agnostic approach on several standard classification tasks and show that the resulting sparse networks achieve higher classification performance while converging faster.

- What is the State of Neural Network Pruning? [[paper]](https://arxiv.org/abs/2003.03033) [[code]](https://github.com/jjgo/shrinkbench)
  - Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag. *MLSys 2020*
  - Key Word: Lottery Ticket Hypothesis; Survey.
  - Digest: Neural network pruning---the task of reducing the size of a network by removing parameters---has been the subject of a great deal of work in recent years. We provide a meta-analysis of the literature, including an overview of approaches to pruning and consistent findings in the literature. After aggregating results across 81 papers and pruning hundreds of models in controlled conditions, our clearest finding is that the community suffers from a lack of standardized benchmarks and metrics. This deficiency is substantial enough that it is hard to compare pruning techniques to one another or determine how much progress the field has made over the past three decades. To address this situation, we identify issues with current practices, suggest concrete remedies, and introduce ShrinkBench, an open-source framework to facilitate standardized evaluations of pruning methods.

- Comparing Rewinding and Fine-tuning in Neural Network Pruning. [[paper]](https://arxiv.org/abs/2003.02389) [[code]](https://github.com/lottery-ticket/rewinding-iclr20-public)
  - Alex Renda, Jonathan Frankle, Michael Carbin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We compare fine-tuning to alternative retraining techniques. Weight rewinding (as proposed by Frankle et al., (2019)), rewinds unpruned weights to their values from earlier in training and retrains them from there using the original training schedule. Learning rate rewinding (which we propose) trains the unpruned weights from their final values using the same learning rate schedule as weight rewinding. Both rewinding techniques outperform fine-tuning, forming the basis of a network-agnostic pruning algorithm that matches the accuracy and compression ratios of several more network-specific state-of-the-art techniques.

- Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection. [[paper]](https://arxiv.org/abs/2003.01794) [[code]](https://github.com/lushleaf/Network-Pruning-Greedy-Forward-Selection)
  - Mao Ye, Chengyue Gong, Lizhen Nie, Denny Zhou, Adam Klivans, Qiang Liu. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Recent empirical works show that large deep neural networks are often highly redundant and one can find much smaller subnetworks without a significant drop of accuracy. However, most existing methods of network pruning are empirical and heuristic, leaving it open whether good subnetworks provably exist, how to find them efficiently, and if network pruning can be provably better than direct training using gradient descent. We answer these problems positively by proposing a simple greedy selection approach for finding good subnetworks, which starts from an empty network and greedily adds important neurons from the large network.

- The Early Phase of Neural Network Training. [[paper]](https://arxiv.org/abs/2002.10365) [[code]](https://github.com/facebookresearch/open_lth)
  - Jonathan Frankle, David J. Schwab, Ari S. Morcos. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest:  We find that, within this framework, deep networks are not robust to reinitializing with random weights while maintaining signs, and that weight distributions are highly non-independent even after only a few hundred iterations.  

- Robust Pruning at Initialization. [[paper]](https://arxiv.org/abs/2002.08797)
  - Soufiane Hayou, Jean-Francois Ton, Arnaud Doucet, Yee Whye Teh.
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: we provide a comprehensive theoretical analysis of Magnitude and Gradient based pruning at initialization and training of sparse architectures. This allows us to propose novel principled approaches which we validate experimentally on a variety of NN architectures.

- Picking Winning Tickets Before Training by Preserving Gradient Flow. [[paper]](https://arxiv.org/abs/2002.07376) [[code]](https://github.com/alecwangcq/GraSP)
  - Chaoqi Wang, Guodong Zhang, Roger Grosse. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We aim to prune networks at initialization, thereby saving resources at training time as well. Specifically, we argue that efficient training requires preserving the gradient flow through the network. This leads to a simple but effective pruning criterion we term Gradient Signal Preservation (GraSP).

- Lookahead: A Far-Sighted Alternative of Magnitude-based Pruning. [[paper]](https://arxiv.org/abs/2002.04809) [[code]](https://github.com/alinlab/lookahead_pruning)
  - Sejun Park, Jaeho Lee, Sangwoo Mo, Jinwoo Shin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: Magnitude-based pruning is one of the simplest methods for pruning neural networks. Despite its simplicity, magnitude-based pruning and its variants demonstrated remarkable performances for pruning modern architectures. Based on the observation that magnitude-based pruning indeed minimizes the Frobenius distortion of a linear operator corresponding to a single layer, we develop a simple pruning method, coined lookahead pruning, by extending the single layer optimization to a multi-layer optimization.

### Lottery Ticket Hypothesis: 2019

- Linear Mode Connectivity and the Lottery Ticket Hypothesis. [[paper]](https://arxiv.org/abs/1912.05671)
  - Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We study whether a neural network optimizes to the same, linearly connected minimum under different samples of SGD noise (e.g., random data order and augmentation). We find that standard vision models become stable to SGD noise in this way early in training. From then on, the outcome of optimization is determined to a linearly connected region. We use this technique to study iterative magnitude pruning (IMP), the procedure used by work on the lottery ticket hypothesis to identify subnetworks that could have trained in isolation to full accuracy.

- What's Hidden in a Randomly Weighted Neural Network? [[paper]](https://arxiv.org/abs/1911.13299) [[code]](https://github.com/allenai/hidden-networks)
  - Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari. *CVPR 2020*
  - Key Word: Lottery Ticket Hypothesis; Neural Architecture Search; Weight Agnositic Neural Networks.
  - Digest: Hidden in a randomly weighted Wide ResNet-50 we show that there is a subnetwork (with random weights) that is smaller than, but matches the performance of a ResNet-34 trained on ImageNet. Not only do these "untrained subnetworks" exist, but we provide an algorithm to effectively find them.  

- Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks. [[paper]](https://arxiv.org/abs/1909.11957) [[code]](https://github.com/RICE-EIC/Early-Bird-Tickets)
  - Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, Yingyan Lin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We discover for the first time that the winning tickets can be identified at the very early training stage, which we term as early-bird (EB) tickets, via low-cost training schemes (e.g., early stopping and low-precision training) at large learning rates. Our finding of EB tickets is consistent with recently reported observations that the key connectivity patterns of neural networks emerge early.

- Rigging the Lottery: Making All Tickets Winners. [[paper]](https://arxiv.org/abs/1911.11134) [[code]](https://github.com/google-research/rigl)
  - Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, Erich Elsen. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We introduce a method to train sparse neural networks with a fixed parameter count and a fixed computational cost throughout training, without sacrificing accuracy relative to existing dense-to-sparse training methods. Our method updates the topology of the sparse network during training by using parameter magnitudes and infrequent gradient calculations. We show that this approach requires fewer floating-point operations (FLOPs) to achieve a given level of accuracy compared to prior techniques.

- The Difficulty of Training Sparse Neural Networks. [[paper]](https://arxiv.org/abs/1906.10732)
  - Utku Evci, Fabian Pedregosa, Aidan Gomez, Erich Elsen.
  - Key Word: Pruning.
  - Digest: We investigate the difficulties of training sparse neural networks and make new observations about optimization dynamics and the energy landscape within the sparse regime. Recent work of has shown that sparse ResNet-50 architectures trained on ImageNet-2012 dataset converge to solutions that are significantly worse than those found by pruning. We show that, despite the failure of optimizers, there is a linear path with a monotonically decreasing objective from the initialization to the "good" solution.

- A Signal Propagation Perspective for Pruning Neural Networks at Initialization. [[paper]](https://arxiv.org/abs/1906.06307) [[code]](https://github.com/namhoonlee/spp-public)
  - Namhoon Lee, Thalaiyasingam Ajanthan, Stephen Gould, Philip H. S. Torr. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis; Mean Field Theory.
  - Digest: In this work, by noting connection sensitivity as a form of gradient, we formally characterize initialization conditions to ensure reliable connection sensitivity measurements, which in turn yields effective pruning results. Moreover, we analyze the signal propagation properties of the resulting pruned networks and introduce a simple, data-free method to improve their trainability.  

- One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. [[paper]](https://arxiv.org/abs/1906.02773)
  - Ari S. Morcos, Haonan Yu, Michela Paganini, Yuandong Tian. *NeurIPS 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest:  Perhaps surprisingly, we found that, within the natural images domain, winning ticket initializations generalized across a variety of datasets, including Fashion MNIST, SVHN, CIFAR-10/100, ImageNet, and Places365, often achieving performance close to that of winning tickets generated on the same dataset.  

- Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask. [[paper]](https://arxiv.org/abs/1905.01067) [[code]](https://github.com/uber-research/deconstructing-lottery-tickets)
  - Hattie Zhou, Janice Lan, Rosanne Liu, Jason Yosinski. *NeurIPS 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: In this paper, we have studied how three components to LT-style network pruning—mask criterion, treatment of kept weights during retraining (mask-1 action), and treatment of pruned weights during retraining (mask-0 action)—come together to produce sparse and performant subnetworks.

- The State of Sparsity in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1902.09574) [[code]](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn)
  - Trevor Gale, Erich Elsen, Sara Hooker.
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We rigorously evaluate three state-of-the-art techniques for inducing sparsity in deep neural networks on two large-scale learning tasks: Transformer trained on WMT 2014 English-to-German, and ResNet-50 trained on ImageNet. Across thousands of experiments, we demonstrate that complex techniques (Molchanov et al., 2017; Louizos et al., 2017b) shown to yield high compression rates on smaller datasets perform inconsistently, and that simple magnitude pruning approaches achieve comparable or better results.

### Lottery Ticket Hypothesis: 2018

- SNIP: Single-shot Network Pruning based on Connection Sensitivity. [[paper]](https://arxiv.org/abs/1810.02340) [[code]](https://github.com/namhoonlee/snip-public)
  - Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr. *ICLR 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: In this work, we present a new approach that prunes a given network once at initialization prior to training. To achieve this, we introduce a saliency criterion based on connection sensitivity that identifies structurally important connections in the network for the given task.  

- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. [[paper]](https://arxiv.org/abs/1803.03635) [[code]](https://github.com/google-research/lottery-ticket-hypothesis)
  - Jonathan Frankle, Michael Carbin *ICLR 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations.  

## Interactions with Neuroscience

### Interactions with Neuroscience: 2021

- Partial success in closing the gap between human and machine vision. [[paper]](https://arxiv.org/abs/2106.07411) [[code]](https://github.com/bethgelab/model-vs-human)
  - Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *NeurIPS 2021*
  - Key Word: Out-of-Distribution Generalization; Psychophysical Experiments.
  - Digest: A few years ago, the first CNN surpassed human performance on ImageNet. However, it soon became clear that machines lack robustness on more challenging test cases, a major obstacle towards deploying machines "in the wild" and towards obtaining better computational models of human visual perception. Here we ask: Are we making progress in closing the gap between human and machine vision? To answer this question, we tested human observers on a broad range of out-of-distribution (OOD) datasets, recording 85,120 psychophysical trials across 90 participants.

- Does enhanced shape bias improve neural network robustness to common corruptions? [[paper]](https://arxiv.org/abs/2104.09789)
  - Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, Jan Hendrik Metzen. *ICLR 2021*
  - Key Word: Shape-Texture; Robustness.
  - Digest: We perform a systematic study of different ways of composing inputs based on natural images, explicit edge information, and stylization. While stylization is essential for achieving high corruption robustness, we do not find a clear correlation between shape bias and robustness. We conclude that the data augmentation caused by style-variation accounts for the improved corruption robustness and increased shape bias is only a byproduct.

### Interactions with Neuroscience: 2020

- Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html)
  - Joel Dapello, Tiago Marques, Martin Schrimpf, Franziska Geiger, David Cox, James J. DiCarlo. *NeurIPS 2020*
  - Key Word: Robustness; V1 Model.
  - Digest: Current state-of-the-art object recognition models are largely based on convolutional neural network (CNN) architectures, which are loosely inspired by the primate visual system. However, these CNNs can be fooled by imperceptibly small, explicitly crafted perturbations, and struggle to recognize objects in corrupted images that are easily recognized by humans. Here, by making comparisons with primate neural data, we first observed that CNN models with a neural hidden layer that better matches primate primary visual cortex (V1) are also more robust to adversarial attacks. Inspired by this observation, we developed VOneNets, a new class of hybrid CNN vision models. Each VOneNet contains a fixed weight neural network front-end that simulates primate V1, called the VOneBlock, followed by a neural network back-end adapted from current CNN vision models.

- Shape-Texture Debiased Neural Network Training. [[paper]](https://arxiv.org/abs/2010.05981) [[code]](https://github.com/LiYingwei/ShapeTextureDebiasedTraining)
  - Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan Yuille, Cihang Xie. *ICLR 2021*
  - Key Word: Shape-Texture; Robustness.
  - Digest: Shape and texture are two prominent and complementary cues for recognizing objects. Nonetheless, Convolutional Neural Networks are often biased towards either texture or shape, depending on the training dataset. Our ablation shows that such bias degenerates model performance. Motivated by this observation, we develop a simple algorithm for shape-texture debiased learning. To prevent models from exclusively attending on a single cue in representation learning, we augment training data with images with conflicting shape and texture information (eg, an image of chimpanzee shape but with lemon texture) and, most importantly, provide the corresponding supervisions from shape and texture simultaneously.

- Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency. [[paper]](https://arxiv.org/abs/2006.16736) [[code]](https://github.com/wichmann-lab/error-consistency)
  - Robert Geirhos, Kristof Meding, Felix A. Wichmann.
  - Key Word: Error Consistency.
  - Digest: Here we introduce trial-by-trial error consistency, a quantitative analysis for measuring whether two decision making systems systematically make errors on the same inputs. Making consistent errors on a trial-by-trial basis is a necessary condition if we want to ascertain similar processing strategies between decision makers.  

- Biologically Inspired Mechanisms for Adversarial Robustness. [[paper]](https://arxiv.org/abs/2006.16427)
  - Manish V. Reddy, Andrzej Banburski, Nishka Pant, Tomaso Poggio. *NeurIPS 2020*
  - Key Word: Robustness; Retinal Fixations.
  - Digest: A convolutional neural network strongly robust to adversarial perturbations at reasonable computational and performance cost has not yet been demonstrated. The primate visual ventral stream seems to be robust to small perturbations in visual stimuli but the underlying mechanisms that give rise to this robust perception are not understood. In this work, we investigate the role of two biologically plausible mechanisms in adversarial robustness. We demonstrate that the non-uniform sampling performed by the primate retina and the presence of multiple receptive fields with a range of receptive field sizes at each eccentricity improve the robustness of neural networks to small adversarial perturbations

- Five Points to Check when Comparing Visual Perception in Humans and Machines. [[paper]](https://arxiv.org/abs/2004.09406) [[code]](https://github.com/bethgelab/notorious_difficulty_of_comparing_human_and_machine_perception)
  - Christina M. Funke, Judy Borowski, Karolina Stosio, Wieland Brendel, Thomas S. A. Wallis, Matthias Bethge. *JOV*
  - Key Word: Model Comparison.
  - Digest: With the rise of machines to human-level performance in complex recognition tasks, a growing amount of work is directed towards comparing information processing in humans and machines. These studies are an exciting chance to learn about one system by studying the other. Here, we propose ideas on how to design, conduct and interpret experiments such that they adequately support the investigation of mechanisms when comparing human and machine perception. We demonstrate and apply these ideas through three case studies.

- Shortcut Learning in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2004.07780) [[code]](https://github.com/rgeirhos/shortcut-perspective)
  - Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. *Nature Machine Intelligence*
  - Key Word: Out-of-Distribution Generalization; Survey.
  - Digest: Deep learning has triggered the current rise of artificial intelligence and is the workhorse of today's machine intelligence. Numerous success stories have rapidly spread all over science, industry and society, but its limitations have only recently come into focus. In this perspective we seek to distil how many of deep learning's problem can be seen as different symptoms of the same underlying problem: shortcut learning. Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios. Related issues are known in Comparative Psychology, Education and Linguistics, suggesting that shortcut learning may be a common characteristic of learning systems, biological and artificial alike. Based on these observations, we develop a set of recommendations for model interpretation and benchmarking, highlighting recent advances in machine learning to improve robustness and transferability from the lab to real-world applications.

### Interactions with Neuroscience: 2019

- White Noise Analysis of Neural Networks. [[paper]](https://arxiv.org/abs/1912.12106) [[code]](https://github.com/aliborji/WhiteNoiseAnalysis)
  - Ali Borji, Sikun Lin. *ICLR 2020*
  - Key Word: Spike-Triggered Analysis.
  - Digest: A white noise analysis of modern deep neural networks is presented to unveil their biases at the whole network level or the single neuron level. Our analysis is based on two popular and related methods in psychophysics and neurophysiology namely classification images and spike triggered analysis.  

- The Origins and Prevalence of Texture Bias in Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1911.09071)
  - Katherine L. Hermann, Ting Chen, Simon Kornblith. *NeurIPS 2020*
  - Key Word: Shape-Texture; Robustness.
  - Digest: Recent work has indicated that, unlike humans, ImageNet-trained CNNs tend to classify images by texture rather than by shape. How pervasive is this bias, and where does it come from? We find that, when trained on datasets of images with conflicting shape and texture, CNNs learn to classify by shape at least as easily as by texture. What factors, then, produce the texture bias in CNNs trained on ImageNet? Different unsupervised training objectives and different architectures have small but significant and largely independent effects on the level of texture bias. However, all objectives and architectures still lead to models that make texture-based classification decisions a majority of the time, even if shape information is decodable from their hidden representations. The effect of data augmentation is much larger.

- Learning From Brains How to Regularize Machines. [[paper]](https://arxiv.org/abs/1911.05072)
  - Zhe Li, Wieland Brendel, Edgar Y. Walker, Erick Cobos, Taliah Muhammad, Jacob Reimer, Matthias Bethge, Fabian H. Sinz, Xaq Pitkow, Andreas S. Tolias. *NeurIPS 2019*
  - Key Word: Neural Representation Similarity.
  - Digest: Despite impressive performance on numerous visual tasks, Convolutional Neural Networks (CNNs) --- unlike brains --- are often highly sensitive to small perturbations of their input, e.g. adversarial noise leading to erroneous decisions. We propose to regularize CNNs using large-scale neuroscience data to learn more robust neural features in terms of representational similarity. We presented natural images to mice and measured the responses of thousands of neurons from cortical visual areas.

### Interactions with Neuroscience: 2018

- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. [[paper]](https://arxiv.org/abs/1811.12231) [[code]](https://github.com/rgeirhos/Stylized-ImageNet)
  - Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *ICLR 2019*
  - Key Word: Shape-Texture; Psychophysical Experiments.
  - Digest: Convolutional Neural Networks (CNNs) are commonly thought to recognise objects by learning increasingly complex representations of object shapes. Some recent studies suggest a more important role of image textures. We here put these conflicting hypotheses to a quantitative test by evaluating CNNs and human observers on images with a texture-shape cue conflict. We show that ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark contrast to human behavioural evidence and reveals fundamentally different classification strategies.

- Generalisation in humans and deep neural networks. [[paper]](https://arxiv.org/abs/1808.08750) [[code]](https://github.com/rgeirhos/generalisation-humans-DNNs)
  - Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Robustness.
  - Digest: We compare the robustness of humans and current convolutional deep neural networks (DNNs) on object recognition under twelve different types of image degradations. First, using three well known DNNs (ResNet-152, VGG-19, GoogLeNet) we find the human visual system to be more robust to nearly all of the tested image manipulations, and we observe progressively diverging classification error-patterns between humans and DNNs when the signal gets weaker. Secondly, we show that DNNs trained directly on distorted images consistently surpass human performance on the exact distortion types they were trained on, yet they display extremely poor generalisation abilities when tested on other distortion types.

### Interactions with Neuroscience: 2017

- Comparing deep neural networks against humans: object recognition when the signal gets weaker. [[paper]](https://arxiv.org/abs/1706.06969) [[code]](https://github.com/rgeirhos/object-recognition)
  - Robert Geirhos, David H. J. Janssen, Heiko H. Schütt, Jonas Rauber, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Model Comparison; Robustness.
  - Digest: Human visual object recognition is typically rapid and seemingly effortless, as well as largely independent of viewpoint and object orientation. Until very recently, animate visual systems were the only ones capable of this remarkable computational feat. This has changed with the rise of a class of computer vision algorithms called deep neural networks (DNNs) that achieve human-level classification performance on object recognition tasks. Furthermore, a growing number of studies report similarities in the way DNNs and the human visual system process objects, suggesting that current DNNs may be good models of human visual object recognition. Yet there clearly exist important architectural and processing differences between state-of-the-art DNNs and the primate visual system. The potential behavioural consequences of these differences are not well understood. We aim to address this issue by comparing human and DNN generalisation abilities towards image degradations.

## Information Bottleneck

## Information Bottleneck: 2022

- Sparsity-Inducing Categorical Prior Improves Robustness of the Information Bottleneck. [[paper]](https://arxiv.org/abs/2203.02592)
  - Anirban Samaddar, Sandeep Madireddy, Prasanna Balaprakash
  - Key Word: Information Bottleneck; Robustness.
  - Digest: We present a novel sparsity-inducing spike-slab prior that uses sparsity as a mechanism to provide flexibility that allows each data point to learn its own dimension distribution. In addition, it provides a mechanism to learn a joint distribution of the latent variable and the sparsity. Thus, unlike other approaches, it can account for the full uncertainty in the latent space.

## Information Bottleneck: 2021

- Information Bottleneck Disentanglement for Identity Swapping. [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Information_Bottleneck_Disentanglement_for_Identity_Swapping_CVPR_2021_paper.html)
  - Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, Ran He. *CVPR 2021*
  - Key Word: Information Bottleneck; Identity Swapping.
  - Digest: We propose a novel information disentangling and swapping network, called InfoSwap, to extract the most expressive information for identity representation from a pre-trained face recognition model. The key insight of our method is to formulate the learning of disentangled representations as optimizing an information bottleneck trade-off, in terms of finding an optimal compression of the pre-trained latent features.

- PAC-Bayes Information Bottleneck. [[paper]](https://arxiv.org/abs/2109.14509) [[code]](https://github.com/ryanwangzf/pac-bayes-ib)
  - Zifeng Wang, Shao-Lun Huang, Ercan E. Kuruoglu, Jimeng Sun, Xi Chen, Yefeng Zheng. *ICLR 2022*
  - Key Word: Information Bottleneck; PAC-Bayes.
  - Digest: There have been a series of theoretical works trying to derive non-vacuous bounds for NNs. Recently, the compression of information stored in weights (IIW) is proved to play a key role in NNs generalization based on the PAC-Bayes theorem. However, no solution of IIW has ever been provided, which builds a barrier for further investigation of the IIW's property and its potential in practical deep learning. In this paper, we propose an algorithm for the efficient approximation of IIW. Then, we build an IIW-based information bottleneck on the trade-off between accuracy and information complexity of NNs, namely PIB.

- Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.06607) [[code]](https://github.com/ahujak/IB-IRM)
  - Kartik Ahuja, Ethan Caballero, Dinghuai Zhang, Yoshua Bengio, Ioannis Mitliagkas, Irina Rish. *NeurIPS 2021*
  - Key Word: Information Bottleneck; Out-of-Distribution Generalization; Invarianct Risk Minimization.
  - Digest: We revisit the fundamental assumptions in linear regression tasks, where invariance-based approaches were shown to provably generalize OOD. In contrast to the linear regression tasks, we show that for linear classification tasks we need much stronger restrictions on the distribution shifts, or otherwise OOD generalization is impossible.  Furthermore, even with appropriate restrictions on distribution shifts in place, we show that the invariance principle alone is insufficient. We prove that a form of the information bottleneck constraint along with invariance helps address the key failures when invariant features capture all the information about the label and also retains the existing success when they do not.

- A Critical Review of Information Bottleneck Theory and its Applications to Deep Learning. [[paper]](https://arxiv.org/abs/2105.04405)
  - Mohammad Ali Alomrani.
  - Key Word: Information Bottleneck; Survey.
  - Digest: A known information-theoretic method called the information bottleneck theory has emerged as a promising approach to better understand the learning dynamics of neural networks. In principle, IB theory models learning as a trade-off between the compression of the data and the retainment of information. The goal of this survey is to provide a comprehensive review of IB theory covering it's information theoretic roots and the recently proposed applications to understand deep learning models.

## Information Bottleneck: 2020

- Graph Information Bottleneck. [[paper]](https://arxiv.org/abs/2010.12811) [[code]](https://github.com/snap-stanford/GIB)
  - Tailin Wu, Hongyu Ren, Pan Li, Jure Leskovec. *NeurIPS 2020*
  - Key Word: Information Bottleneck; Graph Neural Networks.
  - Digest: We introduce Graph Information Bottleneck (GIB), an information-theoretic principle that optimally balances expressiveness and robustness of the learned representation of graph-structured data. Inheriting from the general Information Bottleneck (IB), GIB aims to learn the minimal sufficient representation for a given task by maximizing the mutual information between the representation and the target, and simultaneously constraining the mutual information between the representation and the input data.

- Learning Optimal Representations with the Decodable Information Bottleneck. [[paper]](https://arxiv.org/abs/2009.12789) [[code]](https://github.com/YannDubs/Mini_Decodable_Information_Bottleneck)
  - Yann Dubois, Douwe Kiela, David J. Schwab, Ramakrishna Vedantam. *NeurIPS 2020*
  - Key Word: Information Bottleneck.
  - Digest: We propose the Decodable Information Bottleneck (DIB) that considers information retention and compression from the perspective of the desired predictive family. As a result, DIB gives rise to representations that are optimal in terms of expected test performance and can be estimated with guarantees. Empirically, we show that the framework can be used to enforce a small generalization gap on downstream classifiers and to predict the generalization ability of neural networks.

- Concept Bottleneck Models. [[paper]](https://arxiv.org/abs/2007.04612) [[code]](https://github.com/yewsiang/ConceptBottleneck)
  - Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang. *ICML 2020*
  - Key Word: Information Bottleneck
  - Digest: We seek to learn models that we can interact with using high-level concepts: if the model did not think there was a bone spur in the x-ray, would it still predict severe arthritis? State-of-the-art models today do not typically support the manipulation of concepts like "the existence of bone spurs", as they are trained end-to-end to go directly from raw input (e.g., pixels) to output (e.g., arthritis severity). We revisit the classic idea of first predicting concepts that are provided at training time, and then using these concepts to predict the label. By construction, we can intervene on these concept bottleneck models by editing their predicted concept values and propagating these changes to the final prediction.

- On Information Plane Analyses of Neural Network Classifiers -- A Review. [[paper]](https://arxiv.org/abs/2003.09671)
  - Bernhard C. Geiger. *TNNLS*
  - Key Word: Information Bottleneck; Survey.
  - Digest: We review the current literature concerned with information plane analyses of neural network classifiers. While the underlying information bottleneck theory and the claim that information-theoretic compression is causally linked to generalization are plausible, empirical evidence was found to be both supporting and conflicting. We review this evidence together with a detailed analysis of how the respective information quantities were estimated.

- On the Information Bottleneck Problems: Models, Connections, Applications and Information Theoretic Views. [[paper]](https://arxiv.org/abs/2002.00008)
  - Abdellatif Zaidi, Inaki Estella Aguerri, Shlomo Shamai. *Entropy*
  - Key Word: Information Bottleneck; Survey.
  - Digest: This tutorial paper focuses on the variants of the bottleneck problem taking an information theoretic perspective and discusses practical methods to solve it, as well as its connection to coding and learning aspects. The intimate connections of this setting to remote source-coding under logarithmic loss distortion measure, information combining, common reconstruction, the Wyner-Ahlswede-Korner problem, the efficiency of investment information, as well as, generalization, variational inference, representation learning, autoencoders, and others are highlighted.

- Phase Transitions for the Information Bottleneck in Representation Learning. [[paper]](https://arxiv.org/abs/2001.01878)
  - Tailin Wu, Ian Fischer. *ICLR 2020*
  - Key Word: Information Bottleneck.
  - Digest: Our work provides the first theoretical formula to address IB phase transitions in the most general setting. In addition, we present an algorithm for iteratively finding the IB phase transition points.

- Restricting the Flow: Information Bottlenecks for Attribution. [[paper]](https://arxiv.org/abs/2001.00396) [[code]](https://github.com/BioroboticsLab/IBA-paper-code)
  - Karl Schulz, Leon Sixt, Federico Tombari, Tim Landgraf. *ICLR 2020*
  - Key Word: Information Bottleneck; Attribution.
  - Digest: We adapt the information bottleneck concept for attribution. By adding noise to intermediate feature maps we restrict the flow of information and can quantify (in bits) how much information image regions provide.

## Information Bottleneck: 2019

- Learnability for the Information Bottleneck. [[paper]](https://arxiv.org/abs/1907.07331)
  - Tailin Wu, Ian Fischer, Isaac L. Chuang, Max Tegmark. *UAI 2019*
  - Key Word: Information Bottleneck.
  - Digest: We presented theoretical results for predicting the onset of learning, and have shown that it is determined by the conspicuous subset of the training examples. We gave a practical algorithm for predicting the transition as well as discovering this subset, and showed that those predictions are accurate, even in cases of extreme label noise.

## Information Bottleneck: 2018

- On the Information Bottleneck Theory of Deep Learning. [[paper]](https://openreview.net/forum?id=ry_WPG-A-) [[code]](https://github.com/artemyk/ibsgd/tree/iclr2018)
  - Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox. *ICLR 2018*
  - Key Word: Information Bottleneck.
  - Digest: This submission explores [recent theoretical work](https://arxiv.org/abs/1703.00810) by Shwartz-Ziv and Tishby on explaining the generalization ability of deep networks. The paper gives counter-examples that suggest aspects of the theory might not be relevant for all neural networks.  

## Information Bottleneck: 2017

- Emergence of Invariance and Disentanglement in Deep Representations. [[paper]](https://arxiv.org/abs/1706.01350)
  - Alessandro Achille, Stefano Soatto. *JMLR*
  - Key Word: PAC-Bayes; Information Bottleneck.
  - Digest: Using established principles from Statistics and Information Theory, we show that invariance to nuisance factors in a deep neural network is equivalent to information minimality of the learned representation, and that stacking layers and injecting noise during training naturally bias the network towards learning invariant representations. We then decompose the cross-entropy loss used during training and highlight the presence of an inherent overfitting term. We propose regularizing the loss by bounding such a term in two equivalent ways: One with a Kullbach-Leibler term, which relates to a PAC-Bayes perspective; the other using the information in the weights as a measure of complexity of a learned model, yielding a novel Information Bottleneck for the weights.

- Information-theoretic analysis of generalization capability of learning algorithms. [[paper]](https://arxiv.org/abs/1705.07809)
  - Aolin Xu, Maxim Raginsky. *NeurIPS 2017*
  - Key Word: Information Bottleneck.
  - Digest: We derive upper bounds on the generalization error of a learning algorithm in terms of the mutual information between its input and output. The bounds provide an information-theoretic understanding of generalization in learning problems, and give theoretical guidelines for striking the right balance between data fit and generalization by controlling the input-output mutual information. We propose a number of methods for this purpose, among which are algorithms that regularize the ERM algorithm with relative entropy or with random noise.

- Opening the Black Box of Deep Neural Networks via Information. [[paper]](https://arxiv.org/abs/1703.00810)
  - Ravid Shwartz-Ziv, Naftali Tishby.
  - Key Word: Information Bottleneck.
  - Digest: [Previous work](https://arxiv.org/abs/1503.02406) proposed to analyze DNNs in the *Information Plane*; i.e., the plane of the Mutual Information values that each layer preserves on the input and output variables. They suggested that the goal of the network is to optimize the Information Bottleneck (IB) tradeoff between compression and prediction, successively, for each layer. In this work we follow up on this idea and demonstrate the effectiveness of the Information-Plane visualization of DNNs.  

## Neural Tangent Kernel

## Neural Tangent Kernel: 2021

- Neural Tangent Generalization Attacks. [[paper]](https://proceedings.mlr.press/v139/yuan21b.html) [[code]](https://github.com/lionelmessi6410/ntga)
  - Chia-Hung Yuan, Shan-Hung Wu. *ICML 2021*
  - Key Word: Neural Tangent Kernel; Poisoning Attacks.
  - Digest: We study the generalization attacks against DNNs, where an attacker aims to slightly modify training data in order to spoil the training process such that a trained network lacks generalizability. These attacks can be performed by data owners and protect data from unexpected use. However, there is currently no efficient generalization attack against DNNs due to the complexity of a bilevel optimization involved. We propose the Neural Tangent Generalization Attack (NTGA) that, to the best of our knowledge, is the first work enabling clean-label, black-box generalization attack against DNNs.

- On the Equivalence between Neural Network and Support Vector Machine. [[paper]](https://arxiv.org/abs/2111.06063) [[code]](https://github.com/leslie-ch/equiv-nn-svm)
  - Yilan Chen, Wei Huang, Lam M. Nguyen, Tsui-Wei Weng. *NeurIPS 2021*
  - Key Word: Neural Tangent Kernel; Support Vector Machine.
  - Digest: We prove the equivalence between neural network (NN) and support vector machine (SVM), specifically, the infinitely wide NN trained by soft margin loss and the standard soft margin SVM with NTK trained by subgradient descent. Our main theoretical results include establishing the equivalence between NN and a broad family of L2 regularized kernel machines (KMs) with finite-width bounds, which cannot be handled by prior work, and showing that every finite-width NN trained by such regularized loss functions is approximately a KM.

- An Empirical Study of Neural Kernel Bandits. [[paper]](https://arxiv.org/abs/2111.03543) [[code]](https://github.com/mlisicki/neuralkernelbandits)
  - Michal Lisicki, Arash Afkanpour, Graham W. Taylor.
  - Key Word: Neural Tangent Kernel.
  - Digest: We propose to directly apply NK-induced distributions to guide an upper confidence bound or Thompson sampling-based policy. We show that NK bandits achieve state-of-the-art performance on highly non-linear structured data. Furthermore, we analyze practical considerations such as training frequency and model partitioning.

- A Neural Tangent Kernel Perspective of GANs. [[paper]](https://arxiv.org/abs/2106.05566) [[code]](https://github.com/emited/gantk2)
  - Jean-Yves Franceschi, Emmanuel de Bézenac, Ibrahim Ayed, Mickaël Chen, Sylvain Lamprier, Patrick Gallinari. *ICML 2021*
  - Key Word: Neural Tangent Kernel; Generative Adversarial Networks.
  - Digest: We propose a novel theoretical framework of analysis for Generative Adversarial Networks (GANs). We start by pointing out a fundamental flaw in previous theoretical analyses that leads to ill-defined gradients for the discriminator. We overcome this issue which impedes a principled study of GAN training, solving it within our framework by taking into account the discriminator's architecture. To this end, we leverage the theory of infinite-width neural networks for the discriminator via its Neural Tangent Kernel. We characterize the trained discriminator for a wide range of losses and establish general differentiability properties of the network.

- Reverse Engineering the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2106.03186) [[code]](https://github.com/james-simon/shallow-learning)
  - James B. Simon, Sajant Anand, Michael R. DeWeese.
  - Key Word: Neural Tangent Kernel.
  - Digest: The development of methods to guide the design of neural networks is an important open challenge for deep learning theory. As a paradigm for principled neural architecture design, we propose the translation of high-performing kernels, which are better-understood and amenable to first-principles design, into equivalent network architectures, which have superior efficiency, flexibility, and feature learning. To this end, we constructively prove that, with just an appropriate choice of activation function, any positive-semidefinite dot-product kernel can be realized as either the conjugate or neural tangent kernel of a fully-connected neural network with only one hidden layer.

- Out-of-Distribution Generalization in Kernel Regression. [[paper]](https://arxiv.org/abs/2106.02261) [[code]](https://github.com/pehlevan-group/kernel-ood-generalization)
  - Abdulkadir Canatar, Blake Bordelon, Cengiz Pehlevan. *NeurIPS 2021*
  - Key Word: Out-of-Distribution Generalization; Neural Tangent Kernel.
  - Digest: We study generalization in kernel regression when the training and test distributions are different using methods from statistical physics. Using the replica method, we derive an analytical formula for the out-of-distribution generalization error applicable to any kernel and real datasets. We identify an overlap matrix that quantifies the mismatch between distributions for a given kernel as a key determinant of generalization performance under distribution shift.

- FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis. [[paper]](https://arxiv.org/abs/2105.05001)
  - Baihe Huang, Xiaoxiao Li, Zhao Song, Xin Yang. *ICML 2021*
  - Key Word: Federated Learning; Neural Tangent Kernel.
  - Digest: This paper presents a new class of convergence analysis for FL, Federated Learning Neural Tangent Kernel (FL-NTK), which corresponds to overparamterized ReLU neural networks trained by gradient descent in FL and is inspired by the analysis in Neural Tangent Kernel (NTK). Theoretically, FL-NTK converges to a global-optimal solution at a linear rate with properly tuned learning parameters. Furthermore, with proper distributional assumptions, FL-NTK can also achieve good generalization.

- Explaining Neural Scaling Laws. [[paper]](https://arxiv.org/abs/2102.06701) [[code]](https://github.com/google/neural-tangents)
  - Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma. *ICLR 2022*
  - Key Word: Scaling Laws; Neural Tangent Kernel.
  - Digest: We propose a theory that explains and connects these scaling laws. We identify variance-limited and resolution-limited scaling behavior for both dataset and model size, for a total of four scaling regimes. The variance-limited scaling follows simply from the existence of a well-behaved infinite data or infinite width limit, while the resolution-limited regime can be explained by positing that models are effectively resolving a smooth data manifold.

## Neural Tangent Kernel: 2020

- Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2010.15110)
  - Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M. Roy, Surya Ganguli. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - Digest: In suitably initialized wide networks, small learning rates transform deep neural networks (DNNs) into neural tangent kernel (NTK) machines, whose training dynamics is well-approximated by a linear weight expansion of the network at initialization. Standard training, however, diverges from its linearization in ways that are poorly understood. We study the relationship between the training dynamics of nonlinear deep networks, the geometry of the loss landscape, and the time evolution of a data-dependent NTK.

- Finite Versus Infinite Neural Networks: an Empirical Study. [[paper]](https://arxiv.org/abs/2007.15801) [[code]](https://github.com/google/neural-tangents)
  - Jaehoon Lee, Samuel S. Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, Jascha Sohl-Dickstein. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - Digest: We perform a careful, thorough, and large scale empirical study of the correspondence between wide neural networks and kernel methods. By doing so, we resolve a variety of open questions related to the study of infinitely wide neural networks. Our experimental results include: kernel methods outperform fully-connected finite-width networks, but underperform convolutional finite width networks; neural network Gaussian process (NNGP) kernels frequently outperform neural tangent (NT) kernels; centered and ensembled finite networks have reduced posterior variance and behave more similarly to infinite networks; weight decay and the use of a large learning rate break the correspondence between finite and infinite networks; the NTK parameterization outperforms the standard parameterization for finite width networks; diagonal regularization of kernels acts similarly to early stopping; floating point precision limits kernel performance beyond a critical dataset size; regularized ZCA whitening improves accuracy; finite network performance depends non-monotonically on width in ways not captured by double descent phenomena; equivariance of CNNs is only beneficial for narrow networks far from the kernel regime.

- Bayesian Deep Ensembles via the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2007.05864) [[code]](https://github.com/bobby-he/bayesian-ntk)
  - Bobby He, Balaji Lakshminarayanan, Yee Whye Teh.
  - Key Word: Neural Tangent Kernel.
  - Digest: We explore the link between deep ensembles and Gaussian processes (GPs) through the lens of the Neural Tangent Kernel (NTK): a recent development in understanding the training dynamics of wide neural networks (NNs). Previous work has shown that even in the infinite width limit, when NNs become GPs, there is no GP posterior interpretation to a deep ensemble trained with squared error loss. We introduce a simple modification to standard deep ensembles training, through addition of a computationally-tractable, randomised and untrainable function to each ensemble member, that enables a posterior interpretation in the infinite width limit.

- The Surprising Simplicity of the Early-Time Learning Dynamics of Neural Networks. [[paper]](https://arxiv.org/abs/2006.14599)
  - Wei Hu, Lechao Xiao, Ben Adlam, Jeffrey Pennington. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - Digest: We show that these common perceptions can be completely false in the early phase of learning. In particular, we formally prove that, for a class of well-behaved input distributions, the early-time learning dynamics of a two-layer fully-connected neural network can be mimicked by training a simple linear model on the inputs.

- When Do Neural Networks Outperform Kernel Methods? [[paper]](https://arxiv.org/abs/2006.13409) [[code]](https://github.com/bGhorbani/linearized_neural_networks)
  - Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, Andrea Montanari. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - Digest: How can we reconcile the above claims? For which tasks do NNs outperform RKHS? If covariates are nearly isotropic, RKHS methods suffer from the curse of dimensionality, while NNs can overcome it by learning the best low-dimensional representation. Here we show that this curse of dimensionality becomes milder if the covariates display the same low-dimensional structure as the target function, and we precisely characterize this tradeoff. Building on these results, we present the spiked covariates model that can capture in a unified framework both behaviors observed in earlier work.

- A Generalized Neural Tangent Kernel Analysis for Two-layer Neural Networks. [[paper]](https://arxiv.org/abs/2002.04026)
  - Zixiang Chen, Yuan Cao, Quanquan Gu, Tong Zhang. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel; Mean Field Theory.
  - Digest: We provide a generalized neural tangent kernel analysis and show that noisy gradient descent with weight decay can still exhibit a "kernel-like" behavior. This implies that the training loss converges linearly up to a certain accuracy. We also establish a novel generalization error bound for two-layer neural networks trained by noisy gradient descent with weight decay.

## Neural Tangent Kernel: 2019

- Simple and Effective Regularization Methods for Training on Noisily Labeled Data with Generalization Guarantee. [[paper]](https://arxiv.org/abs/1905.11368)
  - Wei Hu, Zhiyuan Li, Dingli Yu. *ICLR 2020*
  - Key Word: Neural Tangent Kernel; Regularization.
  - Digest: This paper proposes and analyzes two simple and intuitive regularization methods: (i) regularization by the distance between the network parameters to initialization, and (ii) adding a trainable auxiliary variable to the network output for each training example. Theoretically, we prove that gradient descent training with either of these two methods leads to a generalization guarantee on the clean data distribution despite being trained using noisy labels.

- Disentangling Trainability and Generalization in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1912.13053)
  - Lechao Xiao, Jeffrey Pennington, Samuel S. Schoenholz. *ICML 2020*
  - Key Word: Neural Tangent Kernel.
  - Digest: We provide such a characterization in the limit of very wide and very deep networks, for which the analysis simplifies considerably. For wide networks, the trajectory under gradient descent is governed by the Neural Tangent Kernel (NTK), and for deep networks the NTK itself maintains only weak data dependence.

- On Exact Computation with an Infinitely Wide Neural Net. [[paper]](https://arxiv.org/abs/1904.11955) [[code]](https://github.com/ruosongwang/CNTK)
  - Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, Ruosong Wang. *NeurIPS 2019*
  - Key Word: Neural Tangent Kernel.
  - Digest: The current paper gives the first efficient exact algorithm for computing the extension of NTK to convolutional neural nets, which we call Convolutional NTK (CNTK), as well as an efficient GPU implementation of this algorithm.

- Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation. [[paper]](https://arxiv.org/abs/1902.04760)
  - Greg Yang.
  - Key Word: Neural Tangent Kernel.
  - Digest: Several recent trends in machine learning theory and practice, from the design of state-of-the-art Gaussian Process to the convergence analysis of deep neural nets (DNNs) under stochastic gradient descent (SGD), have found it fruitful to study wide random neural networks. Central to these approaches are certain scaling limits of such networks. We unify these results by introducing a notion of a straightline \emph{tensor program} that can express most neural network computations, and we characterize its scaling limit when its tensors are large and randomized.

## Neural Tangent Kernel: 2018

- A Convergence Theory for Deep Learning via Over-Parameterization. [[paper]](https://arxiv.org/abs/1811.03962)
  - Zeyuan Allen-Zhu, Yuanzhi Li, Zhao Song. *ICML 2019*
  - Key Word: Stochastic Gradient Descent; Neural Tangent Kernel.
  - Digest: We prove why stochastic gradient descent (SGD) can find global minima on the training objective of DNNs in polynomial time. We only make two assumptions: the inputs are non-degenerate and the network is over-parameterized. The latter means the network width is sufficiently large: polynomial in L, the number of layers and in n, the number of samples. Our key technique is to derive that, in a sufficiently large neighborhood of the random initialization, the optimization landscape is almost-convex and semi-smooth even with ReLU activations. This implies an equivalence between over-parameterized neural networks and neural tangent kernel (NTK) in the finite (and polynomial) width setting.

- Neural Tangent Kernel: Convergence and Generalization in Neural Networks. [[paper]](https://arxiv.org/abs/1806.07572)
  - Arthur Jacot, Franck Gabriel, Clément Hongler. *NeurIPS 2018*
  - Key Word: Neural Tangent Kernel.
  - Digest: We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK).  

## Others

### Others: 2022

- Resonance in Weight Space: Covariate Shift Can Drive Divergence of SGD with Momentum. [[paper]](https://arxiv.org/abs/2203.11992)
  - Kirby Banman, Liam Peet-Pare, Nidhi Hegde, Alona Fyshe, Martha White. *ICLR 2022*
  - Key Word: Stochastic Gradient Descent; Covariate Shift.
  - Digest: We show that SGDm under covariate shift with a fixed step-size can be unstable and diverge. In particular, we show SGDm under covariate shift is a parametric oscillator, and so can suffer from a phenomenon known as resonance. We approximate the learning system as a time varying system of ordinary differential equations, and leverage existing theory to characterize the system's divergence/convergence as resonant/nonresonant modes.

- Do We Really Need a Learnable Classifier at the End of Deep Neural Network? [[paper]](https://arxiv.org/abs/2203.09081)
  - Yibo Yang, Liang Xie, Shixiang Chen, Xiangtai Li, Zhouchen Lin, Dacheng Tao.
  - Key Word: Neural Collapse.
  - Digest: We study the potential of training a network with the last-layer linear classifier randomly initialized as a simplex ETF and fixed during training. This practice enjoys theoretical merits under the layer-peeled analytical framework. We further develop a simple loss function specifically for the ETF classifier. Its advantage gets verified by both theoretical and experimental results.

- How Many Data Are Needed for Robust Learning? [[paper]](https://arxiv.org/abs/2202.11592)
  - Hongyang Zhang, Yihan Wu, Heng Huang.
  - Key Word: Robustness.
  - Digest: In this work, we study the sample complexity of robust interpolation problem when the data are in a unit ball. We show that both too many data and small data hurt robustness.

### Others: 2021

- The Equilibrium Hypothesis: Rethinking implicit regularization in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2110.11749)
  - Yizhang Lou, Chris Mingard, Soufiane Hayou.
  - Key Word: Implicit Regularization.
  - Digest: We provide the first explanation for this alignment hierarchy. We introduce and empirically validate the Equilibrium Hypothesis which states that the layers that achieve some balance between forward and backward information loss are the ones with the highest alignment to data labels.

- Implicit Sparse Regularization: The Impact of Depth and Early Stopping. [[paper]](https://arxiv.org/abs/2108.05574) [[code]](https://github.com/jiangyuan2li/implicit-sparse-regularization)
  - Jiangyuan Li, Thanh V. Nguyen, Chinmay Hegde, Raymond K. W. Wong. *NeurIPS 2021*
  - Key Word: Implicit Regularization.
  - Digest: In this paper, we study the implicit bias of gradient descent for sparse regression. We extend results on regression with quadratic parametrization, which amounts to depth-2 diagonal linear networks, to more general depth-N networks, under more realistic settings of noise and correlated designs. We show that early stopping is crucial for gradient descent to converge to a sparse model, a phenomenon that we call implicit sparse regularization. This result is in sharp contrast to known results for noiseless and uncorrelated-design cases.

- The Benefits of Implicit Regularization from SGD in Least Squares Problems. [[paper]](https://arxiv.org/abs/2108.04552)
  - Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, Dean P. Foster, Sham M. Kakade. *NeurIPS 2021*
  - Key Word: Implicit Regularization.
  Digest: We show: (1) for every problem instance and for every ridge parameter, (unregularized) SGD, when provided with logarithmically more samples than that provided to the ridge algorithm, generalizes no worse than the ridge solution (provided SGD uses a tuned constant stepsize); (2) conversely, there exist instances (in this wide problem class) where optimally-tuned ridge regression requires quadratically more samples than SGD in order to have the same generalization performance.

- Neural Controlled Differential Equations for Online Prediction Tasks. [[paper]](https://arxiv.org/abs/2106.11028) [[code]](https://github.com/jambo6/online-neural-cdes)
  - James Morrill, Patrick Kidger, Lingyi Yang, Terry Lyons.
  - Key Word: Ordinary Differential Equations.
  - Digest: Neural controlled differential equations (Neural CDEs) are state-of-the-art models for irregular time series. However, due to current implementations relying on non-causal interpolation schemes, Neural CDEs cannot currently be used in online prediction tasks; that is, in real-time as data arrives. This is in contrast to similar ODE models such as the ODE-RNN which can already operate in continuous time. Here we introduce and benchmark new interpolation schemes, most notably, rectilinear interpolation, which allows for an online everywhere causal solution to be defined.

- Differentiable Multiple Shooting Layers. [[paper]](https://arxiv.org/abs/2106.03885)
  - Stefano Massaroli, Michael Poli, Sho Sonoda, Taji Suzuki, Jinkyoo Park, Atsushi Yamashita, Hajime Asama. *NeurIPS 2021*
  - Key Word: Ordinary Differential Equations.
  - Digest: We detail a novel class of implicit neural models. Leveraging time-parallel methods for differential equations, Multiple Shooting Layers (MSLs) seek solutions of initial value problems via parallelizable root-finding algorithms. MSLs broadly serve as drop-in replacements for neural ordinary differential equations (Neural ODEs) with improved efficiency in number of function evaluations (NFEs) and wall-clock inference time.

- Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. [[paper]](https://arxiv.org/abs/2105.14368)
  - Mikhail Belkin.
  - Key Word: Interpolation; Over-parameterization.
  - Digest: In the past decade the mathematical theory of machine learning has lagged far behind the triumphs of deep neural networks on practical challenges. However, the gap between theory and practice is gradually starting to close. In this paper I will attempt to assemble some pieces of the remarkable and still incomplete mathematical mosaic emerging from the efforts to understand the foundations of deep learning. The two key themes will be interpolation, and its sibling, over-parameterization. Interpolation corresponds to fitting data, even noisy data, exactly. Over-parameterization enables interpolation and provides flexibility to select a right interpolating model.

- MALI: A memory efficient and reverse accurate integrator for Neural ODEs. [[paper]](https://arxiv.org/abs/2102.04668) [[code]](https://github.com/juntang-zhuang/TorchDiffEqPack)
  - Juntang Zhuang, Nicha C. Dvornek, Sekhar Tatikonda, James S. Duncan. *ICLR 2021*
  - Key Word: Ordinary Differential Equations.
  - Digest: Based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost \textit{w.r.t} number of solver steps in integration similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation). We validate MALI in various tasks: on image recognition tasks, to our knowledge, MALI is the first to enable feasible training of a Neural ODE on ImageNet and outperform a well-tuned ResNet, while existing methods fail due to either heavy memory burden or inaccuracy.

### Others: 2020

- Understanding the Failure Modes of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2010.15775) [[code]](https://github.com/google-research/OOD-failures)
  - Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Out-of-Distribution Generalization.
  - Digest: We identify that spurious correlations during training can induce two distinct skews in the training set, one geometric and another statistical. These skews result in two complementary ways by which empirical risk minimization (ERM) via gradient descent is guaranteed to rely on those spurious correlations.

- Sharpness-Aware Minimization for Efficiently Improving Generalization. [[paper]](https://arxiv.org/abs/2010.01412) [[code]](https://github.com/google-research/sam)
  - Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Flat Minima.
  - Digest: In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by prior work connecting the geometry of the loss landscape and generalization, we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently.

- Implicit Gradient Regularization. [[paper]](https://arxiv.org/abs/2009.11162)
  - David G.T. Barrett, Benoit Dherin. *ICLR 2021*
  - Key Word: Implicit Regularization.
  - Digest: Gradient descent can be surprisingly good at optimizing deep neural networks without overfitting and without explicit regularization. We find that the discrete steps of gradient descent implicitly regularize models by penalizing gradient descent trajectories that have large loss gradients. We call this Implicit Gradient Regularization (IGR) and we use backward error analysis to calculate the size of this regularization. We confirm empirically that implicit gradient regularization biases gradient descent toward flat minima, where test errors are small and solutions are robust to noisy parameter perturbations.

- Neural Rough Differential Equations for Long Time Series. [[paper]](https://arxiv.org/abs/2009.08295) [[code]](https://github.com/jambo6/neuralRDEs)
  - James Morrill, Cristopher Salvi, Patrick Kidger, James Foster, Terry Lyons. *ICML 2021*
  - Key Word: Ordinary Differential Equations.
  - Digest: Neural Controlled Differential Equations (Neural CDEs) are the continuous-time analogue of an RNN. However, as with RNNs, training can quickly become impractical for long time series. Here we use rough path theory to extend this formulation through application of a pre-existing mathematical tool from rough analysis - the log-ODE method - which allows us to take integration steps larger than the discretisation of the data, resulting in significantly faster training times, with retainment (and often even improvements) in model performance.

- Prevalence of Neural Collapse during the terminal phase of deep learning training. [[paper]](https://arxiv.org/abs/2008.08186) [[code]](https://github.com/neuralcollapse/neuralcollapse)
  - Vardan Papyan, X.Y. Han, David L. Donoho. *PNAS*
  - Key Word: Neural Collapse.
  - Digest: This paper studied the terminal phase of training (TPT) of today’s canonical deepnet training protocol. It documented that during TPT a process called Neural Collapse takes place, involving four fundamental and interconnected phenomena: (NC1)-(NC4).

- Neural Controlled Differential Equations for Irregular Time Series. [[paper]](https://arxiv.org/abs/2005.08926) [[code]](https://github.com/patrick-kidger/NeuralCDE)
  - Patrick Kidger, James Morrill, James Foster, Terry Lyons. *NeurIPS 2020*
  - Key Word: Ordinary Differential Equations.
  - Digest: a fundamental issue is that the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations. Here, we demonstrate how this may be resolved through the well-understood mathematics of controlled differential equations.

- Dissecting Neural ODEs. [[paper]](https://arxiv.org/abs/2002.08071) [[code]](https://github.com/DiffEqML/diffeqml-research/tree/master/dissecting-neural-odes)
  - Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, Hajime Asama. *NeurIPS 2020*
  - Key Word: Ordinary Differential Equations.
  - Digest: Continuous deep learning architectures have recently re-emerged as Neural Ordinary Differential Equations (Neural ODEs). This infinite-depth approach theoretically bridges the gap between deep learning and dynamical systems, offering a novel perspective. However, deciphering the inner working of these models is still an open challenge, as most applications apply them as generic black-box modules. In this work we "open the box", further developing the continuous-depth formulation with the aim of clarifying the influence of several design choices on the underlying dynamics.

- Proving the Lottery Ticket Hypothesis: Pruning is All You Need. [[paper]](https://arxiv.org/abs/2002.00585)
  - Eran Malach, Gilad Yehudai, Shai Shalev-Shwartz, Ohad Shamir. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - Digest: The lottery ticket hypothesis (Frankle and Carbin, 2018), states that a randomly-initialized network contains a small subnetwork such that, when trained in isolation, can compete with the performance of the original network. We prove an even stronger hypothesis (as was also conjectured in Ramanujan et al., 2019), showing that for every bounded distribution and every target network with bounded weights, a sufficiently over-parameterized neural network with random weights contains a subnetwork with roughly the same accuracy as the target network, without any further training.

### Others: 2019

- Deep Learning via Dynamical Systems: An Approximation Perspective. [[paper]](https://arxiv.org/abs/1912.10382)
  - Qianxiao Li, Ting Lin, Zuowei Shen.
  - Key Word: Approximation Theory; Controllability.
  - Digest: We build on the dynamical systems approach to deep learning, where deep residual networks are idealized as continuous-time dynamical systems, from the approximation perspective. In particular, we establish general sufficient conditions for universal approximation using continuous-time deep residual networks, which can also be understood as approximation theories in Lp using flow maps of dynamical systems.

- Why bigger is not always better: on finite and infinite neural networks. [[paper]](https://arxiv.org/abs/1910.08013)
  - Laurence Aitchison. *ICML 2020*
  - Key Word: Gradient Dynamics.
  - Digest: We give analytic results characterising the prior over representations and representation learning in finite deep linear networks. We show empirically that the representations in SOTA architectures such as ResNets trained with SGD are much closer to those suggested by our deep linear results than by the corresponding infinite network.  

- Deep Learning Theory Review: An Optimal Control and Dynamical Systems Perspective. [[paper]](https://arxiv.org/abs/1908.10920) [[code]](https://github.com/ghliu/mean-field-fcdnn)
  - Guan-Horng Liu, Evangelos A. Theodorou.
  - Key Word: Mean Field Theory.
  - Digest: We provide one possible way to align existing branches of deep learning theory through the lens of dynamical system and optimal control. By viewing deep neural networks as discrete-time nonlinear dynamical systems, we can analyze how information propagates through layers using mean field theory.

- Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks. [[paper]](https://arxiv.org/abs/1907.04595) [[code]](https://github.com/cwein3/large-lr-code)
  - Yuanzhi Li, Colin Wei, Tengyu Ma. *NeurIPS 2019*
  - Key Word: Regularization.
  - Digest: The key insight in our analysis is that the order of learning different types of patterns is crucial: because the small learning rate model first memorizes easy-to-generalize, hard-to-fit patterns, it generalizes worse on hard-to-generalize, easier-to-fit patterns than its large learning rate counterpart.  

- Are deep ResNets provably better than linear predictors? [[paper]](https://arxiv.org/abs/1907.03922)
  - Chulhee Yun, Suvrit Sra, Ali Jadbabaie. *NeurIPS 2019*
  - Key Word: ResNets; Local Minima.
  - Digest: We investigated the question whether local minima of risk function of a deep ResNet are better than linear predictors. We showed two motivating examples showing 1) the advantage of ResNets over fully-connected networks, and 2) difficulty in analysis of deep ResNets.

- Invariance-inducing regularization using worst-case transformations suffices to boost accuracy and spatial robustness. [[paper]](https://arxiv.org/abs/1906.11235)
  - Fanny Yang, Zuowen Wang, Christina Heinze-Deml. *NeurIPS 2019*
  - Key Word: Robustness; Regularization.
  - Digest: This work provides theoretical and empirical evidence that invariance-inducing regularizers can increase predictive accuracy for worst-case spatial transformations (spatial robustness). Evaluated on these adversarially transformed examples, we demonstrate that adding regularization on top of standard or adversarial training reduces the relative error by 20% for CIFAR10 without increasing the computational cost.

- Augmented Neural ODEs. [[paper]](https://arxiv.org/abs/1904.01681) [[code]](https://github.com/EmilienDupont/augmented-neural-odes)
  - Emilien Dupont, Arnaud Doucet, Yee Whye Teh. *NeurIPS 2019*
  - Key Word: Ordinary Differential Equations.
  - Digest: We show that Neural Ordinary Differential Equations (ODEs) learn representations that preserve the topology of the input space and prove that this implies the existence of functions Neural ODEs cannot represent. To address these limitations, we introduce Augmented Neural ODEs which, in addition to being more expressive models, are empirically more stable, generalize better and have a lower computational cost than Neural ODEs.

- Mean Field Analysis of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1903.04440)
  - Justin Sirignano, Konstantinos Spiliopoulos.
  - Key Word: Mean Field Theory.
  - Digest: We analyze multi-layer neural networks in the asymptotic regime of simultaneously (A) large network sizes and (B) large numbers of stochastic gradient descent training iterations. We rigorously establish the limiting behavior of the multi-layer neural network output. The limit procedure is valid for any number of hidden layers and it naturally also describes the limiting behavior of the training loss.

- A Mean Field Theory of Batch Normalization. [[paper]](https://arxiv.org/abs/1902.08129)
  - Greg Yang, Jeffrey Pennington, Vinay Rao, Jascha Sohl-Dickstein, Samuel S. Schoenholz. *ICLR 2019*
  - Key Word: Mean Field Theory.
  - Digest: We develop a mean field theory for batch normalization in fully-connected feedforward neural networks. In so doing, we provide a precise characterization of signal propagation and gradient backpropagation in wide batch-normalized networks at initialization. Our theory shows that gradient signals grow exponentially in depth and that these exploding gradients cannot be eliminated by tuning the initial weight variances or by adjusting the nonlinear activation function.

- Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. [[paper]](https://arxiv.org/abs/1902.06720) [[code]](https://github.com/google/neural-tangents)
  - Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, Jeffrey Pennington. *NeurIPS 2019*
  - Key Word: Mean Field Theory.
  - Digest: We show that for wide neural networks the learning dynamics simplify considerably and that, in the infinite width limit, they are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters. Furthermore, mirroring the correspondence between wide Bayesian neural networks and Gaussian processes, gradient-based training of wide neural networks with a squared loss produces test set predictions drawn from a Gaussian process with a particular compositional kernel.

- On Nonconvex Optimization for Machine Learning: Gradients, Stochasticity, and Saddle Points. [[paper]](https://arxiv.org/abs/1902.04811)
  - Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, Michael I. Jordan. *ICML 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - Digest: Traditional analyses of GD and SGD show that both algorithms converge to stationary points efficiently. But these analyses do not take into account the possibility of converging to saddle points. More recent theory has shown that GD and SGD can avoid saddle points, but the dependence on dimension in these analyses is polynomial. For modern machine learning, where the dimension can be in the millions, such dependence would be catastrophic. We analyze perturbed versions of GD and SGD and show that they are truly efficient---their dimension dependence is only polylogarithmic. Indeed, these algorithms converge to second-order stationary points in essentially the same time as they take to converge to classical first-order stationary points.

- Escaping Saddle Points with Adaptive Gradient Methods. [[paper]](https://arxiv.org/abs/1901.09149)
  - Matthew Staib, Sashank J. Reddi, Satyen Kale, Sanjiv Kumar, Suvrit Sra. *ICML 2019*
  - Key Word: Gradient Descent; Saddle Points.
  - Digest: We seek a crisp, clean and precise characterization of their behavior in nonconvex settings. To this end, we first provide a novel view of adaptive methods as preconditioned SGD, where the preconditioner is estimated in an online manner. By studying the preconditioner on its own, we elucidate its purpose: it rescales the stochastic gradient noise to be isotropic near stationary points, which helps escape saddle points.

### Others: 2018

- A Spline Theory of Deep Learning. [[paper]](https://proceedings.mlr.press/v80/balestriero18b.html)
  - Randall Balestriero, Richard G. Baraniuk. *ICML 2018*
  - Key Word: Approximation Theory.
  - Digest: We build a rigorous bridge between deep networks (DNs) and approximation theory via spline functions and operators. Our key result is that a large class of DNs can be written as a composition of max-affine spline operators (MASOs), which provide a powerful portal through which to view and analyze their inner workings.

- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. [[paper]](https://arxiv.org/abs/1812.05720) [[code]](https://github.com/max-andr/relu_networks_overconfident)
  - Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf. *CVPR 2019*
  - Key Wrod: ReLU; Adversarial Example.
  - Digest: We show that ReLU type neural networks which yield a piecewise linear classifier function fail in this regard as they produce almost always high confidence predictions far away from the training data. For bounded domains like images we propose a new robust optimization technique similar to adversarial training which enforces low confidence predictions far away from the training data.

- Gradient Descent Finds Global Minima of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1811.03804)
  - Simon S. Du, Jason D. Lee, Haochuan Li, Liwei Wang, Xiyu Zhai. *ICML 2019*
  - Key Word: Gradient Descent; Gradient Dynamics.
  - Digest: Gradient descent finds a global minimum in training deep neural networks despite the objective function being non-convex. The current paper proves gradient descent achieves zero training loss in polynomial time for a deep over-parameterized neural network with residual connections (ResNet). Our analysis relies on the particular structure of the Gram matrix induced by the neural network architecture. This structure allows us to show the Gram matrix is stable throughout the training process and this stability implies the global optimality of the gradient descent algorithm.

- Memorization in Overparameterized Autoencoders. [[paper]](https://arxiv.org/abs/1810.10333)
  - Adityanarayanan Radhakrishnan, Karren Yang, Mikhail Belkin, Caroline Uhler.
  - Key Word: Autoencoders; Memorization.
  - Digest: We show that overparameterized autoencoders exhibit memorization, a form of inductive bias that constrains the functions learned through the optimization process to concentrate around the training examples, although the network could in principle represent a much larger function class. In particular, we prove that single-layer fully-connected autoencoders project data onto the (nonlinear) span of the training examples.

- Information Geometry of Orthogonal Initializations and Training. [[paper]](https://arxiv.org/abs/1810.03785)
  - Piotr A. Sokol, Il Memming Park. *ICLR 2020*
  - Key Word: Mean Field Theory; Information Geometry.
  - Digest: We show a novel connection between the maximum curvature of the optimization landscape (gradient smoothness) as measured by the Fisher information matrix (FIM) and the spectral radius of the input-output Jacobian, which partially explains why more isometric networks can train much faster.

- Gradient Descent Provably Optimizes Over-parameterized Neural Networks. [[paper]](https://arxiv.org/abs/1810.02054)
  - Simon S. Du, Xiyu Zhai, Barnabas Poczos, Aarti Singh. *ICLR 2019*
  - Key Word: Gradient Descent; Gradient Dynamics.
  - Digest: One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth. This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks. For an m hidden node shallow neural network with ReLU activation and n training data, we show as long as m is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function.

- Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function. [[paper]](https://arxiv.org/abs/1809.08848)
  - Wojciech Tarnowski, Piotr Warchoł, Stanisław Jastrzębski, Jacek Tabor, Maciej A. Nowak. *AISTATS 2019*
  - Key Word: Mean Field Theory.
  - Digest: We demonstrate that in residual neural networks (ResNets) dynamical isometry is achievable irrespectively of the activation function used. We do that by deriving, with the help of Free Probability and Random Matrix Theories, a universal formula for the spectral density of the input-output Jacobian at initialization, in the large network width and depth limit.

- Mean Field Analysis of Neural Networks: A Central Limit Theorem. [[paper]](https://arxiv.org/abs/1808.09372)
  - Justin Sirignano, Konstantinos Spiliopoulos.
  - Key Word: Mean Field Theory.
  - Digest: We rigorously prove a central limit theorem for neural network models with a single hidden layer. The central limit theorem is proven in the asymptotic regime of simultaneously (A) large numbers of hidden units and (B) large numbers of stochastic gradient descent training iterations. Our result describes the neural network's fluctuations around its mean-field limit. The fluctuations have a Gaussian distribution and satisfy a stochastic partial differential equation.

- Deep Convolutional Networks as shallow Gaussian Processes. [[paper]](https://arxiv.org/abs/1808.05587) [[code]](https://github.com/convnets-as-gps/convnets-as-gps)
  - Adrià Garriga-Alonso, Carl Edward Rasmussen, Laurence Aitchison. *ICLR 2019*
  - Key Word: Gaussian Process.
  - Digest: We show that the output of a (residual) convolutional neural network (CNN) with an appropriate prior over the weights and biases is a Gaussian process (GP) in the limit of infinitely many convolutional filters, extending similar results for dense networks. For a CNN, the equivalent kernel can be computed exactly and, unlike "deep kernels", has very few parameters: only the hyperparameters of the original CNN.

- Learning Overparameterized Neural Networks via Stochastic Gradient Descent on Structured Data. [[paper]](https://arxiv.org/abs/1808.01204)
  - Yuanzhi Li, Yingyu Liang. *NeurIPS 2018*
  - Key Word: Stochastic Gradient Descent.
  - Digest: Neural networks have many successful applications, while much less theoretical understanding has been gained. Towards bridging this gap, we study the problem of learning a two-layer overparameterized ReLU neural network for multi-class classification via stochastic gradient descent (SGD) from random initialization. In the overparameterized setting, when the data comes from mixtures of well-separated distributions, we prove that SGD learns a network with a small generalization error, albeit the network has enough capacity to fit arbitrary labels.

- Neural Ordinary Differential Equations. [[paper]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq)
  - Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. *NeurIPS 2018*
  - Key Word: Ordinary Differential Equations; Normalizing Flow.
  - Digest: We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden
state using a neural network. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions. For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models.

- Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1806.05393) [[code]](https://github.com/brain-research/mean-field-cnns)
  - Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, Jeffrey Pennington. *ICML 2018*
  - Key Word: Mean Field Theory.
  - Digest: We demonstrate that it is possible to train vanilla CNNs with ten thousand layers or more simply by using an appropriate initialization scheme. We derive this initialization scheme theoretically by developing a mean field theory for signal propagation and by characterizing the conditions for dynamical isometry, the equilibration of singular values of the input-output Jacobian matrix.

- Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. [[paper]](https://arxiv.org/abs/1806.01316)
  - Ryo Karakida, Shotaro Akaho, Shun-ichi Amari. *AISTATS 2019*
  - Key Word: Mean Field Theory; Fisher Information.
  - Digest: The Fisher information matrix (FIM) is a fundamental quantity to represent the characteristics of a stochastic model, including deep neural networks (DNNs). The present study reveals novel statistics of FIM that are universal among a wide class of DNNs. To this end, we use random weights and large width limits, which enables us to utilize mean field theories. We investigate the asymptotic statistics of the FIM's eigenvalues and reveal that most of them are close to zero while the maximum eigenvalue takes a huge value.

- Understanding Generalization and Optimization Performance of Deep CNNs. [[paper]](https://arxiv.org/abs/1805.10767)
  - Pan Zhou, Jiashi Feng. *ICML 2018*
  - Key Word: Generalization of CNNs.
  - Digest: We make multiple contributions to understand deep CNNs theoretically. To our best knowledge, this work presents the first theoretical guarantees on both generalization error bound without exponential growth over network depth and optimization performance for deep CNNs.

- Towards Understanding the Role of Over-Parametrization in Generalization of Neural Networks. [[paper]](https://arxiv.org/abs/1805.12076) [[code]](https://github.com/bneyshabur/over-parametrization)
  - Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, Nathan Srebro. *ICLR 2019*
  - Key Word: Over-Parametrization.
  - Digest: We suggest a novel complexity measure based on unit-wise capacities resulting in a tighter generalization bound for two layer ReLU networks. Our capacity bound correlates with the behavior of test error with increasing network sizes (within the range reported in the experiments), and could partly explain the improvement in generalization with over-parametrization.

- Gaussian Process Behaviour in Wide Deep Neural Networks. [[paper]](https://arxiv.org/abs/1804.11271) [[code]](https://github.com/widedeepnetworks/widedeepnetworks)
  - Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani. *ICLR 2018*
  - Key Word: Gaussian Process.
  - Digest: We study the relationship between random, wide, fully connected, feedforward networks with more than one hidden layer and Gaussian processes with a recursive kernel definition. We show that, under broad conditions, as we make the architecture increasingly wide, the implied random function converges in distribution to a Gaussian process, formalising and extending existing results by Neal (1996) to deep networks.

- How to Start Training: The Effect of Initialization and Architecture. [[paper]](https://arxiv.org/abs/1803.01719)
  - Boris Hanin, David Rolnick. *NeurIPS 2018*
  - Key Word: Neuron Activation; Weight Initialization.
  - Digest: We identify and study two common failure modes for early training in deep ReLU nets. The first failure mode, exploding/vanishing mean activation length, can be avoided by initializing weights from a symmetric distribution with variance 2/fan-in and, for ResNets, by correctly weighting the residual modules. We prove that the second failure mode, exponentially large variance of activation length, never occurs in residual nets once the first failure mode is avoided.  

- The Emergence of Spectral Universality in Deep Networks. [[paper]](https://arxiv.org/abs/1802.09979)
  - Jeffrey Pennington, Samuel S. Schoenholz, Surya Ganguli. *AISTATS 2018*
  - Key Word: Mean Field Theory.
  - Digest: We leverage powerful tools from free probability theory to provide a detailed analytic understanding of how a deep network's Jacobian spectrum depends on various hyperparameters including the nonlinearity, the weight and bias distributions, and the depth. For a variety of nonlinearities, our work reveals the emergence of new universal limiting spectral distributions that remain concentrated around one even as the depth goes to infinity.

- Stronger generalization bounds for deep nets via a compression approach [[paper]](https://arxiv.org/abs/1802.05296)
  - Sanjeev Arora, Rong Ge, Behnam Neyshabur, Yi Zhang. *ICML 2018*
  - Key Word: PAC-Bayes.
  - Digest: A simple compression framework for proving generalization bounds, perhaps a more explicit and intuitive form of the PAC-Bayes work. It also yields elementary short
proofs of recent generalization results.

- Which Neural Net Architectures Give Rise To Exploding and Vanishing Gradients? [[paper]](https://arxiv.org/abs/1801.03744)
  - Boris Hanin. *NeurIPS 2018*
  - Key Word: Network Architectures.
  - Digest: We give a rigorous analysis of the statistical behavior of gradients in a randomly initialized fully connected network N with ReLU activations. Our results show that the empirical variance of the squares of the entries in the input-output Jacobian of N is exponential in a simple architecture-dependent constant beta, given by the sum of the reciprocals of the hidden layer widths.  

### Others: 2017

- Mean Field Residual Networks: On the Edge of Chaos. [[paper]](https://arxiv.org/abs/1712.08969)
  - Greg Yang, Samuel S. Schoenholz. *NeurIPS 2017*
  - Key Word: Mean Field Theory.
  - Digest: The exponential forward dynamics causes rapid collapsing of the input space geometry, while the exponential backward dynamics causes drastic vanishing or exploding gradients. We show, in contrast, that by adding skip connections, the network will, depending on the nonlinearity, adopt subexponential forward and backward dynamics, and in many cases in fact polynomial.  

- Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. [[paper]](https://arxiv.org/abs/1711.04735)
  - Jeffrey Pennington, Samuel S. Schoenholz, Surya Ganguli. *NeurIPS 2017*
  - Key Word: Mean Field Theory.
  - Digest: We explore the dependence of the singular value distribution on the depth of the network, the weight initialization, and the choice of nonlinearity. Intriguingly, we find that ReLU networks are incapable of dynamical isometry. On the other hand, sigmoidal networks can achieve isometry, but only with orthogonal weight initialization. Moreover, we demonstrate empirically that deep nonlinear networks achieving dynamical isometry learn orders of magnitude faster than networks that do not.

- Deep Neural Networks as Gaussian Processes. [[paper]](https://arxiv.org/abs/1711.00165)
  - Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha Sohl-Dickstein. *ICLR 2018*
  - Key Word: Gaussian Process.
  - Digest: In this work, we derive the exact equivalence between infinitely wide deep networks and GPs. We further develop a computationally efficient pipeline to compute the covariance function for these GPs.  

- Maximum Principle Based Algorithms for Deep Learning. [[paper]](https://arxiv.org/abs/1710.09513)
  - Qianxiao Li, Long Chen, Cheng Tai, Weinan E. *JMLR*
  - Key Word: Optimal control; Pontryagin’s Maximum Principle.
  - Digest: We discuss the viewpoint that deep residual neural networks can be viewed as discretization of a continuous-time dynamical system, and hence supervised deep learning can be regarded as solving an optimal control problem in continuous time.

- When is a Convolutional Filter Easy To Learn? [[paper]](https://arxiv.org/abs/1709.06129)
  - Simon S. Du, Jason D. Lee, Yuandong Tian. *ICLR 2018*
  - Key Word: Gradient Descent.
  - Digest: We show that (stochastic) gradient descent with random initialization can learn the convolutional filter in polynomial time and the convergence rate depends on the smoothness of the input distribution and the closeness of patches. To the best of our knowledge, this is the first recovery guarantee of gradient-based algorithms for convolutional filter on non-Gaussian input distributions.  

- Implicit Regularization in Deep Learning. [[paper]](https://arxiv.org/abs/1709.01953)
  - Behnam Neyshabur. *PhD Thesis*
  - Key Word: Implicit Regularization.
  - Digest: In an attempt to better understand generalization in deep learning, we study several possible explanations. We show that implicit regularization induced by the optimization method is playing a key role in generalization and success of deep learning models. Motivated by this view, we study how different complexity measures can ensure generalization and explain how optimization algorithms can implicitly regularize complexity measures.

- Exploring Generalization in Deep Learning. [[paper]](https://arxiv.org/abs/1706.08947) [[code]](https://github.com/bneyshabur/generalization-bounds)
  - Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, Nathan Srebro. *NeurIPS 2017*
  - Key Word: PAC-Bayes.
  - Digest: With a goal of understanding what drives generalization in deep networks, we consider several recently suggested explanations, including norm-based control, sharpness and robustness. We study how these measures can ensure generalization, highlighting the importance of scale normalization, and making a connection between sharpness and PAC-Bayes theory. We then investigate how well the measures explain different observed phenomena.

- Gradient Descent Can Take Exponential Time to Escape Saddle Points. [[paper]](https://arxiv.org/abs/1705.10412)
  - Simon S. Du, Chi Jin, Jason D. Lee, Michael I. Jordan, Barnabas Poczos, Aarti Singh. *NeurIPS 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - Digest: We established the failure of gradient descent to efficiently escape saddle points for general non-convex smooth functions. We showed that even under a very natural initialization scheme, gradient descent can require exponential time to converge to a local minimum whereas perturbed gradient descent converges in polynomial time. Our results demonstrate the necessity of adding perturbations for efficient non-convex optimization.

- How to Escape Saddle Points Efficiently. [[paper]](https://arxiv.org/abs/1703.00887)
  - Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, Michael I. Jordan. *ICML 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - Digest: This paper presents the first (nearly) dimension-free result for gradient descent in a general nonconvex setting. We present a general convergence result and show how it can be further strengthened when combined with further structure such as strict saddle conditions and/or local regularity/convexity.

### Others: 2016

- Understanding Deep Neural Networks with Rectified Linear Units. [[paper]](https://arxiv.org/abs/1611.01491)
  - Raman Arora, Amitabh Basu, Poorya Mianjy, Anirbit Mukherjee. *ICLR 2018*
  - Key Word: ReLU.
  - Digest: In this paper we investigate the family of functions representable by deep neural networks (DNN) with rectified linear units (ReLU). We give an algorithm to train a ReLU DNN with one hidden layer to *global optimality* with runtime polynomial in the data size albeit exponential in the input dimension. Further, we improve on the known lower bounds on size (from exponential to super exponential) for approximating a ReLU deep net function by a shallower ReLU net.

- Deep Information Propagation. [[paper]](https://arxiv.org/abs/1611.01232)
  - Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, Jascha Sohl-Dickstein. *ICLR 2017*
  - Key Word: Mean Field Theory.
  - Digest: We study the behavior of untrained neural networks whose weights and biases are randomly distributed using mean field theory. We show the existence of depth scales that naturally limit the maximum depth of signal propagation through these random networks. Our main practical result is to show that random networks may be trained precisely when information can travel through them. Thus, the depth scales that we identify provide bounds on how deep a network may be trained for a specific choice of hyperparameters.

- Why does deep and cheap learning work so well? [[paper]](https://arxiv.org/abs/1608.08225)
  - Henry W. Lin, Max Tegmark, David Rolnick. *Journal of Statistical Physics*
  - Key Word: Physics.
  - Digest: We show how the success of deep learning could depend not only on mathematics but also on physics: although well-known mathematical theorems guarantee that neural networks can approximate arbitrary functions well, the class of functions of practical interest can frequently be approximated through "cheap learning" with exponentially fewer parameters than generic ones. We explore how properties frequently encountered in physics such as symmetry, locality, compositionality, and polynomial log-probability translate into exceptionally simple neural networks.

- Exponential expressivity in deep neural networks through transient chaos. [[paper]](https://arxiv.org/abs/1606.05340) [[code]](https://github.com/ganguli-lab/deepchaos)
  - Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, Surya Ganguli. *NeurIPS 2016*
  - Key Word: Mean Field Theory; Riemannian Geometry.
  - Digest: We combine Riemannian geometry with the mean field theory of high dimensional chaos to study the nature of signal propagation in deep neural networks with random weights. Our results reveal a phase transition in the expressivity of random deep networks, with networks in the chaotic phase computing nonlinear functions whose global curvature grows exponentially with depth, but not with width. We prove that this generic class of random functions cannot be efficiently computed by any shallow network, going beyond prior work that restricts their analysis to single functions.
