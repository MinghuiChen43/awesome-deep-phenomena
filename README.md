# Awesome Deep Phenomena [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Our understanding of modern neural networks lags behind their practical successes. This growing gap poses a challenge to the pace of progress in machine learning because fewer pillars of knowledge are available to designers of models and algorithms [(Hanie Sedghi)](https://odsc.com/speakers/understanding-deep-learning-phenomena/). Inspired by the [ICML 2019 workshop Identifying and Understanding Deep Learning Phenomena](http://deep-phenomena.org/), I collect papers and related resources which present interesting empirical study and insight into the nature of deep learning.  

# Table of Contents

- [Empirical Study](#empirical-study)
- [Neural Collapse](#neural-collapse)
- [Deep Double Descent](#deep-double-descent)
- [Lottery Ticket Hypothesis](#lottery-ticket-hypothesis)
- [Interactions with Neuroscience](#interactions-with-neuroscience)
- [Information Bottleneck](#information-bottleneck)
- [Neural Tangent Kernel](#neural-tangent-kernel)
- [Other Papers](#others)
- [Resources](#related-resources)

## Empirical Study

### Empirical Study: 2023

- Sharpness-Aware Minimization Leads to Low-Rank Features. [[paper]](https://arxiv.org/abs/2305.16292)
  - Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion. 
  - Key Word: Sharpness-Aware Minimization; Low-Rank Features.
  - <details><summary>Digest</summary> Sharpness-aware minimization (SAM) is a method that minimizes the sharpness of the training loss of a neural network. It improves generalization and reduces the feature rank at different layers of a neural network. This low-rank effect occurs for different architectures and objectives. A significant number of activations get pruned by SAM, contributing to rank reduction. This effect can also occur in deep networks.

- Are Emergent Abilities of Large Language Models a Mirage? [[paper]](https://arxiv.org/abs/2304.15004)
  - Rylan Schaeffer, Brando Miranda, Sanmi Koyejo.
  - Key Word: Large Language Models; Neural Scaling Laws; Emergent Abilities.
  - <details><summary>Digest</summary> Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. What makes emergent abilities intriguing is two-fold: their sharpness, transitioning seemingly instantaneously from not present to present, and their unpredictability, appearing at seemingly unforeseeable model scales. Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, one can choose a metric which leads to the inference of an emergent ability or another metric which does not. We find strong supporting evidence that emergent abilities may not be a fundamental property of scaling AI models.

- A surprisingly simple technique to control the pretraining bias for better transfer: Expand or Narrow your representation. [[paper]](https://arxiv.org/abs/2304.05369)
  - Florian Bordes, Samuel Lavoie, Randall Balestriero, Nicolas Ballas, Pascal Vincent.
  - Key Word: Pretraining; Fine-Tuning; Information Bottleneck.
  - <details><summary>Digest</summary> A commonly used trick in SSL, shown to make deep networks more robust to such bias, is the addition of a small projector (usually a 2 or 3 layer multi-layer perceptron) on top of a backbone network during training. In contrast to previous work that studied the impact of the projector architecture, we here focus on a simpler, yet overlooked lever to control the information in the backbone representation. We show that merely changing its dimensionality -- by changing only the size of the backbone's very last block -- is a remarkably effective technique to mitigate the pretraining bias.

- Why is the winner the best? [[paper]](https://arxiv.org/abs/2303.17719)
  - <details><summary>Author List</summary> Matthias Eisenmann, Annika Reinke, Vivienn Weru, Minu Dietlinde Tizabi, Fabian Isensee, Tim J. Adler, Sharib Ali, Vincent Andrearczyk, Marc Aubreville, Ujjwal Baid, Spyridon Bakas, Niranjan Balu, Sophia Bano, Jorge Bernal, Sebastian Bodenstedt, Alessandro Casella, Veronika Cheplygina, Marie Daum, Marleen de Bruijne, Adrien Depeursinge, Reuben Dorent, Jan Egger, David G. Ellis, Sandy Engelhardt, Melanie Ganz, Noha Ghatwary, Gabriel Girard, Patrick Godau, Anubha Gupta, Lasse Hansen, Kanako Harada, Mattias Heinrich, Nicholas Heller, Alessa Hering, Arnaud Huaulmé, Pierre Jannin, Ali Emre Kavur, Oldřich Kodym, Michal Kozubek, Jianning Li, Hongwei Li, Jun Ma, Carlos Martín-Isla, Bjoern Menze, Alison Noble, Valentin Oreiller, Nicolas Padoy, Sarthak Pati, Kelly Payette, Tim Rädsch, Jonathan Rafael-Patiño, Vivek Singh Bawa, Stefanie Speidel, Carole H. Sudre, Kimberlin van Wijnen, Martin Wagner, Donglai Wei, Amine Yamlahi, Moi Hoon Yap, Chun Yuan, Maximilian Zenk, Aneeq Zia, David Zimmerer, Dogu Baran Aydogan, Binod Bhattarai, Louise Bloch, Raphael Brüngel, Jihoon Cho, Chanyeol Choi, Qi Dou, Ivan Ezhov, Christoph M. Friedrich, Clifton Fuller, Rebati Raman Gaire, Adrian Galdran, Álvaro García Faura, Maria Grammatikopoulou, SeulGi Hong, Mostafa Jahanifar, Ikbeom Jang, Abdolrahim Kadkhodamohammadi, Inha Kang, Florian Kofler, Satoshi Kondo, Hugo Kuijf, Mingxing Li, Minh Huan Luu, Tomaž Martinčič, Pedro Morais, Mohamed A. Naser, Bruno Oliveira, David Owen, Subeen Pang, Jinah Park, Sung-Hong Park, Szymon Płotka, Elodie Puybareau, Nasir Rajpoot, Kanghyun Ryu, Numan Saeed , Adam Shephard, Pengcheng Shi, Dejan Štepec, Ronast Subedi, Guillaume Tochon, Helena R. Torres, Helene Urien, João L. Vilaça, Kareem Abdul Wahid, Haojie Wang, Jiacheng Wang, Liansheng Wang, Xiyue Wang, Benedikt Wiestler, Marek Wodzinski, Fangfang Xia, Juanying Xie, Zhiwei Xiong, Sen Yang, Yanwu Yang, Zixuan Zhao, Klaus Maier-Hein, Paul F. Jäger, Annette Kopp-Schneider, Lena Maier-Hein.
  - Key Word: Benchmarking Competitions; Medical Imaging.
  - <details><summary>Digest</summary> The article discusses the lack of investigation into what can be learned from international benchmarking competitions for image analysis methods. The authors conducted a multi-center study of 80 competitions conducted in the scope of IEEE ISBI 2021 and MICCAI 2021 to address this gap. Based on comprehensive descriptions of the submitted algorithms and their rankings, as well as participation strategies, statistical analyses revealed common characteristics of winning solutions. These typically include the use of multi-task learning and/or multi-stage pipelines, a focus on augmentation, image preprocessing, data curation, and postprocessing.

- Sparks of Artificial General Intelligence: Early experiments with GPT-4. [[paper]](https://arxiv.org/abs/2303.12712)
  - Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang.
  - Key Word: Artificial General Intelligence; Benchmarking; GPT.
  - <details><summary>Digest</summary> We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. 

- Is forgetting less a good inductive bias for forward transfer? [[paper]](https://arxiv.org/abs/2303.08207)
  - Jiefeng Chen, Timothy Nguyen, Dilan Gorur, Arslan Chaudhry. *ICLR 2023*
  - Key Word: Continual Learning; Catastrophic Forgetting; Forward Transfer; Inductive Bias.
  - <details><summary>Digest</summary> One of the main motivations of studying continual learning is that the problem setting allows a model to accrue knowledge from past tasks to learn new tasks more efficiently. However, recent studies suggest that the key metric that continual learning algorithms optimize, reduction in catastrophic forgetting, does not correlate well with the forward transfer of knowledge. We believe that the conclusion previous works reached is due to the way they measure forward transfer. We argue that the measure of forward transfer to a task should not be affected by the restrictions placed on the continual learner in order to preserve knowledge of previous tasks. 

- Dropout Reduces Underfitting. [[paper]](https://arxiv.org/abs/2303.01500) [[code]](https://github.com/facebookresearch/dropout)
  - Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, Trevor Darrell.
  - Key Word: Dropout; Overfitting.
  - <details><summary>Digest</summary> In this study, we demonstrate that dropout can also mitigate underfitting when used at the start of training. During the early phase, we find dropout reduces the directional variance of gradients across mini-batches and helps align the mini-batch gradients with the entire dataset's gradient. This helps counteract the stochasticity of SGD and limit the influence of individual batches on model training. 

- The Role of Pre-training Data in Transfer Learning. [[paper]](https://arxiv.org/abs/2302.13464)
  - Rahim Entezari, Mitchell Wortsman, Olga Saukh, M.Moein Shariatnia, Hanie Sedghi, Ludwig Schmidt.
  - Key Word: Pre-training; Transfer Learning.
  - <details><summary>Digest</summary> We investigate the impact of pre-training data distribution on the few-shot and full fine-tuning performance using 3 pre-training methods (supervised, contrastive language-image and image-image), 7 pre-training datasets, and 9 downstream datasets. Through extensive controlled experiments, we find that the choice of the pre-training data source is essential for the few-shot transfer, but its role decreases as more data is made available for fine-tuning.

- Cliff-Learning. [[paper]](https://arxiv.org/abs/2302.07348)
  - Tony T. Wang, Igor Zablotchi, Nir Shavit, Jonathan S. Rosenfeld.
  - Key Word: Foundation Models; Fine-Tuning.
  - <details><summary>Digest</summary> We study the data-scaling of transfer learning from foundation models in the low-downstream-data regime. We observe an intriguing phenomenon which we call cliff-learning. Cliff-learning refers to regions of data-scaling laws where performance improves at a faster than power law rate (i.e. regions of concavity on a log-log scaling plot). 

- Progress measures for grokking via mechanistic interpretability. [[paper]](https://arxiv.org/abs/2301.05217)
  - Neel Nanda, Lawrence Chan, Tom Liberum, Jess Smith, Jacob Steinhardt.
  - Key Work: Grokking; Interpretability.
  - <details><summary>Digest</summary> We argue that progress measures can be found via mechanistic interpretability: reverse-engineering learned behaviors into their individual components. As a case study, we investigate the recently-discovered phenomenon of ``grokking'' exhibited by small transformers trained on modular addition tasks. 

- Grokking modular arithmetic. [[paper]](https://arxiv.org/abs/2301.02679)
  - Andrey Gromov.
  - Key Word: Grokking; Modular Addition; Interpretability.
  - <details><summary>Digest</summary> We present a simple neural network that can learn modular arithmetic tasks and exhibits a sudden jump in generalization known as ``grokking''. Concretely, we present (i) fully-connected two-layer networks that exhibit grokking on various modular arithmetic tasks under vanilla gradient descent with the MSE loss function in the absence of any regularization.

### Empirical Study: 2022

- ModelDiff: A Framework for Comparing Learning Algorithms. [[paper]](https://arxiv.org/abs/2211.12491) [[code]](https://github.com/MadryLab/modeldiff)
  - Harshay Shah, Sung Min Park, Andrew Ilyas, Aleksander Madry.
  - Key Word: Representation-based Comparison; Example-level Comparisons; Comparing Feature Attributions.
  - <details><summary>Digest</summary> We study the problem of (learning) algorithm comparison, where the goal is to find differences between models trained with two different learning algorithms. We begin by formalizing this goal as one of finding distinguishing feature transformations, i.e., input transformations that change the predictions of models trained with one learning algorithm but not the other. We then present ModelDiff, a method that leverages the datamodels framework (Ilyas et al., 2022) to compare learning algorithms based on how they use their training data.

- Overfreezing Meets Overparameterization: A Double Descent Perspective on Transfer Learning of Deep Neural Networks. [[paper]](https://arxiv.org/abs/2211.11074)
  - Yehuda Dar, Lorenzo Luzi, Richard G. Baraniuk.
  - Key Word: Transfer Learning; Deep Double Descent; Overfreezing.
  - <details><summary>Digest</summary> We study the generalization behavior of transfer learning of deep neural networks (DNNs). We adopt the overparameterization perspective -- featuring interpolation of the training data (i.e., approximately zero train error) and the double descent phenomenon -- to explain the delicate effect of the transfer learning setting on generalization performance. We study how the generalization behavior of transfer learning is affected by the dataset size in the source and target tasks, the number of transferred layers that are kept frozen in the target DNN training, and the similarity between the source and target tasks.

- How to Fine-Tune Vision Models with SGD. [[paper]](https://arxiv.org/abs/2211.09359)
  - Ananya Kumar, Ruoqi Shen, Sébastien Bubeck, Suriya Gunasekar.
  - Key Word: Fine-Tuning; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We show that fine-tuning with AdamW performs substantially better than SGD on modern Vision Transformer and ConvNeXt models. We find that large gaps in performance between SGD and AdamW occur when the fine-tuning gradients in the first "embedding" layer are much larger than in the rest of the model. Our analysis suggests an easy fix that works consistently across datasets and models: merely freezing the embedding layer (less than 1\% of the parameters) leads to SGD performing competitively with AdamW while using less memory. 

- What Images are More Memorable to Machines? [[paper]](https://arxiv.org/abs/2211.07625) [[code]](https://github.com/JunlinHan/MachineMem)
  - Junlin Han, Huangying Zhan, Jie Hong, Pengfei Fang, Hongdong Li, Lars Petersson, Ian Reid.
  - Key Word: Self-Supervised Memorization Quantification.
  - <details><summary>Digest</summary> This paper studies the problem of measuring and predicting how memorable an image is to pattern recognition machines, as a path to explore machine intelligence. Firstly, we propose a self-supervised machine memory quantification pipeline, dubbed ``MachineMem measurer'', to collect machine memorability scores of images. Similar to humans, machines also tend to memorize certain kinds of images, whereas the types of images that machines and humans memorialize are different. 

- Harmonizing the object recognition strategies of deep neural networks with humans. [[paper]](https://arxiv.org/abs/2211.04533) [[code]](https://serre-lab.github.io/Harmonization/)
  - Thomas Fel, Ivan Felipe, Drew Linsley, Thomas Serre.
  - Key Word: Interpretation; Neural Harmonizer; Psychophysics.
  - <details><summary>Digest</summary> Across 84 different DNNs trained on ImageNet and three independent datasets measuring the where and the how of human visual strategies for object recognition on those images, we find a systematic trade-off between DNN categorization accuracy and alignment with human visual strategies for object recognition. State-of-the-art DNNs are progressively becoming less aligned with humans as their accuracy improves. We rectify this growing issue with our neural harmonizer: a general-purpose training routine that both aligns DNN and human visual strategies and improves categorization accuracy. 

- Broken Neural Scaling Laws. [[paper]](https://arxiv.org/abs/2210.14891) [[code]](https://github.com/ethancaballero/broken_neural_scaling_laws)
  - Ethan Caballero, Kshitij Gupta, Irina Rish, David Krueger.
  - Key Word: Neural Scaling Laws.
  - <details><summary>Digest</summary> We present a smoothly broken power law functional form that accurately models and extrapolates the scaling behaviors of deep neural networks (i.e. how the evaluation metric of interest varies as the amount of compute used for training, number of model parameters, or training dataset size varies) for each task within a large and diverse set of upstream and downstream tasks, in zero-shot, prompted, and fine-tuned settings. This set includes large-scale vision and unsupervised language tasks, diffusion generative modeling of images, arithmetic, and reinforcement learning. 

- Pruning's Effect on Generalization Through the Lens of Training and Regularization. [[paper]](https://arxiv.org/abs/2210.13738)
  - Tian Jin, Michael Carbin, Daniel M. Roy, Jonathan Frankle, Gintare Karolina Dziugaite.
  - Key Word: Pruning; Regularization.
  - <details><summary>Digest</summary> We show that size reduction cannot fully account for the generalization-improving effect of standard pruning algorithms. Instead, we find that pruning leads to better training at specific sparsities, improving the training loss over the dense model. We find that pruning also leads to additional regularization at other sparsities, reducing the accuracy degradation due to noisy examples over the dense model. Pruning extends model training time and reduces model size. These two factors improve training and add regularization respectively. We empirically demonstrate that both factors are essential to fully explaining pruning's impact on generalization.

- How Much Data Are Augmentations Worth? An Investigation into Scaling Laws, Invariance, and Implicit Regularization. [[paper]](https://arxiv.org/abs/2210.06441) [[code]](https://github.com/JonasGeiping/dataaugs)
  - Jonas Geiping, Micah Goldblum, Gowthami Somepalli, Ravid Shwartz-Ziv, Tom Goldstein, Andrew Gordon Wilson.
  - Key Word: Data Augmentation; Neural Scaling Laws; Implicit Regularization.
  - <details><summary>Digest</summary> Despite the clear performance benefits of data augmentations, little is known about why they are so effective. In this paper, we disentangle several key mechanisms through which data augmentations operate. Establishing an exchange rate between augmented and additional real data, we find that in out-of-distribution testing scenarios, augmentations which yield samples that are diverse, but inconsistent with the data distribution can be even more valuable than additional training data.

- What does a deep neural network confidently perceive? The effective dimension of high certainty class manifolds and their low confidence boundaries. [[paper]](https://arxiv.org/abs/2210.05546) [[code]](https://github.com/stanislavfort/slice-dice-optimize/)
  - Stanislav Fort, Ekin Dogus Cubuk, Surya Ganguli, Samuel S. Schoenholz. 
  - Key Word: Class Manifold; Linear Region; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> Deep neural network classifiers partition input space into high confidence regions for each class. The geometry of these class manifolds (CMs) is widely studied and intimately related to model performance; for example, the margin depends on CM boundaries. We exploit the notions of Gaussian width and Gordon's escape theorem to tractably estimate the effective dimension of CMs and their boundaries through tomographic intersections with random affine subspaces of varying dimension. We show several connections between the dimension of CMs, generalization, and robustness.

- In What Ways Are Deep Neural Networks Invariant and How Should We Measure This? [[paper]](https://arxiv.org/abs/2210.03773) 
  - Henry Kvinge, Tegan H. Emerson, Grayson Jorgenson, Scott Vasquez, Timothy Doster, Jesse D. Lew. *NeurIPS 2022*
  - Key Word: Invariance and Equivariance.
  - <details><summary>Digest</summary> We explore the nature of invariance and equivariance of deep learning models with the goal of better understanding the ways in which they actually capture these concepts on a formal level. We introduce a family of invariance and equivariance metrics that allows us to quantify these properties in a way that disentangles them from other metrics such as loss or accuracy. 

- Omnigrok: Grokking Beyond Algorithmic Data. [[paper]](https://arxiv.org/abs/2210.01117)
  - Ziming Liu, Eric J. Michaud, Max Tegmark.
  - Key Word: Grokking Dynamics.
  - <details><summary>Digest</summary> Grokking, the unusual phenomenon for algorithmic datasets where generalization happens long after overfitting the training data, has remained elusive. We aim to understand grokking by analyzing the loss landscapes of neural networks, identifying the mismatch between training and test losses as the cause for grokking. We refer to this as the "LU mechanism" because training and test losses (against model weight norm) typically resemble "L" and "U", respectively. This simple mechanism can nicely explain many aspects of grokking: data size dependence, weight decay dependence, the emergence of representations, etc.

- Relative representations enable zero-shot latent space communication. [[paper]](https://arxiv.org/abs/2209.15430)
  - Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, Emanuele Rodolà.
  - Key Word: Representation Similarity; Model stitching.
  - <details><summary>Digest</summary> Neural networks embed the geometric structure of a data manifold lying in a high-dimensional space into latent representations. Ideally, the distribution of the data points in the latent space should depend only on the task, the data, the loss, and other architecture-specific constraints. However, factors such as the random weights initialization, training hyperparameters, or other sources of randomness in the training phase may induce incoherent latent spaces that hinder any form of reuse. Nevertheless, we empirically observe that, under the same data and modeling choices, distinct latent spaces typically differ by an unknown quasi-isometric transformation: that is, in each space, the distances between the encodings do not change. In this work, we propose to adopt pairwise similarities as an alternative data representation, that can be used to enforce the desired invariance without any additional training.

- Minimalistic Unsupervised Learning with the Sparse Manifold Transform. [[paper]](https://arxiv.org/abs/2209.15261)
  - Yubei Chen, Zeyu Yun, Yi Ma, Bruno Olshausen, Yann LeCun.
  - Key Word: Self-Supervision; Sparse Manifold Transform.
  - <details><summary>Digest</summary> We describe a minimalistic and interpretable method for unsupervised learning, without resorting to data augmentation, hyperparameter tuning, or other engineering designs, that achieves performance close to the SOTA SSL methods. Our approach leverages the sparse manifold transform, which unifies sparse coding, manifold learning, and slow feature analysis. With a one-layer deterministic sparse manifold transform, one can achieve 99.3% KNN top-1 accuracy on MNIST, 81.1% KNN top-1 accuracy on CIFAR-10 and 53.2% on CIFAR-100. 

- Revisiting Neural Scaling Laws in Language and Vision. [[paper]](https://arxiv.org/abs/2209.06640)
  - Ibrahim Alabdulmohsin, Behnam Neyshabur, Xiaohua Zhai.
  - Key Word: Neural Scaling Laws; Multi-modal Learning.
  - <details><summary>Digest</summary> The remarkable progress in deep learning in recent years is largely driven by improvements in scale, where bigger models are trained on larger datasets for longer schedules. To predict the benefit of scale empirically, we argue for a more rigorous methodology based on the extrapolation loss, instead of reporting the best-fitting (interpolating) parameters. We then present a recipe for estimating scaling law parameters reliably from learning curves. We demonstrate that it extrapolates more accurately than previous methods in a wide range of architecture families across several domains, including image classification, neural machine translation (NMT) and language modeling, in addition to tasks from the BIG-Bench evaluation benchmark. 

- A Review of Sparse Expert Models in Deep Learning. [[paper]](https://arxiv.org/abs/2209.01667)
  - William Fedus, Jeff Dean, Barret Zoph.
  - Key Word: Mixture-of-Experts.
  - <details><summary>Digest</summary> Sparse expert models are a thirty-year old concept re-emerging as a popular architecture in deep learning. This class of architecture encompasses Mixture-of-Experts, Switch Transformers, Routing Networks, BASE layers, and others, all with the unifying idea that each example is acted on by a subset of the parameters. By doing so, the degree of sparsity decouples the parameter count from the compute per example allowing for extremely large, but efficient models. The resulting models have demonstrated significant improvements across diverse domains such as natural language processing, computer vision, and speech recognition. We review the concept of sparse expert models, provide a basic description of the common algorithms, contextualize the advances in the deep learning era, and conclude by highlighting areas for future work.

- A Data-Based Perspective on Transfer Learning. [[paper]](https://arxiv.org/abs/2207.05739) [[code]](https://github.com/MadryLab/data-transfer)
  - Saachi Jain, Hadi Salman, Alaa Khaddaj, Eric Wong, Sung Min Park, Aleksander Madry.
  - Key Word: Transfer Learning; Influence Function; Data Leakage.
  - <details><summary>Digest</summary> It is commonly believed that in transfer learning including more pre-training data translates into better performance. However, recent evidence suggests that removing data from the source dataset can actually help too. In this work, we take a closer look at the role of the source dataset's composition in transfer learning and present a framework for probing its impact on downstream performance. Our framework gives rise to new capabilities such as pinpointing transfer learning brittleness as well as detecting pathologies such as data-leakage and the presence of misleading examples in the source dataset.

- How Much More Data Do I Need? Estimating Requirements for Downstream Tasks. [[paper]](https://arxiv.org/abs/2207.01725)
  - Rafid Mahmood, James Lucas, David Acuna, Daiqing Li, Jonah Philion, Jose M. Alvarez, Zhiding Yu, Sanja Fidler, Marc T. Law. *CVPR 2022*
  - Key Word: Neural Scaling Laws; Active Learning.
  - <details><summary>Digest</summary> Prior work on neural scaling laws suggest that the power-law function can fit the validation performance curve and extrapolate it to larger data set sizes. We find that this does not immediately translate to the more difficult downstream task of estimating the required data set size to meet a target performance. In this work, we consider a broad class of computer vision tasks and systematically investigate a family of functions that generalize the power-law function to allow for better estimation of data requirements.

- When Does Re-initialization Work? [[paper]](https://arxiv.org/abs/2206.10011)
  - Sheheryar Zaidi, Tudor Berariu, Hyunjik Kim, Jörg Bornschein, Claudia Clopath, Yee Whye Teh, Razvan Pascanu.
  - Key Word: Re-initialization; Regularization.
  - <details><summary>Digest</summary> We conduct an extensive empirical comparison of standard training with a selection of re-initialization methods to answer this question, training over 15,000 models on a variety of image classification benchmarks. We first establish that such methods are consistently beneficial for generalization in the absence of any other regularization. However, when deployed alongside other carefully tuned regularization techniques, re-initialization methods offer little to no added benefit for generalization, although optimal generalization performance becomes less sensitive to the choice of learning rate and weight decay hyperparameters. To investigate the impact of re-initialization methods on noisy data, we also consider learning under label noise. Surprisingly, in this case, re-initialization significantly improves upon standard training, even in the presence of other carefully tuned regularization techniques.

- How You Start Matters for Generalization. [[paper]](https://arxiv.org/abs/2206.08558)
  - Sameera Ramasinghe, Lachlan MacDonald, Moshiur Farazi, Hemanth Sartachandran, Simon Lucey.
  - Key Word: Implicit regularization; Fourier Spectrum.
  - <details><summary>Digest</summary> We promote a shift of focus towards initialization rather than neural architecture or (stochastic) gradient descent to explain this implicit regularization. Through a Fourier lens, we derive a general result for the spectral bias of neural networks and show that the generalization of neural networks is heavily tied to their initialization. Further, we empirically solidify the developed theoretical insights using practical, deep networks.

- Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? [[paper]](https://arxiv.org/abs/2202.12837) [[code]](https://github.com/Alrope123/rethinking-demonstrations)
  - Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer.
  - Key Word: Natural Language Processing; In-Context Learning.
  - <details><summary>Digest</summary> We show that ground truth demonstrations are in fact not required -- randomly replacing labels in the demonstrations barely hurts performance, consistently over 12 different models including GPT-3. Instead, we find that other aspects of the demonstrations are the key drivers of end task performance, including the fact that they provide a few examples of (1) the label space, (2) the distribution of the input text, and (3) the overall format of the sequence.

- Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. [[paper]](https://arxiv.org/abs/2201.02177) [[code]](https://github.com/openai/grok)
  - Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. 
  - Key Word: Grokking; Overfitting.
  - <details><summary>Digest</summary> In this paper we propose to study generalization of neural networks on small algorithmically generated datasets. In this setting, questions about data efficiency, memorization, generalization, and speed of learning can be studied in great detail. In some situations we show that neural networks learn through a process of "grokking" a pattern in the data, improving generalization performance from random chance level to perfect generalization, and that this improvement in generalization can happen well past the point of overfitting.

### Empirical Study: 2021

- Masked Autoencoders Are Scalable Vision Learners. [[paper]](https://arxiv.org/abs/2111.06377) [[code]](https://github.com/facebookresearch/mae)
  - Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. *CVPR 2022*
  - Key Word: Self-Supervision; Autoencoders.
  - <details><summary>Digest</summary> This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task.

- Learning in High Dimension Always Amounts to Extrapolation. [[paper]](https://arxiv.org/abs/2110.09485)
  - Randall Balestriero, Jerome Pesenti, Yann LeCun.
  - Key Word: Interpolation and Extrapolation.
  - <details><summary>Digest</summary> The notion of interpolation and extrapolation is fundamental in various fields from deep learning to function approximation. Interpolation occurs for a sample x whenever this sample falls inside or on the boundary of the given dataset's convex hull. Extrapolation occurs when x falls outside of that convex hull. One fundamental (mis)conception is that state-of-the-art algorithms work so well because of their ability to correctly interpolate training data. A second (mis)conception is that interpolation happens throughout tasks and datasets, in fact, many intuitions and theories rely on that assumption. We empirically and theoretically argue against those two points and demonstrate that on any high-dimensional (>100) dataset, interpolation almost surely never happens. 

- Understanding Dataset Difficulty with V-Usable Information. [[paper]](https://arxiv.org/abs/2110.08420) [[code]](https://github.com/kawine/dataset_difficulty)
  - Kawin Ethayarajh, Yejin Choi, Swabha Swayamdipta. *ICML 2022*
  - Key Word: Dataset Difficulty Measures; Information Theory.
  - <details><summary>Digest</summary> Estimating the difficulty of a dataset typically involves comparing state-of-the-art models to humans; the bigger the performance gap, the harder the dataset is said to be. However, this comparison provides little understanding of how difficult each instance in a given distribution is, or what attributes make the dataset difficult for a given model. To address these questions, we frame dataset difficulty -- w.r.t. a model V -- as the lack of V-usable information (Xu et al., 2019), where a lower value indicates a more difficult dataset for V. We further introduce pointwise V-information (PVI) for measuring the difficulty of individual instances w.r.t. a given distribution.

- Exploring the Limits of Large Scale Pre-training. [[paper]](https://arxiv.org/abs/2110.02095)
  - Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, Hanie Sedghi. *ICLR 2022*
  - Key Word: Pre-training.
  - <details><summary>Digest</summary> We investigate more than 4800 experiments on Vision Transformers, MLP-Mixers and ResNets with number of parameters ranging from ten million to ten billion, trained on the largest scale of available image data (JFT, ImageNet21K) and evaluated on more than 20 downstream image recognition tasks. We propose a model for downstream performance that reflects the saturation phenomena and captures the nonlinear relationship in performance of upstream and downstream tasks.

- Stochastic Training is Not Necessary for Generalization. [[paper]](https://arxiv.org/abs/2109.14119) [[code]](https://github.com/JonasGeiping/fullbatchtraining)
  - Jonas Geiping, Micah Goldblum, Phillip E. Pope, Michael Moeller, Tom Goldstein. *ICLR 2022*
  - Key Word: Stochastic Gradient Descent; Regularization.
  - <details><summary>Digest</summary> It is widely believed that the implicit regularization of SGD is fundamental to the impressive generalization behavior we observe in neural networks. In this work, we demonstrate that non-stochastic full-batch training can achieve comparably strong performance to SGD on CIFAR-10 using modern architectures. To this end, we show that the implicit regularization of SGD can be completely replaced with explicit regularization even when comparing against a strong and well-researched baseline.

- Pointer Value Retrieval: A new benchmark for understanding the limits of neural network generalization. [[paper]](https://arxiv.org/abs/2107.12580)
  - Chiyuan Zhang, Maithra Raghu, Jon Kleinberg, Samy Bengio.
  - Key Word: Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> In this paper we introduce a novel benchmark, Pointer Value Retrieval (PVR) tasks, that explore the limits of neural network generalization. We demonstrate that this task structure provides a rich testbed for understanding generalization, with our empirical study showing large variations in neural network performance based on dataset size, task complexity and model architecture.

- What can linear interpolation of neural network loss landscapes tell us? [[paper]](https://arxiv.org/abs/2106.16004)
  - Tiffany Vlaar, Jonathan Frankle. *ICML 2022*
  - Key Word: Linear Interpolation; Loss Landscapes.
  - <details><summary>Digest</summary> We put inferences of this kind to the test, systematically evaluating how linear interpolation and final performance vary when altering the data, choice of initialization, and other optimizer and architecture design choices. Further, we use linear interpolation to study the role played by individual layers and substructures of the network. We find that certain layers are more sensitive to the choice of initialization, but that the shape of the linear path is not indicative of the changes in test accuracy of the model. 

- Can Vision Transformers Learn without Natural Images? [[paper]](https://arxiv.org/abs/2103.13023) [[code]](https://hirokatsukataoka16.github.io/Vision-Transformers-without-Natural-Images/)
  - Kodai Nakashima, Hirokatsu Kataoka, Asato Matsumoto, Kenji Iwata, Nakamasa Inoue. *AAAI 2022*
  - Key Word: Formula-driven Supervised Learning; Vision Transformer.
  - <details><summary>Digest</summary> We pre-train ViT without any image collections and annotation labor. We experimentally verify that our proposed framework partially outperforms sophisticated Self-Supervised Learning (SSL) methods like SimCLRv2 and MoCov2 without using any natural images in the pre-training phase. Moreover, although the ViT pre-trained without natural images produces some different visualizations from ImageNet pre-trained ViT, it can interpret natural image datasets to a large extent. 

- The Low-Rank Simplicity Bias in Deep Networks. [[paper]](https://arxiv.org/abs/2103.10427)
  - Minyoung Huh, Hossein Mobahi, Richard Zhang, Brian Cheung, Pulkit Agrawal, Phillip Isola.
  - Key Word: Low-Rank Embedding; Inductive Bias.
  - <details><summary>Digest</summary> We make a series of empirical observations that investigate and extend the hypothesis that deeper networks are inductively biased to find solutions with lower effective rank embeddings. We conjecture that this bias exists because the volume of functions that maps to low effective rank embedding increases with depth. We show empirically that our claim holds true on finite width linear and non-linear models on practical learning paradigms and show that on natural data, these are often the solutions that generalize well. 

- Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. [[paper]](https://arxiv.org/abs/2103.00065) [[code]](https://github.com/locuslab/edge-of-stability)
  - Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar. *ICLR 2021*
  - Key Word: Edge of Stability.
  - <details><summary>Digest</summary> We empirically demonstrate that full-batch gradient descent on neural network training objectives typically operates in a regime we call the Edge of Stability. In this regime, the maximum eigenvalue of the training loss Hessian hovers just above the numerical value 2/(step size), and the training loss behaves non-monotonically over short timescales, yet consistently decreases over long timescales. Since this behavior is inconsistent with several widespread presumptions in the field of optimization, our findings raise questions as to whether these presumptions are relevant to neural network training. 

- Pre-training without Natural Images. [[paper]](https://arxiv.org/abs/2101.08515) [[code]](https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch)
  - Hirokatsu Kataoka, Kazushige Okayasu, Asato Matsumoto, Eisuke Yamagata, Ryosuke Yamada, Nakamasa Inoue, Akio Nakamura, Yutaka Satoh. *ACCV 2020*
  - Key Word: Formula-driven Supervised Learning.
  - <details><summary>Digest</summary> The paper proposes a novel concept, Formula-driven Supervised Learning. We automatically generate image patterns and their category labels by assigning fractals, which are based on a natural law existing in the background knowledge of the real world. Theoretically, the use of automatically generated images instead of natural images in the pre-training phase allows us to generate an infinite scale dataset of labeled images. Although the models pre-trained with the proposed Fractal DataBase (FractalDB), a database without natural images, does not necessarily outperform models pre-trained with human annotated datasets at all settings, we are able to partially surpass the accuracy of ImageNet/Places pre-trained models. 

### Empirical Study: 2020

- When Do Curricula Work? [[paper]](https://arxiv.org/abs/2012.03107) [[code]](https://github.com/google-research/understanding-curricula)
  - Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Curriculum Learning.
  - <details><summary>Digest</summary> We set out to investigate the relative benefits of ordered learning. We first investigate the implicit curricula resulting from architectural and optimization bias and find that samples are learned in a highly consistent order. Next, to quantify the benefit of explicit curricula, we conduct extensive experiments over thousands of orderings spanning three kinds of learning: curriculum, anti-curriculum, and random-curriculum -- in which the size of the training dataset is dynamically increased over time, but the examples are randomly ordered.

- In Search of Robust Measures of Generalization. [[paper]](https://arxiv.org/abs/2010.11924) [[code]](https://github.com/nitarshan/robust-generalization-measures)
  - Gintare Karolina Dziugaite, Alexandre Drouin, Brady Neal, Nitarshan Rajkumar, Ethan Caballero, Linbo Wang, Ioannis Mitliagkas, Daniel M. Roy. *NeurIPS 2020*
  - Key Word: Generalization Measures.
  - <details><summary>Digest</summary> One of the principal scientific challenges in deep learning is explaining generalization, i.e., why the particular way the community now trains networks to achieve small training error also leads to small error on held-out data from the same population. It is widely appreciated that some worst-case theories -- such as those based on the VC dimension of the class of predictors induced by modern neural network architectures -- are unable to explain empirical performance. A large volume of work aims to close this gap, primarily by developing bounds on generalization error, optimization error, and excess risk. When evaluated empirically, however, most of these bounds are numerically vacuous. Focusing on generalization bounds, this work addresses the question of how to evaluate such bounds empirically.

- The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers. [[paper]](https://arxiv.org/abs/2010.08127) [[code]](https://github.com/preetum/deep-bootstrap-code)
  - Preetum Nakkiran, Behnam Neyshabur, Hanie Sedghi. *ICLR 2021*
  - Key Word: Online Learning; Finite-Sample Deviations.
  - <details><summary>Digest</summary> We propose a new framework for reasoning about generalization in deep learning. The core idea is to couple the Real World, where optimizers take stochastic gradient steps on the empirical loss, to an Ideal World, where optimizers take steps on the population loss. This leads to an alternate decomposition of test error into: (1) the Ideal World test error plus (2) the gap between the two worlds. If the gap (2) is universally small, this reduces the problem of generalization in offline learning to the problem of optimization in online learning. 

- Characterising Bias in Compressed Models. [[paper]](https://arxiv.org/abs/2010.03058)
  - Sara Hooker, Nyalleng Moorosi, Gregory Clark, Samy Bengio, Emily Denton.
  - Key Word: Pruning; Fairness.
  - <details><summary>Digest</summary> The popularity and widespread use of pruning and quantization is driven by the severe resource constraints of deploying deep neural networks to environments with strict latency, memory and energy requirements. These techniques achieve high levels of compression with negligible impact on top-line metrics (top-1 and top-5 accuracy). However, overall accuracy hides disproportionately high errors on a small subset of examples; we call this subset Compression Identified Exemplars (CIE).

- Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. [[paper]](https://arxiv.org/abs/2009.10795) [[code]](https://github.com/allenai/cartography)
  - Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, Yejin Choi. *EMNLP 2020*
  - Key Word: Training Dynamics; Data Map; Curriculum Learning.
  - <details><summary>Digest</summary> Large datasets have become commonplace in NLP research. However, the increased emphasis on data quantity has made it challenging to assess the quality of data. We introduce Data Maps---a model-based tool to characterize and diagnose datasets. We leverage a largely ignored source of information: the behavior of the model on individual instances during training (training dynamics) for building data maps.

- What is being transferred in transfer learning? [[paper]](https://arxiv.org/abs/2008.11687) [[code]](https://github.com/google-research/understanding-transfer-learning)
  - Behnam Neyshabur, Hanie Sedghi, Chiyuan Zhang. *NeurIPS 2020*
  - Key Word: Transfer Learning.
  - <details><summary>Digest</summary> We provide new tools and analyses to address these fundamental questions. Through a series of analyses on transferring to block-shuffled images, we separate the effect of feature reuse from learning low-level statistics of data and show that some benefit of transfer learning comes from the latter. We present that when training from pre-trained weights, the model stays in the same basin in the loss landscape and different instances of such model are similar in feature space and close in parameter space.

- Deep Isometric Learning for Visual Recognition. [[paper]](https://arxiv.org/abs/2006.16992) [[code]](https://github.com/HaozhiQi/ISONet)
  - Haozhi Qi, Chong You, Xiaolong Wang, Yi Ma, Jitendra Malik. *ICML 2020*
  - Key Word: Isometric Networks.
  - <details><summary>Digest</summary> This paper shows that deep vanilla ConvNets without normalization nor skip connections can also be trained to achieve surprisingly good performance on standard image recognition benchmarks. This is achieved by enforcing the convolution kernels to be near isometric during initialization and training, as well as by using a variant of ReLU that is shifted towards being isometric.  

- On the Generalization Benefit of Noise in Stochastic Gradient Descent. [[paper]](https://arxiv.org/abs/2006.15081)
  - Samuel L. Smith, Erich Elsen, Soham De. *ICML 2020*
  - Key Word: Stochastic Gradient Descent.
  - <details><summary>Digest</summary> In this paper, we perform carefully designed experiments and rigorous hyperparameter sweeps on a range of popular models, which verify that small or moderately large batch sizes can substantially outperform very large batches on the test set. This occurs even when both models are trained for the same number of iterations and large batches achieve smaller training losses.  

- Do CNNs Encode Data Augmentations? [[paper]](https://arxiv.org/abs/2003.08773)
  - Eddie Yan, Yanping Huang.
  - Key Word: Data Augmentations.
  - <details><summary>Digest</summary> Surprisingly, neural network features not only predict data augmentation transformations, but they predict many transformations with high accuracy. After validating that neural networks encode features corresponding to augmentation transformations, we show that these features are primarily encoded in the early layers of modern CNNs.  

- Do We Need Zero Training Loss After Achieving Zero Training Error? [[paper]](https://arxiv.org/abs/2002.08709) [[code]](https://github.com/takashiishida/flooding)
  - Takashi Ishida, Ikko Yamane, Tomoya Sakai, Gang Niu, Masashi Sugiyama. *ICML 2020*
  - Key Word: Regularization.
  - <details><summary>Digest</summary>  Our approach makes the loss float around the flooding level by doing mini-batched gradient descent as usual but gradient ascent if the training loss is below the flooding level. This can be implemented with one line of code, and is compatible with any stochastic optimizer and other regularizers. We experimentally show that flooding improves performance and as a byproduct, induces a double descent curve of the test loss.  

- Understanding Why Neural Networks Generalize Well Through GSNR of Parameters. [[paper]](https://arxiv.org/abs/2001.07384)
  - Jinlong Liu, Guoqing Jiang, Yunzhi Bai, Ting Chen, Huayan Wang. *ICLR 2020*
  - Key Word: Generalization Indicators.
  - <details><summary>Digest</summary> In this paper, we provide a novel perspective on these issues using the gradient signal to noise ratio (GSNR) of parameters during training process of DNNs. The GSNR of a parameter is defined as the ratio between its gradient's squared mean and variance, over the data distribution.  

### Empirical Study: 2019

- Angular Visual Hardness. [[paper]](https://arxiv.org/abs/1912.02279)
  - Beidi Chen, Weiyang Liu, Zhiding Yu, Jan Kautz, Anshumali Shrivastava, Animesh Garg, Anima Anandkumar. *ICML 2020*
  - Key Word: Calibration; Example Hardness Measures.
  - <details><summary>Digest</summary> We propose a novel measure for CNN models known as Angular Visual Hardness. Our comprehensive empirical studies show that AVH can serve as an indicator of generalization abilities of neural networks, and improving SOTA accuracy entails improving accuracy on hard example

- Fantastic Generalization Measures and Where to Find Them. [[paper]](https://arxiv.org/abs/1912.02178) [[code]](https://github.com/avakanski/Evaluation-of-Complexity-Measures-for-Deep-Learning-Generalization-in-Medical-Image-Analysis)
  - Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, Samy Bengio. *ICLR 2020*
  - Key Word: Complexity Measures; Spurious Correlations.
  - <details><summary>Digest</summary> We present the first large scale study of generalization in deep networks. We investigate more then 40 complexity measures taken from both theoretical bounds and empirical studies. We train over 10,000 convolutional networks by systematically varying commonly used hyperparameters. Hoping to uncover potentially causal relationships between each measure and generalization, we analyze carefully controlled experiments and show surprising failures of some measures as well as promising measures for further research.

- Truth or Backpropaganda? An Empirical Investigation of Deep Learning Theory. [[paper]](https://arxiv.org/abs/1910.00359) [[code]](https://github.com/goldblum/TruthOrBackpropaganda)
  - Micah Goldblum, Jonas Geiping, Avi Schwarzschild, Michael Moeller, Tom Goldstein. *ICLR 2020*
  - Key Word: Local Minima.
  - <details><summary>Digest</summary> The authors take a closer look at widely held beliefs about neural networks. Using a mix of analysis and experiment, they shed some light on the ways these assumptions break down.  

- Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML. [[paper]](https://arxiv.org/abs/1909.09157) [[code]](https://github.com/fmu2/PyTorch-MAML)
  - Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals. *ICLR 2020*
  - Key Word: Meta Learning.
  - <details><summary>Digest</summary> Despite MAML's popularity, a fundamental open question remains -- is the effectiveness of MAML due to the meta-initialization being primed for rapid learning (large, efficient changes in the representations) or due to feature reuse, with the meta initialization already containing high quality features? We investigate this question, via ablation studies and analysis of the latent representations, finding that feature reuse is the dominant factor.

- Finding the Needle in the Haystack with Convolutions: on the benefits of architectural bias. [[paper]](https://arxiv.org/abs/1906.06766) [[code]](https://github.com/sdascoli/anarchitectural-search)
  - Stéphane d'Ascoli, Levent Sagun, Joan Bruna, Giulio Biroli. *NeurIPS 2019*
  - Key Word: Architectural Bias.
  - <details><summary>Digest</summary>  In particular, Convolutional Neural Networks (CNNs) are known to perform much better than Fully-Connected Networks (FCNs) on spatially structured data: the architectural structure of CNNs benefits from prior knowledge on the features of the data, for instance their translation invariance. The aim of this work is to understand this fact through the lens of dynamics in the loss landscape.  

- Adversarial Training Can Hurt Generalization. [[paper]](https://arxiv.org/abs/1906.06032)
  - Aditi Raghunathan, Sang Michael Xie, Fanny Yang, John C. Duchi, Percy Liang.
  - Key Word: Adversarial Examples.
  - <details><summary>Digest</summary> While adversarial training can improve robust accuracy (against an adversary), it sometimes hurts standard accuracy (when there is no adversary). Previous work has studied this tradeoff between standard and robust accuracy, but only in the setting where no predictor performs well on both objectives in the infinite data limit. In this paper, we show that even when the optimal predictor with infinite data performs well on both objectives, a tradeoff can still manifest itself with finite data.

- Bad Global Minima Exist and SGD Can Reach Them. [[paper]](https://arxiv.org/abs/1906.02613) [[code]](https://github.com/chao1224/BadGlobalMinima)
  - Shengchao Liu, Dimitris Papailiopoulos, Dimitris Achlioptas. *NeurIPS 2020*
  - Key Word: Stochastic Gradient Descent.
  - <details><summary>Digest</summary> Several works have aimed to explain why overparameterized neural networks generalize well when trained by Stochastic Gradient Descent (SGD). The consensus explanation that has emerged credits the randomized nature of SGD for the bias of the training process towards low-complexity models and, thus, for implicit regularization. We take a careful look at this explanation in the context of image classification with common deep neural network architectures. We find that if we do not regularize explicitly, then SGD can be easily made to converge to poorly-generalizing, high-complexity models: all it takes is to first train on a random labeling on the data, before switching to properly training with the correct labels.

- Deep ReLU Networks Have Surprisingly Few Activation Patterns. [[paper]](https://arxiv.org/abs/1906.00904)
  - Boris Hanin, David Rolnick. *NeurIPS 2019*
  - <details><summary>Digest</summary> In this paper, we show that the average number of activation patterns for ReLU networks at initialization is bounded by the total number of neurons raised to the input dimension. We show empirically that this bound, which is independent of the depth, is tight both at initialization and during training, even on memorization tasks that should maximize the number of activation patterns.  

- Sensitivity of Deep Convolutional Networks to Gabor Noise. [[paper]](https://arxiv.org/abs/1906.03455) [[code]](https://github.com/kenny-co/procedural-advml)
  - Kenneth T. Co, Luis Muñoz-González, Emil C. Lupu.
  - Key Word: Robustness.
  - <details><summary>Digest</summary> Deep Convolutional Networks (DCNs) have been shown to be sensitive to Universal Adversarial Perturbations (UAPs): input-agnostic perturbations that fool a model on large portions of a dataset. These UAPs exhibit interesting visual patterns, but this phenomena is, as yet, poorly understood. Our work shows that visually similar procedural noise patterns also act as UAPs. In particular, we demonstrate that different DCN architectures are sensitive to Gabor noise patterns. This behaviour, its causes, and implications deserve further in-depth study.

- Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1905.05928)
  - Guangyong Chen, Pengfei Chen, Yujun Shi, Chang-Yu Hsieh, Benben Liao, Shengyu Zhang.
  - Key Word: Batch Normalization; Dropout.
  - <details><summary>Digest</summary> Our work is based on an excellent idea that whitening the inputs of neural networks can achieve a fast convergence speed. Given the well-known fact that independent components must be whitened, we introduce a novel Independent-Component (IC) layer before each weight layer, whose inputs would be made more independent.  

- A critical analysis of self-supervision, or what we can learn from a single image. [[paper]](https://arxiv.org/abs/1904.13132) [[code]](https://github.com/yukimasano/linear-probes)
  - Yuki M. Asano, Christian Rupprecht, Andrea Vedaldi. *ICLR 2020*
  - Key Word: Self-Supervision.
  - <details><summary>Digest</summary> We show that three different and representative methods, BiGAN, RotNet and DeepCluster, can learn the first few layers of a convolutional network from a single image as well as using millions of images and manual labels, provided that strong data augmentation is used. However, for deeper layers the gap with manual supervision cannot be closed even if millions of unlabelled images are used for training.  

- Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet. [[paper]](https://arxiv.org/abs/1904.00760) [[code]](https://github.com/wielandbrendel/bag-of-local-features-models)
  - Wieland Brendel, Matthias Bethge. *ICLR 2019*
  - Key Word: Bag-of-Features.
  - <details><summary>Digest</summary> Our model, a simple variant of the ResNet-50 architecture called BagNet, classifies an image based on the occurrences of small local image features without taking into account their spatial ordering. This strategy is closely related to the bag-of-feature (BoF) models popular before the onset of deep learning and reaches a surprisingly high accuracy on ImageNet.  

- Transfusion: Understanding Transfer Learning for Medical Imaging. [[paper]](https://arxiv.org/abs/1902.07208) [[code]](https://github.com/PasqualeZingo/TransfusionReproducibilityChallenge)
  - Maithra Raghu, Chiyuan Zhang, Jon Kleinberg, Samy Bengio. *NeurIPS 2019*
  - Key Word: Transfer Learning; Medical Imaging.
  - <details><summary>Digest</summary> we explore properties of transfer learning for medical imaging. A performance evaluation on two large scale medical imaging tasks shows that surprisingly, transfer offers little benefit to performance, and simple, lightweight models can perform comparably to ImageNet architectures.

- Identity Crisis: Memorization and Generalization under Extreme Overparameterization. [[paper]](https://arxiv.org/abs/1902.04698)
  - Chiyuan Zhang, Samy Bengio, Moritz Hardt, Michael C. Mozer, Yoram Singer. *ICLR 2020*
  - Key Word: Memorization.
  - <details><summary>Digest</summary> We study the interplay between memorization and generalization of overparameterized networks in the extreme case of a single training example and an identity-mapping task.  

- Are All Layers Created Equal? [[paper]](https://arxiv.org/abs/1902.01996)  
  - Chiyuan Zhang, Samy Bengio, Yoram Singer. *JMLR*
  - Key Word: Robustness.
  - <details><summary>Digest</summary> We show that the layers can be categorized as either "ambient" or "critical". Resetting the ambient layers to their initial values has no negative consequence, and in many cases they barely change throughout training. On the contrary, resetting the critical layers completely destroys the predictor and the performance drops to chance.  

### Empirical Study: 2018

- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. [[paper]](https://arxiv.org/abs/1812.05720) [[code]](https://github.com/max-andr/relu_networks_overconfident)
  - Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf. *CVPR 2019*
  - Key Word: ReLU.
  - <details><summary>Digest</summary> Classifiers used in the wild, in particular for safety-critical systems, should not only have good generalization properties but also should know when they don't know, in particular make low confidence predictions far away from the training data. We show that ReLU type neural networks which yield a piecewise linear classifier function fail in this regard as they produce almost always high confidence predictions far away from the training data.  

- An Empirical Study of Example Forgetting during Deep Neural Network Learning. [[paper]](https://arxiv.org/abs/1812.05159) [[code]](https://github.com/mtoneva/example_forgetting)
  - Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, Geoffrey J. Gordon. *ICLR 2019*
  - Key Word: Curriculum Learning; Sample Weighting; Example Forgetting.
  - <details><summary>Digest</summary> We define a 'forgetting event' to have occurred when an individual training example transitions from being classified correctly to incorrectly over the course of learning. Across several benchmark data sets, we find that: (i) certain examples are forgotten with high frequency, and some not at all; (ii) a data set's (un)forgettable examples generalize across neural architectures; and (iii) based on forgetting dynamics, a significant fraction of examples can be omitted from the training data set while still maintaining state-of-the-art generalization performance.

- On Implicit Filter Level Sparsity in Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1811.12495)
  - Dushyant Mehta, Kwang In Kim, Christian Theobalt. *CVPR 2019*
  - Key Word: Regularization; Sparsification.
  - <details><summary>Digest</summary> We investigate filter level sparsity that emerges in convolutional neural networks (CNNs) which employ Batch Normalization and ReLU activation, and are trained with adaptive gradient descent techniques and L2 regularization or weight decay. We conduct an extensive experimental study casting our initial findings into hypotheses and conclusions about the mechanisms underlying the emergent filter level sparsity. This study allows new insight into the performance gap obeserved between adapative and non-adaptive gradient descent methods in practice.

- Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. [[paper]](https://arxiv.org/abs/1811.12359) [[code]](https://github.com/google-research/disentanglement_lib)
  - Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem. *ICML 2019*
  - Key Word: Disentanglement.
  - <details><summary>Digest</summary> Our results suggest that future work on disentanglement learning should be explicit about the role of inductive biases and (implicit) supervision, investigate concrete benefits of enforcing disentanglement of the learned representations, and consider a reproducible experimental setup covering several data sets.  

- Insights on representational similarity in neural networks with canonical correlation. [[paper]](https://arxiv.org/abs/1806.05759) [[code]](https://github.com/google/svcca)
  - Ari S. Morcos, Maithra Raghu, Samy Bengio. *NeurIPS 2018*
  - Key Word: Representational Similarity.
  - <details><summary>Digest</summary> Comparing representations in neural networks is fundamentally difficult as the structure of representations varies greatly, even across groups of networks trained on identical tasks, and over the course of training. Here, we develop projection weighted CCA (Canonical Correlation Analysis) as a tool for understanding neural networks, building off of SVCCA.

- Layer rotation: a surprisingly powerful indicator of generalization in deep networks? [[paper]](https://arxiv.org/abs/1806.01603) [[code]](https://github.com/ispgroupucl/layer-rotation-paper-experiments)
  - Simon Carbonnelle, Christophe De Vleeschouwer.
  - Key Word: Weight Evolution.
  - <details><summary>Digest</summary> Our work presents extensive empirical evidence that layer rotation, i.e. the evolution across training of the cosine distance between each layer's weight vector and its initialization, constitutes an impressively consistent indicator of generalization performance. In particular, larger cosine distances between final and initial weights of each layer consistently translate into better generalization performance of the final model.

- Sensitivity and Generalization in Neural Networks: an Empirical Study. [[paper]](https://arxiv.org/abs/1802.08760)
  - Roman Novak, Yasaman Bahri, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein. *ICLR 2018*
  - Key Word: Sensitivity.
  - <details><summary>Digest</summary> In this work, we investigate this tension between complexity and generalization through an extensive empirical exploration of two natural metrics of complexity related to sensitivity to input perturbations. We find that trained neural networks are more robust to input perturbations in the vicinity of the training data manifold, as measured by the norm of the input-output Jacobian of the network, and that it correlates well with generalization.  

### Empirical Study: 2017

- Deep Image Prior. [[paper]](https://arxiv.org/abs/1711.10925) [[code]](https://dmitryulyanov.github.io/deep_image_prior)
  - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky.
  - Key Word: Low-Level Vision.
  - <details><summary>Digest</summary> In this paper, we show that, on the contrary, the structure of a generator network is sufficient to capture a great deal of low-level image statistics prior to any learning. In order to do so, we show that a randomly-initialized neural network can be used as a handcrafted prior with excellent results in standard inverse problems such as denoising, super-resolution, and inpainting.  

- Critical Learning Periods in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1711.08856)
  - Alessandro Achille, Matteo Rovere, Stefano Soatto. *ICLR 2019*
  - Key Word: Memorization.
  - <details><summary>Digest</summary> Our findings indicate that the early transient is critical in determining the final solution of the optimization associated with training an artificial neural network. In particular, the effects of sensory deficits during a critical period cannot be overcome, no matter how much additional training is performed.  

- A Closer Look at Memorization in Deep Networks. [[paper]](https://arxiv.org/abs/1706.05394)
  - Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, Simon Lacoste-Julien. *ICML 2017*
  - Key Word: Memorization.
  - <details><summary>Digest</summary> In our experiments, we expose qualitative differences in gradient-based optimization of deep neural networks (DNNs) on noise vs. real data. We also demonstrate that for appropriately tuned explicit regularization (e.g., dropout) we can degrade DNN training performance on noise datasets without compromising generalization on real data.  

### Empirical Study: 2016

- Understanding deep learning requires rethinking generalization. [[paper]](https://arxiv.org/abs/1611.03530)
  - Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. *ICLR 2017*
  - Key Word: Memorization.
  - <details><summary>Digest</summary> Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice. Specifically, our experiments establish that state-of-the-art convolutional networks for image classification trained with stochastic gradient methods easily fit a random labeling of the training data.  

## Neural Collapse

### Neural Collapse: 2023

- Neural (Tangent Kernel) Collapse. [[paper]](https://arxiv.org/abs/2305.16427)
  - Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou.
  - Key Word: Neural Collapse; Neural Tangent Kernel.
  - <details><summary>Digest</summary> This paper investigates how the Neural Tangent Kernel (NTK), which tracks how deep neural networks (DNNs) change during training, and the Neural Collapse (NC) phenomenon, which refers to the symmetry and structure in the last-layer features of trained classification DNNs, are related. They assume that the empirical NTK has a block structure that matches the class labels, meaning that samples of the same class are more correlated than samples of different classes. They show how this assumption leads to the dynamics of DNNs trained with mean squared (MSE) loss and the emergence of NC in DNNs with block-structured NTK. They support their theory with large-scale experiments on three DNN architectures and three datasets.

- Neural Collapse Inspired Feature-Classifier Alignment for Few-Shot Class Incremental Learning. [[paper]](https://arxiv.org/abs/2302.03004) [[code]](https://github.com/NeuralCollapseApplications/FSCIL)
  - Yibo Yang, Haobo Yuan, Xiangtai Li, Zhouchen Lin, Philip Torr, Dacheng Tao. *ICLR 2023*
  - Key Word: Few-Shot Class Incremental Learning; Neural Collapse.
  - <details><summary>Digest</summary> We deal with this misalignment dilemma in FSCIL inspired by the recently discovered phenomenon named neural collapse, which reveals that the last-layer features of the same class will collapse into a vertex, and the vertices of all classes are aligned with the classifier prototypes, which are formed as a simplex equiangular tight frame (ETF). It corresponds to an optimal geometric structure for classification due to the maximized Fisher Discriminant Ratio. 

- Neural Collapse in Deep Linear Network: From Balanced to Imbalanced Data. [[paper]](https://arxiv.org/abs/2301.00437)
  - Hien Dang, Tan Nguyen, Tho Tran, Hung Tran, Nhat Ho.
  - Key Word: Neural Collapse; Imbalanced Learning.
  - <details><summary>Digest</summary> We take a step further and prove the Neural Collapse occurrence for deep linear network for the popular mean squared error (MSE) and cross entropy (CE) loss. Furthermore, we extend our research to imbalanced data for MSE loss and present the first geometric analysis for Neural Collapse under this setting.

### Neural Collapse: 2022

- Principled and Efficient Transfer Learning of Deep Models via Neural Collapse. [[paper]](https://arxiv.org/abs/2212.12206)
  - Xiao Li, Sheng Liu, Jinxin Zhou, Xinyu Lu, Carlos Fernandez-Granda, Zhihui Zhu, Qing Qu.
  - Key Word: Neural Collapse; Transfer Learning.
  - <details><summary>Digest</summary> This work delves into the mystery of transfer learning through an intriguing phenomenon termed neural collapse (NC), where the last-layer features and classifiers of learned deep networks satisfy: (i) the within-class variability of the features collapses to zero, and (ii) the between-class feature means are maximally and equally separated. Through the lens of NC, our findings for transfer learning are the following: (i) when pre-training models, preventing intra-class variability collapse (to a certain extent) better preserves the intrinsic structures of the input data, so that it leads to better model transferability; (ii) when fine-tuning models on downstream tasks, obtaining features with more NC on downstream data results in better test accuracy on the given task. 

- Perturbation Analysis of Neural Collapse. [[paper]](https://arxiv.org/abs/2210.16658)
  - Tom Tirer, Haoxiang Huang, Jonathan Niles-Weed.
  - Key Word: Neural Collapse.
  - <details><summary>Digest</summary> We propose a richer model that can capture this phenomenon by forcing the features to stay in the vicinity of a predefined features matrix (e.g., intermediate features). We explore the model in the small vicinity case via perturbation analysis and establish results that cannot be obtained by the previously studied models. 

- Grokking phase transitions in learning local rules with gradient descent. [[paper]](https://arxiv.org/abs/2210.15435)
  - Bojan Žunkovič, Enej Ilievski.
  - Key Word: Tensor Network; Grokking; Many-Body Quantum Mechanics; Neural Collapse.
  - <details><summary>Digest</summary> We discuss two solvable grokking (generalisation beyond overfitting) models in a rule learning scenario. We show that grokking is a phase transition and find exact analytic expressions for the critical exponents, grokking probability, and grokking time distribution. Further, we introduce a tensor-network map that connects the proposed grokking setup with the standard (perceptron) statistical learning theory and show that grokking is a consequence of the locality of the teacher model. As an example, we analyse the cellular automata learning task, numerically determine the critical exponent and the grokking time distributions and compare them with the prediction of the proposed grokking model. Finally, we numerically analyse the connection between structure formation and grokking.

- Imbalance Trouble: Revisiting Neural-Collapse Geometry. [[paper]](https://arxiv.org/abs/2208.05512)
  - Christos Thrampoulidis, Ganesh R. Kini, Vala Vakilian, Tina Behnia.
  - Key Word: Neural Collapse; Class Imbalance.
  - <details><summary>Digest</summary> Neural Collapse refers to the remarkable structural properties characterizing the geometry of class embeddings and classifier weights, found by deep nets when trained beyond zero training error. However, this characterization only holds for balanced data. Here we thus ask whether it can be made invariant to class imbalances. Towards this end, we adopt the unconstrained-features model (UFM), a recent theoretical model for studying neural collapse, and introduce Simplex-Encoded-Labels Interpolation (SELI) as an invariant characterization of the neural collapse phenomenon. 

- Neural Collapse: A Review on Modelling Principles and Generalization. [[paper]](https://arxiv.org/abs/2206.04041)
  - Vignesh Kothapalli, Ebrahim Rasromani, Vasudev Awatramani.
  - Key Word: Neural Collapse.
  - <details><summary>Digest</summary> We analyse the principles which aid in modelling such a phenomena from the ground up and show how they can build a common understanding of the recently proposed models that try to explain NC. We hope that our analysis presents a multifaceted perspective on modelling NC and aids in forming connections with the generalization capabilities of neural networks. Finally, we conclude by discussing the avenues for further research and propose potential research problems.

- Do We Really Need a Learnable Classifier at the End of Deep Neural Network? [[paper]](https://arxiv.org/abs/2203.09081)
  - Yibo Yang, Liang Xie, Shixiang Chen, Xiangtai Li, Zhouchen Lin, Dacheng Tao.
  - Key Word: Neural Collapse.
  - <details><summary>Digest</summary> We study the potential of training a network with the last-layer linear classifier randomly initialized as a simplex ETF and fixed during training. This practice enjoys theoretical merits under the layer-peeled analytical framework. We further develop a simple loss function specifically for the ETF classifier. Its advantage gets verified by both theoretical and experimental results.

- Limitations of Neural Collapse for Understanding Generalization in Deep Learning. [[paper]](https://arxiv.org/abs/2202.08384)
  - Like Hui, Mikhail Belkin, Preetum Nakkiran.
  - Key Word: Neural Collapse.
  - <details><summary>Digest</summary> We point out that Neural Collapse is primarily an optimization phenomenon, not a generalization one, by investigating the train collapse and test collapse on various dataset and architecture combinations. We propose more precise definitions — "strong" and "weak" Neural Collapse for both the train set and the test set — and discuss their theoretical feasibility.

### Neural Collapse: 2021

- On the Role of Neural Collapse in Transfer Learning. [[paper]](https://arxiv.org/abs/2112.15121)
  - Tomer Galanti, András György, Marcus Hutter. *ICLR 2022*
  - Key Word: Neural Collapse; Transfer Learning.
  - <details><summary>Digest</summary> We provide an explanation for this behavior based on the recently observed phenomenon that the features learned by overparameterized classification networks show an interesting clustering property, called neural collapse.

- An Unconstrained Layer-Peeled Perspective on Neural Collapse. [[paper]](https://arxiv.org/abs/2110.02796)
  - Wenlong Ji, Yiping Lu, Yiliang Zhang, Zhun Deng, Weijie J. Su. *ICLR 2022*
  - Key Word: Neural Collapse; Uncostrained Model; Implicit Regularization.
  - <details><summary>Digest</summary> We introduce a surrogate model called the unconstrained layer-peeled model (ULPM). We prove that gradient flow on this model converges to critical points of a minimum-norm separation problem exhibiting neural collapse in its global minimizer. Moreover, we show that the ULPM with the cross-entropy loss has a benign global landscape for its loss function, which allows us to prove that all the critical points are strict saddle points except the global minimizers that exhibit the neural collapse phenomenon.

- Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path. [[paper]](https://arxiv.org/abs/2106.02073)
  - X.Y. Han, Vardan Papyan, David L. Donoho. *ICLR 2022*
  - Key Word: Neural Collapse; Gradient Flow.
  - <details><summary>Digest</summary> The analytically-tractable MSE loss offers more mathematical opportunities than the hard-to-analyze CE loss, inspiring us to leverage MSE loss towards the theoretical investigation of NC. We develop three main contributions: (I) We show a new decomposition of the MSE loss into (A) terms directly interpretable through the lens of NC and which assume the last-layer classifier is exactly the least-squares classifier; and (B) a term capturing the deviation from this least-squares classifier. (II) We exhibit experiments on canonical datasets and networks demonstrating that term-(B) is negligible during training. This motivates us to introduce a new theoretical construct: the central path, where the linear classifier stays MSE-optimal for feature activations throughout the dynamics. (III) By studying renormalized gradient flow along the central path, we derive exact dynamics that predict NC.

- A Geometric Analysis of Neural Collapse with Unconstrained Features. [[paper]](https://arxiv.org/abs/2105.02375) [[code]](https://github.com/tding1/Neural-Collapse)
  - Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, Qing Qu. *NeurIPS 2021*
  - Key Word: Neural Collapse, Nonconvex Optimization.
  - <details><summary>Digest</summary> We provide the first global optimization landscape analysis of Neural Collapse -- an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. As recently reported by Papyan et al., this phenomenon implies that (i) the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling, and (ii) cross-example within-class variability of last-layer activations collapses to zero. We study the problem based on a simplified unconstrained feature model, which isolates the topmost layers from the classifier of the neural network.

- Exploring Deep Neural Networks via Layer-Peeled Model: Minority Collapse in Imbalanced Training. [[paper]](https://arxiv.org/abs/2101.12699) [[code]](https://github.com/HornHehhf/LPM)
  - Cong Fang, Hangfeng He, Qi Long, Weijie J. Su. *PNAS*
  - Key Word: Neural Collapse; Imbalanced Training.
  - <details><summary>Digest</summary> In this paper, we introduce the Layer-Peeled Model, a nonconvex yet analytically tractable optimization program, in a quest to better understand deep neural networks that are trained for a sufficiently long time. As the name suggests, this new model is derived by isolating the topmost layer from the remainder of the neural network, followed by imposing certain constraints separately on the two parts of the network. When moving to the imbalanced case, our analysis of the Layer-Peeled Model reveals a hitherto unknown phenomenon that we term Minority Collapse, which fundamentally limits the performance of deep learning models on the minority classes.

### Neural Collapse: 2020

- Prevalence of Neural Collapse during the terminal phase of deep learning training. [[paper]](https://arxiv.org/abs/2008.08186) [[code]](https://github.com/neuralcollapse/neuralcollapse)
  - Vardan Papyan, X.Y. Han, David L. Donoho. *PNAS*
  - Key Word: Neural Collapse.
  - <details><summary>Digest</summary> This paper studied the terminal phase of training (TPT) of today’s canonical deepnet training protocol. It documented that during TPT a process called Neural Collapse takes place, involving four fundamental and interconnected phenomena: (NC1)-(NC4).

## Deep Double Descent

### Deep Double Descent: 2023

- Dropout Drops Double Descent. [[paper]](https://arxiv.org/abs/2305.16179)
  - Tian-Le Yang, Joe Suzuki.
  - Key Word: Dropout; Deep Double Descent.
  - <details><summary>Digest</summary> The paper finds that adding a dropout layer before the fully-connected linear layer can drop the double descent phenomenon. Double descent is when the prediction error rises and drops as sample or model size increases. Optimal dropout can alleviate this in linear and nonlinear regression models, both theoretically and empirically. Optimal dropout can achieve a monotonic test error curve in nonlinear neural networks. Previous deep learning models do not encounter double-descent because they already apply regularization approaches like dropout.

- Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle. [[paper]](https://arxiv.org/abs/2303.14151)
  - Rylan Schaeffer, Mikail Khona, Zachary Robertson, Akhilan Boopathy, Kateryna Pistunova, Jason W. Rocks, Ila Rani Fiete, Oluwasanmi Koyejo.
  - Key Word: Deep Double Descent; Tutorial.
  - <details><summary>Digest</summary> We briefly describe double descent, then provide an explanation of why double descent occurs in an informal and approachable manner, requiring only familiarity with linear algebra and introductory probability. We provide visual intuition using polynomial regression, then mathematically analyze double descent with ordinary linear regression and identify three interpretable factors that, when simultaneously all present, together create double descent.

- Unifying Grokking and Double Descent. [[paper]](https://arxiv.org/abs/2303.06173) [[code]](https://github.com/xanderdavies/unifying-grok-dd)
  - Xander Davies, Lauro Langosco, David Krueger.
  - Key Word: Deep Double Descent; Grokking.
  - <details><summary>Digest</summary> We hypothesize that grokking and double descent can be understood as instances of the same learning dynamics within a framework of pattern learning speeds. We propose that this framework also applies when varying model capacity instead of optimization steps, and provide the first demonstration of model-wise grokking.

### Deep Double Descent: 2022

- Sparse Double Descent: Where Network Pruning Aggravates Overfitting. [[paper]](https://arxiv.org/abs/2206.08684) [[code]](https://github.com/hezheug/sparse-double-descent)
  - Zheng He, Zeke Xie, Quanzhi Zhu, Zengchang Qin. *ICML 2022*
  - Key Word: Deep Double Descent; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> While recent studies focused on the deep double descent with respect to model overparameterization, they failed to recognize that sparsity may also cause double descent. In this paper, we have three main contributions. First, we report the novel sparse double descent phenomenon through extensive experiments. Second, for this phenomenon, we propose a novel learning distance interpretation that the curve of ℓ2 learning distance of sparse models (from initialized parameters to final parameters) may correlate with the sparse double descent curve well and reflect generalization better than minima flatness. Third, in the context of sparse double descent, a winning ticket in the lottery ticket hypothesis surprisingly may not always win.

- Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective. [[paper]](https://arxiv.org/abs/2203.08124) [[code]](https://github.com/somepago/dbviz)
  - Gowthami Somepalli, Liam Fowl, Arpit Bansal, Ping Yeh-Chiang, Yehuda Dar, Richard Baraniuk, Micah Goldblum, Tom Goldstein. *CVPR 2022*
  - Key Word: Deep Double Descent; Manifold.
  - <details><summary>Digest</summary> We discuss methods for visualizing neural network decision boundaries and decision regions. We use these visualizations to investigate issues related to reproducibility and generalization in neural network training. We observe that changes in model architecture (and its associate inductive bias) cause visible changes in decision boundaries, while multiple runs with the same architecture yield results with strong similarities, especially in the case of wide architectures. We also use decision boundary methods to visualize double descent phenomena.

- Phenomenology of Double Descent in Finite-Width Neural Networks. [[paper]](https://arxiv.org/abs/2203.07337) [[code]](https://github.com/sidak/double-descent)
  - Sidak Pal Singh, Aurelien Lucchi, Thomas Hofmann, Bernhard Schölkopf. *ICLR 2022*
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> 'Double descent' delineates the generalization behaviour of models depending on the regime they belong to: under- or over-parameterized. The current theoretical understanding behind the occurrence of this phenomenon is primarily based on linear and kernel regression models -- with informal parallels to neural networks via the Neural Tangent Kernel. Therefore such analyses do not adequately capture the mechanisms behind double descent in finite-width neural networks, as well as, disregard crucial components -- such as the choice of the loss function. We address these shortcomings by leveraging influence functions in order to derive suitable expressions of the population loss and its lower bound, while imposing minimal assumptions on the form of the parametric model.

### Deep Double Descent: 2021

- Multi-scale Feature Learning Dynamics: Insights for Double Descent. [[paper]](https://arxiv.org/abs/2112.03215) [[code]](https://github.com/nndoubledescent/doubledescent)
  - Mohammad Pezeshki, Amartya Mitra, Yoshua Bengio, Guillaume Lajoie.
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> We investigate the origins of the less studied epoch-wise double descent in which the test error undergoes two non-monotonous transitions, or descents as the training time increases. By leveraging tools from statistical physics, we study a linear teacher-student setup exhibiting epoch-wise double descent similar to that in deep neural networks. In this setting, we derive closed-form analytical expressions for the evolution of generalization error over training. We find that double descent can be attributed to distinct features being learned at different scales: as fast-learning features overfit, slower-learning features start to fit, resulting in a second descent in test error.

- Asymptotic Risk of Overparameterized Likelihood Models: Double Descent Theory for Deep Neural Networks. [[paper]](https://arxiv.org/abs/2103.00500)
  - Ryumei Nakada, Masaaki Imaizumi.
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> We consider a likelihood maximization problem without the model constraints and analyze the upper bound of an asymptotic risk of an estimator with penalization. Technically, we combine a property of the Fisher information matrix with an extended Marchenko-Pastur law and associate the combination with empirical process techniques. The derived bound is general, as it describes both the double descent and the regularized risk curves, depending on the penalization.

- Distilling Double Descent. [[paper]](https://arxiv.org/abs/2102.06849)
  - Andrew Cotter, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, Sashank J. Reddi, Yichen Zhou.
  - Key Word: Deep Double Descent; Distillation.
  - <details><summary>Digest</summary> Distillation is the technique of training a "student" model based on examples that are labeled by a separate "teacher" model, which itself is trained on a labeled dataset. The most common explanations for why distillation "works" are predicated on the assumption that student is provided with soft labels, e.g. probabilities or confidences, from the teacher model. In this work, we show, that, even when the teacher model is highly overparameterized, and provides hard labels, using a very large held-out unlabeled dataset to train the student model can result in a model that outperforms more "traditional" approaches.

### Deep Double Descent: 2020

- Understanding Double Descent Requires a Fine-Grained Bias-Variance Decomposition. [[paper]](https://arxiv.org/abs/2011.03321)
  - Ben Adlam, Jeffrey Pennington. *NeurIPS 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - <details><summary>Digest</summary> Classical learning theory suggests that the optimal generalization performance of a machine learning model should occur at an intermediate model complexity, with simpler models exhibiting high bias and more complex models exhibiting high variance of the predictive function. However, such a simple trade-off does not adequately describe deep learning models that simultaneously attain low bias and variance in the heavily overparameterized regime. A primary obstacle in explaining this behavior is that deep learning algorithms typically involve multiple sources of randomness whose individual contributions are not visible in the total variance. To enable fine-grained analysis, we describe an interpretable, symmetric decomposition of the variance into terms associated with the randomness from sampling, initialization, and the labels.

- Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win. [[paper]](https://arxiv.org/abs/2010.03533) [[code]](https://github.com/google-research/rigl)
  - Utku Evci, Yani A. Ioannou, Cem Keskin, Yann Dauphin. *AAAI 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Sparse Neural Networks (NNs) can match the generalization of dense NNs using a fraction of the compute/storage for inference, and also have the potential to enable efficient training. However, naively training unstructured sparse NNs from random initialization results in significantly worse generalization, with the notable exceptions of Lottery Tickets (LTs) and Dynamic Sparse Training (DST). Through our analysis of gradient flow during training we attempt to answer: (1) why training unstructured sparse networks from random initialization performs poorly and; (2) what makes LTs and DST the exceptions?

- Multiple Descent: Design Your Own Generalization Curve. [[paper]](https://arxiv.org/abs/2008.01036)
  - Lin Chen, Yifei Min, Mikhail Belkin, Amin Karbasi. *NeurIPS 2021*
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> This paper explores the generalization loss of linear regression in variably parameterized families of models, both under-parameterized and over-parameterized. We show that the generalization curve can have an arbitrary number of peaks, and moreover, locations of those peaks can be explicitly controlled. Our results highlight the fact that both classical U-shaped generalization curve and the recently observed double descent curve are not intrinsic properties of the model family. Instead, their emergence is due to the interaction between the properties of the data and the inductive biases of learning algorithms.

- Early Stopping in Deep Networks: Double Descent and How to Eliminate it. [[paper]](https://arxiv.org/abs/2007.10099) [[code]](https://github.com/MLI-lab/early_stopping_double_descent)
  - Reinhard Heckel, Fatih Furkan Yilmaz. *ICLR 2021*
  - Key Word: Deep Double Descent; Early Stopping.
  - <details><summary>Digest</summary> We show that such epoch-wise double descent arises for a different reason: It is caused by a superposition of two or more bias-variance tradeoffs that arise because different parts of the network are learned at different epochs, and eliminating this by proper scaling of stepsizes can significantly improve the early stopping performance. We show this analytically for i) linear regression, where differently scaled features give rise to a superposition of bias-variance tradeoffs, and for ii) a two-layer neural network, where the first and second layer each govern a bias-variance tradeoff. Inspired by this theory, we study two standard convolutional networks empirically and show that eliminating epoch-wise double descent through adjusting stepsizes of different layers improves the early stopping performance significantly.

- Triple descent and the two kinds of overfitting: Where & why do they appear? [[paper]](https://arxiv.org/abs/2006.03509) [[code]](https://github.com/sdascoli/triple-descent-paper)
  - Stéphane d'Ascoli, Levent Sagun, Giulio Biroli.
  - Key Word:Deep Double Descent.
  - <details><summary>Digest</summary> In this paper, we show that despite their apparent similarity, these two scenarios are inherently different. In fact, both peaks can co-exist when neural networks are applied to noisy regression tasks. The relative size of the peaks is governed by the degree of nonlinearity of the activation function. Building on recent developments in the analysis of random feature models, we provide a theoretical ground for this sample-wise triple descent.  

- A Brief Prehistory of Double Descent. [[paper]](https://arxiv.org/abs/2004.04328)
  - Marco Loog, Tom Viering, Alexander Mey, Jesse H. Krijthe, David M.J. Tax.
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> This letter draws attention to some original, earlier findings, of interest to double descent. 

- Double Trouble in Double Descent : Bias and Variance(s) in the Lazy Regime. [[paper]](https://arxiv.org/abs/2003.01054) [[code]](https://github.com/lightonai/double-trouble-in-double-descent)
  - Stéphane d'Ascoli, Maria Refinetti, Giulio Biroli, Florent Krzakala. *ICML 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - <details><summary>Digest</summary> Deep neural networks can achieve remarkable generalization performances while interpolating the training data perfectly. Rather than the U-curve emblematic of the bias-variance trade-off, their test error often follows a "double descent" - a mark of the beneficial role of overparametrization. In this work, we develop a quantitative theory for this phenomenon in the so-called lazy learning regime of neural networks, by considering the problem of learning a high-dimensional function with random features regression. We obtain a precise asymptotic expression for the bias-variance decomposition of the test error, and show that the bias displays a phase transition at the interpolation threshold, beyond which it remains constant.

- Rethinking Bias-Variance Trade-off for Generalization of Neural Networks. [[paper]](https://arxiv.org/abs/2002.11328) [[code]](https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff)
  - Zitong Yang, Yaodong Yu, Chong You, Jacob Steinhardt, Yi Ma. *ICML 2020*
  - Key Word: Deep Double Descent; Bias-Variance.
  - <details><summary>Digest</summary> The classical bias-variance trade-off predicts that bias decreases and variance increase with model complexity, leading to a U-shaped risk curve. Recent work calls this into question for neural networks and other over-parameterized models, for which it is often observed that larger models generalize better. We provide a simple explanation for this by measuring the bias and variance of neural networks: while the bias is monotonically decreasing as in the classical theory, the variance is unimodal or bell-shaped: it increases then decreases with the width of the network.

- The Curious Case of Adversarially Robust Models: More Data Can Help, Double Descend, or Hurt Generalization. [[paper]](https://arxiv.org/abs/2002.11080)
  - Yifei Min, Lin Chen, Amin Karbasi. *UAI 2021*
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> We challenge this conventional belief and show that more training data can hurt the generalization of adversarially robust models in the classification problems. We first investigate the Gaussian mixture classification with a linear loss and identify three regimes based on the strength of the adversary. In the weak adversary regime, more data improves the generalization of adversarially robust models. In the medium adversary regime, with more training data, the generalization loss exhibits a double descent curve, which implies the existence of an intermediate stage where more training data hurts the generalization. In the strong adversary regime, more data almost immediately causes the generalization error to increase.

### Deep Double Descent: 2019

- Deep Double Descent: Where Bigger Models and More Data Hurt. [[paper]](https://arxiv.org/abs/1912.02292)  
  - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever. *ICLR 2020*
  - Key Word: Deep Double Descent.
  - <details><summary>Digest</summary> We show that a variety of modern deep learning tasks exhibit a "double-descent" phenomenon where, as we increase model size, performance first gets worse and then gets better.  

### Deep Double Descent: 2018

- Reconciling modern machine learning practice and the bias-variance trade-off. [[paper]](https://arxiv.org/abs/1812.11118)
  - Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal. *PNAS*
  - Key Word: Bias-Variance; Over-Parameterization.
  - <details><summary>Digest</summary> In this paper, we reconcile the classical understanding and the modern practice within a unified performance curve. This "double descent" curve subsumes the textbook U-shaped bias-variance trade-off curve by showing how increasing model capacity beyond the point of interpolation results in improved performance.  

- A Modern Take on the Bias-Variance Tradeoff in Neural Networks. [[paper]](https://arxiv.org/abs/1810.08591)
  - Brady Neal, Sarthak Mittal, Aristide Baratin, Vinayak Tantia, Matthew Scicluna, Simon Lacoste-Julien, Ioannis Mitliagkas.
  - Key Word: Bias-Variance; Over-Parameterization.
  - <details><summary>Digest</summary> The bias-variance tradeoff tells us that as model complexity increases, bias falls and variances increases, leading to a U-shaped test error curve. However, recent empirical results with over-parameterized neural networks are marked by a striking absence of the classic U-shaped test error curve: test error keeps decreasing in wider networks. Motivated by the shaky evidence used to support this claim in neural networks, we measure bias and variance in the modern setting. We find that both bias and variance can decrease as the number of parameters grows. To better understand this, we introduce a new decomposition of the variance to disentangle the effects of optimization and data sampling.

## Lottery Ticket Hypothesis

### Lottery Ticket Hypothesis: 2023

- Pruning at Initialization -- A Sketching Perspective. [[paper]](https://arxiv.org/abs/2305.17559)
  - Noga Bar, Raja Giryes.
  - Key Word: Pruning at Ininitialization; Sketching Algorithm; Neural Tangent Kernel.
  - <details><summary>Digest</summary> The paper studies how to prune linear neural networks before training. They show that this problem is related to the sketching problem for fast matrix multiplication. They use this connection to analyze the error and data dependence of pruning at initialization. They also propose a general improvement to existing pruning algorithms based on sketching techniques.

- NTK-SAP: Improving neural network pruning by aligning training dynamics. [[paper]](https://arxiv.org/abs/2304.02840) [[code]](https://github.com/YiteWang/NTK-SAP)
  - Yite Wang, Dawei Li, Ruoyu Sun. *ICLR 2023*
  - Key Word: Pruning at Ininitialization; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose to prune the connections that have the least influence on the spectrum of the NTK. This method can help maintain the NTK spectrum, which may help align the training dynamics to that of its dense counterpart. However, one possible issue is that the fixed-weight-NTK corresponding to a given initial point can be very different from the NTK corresponding to later iterates during the training phase. 

- Sparsity May Cry: Let Us Fail (Current) Sparse Neural Networks Together! [[paper]](https://arxiv.org/abs/2303.02141) [[code]](https://github.com/VITA-Group/SMC-Bench)
  - Shiwei Liu, Tianlong Chen, Zhenyu Zhang, Xuxi Chen, Tianjin Huang, Ajay Jaiswal, Zhangyang Wang. *ICLR 2023*
  - Key Word: Sparse Neural Network; Benchmark.
  - <details><summary>Digest</summary> In absence of a carefully crafted evaluation benchmark, most if not all, sparse algorithms are evaluated against fairly simple and naive tasks (eg. CIFAR, ImageNet, GLUE, etc.), which can potentially camouflage many advantages as well unexpected predicaments of SNNs. In pursuit of a more general evaluation and unveiling the true potential of sparse algorithms, we introduce "Sparsity May Cry" Benchmark (SMC-Bench), a collection of carefully-curated 4 diverse tasks with 10 datasets, that accounts for capturing a wide range of domain-specific and sophisticated knowledge. 

- Pruning Deep Neural Networks from a Sparsity Perspective. [[paper]](https://arxiv.org/abs/2302.05601) [[code]](https://github.com/dem123456789/Pruning-Deep-Neural-Networks-from-a-Sparsity-Perspective)
  - Enmao Diao, Ganghua Wang, Jiawei Zhan, Yuhong Yang, Jie Ding, Vahid Tarokh. *ICLR 2023*
  - Key Word: Theory of Model Compression; Sparsity Measure.
  - <details><summary>Digest</summary> Many deep pruning algorithms have been proposed with impressive empirical success. However, existing approaches lack a quantifiable measure to estimate the compressibility of a sub-network during each pruning iteration and thus may under-prune or over-prune the model. In this work, we propose PQ Index (PQI) to measure the potential compressibility of deep neural networks and use this to develop a Sparsity-informed Adaptive Pruning (SAP) algorithm. 

- Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning. [[paper]](https://arxiv.org/abs/2301.05219) [[code]](https://github.com/mingsun-tse/why-the-state-of-pruning-so-confusing)
  - Huan Wang, Can Qin, Yue Bai, Yun Fu.
  - Key Word: Pruning; Empirical Study.
  - <details><summary>Digest</summary> Two mysteries in pruning represent such a confusing status: the performance-boosting effect of a larger finetuning learning rate, and the no-value argument of inheriting pretrained weights in filter pruning. In this work, we attempt to explain the confusing state of network pruning by demystifying the two mysteries. 

- Theoretical Characterization of How Neural Network Pruning Affects its Generalization. [[paper]](https://arxiv.org/abs/2301.00335)
  - Hongru Yang, Yingbin Liang, Xiaojie Guo, Lingfei Wu, Zhangyang Wang.
  - Key Word: Lottery Ticket Hypothesis; Generalization Bound.
  - <details><summary>Digest</summary> This work considers a classification task for overparameterized two-layer neural networks, where the network is randomly pruned according to different rates at the initialization. It is shown that as long as the pruning fraction is below a certain threshold, gradient descent can drive the training loss toward zero and the network exhibits good generalization performance. More surprisingly, the generalization bound gets better as the pruning fraction gets larger.

### Lottery Ticket Hypothesis: 2022

- Revisit Kernel Pruning with Lottery Regulated Grouped Convolutions. [[paper]](https://openreview.net/forum?id=LdEhiMG9WLO) [[code]](https://github.com/choH/lottery_regulated_grouped_kernel_pruning)
  - Shaochen Zhong, Guanqun Zhang, Ningjia Huang, Shuai Xu. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We revisit the idea of kernel pruning, a heavily overlooked approach under the context of structured pruning. This is because kernel pruning will naturally introduce sparsity to filters within the same convolutional layer — thus, making the remaining network no longer dense. We address this problem by proposing a versatile grouped pruning framework where we first cluster filters from each convolutional layer into equal-sized groups, prune the grouped kernels we deem unimportant from each filter group, then permute the remaining filters to form a densely grouped convolutional architecture (which also enables the parallel computing capability) for fine-tuning.

- Proving the Lottery Ticket Hypothesis for Convolutional Neural Networks. [[paper]](https://openreview.net/forum?id=Vjki79-619-)
  - Arthur da Cunha, Emanuele Natale, Laurent Viennot, Laurent_Viennot. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent theoretical works proved an even stronger version: every sufficiently overparameterized (dense) neural network contains a subnetwork that, even without training, achieves accuracy comparable to that of the trained large network. These works left as an open problem to extend the result to convolutional neural networks (CNNs). In this work we provide such generalization by showing that, with high probability, it is possible to approximate any CNN by pruning a random CNN whose size is larger by a logarithmic factor.

- Audio Lottery: Speech Recognition Made Ultra-Lightweight, Noise-Robust, and Transferable. [[paper]](https://openreview.net/forum?id=9Nk6AJkVYB) [[code]](https://github.com/VITA-Group/Audio-Lottery)
  - Shaojin Ding, Tianlong Chen, Zhangyang Wang. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis; Speech Recognition.
  - <details><summary>Digest</summary> We investigate the tantalizing possibility of using lottery ticket hypothesis to discover lightweight speech recognition models, that are (1) robust to various noise existing in speech; (2) transferable to fit the open-world personalization; and 3) compatible with structured sparsity.

- Strong Lottery Ticket Hypothesis with ε--perturbation. [[paper]](https://arxiv.org/abs/2210.16589)
  - Zheyang Xiong, Fangshuo Liao, Anastasios Kyrillidis.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The strong Lottery Ticket Hypothesis (LTH) claims the existence of a subnetwork in a sufficiently large, randomly initialized neural network that approximates some target neural network without the need of training. We extend the theoretical guarantee of the strong LTH literature to a scenario more similar to the original LTH, by generalizing the weight change in the pre-training step to some perturbation around initialization. 

- Large Models are Parsimonious Learners: Activation Sparsity in Trained Transformers. [[paper]](https://arxiv.org/abs/2210.06313)
  - Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, Sanjiv Kumar.
  - Key Word: Sparse Activation; Large Models; Transformers.
  - <details><summary>Digest</summary> This paper studies the curious phenomenon for machine learning models with Transformer architectures that their activation maps are sparse. By activation map we refer to the intermediate output of the multi-layer perceptrons (MLPs) after a ReLU activation function, and by "sparse" we mean that on average very few entries (e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input to MLP. 

- Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask? [[paper]](https://arxiv.org/abs/2210.03044)
  - Mansheej Paul, Feng Chen, Brett W. Larsen, Jonathan Frankle, Surya Ganguli, Gintare Karolina Dziugaite.
  - Key Word: Lottery Ticket Hypothesis; Mode Connectivity.
  - <details><summary>Digest</summary> First, we find that—at higher sparsities—pairs of pruned networks at successive pruning iterations are connected by a linear path with zero error barrier if and only if they are matching. This indicates that masks found at the end of training convey the identity of an axial subspace that intersects a desired linearly connected mode of a matching sublevel set. Second, we show SGD can exploit this information due to a strong form of robustness: it can return to this mode despite strong perturbations early in training. Third, we show how the flatness of the error landscape at the end of training determines a limit on the fraction of weights that can be pruned at each iteration of IMP. Finally, we show that the role of retraining in IMP is to find a network with new small weights to prune.

- How Erdös and Rényi Win the Lottery. [[paper]](https://arxiv.org/abs/2210.02412)
  - Advait Gadhikar, Sohum Mukherjee, Rebekka Burkholz.
  - Key Word: Lottery Ticket Hypothesis; Erdös-Rényi Random Graphs.
  - <details><summary>Digest</summary> Random masks define surprisingly effective sparse neural network models, as has been shown empirically. The resulting Erdös-Rényi (ER) random graphs can often compete with dense architectures and state-of-the-art lottery ticket pruning algorithms struggle to outperform them, even though the random baselines do not rely on computationally expensive pruning-training iterations but can be drawn initially without significant computational overhead. We offer a theoretical explanation of how such ER masks can approximate arbitrary target networks if they are wider by a logarithmic factor in the inverse sparsity 1/log(1/sparsity). 

- SparCL: Sparse Continual Learning on the Edge. [[paper]](https://arxiv.org/abs/2209.09476)
  - Zifeng Wang, Zheng Zhan, Yifan Gong, Geng Yuan, Wei Niu, Tong Jian, Bin Ren, Stratis Ioannidis, Yanzhi Wang, Jennifer Dy. *NeurIPS 2022*
  - Key Word: Continual Learning; Sparse Training.
  - <details><summary>Digest</summary> We propose a novel framework called Sparse Continual Learning(SparCL), which is the first study that leverages sparsity to enable cost-effective continual learning on edge devices. SparCL achieves both training acceleration and accuracy preservation through the synergy of three aspects: weight sparsity, data efficiency, and gradient sparsity. Specifically, we propose task-aware dynamic masking (TDM) to learn a sparse network throughout the entire CL process, dynamic data removal (DDR) to remove less informative training data, and dynamic gradient masking (DGM) to sparsify the gradient updates. 

- One-shot Network Pruning at Initialization with Discriminative Image Patches. [[paper]](https://arxiv.org/abs/2209.05683)
  - Yinan Yang, Ying Ji, Yu Wang, Heng Qi, Jien Kato.
  - Key Word: One-Shot Network Pruning.
  - <details><summary>Digest</summary> We propose two novel methods, Discriminative One-shot Network Pruning (DOP) and Super Stitching, to prune the network by high-level visual discriminative image patches. Our contributions are as follows. (1) Extensive experiments reveal that OPaI is data-dependent. (2) Super Stitching performs significantly better than the original OPaI method on benchmark ImageNet, especially in a highly compressed model.

- SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning. [[paper]](https://arxiv.org/abs/2207.03677) [[code]](https://github.com/RICE-EIC/SuperTickets)
  - Haoran You, Baopu Li, Zhanyi Sun, Xu Ouyang, Yingyan Lin. *ECCV 2022*
  - Key Word: Lottery Ticket Hypothesis; Neural Architecture Search.
  - <details><summary>Digest</summary> We discover for the first time that both efficient DNNs and their lottery subnetworks (i.e., lottery tickets) can be directly identified from a supernet, which we term as SuperTickets, via a two-in-one training scheme with jointly architecture searching and parameter pruning. Moreover, we develop a progressive and unified SuperTickets identification strategy that allows the connectivity of subnetworks to change during supernet training, achieving better accuracy and efficiency trade-offs than conventional sparse training.

- Lottery Ticket Hypothesis for Spiking Neural Networks. [[paper]](https://arxiv.org/abs/2207.01382)
  - Youngeun Kim, Yuhang Li, Hyoungseob Park, Yeshwanth Venkatesha, Ruokai Yin, Priyadarshini Panda. *ECCV 2022*
  - Key Word: Lottery Ticket Hypothesis; Spiking Neural Networks.
  - <details><summary>Digest</summary> Spiking Neural Networks (SNNs) have recently emerged as a new generation of low-power deep neural networks where binary spikes convey information across multiple timesteps. Pruning for SNNs is highly important as they become deployed on a resource-constraint mobile/edge device. The previous SNN pruning works focus on shallow SNNs (2~6 layers), however, deeper SNNs (>16 layers) are proposed by state-of-the-art SNN works, which is difficult to be compatible with the current pruning work. To scale up a pruning technique toward deep SNNs, we investigate Lottery Ticket Hypothesis (LTH) which states that dense networks contain smaller subnetworks (i.e., winning tickets) that achieve comparable performance to the dense networks. Our studies on LTH reveal that the winning tickets consistently exist in deep SNNs across various datasets and architectures, providing up to 97% sparsity without huge performance degradation.

- Winning the Lottery Ahead of Time: Efficient Early Network Pruning. [[paper]](https://arxiv.org/abs/2206.10451)
  - John Rachwan, Daniel Zügner, Bertrand Charpentier, Simon Geisler, Morgane Ayle, Stephan Günnemann. *ICML 2022*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - <details><summary>Digest</summary> Although state-of-the-art pruning methods extract highly sparse models, they neglect two main challenges: (1) the process of finding these sparse models is often very expensive; (2) unstructured pruning does not provide benefits in terms of GPU memory, training time, or carbon emissions. We propose Early Compression via Gradient Flow Preservation (EarlyCroP), which efficiently extracts state-of-the-art sparse models before or early in training addressing challenge (1), and can be applied in a structured manner addressing challenge (2). This enables us to train sparse networks on commodity GPUs whose dense versions would be too large, thereby saving costs and reducing hardware requirements.

- "Understanding Robustness Lottery": A Comparative Visual Analysis of Neural Network Pruning Approaches. [[paper]](https://arxiv.org/abs/2206.07918)
  - Zhimin Li, Shusen Liu, Xin Yu, Kailkhura Bhavya, Jie Cao, Diffenderfer James Daniel, Peer-Timo Bremer, Valerio Pascucci.
  - Key Word: Lottery Ticket Hypothesis; Out-of-Distribution Generalization; Visualization.
  - <details><summary>Digest</summary> This work aims to shed light on how different pruning methods alter the network's internal feature representation, and the corresponding impact on model performance. To provide a meaningful comparison and characterization of model feature space, we use three geometric metrics that are decomposed from the common adopted classification loss. With these metrics, we design a visualization system to highlight the impact of pruning on model prediction as well as the latent feature embedding.

- Data-Efficient Double-Win Lottery Tickets from Robust Pre-training. [[paper]](https://arxiv.org/abs/2206.04762) [[code]](https://github.com/VITA-Group/Double-Win-LTH)
  - Tianlong Chen, Zhenyu Zhang, Sijia Liu, Yang Zhang, Shiyu Chang, Zhangyang Wang. *ICML 2022*
  - Key Word: Lottery Ticket Hypothesis; Adversarial Training; Robust Pre-training.
  - <details><summary>Digest</summary> We formulate a more rigorous concept, Double-Win Lottery Tickets, in which a located subnetwork from a pre-trained model can be independently transferred on diverse downstream tasks, to reach BOTH the same standard and robust generalization, under BOTH standard and adversarial training regimes, as the full pre-trained model can do. We comprehensively examine various pre-training mechanisms and find that robust pre-training tends to craft sparser double-win lottery tickets with superior performance over the standard counterparts.

- HideNseek: Federated Lottery Ticket via Server-side Pruning and Sign Supermask. [[paper]](https://arxiv.org/abs/2206.04385)
  - Anish K. Vallapuram, Pengyuan Zhou, Young D. Kwon, Lik Hang Lee, Hengwei Xu, Pan Hui.
  - Key Word: Lottery Ticket Hypothesis; Federated Learning.
  - <details><summary>Digest</summary> We propose HideNseek which employs one-shot data-agnostic pruning at initialization to get a subnetwork based on weights' synaptic saliency. Each client then optimizes a sign supermask multiplied by the unpruned weights to allow faster convergence with the same compression rates as state-of-the-art.

- Lottery Tickets on a Data Diet: Finding Initializations with Sparse Trainable Networks. [[paper]](https://arxiv.org/abs/2206.01278) [[code]](https://github.com/mansheej/lth_diet)
  - Mansheej Paul, Brett W. Larsen, Surya Ganguli, Jonathan Frankle, Gintare Karolina Dziugaite. *NeurIPS 2022*
  - Key Word: Lottery Ticket Hypothesis; Pre-training.
  - <details><summary>Digest</summary> We seek to understand how this early phase of pre-training leads to a good initialization for IMP both through the lens of the data distribution and the loss landscape geometry. Empirically we observe that, holding the number of pre-training iterations constant, training on a small fraction of (randomly chosen) data suffices to obtain an equally good initialization for IMP. We additionally observe that by pre-training only on "easy" training data, we can decrease the number of steps necessary to find a good initialization for IMP compared to training on the full dataset or a randomly chosen subset. Finally, we identify novel properties of the loss landscape of dense networks that are predictive of IMP performance, showing in particular that more examples being linearly mode connected in the dense network correlates well with good initializations for IMP.

- Analyzing Lottery Ticket Hypothesis from PAC-Bayesian Theory Perspective. [[paper]](https://arxiv.org/abs/2205.07320)
  - Keitaro Sakamoto, Issei Sato. *NeurIPS 2022*
  - Key Word: Lottery Ticket Hypothesis; PAC-Bayes.
  - <details><summary>Digest</summary> We confirm this hypothesis and show that the PAC-Bayesian theory can provide an explicit understanding of the relationship between LTH and generalization behavior. On the basis of our experimental findings that flatness is useful for improving accuracy and robustness to label noise and that the distance from the initial weights is deeply involved in winning tickets, we offer the PAC-Bayes bound using a spike-and-slab distribution to analyze winning tickets.

- Dual Lottery Ticket Hypothesis. [[paper]](https://arxiv.org/abs/2203.04248) [[code]](https://github.com/yueb17/dlth)
  - Yue Bai, Huan Wang, Zhiqiang Tao, Kunpeng Li, Yun Fu. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> This paper articulates a Dual Lottery Ticket Hypothesis (DLTH) as a dual format of original Lottery Ticket Hypothesis (LTH). Correspondingly, a simple regularization based sparse network training strategy, Random Sparse Network Transformation (RST), is proposed to validate DLTH and enhance sparse network training.

- Rare Gems: Finding Lottery Tickets at Initialization. [[paper]](https://arxiv.org/abs/2202.12002)
  - Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grinde, Alliot Nagle, Hongyi Wang, Eric Xing, Kangwook Lee, Dimitris Papailiopoulos. *NeurIPS 2022*
  - Key Word: Lottery Ticket Hypothesis; Sanity Checks; Pruning at Initialization.
  - <details><summary>Digest</summary> Finding lottery tickets that train to better accuracy compared to simple baselines remains an open problem. In this work, we resolve this open problem by proposing Gem-Miner which finds lottery tickets at initialization that beat current baselines. Gem-Miner finds lottery tickets trainable to accuracy competitive or better than Iterative Magnitude Pruning (IMP), and does so up to 19× faster.

- Reconstruction Task Finds Universal Winning Tickets. [[paper]](https://arxiv.org/abs/2202.11484)
  - Ruichen Li, Binghui Li, Qi Qian, Liwei Wang.
  - Key Word: Lottery Ticket Hypothesis; Self-Supervision.
  - <details><summary>Digest</summary> We show that the image-level pretrain task is not capable of pruning models for diverse downstream tasks. To mitigate this problem, we introduce image reconstruction, a pixel-level task, into the traditional pruning framework. Concretely, an autoencoder is trained based on the original model, and then the pruning process is optimized with both autoencoder and classification losses.

- Finding Dynamics Preserving Adversarial Winning Tickets. [[paper]](https://arxiv.org/abs/2202.06488) [[code]](https://github.com/google/neural-tangents)
  - Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang. *AISTATS 2022*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - <details><summary>Digest</summary> Based on recent works of Neural Tangent Kernel (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the lottery ticket hypothesis in adversarial context and we refer such sub-network structure as Adversarial Winning Ticket (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

### Lottery Ticket Hypothesis: 2021

- Plant 'n' Seek: Can You Find the Winning Ticket? [[paper]](https://arxiv.org/abs/2111.11153) [[code]](https://github.com/RelationalML/PlantNSeek)
  - Jonas Fischer, Rebekka Burkholz. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Currently, such algorithms are primarily evaluated on imaging data, for which we lack ground truth information and thus the understanding of how sparse lottery tickets could be. To fill this gap, we develop a framework that allows us to plant and hide winning tickets with desirable properties in randomly initialized neural networks. To analyze the ability of state-of-the-art pruning to identify tickets of extreme sparsity, we design and hide such tickets solving four challenging tasks.

- On the Existence of Universal Lottery Tickets. [[paper]](https://arxiv.org/abs/2111.11146) [[code]](https://github.com/relationalml/universallt)
  - Rebekka Burkholz, Nilanjana Laha, Rajarshi Mukherjee, Alkis Gotovos. *ICLR 2022*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The lottery ticket hypothesis conjectures the existence of sparse subnetworks of large randomly initialized deep neural networks that can be successfully trained in isolation. Recent work has experimentally observed that some of these tickets can be practically reused across a variety of tasks, hinting at some form of universality. We formalize this concept and theoretically prove that not only do such universal tickets exist but they also do not require further training.

- Universality of Winning Tickets: A Renormalization Group Perspective. [[paper]](https://arxiv.org/abs/2110.03210)
  - William T. Redman, Tianlong Chen, Zhangyang Wang, Akshunna S. Dogra. *ICML 2022*
  - Key Word: Lottery Ticket Hypothesis; Renormalization Group Theory.
  - <details><summary>Digest</summary> Foundational work on the Lottery Ticket Hypothesis has suggested an exciting corollary: winning tickets found in the context of one task can be transferred to similar tasks, possibly even across different architectures. This has generated broad interest, but methods to study this universality are lacking. We make use of renormalization group theory, a powerful tool from theoretical physics, to address this need. We find that iterative magnitude pruning, the principal algorithm used for discovering winning tickets, is a renormalization group scheme, and can be viewed as inducing a flow in parameter space. 

- How many degrees of freedom do we need to train deep networks: a loss landscape perspective. [[paper]](https://arxiv.org/abs/2107.05802) [[code]](https://github.com/ganguli-lab/degrees-of-freedom)
  - Brett W. Larsen, Stanislav Fort, Nic Becker, Surya Ganguli. *ICLR 2022*
  - Key Word: Loss Landscape; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> A variety of recent works, spanning pruning, lottery tickets, and training within random subspaces, have shown that deep neural networks can be trained using far fewer degrees of freedom than the total number of parameters. We analyze this phenomenon for random subspaces by first examining the success probability of hitting a training loss sublevel set when training within a random subspace of a given training dimensionality.  

- A Winning Hand: Compressing Deep Networks Can Improve Out-Of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2106.09129)
  - James Diffenderfer, Brian R. Bartoldson, Shreya Chaganti, Jize Zhang, Bhavya Kailkhura. *NeurIPS 2021*
  - Key Word: Lottery Ticket Hypothesis; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We perform a large-scale analysis of popular model compression techniques which uncovers several intriguing patterns. Notably, in contrast to traditional pruning approaches (e.g., fine tuning and gradual magnitude pruning), we find that "lottery ticket-style" approaches can surprisingly be used to produce CARDs, including binary-weight CARDs. Specifically, we are able to create extremely compact CARDs that, compared to their larger counterparts, have similar test accuracy and matching (or better) robustness -- simply by pruning and (optionally) quantizing.

- Efficient Lottery Ticket Finding: Less Data is More. [[paper]](https://arxiv.org/abs/2106.03225) [[code]](https://github.com/VITA-Group/PrAC-LTH)
  - Zhenyu Zhang, Xuxi Chen, Tianlong Chen, Zhangyang Wang. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> This paper explores a new perspective on finding lottery tickets more efficiently, by doing so only with a specially selected subset of data, called Pruning-Aware Critical set (PrAC set), rather than using the full training set. The concept of PrAC set was inspired by the recent observation, that deep networks have samples that are either hard to memorize during training, or easy to forget during pruning.

- A Probabilistic Approach to Neural Network Pruning. [[paper]](https://arxiv.org/abs/2105.10065)
  - Xin Qian, Diego Klabjan. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We theoretically study the performance of two pruning techniques (random and magnitude-based) on FCNs and CNNs. Given a target network whose weights are independently sampled from appropriate distributions, we provide a universal approach to bound the gap between a pruned and the target network in a probabilistic sense. The results establish that there exist pruned networks with expressive power within any specified bound from the target network.

- On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning. [[paper]](https://arxiv.org/abs/2105.01648)
  - Marc Aurel Vischer, Robert Tjarko Lange, Henning Sprekeler. *ICLR 2022*
  - Key Word: Reinforcement Learning; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The lottery ticket hypothesis questions the role of overparameterization in supervised deep learning. But how is the performance of winning lottery tickets affected by the distributional shift inherent to reinforcement learning problems? In this work, we address this question by comparing sparse agents who have to address the non-stationarity of the exploration-exploitation problem with supervised agents trained to imitate an expert. We show that feed-forward networks trained with behavioural cloning compared to reinforcement learning can be pruned to higher levels of sparsity without performance degradation.

- Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network. [[paper]](https://arxiv.org/abs/2103.09377) [[code]](https://github.com/chrundle/biprop)
  - James Diffenderfer, Bhavya Kailkhura. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis; Binary Neural Networks.
  - <details><summary>Digest</summary> This provides a new paradigm for learning compact yet highly accurate binary neural networks simply by pruning and quantizing randomly weighted full precision neural networks. We also propose an algorithm for finding multi-prize tickets (MPTs) and test it by performing a series of experiments on CIFAR-10 and ImageNet datasets. Empirical results indicate that as models grow deeper and wider, multi-prize tickets start to reach similar (and sometimes even higher) test accuracy compared to their significantly larger and full-precision counterparts that have been weight-trained.

- Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training. [[paper]](https://arxiv.org/abs/2102.02887) [[code]](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization)
  - Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, Mykola Pechenizkiy. *ICML 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> In this paper, we introduce a new perspective on training deep neural networks capable of state-of-the-art performance without the need for the expensive over-parameterization by proposing the concept of In-Time Over-Parameterization (ITOP) in sparse training. By starting from a random sparse network and continuously exploring sparse connectivities during training, we can perform an Over-Parameterization in the space-time manifold, closing the gap in the expressibility between sparse training and dense training.

- Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks. [[paper]](https://arxiv.org/abs/2102.00554)
  - Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, Alexandra Peste.
  - Key Word: Sparsity; Survey.
  - <details><summary>Digest</summary> We survey prior work on sparsity in deep learning and provide an extensive tutorial of sparsification for both inference and training. We describe approaches to remove and add elements of neural networks, different training strategies to achieve model sparsity, and mechanisms to exploit sparsity in practice. Our work distills ideas from more than 300 research papers and provides guidance to practitioners who wish to utilize sparsity today, as well as to researchers whose goal is to push the frontier forward. 

- A Unified Paths Perspective for Pruning at Initialization. [[paper]](https://arxiv.org/abs/2101.10552)
  - Thomas Gebhart, Udit Saxena, Paul Schrater.
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - <details><summary>Digest</summary> Leveraging recent theoretical approximations provided by the Neural Tangent Kernel, we unify a number of popular approaches for pruning at initialization under a single path-centric framework. We introduce the Path Kernel as the data-independent factor in a decomposition of the Neural Tangent Kernel and show the global structure of the Path Kernel can be computed efficiently. This Path Kernel decomposition separates the architectural effects from the data-dependent effects within the Neural Tangent Kernel, providing a means to predict the convergence dynamics of a network from its architecture alone.

### Lottery Ticket Hypothesis: 2020

- PHEW: Constructing Sparse Networks that Learn Fast and Generalize Well without Training Data. [[paper]](https://arxiv.org/abs/2010.11354) [[code]](https://github.com/ShreyasMalakarjunPatil/PHEW)
  - Shreyas Malakarjun Patil, Constantine Dovrolis. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - <details><summary>Digest</summary>  Our work is based on a recently proposed decomposition of the Neural Tangent Kernel (NTK) that has decoupled the dynamics of the training process into a data-dependent component and an architecture-dependent kernel - the latter referred to as Path Kernel. That work has shown how to design sparse neural networks for faster convergence, without any training data, using the Synflow-L2 algorithm. We first show that even though Synflow-L2 is optimal in terms of convergence, for a given network density, it results in sub-networks with "bottleneck" (narrow) layers - leading to poor performance as compared to other data-agnostic methods that use the same number of parameters.

- A Gradient Flow Framework For Analyzing Network Pruning. [[paper]](https://arxiv.org/abs/2009.11839) [[code]](https://github.com/EkdeepSLubana/flowandprune)
  - Ekdeep Singh Lubana, Robert P. Dick. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent network pruning methods focus on pruning models early-on in training. To estimate the impact of removing a parameter, these methods use importance measures that were originally designed to prune trained models. Despite lacking justification for their use early-on in training, such measures result in surprisingly low accuracy loss. To better explain this behavior, we develop a general framework that uses gradient flow to unify state-of-the-art importance measures through the norm of model parameters.

- Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot. [[paper]](https://arxiv.org/abs/2009.11094) [[code]](https://github.com/JingtongSu/sanity-checking-pruning)
  - Jingtong Su, Yihang Chen, Tianle Cai, Tianhao Wu, Ruiqi Gao, Liwei Wang, Jason D. Lee. *NeurIPS 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We conduct sanity checks for the above beliefs on several recent unstructured pruning methods and surprisingly find that: (1) A set of methods which aims to find good subnetworks of the randomly-initialized network (which we call "initial tickets"), hardly exploits any information from the training data; (2) For the pruned networks obtained by these methods, randomly changing the preserved weights in each layer, while keeping the total number of preserved weights unchanged per layer, does not affect the final performance.

- Pruning Neural Networks at Initialization: Why are We Missing the Mark? [[paper]](https://arxiv.org/abs/2009.08576)
  - Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent work has explored the possibility of pruning neural networks at initialization. We assess proposals for doing so: SNIP (Lee et al., 2019), GraSP (Wang et al., 2020), SynFlow (Tanaka et al., 2020), and magnitude pruning. Although these methods surpass the trivial baseline of random pruning, they remain below the accuracy of magnitude pruning after training, and we endeavor to understand why. We show that, unlike pruning after training, randomly shuffling the weights these methods prune within each layer or sampling new initial values preserves or improves accuracy. As such, the per-weight pruning decisions made by these methods can be replaced by a per-layer choice of the fraction of weights to prune. This property suggests broader challenges with the underlying pruning heuristics, the desire to prune at initialization, or both.

- ESPN: Extremely Sparse Pruned Networks. [[paper]](https://arxiv.org/abs/2006.15741) [[code]](https://github.com/chomd90/extreme_sparse)
  - Minsu Cho, Ameya Joshi, Chinmay Hegde.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Deep neural networks are often highly overparameterized, prohibiting their use in compute-limited systems. However, a line of recent works has shown that the size of deep networks can be considerably reduced by identifying a subset of neuron indicators (or mask) that correspond to significant weights prior to training. We demonstrate that an simple iterative mask discovery method can achieve state-of-the-art compression of very deep networks. Our algorithm represents a hybrid approach between single shot network pruning methods (such as SNIP) with Lottery-Ticket type approaches. We validate our approach on several datasets and outperform several existing pruning approaches in both test accuracy and compression ratio.

- Logarithmic Pruning is All You Need. [[paper]](https://arxiv.org/abs/2006.12156)
  - Laurent Orseau, Marcus Hutter, Omar Rivasplata. *NeurIPS 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The Lottery Ticket Hypothesis is a conjecture that every large neural network contains a subnetwork that, when trained in isolation, achieves comparable performance to the large network. An even stronger conjecture has been proven recently: Every sufficiently overparameterized network contains a subnetwork that, at random initialization, but without training, achieves comparable accuracy to the trained large network. This latter result, however, relies on a number of strong assumptions and guarantees a polynomial factor on the size of the large network compared to the target function. In this work, we remove the most limiting assumptions of this previous work while providing significantly tighter bounds:the overparameterized network only needs a logarithmic factor (in all variables but depth) number of neurons per weight of the target subnetwork.

- Exploring Weight Importance and Hessian Bias in Model Pruning. [[paper]](https://arxiv.org/abs/2006.10903)
  - Mingchen Li, Yahya Sattar, Christos Thrampoulidis, Samet Oymak.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Model pruning is an essential procedure for building compact and computationally-efficient machine learning models. A key feature of a good pruning algorithm is that it accurately quantifies the relative importance of the model weights. While model pruning has a rich history, we still don't have a full grasp of the pruning mechanics even for relatively simple problems involving linear models or shallow neural nets. In this work, we provide a principled exploration of pruning by building on a natural notion of importance.

- Progressive Skeletonization: Trimming more fat from a network at initialization. [[paper]](https://arxiv.org/abs/2006.09081) [[code]](https://github.com/naver/force)
  - Pau de Jorge, Amartya Sanyal, Harkirat S. Behl, Philip H.S. Torr, Gregory Rogez, Puneet K. Dokania. *ICLR 2021*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent studies have shown that skeletonization (pruning parameters) of networks at initialization provides all the practical benefits of sparsity both at inference and training time, while only marginally degrading their performance. However, we observe that beyond a certain level of sparsity (approx 95%), these approaches fail to preserve the network performance, and to our surprise, in many cases perform even worse than trivial random pruning. To this end, we propose an objective to find a skeletonized network with maximum foresight connection sensitivity (FORCE) whereby the trainability, in terms of connection sensitivity, of a pruned network is taken into consideration.

- Pruning neural networks without any data by iteratively conserving synaptic flow. [[paper]](https://arxiv.org/abs/2006.05467) [[code]](https://github.com/ganguli-lab/Synaptic-Flow)
  - Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, Surya Ganguli.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent works have identified, through an expensive sequence of training and pruning cycles, the existence of winning lottery tickets or sparse trainable subnetworks at initialization. This raises a foundational question: can we identify highly sparse trainable subnetworks at initialization, without ever training, or indeed without ever looking at the data? We provide an affirmative answer to this question through theory driven algorithm design.  

- Finding trainable sparse networks through Neural Tangent Transfer. [[paper]](https://arxiv.org/abs/2006.08228) [[code]](https://github.com/fmi-basel/neural-tangent-transfer)
  - Tianlin Liu, Friedemann Zenke. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We introduce Neural Tangent Transfer, a method that instead finds trainable sparse networks in a label-free manner. Specifically, we find sparse networks whose training dynamics, as characterized by the neural tangent kernel, mimic those of dense networks in function space. Finally, we evaluate our label-agnostic approach on several standard classification tasks and show that the resulting sparse networks achieve higher classification performance while converging faster.

- What is the State of Neural Network Pruning? [[paper]](https://arxiv.org/abs/2003.03033) [[code]](https://github.com/jjgo/shrinkbench)
  - Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag. *MLSys 2020*
  - Key Word: Lottery Ticket Hypothesis; Survey.
  - <details><summary>Digest</summary> Neural network pruning---the task of reducing the size of a network by removing parameters---has been the subject of a great deal of work in recent years. We provide a meta-analysis of the literature, including an overview of approaches to pruning and consistent findings in the literature. After aggregating results across 81 papers and pruning hundreds of models in controlled conditions, our clearest finding is that the community suffers from a lack of standardized benchmarks and metrics. This deficiency is substantial enough that it is hard to compare pruning techniques to one another or determine how much progress the field has made over the past three decades. To address this situation, we identify issues with current practices, suggest concrete remedies, and introduce ShrinkBench, an open-source framework to facilitate standardized evaluations of pruning methods.

- Comparing Rewinding and Fine-tuning in Neural Network Pruning. [[paper]](https://arxiv.org/abs/2003.02389) [[code]](https://github.com/lottery-ticket/rewinding-iclr20-public)
  - Alex Renda, Jonathan Frankle, Michael Carbin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We compare fine-tuning to alternative retraining techniques. Weight rewinding (as proposed by Frankle et al., (2019)), rewinds unpruned weights to their values from earlier in training and retrains them from there using the original training schedule. Learning rate rewinding (which we propose) trains the unpruned weights from their final values using the same learning rate schedule as weight rewinding. Both rewinding techniques outperform fine-tuning, forming the basis of a network-agnostic pruning algorithm that matches the accuracy and compression ratios of several more network-specific state-of-the-art techniques.

- Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection. [[paper]](https://arxiv.org/abs/2003.01794) [[code]](https://github.com/lushleaf/Network-Pruning-Greedy-Forward-Selection)
  - Mao Ye, Chengyue Gong, Lizhen Nie, Denny Zhou, Adam Klivans, Qiang Liu. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Recent empirical works show that large deep neural networks are often highly redundant and one can find much smaller subnetworks without a significant drop of accuracy. However, most existing methods of network pruning are empirical and heuristic, leaving it open whether good subnetworks provably exist, how to find them efficiently, and if network pruning can be provably better than direct training using gradient descent. We answer these problems positively by proposing a simple greedy selection approach for finding good subnetworks, which starts from an empty network and greedily adds important neurons from the large network.

- The Early Phase of Neural Network Training. [[paper]](https://arxiv.org/abs/2002.10365) [[code]](https://github.com/facebookresearch/open_lth)
  - Jonathan Frankle, David J. Schwab, Ari S. Morcos. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary>  We find that, within this framework, deep networks are not robust to reinitializing with random weights while maintaining signs, and that weight distributions are highly non-independent even after only a few hundred iterations.  

- Robust Pruning at Initialization. [[paper]](https://arxiv.org/abs/2002.08797)
  - Soufiane Hayou, Jean-Francois Ton, Arnaud Doucet, Yee Whye Teh.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> we provide a comprehensive theoretical analysis of Magnitude and Gradient based pruning at initialization and training of sparse architectures. This allows us to propose novel principled approaches which we validate experimentally on a variety of NN architectures.

- Picking Winning Tickets Before Training by Preserving Gradient Flow. [[paper]](https://arxiv.org/abs/2002.07376) [[code]](https://github.com/alecwangcq/GraSP)
  - Chaoqi Wang, Guodong Zhang, Roger Grosse. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We aim to prune networks at initialization, thereby saving resources at training time as well. Specifically, we argue that efficient training requires preserving the gradient flow through the network. This leads to a simple but effective pruning criterion we term Gradient Signal Preservation (GraSP).

- Lookahead: A Far-Sighted Alternative of Magnitude-based Pruning. [[paper]](https://arxiv.org/abs/2002.04809) [[code]](https://github.com/alinlab/lookahead_pruning)
  - Sejun Park, Jaeho Lee, Sangwoo Mo, Jinwoo Shin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> Magnitude-based pruning is one of the simplest methods for pruning neural networks. Despite its simplicity, magnitude-based pruning and its variants demonstrated remarkable performances for pruning modern architectures. Based on the observation that magnitude-based pruning indeed minimizes the Frobenius distortion of a linear operator corresponding to a single layer, we develop a simple pruning method, coined lookahead pruning, by extending the single layer optimization to a multi-layer optimization.

### Lottery Ticket Hypothesis: 2019

- Linear Mode Connectivity and the Lottery Ticket Hypothesis. [[paper]](https://arxiv.org/abs/1912.05671)
  - Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We study whether a neural network optimizes to the same, linearly connected minimum under different samples of SGD noise (e.g., random data order and augmentation). We find that standard vision models become stable to SGD noise in this way early in training. From then on, the outcome of optimization is determined to a linearly connected region. We use this technique to study iterative magnitude pruning (IMP), the procedure used by work on the lottery ticket hypothesis to identify subnetworks that could have trained in isolation to full accuracy.

- What's Hidden in a Randomly Weighted Neural Network? [[paper]](https://arxiv.org/abs/1911.13299) [[code]](https://github.com/allenai/hidden-networks)
  - Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari. *CVPR 2020*
  - Key Word: Lottery Ticket Hypothesis; Neural Architecture Search; Weight Agnositic Neural Networks.
  - <details><summary>Digest</summary> Hidden in a randomly weighted Wide ResNet-50 we show that there is a subnetwork (with random weights) that is smaller than, but matches the performance of a ResNet-34 trained on ImageNet. Not only do these "untrained subnetworks" exist, but we provide an algorithm to effectively find them.  

- Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks. [[paper]](https://arxiv.org/abs/1909.11957) [[code]](https://github.com/RICE-EIC/Early-Bird-Tickets)
  - Haoran You, Chaojian Li, Pengfei Xu, Yonggan Fu, Yue Wang, Xiaohan Chen, Richard G. Baraniuk, Zhangyang Wang, Yingyan Lin. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We discover for the first time that the winning tickets can be identified at the very early training stage, which we term as early-bird (EB) tickets, via low-cost training schemes (e.g., early stopping and low-precision training) at large learning rates. Our finding of EB tickets is consistent with recently reported observations that the key connectivity patterns of neural networks emerge early.

- Rigging the Lottery: Making All Tickets Winners. [[paper]](https://arxiv.org/abs/1911.11134) [[code]](https://github.com/google-research/rigl)
  - Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, Erich Elsen. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We introduce a method to train sparse neural networks with a fixed parameter count and a fixed computational cost throughout training, without sacrificing accuracy relative to existing dense-to-sparse training methods. Our method updates the topology of the sparse network during training by using parameter magnitudes and infrequent gradient calculations. We show that this approach requires fewer floating-point operations (FLOPs) to achieve a given level of accuracy compared to prior techniques.

- The Difficulty of Training Sparse Neural Networks. [[paper]](https://arxiv.org/abs/1906.10732)
  - Utku Evci, Fabian Pedregosa, Aidan Gomez, Erich Elsen.
  - Key Word: Pruning.
  - <details><summary>Digest</summary> We investigate the difficulties of training sparse neural networks and make new observations about optimization dynamics and the energy landscape within the sparse regime. Recent work of has shown that sparse ResNet-50 architectures trained on ImageNet-2012 dataset converge to solutions that are significantly worse than those found by pruning. We show that, despite the failure of optimizers, there is a linear path with a monotonically decreasing objective from the initialization to the "good" solution.

- A Signal Propagation Perspective for Pruning Neural Networks at Initialization. [[paper]](https://arxiv.org/abs/1906.06307) [[code]](https://github.com/namhoonlee/spp-public)
  - Namhoon Lee, Thalaiyasingam Ajanthan, Stephen Gould, Philip H. S. Torr. *ICLR 2020*
  - Key Word: Lottery Ticket Hypothesis; Mean Field Theory.
  - <details><summary>Digest</summary> In this work, by noting connection sensitivity as a form of gradient, we formally characterize initialization conditions to ensure reliable connection sensitivity measurements, which in turn yields effective pruning results. Moreover, we analyze the signal propagation properties of the resulting pruned networks and introduce a simple, data-free method to improve their trainability.  

- One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers. [[paper]](https://arxiv.org/abs/1906.02773)
  - Ari S. Morcos, Haonan Yu, Michela Paganini, Yuandong Tian. *NeurIPS 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary>  Perhaps surprisingly, we found that, within the natural images domain, winning ticket initializations generalized across a variety of datasets, including Fashion MNIST, SVHN, CIFAR-10/100, ImageNet, and Places365, often achieving performance close to that of winning tickets generated on the same dataset.  

- Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask. [[paper]](https://arxiv.org/abs/1905.01067) [[code]](https://github.com/uber-research/deconstructing-lottery-tickets)
  - Hattie Zhou, Janice Lan, Rosanne Liu, Jason Yosinski. *NeurIPS 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> In this paper, we have studied how three components to LT-style network pruning—mask criterion, treatment of kept weights during retraining (mask-1 action), and treatment of pruned weights during retraining (mask-0 action)—come together to produce sparse and performant subnetworks.

- The State of Sparsity in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1902.09574) [[code]](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn)
  - Trevor Gale, Erich Elsen, Sara Hooker.
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We rigorously evaluate three state-of-the-art techniques for inducing sparsity in deep neural networks on two large-scale learning tasks: Transformer trained on WMT 2014 English-to-German, and ResNet-50 trained on ImageNet. Across thousands of experiments, we demonstrate that complex techniques (Molchanov et al., 2017; Louizos et al., 2017b) shown to yield high compression rates on smaller datasets perform inconsistently, and that simple magnitude pruning approaches achieve comparable or better results.

### Lottery Ticket Hypothesis: 2018

- SNIP: Single-shot Network Pruning based on Connection Sensitivity. [[paper]](https://arxiv.org/abs/1810.02340) [[code]](https://github.com/namhoonlee/snip-public)
  - Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr. *ICLR 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> In this work, we present a new approach that prunes a given network once at initialization prior to training. To achieve this, we introduce a saliency criterion based on connection sensitivity that identifies structurally important connections in the network for the given task.  

- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. [[paper]](https://arxiv.org/abs/1803.03635) [[code]](https://github.com/google-research/lottery-ticket-hypothesis)
  - Jonathan Frankle, Michael Carbin *ICLR 2019*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We find that a standard pruning technique naturally uncovers subnetworks whose initializations made them capable of training effectively. Based on these results, we articulate the "lottery ticket hypothesis:" dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations.  

## Interactions with Neuroscience

### Interactions with Neuroscience: 2023

- Finding Neurons in a Haystack: Case Studies with Sparse Probing. [[paper]](https://arxiv.org/abs/2305.01610) [[code]](https://github.com/wesg52/sparse-probing-paper)
  - Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas. 
  - Key Word: Probing; Mechanistic Interpretability; Superposition; Sparse Coding.
  - <details><summary>Digest</summary> We seek to understand how high-level human-interpretable features are represented within the internal neuron activations of LLMs. We train k-sparse linear classifiers (probes) on these internal activations to predict the presence of features in the input; by varying the value of k we study the sparsity of learned representations and how this varies with model scale. With k=1, we localize individual neurons which are highly relevant for a particular feature, and perform a number of case studies to illustrate general properties of LLMs.

### Interactions with Neuroscience: 2022

- Multilevel development of cognitive abilities in an artificial neural network. [[paper]](https://www.pnas.org/doi/abs/10.1073/pnas.2201304119?af=R)
  - Konstantin Volzhenin, Jean-Pierre Changeux, Guillaume Dumas. *PNAS*
   - Key Word: Global Neuronal Workspace.
   - <details><summary>Digest</summary> We introduce a three-level computational model of information processing and acquisition of cognitive abilities. We propose minimal architectural requirements to build these levels, and how the parameters affect their performance and relationships. The first sensorimotor level handles local nonconscious processing, here during a visual classification task. The second level or cognitive level globally integrates the information from multiple local processors via long-ranged connections and synthesizes it in a global, but still nonconscious, manner. The third and cognitively highest level handles the information globally and consciously. It is based on the global neuronal workspace (GNW) theory and is referred to as the conscious level. 

- Deep Problems with Neural Network Models of Human Vision. [[paper]](https://psyarxiv.com/5zf4s/)
  - Jeffrey S BowersGaurav MalhotraMarin DujmovićMilton Llera MonteroChristian TsvetkovValerio BiscioneGuillermo PueblaFederico G AdolfiJohn HummelRachel Flood HeatonBenjamin EvansJeff MitchellRyan Blything.
  - Key Word: Brain-Score; Computational Neuroscience; Convolutional Neural Networks; Representational Similarity Analysis.
  - <details><summary>Digest</summary> We show that the good prediction on these datasets may be mediated by DNNs that share little overlap with biological vision. More problematically, we show that DNNs account for almost no results from psychological research. This contradicts the common claim that DNNs are good, let alone the best, models of human object recognition.

- Reassessing hierarchical correspondences between brain and deep networks through direct interface. [[paper]](https://www.science.org/doi/10.1126/sciadv.abm2219)
  - Nicholas J Sexton, Bradley C Love. *Science Advances*
  - Key Word: Neural Interfacing Analysis; Shared Neural Variance.
  - <details><summary>Digest</summary> Functional correspondences between deep convolutional neural networks (DCNNs) and the mammalian visual system support a hierarchical account in which successive stages of processing contain ever higher-level information. However, these correspondences between brain and model activity involve shared, not task-relevant, variance. We propose a stricter account of correspondence: If a DCNN layer corresponds to a brain region, then replacing model activity with brain activity should successfully drive the DCNN’s object recognition decision. Using this approach on three datasets, we found that all regions along the ventral visual stream best corresponded with later model layers, indicating that all stages of processing contained higher-level information about object category.

- Wiring Up Vision: Minimizing Supervised Synaptic Updates Needed to Produce a Primate Ventral Stream. [[paper]](https://openreview.net/forum?id=g1SzIRLQXMM)
  - Franziska Geiger, Martin Schrimpf, Tiago Marques, James J. DiCarlo. *ICLR 2022*
  - Key Word: Computational Neuroscience; Primate Visual Ventral Stream.
  - <details><summary>Digest</summary> We develop biologically-motivated initialization and training procedures to train models with 200x fewer synaptic updates (epochs x labeled images x weights) while maintaining 80% of brain predictivity on a set of neural and behavioral benchmarks.

- Curriculum learning as a tool to uncover learning principles in the brain. [[paper]](https://openreview.net/forum?id=TpJMvo0_pu-)
  - Daniel R. Kepple, Rainer Engelken, Kanaka Rajan. *ICLR 2022*
  - Key Word: Curriculum Learning; Neuroscience.
  - <details><summary>Digest</summary> We present a novel approach to use curricula to identify principles by which a system learns. Previous work in curriculum learning has focused on how curricula can be designed to improve learning of a model on particular tasks. We consider the inverse problem: what can a curriculum tell us about how a learning system acquired a task? Using recurrent neural networks (RNNs) and models of common experimental neuroscience tasks, we demonstrate that curricula can be used to differentiate learning principles using target-based and a representation-based loss functions as use cases.

- Building Transformers from Neurons and Astrocytes. [[paper]](https://www.biorxiv.org/content/10.1101/2022.10.12.511910v1)
  - Leo Kozachkov, Ksenia V. Kastanenka, Dmitry Krotov.
  - Key Word: Transformers; Glia; Astrocytes.
  - <details><summary>Digest</summary> In this work we hypothesize that neuron-astrocyte networks can naturally implement the core computation performed by the Transformer block in AI. The omnipresence of astrocytes in almost any brain area may explain the success of Transformers across a diverse set of information domains and computational tasks. 

- High-performing neural network models of visual cortex benefit from high latent dimensionality. [[paper]](https://www.biorxiv.org/content/10.1101/2022.07.13.499969v1)
  - Eric Elmoznino, Michael F. Bonner.
  - Key Word: Dimensionality and Alignment in Computational Brain Models.
  - <details><summary>Digest</summary> The prevailing view holds that optimal DNNs compress their representations onto low-dimensional manifolds to achieve invariance and robustness, which suggests that better models of visual cortex should have low-dimensional geometries. Surprisingly, we found a strong trend in the opposite direction—neural networks with high-dimensional image manifolds tend to have better generalization performance when predicting cortical responses to held-out stimuli in both monkey electrophysiology and human fMRI data.

- On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence. [[paper]](https://arxiv.org/abs/2207.04630)
  - Yi Ma, Doris Tsao, Heung-Yeung Shum.
  - Key Word: Intelligence; Parsimony; Self-Consistency; Rate Reduction.
  - <details><summary>Digest</summary> Ten years into the revival of deep networks and artificial intelligence, we propose a theoretical framework that sheds light on understanding deep networks within a bigger picture of Intelligence in general. We introduce two fundamental principles, Parsimony and Self-consistency, that we believe to be cornerstones for the emergence of Intelligence, artificial or natural. While these two principles have rich classical roots, we argue that they can be stated anew in entirely measurable and computable ways.

- Painful intelligence: What AI can tell us about human suffering. [[paper]](https://arxiv.org/abs/2205.15409)
  - Aapo Hyvärinen.
  - Key Word: Neuroscience.
  - <details><summary>Digest</summary> This book uses the modern theory of artificial intelligence (AI) to understand human suffering or mental pain. Both humans and sophisticated AI agents process information about the world in order to achieve goals and obtain rewards, which is why AI can be used as a model of the human brain and mind. This book intends to make the theory accessible to a relatively general audience, requiring only some relevant scientific background. The book starts with the assumption that suffering is mainly caused by frustration. Frustration means the failure of an agent (whether AI or human) to achieve a goal or a reward it wanted or expected.

- The developmental trajectory of object recognition robustness: children are like small adults but unlike big deep neural networks. [[paper]](https://arxiv.org/abs/2205.10144) [[code]](https://github.com/wichmann-lab/robustness-development)
  - Lukas S. Huber, Robert Geirhos, Felix A. Wichmann.
  - Key Word: Object Recognition; Out-of-Distribution Generalization; Children.
  - <details><summary>Digest</summary>  We find, first, that already 4–6 year-olds showed remarkable robustness to image distortions and outperform DNNs trained on ImageNet. Second, we estimated the number of “images” children have been exposed to during their lifetime. Compared to various DNNs, children's high robustness requires relatively little data. Third, when recognizing objects children—like adults but unlike DNNs—rely heavily on shape but not on texture cues. Together our results suggest that the remarkable robustness to distortions emerges early in the developmental trajectory of human object recognition and is unlikely the result of a mere accumulation of experience with distorted visual input.

- Finding Biological Plausibility for Adversarially Robust Features via Metameric Tasks. [[paper]](https://arxiv.org/abs/2202.00838) [[code]](https://github.com/anneharrington/adversarially-robust-periphery)
  - Anne Harrington, Arturo Deza. *ICLR 2022*
  - Key Word: Adversarial Robustness; Peripheral Computation; Psychophysics.
  - <details><summary>Digest</summary> To understand how adversarially robust optimizations/representations compare to human vision, we performed a psychophysics experiment using a metamer task where we evaluated how well human observers could distinguish between images synthesized to match adversarially robust representations compared to non-robust representations and a texture synthesis model of peripheral vision.  We found that the discriminability of robust representation and texture model images decreased to near chance performance as stimuli were presented farther in the periphery.  

### Interactions with Neuroscience: 2021

- Relating transformers to models and neural representations of the hippocampal formation. [[paper]](https://arxiv.org/abs/2112.04035)
  - James C.R. Whittington, Joseph Warren, Timothy E.J. Behrens. *ICLR 2022*
  - Key Word: Transformers; Hippocampus; Cortex.
  - <details><summary>Digest</summary> We show that transformers, when equipped with recurrent position encodings, replicate the precisely tuned spatial representations of the hippocampal formation; most notably place and grid cells. Furthermore, we show that this result is no surprise since it is closely related to current hippocampal models from neuroscience.

- Partial success in closing the gap between human and machine vision. [[paper]](https://arxiv.org/abs/2106.07411) [[code]](https://github.com/bethgelab/model-vs-human)
  - Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *NeurIPS 2021*
  - Key Word: Out-of-Distribution Generalization; Psychophysical Experiments.
  - <details><summary>Digest</summary> A few years ago, the first CNN surpassed human performance on ImageNet. However, it soon became clear that machines lack robustness on more challenging test cases, a major obstacle towards deploying machines "in the wild" and towards obtaining better computational models of human visual perception. Here we ask: Are we making progress in closing the gap between human and machine vision? To answer this question, we tested human observers on a broad range of out-of-distribution (OOD) datasets, recording 85,120 psychophysical trials across 90 participants.

- Does enhanced shape bias improve neural network robustness to common corruptions? [[paper]](https://arxiv.org/abs/2104.09789)
  - Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, Jan Hendrik Metzen. *ICLR 2021*
  - Key Word: Shape-Texture; Robustness.
  - <details><summary>Digest</summary> We perform a systematic study of different ways of composing inputs based on natural images, explicit edge information, and stylization. While stylization is essential for achieving high corruption robustness, we do not find a clear correlation between shape bias and robustness. We conclude that the data augmentation caused by style-variation accounts for the improved corruption robustness and increased shape bias is only a byproduct.

### Interactions with Neuroscience: 2020

- Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. [[paper]](https://proceedings.neurips.cc/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html)
  - Joel Dapello, Tiago Marques, Martin Schrimpf, Franziska Geiger, David Cox, James J. DiCarlo. *NeurIPS 2020*
  - Key Word: Robustness; V1 Model.
  - <details><summary>Digest</summary> Current state-of-the-art object recognition models are largely based on convolutional neural network (CNN) architectures, which are loosely inspired by the primate visual system. However, these CNNs can be fooled by imperceptibly small, explicitly crafted perturbations, and struggle to recognize objects in corrupted images that are easily recognized by humans. Here, by making comparisons with primate neural data, we first observed that CNN models with a neural hidden layer that better matches primate primary visual cortex (V1) are also more robust to adversarial attacks. Inspired by this observation, we developed VOneNets, a new class of hybrid CNN vision models. Each VOneNet contains a fixed weight neural network front-end that simulates primate V1, called the VOneBlock, followed by a neural network back-end adapted from current CNN vision models.

- On 1/n neural representation and robustness. [[paper]](https://arxiv.org/abs/2012.04729) [[code]](https://github.com/josuenassar/power_law)
  - Josue Nassar, Piotr Aleksander Sokol, SueYeon Chung, Kenneth D. Harris, Il Memming Park. *NeurIPS 2020*
  - Key Word: Adversarial Robustness; 1/n Power Law.
  - <details><summary>Digest</summary> We investigate the latter by juxtaposing experimental results regarding the covariance spectrum of neural representations in the mouse V1 (Stringer et al) with artificial neural networks. We use adversarial robustness to probe Stringer et al's theory regarding the causal role of a 1/n covariance spectrum. We empirically investigate the benefits such a neural code confers in neural networks, and illuminate its role in multi-layer architectures. Our results show that imposing the experimentally observed structure on artificial neural networks makes them more robust to adversarial attacks. Moreover, our findings complement the existing theory relating wide neural networks to kernel methods, by showing the role of intermediate representations.

- Shape-Texture Debiased Neural Network Training. [[paper]](https://arxiv.org/abs/2010.05981) [[code]](https://github.com/LiYingwei/ShapeTextureDebiasedTraining)
  - Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan Yuille, Cihang Xie. *ICLR 2021*
  - Key Word: Shape-Texture; Robustness.
  - <details><summary>Digest</summary> Shape and texture are two prominent and complementary cues for recognizing objects. Nonetheless, Convolutional Neural Networks are often biased towards either texture or shape, depending on the training dataset. Our ablation shows that such bias degenerates model performance. Motivated by this observation, we develop a simple algorithm for shape-texture debiased learning. To prevent models from exclusively attending on a single cue in representation learning, we augment training data with images with conflicting shape and texture information (eg, an image of chimpanzee shape but with lemon texture) and, most importantly, provide the corresponding supervisions from shape and texture simultaneously.

- Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency. [[paper]](https://arxiv.org/abs/2006.16736) [[code]](https://github.com/wichmann-lab/error-consistency)
  - Robert Geirhos, Kristof Meding, Felix A. Wichmann.
  - Key Word: Error Consistency.
  - <details><summary>Digest</summary> Here we introduce trial-by-trial error consistency, a quantitative analysis for measuring whether two decision making systems systematically make errors on the same inputs. Making consistent errors on a trial-by-trial basis is a necessary condition if we want to ascertain similar processing strategies between decision makers.  

- Biologically Inspired Mechanisms for Adversarial Robustness. [[paper]](https://arxiv.org/abs/2006.16427)
  - Manish V. Reddy, Andrzej Banburski, Nishka Pant, Tomaso Poggio. *NeurIPS 2020*
  - Key Word: Robustness; Retinal Fixations.
  - <details><summary>Digest</summary> A convolutional neural network strongly robust to adversarial perturbations at reasonable computational and performance cost has not yet been demonstrated. The primate visual ventral stream seems to be robust to small perturbations in visual stimuli but the underlying mechanisms that give rise to this robust perception are not understood. In this work, we investigate the role of two biologically plausible mechanisms in adversarial robustness. We demonstrate that the non-uniform sampling performed by the primate retina and the presence of multiple receptive fields with a range of receptive field sizes at each eccentricity improve the robustness of neural networks to small adversarial perturbations

- Five Points to Check when Comparing Visual Perception in Humans and Machines. [[paper]](https://arxiv.org/abs/2004.09406) [[code]](https://github.com/bethgelab/notorious_difficulty_of_comparing_human_and_machine_perception)
  - Christina M. Funke, Judy Borowski, Karolina Stosio, Wieland Brendel, Thomas S. A. Wallis, Matthias Bethge. *JOV*
  - Key Word: Model Comparison.
  - <details><summary>Digest</summary> With the rise of machines to human-level performance in complex recognition tasks, a growing amount of work is directed towards comparing information processing in humans and machines. These studies are an exciting chance to learn about one system by studying the other. Here, we propose ideas on how to design, conduct and interpret experiments such that they adequately support the investigation of mechanisms when comparing human and machine perception. We demonstrate and apply these ideas through three case studies.

- Shortcut Learning in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2004.07780) [[code]](https://github.com/rgeirhos/shortcut-perspective)
  - Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. *Nature Machine Intelligence*
  - Key Word: Out-of-Distribution Generalization; Survey.
  - <details><summary>Digest</summary> Deep learning has triggered the current rise of artificial intelligence and is the workhorse of today's machine intelligence. Numerous success stories have rapidly spread all over science, industry and society, but its limitations have only recently come into focus. In this perspective we seek to distil how many of deep learning's problem can be seen as different symptoms of the same underlying problem: shortcut learning. Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios. Related issues are known in Comparative Psychology, Education and Linguistics, suggesting that shortcut learning may be a common characteristic of learning systems, biological and artificial alike. Based on these observations, we develop a set of recommendations for model interpretation and benchmarking, highlighting recent advances in machine learning to improve robustness and transferability from the lab to real-world applications.

### Interactions with Neuroscience: 2019

- A deep learning framework for neuroscience. [[paper]](https://www.nature.com/articles/s41593-019-0520-2)
  - <details><summary>Author List</summary> Blake A. Richards, Timothy P. Lillicrap, Philippe Beaudoin, Yoshua Bengio, Rafal Bogacz, Amelia Christensen, Claudia Clopath, Rui Ponte Costa, Archy de Berker, Surya Ganguli, Colleen J. Gillon, Danijar Hafner, Adam Kepecs, Nikolaus Kriegeskorte, Peter Latham, Grace W. Lindsay, Kenneth D. Miller, Richard Naud, Christopher C. Pack, Panayiota Poirazi, Pieter Roelfsema, João Sacramento, Andrew Saxe, Benjamin Scellier, Anna C. Schapiro, Walter Senn, Greg Wayne, Daniel Yamins, Friedemann Zenke, Joel Zylberberg, Denis Therien & Konrad P. Kording. *Nature Neuroscience*
  - Key Word: Deep Learning; Neuroscience.
  - <details><summary>Digest</summary> The article discusses the similarities and differences between systems neuroscience and artificial intelligence. It argues that the three components in artificial neural networks - objective functions, learning rules, and architectures - are crucial for modeling and optimizing complex artificial learning systems. The authors suggest that a greater focus on these components could benefit systems neuroscience and drive theoretical and experimental progress. 

- White Noise Analysis of Neural Networks. [[paper]](https://arxiv.org/abs/1912.12106) [[code]](https://github.com/aliborji/WhiteNoiseAnalysis)
  - Ali Borji, Sikun Lin. *ICLR 2020*
  - Key Word: Spike-Triggered Analysis.
  - <details><summary>Digest</summary> A white noise analysis of modern deep neural networks is presented to unveil their biases at the whole network level or the single neuron level. Our analysis is based on two popular and related methods in psychophysics and neurophysiology namely classification images and spike triggered analysis.  

- The Origins and Prevalence of Texture Bias in Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1911.09071)
  - Katherine L. Hermann, Ting Chen, Simon Kornblith. *NeurIPS 2020*
  - Key Word: Shape-Texture; Robustness.
  - <details><summary>Digest</summary> Recent work has indicated that, unlike humans, ImageNet-trained CNNs tend to classify images by texture rather than by shape. How pervasive is this bias, and where does it come from? We find that, when trained on datasets of images with conflicting shape and texture, CNNs learn to classify by shape at least as easily as by texture. What factors, then, produce the texture bias in CNNs trained on ImageNet? Different unsupervised training objectives and different architectures have small but significant and largely independent effects on the level of texture bias. However, all objectives and architectures still lead to models that make texture-based classification decisions a majority of the time, even if shape information is decodable from their hidden representations. The effect of data augmentation is much larger.

- Learning From Brains How to Regularize Machines. [[paper]](https://arxiv.org/abs/1911.05072)
  - Zhe Li, Wieland Brendel, Edgar Y. Walker, Erick Cobos, Taliah Muhammad, Jacob Reimer, Matthias Bethge, Fabian H. Sinz, Xaq Pitkow, Andreas S. Tolias. *NeurIPS 2019*
  - Key Word: Neural Representation Similarity.
  - <details><summary>Digest</summary> Despite impressive performance on numerous visual tasks, Convolutional Neural Networks (CNNs) --- unlike brains --- are often highly sensitive to small perturbations of their input, e.g. adversarial noise leading to erroneous decisions. We propose to regularize CNNs using large-scale neuroscience data to learn more robust neural features in terms of representational similarity. We presented natural images to mice and measured the responses of thousands of neurons from cortical visual areas.

- A Unified Theory of Early Visual Representations from Retina to Cortex through Anatomically Constrained Deep CNNs. [[paper]](https://arxiv.org/abs/1901.00945) [[code]](https://github.com/ganguli-lab/RetinalResources)
  - Jack Lindsey, Samuel A. Ocko, Surya Ganguli, Stephane Deny. *ICLR 2019*
  - Key Word: Visual System; Convolutional Neural Networks; Efficient Coding; Retina.
  - <details><summary>Digest</summary> There is currently no unified theory explaining these differences in representations across layers. Here, using a deep convolutional neural network trained on image recognition as a model of the visual system, we show that such differences in representation can emerge as a direct consequence of different neural resource constraints on the retinal and cortical networks, and we find a single model from which both geometries spontaneously emerge at the appropriate stages of visual processing. The key constraint is a reduced number of neurons at the retinal output, consistent with the anatomy of the optic nerve as a stringent bottleneck.

### Interactions with Neuroscience: 2018

- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. [[paper]](https://arxiv.org/abs/1811.12231) [[code]](https://github.com/rgeirhos/Stylized-ImageNet)
  - Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *ICLR 2019*
  - Key Word: Shape-Texture; Psychophysical Experiments.
  - <details><summary>Digest</summary> Convolutional Neural Networks (CNNs) are commonly thought to recognise objects by learning increasingly complex representations of object shapes. Some recent studies suggest a more important role of image textures. We here put these conflicting hypotheses to a quantitative test by evaluating CNNs and human observers on images with a texture-shape cue conflict. We show that ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark contrast to human behavioural evidence and reveals fundamentally different classification strategies.

- Generalisation in humans and deep neural networks. [[paper]](https://arxiv.org/abs/1808.08750) [[code]](https://github.com/rgeirhos/generalisation-humans-DNNs)
  - Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Robustness.
  - <details><summary>Digest</summary> We compare the robustness of humans and current convolutional deep neural networks (DNNs) on object recognition under twelve different types of image degradations. First, using three well known DNNs (ResNet-152, VGG-19, GoogLeNet) we find the human visual system to be more robust to nearly all of the tested image manipulations, and we observe progressively diverging classification error-patterns between humans and DNNs when the signal gets weaker. Secondly, we show that DNNs trained directly on distorted images consistently surpass human performance on the exact distortion types they were trained on, yet they display extremely poor generalisation abilities when tested on other distortion types.

### Interactions with Neuroscience: 2017

- Comparing deep neural networks against humans: object recognition when the signal gets weaker. [[paper]](https://arxiv.org/abs/1706.06969) [[code]](https://github.com/rgeirhos/object-recognition)
  - Robert Geirhos, David H. J. Janssen, Heiko H. Schütt, Jonas Rauber, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Model Comparison; Robustness.
  - <details><summary>Digest</summary> Human visual object recognition is typically rapid and seemingly effortless, as well as largely independent of viewpoint and object orientation. Until very recently, animate visual systems were the only ones capable of this remarkable computational feat. This has changed with the rise of a class of computer vision algorithms called deep neural networks (DNNs) that achieve human-level classification performance on object recognition tasks. Furthermore, a growing number of studies report similarities in the way DNNs and the human visual system process objects, suggesting that current DNNs may be good models of human visual object recognition. Yet there clearly exist important architectural and processing differences between state-of-the-art DNNs and the primate visual system. The potential behavioural consequences of these differences are not well understood. We aim to address this issue by comparing human and DNN generalisation abilities towards image degradations.

## Information Bottleneck

### Information Bottleneck: 2023

- To Compress or Not to Compress -- Self-Supervised Learning and Information Theory: A Review. [[paper]](https://arxiv.org/abs/2304.09355)
  - Ravid Shwartz-Ziv, Yann LeCun.
  - Key Word: Self-Supervised Learning; Information Theory; Survey.
  - <details><summary>Digest</summary> We review various approaches to self-supervised learning from an information-theoretic standpoint and present a unified framework that formalizes the self-supervised information-theoretic learning problem. We integrate existing research into a coherent framework, examine recent self-supervised methods, and identify research opportunities and challenges.

### Information Bottleneck: 2022

- Sparsity-Inducing Categorical Prior Improves Robustness of the Information Bottleneck. [[paper]](https://arxiv.org/abs/2203.02592)
  - Anirban Samaddar, Sandeep Madireddy, Prasanna Balaprakash
  - Key Word: Information Bottleneck; Robustness.
  - <details><summary>Digest</summary> We present a novel sparsity-inducing spike-slab prior that uses sparsity as a mechanism to provide flexibility that allows each data point to learn its own dimension distribution. In addition, it provides a mechanism to learn a joint distribution of the latent variable and the sparsity. Thus, unlike other approaches, it can account for the full uncertainty in the latent space.

### Information Bottleneck: 2021

- Information Bottleneck Disentanglement for Identity Swapping. [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Information_Bottleneck_Disentanglement_for_Identity_Swapping_CVPR_2021_paper.html)
  - Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, Ran He. *CVPR 2021*
  - Key Word: Information Bottleneck; Identity Swapping.
  - <details><summary>Digest</summary> We propose a novel information disentangling and swapping network, called InfoSwap, to extract the most expressive information for identity representation from a pre-trained face recognition model. The key insight of our method is to formulate the learning of disentangled representations as optimizing an information bottleneck trade-off, in terms of finding an optimal compression of the pre-trained latent features.

- PAC-Bayes Information Bottleneck. [[paper]](https://arxiv.org/abs/2109.14509) [[code]](https://github.com/ryanwangzf/pac-bayes-ib)
  - Zifeng Wang, Shao-Lun Huang, Ercan E. Kuruoglu, Jimeng Sun, Xi Chen, Yefeng Zheng. *ICLR 2022*
  - Key Word: Information Bottleneck; PAC-Bayes.
  - <details><summary>Digest</summary> There have been a series of theoretical works trying to derive non-vacuous bounds for NNs. Recently, the compression of information stored in weights (IIW) is proved to play a key role in NNs generalization based on the PAC-Bayes theorem. However, no solution of IIW has ever been provided, which builds a barrier for further investigation of the IIW's property and its potential in practical deep learning. In this paper, we propose an algorithm for the efficient approximation of IIW. Then, we build an IIW-based information bottleneck on the trade-off between accuracy and information complexity of NNs, namely PIB.

- Information Bottleneck: Exact Analysis of (Quantized) Neural Networks. [[paper]](https://arxiv.org/abs/2106.12912) [[code]](https://github.com/StephanLorenzen/ExactIBAnalysisInQNNs)
  - Stephan Sloth Lorenzen, Christian Igel, Mads Nielsen. *ICLR 2022*
  - Key Word: Information Bottleneck; Quantization.
  - <details><summary>Digest</summary> We study the IB principle in settings where MI is non-trivial and can be computed exactly. We monitor the dynamics of quantized neural networks, that is, we discretize the whole deep learning system so that no approximation is required when computing the MI. This allows us to quantify the information flow without measurement errors.

- Compressive Visual Representations. [[paper]](https://arxiv.org/abs/2109.12909) [[code]](https://github.com/google-research/compressive-visual-representations)
  - Kuang-Huei Lee, Anurag Arnab, Sergio Guadarrama, John Canny, Ian Fischer. *NeurIPS*
  - Key Word: Self-Supervision; Contrastive Learning; Conditional Entropy Bottleneck; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We hypothesize that adding explicit information compression to these algorithms yields better and more robust representations. We verify this by developing SimCLR and BYOL formulations compatible with the Conditional Entropy Bottleneck (CEB) objective, allowing us to both measure and control the amount of compression in the learned representation, and observe their impact on downstream tasks. Furthermore, we explore the relationship between Lipschitz continuity and compression, showing a tractable lower bound on the Lipschitz constant of the encoders we learn.

- Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.06607) [[code]](https://github.com/ahujak/IB-IRM)
  - Kartik Ahuja, Ethan Caballero, Dinghuai Zhang, Yoshua Bengio, Ioannis Mitliagkas, Irina Rish. *NeurIPS 2021*
  - Key Word: Information Bottleneck; Out-of-Distribution Generalization; Invarianct Risk Minimization.
  - <details><summary>Digest</summary> We revisit the fundamental assumptions in linear regression tasks, where invariance-based approaches were shown to provably generalize OOD. In contrast to the linear regression tasks, we show that for linear classification tasks we need much stronger restrictions on the distribution shifts, or otherwise OOD generalization is impossible.  Furthermore, even with appropriate restrictions on distribution shifts in place, we show that the invariance principle alone is insufficient. We prove that a form of the information bottleneck constraint along with invariance helps address the key failures when invariant features capture all the information about the label and also retains the existing success when they do not.

- Perturbation Theory for the Information Bottleneck. [[paper]](https://arxiv.org/abs/2105.13977)
  - Vudtiwat Ngampruetikorn, David J. Schwab. *NeurIPS 2021*
  - Key Word: Information Bottleneck; Perturbation Theory.
  - <details><summary>Digest</summary> Extracting relevant information from data is crucial for all forms of learning. The information bottleneck (IB) method formalizes this, offering a mathematically precise and conceptually appealing framework for understanding learning phenomena. However the nonlinearity of the IB problem makes it computationally expensive and analytically intractable in general. Here we derive a perturbation theory for the IB method and report the first complete characterization of the learning onset, the limit of maximum relevant information per bit extracted from data. 

- A Critical Review of Information Bottleneck Theory and its Applications to Deep Learning. [[paper]](https://arxiv.org/abs/2105.04405)
  - Mohammad Ali Alomrani.
  - Key Word: Information Bottleneck; Survey.
  - <details><summary>Digest</summary> A known information-theoretic method called the information bottleneck theory has emerged as a promising approach to better understand the learning dynamics of neural networks. In principle, IB theory models learning as a trade-off between the compression of the data and the retainment of information. The goal of this survey is to provide a comprehensive review of IB theory covering it's information theoretic roots and the recently proposed applications to understand deep learning models.

### Information Bottleneck: 2020

- Graph Information Bottleneck. [[paper]](https://arxiv.org/abs/2010.12811) [[code]](https://github.com/snap-stanford/GIB)
  - Tailin Wu, Hongyu Ren, Pan Li, Jure Leskovec. *NeurIPS 2020*
  - Key Word: Information Bottleneck; Graph Neural Networks.
  - <details><summary>Digest</summary> We introduce Graph Information Bottleneck (GIB), an information-theoretic principle that optimally balances expressiveness and robustness of the learned representation of graph-structured data. Inheriting from the general Information Bottleneck (IB), GIB aims to learn the minimal sufficient representation for a given task by maximizing the mutual information between the representation and the target, and simultaneously constraining the mutual information between the representation and the input data.

- Learning Optimal Representations with the Decodable Information Bottleneck. [[paper]](https://arxiv.org/abs/2009.12789) [[code]](https://github.com/YannDubs/Mini_Decodable_Information_Bottleneck)
  - Yann Dubois, Douwe Kiela, David J. Schwab, Ramakrishna Vedantam. *NeurIPS 2020*
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> We propose the Decodable Information Bottleneck (DIB) that considers information retention and compression from the perspective of the desired predictive family. As a result, DIB gives rise to representations that are optimal in terms of expected test performance and can be estimated with guarantees. Empirically, we show that the framework can be used to enforce a small generalization gap on downstream classifiers and to predict the generalization ability of neural networks.

- Concept Bottleneck Models. [[paper]](https://arxiv.org/abs/2007.04612) [[code]](https://github.com/yewsiang/ConceptBottleneck)
  - Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, Percy Liang. *ICML 2020*
  - Key Word: Information Bottleneck
  - <details><summary>Digest</summary> We seek to learn models that we can interact with using high-level concepts: if the model did not think there was a bone spur in the x-ray, would it still predict severe arthritis? State-of-the-art models today do not typically support the manipulation of concepts like "the existence of bone spurs", as they are trained end-to-end to go directly from raw input (e.g., pixels) to output (e.g., arthritis severity). We revisit the classic idea of first predicting concepts that are provided at training time, and then using these concepts to predict the label. By construction, we can intervene on these concept bottleneck models by editing their predicted concept values and propagating these changes to the final prediction.

- On Information Plane Analyses of Neural Network Classifiers -- A Review. [[paper]](https://arxiv.org/abs/2003.09671)
  - Bernhard C. Geiger. *TNNLS*
  - Key Word: Information Bottleneck; Survey.
  - <details><summary>Digest</summary> We review the current literature concerned with information plane analyses of neural network classifiers. While the underlying information bottleneck theory and the claim that information-theoretic compression is causally linked to generalization are plausible, empirical evidence was found to be both supporting and conflicting. We review this evidence together with a detailed analysis of how the respective information quantities were estimated.

- On the Information Bottleneck Problems: Models, Connections, Applications and Information Theoretic Views. [[paper]](https://arxiv.org/abs/2002.00008)
  - Abdellatif Zaidi, Inaki Estella Aguerri, Shlomo Shamai. *Entropy*
  - Key Word: Information Bottleneck; Survey.
  - <details><summary>Digest</summary> This tutorial paper focuses on the variants of the bottleneck problem taking an information theoretic perspective and discusses practical methods to solve it, as well as its connection to coding and learning aspects. The intimate connections of this setting to remote source-coding under logarithmic loss distortion measure, information combining, common reconstruction, the Wyner-Ahlswede-Korner problem, the efficiency of investment information, as well as, generalization, variational inference, representation learning, autoencoders, and others are highlighted.

- Phase Transitions for the Information Bottleneck in Representation Learning. [[paper]](https://arxiv.org/abs/2001.01878)
  - Tailin Wu, Ian Fischer. *ICLR 2020*
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> Our work provides the first theoretical formula to address IB phase transitions in the most general setting. In addition, we present an algorithm for iteratively finding the IB phase transition points.

- Restricting the Flow: Information Bottlenecks for Attribution. [[paper]](https://arxiv.org/abs/2001.00396) [[code]](https://github.com/BioroboticsLab/IBA-paper-code)
  - Karl Schulz, Leon Sixt, Federico Tombari, Tim Landgraf. *ICLR 2020*
  - Key Word: Information Bottleneck; Attribution.
  - <details><summary>Digest</summary> We adapt the information bottleneck concept for attribution. By adding noise to intermediate feature maps we restrict the flow of information and can quantify (in bits) how much information image regions provide.

### Information Bottleneck: 2019

- Learnability for the Information Bottleneck. [[paper]](https://arxiv.org/abs/1907.07331)
  - Tailin Wu, Ian Fischer, Isaac L. Chuang, Max Tegmark. *UAI 2019*
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> We presented theoretical results for predicting the onset of learning, and have shown that it is determined by the conspicuous subset of the training examples. We gave a practical algorithm for predicting the transition as well as discovering this subset, and showed that those predictions are accurate, even in cases of extreme label noise.

### Information Bottleneck: 2018

- On the Information Bottleneck Theory of Deep Learning. [[paper]](https://openreview.net/forum?id=ry_WPG-A-) [[code]](https://github.com/artemyk/ibsgd/tree/iclr2018)
  - Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox. *ICLR 2018*
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> This submission explores [recent theoretical work](https://arxiv.org/abs/1703.00810) by Shwartz-Ziv and Tishby on explaining the generalization ability of deep networks. The paper gives counter-examples that suggest aspects of the theory might not be relevant for all neural networks.  

### Information Bottleneck: 2017

- Emergence of Invariance and Disentanglement in Deep Representations. [[paper]](https://arxiv.org/abs/1706.01350)
  - Alessandro Achille, Stefano Soatto. *JMLR*
  - Key Word: PAC-Bayes; Information Bottleneck.
  - <details><summary>Digest</summary> Using established principles from Statistics and Information Theory, we show that invariance to nuisance factors in a deep neural network is equivalent to information minimality of the learned representation, and that stacking layers and injecting noise during training naturally bias the network towards learning invariant representations. We then decompose the cross-entropy loss used during training and highlight the presence of an inherent overfitting term. We propose regularizing the loss by bounding such a term in two equivalent ways: One with a Kullbach-Leibler term, which relates to a PAC-Bayes perspective; the other using the information in the weights as a measure of complexity of a learned model, yielding a novel Information Bottleneck for the weights.

- Information-theoretic analysis of generalization capability of learning algorithms. [[paper]](https://arxiv.org/abs/1705.07809)
  - Aolin Xu, Maxim Raginsky. *NeurIPS 2017*
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> We derive upper bounds on the generalization error of a learning algorithm in terms of the mutual information between its input and output. The bounds provide an information-theoretic understanding of generalization in learning problems, and give theoretical guidelines for striking the right balance between data fit and generalization by controlling the input-output mutual information. We propose a number of methods for this purpose, among which are algorithms that regularize the ERM algorithm with relative entropy or with random noise.

- Opening the Black Box of Deep Neural Networks via Information. [[paper]](https://arxiv.org/abs/1703.00810)
  - Ravid Shwartz-Ziv, Naftali Tishby.
  - Key Word: Information Bottleneck.
  - <details><summary>Digest</summary> [Previous work](https://arxiv.org/abs/1503.02406) proposed to analyze DNNs in the *Information Plane*; i.e., the plane of the Mutual Information values that each layer preserves on the input and output variables. They suggested that the goal of the network is to optimize the Information Bottleneck (IB) tradeoff between compression and prediction, successively, for each layer. In this work we follow up on this idea and demonstrate the effectiveness of the Information-Plane visualization of DNNs.  

## Neural Tangent Kernel

### Neural Tangent Kernel: 2023

- The NTK approximation is valid for longer than you think. [[paper]](https://arxiv.org/abs/2305.13141)
  - Enric Boix-Adsera, Etai Littwin.
  - Key Word: Neural Tangent Kernel Approximation.
  - <details><summary>Digest</summary> We study when the neural tangent kernel (NTK) approximation is valid for training a model with the square loss. In the lazy training setting of Chizat et al. 2019, we show that rescaling the model by a factor of α=O(T) suffices for the NTK approximation to be valid until training time T. Our bound is tight and improves on the previous bound of Chizat et al. 2019, which required a larger rescaling factor of α=O(T2).

- Effective Theory of Transformers at Initialization. [[paper]](https://arxiv.org/abs/2304.02034)
  - Emily Dinan, Sho Yaida, Susan Zhang.
  - Key Word: Transformers; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We perform an effective-theory analysis of forward-backward signal propagation in wide and deep Transformers, i.e., residual neural networks with multi-head self-attention blocks and multilayer perceptron blocks. This analysis suggests particular width scalings of initialization and training hyperparameters for these models. We then take up such suggestions, training Vision and Language Transformers in practical setups.

- Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels. [[paper]](https://arxiv.org/abs/2302.01629)
  - Simone Bombari, Shayan Kiyani, Marco Mondelli.
  - Key Word: Neural Tangent Kernel; Random Feature.
  - <details><summary>Digest</summary> Machine learning models are vulnerable to adversarial perturbations, and a thought-provoking paper by Bubeck and Sellke has analyzed this phenomenon through the lens of over-parameterization: interpolating smoothly the data requires significantly more parameters than simply memorizing it. However, this "universal" law provides only a necessary condition for robustness, and it is unable to discriminate between models. In this paper, we address these gaps by focusing on empirical risk minimization in two prototypical settings, namely, random features and the neural tangent kernel (NTK). 

- Over-parameterised Shallow Neural Networks with Asymmetrical Node Scaling: Global Convergence Guarantees and Feature Learning. [[paper]](https://arxiv.org/abs/2302.01002)
  - Francois Caron, Fadhel Ayed, Paul Jung, Hoil Lee, Juho Lee, Hongseok Yang.
  - Key Word: Neural Tangent Kernel; Feature Learning.
  - <details><summary>Digest</summary> We consider the optimisation of large and shallow neural networks via gradient flow, where the output of each hidden node is scaled by some positive parameter. We focus on the case where the node scalings are non-identical, differing from the classical Neural Tangent Kernel (NTK) parameterisation. We prove that, for large neural networks, with high probability, gradient flow converges to a global minimum AND can learn features, unlike in the NTK regime. 

### Neural Tangent Kernel: 2022

- A Kernel Perspective of Skip Connections in Convolutional Networks. [[paper]](https://arxiv.org/abs/2211.14810)
  - Daniel Barzilai, Amnon Geifman, Meirav Galun, Ronen Basri.
  - Key Word: Neural Tangent Kernel; Gaussian Process; Understanding Skip Connections.
  - <details><summary>Digest</summary> Over-parameterized residual networks (ResNets) are amongst the most successful convolutional neural architectures for image processing. Here we study their properties through their Gaussian Process and Neural Tangent kernels. We derive explicit formulas for these kernels, analyze their spectra, and provide bounds on their implied condition numbers. Our results indicate that (1) with ReLU activation, the eigenvalues of these residual kernels decay polynomially at a similar rate compared to the same kernels when skip connections are not used, thus maintaining a similar frequency bias; (2) however, residual kernels are more locally biased. 

- Transfer Learning with Kernel Methods. [[paper]](https://arxiv.org/abs/2211.00227)
  - Adityanarayanan Radhakrishnan, Max Ruiz Luyten, Neha Prasad, Caroline Uhler. 
  - Key Word: Transfer Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose a transfer learning framework for kernel methods by projecting and translating the source model to the target task. We demonstrate the effectiveness of our framework in applications to image classification and virtual drug screening. In particular, we show that transferring modern kernels trained on large-scale image datasets can result in substantial performance increase as compared to using the same kernel trained directly on the target task.

- Neural Tangent Kernel: A Survey. [[paper]](https://arxiv.org/abs/2208.13614)
  - Eugene Golikov, Eduard Pokonechnyy, Vladimir Korviakov.
  - Key Word: Neural Tangent Kernel; Survey.
  - <details><summary>Digest</summary> A seminal work [Jacot et al., 2018] demonstrated that training a neural network under specific parameterization is equivalent to performing a particular kernel method as width goes to infinity. This equivalence opened a promising direction for applying the results of the rich literature on kernel methods to neural nets which were much harder to tackle. The present survey covers key results on kernel convergence as width goes to infinity, finite-width corrections, applications, and a discussion of the limitations of the corresponding method.

- Limitations of the NTK for Understanding Generalization in Deep Learning. [[paper]](https://arxiv.org/abs/2206.10012)
  - Nikhil Vyas, Yamini Bansal, Preetum Nakkiran.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> In this work, we study NTKs through the lens of scaling laws, and demonstrate that they fall short of explaining important aspects of neural network generalization. In particular, we demonstrate realistic settings where finite-width neural networks have significantly better data scaling exponents as compared to their corresponding empirical and infinite NTKs at initialization. This reveals a more fundamental difference between the real networks and NTKs, beyond just a few percentage points of test accuracy. Further, we show that even if the empirical NTK is allowed to be pre-trained on a constant number of samples, the kernel scaling does not catch up to the neural network scaling. Finally, we show that the empirical NTK continues to evolve throughout most of the training, in contrast with prior work which suggests that it stabilizes after a few epochs of training. Altogether, our work establishes concrete limitations of the NTK approach in understanding generalization of real networks on natural datasets.

- Fast Finite Width Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2206.08720) [[code]](https://github.com/google/neural-tangents)
  - Roman Novak, Jascha Sohl-Dickstein, Samuel S. Schoenholz. *ICML 2022*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> In the infinite width limit, the NTK can sometimes be computed analytically and is useful for understanding training and generalization of NN architectures. At finite widths, the NTK is also used to better initialize NNs, compare the conditioning across models, perform architecture search, and do meta-learning. Unfortunately, the finite width NTK is notoriously expensive to compute, which severely limits its practical utility. We perform the first in-depth analysis of the compute and memory requirements for NTK computation in finite width networks. Leveraging the structure of neural networks, we further propose two novel algorithms that change the exponent of the compute and memory requirements of the finite width NTK, dramatically improving efficiency.

- On the Generalization Power of the Overfitted Three-Layer Neural Tangent Kernel Model. [[paper]](https://arxiv.org/abs/2206.02047)
  - Peizhong Ju, Xiaojun Lin, Ness B. Shroff.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We study the generalization performance of overparameterized 3-layer NTK models. We show that, for a specific set of ground-truth functions (which we refer to as the "learnable set"), the test error of the overfitted 3-layer NTK is upper bounded by an expression that decreases with the number of neurons of the two hidden layers. Different from 2-layer NTK where there exists only one hidden-layer, the 3-layer NTK involves interactions between two hidden-layers. Our upper bound reveals that, between the two hidden-layers, the test error descends faster with respect to the number of neurons in the second hidden-layer (the one closer to the output) than with respect to that in the first hidden-layer (the one closer to the input).

- Self-Consistent Dynamical Field Theory of Kernel Evolution in Wide Neural Networks. [[paper]](https://arxiv.org/abs/2205.09653)
  - Blake Bordelon, Cengiz Pehlevan.
  - Key Word: Neural Tangent Kernel; Mean Field Theory.
  - <details><summary>Digest</summary> We analyze feature learning in infinite width neural networks trained with gradient flow through a self-consistent dynamical field theory. We construct a collection of deterministic dynamical order parameters which are inner-product kernels for hidden unit activations and gradients in each layer at pairs of time points, providing a reduced description of network activity through training.

### Neural Tangent Kernel: 2021

- Neural Tangent Generalization Attacks. [[paper]](https://proceedings.mlr.press/v139/yuan21b.html) [[code]](https://github.com/lionelmessi6410/ntga)
  - Chia-Hung Yuan, Shan-Hung Wu. *ICML 2021*
  - Key Word: Neural Tangent Kernel; Poisoning Attacks.
  - <details><summary>Digest</summary> We study the generalization attacks against DNNs, where an attacker aims to slightly modify training data in order to spoil the training process such that a trained network lacks generalizability. These attacks can be performed by data owners and protect data from unexpected use. However, there is currently no efficient generalization attack against DNNs due to the complexity of a bilevel optimization involved. We propose the Neural Tangent Generalization Attack (NTGA) that, to the best of our knowledge, is the first work enabling clean-label, black-box generalization attack against DNNs.

- On the Equivalence between Neural Network and Support Vector Machine. [[paper]](https://arxiv.org/abs/2111.06063) [[code]](https://github.com/leslie-ch/equiv-nn-svm)
  - Yilan Chen, Wei Huang, Lam M. Nguyen, Tsui-Wei Weng. *NeurIPS 2021*
  - Key Word: Neural Tangent Kernel; Support Vector Machine.
  - <details><summary>Digest</summary> We prove the equivalence between neural network (NN) and support vector machine (SVM), specifically, the infinitely wide NN trained by soft margin loss and the standard soft margin SVM with NTK trained by subgradient descent. Our main theoretical results include establishing the equivalence between NN and a broad family of L2 regularized kernel machines (KMs) with finite-width bounds, which cannot be handled by prior work, and showing that every finite-width NN trained by such regularized loss functions is approximately a KM.

- An Empirical Study of Neural Kernel Bandits. [[paper]](https://arxiv.org/abs/2111.03543) [[code]](https://github.com/mlisicki/neuralkernelbandits)
  - Michal Lisicki, Arash Afkanpour, Graham W. Taylor.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose to directly apply NK-induced distributions to guide an upper confidence bound or Thompson sampling-based policy. We show that NK bandits achieve state-of-the-art performance on highly non-linear structured data. Furthermore, we analyze practical considerations such as training frequency and model partitioning.

- A Neural Tangent Kernel Perspective of GANs. [[paper]](https://arxiv.org/abs/2106.05566) [[code]](https://github.com/emited/gantk2)
  - Jean-Yves Franceschi, Emmanuel de Bézenac, Ibrahim Ayed, Mickaël Chen, Sylvain Lamprier, Patrick Gallinari. *ICML 2021*
  - Key Word: Neural Tangent Kernel; Generative Adversarial Networks.
  - <details><summary>Digest</summary> We propose a novel theoretical framework of analysis for Generative Adversarial Networks (GANs). We start by pointing out a fundamental flaw in previous theoretical analyses that leads to ill-defined gradients for the discriminator. We overcome this issue which impedes a principled study of GAN training, solving it within our framework by taking into account the discriminator's architecture. To this end, we leverage the theory of infinite-width neural networks for the discriminator via its Neural Tangent Kernel. We characterize the trained discriminator for a wide range of losses and establish general differentiability properties of the network.

- Reverse Engineering the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2106.03186) [[code]](https://github.com/james-simon/shallow-learning)
  - James B. Simon, Sajant Anand, Michael R. DeWeese.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> The development of methods to guide the design of neural networks is an important open challenge for deep learning theory. As a paradigm for principled neural architecture design, we propose the translation of high-performing kernels, which are better-understood and amenable to first-principles design, into equivalent network architectures, which have superior efficiency, flexibility, and feature learning. To this end, we constructively prove that, with just an appropriate choice of activation function, any positive-semidefinite dot-product kernel can be realized as either the conjugate or neural tangent kernel of a fully-connected neural network with only one hidden layer.

- Out-of-Distribution Generalization in Kernel Regression. [[paper]](https://arxiv.org/abs/2106.02261) [[code]](https://github.com/pehlevan-group/kernel-ood-generalization)
  - Abdulkadir Canatar, Blake Bordelon, Cengiz Pehlevan. *NeurIPS 2021*
  - Key Word: Out-of-Distribution Generalization; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We study generalization in kernel regression when the training and test distributions are different using methods from statistical physics. Using the replica method, we derive an analytical formula for the out-of-distribution generalization error applicable to any kernel and real datasets. We identify an overlap matrix that quantifies the mismatch between distributions for a given kernel as a key determinant of generalization performance under distribution shift.

- FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis. [[paper]](https://arxiv.org/abs/2105.05001)
  - Baihe Huang, Xiaoxiao Li, Zhao Song, Xin Yang. *ICML 2021*
  - Key Word: Federated Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> This paper presents a new class of convergence analysis for FL, Federated Learning Neural Tangent Kernel (FL-NTK), which corresponds to overparamterized ReLU neural networks trained by gradient descent in FL and is inspired by the analysis in Neural Tangent Kernel (NTK). Theoretically, FL-NTK converges to a global-optimal solution at a linear rate with properly tuned learning parameters. Furthermore, with proper distributional assumptions, FL-NTK can also achieve good generalization.

- Random Features for the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2104.01351)
  - Insu Han, Haim Avron, Neta Shoham, Chaewon Kim, Jinwoo Shin. 
  - Key Word: Neural Tangent Kernel; Random Features.
  - <details><summary>Digest</summary> We propose an efficient feature map construction of the NTK of fully-connected ReLU network which enables us to apply it to large-scale datasets. We combine random features of the arc-cosine kernels with a sketching-based algorithm which can run in linear with respect to both the number of data points and input dimension. We show that dimension of the resulting features is much smaller than other baseline feature map constructions to achieve comparable error bounds both in theory and practice. 

- Explaining Neural Scaling Laws. [[paper]](https://arxiv.org/abs/2102.06701) [[code]](https://github.com/google/neural-tangents)
  - Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma. *ICLR 2022*
  - Key Word: Scaling Laws; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose a theory that explains and connects these scaling laws. We identify variance-limited and resolution-limited scaling behavior for both dataset and model size, for a total of four scaling regimes. The variance-limited scaling follows simply from the existence of a well-behaved infinite data or infinite width limit, while the resolution-limited regime can be explained by positing that models are effectively resolving a smooth data manifold.

### Neural Tangent Kernel: 2020

- Mathematical Models of Overparameterized Neural Networks. [[paper]](https://arxiv.org/abs/2012.13982) [[code]](https://github.com/hendrydong/NTK-and-MF-examples)\
  - Cong Fang, Hanze Dong, Tong Zhang. *Proceedings of the IEEE*
  - Key Word: Neural Tangent Kernel; Mean-Field Theory.
  - <details><summary>Digest</summary> Known by practitioners that overparameterized neural networks are easy to learn, in the past few years there have been important theoretical developments in the analysis of overparameterized neural networks. In particular, it was shown that such systems behave like convex systems under various restricted settings, such as for two-layer NNs, and when learning is restricted locally in the so-called neural tangent kernel space around specialized initializations. This paper discusses some of these recent progresses leading to significant better understanding of neural networks. We will focus on the analysis of two-layer neural networks, and explain the key mathematical models, with their algorithmic implications.

- Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2010.15110)
  - Stanislav Fort, Gintare Karolina Dziugaite, Mansheej Paul, Sepideh Kharaghani, Daniel M. Roy, Surya Ganguli. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> In suitably initialized wide networks, small learning rates transform deep neural networks (DNNs) into neural tangent kernel (NTK) machines, whose training dynamics is well-approximated by a linear weight expansion of the network at initialization. Standard training, however, diverges from its linearization in ways that are poorly understood. We study the relationship between the training dynamics of nonlinear deep networks, the geometry of the loss landscape, and the time evolution of a data-dependent NTK.

- Increasing Depth Leads to U-Shaped Test Risk in Over-parameterized Convolutional Networks. [[paper]](https://arxiv.org/abs/2010.09610)
  - Eshaan Nichani, Adityanarayanan Radhakrishnan, Caroline Uhler.
  - Key Word: Convolutional Neural Tangent Kernel.
  - <details><summary>Digest</summary> We demonstrate that the test risk of over-parameterized convolutional networks is a U-shaped curve (i.e. monotonically decreasing, then increasing) with increasing depth. We first provide empirical evidence for this phenomenon via image classification experiments using both ResNets and the convolutional neural tangent kernel (CNTK). We then present a novel linear regression framework for characterizing the impact of depth on test risk, and show that increasing depth leads to a U-shaped test risk for the linear CNTK.

- Finite Versus Infinite Neural Networks: an Empirical Study. [[paper]](https://arxiv.org/abs/2007.15801) [[code]](https://github.com/google/neural-tangents)
  - Jaehoon Lee, Samuel S. Schoenholz, Jeffrey Pennington, Ben Adlam, Lechao Xiao, Roman Novak, Jascha Sohl-Dickstein. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We perform a careful, thorough, and large scale empirical study of the correspondence between wide neural networks and kernel methods. By doing so, we resolve a variety of open questions related to the study of infinitely wide neural networks. Our experimental results include: kernel methods outperform fully-connected finite-width networks, but underperform convolutional finite width networks; neural network Gaussian process (NNGP) kernels frequently outperform neural tangent (NT) kernels; centered and ensembled finite networks have reduced posterior variance and behave more similarly to infinite networks; weight decay and the use of a large learning rate break the correspondence between finite and infinite networks; the NTK parameterization outperforms the standard parameterization for finite width networks; diagonal regularization of kernels acts similarly to early stopping; floating point precision limits kernel performance beyond a critical dataset size; regularized ZCA whitening improves accuracy; finite network performance depends non-monotonically on width in ways not captured by double descent phenomena; equivariance of CNNs is only beneficial for narrow networks far from the kernel regime.

- Bayesian Deep Ensembles via the Neural Tangent Kernel. [[paper]](https://arxiv.org/abs/2007.05864) [[code]](https://github.com/bobby-he/bayesian-ntk)
  - Bobby He, Balaji Lakshminarayanan, Yee Whye Teh.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We explore the link between deep ensembles and Gaussian processes (GPs) through the lens of the Neural Tangent Kernel (NTK): a recent development in understanding the training dynamics of wide neural networks (NNs). Previous work has shown that even in the infinite width limit, when NNs become GPs, there is no GP posterior interpretation to a deep ensemble trained with squared error loss. We introduce a simple modification to standard deep ensembles training, through addition of a computationally-tractable, randomised and untrainable function to each ensemble member, that enables a posterior interpretation in the infinite width limit.

- The Surprising Simplicity of the Early-Time Learning Dynamics of Neural Networks. [[paper]](https://arxiv.org/abs/2006.14599)
  - Wei Hu, Lechao Xiao, Ben Adlam, Jeffrey Pennington. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We show that these common perceptions can be completely false in the early phase of learning. In particular, we formally prove that, for a class of well-behaved input distributions, the early-time learning dynamics of a two-layer fully-connected neural network can be mimicked by training a simple linear model on the inputs.

- When Do Neural Networks Outperform Kernel Methods? [[paper]](https://arxiv.org/abs/2006.13409) [[code]](https://github.com/bGhorbani/linearized_neural_networks)
  - Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, Andrea Montanari. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> How can we reconcile the above claims? For which tasks do NNs outperform RKHS? If covariates are nearly isotropic, RKHS methods suffer from the curse of dimensionality, while NNs can overcome it by learning the best low-dimensional representation. Here we show that this curse of dimensionality becomes milder if the covariates display the same low-dimensional structure as the target function, and we precisely characterize this tradeoff. Building on these results, we present the spiked covariates model that can capture in a unified framework both behaviors observed in earlier work.

- A Generalized Neural Tangent Kernel Analysis for Two-layer Neural Networks. [[paper]](https://arxiv.org/abs/2002.04026)
  - Zixiang Chen, Yuan Cao, Quanquan Gu, Tong Zhang. *NeurIPS 2020*
  - Key Word: Neural Tangent Kernel; Mean Field Theory.
  - <details><summary>Digest</summary> We provide a generalized neural tangent kernel analysis and show that noisy gradient descent with weight decay can still exhibit a "kernel-like" behavior. This implies that the training loss converges linearly up to a certain accuracy. We also establish a novel generalization error bound for two-layer neural networks trained by noisy gradient descent with weight decay.

### Neural Tangent Kernel: 2019

- Disentangling Trainability and Generalization in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1912.13053)
  - Lechao Xiao, Jeffrey Pennington, Samuel S. Schoenholz. *ICML 2020*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We provide such a characterization in the limit of very wide and very deep networks, for which the analysis simplifies considerably. For wide networks, the trajectory under gradient descent is governed by the Neural Tangent Kernel (NTK), and for deep networks the NTK itself maintains only weak data dependence.

- Simple and Effective Regularization Methods for Training on Noisily Labeled Data with Generalization Guarantee. [[paper]](https://arxiv.org/abs/1905.11368)
  - Wei Hu, Zhiyuan Li, Dingli Yu. *ICLR 2020*
  - Key Word: Neural Tangent Kernel; Regularization.
  - <details><summary>Digest</summary> This paper proposes and analyzes two simple and intuitive regularization methods: (i) regularization by the distance between the network parameters to initialization, and (ii) adding a trainable auxiliary variable to the network output for each training example. Theoretically, we prove that gradient descent training with either of these two methods leads to a generalization guarantee on the clean data distribution despite being trained using noisy labels.

- On Exact Computation with an Infinitely Wide Neural Net. [[paper]](https://arxiv.org/abs/1904.11955) [[code]](https://github.com/ruosongwang/CNTK)
  - Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, Ruosong Wang. *NeurIPS 2019*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> The current paper gives the first efficient exact algorithm for computing the extension of NTK to convolutional neural nets, which we call Convolutional NTK (CNTK), as well as an efficient GPU implementation of this algorithm.

- Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation. [[paper]](https://arxiv.org/abs/1902.04760)
  - Greg Yang.
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> Several recent trends in machine learning theory and practice, from the design of state-of-the-art Gaussian Process to the convergence analysis of deep neural nets (DNNs) under stochastic gradient descent (SGD), have found it fruitful to study wide random neural networks. Central to these approaches are certain scaling limits of such networks. We unify these results by introducing a notion of a straightline \emph{tensor program} that can express most neural network computations, and we characterize its scaling limit when its tensors are large and randomized.

### Neural Tangent Kernel: 2018

- A Convergence Theory for Deep Learning via Over-Parameterization. [[paper]](https://arxiv.org/abs/1811.03962)
  - Zeyuan Allen-Zhu, Yuanzhi Li, Zhao Song. *ICML 2019*
  - Key Word: Stochastic Gradient Descent; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We prove why stochastic gradient descent (SGD) can find global minima on the training objective of DNNs in polynomial time. We only make two assumptions: the inputs are non-degenerate and the network is over-parameterized. The latter means the network width is sufficiently large: polynomial in L, the number of layers and in n, the number of samples. Our key technique is to derive that, in a sufficiently large neighborhood of the random initialization, the optimization landscape is almost-convex and semi-smooth even with ReLU activations. This implies an equivalence between over-parameterized neural networks and neural tangent kernel (NTK) in the finite (and polynomial) width setting.

- Neural Tangent Kernel: Convergence and Generalization in Neural Networks. [[paper]](https://arxiv.org/abs/1806.07572)
  - Arthur Jacot, Franck Gabriel, Clément Hongler. *NeurIPS 2018*
  - Key Word: Neural Tangent Kernel.
  - <details><summary>Digest</summary> We prove that the evolution of an ANN during training can also be described by a kernel: during gradient descent on the parameters of an ANN, the network function (which maps input vectors to output vectors) follows the kernel gradient of the functional cost (which is convex, in contrast to the parameter cost) w.r.t. a new kernel: the Neural Tangent Kernel (NTK).  

## Others

### Others: 2023

- Learning Theory from First Principles. [[paper]](https://www.di.ens.fr/~fbach/ltfp/)
  - Francis Bach.
  - Key Word: Learning Theory; Book.
  - <details><summary>Digest</summary> The goal of the class (and thus of this textbook) is to present old and recent results in learning theory for the most widely-used learning architectures. This class is geared towards theory-oriented students as well as students who want to acquire a basic mathematical understanding of algorithms used throughout machine learning and associated fields that are significant users of learning methods such as computer vision or natural language processing.

- Most Neural Networks Are Almost Learnable. [[paper]](https://arxiv.org/abs/2305.16508)
  - Amit Daniely, Nathan Srebro, Gal Vardi.
  - Key Word: Neural Network Learnability.
  - <details><summary>Digest</summary> They assume that the network’s weights are initialized randomly using a standard scheme and that the input distribution is uniform on a sphere. They show that random networks with Lipschitz activation functions can be approximated by low-degree polynomials, and use this to derive a polynomial-time approximation scheme (PTAS) for learning them. They also show that for sigmoid and ReLU-like activation functions, the PTAS can be improved to a quasi-polynomial-time algorithm. They support their theory with experiments on three network architectures and three datasets.

- The Crucial Role of Normalization in Sharpness-Aware Minimization. [[paper]](https://arxiv.org/abs/2305.15287)
  - Yan Dai, Kwangjun Ahn, Suvrit Sra.
  - Key Word: Sharpness-Awareness Minimization; Normalization.
  - <details><summary>Digest</summary> Sharpness-Aware Minimization (SAM) is a recently proposed gradient-based optimizer that greatly improves the prediction performance of deep neural networks. There has been a surge of interest in explaining its empirical success. We focus on understanding the role played by normalization, a key component of the SAM updates. We study the effect of normalization in SAM for both convex and non-convex functions, revealing two key roles played by normalization. These two properties of normalization make SAM robust against the choice of hyper-parameters, supporting the practicality of SAM.

- From Tempered to Benign Overfitting in ReLU Neural Networks. [[paper]](https://arxiv.org/abs/2305.15141)
  - Guy Kornowski, Gilad Yehudai, Ohad Shamir.
  - Key Word: Overparameterized neural networks; Benign overfitting; Tempered overfitting.
  - <details><summary>Digest</summary> Overparameterized neural networks (NNs) are observed to generalize well even when trained to perfectly fit noisy data. This phenomenon motivated a large body of work on “benign overfitting”, where interpolating predictors achieve near-optimal performance. Recently, it was conjectured and empirically observed that the behavior of NNs is often better described as “tempered overfitting”. In this work, we provide several results that aim at bridging these complementing views. We study a simple classification setting with 2-layer ReLU NNs, and prove that under various assumptions, the type of overfitting transitions from tempered in the extreme case of one-dimensional data, to benign in high dimensions.

- When are ensembles really effective? [[paper]](https://arxiv.org/abs/2305.12313)
  - Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, Michael W. Mahoney.
  - Key Word: Ensemble; Disagreement-Error Ratio.
  - <details><summary>Digest</summary> Ensembling is a machine learning technique that combines multiple models to improve the overall performance. Ensembling has a long history in statistical data analysis, but its benefits are not always obvious in modern machine learning settings. We study the fundamental question of when ensembling yields significant performance improvements in classification tasks. We prove new results relating the ensemble improvement rate to the disagreement-error ratio. We show that ensembling improves performance significantly whenever the disagreement rate is large relative to the average error rate. 

- The Hessian perspective into the Nature of Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/2305.09088)
  - Sidak Pal Singh, Thomas Hofmann, Bernhard Schölkopf. *ICML 2023*
  - Key Word: Hessian Maps; Convolutional Neural Networks.
  - <details><summary>Digest</summary> We provide a novel perspective on Convolutional Neural Networks (CNNs) by studying their Hessian maps, which capture parameter interactions. Using a Toeplitz representation framework, we reveal the Hessian structure and establish tight upper bounds on its rank. Our findings show that the Hessian rank in CNNs grows as the square root of the number of parameters, challenging previous assumptions.

- Model-agnostic Measure of Generalization Difficulty. [[paper]](https://arxiv.org/abs/2305.01034) [[code]](https://github.com/fietelab/inductive-bias-complexity)
  - Akhilan Boopathy, Kevin Liu, Jaedong Hwang, Shu Ge, Asaad Mohammedsaleh, Ila Fiete. *ICML 2023*
  - Key Word: Generalization Difficulty; Information Content of Inductive Biases.
  - <details><summary>Digest</summary> The measure of a machine learning algorithm is the difficulty of the tasks it can perform, and sufficiently difficult tasks are critical drivers of strong machine learning models. However, quantifying the generalization difficulty of machine learning benchmarks has remained challenging. We propose what is to our knowledge the first model-agnostic measure of the inherent generalization difficulty of tasks. Our inductive bias complexity measure quantifies the total information required to generalize well on a task minus the information provided by the data. 

- Wasserstein PAC-Bayes Learning: A Bridge Between Generalisation and Optimisation. [[paper]](https://arxiv.org/abs/2304.07048)
  - Maxime Haddouche, Benjamin Guedj.
  - Key Word: PAC-Bayes Bound; Wasserstein Distances.
  - <details><summary>Digest</summary> PAC-Bayes learning is an established framework to assess the generalisation ability of learning algorithm during the training phase. However, it remains challenging to know whether PAC-Bayes is useful to understand, before training, why the output of well-known algorithms generalise well. We positively answer this question by expanding the Wasserstein PAC-Bayes framework, briefly introduced in \cite{amit2022ipm}. We provide new generalisation bounds exploiting geometric assumptions on the loss function. Using our framework, we prove, before any training, that the output of an algorithm from \citet{lambert2022variational} has a strong asymptotic generalisation ability. More precisely, we show that it is possible to incorporate optimisation results within a generalisation framework, building a bridge between PAC-Bayes and optimisation algorithms.

- Do deep neural networks have an inbuilt Occam's razor? [[paper]](https://arxiv.org/abs/2304.06670)
  - Key Word: Kolmogorov Complexity; Inductive Bias; Occam’s Razor; No Free Lunch Theorems.
  - <details><summary>Digest</summary> The remarkable performance of overparameterized deep neural networks (DNNs) must arise from an interplay between network architecture, training algorithms, and structure in the data. To disentangle these three components, we apply a Bayesian picture, based on the functions expressed by a DNN, to supervised learning. The prior over functions is determined by the network, and is varied by exploiting a transition between ordered and chaotic regimes. For Boolean function classification, we approximate the likelihood using the error spectrum of functions on data. When combined with the prior, this accurately predicts the posterior, measured for DNNs trained with stochastic gradient descent. This analysis reveals that structured data, combined with an intrinsic Occam's razor-like inductive bias towards (Kolmogorov) simple functions that is strong enough to counteract the exponential growth of the number of functions with complexity, is a key to the success of DNNs.

- The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning. [[paper]](https://arxiv.org/abs/2304.05366)
  - Micah Goldblum, Marc Finzi, Keefer Rowan, Andrew Gordon Wilson.
  - Key Word: No Free Lunch Theorem; Kolmogorov Complexity; Model Selection.
  - <details><summary>Digest</summary> No free lunch theorems for supervised learning state that no learner can solve all problems or that all learners achieve exactly the same accuracy on average over a uniform distribution on learning problems. Accordingly, these theorems are often referenced in support of the notion that individual problems require specially tailored inductive biases. While virtually all uniformly sampled datasets have high complexity, real-world problems disproportionately generate low-complexity data, and we argue that neural network models share this same preference, formalized using Kolmogorov complexity. Notably, we show that architectures designed for a particular domain, such as computer vision, can compress datasets on a variety of seemingly unrelated domains. 

- The Benefits of Mixup for Feature Learning. [[paper]](https://arxiv.org/abs/2303.08433)
  - Difan Zou, Yuan Cao, Yuanzhi Li, Quanquan Gu.
  - Key Word: Mixup; Data Augmentation; Feature Learning.
  - <details><summary>Digest</summary> We first show that Mixup using different linear interpolation parameters for features and labels can still achieve similar performance to the standard Mixup. This indicates that the intuitive linearity explanation in Zhang et al., (2018) may not fully explain the success of Mixup. Then we perform a theoretical study of Mixup from the feature learning perspective. We consider a feature-noise data model and show that Mixup training can effectively learn the rare features (appearing in a small fraction of data) from its mixture with the common features (appearing in a large fraction of data). In contrast, standard training can only learn the common features but fails to learn the rare features, thus suffering from bad generalization performance. Moreover, our theoretical analysis also shows that the benefits of Mixup for feature learning are mostly gained in the early training phase, based on which we propose to apply early stopping in Mixup.

- Bayes Complexity of Learners vs Overfitting. [[paper]](https://arxiv.org/abs/2303.07874)
  - Grzegorz Głuch, Rudiger Urbanke.
  - Key Word: PAC-Bayes; Bayes Complexity; Overfitting.
  - <details><summary>Digest</summary> We introduce a new notion of complexity of functions and we show that it has the following properties: (i) it governs a PAC Bayes-like generalization bound, (ii) for neural networks it relates to natural notions of complexity of functions (such as the variation), and (iii) it explains the generalization gap between neural networks and linear schemes. 

- Benign Overfitting in Linear Classifiers and Leaky ReLU Networks from KKT Conditions for Margin Maximization. [[paper]](https://arxiv.org/abs/2303.01462)
  - Spencer Frei, Gal Vardi, Peter L. Bartlett, Nathan Srebro.
  - Key Word: Benign Overfitting; Implicit Bias.
  - <details><summary>Digest</summary> Linear classifiers and leaky ReLU networks trained by gradient flow on the logistic loss have an implicit bias towards solutions which satisfy the Karush--Kuhn--Tucker (KKT) conditions for margin maximization. In this work we establish a number of settings where the satisfaction of these KKT conditions implies benign overfitting in linear classifiers and in two-layer leaky ReLU networks: the estimators interpolate noisy training data and simultaneously generalize well to test data.

- The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks. [[paper]](https://arxiv.org/abs/2303.01456)
  - Spencer Frei, Gal Vardi, Peter L. Bartlett, Nathan Srebro.
  - Key Word: Implicit Bias; Adversarial Robustness.
  - <details><summary>Digest</summary> In this work, we study the implications of the implicit bias of gradient flow on generalization and adversarial robustness in ReLU networks. We focus on a setting where the data consists of clusters and the correlations between cluster means are small, and show that in two-layer ReLU networks gradient flow is biased towards solutions that generalize well, but are highly vulnerable to adversarial examples. Our results hold even in cases where the network has many more parameters than training examples.

- Why (and When) does Local SGD Generalize Better than SGD? [[paper]](https://arxiv.org/abs/2303.01215)
  - Xinran Gu, Kaifeng Lyu, Longbo Huang, Sanjeev Arora. *ICLR 2023*
  - Key Word: Local Stochastic Gradient Descent; Stochastic Differential Equations.
  - <details><summary>Digest</summary> This paper aims to understand why (and when) Local SGD generalizes better based on Stochastic Differential Equation (SDE) approximation. The main contributions of this paper include (i) the derivation of an SDE that captures the long-term behavior of Local SGD in the small learning rate regime, showing how noise drives the iterate to drift and diffuse after it has reached close to the manifold of local minima, (ii) a comparison between the SDEs of Local SGD and SGD, showing that Local SGD induces a stronger drift term that can result in a stronger effect of regularization, e.g., a faster reduction of sharpness, and (iii) empirical evidence validating that having a small learning rate and long enough training time enables the generalization improvement over SGD but removing either of the two conditions leads to no improvement.

- Hiding Data Helps: On the Benefits of Masking for Sparse Coding. [[paper]](https://arxiv.org/abs/2302.12715)
  - Muthu Chidambaram, Chenwei Wu, Yu Cheng, Rong Ge.
  - Key Word: Sparse Coding; Self-Supervised Learning.
  - <details><summary>Digest</summary> We show that for over-realized sparse coding in the presence of noise, minimizing the standard dictionary learning objective can fail to recover the ground-truth dictionary, regardless of the magnitude of the signal in the data-generating process. Furthermore, drawing from the growing body of work on self-supervised learning, we propose a novel masking objective and we prove that minimizing this new objective can recover the ground-truth dictionary. 

- Phase diagram of training dynamics in deep neural networks: effect of learning rate, depth, and width. [[paper]](https://arxiv.org/abs/2302.12250)
  - Dayal Singh Kalra, Maissam Barkeshli.
  - Key Word: Sharpness; Neural Tangent Kernel.
  - <details><summary>Digest</summary> By analyzing the maximum eigenvalue λHt of the Hessian of the loss, which is a measure of sharpness of the loss landscape, we find that the dynamics can show four distinct regimes: (i) an early time transient regime, (ii) an intermediate saturation regime, (iii) a progressive sharpening regime, and finally (iv) a late time ``edge of stability" regime. 

- Sharpness-Aware Minimization: An Implicit Regularization Perspective. [[paper]](https://arxiv.org/abs/2302.11836)
  - Kayhan Behdin, Rahul Mazumder.
  - Key Word: Sharpness-Aware Minimization; Implicit Regularization.
  - <details><summary>Digest</summary> We study SAM through an implicit regularization lens, and present a new theoretical explanation of why SAM generalizes well. To this end, we study the least-squares linear regression problem and show a bias-variance trade-off for SAM's error over the course of the algorithm. We show SAM has lower bias compared to Gradient Descent (GD), while having higher variance.

- Modular Deep Learning. [[paper]](https://arxiv.org/abs/2302.11529)
  - Jonas Pfeiffer, Sebastian Ruder, Ivan Vulić, Edoardo Maria Ponti.
  - Key Word: Parameter-Efficient Fine-Tuning; Mixture-of-Expert; Rounting; Model Aggregation.
  - <details><summary>Digest</summary> Modular deep learning has emerged as a promising solution to these challenges. In this framework, units of computation are often implemented as autonomous parameter-efficient modules. Information is conditionally routed to a subset of modules and subsequently aggregated. These properties enable positive transfer and systematic generalisation by separating computation from routing and updating modules locally. We offer a survey of modular architectures, providing a unified view over several threads of research that evolved independently in the scientific literature. 

- mSAM: Micro-Batch-Averaged Sharpness-Aware Minimization. [[paper]](https://arxiv.org/abs/2302.09693)
  - Key Word: Sharpness-Aware Minimization.
  - <details><summary>Digest</summary> We focus on a variant of SAM known as micro-batch SAM (mSAM), which, during training, averages the updates generated by adversarial perturbations across several disjoint shards (micro batches) of a mini-batch. We extend a recently developed and well-studied general framework for flatness analysis to show that distributed gradient computation for sharpness-aware minimization theoretically achieves even flatter minima. 

- Machine Love. [[paper]](https://arxiv.org/abs/2302.09248)
  - Key Word: Maslow’s Gridworld; Psychology.
  - <details><summary>Digest</summary> While ML generates much economic value, many of us have problematic relationships with social media and other ML-powered applications. One reason is that ML often optimizes for what we want in the moment, which is easy to quantify but at odds with what is known scientifically about human flourishing. Thus, through its impoverished models of us, ML currently falls far short of its exciting potential, which is for it to help us to reach ours. While there is no consensus on defining human flourishing, from diverse perspectives across psychology, philosophy, and spiritual traditions, love is understood to be one of its primary catalysts. Motivated by this view, this paper explores whether there is a useful conception of love fitting for machines to embody, as historically it has been generative to explore whether a nebulous concept, such as life or intelligence, can be thoughtfully abstracted and reimagined, as in the fields of machine intelligence or artificial life. 

- PAC-Bayesian Generalization Bounds for Adversarial Generative Models. [[paper]](https://arxiv.org/abs/2302.08942)
  - Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain.
  - Key Word: PAC-Bayes; Generative Model Generalization Bound.
  - <details><summary>Digest</summary> We extend PAC-Bayesian theory to generative models and develop generalization bounds for models based on the Wasserstein distance and the total variation distance. Our first result on the Wasserstein distance assumes the instance space is bounded, while our second result takes advantage of dimensionality reduction. Our results naturally apply to Wasserstein GANs and Energy-Based GANs, and our bounds provide new training objectives for these two. 

- SAM operates far from home: eigenvalue regularization as a dynamical phenomenon. [[paper]](https://arxiv.org/abs/2302.08692)
  - Atish Agarwala, Yann N. Dauphin.
  - Key Word: Sharpness-Aware Minimization.
  - <details><summary>Digest</summary> Our work reveals that SAM provides a strong regularization of the eigenvalues throughout the learning trajectory. We show that in a simplified setting, SAM dynamically induces a stabilization related to the edge of stability (EOS) phenomenon observed in large learning rate gradient descent. Our theory predicts the largest eigenvalue as a function of the learning rate and SAM radius parameters. 

- Interpolation Learning With Minimum Description Length. [[paper]](https://arxiv.org/abs/2302.07263)
  - Naren Sarayu Manoj, Nathan Srebro.
  - Key Word: Minimum Description Length; Benign Overfitting; Tempered Overfitting.
  - <details><summary>Digest</summary> We prove that the Minimum Description Length learning rule exhibits tempered overfitting. We obtain tempered agnostic finite sample learning guarantees and characterize the asymptotic behavior in the presence of random label noise.

- A modern look at the relationship between sharpness and generalization. [[paper]](https://arxiv.org/abs/2302.07011)
  - Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, Nicolas Flammarion.
  - Key Word: Sharpness; Generalization.
  - <details><summary>Digest</summary> We comprehensively explore this question in a detailed study of various definitions of adaptive sharpness in settings ranging from training from scratch on ImageNet and CIFAR-10 to fine-tuning CLIP on ImageNet and BERT on MNLI. We focus mostly on transformers for which little is known in terms of sharpness despite their widespread usage. Overall, we observe that sharpness does not correlate well with generalization but rather with some training parameters like the learning rate that can be positively or negatively correlated with generalization depending on the setup.

- Data pruning and neural scaling laws: fundamental limitations of score-based algorithms. [[paper]](https://arxiv.org/abs/2302.06960)
  - Fadhel Ayed, Soufiane Hayou.
  - Key Word: Data Pruning; Neural Scaling Laws.
  - <details><summary>Digest</summary> In this work, we focus on score-based data pruning algorithms and show theoretically and empirically why such algorithms fail in the high compression regime. We demonstrate ``No Free Lunch" theorems for data pruning and present calibration protocols that enhance the performance of existing pruning algorithms in this high compression regime using randomization.

- A Theoretical Understanding of shallow Vision Transformers: Learning, Generalization, and Sample Complexity. [[paper]](https://arxiv.org/abs/2302.06015)
  - Hongkang Li, Meng Wang, Sijia Liu, Pin-yu Chen. *ICLR 2023*
  - Key Word: Vision Transformer; Token Sparsification; Sample Complexity Bound.
  - <details><summary>Digest</summary> Based on a data model characterizing both label-relevant and label-irrelevant tokens, this paper provides the first theoretical analysis of training a shallow ViT, i.e., one self-attention layer followed by a two-layer perceptron, for a classification task. We characterize the sample complexity to achieve a zero generalization error. Our sample complexity bound is positively correlated with the inverse of the fraction of label-relevant tokens, the token noise level, and the initial model error. 

- Tighter PAC-Bayes Bounds Through Coin-Betting. [[paper]](https://arxiv.org/abs/2302.05829)
  - Kyoungseok Jang, Kwang-Sung Jun, Ilja Kuzborskij, Francesco Orabona.
  - Key Word: PAC-Bayes Bounds.
  - <details><summary>Digest</summary> Recently, the PAC-Bayes framework has been proposed as a better alternative for this class of problems for its ability to often give numerically non-vacuous bounds. In this paper, we show that we can do even better: we show how to refine the proof strategy of the PAC-Bayes bounds and achieve even tighter guarantees. Our approach is based on the coin-betting framework that derives the numerically tightest known time-uniform concentration inequalities from the regret guarantees of online gambling algorithms. 

- A unified recipe for deriving (time-uniform) PAC-Bayes bounds. [[paper]](https://arxiv.org/abs/2302.03421)
  - Ben Chugg, Hongjian Wang, Aaditya Ramdas.
  - Key Word: PAC-Bayes Bounds.
  - <details><summary>Digest</summary> We present a unified framework for deriving PAC-Bayesian generalization bounds. Unlike most previous literature on this topic, our bounds are anytime-valid (i.e., time-uniform), meaning that they hold at all stopping times, not only for a fixed sample size. Our approach combines four tools in the following order: (a) nonnegative supermartingales or reverse submartingales, (b) the method of mixtures, (c) the Donsker-Varadhan formula (or other convex duality principles), and (d) Ville's inequality. 

- The SSL Interplay: Augmentations, Inductive Bias, and Generalization. [[paper]](https://arxiv.org/abs/2302.02774)
  - Vivien Cabannes, Bobak T. Kiani, Randall Balestriero, Yann LeCun, Alberto Bietti.
  - Key Word: Self-Supervised Learning; Data Augmentation; Inductive Bias.
  - <details><summary>Digest</summary> Self-supervised learning (SSL) has emerged as a powerful framework to learn representations from raw data without supervision. Yet in practice, engineers face issues such as instability in tuning optimizers and collapse of representations during training. Such challenges motivate the need for a theory to shed light on the complex interplay between the choice of data augmentation, network architecture, and training algorithm. We study such an interplay with a precise analysis of generalization performance on both pretraining and downstream tasks in a theory friendly setup, and highlight several insights for SSL practitioners that arise from our theory.

- A Stability Analysis of Fine-Tuning a Pre-Trained Model. [[paper]](https://arxiv.org/abs/2301.09820)
  - Zihao Fu, Anthony Man-Cho So, Nigel Collier.
  - Key Word: Fine-Tuning; Stability Analysis.
  - <details><summary>Digest</summary> We propose a novel theoretical stability analysis of fine-tuning that focuses on two commonly used settings, namely, full fine-tuning and head tuning. We define the stability under each setting and prove the corresponding stability bounds. The theoretical bounds explain why and how several existing methods can stabilize the fine-tuning procedure. 

- Strong inductive biases provably prevent harmless interpolation. [[paper]](https://arxiv.org/abs/2301.07605) [[code]](https://github.com/michaelaerni/iclr23-inductivebiasesharmlessinterpolation)
  - Michael Aerni, Marco Milanta, Konstantin Donhauser, Fanny Yang.
  - Key Word: Benign Overfitting; Inductive Bias.
  - <details><summary>Digest</summary> This paper argues that the degree to which interpolation is harmless hinges upon the strength of an estimator's inductive bias, i.e., how heavily the estimator favors solutions with a certain structure: while strong inductive biases prevent harmless interpolation, weak inductive biases can even require fitting noise to generalize well. 

### Others: 2022

- Feature learning in neural networks and kernel machines that recursively learn features. [[paper]](https://arxiv.org/abs/2212.13881) [[code]](https://github.com/aradha/recursive_feature_machines)
  - Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, Mikhail Belkin.
  - Key Word: Feature Learning; Kernel Machines; Grokking; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We isolate the key mechanism driving feature learning in fully connected neural networks by connecting neural feature learning to the average gradient outer product. We subsequently leverage this mechanism to design Recursive Feature Machines (RFMs), which are kernel machines that learn features. We show that RFMs (1) accurately capture features learned by deep fully connected neural networks, (2) close the gap between kernel machines and fully connected networks, and (3) surpass a broad spectrum of models including neural networks on tabular data. 

- PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization. [[paper]](https://arxiv.org/abs/2211.13609) [[code]](https://github.com/activatedgeek/tight-pac-bayes)
  - Sanae Lotfi, Marc Finzi, Sanyam Kapoor, Andres Potapczynski, Micah Goldblum, Andrew Gordon Wilson. 
  - Key Word: PAC-Bayes; Model Compression.
  - <details><summary>Digest</summary> We develop a compression approach based on quantizing neural network parameters in a linear subspace, profoundly improving on previous results to provide state-of-the-art generalization bounds on a variety of tasks, including transfer learning. We use these tight bounds to better understand the role of model size, equivariance, and the implicit biases of optimization, for generalization in deep learning. 

- Instance-Dependent Generalization Bounds via Optimal Transport. [[paper]](https://arxiv.org/abs/2211.01258)
  - Songyan Hou, Parnian Kassraie, Anastasis Kratsios, Jonas Rothfuss, Andreas Krause.
  - Key Word: Generalization Bounds; Optimal Transport; Distribution Shifts.
  - <details><summary>Digest</summary> We propose a novel optimal transport interpretation of the generalization problem. This allows us to derive instance-dependent generalization bounds that depend on the local Lipschitz regularity of the learned prediction function} in the data space. Therefore, our bounds are agnostic to the parametrization of the model and work well when the number of training samples is much smaller than the number of parameters. With small modifications, our approach yields accelerated rates for data on low-dimensional manifolds, and guarantees under distribution shifts. We empirically analyze our generalization bounds for neural networks, showing that the bound values are meaningful and capture the effect of popular regularization methods during training.

- How Does Sharpness-Aware Minimization Minimize Sharpness? [[paper]](https://arxiv.org/abs/2211.05729)
  - Kaiyue Wen, Tengyu Ma, Zhiyuan Li.
  - Key Word: Sharpness-Aware Minimization.
  - <details><summary>Digest</summary> This paper rigorously nails down the exact sharpness notion that SAM regularizes and clarifies the underlying mechanism. We also show that the two steps of approximations in the original motivation of SAM individually lead to inaccurate local conclusions, but their combination accidentally reveals the correct effect, when full-batch gradients are applied. Furthermore, we also prove that the stochastic version of SAM in fact regularizes the third notion of sharpness mentioned above, which is most likely to be the preferred notion for practical performance. The key mechanism behind this intriguing phenomenon is the alignment between the gradient and the top eigenvector of Hessian when SAM is applied.

- Augmentation Invariant Manifold Learning. [[paper]](https://arxiv.org/abs/2211.00460)
  - Shulei Wang.
  - Key Word: Manifold Learning; Data Augmentation.
  - <details><summary>Digest</summary> We develop a statistical framework on a low-dimension product manifold to theoretically understand why the unlabeled augmented data can lead to useful data representation. Under this framework, we propose a new representation learning method called augmentation invariant manifold learning and develop the corresponding loss function, which can work with a deep neural network to learn data representations.

- The Curious Case of Benign Memorization. [[paper]](https://arxiv.org/abs/2210.14019)
  - Sotiris Anagnostidis, Gregor Bachmann, Lorenzo Noci, Thomas Hofmann.
  - Key Word: Memorization; Data Augmentation.
  - <details><summary>Digest</summary> We show that under training protocols that include data augmentation, neural networks learn to memorize entirely random labels in a benign way, i.e. they learn embeddings that lead to highly non-trivial performance under nearest neighbour probing. We demonstrate that deep models have the surprising ability to separate noise from signal by distributing the task of memorization and feature learning to different layers.

- Provably Learning Diverse Features in Multi-View Data with Midpoint Mixup. [[paper]](https://arxiv.org/abs/2210.13512)
  - Muthu Chidambaram, Xiang Wang, Chenwei Wu, Rong Ge.
  - Key Word: Mixup; Feature Learning.
  - <details><summary>Digest</summary> We try to explain some of this success from a feature learning perspective. We focus our attention on classification problems in which each class may have multiple associated features (or views) that can be used to predict the class correctly. Our main theoretical results demonstrate that, for a non-trivial class of data distributions with two features per class, training a 2-layer convolutional network using empirical risk minimization can lead to learning only one feature for almost all classes while training with a specific instantiation of Mixup succeeds in learning both features for every class. 

- A PAC-Bayesian Generalization Bound for Equivariant Networks. [[paper]](https://arxiv.org/abs/2210.13150)
  - Arash Behboodi, Gabriele Cesa, Taco Cohen. *NeurIPS 2022*
  - Key Word: PAC-Bayes; Equivariant Networks.
  - <details><summary>Digest</summary> We study how equivariance relates to generalization error utilizing PAC Bayesian analysis for equivariant networks, where the transformation laws of feature spaces are determined by group representations. By using perturbation analysis of equivariant networks in Fourier domain for each layer, we derive norm-based PAC-Bayesian generalization bounds. The bound characterizes the impact of group size, and multiplicity and degree of irreducible representations on the generalization error and thereby provide a guideline for selecting them.

- Tighter PAC-Bayes Generalisation Bounds by Leveraging Example Difficulty. [[paper]](https://arxiv.org/abs/2210.11289)
  - Felix Biggs, Benjamin Guedj.
  - Key Word: PAC-Bayes.
  - <details><summary>Digest</summary> We introduce a modified version of the excess risk, which can be used to obtain tighter, fast-rate PAC-Bayesian generalisation bounds. This modified excess risk leverages information about the relative hardness of data examples to reduce the variance of its empirical counterpart, tightening the bound. We combine this with a new bound for [−1,1]-valued (and potentially non-independent) signed losses, which is more favourable when they empirically have low variance around 0. The primary new technical tool is a novel result for sequences of interdependent random vectors which may be of independent interest. We empirically evaluate these new bounds on a number of real-world datasets.

- How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders. [[paper]](https://arxiv.org/abs/2210.08344) [[code]](https://github.com/zhangq327/u-mae)
  - Qi Zhang, Yifei Wang, Yisen Wang. *NeurIPS 2022*
  - Key Word: Masked Autoencoders.
  - <details><summary>Digest</summary> We propose a theoretical understanding of how masking matters for MAE to learn meaningful features. We establish a close connection between MAE and contrastive learning, which shows that MAE implicit aligns the mask-induced positive pairs. Built upon this connection, we develop the first downstream guarantees for MAE methods, and analyze the effect of mask ratio. Besides, as a result of the implicit alignment, we also point out the dimensional collapse issue of MAE, and propose a Uniformity-enhanced MAE (U-MAE) loss that can effectively address this issue and bring significant improvements on real-world datasets, including CIFAR-10, ImageNet-100, and ImageNet-1K.

- On the Importance of Gradient Norm in PAC-Bayesian Bounds. [[paper]](https://arxiv.org/abs/2210.06143)
  - Itai Gat, Yossi Adi, Alexander Schwing, Tamir Hazan. *NeurIPS 2022*
  - Key Word: PAC-Bayes.
  - <details><summary>Digest</summary> Generalization bounds which assess the difference between the true risk and the empirical risk, have been studied extensively. However, to obtain bounds, current techniques use strict assumptions such as a uniformly bounded or a Lipschitz loss function. To avoid these assumptions, in this paper, we follow an alternative approach: we relax uniform bounds assumptions by using on-average bounded loss and on-average bounded gradient norm assumptions. Following this relaxation, we propose a new generalization bound that exploits the contractivity of the log-Sobolev inequalities. 

- SGD with large step sizes learns sparse features. [[paper]](https://arxiv.org/abs/2210.05337)
  - Maksym Andriushchenko, Aditya Varre, Loucas Pillaud-Vivien, Nicolas Flammarion.
  - Key Word: Stochastic Gradient Descent; Sparse Features.
  - <details><summary>Digest</summary> We showcase important features of the dynamics of the Stochastic Gradient Descent (SGD) in the training of neural networks. We present empirical observations that commonly used large step sizes (i) lead the iterates to jump from one side of a valley to the other causing loss stabilization, and (ii) this stabilization induces a hidden stochastic dynamics orthogonal to the bouncing directions that biases it implicitly toward simple predictors. 

- The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective. [[paper]](https://arxiv.org/abs/2210.05021)
  - Chi-Heng Lin, Chiraag Kaushik, Eva L. Dyer, Vidya Muthukumar. 
  - Key Word: Data Augmentation; Spectral Regularization.
  - <details><summary>Digest</summary> We develop a new theoretical framework to characterize the impact of a general class of DA on underparameterized and overparameterized linear model generalization. Our framework reveals that DA induces implicit spectral regularization through a combination of two distinct effects: a) manipulating the relative proportion of eigenvalues of the data covariance matrix in a training-data-dependent manner, and b) uniformly boosting the entire spectrum of the data covariance matrix through ridge regression. 

- Understanding Gradient Regularization in Deep Learning: Efficient Finite-Difference Computation and Implicit Bias. [[paper]](https://arxiv.org/abs/2210.02720)
  - Ryo Karakida, Tomoumi Takase, Tomohiro Hayase, Kazuki Osawa.
  - Key Word: Gradient Regularization; Implicit Bias.
  - <details><summary>Digest</summary> We first reveal that a specific finite-difference computation, composed of both gradient ascent and descent steps, reduces the computational cost for GR. In addition, this computation empirically achieves better generalization performance. Next, we theoretically analyze a solvable model, a diagonal linear network, and clarify that GR has a desirable implicit bias in a certain problem. In particular, learning with the finite-difference GR chooses better minima as the ascent step size becomes larger. 

- The Dynamics of Sharpness-Aware Minimization: Bouncing Across Ravines and Drifting Towards Wide Minima. [[paper]](https://arxiv.org/abs/2210.01513)
  - Peter L. Bartlett, Philip M. Long, Olivier Bousquet.
  - Key Word: Sharpness-Aware Minimization.
  - <details><summary>Digest</summary> We consider Sharpness-Aware Minimization (SAM), a gradient-based optimization method for deep networks that has exhibited performance improvements on image and language prediction problems. We show that when SAM is applied with a convex quadratic objective, for most random initializations it converges to a cycle that oscillates between either side of the minimum in the direction with the largest curvature, and we provide bounds on the rate of convergence.

- SAM as an Optimal Relaxation of Bayes. [[paper]](https://arxiv.org/abs/2210.01620)
  - Thomas Möllenhoff, Mohammad Emtiyaz Khan.
  - Key Word: Sharpness-Aware Minimization; Bayesian Methods.
  - <details><summary>Digest</summary> Sharpness-aware minimization (SAM) and related adversarial deep-learning methods can drastically improve generalization, but their underlying mechanisms are not yet fully understood. Here, we establish SAM as a relaxation of the Bayes objective where the expected negative-loss is replaced by the optimal convex lower bound, obtained by using the so-called Fenchel biconjugate. The connection enables a new Adam-like extension of SAM to automatically obtain reasonable uncertainty estimates, while sometimes also improving its accuracy. By connecting adversarial and Bayesian methods, our work opens a new path to robustness.

- Understanding Influence Functions and Datamodels via Harmonic Analysis. [[paper]](https://arxiv.org/abs/2210.01072)
  - Nikunj Saunshi, Arushi Gupta, Mark Braverman, Sanjeev Arora.
  - Key Word: Influence Functions; Harmonic Analysis.
  - <details><summary>Digest</summary> The current paper seeks to provide a better theoretical understanding of such interesting empirical phenomena. The primary tool is harmonic analysis and the idea of noise stability. Contributions include: (a) Exact characterization of the learnt datamodel in terms of Fourier coefficients. (b) An efficient method to estimate the residual error and quality of the optimum linear datamodel without having to train the datamodel. (c) New insights into when influences of groups of datapoints may or may not add up linearly.

- Plateau in Monotonic Linear Interpolation -- A "Biased" View of Loss Landscape for Deep Networks. [[paper]](https://arxiv.org/abs/2210.01019)
  - Xiang Wang, Annie N. Wang, Mo Zhou, Rong Ge.
  - Key Word: Monotonic Linear Interpolation; Loss Landscapes.
  - <details><summary>Digest</summary> We show that the MLI property is not necessarily related to the hardness of optimization problems, and empirical observations on MLI for deep neural networks depend heavily on biases. In particular, we show that interpolating both weights and biases linearly leads to very different influences on the final output, and when different classes have different last-layer biases on a deep network, there will be a long plateau in both the loss and accuracy interpolation (which existing theory of MLI cannot explain). 

- Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability. [[paper]](https://arxiv.org/abs/2209.15594)
  - Alex Damian, Eshaan Nichani, Jason D. Lee.
  - Key Word: Implicit Bias; Edge of Stability.
  - <details><summary>Digest</summary> Traditional analyses of gradient descent show that when the largest eigenvalue of the Hessian, also known as the sharpness S(θ), is bounded by 2/η, training is "stable" and the training loss decreases monotonically. Recent works, however, have observed that this assumption does not hold when training modern neural networks with full batch or large batch gradient descent. Most recently, Cohen et al. (2021) observed two important phenomena. The first, dubbed progressive sharpening, is that the sharpness steadily increases throughout training until it reaches the instability cutoff 2/η. The second, dubbed edge of stability, is that the sharpness hovers at 2/η for the remainder of training while the loss continues decreasing, albeit non-monotonically. We demonstrate that, far from being chaotic, the dynamics of gradient descent at the edge of stability can be captured by a cubic Taylor expansion: as the iterates diverge in direction of the top eigenvector of the Hessian due to instability, the cubic term in the local Taylor expansion of the loss function causes the curvature to decrease until stability is restored.

- Implicit Bias of Large Depth Networks: a Notion of Rank for Nonlinear Functions. [[paper]](https://arxiv.org/abs/2209.15055)
  - Arthur Jacot.
  - Key Word: Non-Linear Rank; Implicit Bias.
  - <details><summary>Digest</summary> We show that the representation cost of fully connected neural networks with homogeneous nonlinearities - which describes the implicit bias in function space of networks with L2-regularization or with losses such as the cross-entropy - converges as the depth of the network goes to infinity to a notion of rank over nonlinear functions. We then inquire under which conditions the global minima of the loss recover the `true' rank of the data: we show that for too large depths the global minimum will be approximately rank 1 (underestimating the rank); we then argue that there is a range of depths which grows with the number of datapoints where the true rank is recovered.

- Scaling Laws For Deep Learning Based Image Reconstruction. [[paper]](https://arxiv.org/abs/2209.13435)
  - Tobit Klug, Reinhard Heckel.
  - Key Word: Scaling Laws; Inverse Problems.
  - <details><summary>Digest</summary> We study whether major performance gains are expected from scaling up the training set size. We consider image denoising, accelerated magnetic resonance imaging, and super-resolution and empirically determine the reconstruction quality as a function of training set size, while optimally scaling the network size. For all three tasks we find that an initially steep power-law scaling slows significantly already at moderate training set sizes. Interpolating those scaling laws suggests that even training on millions of images would not significantly improve performance. 

- Why neural networks find simple solutions: the many regularizers of geometric complexity. [[paper]](https://arxiv.org/abs/2209.13083)
  - Benoit Dherin, Michael Munn, Mihaela C. Rosca, David G.T. Barrett. *NeurIPS 2022*
  - Key Word: Regularization; Geometric Complexity; Dirichlet Energy.
  - <details><summary>Digest</summary> In many contexts, simpler models are preferable to more complex models and the control of this model complexity is the goal for many methods in machine learning such as regularization, hyperparameter tuning and architecture design. In deep learning, it has been difficult to understand the underlying mechanisms of complexity control, since many traditional measures are not naturally suitable for deep neural networks. Here we develop the notion of geometric complexity, which is a measure of the variability of the model function, computed using a discrete Dirichlet energy. Using a combination of theoretical arguments and empirical results, we show that many common training heuristics such as parameter norm regularization, spectral norm regularization, flatness regularization, implicit gradient regularization, noise regularization and the choice of parameter initialization all act to control geometric complexity, providing a unifying framework in which to characterize the behavior of deep learning models.

- Variational Inference for Infinitely Deep Neural Networks. [[paper]](https://arxiv.org/abs/2209.10091)
  - Achille Nazaret, David Blei. *ICML 2022*
  - Key Word: Unbounded Depth Neural Networks; Variational Inference.
  - <details><summary>Digest</summary> We develop a novel variational inference algorithm to approximate this posterior, optimizing a distribution of the neural network weights and of the truncation depth L, and without any upper limit on L. To this end, the variational family has a special structure: it models neural network weights of arbitrary depth, and it dynamically creates or removes free variational parameters as its distribution of the truncation is optimized. 

- Deep Linear Networks can Benignly Overfit when Shallow Ones Do. [[paper]](https://arxiv.org/abs/2209.09315)
  - Niladri S. Chatterji, Philip M. Long.
  - Key Word: Benign Overfitting; Double Descent; Implicit Bias.
  - <details><summary>Digest</summary> We bound the excess risk of interpolating deep linear networks trained using gradient flow. In a setting previously used to establish risk bounds for the minimum ℓ2-norm interpolant, we show that randomly initialized deep linear networks can closely approximate or even match known bounds for the minimum ℓ2-norm interpolant. Our analysis also reveals that interpolating deep linear models have exactly the same conditional variance as the minimum ℓ2-norm solution. 

- Robustness in deep learning: The good (width), the bad (depth), and the ugly (initialization). [[paper]](https://arxiv.org/abs/2209.07263)
  - Zhenyu Zhu, Fanghui Liu, Grigorios G Chrysos, Volkan Cevher. *NeurIPS 2022*
  - Key Word: Lazy Training; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We study the average robustness notion in deep neural networks in (selected) wide and narrow, deep and shallow, as well as lazy and non-lazy training settings. We prove that in the under-parameterized setting, width has a negative effect while it improves robustness in the over-parameterized setting. The effect of depth closely depends on the initialization and the training mode. In particular, when initialized with LeCun initialization, depth helps robustness with lazy training regime. In contrast, when initialized with Neural Tangent Kernel (NTK) and He-initialization, depth hurts the robustness. 

- Git Re-Basin: Merging Models modulo Permutation Symmetries. [[paper]](https://arxiv.org/abs/2209.04836)
  - Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa.
  - Key Word: Mode Connectivity.
  - <details><summary>Digest</summary> We argue that neural network loss landscapes contain (nearly) a single basin, after accounting for all possible permutation symmetries of hidden units. We introduce three algorithms to permute the units of one model to bring them into alignment with units of a reference model. This transformation produces a functionally equivalent set of weights that lie in an approximately convex basin near the reference model. Experimentally, we demonstrate the single basin phenomenon across a variety of model architectures and datasets, including the first (to our knowledge) demonstration of zero-barrier linear mode connectivity between independently trained ResNet models on CIFAR-10 and CIFAR-100. 

- Normalization effects on deep neural networks. [[paper]](https://arxiv.org/abs/2209.01018)
  - Jiahui Yu, Konstantinos Spiliopoulos.
  - Key Word: Normalization.
  - <details><summary>Digest</summary> We find that in terms of variance of the neural network's output and test accuracy the best choice is to choose the γi's to be equal to one, which is the mean-field scaling. We also find that this is particularly true for the outer layer, in that the neural network's behavior is more sensitive in the scaling of the outer layer as opposed to the scaling of the inner layers. The mechanism for the mathematical analysis is an asymptotic expansion for the neural network's output.

- Do Quantum Circuit Born Machines Generalize? [[paper]](https://arxiv.org/abs/2207.13645)
  - Kaitlin Gili, Mohamed Hibat-Allah, Marta Mauri, Chris Ballance, Alejandro Perdomo-Ortiz.
  - Key Word: Quantum Machine Learning; Quantum Circuit Born Machines; Unsupervised Generative Models.
  - <details><summary>Digest</summary> There has been little understanding of a model's generalization performance and the relation between such capability and the resource requirements, e.g., the circuit depth and the amount of training data. In this work, we leverage upon a recently proposed generalization evaluation framework to begin addressing this knowledge gap. We first investigate the QCBM's learning process of a cardinality-constrained distribution and see an increase in generalization performance while increasing the circuit depth. 

- Benign, Tempered, or Catastrophic: A Taxonomy of Overfitting. [[paper]](https://arxiv.org/abs/2207.06569)
  - Neil Mallinar, James B. Simon, Amirhesam Abedsoltan, Parthe Pandit, Mikhail Belkin, Preetum Nakkiran.
  - Key Word: Overfitting; Kernel Regression.
  - <details><summary>Digest</summary> The practical success of overparameterized neural networks has motivated the recent scientific study of interpolating methods, which perfectly fit their training data. Certain interpolating methods, including neural networks, can fit noisy training data without catastrophically bad test performance, in defiance of standard intuitions from statistical learning theory. Aiming to explain this, a body of recent work has studied benign overfitting, a phenomenon where some interpolating methods approach Bayes optimality, even in the presence of noise. In this work we argue that while benign overfitting has been instructive and fruitful to study, many real interpolating methods like neural networks do not fit benignly: modest noise in the training set causes nonzero (but non-infinite) excess risk at test time, implying these models are neither benign nor catastrophic but rather fall in an intermediate regime. We call this intermediate regime tempered overfitting, and we initiate its systematic study.

- Towards understanding how momentum improves generalization in deep learning. [[paper]](https://arxiv.org/abs/2207.05931)
  - Samy Jelassi, Yuanzhi Li. *ICML 2022*
  - Key Word: Gradient Descent with Momentum.
  - <details><summary>Digest</summary> We adopt another perspective and first empirically show that gradient descent with momentum (GD+M) significantly improves generalization compared to gradient descent (GD) in some deep learning problems. From this observation, we formally study how momentum improves generalization. We devise a binary classification setting where a one-hidden layer (over-parameterized) convolutional neural network trained with GD+M provably generalizes better than the same network trained with GD, when both algorithms are similarly initialized.

- Synergy and Symmetry in Deep Learning: Interactions between the Data, Model, and Inference Algorithm. [[paper]](https://arxiv.org/abs/2207.04612)
  - Lechao Xiao, Jeffrey Pennington. *ICML 2022*
  - Key Word: Synergy; Symmetry; Implicit Bias; Neural Tangent Kernel; Neural Scaling Laws.
  - <details><summary>Digest</summary> Although learning in high dimensions is commonly believed to suffer from the curse of dimensionality, modern machine learning methods often exhibit an astonishing power to tackle a wide range of challenging real-world learning problems without using abundant amounts of data. How exactly these methods break this curse remains a fundamental open question in the theory of deep learning. While previous efforts have investigated this question by studying the data (D), model (M), and inference algorithm (I) as independent modules, in this paper, we analyze the triplet (D, M, I) as an integrated system and identify important synergies that help mitigate the curse of dimensionality.

- Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent. [[paper]](https://arxiv.org/abs/2207.04036)
  - Zhiyuan Li, Tianhao Wang, JasonD. Lee, Sanjeev Arora.
  - Key Word: Implicit Bias; Mirror Descent.
  - <details><summary>Digest</summary> As part of the effort to understand implicit bias of gradient descent in overparametrized models, several results have shown how the training trajectory on the overparametrized model can be understood as mirror descent on a different objective. The main result here is a characterization of this phenomenon under a notion termed commuting parametrization, which encompasses all the previous results in this setting. It is shown that gradient flow with any commuting parametrization is equivalent to continuous mirror descent with a related Legendre function.

- A law of adversarial risk, interpolation, and label noise. [[paper]](https://arxiv.org/abs/2207.03933)
  - Daniel Paleka, Amartya Sanyal. *ICLR 2023*
  - Key Word: Benign Overfitting; Adversarial Robustness.
  - <details><summary>Digest</summary> We show that interpolating label noise induces adversarial vulnerability, and prove the first theorem showing the relationship between label noise and adversarial risk for any data distribution. Our results are almost tight if we do not make any assumptions on the inductive bias of the learning algorithm. 

- Integral Probability Metrics PAC-Bayes Bounds. [[paper]](https://arxiv.org/abs/2207.00614)
  - Ron Amit, Baruch Epstein, Shay Moran, Ron Meir. *NeurIPS 2022*
  - Key Word: PAC-Bayes Bound.
  - <details><summary>Digest</summary> We present a PAC-Bayes-style generalization bound which enables the replacement of the KL-divergence with a variety of Integral Probability Metrics (IPM). We provide instances of this bound with the IPM being the total variation metric and the Wasserstein distance. A notable feature of the obtained bounds is that they naturally interpolate between classical uniform convergence bounds in the worst case (when the prior and posterior are far away from each other), and improved bounds in favorable cases (when the posterior and prior are close). This illustrates the possibility of reinforcing classical generalization bounds with algorithm- and data-dependent components, thus making them more suitable to analyze algorithms that use a large hypothesis space.

- Beyond neural scaling laws: beating power law scaling via data pruning. [[paper]](https://arxiv.org/abs/2206.14486)
  - Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, Ari S. Morcos.
  - Key Word: Dataset Pruning; Ensemble Active Learning.
  - <details><summary>Digest</summary> Widely observed neural scaling laws, in which error falls off as a power of the training set size, model size, or both, have driven substantial performance improvements in deep learning. However, these improvements through scaling alone require considerable costs in compute and energy. Here we focus on the scaling of error with dataset size and show how both in theory and practice we can break beyond power law scaling and reduce it to exponential scaling instead if we have access to a high-quality data pruning metric that ranks the order in which training examples should be discarded to achieve any pruned dataset size. We then test this new exponential scaling prediction with pruned dataset size empirically, and indeed observe better than power law scaling performance on ResNets trained on CIFAR-10, SVHN, and ImageNet.

- Robustness Implies Generalization via Data-Dependent Generalization Bounds. [[paper]](https://arxiv.org/abs/2206.13497)
  - Kenji Kawaguchi, Zhun Deng, Kyle Luh, Jiaoyang Huang. *ICML 2022*
  - Key Word: Algorithmic Robustness Bound.
  - <details><summary>Digest</summary> This paper proves that robustness implies generalization via data-dependent generalization bounds. As a result, robustness and generalization are shown to be connected closely in a data-dependent manner. Our bounds improve previous bounds in two directions, to solve an open problem that has seen little development since 2010. The first is to reduce the dependence on the covering number. The second is to remove the dependence on the hypothesis space. We present several examples, including ones for lasso and deep learning, in which our bounds are provably preferable.

- Learning sparse features can lead to overfitting in neural networks. [[paper]](https://arxiv.org/abs/2206.12314) [[code]](https://github.com/pcsl-epfl/regressionsphere)
  - Leonardo Petrini, Francesco Cagnetta, Eric Vanden-Eijnden, Matthieu Wyart.
  - Key Word: Sparse Representation; Neural Tangent Kernel.
  - <details><summary>Digest</summary> It is widely believed that the success of deep networks lies in their ability to learn a meaningful representation of the features of the data. Yet, understanding when and how this feature learning improves performance remains a challenge: for example, it is beneficial for modern architectures trained to classify images, whereas it is detrimental for fully-connected networks trained for the same task on the same data. Here we propose an explanation for this puzzle, by showing that feature learning can perform worse than lazy training (via random feature kernel or the NTK) as the former can lead to a sparser neural representation. Although sparsity is known to be essential for learning anisotropic data, it is detrimental when the target function is constant or smooth along certain directions of input space. We illustrate this phenomenon in two settings: (i) regression of Gaussian random functions on the d-dimensional unit sphere and (ii) classification of benchmark datasets of images.

- Towards Understanding Why Mask-Reconstruction Pretraining Helps in Downstream Tasks. [[paper]](https://arxiv.org/abs/2206.03826)
  - Jiachun Pan, Pan Zhou, Shuicheng Yan.
  - Key Word: Mask-Reconstruction Pretraining; Self-Supervision.
  - <details><summary>Digest</summary> Supervised fine-tuning the pretrained encoder remarkably surpasses the conventional supervised learning (SL) trained from scratch. However, it is still unclear 1) how MRP performs semantic learning in the pretraining phase and 2) why it helps in downstream tasks. To solve these problems, we theoretically show that on an auto-encoder of a two/one-layered convolution encoder/decoder, MRP can capture all discriminative semantics in the pretraining dataset, and accordingly show its provable improvement over SL on the classification downstream task.

- Why do CNNs Learn Consistent Representations in their First Layer Independent of Labels and Architecture? [[paper]](https://arxiv.org/abs/2206.02454)
  - Rhea Chowers, Yair Weiss.
  - Key Word: Architecture Inductive Bias.
  - <details><summary>Digest</summary> It has previously been observed that the filters learned in the first layer of a CNN are qualitatively similar for different networks and tasks. We extend this finding and show a high quantitative similarity between filters learned by different networks. We consider the CNN filters as a filter bank and measure the sensitivity of the filter bank to different frequencies. We show that the sensitivity profile of different networks is almost identical, yet far from initialization. Remarkably, we show that it remains the same even when the network is trained with random labels. To understand this effect, we derive an analytic formula for the sensitivity of the filters in the first layer of a linear CNN. We prove that when the average patch in images of the two classes is identical, the sensitivity profile of the filters in the first layer will be identical in expectation when using the true labels or random labels and will only depend on the second-order statistics of image patches.

- A Theoretical Analysis on Feature Learning in Neural Networks: Emergence from Inputs and Advantage over Fixed Features. [[paper]](https://arxiv.org/abs/2206.01717)
  - Zhenmei Shi, Junyi Wei, Yingyu Liang. *ICLR 2022*
  - Key Word: Linearization of Neural Networks; Neural Tangent Kernel.
  - <details><summary>Digest</summary> To better understand the source and benefit of feature learning in neural networks, we consider learning problems motivated by practical data, where the labels are determined by a set of class relevant patterns and the inputs are generated from these along with some background patterns. We prove that neural networks trained by gradient descent can succeed on these problems. The success relies on the emergence and improvement of effective features, which are learned among exponentially many candidates efficiently by exploiting the data (in particular, the structure of the input distribution).

- Realistic Deep Learning May Not Fit Benignly. [[paper]](https://arxiv.org/abs/2206.00501)
  - Kaiyue Wen, Jiaye Teng, Jingzhao Zhang.
  - Key Word: Benign Overfitting.
  - <details><summary>Digest</summary> We examine the benign overfitting phenomena in real-world settings. We found that for tasks such as training a ResNet model on ImageNet dataset, the model does not fit benignly. To understand why benign overfitting fails in the ImageNet experiment, we analyze previous benign overfitting models under a more restrictive setup where the number of parameters is not significantly larger than the number of data points.

- A Model of One-Shot Generalization. [[paper]](https://arxiv.org/abs/2205.14553)
  - Thomas Laurent, James H. von Brecht, Xavier Bresson.
  - Key Word: One-Shot Generalization; PAC Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We provide a theoretical framework to study a phenomenon that we call one-shot generalization. This phenomenon refers to the ability of an algorithm to perform transfer learning within a single task, meaning that it correctly classifies a test point that has a single exemplar in the training set. We propose a simple data model and use it to study this phenomenon in two ways. First, we prove a non-asymptotic base-line -- kernel methods based on nearest-neighbor classification cannot perform one-shot generalization, independently of the choice of the kernel and the size of the training set. Second, we empirically show that the most direct neural network architecture for our data model performs one-shot generalization almost perfectly. This stark differential leads us to believe that the one-shot generalization mechanism is partially responsible for the empirical success of neural networks.

- Towards Understanding Grokking: An Effective Theory of Representation Learning. [[paper]](https://arxiv.org/abs/2205.10343)
  - Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, Mike Williams.
  - Key Word: Grokking; Physics of Learning; Deep Double Descent.
  - <details><summary>Digest</summary> We aim to understand grokking, a phenomenon where models generalize long after overfitting their training set. We present both a microscopic analysis anchored by an effective theory and a macroscopic analysis of phase diagrams describing learning performance across hyperparameters. We find that generalization originates from structured representations whose training dynamics and dependence on training set size can be predicted by our effective theory in a toy setting. We observe empirically the presence of four learning phases: comprehension, grokking, memorization, and confusion.

- Empirical Evaluation and Theoretical Analysis for Representation Learning: A Survey. [[paper]](https://arxiv.org/abs/2204.08226)
  - Kento Nozawa, Issei Sato. *IJCAI 2022*
  - Key Word: Representation Learning; Pre-training; Regularization.
  - <details><summary>Digest</summary> Representation learning enables us to automatically extract generic feature representations from a dataset to solve another machine learning task. Recently, extracted feature representations by a representation learning algorithm and a simple predictor have exhibited state-of-the-art performance on several machine learning tasks. Despite its remarkable progress, there exist various ways to evaluate representation learning algorithms depending on the application because of the flexibility of representation learning. To understand the current representation learning, we review evaluation methods of representation learning algorithms and theoretical analyses.

- The Effects of Regularization and Data Augmentation are Class Dependent. [[paper]](https://arxiv.org/abs/2204.03632)
  - Randall Balestriero, Leon Bottou, Yann LeCun. *NeurIPS 2022*
  - Key Word: Data Augmentation.
  - <details><summary>Digest</summary> We demonstrate that techniques such as DA or weight decay produce a model with a reduced complexity that is unfair across classes. The optimal amount of DA or weight decay found from cross-validation leads to disastrous model performances on some classes e.g. on Imagenet with a resnet50, the "barn spider" classification test accuracy falls from 68% to 46% only by introducing random crop DA during training. Even more surprising, such performance drop also appears when introducing uninformative regularization techniques such as weight decay. 

- Resonance in Weight Space: Covariate Shift Can Drive Divergence of SGD with Momentum. [[paper]](https://arxiv.org/abs/2203.11992)
  - Kirby Banman, Liam Peet-Pare, Nidhi Hegde, Alona Fyshe, Martha White. *ICLR 2022*
  - Key Word: Stochastic Gradient Descent; Covariate Shift.
  - <details><summary>Digest</summary> We show that SGDm under covariate shift with a fixed step-size can be unstable and diverge. In particular, we show SGDm under covariate shift is a parametric oscillator, and so can suffer from a phenomenon known as resonance. We approximate the learning system as a time varying system of ordinary differential equations, and leverage existing theory to characterize the system's divergence/convergence as resonant/nonresonant modes.

- Data Augmentation as Feature Manipulation. [[paper]](https://arxiv.org/abs/2203.01572)
  - Ruoqi Shen, Sébastien Bubeck, Suriya Gunasekar.
  - Key Word: Data Augmentation; Feature Learning.
  - <details><summary>Digest</summary> In this work we consider another angle, and we study the effect of data augmentation on the dynamic of the learning process. We find that data augmentation can alter the relative importance of various features, effectively making certain informative but hard to learn features more likely to be captured in the learning process. Importantly, we show that this effect is more pronounced for non-linear models, such as neural networks. Our main contribution is a detailed analysis of data augmentation on the learning dynamic for a two layer convolutional neural network in the recently proposed multi-view data model by Allen-Zhu and Li [2020]. 

- How Many Data Are Needed for Robust Learning? [[paper]](https://arxiv.org/abs/2202.11592)
  - Hongyang Zhang, Yihan Wu, Heng Huang.
  - Key Word: Robustness.
  - <details><summary>Digest</summary> In this work, we study the sample complexity of robust interpolation problem when the data are in a unit ball. We show that both too many data and small data hurt robustness.

- A Data-Augmentation Is Worth A Thousand Samples: Exact Quantification From Analytical Augmented Sample Moments. [[paper]](https://arxiv.org/abs/2202.08325)
  - Randall Balestriero, Ishan Misra, Yann LeCun. *NeurIPS 2022*
  - Key Word: Data Augmentation.
  - <details><summary>Digest</summary> We derive several quantities in close-form, such as the expectation and variance of an image, loss, and model's output under a given DA distribution. Those derivations open new avenues to quantify the benefits and limitations of DA. For example, we show that common DAs require tens of thousands of samples for the loss at hand to be correctly estimated and for the model training to converge.

### Others: 2021

- Discovering and Explaining the Representation Bottleneck of DNNs. [[paper]](https://arxiv.org/abs/2111.06236)
  - Huiqi Deng, Qihan Ren, Hao Zhang, Quanshi Zhang. *ICLR 2022*
  - Key Word: Representation Bottleneck; Explanation.
  - <details><summary>Digest</summary> This paper explores the bottleneck of feature representations of deep neural networks (DNNs), from the perspective of the complexity of interactions between input variables encoded in DNNs. To this end, we focus on the multi-order interaction between input variables, where the order represents the complexity of interactions. We discover that a DNN is more likely to encode both too simple and too complex interactions, but usually fails to learn interactions of intermediate complexity. Such a phenomenon is widely shared by different DNNs for different tasks. This phenomenon indicates a cognition gap between DNNs and humans, and we call it a representation bottleneck. We theoretically prove the underlying reason for the representation bottleneck.

- Generalization in quantum machine learning from few training data. [[paper]](https://arxiv.org/abs/2111.05292)
  - Matthias C. Caro, Hsin-Yuan Huang, M. Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, Patrick J. Coles. *Nature Communications*
  - Key Word: Quantum Machine Learning; Generalization Bounds.
  - <details><summary>Digest</summary> We provide a comprehensive study of generalization performance in QML after training on a limited number N of training data points. We also show that classification of quantum states across a phase transition with a quantum convolutional neural network requires only a very small training data set. Other potential applications include learning quantum error correcting codes or quantum dynamical simulation. Our work injects new hope into the field of QML, as good generalization is guaranteed from few training data.

- The Equilibrium Hypothesis: Rethinking implicit regularization in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2110.11749)
  - Yizhang Lou, Chris Mingard, Soufiane Hayou.
  - Key Word: Implicit Regularization.
  - <details><summary>Digest</summary> We provide the first explanation for this alignment hierarchy. We introduce and empirically validate the Equilibrium Hypothesis which states that the layers that achieve some balance between forward and backward information loss are the ones with the highest alignment to data labels.

- Understanding Dimensional Collapse in Contrastive Self-supervised Learning. [[paper]](https://arxiv.org/abs/2110.09348) [[code]](https://github.com/facebookresearch/directclr)
  - Li Jing, Pascal Vincent, Yann LeCun, Yuandong Tian. *ICLR 2022*
  - Key Word: Self-Supervision; Contrastive Learning; Implicit Regularization; Dimensional Collapse.
  - <details><summary>Digest</summary> We show that dimensional collapse also happens in contrastive learning. In this paper, we shed light on the dynamics at play in contrastive learning that leads to dimensional collapse. Inspired by our theory,  we propose a novel contrastive learning method, called DirectCLR, which directly optimizes the representation space without relying on a trainable projector.

- Implicit Sparse Regularization: The Impact of Depth and Early Stopping. [[paper]](https://arxiv.org/abs/2108.05574) [[code]](https://github.com/jiangyuan2li/implicit-sparse-regularization)
  - Jiangyuan Li, Thanh V. Nguyen, Chinmay Hegde, Raymond K. W. Wong. *NeurIPS 2021*
  - Key Word: Implicit Regularization.
  - <details><summary>Digest</summary> In this paper, we study the implicit bias of gradient descent for sparse regression. We extend results on regression with quadratic parametrization, which amounts to depth-2 diagonal linear networks, to more general depth-N networks, under more realistic settings of noise and correlated designs. We show that early stopping is crucial for gradient descent to converge to a sparse model, a phenomenon that we call implicit sparse regularization. This result is in sharp contrast to known results for noiseless and uncorrelated-design cases.

- The Benefits of Implicit Regularization from SGD in Least Squares Problems. [[paper]](https://arxiv.org/abs/2108.04552)
  - Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, Dean P. Foster, Sham M. Kakade. *NeurIPS 2021*
  - Key Word: Implicit Regularization.
  <details><summary>Digest</summary> We show: (1) for every problem instance and for every ridge parameter, (unregularized) SGD, when provided with logarithmically more samples than that provided to the ridge algorithm, generalizes no worse than the ridge solution (provided SGD uses a tuned constant stepsize); (2) conversely, there exist instances (in this wide problem class) where optimally-tuned ridge regression requires quadratically more samples than SGD in order to have the same generalization performance.

- Neural Controlled Differential Equations for Online Prediction Tasks. [[paper]](https://arxiv.org/abs/2106.11028) [[code]](https://github.com/jambo6/online-neural-cdes)
  - James Morrill, Patrick Kidger, Lingyi Yang, Terry Lyons.
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> Neural controlled differential equations (Neural CDEs) are state-of-the-art models for irregular time series. However, due to current implementations relying on non-causal interpolation schemes, Neural CDEs cannot currently be used in online prediction tasks; that is, in real-time as data arrives. This is in contrast to similar ODE models such as the ODE-RNN which can already operate in continuous time. Here we introduce and benchmark new interpolation schemes, most notably, rectilinear interpolation, which allows for an online everywhere causal solution to be defined.

- The Principles of Deep Learning Theory. [[paper]](https://arxiv.org/abs/2106.10165)
  - Daniel A. Roberts, Sho Yaida, Boris Hanin.
  - Key Word: Bayesian Learning; Neural Tangent Kernel; Statistical Physics; Information Theory; Residual Learning; Book.
  - <details><summary>Digest</summary> This book develops an effective theory approach to understanding deep neural networks of practical relevance. Beginning from a first-principles component-level picture of networks, we explain how to determine an accurate description of the output of trained networks by solving layer-to-layer iteration equations and nonlinear learning dynamics. 

- Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning. [[paper]](https://arxiv.org/abs/2106.09226) [[code]](https://github.com/sangmichaelxie/pretraining_analysis)
  - Colin Wei, Sang Michael Xie, Tengyu Ma. *NeurIPS 2021*
  - Key Word: Natural Language Processing; Pre-training; Prompting.
  - <details><summary>Digest</summary> We propose an analysis framework that links the pretraining and downstream tasks with an underlying latent variable generative model of text -- the downstream classifier must recover a function of the posterior distribution over the latent variables. We analyze head tuning (learning a classifier on top of the frozen pretrained model) and prompt tuning in this setting. The generative model in our analysis is either a Hidden Markov Model (HMM) or an HMM augmented with a latent memory component, motivated by long-term dependencies in natural language.

- Differentiable Multiple Shooting Layers. [[paper]](https://arxiv.org/abs/2106.03885) [[code]](https://colab.research.google.com/github/neuralcollapse/neuralcollapse/blob/main/neuralcollapse.ipynb)
  - Stefano Massaroli, Michael Poli, Sho Sonoda, Taji Suzuki, Jinkyoo Park, Atsushi Yamashita, Hajime Asama. *NeurIPS 2021*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> We detail a novel class of implicit neural models. Leveraging time-parallel methods for differential equations, Multiple Shooting Layers (MSLs) seek solutions of initial value problems via parallelizable root-finding algorithms. MSLs broadly serve as drop-in replacements for neural ordinary differential equations (Neural ODEs) with improved efficiency in number of function evaluations (NFEs) and wall-clock inference time.

- Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning. [[paper]](https://arxiv.org/abs/2106.02584) [[code]](https://github.com/OATML/Non-Parametric-Transformers)
  - Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, Yarin Gal. *NeurIPS 2021*
  - Key Word: Samplie-Wise Self-Attention; Meta Learning; Metric Learning.
  - <details><summary>Digest</summary> We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms.

- Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. [[paper]](https://arxiv.org/abs/2105.14368)
  - Mikhail Belkin.
  - Key Word: Interpolation; Over-parameterization.
  - <details><summary>Digest</summary> In the past decade the mathematical theory of machine learning has lagged far behind the triumphs of deep neural networks on practical challenges. However, the gap between theory and practice is gradually starting to close. In this paper I will attempt to assemble some pieces of the remarkable and still incomplete mathematical mosaic emerging from the efforts to understand the foundations of deep learning. The two key themes will be interpolation, and its sibling, over-parameterization. Interpolation corresponds to fitting data, even noisy data, exactly. Over-parameterization enables interpolation and provides flexibility to select a right interpolating model.

- A Universal Law of Robustness via Isoperimetry. [[paper]](https://arxiv.org/abs/2105.12806)
  - Sébastien Bubeck, Mark Sellke.
  - Key Word: Overparameterized Memorization; Lipschitz Neural Network.
  - <details><summary>Digest</summary> A puzzling phenomenon in deep learning is that models are trained with many more parameters than what this classical theory would suggest. We propose a theoretical explanation for this phenomenon. We prove that for a broad class of data distributions and model classes, overparametrization is necessary if one wants to interpolate the data smoothly. Namely we show that smooth interpolation requires d times more parameters than mere interpolation, where d is the ambient data dimension. 

- Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes. [[paper]](https://arxiv.org/abs/2104.11044)
  - James Lucas, Juhan Bae, Michael R. Zhang, Stanislav Fort, Richard Zemel, Roger Grosse.
  - Key Word: Monotonic Linear Interpolation; Loss Landscapes.
  - <details><summary>Digest</summary> We evaluate several hypotheses for this property that, to our knowledge, have not yet been explored. Using tools from differential geometry, we draw connections between the interpolated paths in function space and the monotonicity of the network - providing sufficient conditions for the MLI property under mean squared error. While the MLI property holds under various settings (e.g. network architectures and learning problems), we show in practice that networks violating the MLI property can be produced systematically, by encouraging the weights to move far from initialization. 

- On the Validity of Modeling SGD with Stochastic Differential Equations (SDEs). [[paper]](https://arxiv.org/abs/2102.12470)
  - Zhiyuan Li, Sadhika Malladi, Sanjeev Arora. *NeurIPS 2021*
  - Key Word: Stochastic Gradient Descent Dynamics; Stochastic Differential Equations.
  - <details><summary>Digest</summary> The current paper clarifies the picture with the following contributions: (a) An efficient simulation algorithm SVAG that provably converges to the conventionally used Ito SDE approximation. (b) A theoretically motivated testable necessary condition for the SDE approximation and its most famous implication, the linear scaling rule (Goyal et al., 2017), to hold. (c) Experiments using this simulation to demonstrate that the previously proposed SDE approximation can meaningfully capture the training and generalization properties of common deep nets.

- MALI: A memory efficient and reverse accurate integrator for Neural ODEs. [[paper]](https://arxiv.org/abs/2102.04668) [[code]](https://github.com/juntang-zhuang/TorchDiffEqPack)
  - Juntang Zhuang, Nicha C. Dvornek, Sekhar Tatikonda, James S. Duncan. *ICLR 2021*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> Based on the asynchronous leapfrog (ALF) solver, we propose the Memory-efficient ALF Integrator (MALI), which has a constant memory cost w.r.t number of solver steps in integration similar to the adjoint method, and guarantees accuracy in reverse-time trajectory (hence accuracy in gradient estimation). We validate MALI in various tasks: on image recognition tasks, to our knowledge, MALI is the first to enable feasible training of a Neural ODE on ImageNet and outperform a well-tuned ResNet, while existing methods fail due to either heavy memory burden or inaccuracy.

- Learning Curve Theory. [[paper]](https://arxiv.org/abs/2102.04074)
  - Marcus Hutter.
  - Key Word: Neural Scaling Law; Learning Curve Theory.
  - <details><summary>Digest</summary> Recently a number of empirical "universal" scaling law papers have been published, most notably by OpenAI. `Scaling laws' refers to power-law decreases of training or test error w.r.t. more data, larger neural networks, and/or more compute. In this work we focus on scaling w.r.t. data size n. Theoretical understanding of this phenomenon is largely lacking, except in finite-dimensional models for which error typically decreases with n−1/2 or n−1, where n is the sample size. We develop and theoretically analyse the simplest possible (toy) model that can exhibit n−β learning curves for arbitrary power β>0, and determine whether power laws are universal or depend on the data distribution.


### Others: 2020

- Understanding the Failure Modes of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2010.15775) [[code]](https://github.com/google-research/OOD-failures)
  - Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We identify that spurious correlations during training can induce two distinct skews in the training set, one geometric and another statistical. These skews result in two complementary ways by which empirical risk minimization (ERM) via gradient descent is guaranteed to rely on those spurious correlations.

- Deep Networks from the Principle of Rate Reduction. [[paper]](https://arxiv.org/abs/2010.14765) [[code]](https://github.com/Ma-Lab-Berkeley/ReduNet)
  - Kwan Ho Ryan Chan, Yaodong Yu, Chong You, Haozhi Qi, John Wright, Yi Ma.
  - Key Word: Maximal Coding Rate Reduction.
  - <details><summary>Digest</summary> This work attempts to interpret modern deep (convolutional) networks from the principles of rate reduction and (shift) invariant classification. We show that the basic iterative gradient ascent scheme for optimizing the rate reduction of learned features naturally leads to a multi-layer deep network, one iteration per layer. The layered architectures, linear and nonlinear operators, and even parameters of the network are all explicitly constructed layer-by-layer in a forward propagation fashion by emulating the gradient scheme. 

- Sharpness-Aware Minimization for Efficiently Improving Generalization. [[paper]](https://arxiv.org/abs/2010.01412) [[code]](https://github.com/google-research/sam)
  - Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Flat Minima.
  - <details><summary>Digest</summary> In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by prior work connecting the geometry of the loss landscape and generalization, we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently.

- Implicit Gradient Regularization. [[paper]](https://arxiv.org/abs/2009.11162)
  - David G.T. Barrett, Benoit Dherin. *ICLR 2021*
  - Key Word: Implicit Regularization.
  - <details><summary>Digest</summary> Gradient descent can be surprisingly good at optimizing deep neural networks without overfitting and without explicit regularization. We find that the discrete steps of gradient descent implicitly regularize models by penalizing gradient descent trajectories that have large loss gradients. We call this Implicit Gradient Regularization (IGR) and we use backward error analysis to calculate the size of this regularization. We confirm empirically that implicit gradient regularization biases gradient descent toward flat minima, where test errors are small and solutions are robust to noisy parameter perturbations.

- Neural Rough Differential Equations for Long Time Series. [[paper]](https://arxiv.org/abs/2009.08295) [[code]](https://github.com/jambo6/neuralRDEs)
  - James Morrill, Cristopher Salvi, Patrick Kidger, James Foster, Terry Lyons. *ICML 2021*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> Neural Controlled Differential Equations (Neural CDEs) are the continuous-time analogue of an RNN. However, as with RNNs, training can quickly become impractical for long time series. Here we use rough path theory to extend this formulation through application of a pre-existing mathematical tool from rough analysis - the log-ODE method - which allows us to take integration steps larger than the discretisation of the data, resulting in significantly faster training times, with retainment (and often even improvements) in model performance.

- Optimizing Mode Connectivity via Neuron Alignment. [[paper]](https://arxiv.org/abs/2009.02439) [[code]](https://github.com/IBM/NeuronAlignment)
  - N. Joseph Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, Rongjie Lai. *NeurIPS 2020*
  - Key Word: Mode Connectivity; Neuron Alignment; Adversarial Training.
  - <details><summary>Digest</summary> We propose a more general framework to investigate the effect of symmetry on landscape connectivity by accounting for the weight permutations of the networks being connected. To approximate the optimal permutation, we introduce an inexpensive heuristic referred to as neuron alignment. Neuron alignment promotes similarity between the distribution of intermediate activations of models along the curve. 

- Benign Overfitting and Noisy Features. [[paper]](https://arxiv.org/abs/2008.02901)
  - Zhu Li, Weijie Su, Dino Sejdinovic. 
  - Key Word: Benign Overfitting; Random Feature Approximation; Deep Double Descent.
  - <details><summary>Digest</summary> We examine the conditions under which Benign Overfitting occurs in the random feature (RF) models, i.e. in a two-layer neural network with fixed first layer weights. We adopt a new view of random feature and show that benign overfitting arises due to the noise which resides in such features (the noise may already be present in the data and propagate to the features or it may be added by the user to the features directly) and plays an important implicit regularization role in the phenomenon.

- Expressivity of Deep Neural Networks. [[paper]](https://arxiv.org/abs/2007.04759)
  - Ingo Gühring, Mones Raslan, Gitta Kutyniok.
  - Key Word: Approximation; Expressivity; Function Classes
  - <details><summary>Digest</summary> In this review paper, we give a comprehensive overview of the large variety of approximation results for neural networks. Approximation rates for classical function spaces as well as benefits of deep neural networks over shallow ones for specifically structured function classes are discussed. While the mainbody of existing results is for general feedforward architectures, we also depict approximation results for convolutional, residual and recurrent neural networks.

- How benign is benign overfitting? [[paper]](https://arxiv.org/abs/2007.04028)
  - Amartya Sanyal, Puneet K Dokania, Varun Kanade, Philip H.S. Torr. *ICLR 2021*
  - Key Word: Benign Overfitting; Adversarial Robustness.
  - <details><summary>Digest</summary> We investigate two causes for adversarial vulnerability in deep neural networks: bad data and (poorly) trained models. When trained with SGD, deep neural networks essentially achieve zero training error, even in the presence of label noise, while also exhibiting good generalization on natural test data, something referred to as benign overfitting. However, these models are vulnerable to adversarial attacks. We identify label noise as one of the causes for adversarial vulnerability, and provide theoretical and empirical evidence in support of this. Surprisingly, we find several instances of label noise in datasets such as MNIST and CIFAR, and that robustly trained models incur training error on some of these, i.e. they don’t fit the noise.

- On the Theory of Transfer Learning: The Importance of Task Diversity. [[paper]](https://arxiv.org/abs/2006.11650)
  - Nilesh Tripuraneni, Michael I. Jordan, Chi Jin. *NeurIPS 2020*
  - Key Word: Transfer Learning; Task Diversity; Generalization Bound.
  - <details><summary>Digest</summary> We introduce a problem-agnostic definition of task diversity which can be integrated into a uniform convergence framework to provide generalization bounds for transfer learning problems with general losses, tasks, and features. Our framework puts this notion of diversity together with a common-design assumption across tasks to provide guarantees of a fast convergence rate, decaying with all of the samples for the transfer learning problem.

- Neural Controlled Differential Equations for Irregular Time Series. [[paper]](https://arxiv.org/abs/2005.08926) [[code]](https://github.com/patrick-kidger/NeuralCDE)
  - Patrick Kidger, James Morrill, James Foster, Terry Lyons. *NeurIPS 2020*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> a fundamental issue is that the solution to an ordinary differential equation is determined by its initial condition, and there is no mechanism for adjusting the trajectory based on subsequent observations. Here, we demonstrate how this may be resolved through the well-understood mathematics of controlled differential equations.

- Finite-sample Analysis of Interpolating Linear Classifiers in the Overparameterized Regime. [[paper]](https://arxiv.org/abs/2004.12019)
  - Niladri S. Chatterji, Philip M. Long. *JMLR*
  - Key Word: Benign Overfitting; Finite-Sample Analysis.
  - <details><summary>Digest</summary> We prove bounds on the population risk of the maximum margin algorithm for two-class linear classification. For linearly separable training data, the maximum margin algorithm has been shown in previous work to be equivalent to a limit of training with logistic loss using gradient descent, as the training error is driven to zero. We analyze this algorithm applied to random data including misclassification noise. Our assumptions on the clean data include the case in which the class-conditional distributions are standard normal distributions. The misclassification noise may be chosen by an adversary, subject to a limit on the fraction of corrupted labels. Our bounds show that, with sufficient over-parameterization, the maximum margin algorithm trained on noisy data can achieve nearly optimal population risk.

- Dissecting Neural ODEs. [[paper]](https://arxiv.org/abs/2002.08071) [[code]](https://github.com/DiffEqML/diffeqml-research/tree/master/dissecting-neural-odes)
  - Stefano Massaroli, Michael Poli, Jinkyoo Park, Atsushi Yamashita, Hajime Asama. *NeurIPS 2020*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> Continuous deep learning architectures have recently re-emerged as Neural Ordinary Differential Equations (Neural ODEs). This infinite-depth approach theoretically bridges the gap between deep learning and dynamical systems, offering a novel perspective. However, deciphering the inner working of these models is still an open challenge, as most applications apply them as generic black-box modules. In this work we "open the box", further developing the continuous-depth formulation with the aim of clarifying the influence of several design choices on the underlying dynamics.

- Proving the Lottery Ticket Hypothesis: Pruning is All You Need. [[paper]](https://arxiv.org/abs/2002.00585)
  - Eran Malach, Gilad Yehudai, Shai Shalev-Shwartz, Ohad Shamir. *ICML 2020*
  - Key Word: Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The lottery ticket hypothesis (Frankle and Carbin, 2018), states that a randomly-initialized network contains a small subnetwork such that, when trained in isolation, can compete with the performance of the original network. We prove an even stronger hypothesis (as was also conjectured in Ramanujan et al., 2019), showing that for every bounded distribution and every target network with bounded weights, a sufficiently over-parameterized neural network with random weights contains a subnetwork with roughly the same accuracy as the target network, without any further training.

### Others: 2019

- Deep Learning via Dynamical Systems: An Approximation Perspective. [[paper]](https://arxiv.org/abs/1912.10382)
  - Qianxiao Li, Ting Lin, Zuowei Shen.
  - Key Word: Approximation Theory; Controllability.
  - <details><summary>Digest</summary> We build on the dynamical systems approach to deep learning, where deep residual networks are idealized as continuous-time dynamical systems, from the approximation perspective. In particular, we establish general sufficient conditions for universal approximation using continuous-time deep residual networks, which can also be understood as approximation theories in Lp using flow maps of dynamical systems.

- Why bigger is not always better: on finite and infinite neural networks. [[paper]](https://arxiv.org/abs/1910.08013)
  - Laurence Aitchison. *ICML 2020*
  - Key Word: Gradient Dynamics.
  - <details><summary>Digest</summary> We give analytic results characterising the prior over representations and representation learning in finite deep linear networks. We show empirically that the representations in SOTA architectures such as ResNets trained with SGD are much closer to those suggested by our deep linear results than by the corresponding infinite network.  

- Deep Learning Theory Review: An Optimal Control and Dynamical Systems Perspective. [[paper]](https://arxiv.org/abs/1908.10920) [[code]](https://github.com/ghliu/mean-field-fcdnn)
  - Guan-Horng Liu, Evangelos A. Theodorou.
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We provide one possible way to align existing branches of deep learning theory through the lens of dynamical system and optimal control. By viewing deep neural networks as discrete-time nonlinear dynamical systems, we can analyze how information propagates through layers using mean field theory.

- Towards Explaining the Regularization Effect of Initial Large Learning Rate in Training Neural Networks. [[paper]](https://arxiv.org/abs/1907.04595) [[code]](https://github.com/cwein3/large-lr-code)
  - Yuanzhi Li, Colin Wei, Tengyu Ma. *NeurIPS 2019*
  - Key Word: Regularization.
  - <details><summary>Digest</summary> The key insight in our analysis is that the order of learning different types of patterns is crucial: because the small learning rate model first memorizes easy-to-generalize, hard-to-fit patterns, it generalizes worse on hard-to-generalize, easier-to-fit patterns than its large learning rate counterpart.  

- Are deep ResNets provably better than linear predictors? [[paper]](https://arxiv.org/abs/1907.03922)
  - Chulhee Yun, Suvrit Sra, Ali Jadbabaie. *NeurIPS 2019*
  - Key Word: ResNets; Local Minima.
  - <details><summary>Digest</summary> We investigated the question whether local minima of risk function of a deep ResNet are better than linear predictors. We showed two motivating examples showing 1) the advantage of ResNets over fully-connected networks, and 2) difficulty in analysis of deep ResNets.

- Benign Overfitting in Linear Regression. [[paper]](https://arxiv.org/abs/1906.11300)
  - Peter L. Bartlett, Philip M. Long, Gábor Lugosi, Alexander Tsigler. *PNAS*
  - Key Word: Benign Overfitting.
  - <details><summary>Digest</summary> The phenomenon of benign overfitting is one of the key mysteries uncovered by deep learning methodology: deep neural networks seem to predict well, even with a perfect fit to noisy training data. Motivated by this phenomenon, we consider when a perfect fit to training data in linear regression is compatible with accurate prediction. We give a characterization of linear regression problems for which the minimum norm interpolating prediction rule has near-optimal prediction accuracy. The characterization is in terms of two notions of the effective rank of the data covariance. It shows that overparameterization is essential for benign overfitting in this setting: the number of directions in parameter space that are unimportant for prediction must significantly exceed the sample size. 

- Invariance-inducing regularization using worst-case transformations suffices to boost accuracy and spatial robustness. [[paper]](https://arxiv.org/abs/1906.11235)
  - Fanny Yang, Zuowen Wang, Christina Heinze-Deml. *NeurIPS 2019*
  - Key Word: Robustness; Regularization.
  - <details><summary>Digest</summary> This work provides theoretical and empirical evidence that invariance-inducing regularizers can increase predictive accuracy for worst-case spatial transformations (spatial robustness). Evaluated on these adversarially transformed examples, we demonstrate that adding regularization on top of standard or adversarial training reduces the relative error by 20% for CIFAR10 without increasing the computational cost.

- Augmented Neural ODEs. [[paper]](https://arxiv.org/abs/1904.01681) [[code]](https://github.com/EmilienDupont/augmented-neural-odes)
  - Emilien Dupont, Arnaud Doucet, Yee Whye Teh. *NeurIPS 2019*
  - Key Word: Ordinary Differential Equations.
  - <details><summary>Digest</summary> We show that Neural Ordinary Differential Equations (ODEs) learn representations that preserve the topology of the input space and prove that this implies the existence of functions Neural ODEs cannot represent. To address these limitations, we introduce Augmented Neural ODEs which, in addition to being more expressive models, are empirically more stable, generalize better and have a lower computational cost than Neural ODEs.

- On the Power and Limitations of Random Features for Understanding Neural Networks. [[paper]](https://arxiv.org/abs/1904.00687)
  - Gilad Yehudai, Ohad Shamir.
  - Key Word: Random Features.
  - <details><summary>Digest</summary> Recently, a spate of papers have provided positive theoretical results for training over-parameterized neural networks (where the network size is larger than what is needed to achieve low error). The key insight is that with sufficient over-parameterization, gradient-based methods will implicitly leave some components of the network relatively unchanged, so the optimization dynamics will behave as if those components are essentially fixed at their initial random values. In fact, fixing these explicitly leads to the well-known approach of learning with random features. In other words, these techniques imply that we can successfully learn with neural networks, whenever we can successfully learn with random features. In this paper, we first review these techniques, providing a simple and self-contained analysis for one-hidden-layer networks. 

- Mean Field Analysis of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1903.04440)
  - Justin Sirignano, Konstantinos Spiliopoulos.
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We analyze multi-layer neural networks in the asymptotic regime of simultaneously (A) large network sizes and (B) large numbers of stochastic gradient descent training iterations. We rigorously establish the limiting behavior of the multi-layer neural network output. The limit procedure is valid for any number of hidden layers and it naturally also describes the limiting behavior of the training loss.

- Machine learning meets quantum physics. [[paper]](https://arxiv.org/abs/1903.03516) [[book]](https://link.springer.com/book/10.1007/978-3-030-40245-7)
  - Sankar Das Sarma, Dong-Ling Deng, Lu-Ming Duan.
  - Key Word: Physics-based Machine Learning; Quantum Physics; Quantum Chemistry.
  - <details><summary>Digest</summary> The marriage of machine learning and quantum physics may give birth to a new research frontier that could transform both.

- A Mean Field Theory of Batch Normalization. [[paper]](https://arxiv.org/abs/1902.08129)
  - Greg Yang, Jeffrey Pennington, Vinay Rao, Jascha Sohl-Dickstein, Samuel S. Schoenholz. *ICLR 2019*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We develop a mean field theory for batch normalization in fully-connected feedforward neural networks. In so doing, we provide a precise characterization of signal propagation and gradient backpropagation in wide batch-normalized networks at initialization. Our theory shows that gradient signals grow exponentially in depth and that these exploding gradients cannot be eliminated by tuning the initial weight variances or by adjusting the nonlinear activation function.

- Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. [[paper]](https://arxiv.org/abs/1902.06720) [[code]](https://github.com/google/neural-tangents)
  - Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, Jeffrey Pennington. *NeurIPS 2019*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We show that for wide neural networks the learning dynamics simplify considerably and that, in the infinite width limit, they are governed by a linear model obtained from the first-order Taylor expansion of the network around its initial parameters. Furthermore, mirroring the correspondence between wide Bayesian neural networks and Gaussian processes, gradient-based training of wide neural networks with a squared loss produces test set predictions drawn from a Gaussian process with a particular compositional kernel.

- Superposition of many models into one. [[paper]](https://arxiv.org/abs/1902.05522) [[code]](https://github.com/briancheung/superposition)
  - Brian Cheung, Alex Terekhov, Yubei Chen, Pulkit Agrawal, Bruno Olshausen. *NeurIPS 2019*
  - Key Word: Parameter Superposition; Catastrophic Forgetting.
  - <details><summary>Digest</summary> We present a method for storing multiple models within a single set of parameters. Models can coexist in superposition and still be retrieved individually. In experiments with neural networks, we show that a surprisingly large number of models can be effectively stored within a single parameter instance. Furthermore, each of these models can undergo thousands of training steps without significantly interfering with other models within the superposition. This approach may be viewed as the online complement of compression: rather than reducing the size of a network after training, we make use of the unrealized capacity of a network during training.

- On Nonconvex Optimization for Machine Learning: Gradients, Stochasticity, and Saddle Points. [[paper]](https://arxiv.org/abs/1902.04811)
  - Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, Michael I. Jordan. *ICML 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - <details><summary>Digest</summary> Traditional analyses of GD and SGD show that both algorithms converge to stationary points efficiently. But these analyses do not take into account the possibility of converging to saddle points. More recent theory has shown that GD and SGD can avoid saddle points, but the dependence on dimension in these analyses is polynomial. For modern machine learning, where the dimension can be in the millions, such dependence would be catastrophic. We analyze perturbed versions of GD and SGD and show that they are truly efficient---their dimension dependence is only polylogarithmic. Indeed, these algorithms converge to second-order stationary points in essentially the same time as they take to converge to classical first-order stationary points.

- Escaping Saddle Points with Adaptive Gradient Methods. [[paper]](https://arxiv.org/abs/1901.09149)
  - Matthew Staib, Sashank J. Reddi, Satyen Kale, Sanjiv Kumar, Suvrit Sra. *ICML 2019*
  - Key Word: Gradient Descent; Saddle Points.
  - <details><summary>Digest</summary> We seek a crisp, clean and precise characterization of their behavior in nonconvex settings. To this end, we first provide a novel view of adaptive methods as preconditioned SGD, where the preconditioner is estimated in an online manner. By studying the preconditioner on its own, we elucidate its purpose: it rescales the stochastic gradient noise to be isotropic near stationary points, which helps escape saddle points.

### Others: 2018

- A Spline Theory of Deep Learning. [[paper]](https://proceedings.mlr.press/v80/balestriero18b.html)
  - Randall Balestriero, Richard G. Baraniuk. *ICML 2018*
  - Key Word: Approximation Theory.
  - <details><summary>Digest</summary> We build a rigorous bridge between deep networks (DNs) and approximation theory via spline functions and operators. Our key result is that a large class of DNs can be written as a composition of max-affine spline operators (MASOs), which provide a powerful portal through which to view and analyze their inner workings.

- On Lazy Training in Differentiable Programming. [[paper]](https://arxiv.org/abs/1812.07956) [[code]](https://github.com/edouardoyallon/lazy-training-CNN)
  - Lenaic Chizat, Edouard Oyallon, Francis Bach. *NeurIPS 2019*
  - Key Word: Lazy Training.
  - <details><summary>Digest</summary> In a series of recent theoretical works, it was shown that strongly over-parameterized neural networks trained with gradient-based methods could converge exponentially fast to zero training loss, with their parameters hardly varying. In this work, we show that this "lazy training" phenomenon is not specific to over-parameterized neural networks, and is due to a choice of scaling, often implicit, that makes the model behave as its linearization around the initialization, thus yielding a model equivalent to learning with positive-definite kernels. Through a theoretical analysis, we exhibit various situations where this phenomenon arises in non-convex optimization and we provide bounds on the distance between the lazy and linearized optimization paths.

- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. [[paper]](https://arxiv.org/abs/1812.05720) [[code]](https://github.com/max-andr/relu_networks_overconfident)
  - Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf. *CVPR 2019*
  - Key Wrod: ReLU; Adversarial Example.
  - <details><summary>Digest</summary> We show that ReLU type neural networks which yield a piecewise linear classifier function fail in this regard as they produce almost always high confidence predictions far away from the training data. For bounded domains like images we propose a new robust optimization technique similar to adversarial training which enforces low confidence predictions far away from the training data.

- Gradient Descent Finds Global Minima of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1811.03804)
  - Simon S. Du, Jason D. Lee, Haochuan Li, Liwei Wang, Xiyu Zhai. *ICML 2019*
  - Key Word: Gradient Descent; Gradient Dynamics.
  - <details><summary>Digest</summary> Gradient descent finds a global minimum in training deep neural networks despite the objective function being non-convex. The current paper proves gradient descent achieves zero training loss in polynomial time for a deep over-parameterized neural network with residual connections (ResNet). Our analysis relies on the particular structure of the Gram matrix induced by the neural network architecture. This structure allows us to show the Gram matrix is stable throughout the training process and this stability implies the global optimality of the gradient descent algorithm.

- Memorization in Overparameterized Autoencoders. [[paper]](https://arxiv.org/abs/1810.10333)
  - Adityanarayanan Radhakrishnan, Karren Yang, Mikhail Belkin, Caroline Uhler.
  - Key Word: Autoencoders; Memorization.
  - <details><summary>Digest</summary> We show that overparameterized autoencoders exhibit memorization, a form of inductive bias that constrains the functions learned through the optimization process to concentrate around the training examples, although the network could in principle represent a much larger function class. In particular, we prove that single-layer fully-connected autoencoders project data onto the (nonlinear) span of the training examples.

- Information Geometry of Orthogonal Initializations and Training. [[paper]](https://arxiv.org/abs/1810.03785)
  - Piotr A. Sokol, Il Memming Park. *ICLR 2020*
  - Key Word: Mean Field Theory; Information Geometry.
  - <details><summary>Digest</summary> We show a novel connection between the maximum curvature of the optimization landscape (gradient smoothness) as measured by the Fisher information matrix (FIM) and the spectral radius of the input-output Jacobian, which partially explains why more isometric networks can train much faster.

- Gradient Descent Provably Optimizes Over-parameterized Neural Networks. [[paper]](https://arxiv.org/abs/1810.02054)
  - Simon S. Du, Xiyu Zhai, Barnabas Poczos, Aarti Singh. *ICLR 2019*
  - Key Word: Gradient Descent; Gradient Dynamics.
  - <details><summary>Digest</summary> One of the mysteries in the success of neural networks is randomly initialized first order methods like gradient descent can achieve zero training loss even though the objective function is non-convex and non-smooth. This paper demystifies this surprising phenomenon for two-layer fully connected ReLU activated neural networks. For an m hidden node shallow neural network with ReLU activation and n training data, we show as long as m is large enough and no two inputs are parallel, randomly initialized gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function.

- Dynamical Isometry is Achieved in Residual Networks in a Universal Way for any Activation Function. [[paper]](https://arxiv.org/abs/1809.08848)
  - Wojciech Tarnowski, Piotr Warchoł, Stanisław Jastrzębski, Jacek Tabor, Maciej A. Nowak. *AISTATS 2019*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We demonstrate that in residual neural networks (ResNets) dynamical isometry is achievable irrespectively of the activation function used. We do that by deriving, with the help of Free Probability and Random Matrix Theories, a universal formula for the spectral density of the input-output Jacobian at initialization, in the large network width and depth limit.

- Mean Field Analysis of Neural Networks: A Central Limit Theorem. [[paper]](https://arxiv.org/abs/1808.09372)
  - Justin Sirignano, Konstantinos Spiliopoulos.
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We rigorously prove a central limit theorem for neural network models with a single hidden layer. The central limit theorem is proven in the asymptotic regime of simultaneously (A) large numbers of hidden units and (B) large numbers of stochastic gradient descent training iterations. Our result describes the neural network's fluctuations around its mean-field limit. The fluctuations have a Gaussian distribution and satisfy a stochastic partial differential equation.

- An elementary introduction to information geometry. [[paper]](https://arxiv.org/abs/1808.08271)
  - Frank Nielsen.
  - Key Word: Survey; Information Geometry.
  - <details><summary>Digest</summary> In this survey, we describe the fundamental differential-geometric structures of information manifolds, state the fundamental theorem of information geometry, and illustrate some use cases of these information manifolds in information sciences. The exposition is self-contained by concisely introducing the necessary concepts of differential geometry, but proofs are omitted for brevity.

- Deep Convolutional Networks as shallow Gaussian Processes. [[paper]](https://arxiv.org/abs/1808.05587) [[code]](https://github.com/convnets-as-gps/convnets-as-gps)
  - Adrià Garriga-Alonso, Carl Edward Rasmussen, Laurence Aitchison. *ICLR 2019*
  - Key Word: Gaussian Process.
  - <details><summary>Digest</summary> We show that the output of a (residual) convolutional neural network (CNN) with an appropriate prior over the weights and biases is a Gaussian process (GP) in the limit of infinitely many convolutional filters, extending similar results for dense networks. For a CNN, the equivalent kernel can be computed exactly and, unlike "deep kernels", has very few parameters: only the hyperparameters of the original CNN.

- Learning Overparameterized Neural Networks via Stochastic Gradient Descent on Structured Data. [[paper]](https://arxiv.org/abs/1808.01204)
  - Yuanzhi Li, Yingyu Liang. *NeurIPS 2018*
  - Key Word: Stochastic Gradient Descent.
  - <details><summary>Digest</summary> Neural networks have many successful applications, while much less theoretical understanding has been gained. Towards bridging this gap, we study the problem of learning a two-layer overparameterized ReLU neural network for multi-class classification via stochastic gradient descent (SGD) from random initialization. In the overparameterized setting, when the data comes from mixtures of well-separated distributions, we prove that SGD learns a network with a small generalization error, albeit the network has enough capacity to fit arbitrary labels.

- Neural Ordinary Differential Equations. [[paper]](https://arxiv.org/abs/1806.07366) [[code]](https://github.com/rtqichen/torchdiffeq)
  - Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. *NeurIPS 2018*
  - Key Word: Ordinary Differential Equations; Normalizing Flow.
  - <details><summary>Digest</summary> We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions. For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models.

- Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1806.05393) [[code]](https://github.com/brain-research/mean-field-cnns)
  - Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, Jeffrey Pennington. *ICML 2018*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We demonstrate that it is possible to train vanilla CNNs with ten thousand layers or more simply by using an appropriate initialization scheme. We derive this initialization scheme theoretically by developing a mean field theory for signal propagation and by characterizing the conditions for dynamical isometry, the equilibration of singular values of the input-output Jacobian matrix.

- Universal Statistics of Fisher Information in Deep Neural Networks: Mean Field Approach. [[paper]](https://arxiv.org/abs/1806.01316)
  - Ryo Karakida, Shotaro Akaho, Shun-ichi Amari. *AISTATS 2019*
  - Key Word: Mean Field Theory; Fisher Information.
  - <details><summary>Digest</summary> The Fisher information matrix (FIM) is a fundamental quantity to represent the characteristics of a stochastic model, including deep neural networks (DNNs). The present study reveals novel statistics of FIM that are universal among a wide class of DNNs. To this end, we use random weights and large width limits, which enables us to utilize mean field theories. We investigate the asymptotic statistics of the FIM's eigenvalues and reveal that most of them are close to zero while the maximum eigenvalue takes a huge value.

- Towards Understanding the Role of Over-Parametrization in Generalization of Neural Networks. [[paper]](https://arxiv.org/abs/1805.12076) [[code]](https://github.com/bneyshabur/over-parametrization)
  - Behnam Neyshabur, Zhiyuan Li, Srinadh Bhojanapalli, Yann LeCun, Nathan Srebro. *ICLR 2019*
  - Key Word: Over-Parametrization.
  - <details><summary>Digest</summary> We suggest a novel complexity measure based on unit-wise capacities resulting in a tighter generalization bound for two layer ReLU networks. Our capacity bound correlates with the behavior of test error with increasing network sizes (within the range reported in the experiments), and could partly explain the improvement in generalization with over-parametrization.

- Understanding Generalization and Optimization Performance of Deep CNNs. [[paper]](https://arxiv.org/abs/1805.10767)
  - Pan Zhou, Jiashi Feng. *ICML 2018*
  - Key Word: Generalization of CNNs.
  - <details><summary>Digest</summary> We make multiple contributions to understand deep CNNs theoretically. To our best knowledge, this work presents the first theoretical guarantees on both generalization error bound without exponential growth over network depth and optimization performance for deep CNNs.

- Geometric Understanding of Deep Learning. [[paper]](https://arxiv.org/abs/1805.10451)
  - Na Lei, Zhongxuan Luo, Shing-Tung Yau, David Xianfeng Gu.
  - Key Word: Manifold Representation; Learning Capability; Learning Capability; Latent Probability Distribution Control.
  - <details><summary>Digest</summary> In this work, we give a geometric view to understand deep learning: we show that the fundamental principle attributing to the success is the manifold structure in data, namely natural high dimensional data concentrates close to a low-dimensional manifold, deep learning learns the manifold and the probability distribution on it. 

- Tropical Geometry of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1805.07091)
  - Liwen Zhang, Gregory Naitzat, Lek-Heng Lim.
  - Key Word: Tropical Geometry; Geometric Complexity.
  -  <details><summary>Digest</summary> We establish a novel connection between feedforward neural networks with ReLU activation and tropical geometry. This equivalence allows us to characterize these neural networks using zonotopes, relate decision boundaries to tropical hypersurfaces, and establish a correspondence between linear regions and vertices of polytopes associated with tropical rational functions. Our tropical formulation reveals that deeper networks exhibit exponentially higher expressiveness compared to shallow networks. This work provides new insights into the relationship between neural networks and tropical geometry.

- Gaussian Process Behaviour in Wide Deep Neural Networks. [[paper]](https://arxiv.org/abs/1804.11271) [[code]](https://github.com/widedeepnetworks/widedeepnetworks)
  - Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani. *ICLR 2018*
  - Key Word: Gaussian Process.
  - <details><summary>Digest</summary> We study the relationship between random, wide, fully connected, feedforward networks with more than one hidden layer and Gaussian processes with a recursive kernel definition. We show that, under broad conditions, as we make the architecture increasingly wide, the implied random function converges in distribution to a Gaussian process, formalising and extending existing results by Neal (1996) to deep networks.

- How to Start Training: The Effect of Initialization and Architecture. [[paper]](https://arxiv.org/abs/1803.01719)
  - Boris Hanin, David Rolnick. *NeurIPS 2018*
  - Key Word: Neuron Activation; Weight Initialization.
  - <details><summary>Digest</summary> We identify and study two common failure modes for early training in deep ReLU nets. The first failure mode, exploding/vanishing mean activation length, can be avoided by initializing weights from a symmetric distribution with variance 2/fan-in and, for ResNets, by correctly weighting the residual modules. We prove that the second failure mode, exponentially large variance of activation length, never occurs in residual nets once the first failure mode is avoided.  

- The Emergence of Spectral Universality in Deep Networks. [[paper]](https://arxiv.org/abs/1802.09979)
  - Jeffrey Pennington, Samuel S. Schoenholz, Surya Ganguli. *AISTATS 2018*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We leverage powerful tools from free probability theory to provide a detailed analytic understanding of how a deep network's Jacobian spectrum depends on various hyperparameters including the nonlinearity, the weight and bias distributions, and the depth. For a variety of nonlinearities, our work reveals the emergence of new universal limiting spectral distributions that remain concentrated around one even as the depth goes to infinity.

- Generalization in Machine Learning via Analytical Learning Theory. [[paper]](https://arxiv.org/abs/1802.07426) [[code]](https://github.com/Learning-and-Intelligent-Systems/Analytical-Learning-Theory)
  - Kenji Kawaguchi, Yoshua Bengio, Vikas Verma, Leslie Pack Kaelbling.
  - Key Word: Regularization; Measure Theory.
  - <details><summary>Digest</summary> This paper introduces a novel measure-theoretic theory for machine learning that does not require statistical assumptions. Based on this theory, a new regularization method in deep learning is derived and shown to outperform previous methods in CIFAR-10, CIFAR-100, and SVHN. Moreover, the proposed theory provides a theoretical basis for a family of practically successful regularization methods in deep learning. We discuss several consequences of our results on one-shot learning, representation learning, deep learning, and curriculum learning. Unlike statistical learning theory, the proposed learning theory analyzes each problem instance individually via measure theory, rather than a set of problem instances via statistics. As a result, it provides different types of results and insights when compared to statistical learning theory.

- Stronger generalization bounds for deep nets via a compression approach [[paper]](https://arxiv.org/abs/1802.05296)
  - Sanjeev Arora, Rong Ge, Behnam Neyshabur, Yi Zhang. *ICML 2018*
  - Key Word: PAC-Bayes; Compression-Based Generalization Bound.
  - <details><summary>Digest</summary> A simple compression framework for proving generalization bounds, perhaps a more explicit and intuitive form of the PAC-Bayes work. It also yields elementary short proofs of recent generalization results.

- Which Neural Net Architectures Give Rise To Exploding and Vanishing Gradients? [[paper]](https://arxiv.org/abs/1801.03744)
  - Boris Hanin. *NeurIPS 2018*
  - Key Word: Network Architectures.
  - <details><summary>Digest</summary> We give a rigorous analysis of the statistical behavior of gradients in a randomly initialized fully connected network N with ReLU activations. Our results show that the empirical variance of the squares of the entries in the input-output Jacobian of N is exponential in a simple architecture-dependent constant beta, given by the sum of the reciprocals of the hidden layer widths.  

### Others: 2017

- Mean Field Residual Networks: On the Edge of Chaos. [[paper]](https://arxiv.org/abs/1712.08969)
  - Greg Yang, Samuel S. Schoenholz. *NeurIPS 2017*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> The exponential forward dynamics causes rapid collapsing of the input space geometry, while the exponential backward dynamics causes drastic vanishing or exploding gradients. We show, in contrast, that by adding skip connections, the network will, depending on the nonlinearity, adopt subexponential forward and backward dynamics, and in many cases in fact polynomial.  

- Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice. [[paper]](https://arxiv.org/abs/1711.04735)
  - Jeffrey Pennington, Samuel S. Schoenholz, Surya Ganguli. *NeurIPS 2017*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We explore the dependence of the singular value distribution on the depth of the network, the weight initialization, and the choice of nonlinearity. Intriguingly, we find that ReLU networks are incapable of dynamical isometry. On the other hand, sigmoidal networks can achieve isometry, but only with orthogonal weight initialization. Moreover, we demonstrate empirically that deep nonlinear networks achieving dynamical isometry learn orders of magnitude faster than networks that do not.

- Deep Neural Networks as Gaussian Processes. [[paper]](https://arxiv.org/abs/1711.00165)
  - Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha Sohl-Dickstein. *ICLR 2018*
  - Key Word: Gaussian Process.
  - <details><summary>Digest</summary> In this work, we derive the exact equivalence between infinitely wide deep networks and GPs. We further develop a computationally efficient pipeline to compute the covariance function for these GPs.  

- Maximum Principle Based Algorithms for Deep Learning. [[paper]](https://arxiv.org/abs/1710.09513)
  - Qianxiao Li, Long Chen, Cheng Tai, Weinan E. *JMLR*
  - Key Word: Optimal control; Pontryagin’s Maximum Principle.
  - <details><summary>Digest</summary> We discuss the viewpoint that deep residual neural networks can be viewed as discretization of a continuous-time dynamical system, and hence supervised deep learning can be regarded as solving an optimal control problem in continuous time.

- When is a Convolutional Filter Easy To Learn? [[paper]](https://arxiv.org/abs/1709.06129)
  - Simon S. Du, Jason D. Lee, Yuandong Tian. *ICLR 2018*
  - Key Word: Gradient Descent.
  - <details><summary>Digest</summary> We show that (stochastic) gradient descent with random initialization can learn the convolutional filter in polynomial time and the convergence rate depends on the smoothness of the input distribution and the closeness of patches. To the best of our knowledge, this is the first recovery guarantee of gradient-based algorithms for convolutional filter on non-Gaussian input distributions.  

- Implicit Regularization in Deep Learning. [[paper]](https://arxiv.org/abs/1709.01953)
  - Behnam Neyshabur. *PhD Thesis*
  - Key Word: Implicit Regularization.
  - <details><summary>Digest</summary> In an attempt to better understand generalization in deep learning, we study several possible explanations. We show that implicit regularization induced by the optimization method is playing a key role in generalization and success of deep learning models. Motivated by this view, we study how different complexity measures can ensure generalization and explain how optimization algorithms can implicitly regularize complexity measures.

- Exploring Generalization in Deep Learning. [[paper]](https://arxiv.org/abs/1706.08947) [[code]](https://github.com/bneyshabur/generalization-bounds)
  - Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, Nathan Srebro. *NeurIPS 2017*
  - Key Word: PAC-Bayes.
  - <details><summary>Digest</summary> With a goal of understanding what drives generalization in deep networks, we consider several recently suggested explanations, including norm-based control, sharpness and robustness. We study how these measures can ensure generalization, highlighting the importance of scale normalization, and making a connection between sharpness and PAC-Bayes theory. We then investigate how well the measures explain different observed phenomena.

- Gradient Descent Can Take Exponential Time to Escape Saddle Points. [[paper]](https://arxiv.org/abs/1705.10412)
  - Simon S. Du, Chi Jin, Jason D. Lee, Michael I. Jordan, Barnabas Poczos, Aarti Singh. *NeurIPS 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - <details><summary>Digest</summary> We established the failure of gradient descent to efficiently escape saddle points for general non-convex smooth functions. We showed that even under a very natural initialization scheme, gradient descent can require exponential time to converge to a local minimum whereas perturbed gradient descent converges in polynomial time. Our results demonstrate the necessity of adding perturbations for efficient non-convex optimization.

- Stochastic Gradient Descent as Approximate Bayesian Inference. [[paper]](https://arxiv.org/abs/1704.04289#)
  - Stephan Mandt, Matthew D. Hoffman, David M. Blei. *JMLR*
  - Key Word: Stochastic Gradient Descent; Stochastic Differential Equations; Ornstein-Uhlenbeck Process.
  - <details><summary>Digest</summary> The article discusses the use of Stochastic Gradient Descent (SGD) with a constant learning rate as a simulation of a Markov chain with a stationary distribution. This perspective leads to several new findings, including using constant SGD as an approximate Bayesian posterior inference algorithm by adjusting tuning parameters to match the stationary distribution to a posterior. Additionally, constant SGD can optimize hyperparameters in complex probabilistic models and be used for sampling with momentum. The article also analyzes MCMC algorithms and provides a proof of why Polyak averaging is optimal. Finally, a scalable approximate MCMC algorithm, the Averaged Stochastic Gradient Sampler, is proposed based on this stochastic process perspective.

- How to Escape Saddle Points Efficiently. [[paper]](https://arxiv.org/abs/1703.00887)
  - Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, Michael I. Jordan. *ICML 2017*
  - Key Word: Gradient Descent; Saddle Points.
  - <details><summary>Digest</summary> This paper presents the first (nearly) dimension-free result for gradient descent in a general nonconvex setting. We present a general convergence result and show how it can be further strengthened when combined with further structure such as strict saddle conditions and/or local regularity/convexity.

### Others: 2016

- Understanding Deep Neural Networks with Rectified Linear Units. [[paper]](https://arxiv.org/abs/1611.01491)
  - Raman Arora, Amitabh Basu, Poorya Mianjy, Anirbit Mukherjee. *ICLR 2018*
  - Key Word: ReLU.
  - <details><summary>Digest</summary> In this paper we investigate the family of functions representable by deep neural networks (DNN) with rectified linear units (ReLU). We give an algorithm to train a ReLU DNN with one hidden layer to *global optimality* with runtime polynomial in the data size albeit exponential in the input dimension. Further, we improve on the known lower bounds on size (from exponential to super exponential) for approximating a ReLU deep net function by a shallower ReLU net.

- Deep Information Propagation. [[paper]](https://arxiv.org/abs/1611.01232)
  - Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, Jascha Sohl-Dickstein. *ICLR 2017*
  - Key Word: Mean Field Theory.
  - <details><summary>Digest</summary> We study the behavior of untrained neural networks whose weights and biases are randomly distributed using mean field theory. We show the existence of depth scales that naturally limit the maximum depth of signal propagation through these random networks. Our main practical result is to show that random networks may be trained precisely when information can travel through them. Thus, the depth scales that we identify provide bounds on how deep a network may be trained for a specific choice of hyperparameters.

- Why Deep Neural Networks for Function Approximation? [[paper]](https://arxiv.org/abs/1610.04161)
  - Shiyu Liang, R. Srikant. *ICLR 2017*
  - Key Word: Function Approximation.
  - <details><summary>Digest</summary> Recently there has been much interest in understanding why deep neural networks are preferred to shallow networks. We show that, for a large class of piecewise smooth functions, the number of neurons needed by a shallow network to approximate a function is exponentially larger than the corresponding number of neurons needed by a deep network for a given degree of function approximation. First, we consider univariate functions on a bounded interval and require a neural network to achieve an approximation error of ε uniformly over the interval. We show that shallow networks (i.e., networks whose depth does not depend on ε) require Ω(poly(1/ε)) neurons while deep networks (i.e., networks whose depth grows with 1/ε) require O(polylog(1/ε)) neurons. 

- Why does deep and cheap learning work so well? [[paper]](https://arxiv.org/abs/1608.08225)
  - Henry W. Lin, Max Tegmark, David Rolnick. *Journal of Statistical Physics*
  - Key Word: Physics.
  - <details><summary>Digest</summary> We show how the success of deep learning could depend not only on mathematics but also on physics: although well-known mathematical theorems guarantee that neural networks can approximate arbitrary functions well, the class of functions of practical interest can frequently be approximated through "cheap learning" with exponentially fewer parameters than generic ones. We explore how properties frequently encountered in physics such as symmetry, locality, compositionality, and polynomial log-probability translate into exceptionally simple neural networks.

- Exponential expressivity in deep neural networks through transient chaos. [[paper]](https://arxiv.org/abs/1606.05340) [[code]](https://github.com/ganguli-lab/deepchaos)
  - Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, Surya Ganguli. *NeurIPS 2016*
  - Key Word: Mean Field Theory; Riemannian Geometry.
  - <details><summary>Digest</summary> We combine Riemannian geometry with the mean field theory of high dimensional chaos to study the nature of signal propagation in deep neural networks with random weights. Our results reveal a phase transition in the expressivity of random deep networks, with networks in the chaotic phase computing nonlinear functions whose global curvature grows exponentially with depth, but not with width. We prove that this generic class of random functions cannot be efficiently computed by any shallow network, going beyond prior work that restricts their analysis to single functions.

## Related Resources

- [Off the Convex Path Blog](https://www.offconvex.org/)

- [Machine Learning Research Blog](https://francisbach.com/)

- [Awesome Information Bottleneck Paper List](https://github.com/ZIYU-DEEP/Awesome-Information-Bottleneck) ![ ](https://img.shields.io/github/stars/ZIYU-DEEP/Awesome-Information-Bottleneck) ![ ](https://img.shields.io/github/last-commit/ZIYU-DEEP/Awesome-Information-Bottleneck)

- [Neural Tangent Kernel Papers](https://github.com/kwignb/NeuralTangentKernel-Papers) ![ ](https://img.shields.io/github/stars/kwignb/NeuralTangentKernel-Papers) ![ ](https://img.shields.io/github/last-commit/kwignb/NeuralTangentKernel-Papers)

- [Awesome Trustworthy Deep Learning](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning) ![ ](https://img.shields.io/github/stars/MinghuiChen43/awesome-trustworthy-deep-learning) ![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-trustworthy-deep-learning)
