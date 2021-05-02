# Deep Learning
> The central goal of AI is to provide a set of algorithms and techniques that can be used to solve problems that humans perform intuitively and near automatically, but are otherwise very challenging for computers.

- Artificial Intelligence = automatic machine reasoning (inferencing, planning, heuristics, etc.)
    - Machine Learning = specifically interested in pattern recognition and learning from data
        - Deep Learning = Artificial Neural Network (ANN) learn from data and specialize in pattern recognition
           - backpropagation = cornerstone algorithm of modern day neural networks, learning from mistakes
    
Semantic Gap = deeper understanding

## Single Layer Networks
- solves linear problems

## Multi-Layer (Neural) Networks
- train via backpropogation
- hidden layers  
- can solve non-linear (most real world problems) problems WHEN the correct activation function used
- ReLu enhanced deeper network 

## ML vs. DL
|Machine Learning|Deep Learning|
|-|-|
|Input images|Input images (raw pixel intensities data)|
|Handcraft features extraction algorithms|Simple features (edges)|
|ML Classifier|Intermediate features (corners)|
|Output|Abstract features (object parts|
||Output|
|More data != better learning|More data == better learning|

## Hierarchical Feature Learning
1. supervised = both inputs and outputs
1. unsupervised = auto discover features without hints
1. semisupervised = mix

## Then vs. Now
|Then|Now|
|-|-|
|datasets too small|large, labeled datasets|
|computers too slow|faster computers (e.g. GPUs)|
|inferior weight init|better weights init|
|wrong type of non-linear activation|superior activation functions|

# Machine Learning
- Definition computers should be able to learn from experience (i.e., examples) of the problem they are trying to solve.

# Deep Learning
- Definition of deep changes weekly. Define based on types of layers and network architectures (CNNs, RNNs, LTSMs).
  - Using deep learning, we try to understand the problem in terms of a hierarchy of concepts that removes the need for hand-designed feature extraction. 
  - Convolutional Neural Network (CNN)
Reference: https://www.pyimagesearch.com/2021/04/17/what-is-deep-learning/

# Image Classification
> also exists in object detection, instance segmentation, supervised, semi-supervised, unsupervised learning)

Definition is assigning a label to an image from a predefined set of categories.
- More formally, given our input image of W×H pixels with three channels, Red, Green, and Blue, respectively, our goal is to take the W×H×3 = N pixel image and figure out how to correctly classify the contents of the image.

Goal:
- apply machine learning and deep learning algorithms to discover underlying patterns in the dataset, enabling us to correctly classify data points that our algorithm has not encountered yet.

Challenges: 
1. images = numpy arrays (matrix of numbers)...semantic gaps is what the computer sees and understands.
1. viewpoint variation = different viewing angles
1. scale variation = relative size (near vs. far)
1. deformation = bending, changing, deforming
1. occlusion variation = (HARDEST), partially blocking object from camera
1. background clutter = where's waldo
1. illumination variation = (HARD) lighting conditions impact (color, edges, etc.) 
1. intra-class variation = how many different "chairs" are there
1. combination of the above (HARDEST-est)

Reference: https://www.pyimagesearch.com/2021/04/17/image-classification-basics/

# Image Classification Pipeline
> dataset == what we are trying to extract knowledge from; collection of datapoints (e.g. collection of images, labels, etc.). datapoint == item in the dataset (each image.

1. gather dataset (dataset directory, with subdirs of each category...uniform sub-totals)
1. split dataset (train [learn] : test [evaluate] | 75% : 25% | 80% : 20% | 66.7% : 33.3% | )
   - validation set [fake test] tune hyperparameters = 10-20% of the training set.
1. train network/model (epoch) 
1. evaluate (compare to ground truth)

Reference: https://www.pyimagesearch.com/2021/04/17/the-deep-learning-classification-pipeline/

# Considerations

- what is the total size of the dataset in bytes?
- is it large enough to fit in available RAM?
- does it exceed my current machine?

- Machine learning algorithms such as k-NN, SVMs, and even Convolutional Neural Networks require all images in a dataset to have a fixed feature vector size.
- n the case of images, this requirement implies that our images must be preprocessed and scaled to have identical widths and heights.
