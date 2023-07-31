# DeepLr CNNPartiallySharedKernels
 The proposed approach you have a bigger weight matrix (equal size to the input image/feature map) that is randomly initialized. From this matrix, each time you have to select a receptive field as a weight kernel that will do dot product with the same position receptive field in the input image/feature map to calculate a weighted output value for an output neuron.

Code Files:
1. CT5135_Group 16_Assignment_2.ipynb

# University of Galway
## Research Topics in AI (CT5135)

You are working as a research engineer or a data scientist in a company. Your team lead asked you to implement a new neural network. The network is like a Convolutional Neural Network (CNN). The difference is that instead of a Convolution layer in a CNN, you have to introduce a new layer to bring novelty in their model.

Below you can find an image from a PowerPoint (PowerPoint is attached) in which the difference between the traditional 2DCNN kernel operation is shown vs the proposed approach that you have to implement.

<p align="center">
<img src="Screenshot 2023-07-31 204516.png" alt="ProposedCNNModel">
</p>


**Forward propagation:** The main difference is that in a CNN convolve layer, you have a small (e.g. 3x3, 5x5, or 7x7) weight kernel that slides over the whole input image to generate output feature map. This weight kernel is randomly initialized. You can use more than one weight kernel to generate multiple output feature maps.

Whereas, in the proposed approach you have a bigger weight matrix (equal size to the input image/feature map) that is randomly initialized. From this matrix, each time you have to select a receptive field as a weight kernel that will do dot product with the same position receptive field in the input image/feature map (as shown in Figure/ppt) to calculate a weighted output value for an output neuron. The same process is repeated until all the neurons/pixels/values in the output map/s are calculated. 

**Backward Pass (to calculate/update weights):** Once you reach the output layer, you have to use the backpropagation algorithm to backpropagate the error that will be using the same receptive field i.e., each output neuron will use the same weights in the respected receptive field to calculate gradients (or local error) that it used in forward propagation (this time in backward direction). Backpropagation is the step that apply the chain rule to compute the gradient of the loss function with respect to the inputs.

Once you figure how to do that, then create the following network as a baseline:<br>

Input layer – ProposedLayer1– ActivationLayer1 – PoolingLayer1 – ProposedLayer2 –
ActivationLayer2 – PoolingLayer2 – ProposedLayer3 – ActivationLayer3 –
FullyConnectedLayer1/DenseLayer1 – OutputLayer/Softmax


* At ProposedLayer1 use 16 weight matrices to get 16 feature maps while using a 3x3 receptive field and stride of 1.
* At ProposedLayer2 use 12 weight matrices to get 12 feature maps while using a 3x3 receptive field and stride of 1.
* At ProposedLayer3 use 8 weight matrices to get 8 feature maps while using a 3x3 receptive field and stride of 1.
* At ActivationLayer1, ActivationLayer2, and ActivationLayer3 use ReLU activation or other recent functions.
* At PoolingLayer1 and PoolingLayer2, use 2x2 receptive field.

Once you implement the above model, distribute the data in training, validation, and testing
set, and train and test it to bring optimal accuracy, precision, and recall.