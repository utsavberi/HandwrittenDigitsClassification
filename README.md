# HandwrittenDigitsClassification
Handwritten digits recognition using neural network

##File Structure
<ul>
<li>mnist all.mat: original dataset from MNIST. In this file, there are 10 matrices for testing set and 10
matrices for training set, which corresponding to 10 digits. You will have to split the training data
into training and validation data.
<li>nnScript.py: Python script for this programming project. Contains function definitions -
<ul>
<li> preprocess(): performs some preprocess tasks, and output the preprocessed train, validation and
test data with their corresponding labels.
<li>sigmoid(): compute sigmoid function. The input can be a scalar value, a vector or a matrix. 
<li>nnObjFunction(): compute the error function of Neural Network. 
<li>nnPredict(): predicts the label of data given the parameters of Neural Network.
<li>initializeWeights(): return the random weights for Neural Network given the number of unit in
the input layer and output layer.
</ul>
</ul>

##Datasets
The MNIST dataset [1] consists of a training set of 60000 examples and test set of 10000 examples. All
digits have been size-normalized and centered in a fixed image of 28 × 28 size. In original dataset, each pixel
in the image is represented by an integer between 0 and 255, where 0 is black, 255 is white and anything
between represents different shade of gray.
In many research papers, the official training set of 60000 examples is divided into an actual training set of
50000 examples and validation set of 10000 examples.

#Neural Network
##3.2.1 Neural Network Representation
Neural network can be graphically represented as in Figure 1.
As observed in the Figure 1, there are totally 3 layers in the neural network:
• The first layer comprises of (d + 1) units, each represents a feature of image (there is one extra unit
representing the bias).
• The second layer in neural network is called the hidden units. In this document, we denote m + 1
as the number of hidden units in hidden layer. There is an additional bias node at the hidden layer
as well. Hidden units can be considered as the learned features extracted from the original data set.
Since number of hidden units will represent the dimension of learned features in neural network, it’s
our choice to choose an appropriate number of hidden units. Too many hidden units may lead to the
slow training phase while too few hidden units may cause the the under-fitting problem.
• The third layer is also called the output layer. The value of l th unit in the output layer represents
the probability of a certain hand-written image belongs to digit l. Since we have 10 possible digits,
there are 10 units in the output layer. In this document, we denote k as the number of output units
in output layer.
The parameters in Neural Network model are the weights associated with the hidden layer units and the
output layers units. In our standard Neural Network with 3 layers (input, hidden, output), in order to
represent the model parameters, we use 2 matrices:
• W (1) ∈ R m×(d+1) is the weight matrix of connections from input layer to hidden layer. Each row in
this matrix corresponds to the weight vector at each hidden layer unit.
• W (2) ∈ R k×(m+1) is the weight matrix of connections from hidden layer to output layer. Each row in
this matrix corresponds to the weight vector at each output layer unit.
We also further assume that there are n training samples when performing learning task of Neural Network.
In the next section, we will explain how to perform learning in Neural Network.

##3.2.2 Feedforward Propagation
In Feedforward Propagation, given parameters of Neural Network and a feature vector x, we want to compute
the probability that this feature vector belongs to a particular digit.
Suppose that we have totally m hidden units. Let a j for 1 ≤ j ≤ m be the linear combination of input
data and let z j be the output from the hidden unit j after applying an activation function (in this exercise,
we use sigmoid as an activation function). For each hidden unit j (j = 1, 2, · · · , m), we can compute its
value as follow:
D+1
(1)
a j =
w ji x i
(1)
i=1
z j = σ(a j ) =
1
1 + exp(−a j )
(2)
(1)
where w ji = W (1) [j][i] is the weight of connection from unit i in input layer to unit j in hidden layer. Note
that we do not compute the output for the bias hidden node (m + 1); z m+1 is directly set to 1.
The third layer in neural network is called the output layer where the learned features in hidden units
are linearly combined and a sigmoid function is applied to produce the output. Since in this assignment,
we want to classify a hand-written digit image to its corresponding class, we can use the one-vs-all binary
classification in which each output unit l (l = 1, 2, · · · , 10) in neural network represents the probability of an
image belongs to a particular digit. For this reason, the total number of output unit is k = 10. Concretely,
for each output unit l (l = 1, 2, · · · , 10), we can compute its value as follow:
m+1
(2)
w lj z j
b l =
(3)
j=1
o l = σ(b l ) =
1
1 + exp(−b l )
Now we have finished the Feedforward pass.

##3.2.3 Error function and Backpropagation
The error function in this case is the negative log-likelihood error function which can be written as follow:
J(W (1) , W (2) ) = −
1
n
n
k
(y il ln o il + (1 − y il ) ln(1 − o il ))
(5)
i=1 l=1
where y il indicates the l th target value in 1-of-K coding scheme of input data i and o il is the output at l th
output node for the i th data example (See (4)).
Because of the form of error function in equation (5), we can separate its error function in terms of error
for each input data x i :
n
1
J i (W (1) , W (2) )
(6)
J(W (1) , W (2) ) =
n i=1
where
k
J i (W (1) , W (2) ) =
(y il ln o il + (1 − y il ) ln(1 − o il ))
(7)
l=1
One way to learn the model parameters in neural networks is to initialize the weights to some random
numbers and compute the output value (feed-forward), then compute the error in prediction, transmits this
error backward and update the weights accordingly (error backpropagation).
The feed-forward step can be computed directly using formula (1), (2), (3) and (4).
On the other hand, the error backpropagation step requires computing the derivative of error function
with respect to the weight.
Consider the derivative of error function with respect to theweight from the hidden unit j to output
unit l where j = 1, 2, · · · , m + 1 and l = 1, · · · , 10:
  ∂J i
  =
  (2)
  ∂w lj
  ∂J i ∂o l ∂b l
  ∂o l ∂b l ∂w (2)
  = δ l z j
  where
  δ l =
  (8)
  lj
  (9)
  ∂J i ∂o l
  y l
  1 − y l
  = −( −
  )(1 − o l )o l = o l − y l
  ∂o l ∂b l
  o l
  1 − o l
Note that we are dropping the subscript i for simplicity. The error function (log loss) that we are using
in (5) is different from the the squared loss error function that we have discussed in class. Note that the
choice of the error function has “simplified” the expressions for the error!
On the other hand, the derivative of error function with respect to the weight from the input unit l to
output unit j where p = 1, 2, · · · , d + 1 and j = 1, · · · , m can be computed as follow:
  k
  ∂J i
  (1)
  ∂w jp
  =
  l=1
  ∂J i ∂o l ∂b l ∂z j ∂a j
  ∂o l ∂b l ∂z j ∂a j ∂w (1)
  jp
  (10)
  k
  (2)
  δ l w lj (1 − z j )z j x p
  =
  (11)
  l=1
  k
  =
  (2)
  (1 − z j )z j (
  δ k w kj )x p
  (12)
Note that we do not compute the gradient for the weights at the bias hidden node.
After finish computing the derivative of error function with respect to weight of each connection in neural
network, we now can write the formula for the gradient of error function:
  ∇J(W
  (1)
  ,W
  (2)
  1
  )=
  n
  4
  n
  ∇J i (W (1) , W (2) )
  i=1
  (13)
We again can use the gradient descent to update each weight (denoted in general as w) with the following
rule:
  w new = w old − γ∇E(w old )

##3.2.4 Regularization in Neural Network
In order to avoid overfitting problem (the learning model is best fit with the training data but give poor
generalization when test with validation data), we can add a regularization term into our error function to
control the magnitude of parameters in Neural Network. Therefore, our objective function can be rewritten
as follow:
  
  
  m d+1
  k m+1
  λ 
  (1)
  (2)
  (w ) 2 +
  (15)
  (w lj ) 2 
  J(W (1) , W (2) ) = J(W (1) , W (2) ) +
  2n j=1 p=1 jp
  j=1
  l=1
where λ is the regularization coefficient.
With this new objective function, the partial derivative of new objective function with respect to weight
from hidden layer to output layer can be calculated as follow:
  ∂ J
  (2)
  ∂w lj
  =
  1
  n
  n
  ∂J i
  (2)
  i=1 ∂w lj
  (2)
  + λw lj
  (16)
Similarly, the partial derivative of new objective function with respect to weight from input layer to hidden
layer can be calculated as follow:
  ∂ J
  (1)
  ∂w jp
  =
  1
  n
  n
  ∂J i
  (1)
  i=1 ∂w jp
  (1)
  + λw jp
  (17)
With this new formulas for computing objective function (15) and its partial derivative with respect to
weights (16) (17) , we can again use gradient descent to find the minimum of objective function.
