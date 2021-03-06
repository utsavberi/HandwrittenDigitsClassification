import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data
    #Your code here
    train_data = np.array([], dtype=np.dtype('f')).reshape(0, 784)
    validation_data = np.array([], dtype=np.dtype('f')).reshape(0, 784)
    test_data = np.array([], dtype=np.dtype('f')).reshape(0, 784)
    train_label = np.array([])
    validation_label = np.array([])
    test_label = np.array([])

    for i in range(0, 10):
        train_data = np.vstack((train_data, mat["train" + str(i)][0:5000]))
        train_label = np.append(train_label, np.repeat(i, mat["train" + str(i)][0:5000].shape[0]))
        validation_data = np.vstack((validation_data, mat["train" + str(i)][5000:]))
        validation_label = np.append(validation_label, np.repeat(i, mat["train" + str(i)][5000:].shape[0]))
        test_data = np.vstack((test_data, mat["test" + str(i)]))
        test_label = np.append(test_label, np.repeat(i, mat["test" + str(i)].shape[0]))

    def convert_label_to_2d(arr):
        tmp = np.array([[]]).reshape(0,10)
        for i in range(0,arr.size):
            tmp2 = np.repeat(0,10)
            tmp2[arr[i]]=1
            tmp = np.vstack((tmp,tmp2))
        return tmp

    train_label = convert_label_to_2d(train_label)#tmp
    validation_label = convert_label_to_2d(validation_label)
    test_label = convert_label_to_2d(test_label)

    #normalize data
    np.place(train_data, train_data > 0, 1)
    np.place(validation_data,  validation_data > 0, 1)
    np.place(test_data, test_data > 0, 1)

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    m = training_data.shape[0]
    Y = np.eye(n_class)[training_label]
    h = nnPredictHelper(w1,w2,training_data)
    cost = (-Y * np.log(h).T )- ((1 - Y) * np.log(1 - h).T)
    obj_val = np.sum(cost) / m

    if lambdaval != 0:
        t1f = w1[:, 1:]
        t2f = w2[:, 1:]
        reg = (lambdaval / (2 * m)) * (np.sum(t1f)**2 + np.sum(t2f)**2)
        obj_val = obj_val + reg

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    return (obj_val, obj_grad)

def nnPredictHelper(w1,w2,data):
    bias_col = np.ones_like(data[:,-1]).reshape(-1, 1)
    data_bias = np.hstack((data, bias_col))
    data_bias_w1 = sigmoid(np.dot(data_bias,np.transpose(w1)))
    data_bias_w1_bias = np.hstack((data_bias_w1,bias_col))
    labels = sigmoid(np.dot(data_bias_w1_bias,np.transpose(w2)))
    return labels

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    # labels = np.array([])
    # Your code here
    labels = nnPredictHelper(w1,w2,data)
    amax = np.argmax(labels,1);
    tmparr = np.array([[]]).reshape(0,10)
    for i in range(0,amax.size):
        tmptmp = np.repeat(0,10);
        tmptmp[amax[i]] = 1;
        tmparr = np.vstack((tmparr,tmptmp))
    # return labels
    # print tmparr == labels.argmax(0)
    return tmparr


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();


# Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0;

# args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

# nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = initial_w1#nn_params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = initial_w2#nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

#find the accuracy on Training Dataset

print '\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%'

predicted_label = nnPredict(w1, w2, validation_data)

#find the accuracy on Validation Dataset

print '\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%'

predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset

print '\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%'
