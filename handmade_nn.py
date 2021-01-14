import pandas as pd
import numpy as np

class handmade_nn ():
    def __init__ (self, input_dim=0):
        self.weights=[]
        self.bias=[]
        self.activation_types=[]
        self.input_dim=input_dim

    def activation(self,x,activation_type):
        # Defining activation functions
        # Takes a nparray or a single value
        # Returns in the same format
        if activation_type == 'relu':
            return np.maximum(x,0)
        elif activation_type == 'sigmoid':
            return 1/(1+np.exp(-x))
        elif activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'linear':
            return x
        elif activation_type == 'softmax':
            exp_x = np.exp(x)
            return exp_x / exp_x.sum()

        #raise error if unknown type
        else:
            raise ValueError(f'Unknow activation type {activation_type}. Supported types : linear, relu, sigmoid, tanh, softmax')

    def set_input_dim (self,input_dim):
        self.input_dim = input_dim

    def add_dense_layer (self, n_neurons, activation_type):
        #check if the input_dim is set
        if self.input_dim == 0:
            raise ValueError('input_dim = 0 . Use set_input_dim before creating first layer')

        #get the size of the input os this layer
        if len(self.bias) == 0:
            previous_dim=self.input_dim
        else:
            previous_dim=(self.bias[-1].shape[0])

        #initialize the layer parameters
        self.weights.append(np.zeros((n_neurons, previous_dim)))
        self.bias.append(np.expand_dims(np.zeros(n_neurons), axis=0))
        self.activation_types.append(activation_type)

        #test the activation type
        self.activation(0, activation_type)

    def predict (self,X):
        #converting DataFrames, lists or lists of lists to nparray
        X = np.array(X)

        #deal with 1D inputs to forge a 1*n_features 2D-array
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis = 0)

        #raise errors for unconsistant inputs
        if len(X.shape) > 2:
            raise ValueError('X vector dimension too high. Must be 2 max')
        if X.shape[1] != self.input_dim:
            raise ValueError(f'Unconsistent number of features. The network input_dim is {self.input_dim}')

        #compute the prediction
        for layer_index, activation_type in enumerate(self.activation_types):
            activation_input=np.dot(self.weights[layer_index],X.T) + self.bias[layer_index].T
            X = self.activation(activation_input, activation_type).T
        return X


if __name__ == "__main__":
    #test all the features of the handmade_nn class
    from unittest import TestCase

    ## tests for method add_dense_layer

    my_first_nn=handmade_nn()
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.add_dense_layer(5,'relu')
    assert 'input_dim = 0 . Use set_input_dim before creating first layer' in str(context.exception),\
        "no or wrong Exception raised when adding first layer to a network without setting input_dim"

    my_first_nn=handmade_nn(5)
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.add_dense_layer(10,'typo_error')
    assert 'Unknow activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing an unvalid activation_type"


    ## input compatibility tests for method predict

    my_first_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs

    assert my_first_nn.predict([2,3,2,3,4]).shape == (1,5),\
        "list not supported as an input for predict"

    assert my_first_nn.predict([[2,3,2,3,4],[-2,-1,1,3,4]]).shape == (2,5),\
        "list of list not supported as an input for predict"

    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.predict(np.array([[[1,1],[1,2],[1,3],[1,4],[1,5]],
                                    [[2,1],[2,2],[2,3],[3,4],[3,5]]]))
    assert 'X vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array in predict method"

    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.predict(np.array([[1,1],[1,2],[1,3],[1,4],[1,5]]))
    assert 'Unconsistent number of features' in str(context.exception),\
        "no or wrong Exception raised when inputing a X with unconsistant size vs. network input_dim"

    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'linear')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert my_first_nn.predict(np.array([-2,-1,2,3,4])).shape == (1,10),\
        "1-D array not supported as an input for predict"

    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'linear')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert my_first_nn.predict(np.array([[-2,-1,2,3,4],[-12,-11,12,13,14]])).shape == (2,10),\
        "the shape of the prediction for a 2*5 X input by a network having 10neurons on last layer should be 2*10"


    ## testing activation functions

    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'relu')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert (my_first_nn.predict([-2,-1,2,3,4]) ==\
            np.array([[0., 0., 2., 3., 4., 0., 0., 0., 0., 0.]]))\
            .all(),\
        "uncorrect relu function behaviour"

    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'sigmoid')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert (np.round(my_first_nn.predict([-2,-1,2,3,4]), 8) ==\
            np.array([[0.11920292, 0.26894142, 0.88079708, 0.95257413, 0.98201379,
            0.5       , 0.5       , 0.5       , 0.5       , 0.5       ]]))\
            .all(),\
        "uncorrect sigmoid function behaviour"

    print ('all tests successfully passed')



