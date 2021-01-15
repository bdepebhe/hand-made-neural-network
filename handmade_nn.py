import pandas as pd
import numpy as np

## Functions

def compute_activation (X, activation_type):
    # Defining activation functions
    # Takes a nparray or a single value
    # Returns in the same format
    X=np.array(X)
    if activation_type == 'relu':
        return np.maximum(X,0)
    elif activation_type == 'sigmoid':
        return 1/(1+np.exp(-X))
    elif activation_type == 'tanh':
        return np.tanh(X)
    elif activation_type == 'linear':
        return X
    elif activation_type == 'softmax':
        exp_x = np.exp(X)
        return exp_x / exp_x.sum()

    #raise error if unknown type
    else:
        raise ValueError(f'Unknown activation type {activation_type}.\
                           Supported types : linear, relu, sigmoid, tanh, softmax')

def compute_metric (y, y_pred, metric):
    # Defining loss and metric functions
    # Takes a nparray, a list or a single value
    # Always returns a scalar : in case of multioutputs (y.shape[1]>1),
    #     uniform average of the errors along axis1

    #converting DataFrames, lists or lists of lists to nparray
    y = np.array(y)
    y_pred = np.array(y_pred)

    #deal with 1D inputs to forge a n_samples * 1 2D-array
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis = 1)
    if len(y_pred.shape) == 1:
        y_pred = np.expand_dims(y_pred, axis = 1)

    #raise errors for unconsistant inputs
    if len(y.shape) > 2:
        raise ValueError('y vector dimension too high. Must be 2 max')
    if len(y_pred.shape) > 2:
        raise ValueError('y_pred vector dimension too high. Must be 2 max')
    if y.shape != y_pred.shape:
        raise ValueError(f'unconsistent vectors dimensions during scoring :\
                           y.shape= {y.shape} and y_pred.shape= {y_pred.shape}')

    #compute loss funtions
    if metric == 'mse':
        return np.square(y-y_pred).mean()
    elif metric == 'mae':
        return np.abs(y-y_pred).mean()
    elif metric == 'categorical_crossentropy':
        return -1/y.shape[0] * (np.multiply(y, np.log(y_pred)).sum())
    elif metric == 'binary_crossentropy':
        if y.shape[1]>1:
            raise ValueError('y vector dimension too high.\
                              Must be 1 max for binary_crossentropy')
        return -(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)).mean()

    # compute other metrics functions
    ## TODO ## accuracy, f1-score, recall, etc..
    else:
        raise ValueError(f'Unknown metric {metric}. Supported types :\
                           mse, mae, categorical_crossentropy, binary_crossentropy')

class handmade_nn ():
    '''
    hand-made version of neural network
    so far, the possibilities are :

        - layers activation functions :
            'linear', 'relu', 'sigmoid', 'tanh', 'softmax'

        - loss functions :
            'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy'

        - solver :
            SGD without momentum
    '''
    def __init__ (self,
                  input_dim=0,
                  loss=None,
                  learning_rate=0.01,
                  scoring=None):
        self.weights=[]
        self.bias=[]
        self.activation_types=[]
        self.input_dim=input_dim
        self.loss=loss
        self.learning_rate=learning_rate
        self.scoring=scoring

    def set_input_dim (self,input_dim):
        self.input_dim = input_dim

    def set_loss (self,loss):
        self.loss = loss

    def set_learning_rate (self,learning_rate):
        self.learning_rate = learning_rate

    def add_dense_layer (self, n_neurons, activation_type):
        #check if the input_dim is set
        if self.input_dim == 0:
            raise ValueError('input_dim = 0 .\
                              Use set_input_dim before creating first layer')

        #get the size of the input os this layer
        if len(self.bias) == 0:
            previous_dim=self.input_dim
        else:
            previous_dim=(self.bias[-1].shape[0])

        #initialize the layer parameters
        self.weights.append(np.zeros((n_neurons, previous_dim)))
        self.bias.append(np.expand_dims(np.zeros(n_neurons), axis=1))
        self.activation_types.append(activation_type)

        #test the activation type
        compute_activation(0, activation_type)

    def predict (self,X):
        #converting DataFrames, lists or lists of lists to nparray
        X = np.array(X)

        #deal with 1D inputs to forge a 1 * n_features 2D-array
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis = 0)

        #raise errors for unconsistant inputs
        if len(X.shape) > 2:
            raise ValueError('X vector dimension too high. Must be 2 max')
        if X.shape[1] != self.input_dim:
            raise ValueError(f'Unconsistent number of features.\
                               The network input_dim is {self.input_dim}')

        #compute the prediction
        for layer_index, activation_type in enumerate(self.activation_types):
            activation_input = np.dot(self.weights[layer_index], X.T)\
                               + self.bias[layer_index]
            X = compute_activation(activation_input, activation_type).T
        return X


if __name__ == "__main__":
    #test all the features of the handmade_nn class
    from unittest import TestCase


    #---------------------------------------------------------------------------
    ## tests for function compute_activation
    #---------------------------------------------------------------------------

    assert (compute_activation([-2,-1,2,3,4],'relu') ==\
            np.array([[0, 0, 2, 3, 4]]))\
            .all(), "uncorrect relu function behaviour"

    #---------------------------------------------------------------------------
    assert (compute_activation([-2,-1,2,3,4],'linear') ==\
            np.array([[-2, -1, 2, 3, 4]]))\
            .all(), "uncorrect linear function behaviour"

    #---------------------------------------------------------------------------
    assert (np.round(compute_activation([-2,-1,2,3,4],'sigmoid'), decimals= 8) ==\
            np.array([[0.11920292, 0.26894142, 0.88079708, 0.95257413, 0.98201379]]))\
            .all(), "uncorrect sigmoid function behaviour"

    #---------------------------------------------------------------------------
    assert (np.round(compute_activation([-2,-1,2,3,4],'tanh'), decimals= 8) ==\
            np.array([[-0.96402758, -0.76159416,  0.96402758,  0.99505475,  0.9993293]]))\
            .all(), "uncorrect tanh function behaviour"

    #---------------------------------------------------------------------------
    assert (np.round(compute_activation([-2,-1,2,3,4],'softmax'), decimals= 8) ==\
            np.array([[0.00163892, 0.00445504, 0.08948193, 0.24323711, 0.66118700]]))\
            .all(), "uncorrect softmax function behaviour"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_activation(0,'typo_error')
    assert 'Unknown activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing an unknown activation_type\
         while calling compute_activation"


    #---------------------------------------------------------------------------
    ## tests for method add_dense_layer
    #---------------------------------------------------------------------------

    my_first_nn=handmade_nn()
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.add_dense_layer(5,'relu')
    assert 'Use set_input_dim before creating first layer'\
           in str(context.exception),\
        "no or wrong Exception raised when adding first layer\
         to a network without setting input_dim"

    #---------------------------------------------------------------------------
    my_first_nn=handmade_nn(5)
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.add_dense_layer(10,'typo_error')
    assert 'Unknown activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing\
         an unknown activation_type while adding layer"


    #---------------------------------------------------------------------------
    ## tests for method predict ################################################
    #---------------------------------------------------------------------------

    my_first_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs

    assert my_first_nn.predict([2,3,2,3,4]).shape == (1,5),\
        "list not supported as an input for predict"
    #---------------------------------------------------------------------------

    assert my_first_nn.predict([[2,3,2,3,4],[-2,-1,1,3,4]]).shape == (2,5),\
        "list of list not supported as an input for predict"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.predict(np.array([[[1,1],[1,2],[1,3],[1,4],[1,5]],
                                    [[2,1],[2,2],[2,3],[3,4],[3,5]]]))
    assert 'X vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array in predict method"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        my_first_nn.predict(np.array([[1,1],[1,2],[1,3],[1,4],[1,5]]))
    assert 'Unconsistent number of features' in str(context.exception),\
        "no or wrong Exception raised when inputing a X\
         with unconsistant size vs. network input_dim in predict method"

    #---------------------------------------------------------------------------
    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'linear')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert my_first_nn.predict(np.array([-2,-1,2,3,4])).shape == (1,10),\
        "1-D array not supported as an input for predict method"

    #---------------------------------------------------------------------------
    my_first_nn=handmade_nn(5)
    my_first_nn.add_dense_layer(10, 'linear')
    my_first_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    assert my_first_nn.predict(np.array([[-2,-1,2,3,4],
                                         [-12,-11,12,13,14]])).shape == (2,10),\
        "the shape of the prediction for a 2*5 X input\
         by a network having 10neurons on last layer should be 2*10"


    #---------------------------------------------------------------------------
    ## general tests of method predict with all activation types ###############
    #---------------------------------------------------------------------------

    my_first_nn=handmade_nn(5)

    my_first_nn.add_dense_layer(10, 'relu')
    my_first_nn.weights[-1] = np.concatenate([np.identity(5), np.zeros((5,5))], axis=0)
    my_first_nn.bias[-1] = np.expand_dims([0,0,0,0,1,1,1,0,0,0], axis=1)

    my_first_nn.add_dense_layer(10, 'linear')
    my_first_nn.weights[-1] = np.flip(np.identity(10), 1)
    my_first_nn.bias[-1] = np.expand_dims([1,1,1,1,1,1,0,0,0,0], axis=1)

    my_first_nn.add_dense_layer(10, 'tanh')
    my_first_nn.weights[-1] = np.identity(10)
    my_first_nn.bias[-1] = np.expand_dims([0,0,0,0,1,1,1,1,0,0], axis=1)

    my_first_nn.add_dense_layer(10, 'softmax')
    my_first_nn.weights[-1] = np.flip(np.identity(10), 1)
    my_first_nn.bias[-1] = np.expand_dims([0,0,0,0,0,0,1,1,1,1], axis=1)

    my_first_nn.add_dense_layer(1, 'sigmoid')
    my_first_nn.weights[-1] = np.expand_dims(np.arange(1,11,1), axis=0)
    my_first_nn.bias[-1] = np.expand_dims([0.5], axis=1)

    assert np.round(my_first_nn.predict([-2,-1,2,3,4])[0,0], decimals=8) == 0.99939824,\
        "the general test of predict method on a network involving\
         all activation types and manually set bias and weights\
         did not return the correct value"


    #---------------------------------------------------------------------------
    ### tests for function compute_metric ######################################
    #---------------------------------------------------------------------------

    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[[1,1],[1,2]],
                                 [[2,1],[2,2]]]),
                       np.array([[1,2],
                                 [3,4]]),
                       'mse')
    assert 'y vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array as y\
         in compute_metric function"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[1,2],
                                 [3,4]]),
                       np.array([[[1,1],[1,2]],
                                 [[2,1],[2,2]]]),
                       'mse')
    assert 'y_pred vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array as y_pred\
         in compute_metric function"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[1,2,3],
                                 [4,5,6]]),
                       np.array([[1,2],
                                 [3,4]]),
                       'mse')
    assert 'unconsistent vectors dimensions' in str(context.exception),\
        "no or wrong Exception raised when inputing unconsistent\
         y vs y_pred vectors shapes in compute_metric function"

    #---------------------------------------------------------------------------
    assert compute_metric([1,0],[0.5,1],'mse') == 0.625,\
        "uncorrect mse metric behaviour"

    #---------------------------------------------------------------------------
    assert compute_metric([[1,0],[0,0]],[[0.5,1],[1,1]],'mse') == 0.8125,\
        "uncorrect mse metric behaviour for multi-features regressions\
         (2D y and y_pred vectors)"

    #---------------------------------------------------------------------------
    assert compute_metric([1,0],[0.5,1],'mae') == 0.75,\
        "uncorrect mae metric behaviour"

    #---------------------------------------------------------------------------
    assert np.round(compute_metric([[1,0,1],[0,0,0]],[[0.5,0.9,0.1],[0.9,0.9,0.1]],
                                   'categorical_crossentropy'),
                    decimals=8) == 1.49786614,\
        "uncorrect categorical_crossentropy metric behaviour"

    #---------------------------------------------------------------------------
    assert np.round(compute_metric([1,0],[0.9,0.1],'binary_crossentropy'),
                    decimals=8) == 0.10536052,\
        "uncorrect binary_crossentropy metric behaviour"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric([[1,0,1],[0,0,0]],
                       [[0.5,0.9,0.1],
                        [0.9,0.9,0.1]],
                       'binary_crossentropy')
    assert '1 max for binary_crossentropy' in str(context.exception),\
        "no or wrong Exception raised when inputing 2D y/y_pred vectors\
         with binary_crossentropy selected in compute_metric function"

    #---------------------------------------------------------------------------
    test=TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric([0],[0],'typo_error')
    assert 'Unknown metric' in str(context.exception),\
        "no or wrong Exception raised when inputing\
         unknown metric in compute_metric function"

    #---------------------------------------------------------------------------

    print ('all tests successfully passed')



