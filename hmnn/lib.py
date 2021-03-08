'''This module contains all functions needed for
the fully handmade neural network, as well as the
model class itself'''

#import pandas as pd
import numpy as np
import seaborn as sns
#from progressbar.bar import ProgressBar

## Functions

def compute_activation (X, activation_type):
    '''Defining activation functions
     Takes a nparray or a single value
    # Returns in the same format

    For softmax : assuming that X.shape[0]== n_neurons,
        the axis0 of array X is used for computing the mean
    '''
    X=np.array(X)
    if activation_type == 'relu':
        return np.maximum(X,0)
    if activation_type == 'sigmoid':
        return 1/(1+np.exp(-X))
    if activation_type == 'tanh':
        return np.tanh(X)
    if activation_type == 'linear':
        return X
    if activation_type == 'softmax':
        exp_x = np.exp(X)
        return exp_x / exp_x.sum(axis=0)

    #raise error if unknown type
    raise ValueError(f'Unknown activation type {activation_type}.\
Supported types : linear, relu, sigmoid, tanh, softmax')


def compute_activation_derivative (layer_output, activation_type):
    '''Computes the derivative of the activation functions,
       depending of the outputs of the output of these functions
           nota : if occures that for each of the 5 basic activations,
           f'(X) can be expressed simply as a function of f(X)

           Takes a nparray or a single value
        # Returns in the same format
           '''
    X_output=np.array(layer_output)
    if activation_type == 'relu':
        return (X_output > 0).astype(int)
    if activation_type == 'linear':
        return np.ones(X_output.shape)
    if activation_type == 'sigmoid':
        return X_output - np.square(X_output)
    if activation_type == 'tanh':
        return 1 - np.square(X_output)
    if activation_type == 'softmax':
        return X_output - np.square(X_output)

    #raise error if unknown type
    raise ValueError(f'Unknown activation type {activation_type}.\
Supported types : linear, relu, sigmoid, tanh, softmax')


def compute_metric (y, y_pred, metric, loss_derivative=False):
    '''Defining loss and metric functions
     Takes nparrays, lists or a single values

     ## IF loss_derivative==False:
         output: always scalar

     ## IF loss_derivative==True: (True will be ignored for non-loss metrics)
         Computes the partial derivative of the loss function
           with respect to each component of each sample
         output: 2Darray
            n_samples * 1 for binary_crossentropy or single output regression
            n_samples * n_class for categorical_crossentropy
            n_samples * n_features for multifeatures regression)
    '''

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

    #compute loss funtions (or derivatives if loss_derivative==True)
    if metric == 'mse':
        if not loss_derivative:
            return np.square(y-y_pred).mean()
        return 1 / y.size * 2 * (y_pred - y)

    if metric == 'mae':
        if not loss_derivative:
            return np.abs(y-y_pred).mean()
        return 1 / y.size * (y_pred - y) / np.abs(y - y_pred)

    if metric == 'categorical_crossentropy':
        if not loss_derivative:
            return -1 / y.shape[0] * ((y * np.log(y_pred)).sum())
        return -1 / y.shape[0] * (y / y_pred)

    if metric == 'binary_crossentropy':
        if y.shape[1]>1:
            raise ValueError('y vector dimension too high.\
Must be 1 max for binary_crossentropy')
        if not loss_derivative:
            return -(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)).mean()
        return -1 / y.size * (y / y_pred - (1-y) / (1-y_pred))

    # compute other metrics functions
    #### accuracy, f1-score, recall, etc.. : not implemented yet

    raise ValueError(f'Unknown metric {metric}. Supported types :\
mse, mae, categorical_crossentropy, binary_crossentropy')

class adam_optimizer():
    '''adam optimizer object
    This object in instanciated by the .fit() method of the model class
        each time it is triggered
    Unlike in Keras, this object should not be instanciated by the user'''

    def __init__(self, weights, bias, alpha_init=0.001, beta_1=0.9,
             beta_2=0.999, epsilon=1e-8):

        self.alpha_init=alpha_init
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon

        self.t=0
        #initializing first and second momentum
        self.m_weights = [np.zeros_like(w) for w in weights]
        self.m_bias = [np.zeros_like(b) for b in bias]
        self.v_weights = self.m_weights.copy()
        self.v_bias = self.m_bias.copy()

    def get_update(self, gradient_weights, gradient_bias):
        '''computes the values to be added to weights and bias arrays at
        the end of the train step'''
        self.t+=1
        alpha=self.alpha_init*np.sqrt(1-self.beta_2**self.t)/(1-self.beta_1**self.t)

        # updating 1st and 2nd momenta
        self.m_weights=[self.beta_1 * m + (1-self.beta_1) * grad\
                   for m, grad in zip(self.m_weights, gradient_weights)]
        self.m_bias=[self.beta_1 * m + (1-self.beta_1) * grad\
                   for m, grad in zip(self.m_bias, gradient_bias)]
        self.v_weights=[self.beta_2 * v + (1-self.beta_2) * grad**2\
                   for v, grad in zip(self.v_weights, gradient_weights)]
        self.v_bias=[self.beta_2 * v + (1-self.beta_2) * grad**2\
                   for v, grad in zip(self.v_bias, gradient_bias)]

        #computing the updates
        weights_update = [- alpha * m / (np.sqrt(v) + self.epsilon)\
                                  for m, v in zip( self.m_weights, self.v_weights)]
        bias_update = [- alpha * m / (np.sqrt(v) + self.epsilon)\
                                  for m, v in zip( self.m_bias, self.v_bias)]

        return weights_update, bias_update


class handmade_nn ():
    '''
    hand-made version of neural network
    so far, the possibilities are :

        - layers activation functions :
            'linear', 'relu', 'sigmoid', 'tanh', 'softmax'

        - weights initializers : 'ones', 'glorot_uniform'
        - bias initializers : 'zeros', 'ones'

        - loss functions :
            'mse', 'mae', 'binary_crossentropy', 'categorical_crossentropy'

        - solver :
            SGD without momentum
    '''
    def __init__ (self, input_dim=0):
        self.weights=[]
        self.bias=[]
        self.activation_types=[]
        self.input_dim=input_dim
        self.n_layers=0

        self.loss_history=[]

    def set_input_dim (self, input_dim):
        '''manually sets the input_dim attribute of the model instance'''
        self.input_dim = input_dim

    def set_loss (self, loss):
        '''manually sets the loss attribute of the model instance'''
        self.loss = loss

    def add_dense_layer (self, n_neurons, activation_type,
                         weights_initializer='glorot_uniform', bias_initializer='zeros'):
        '''add a dense (fully connected) layer of neurons to the model
        This initializes the weights and bias according to selected initializer type,
        wich are yet implemented directly here'''
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
        if weights_initializer == 'ones':
            self.weights.append(np.ones((n_neurons, previous_dim)))
        elif weights_initializer == 'glorot_uniform':
            limit = np.square(6 / (n_neurons + previous_dim))
            self.weights.append(np.random.uniform(-limit, limit, size = (n_neurons, previous_dim)))
        else:
            raise ValueError(f'Unknown weights initializer {weights_initializer}.\
Supported types : ones, glorot_uniform')

        if bias_initializer == 'zeros':
            self.bias.append(np.zeros(n_neurons))
        elif bias_initializer == 'ones':
            self.bias.append(np.ones(n_neurons))
        else:
            raise ValueError(f'Unknown bias initializer {bias_initializer}.\
Supported types : zeros, ones')

        self.activation_types.append(activation_type)
        self.n_layers += 1

        #test the activation type
        compute_activation(0, activation_type)

    def predict (self, X, keep_hidden_layers=False):
        '''input X : list, list of lists, np array, pd DataFrame
               axis 0 = samples
               axis 1 = features

           ## IF keep_hidden_layers==False:
           output = y_pred: 2D np-array
               axis 0 = samples
               axis 1 = output features, depending of the size of last layer

           ## IF keep_hidden_layers==True:
           outputs = layers_outputs, layers_activation_derivatives
           -output1 = layers_outputs:
               list of 2D np-arrays of outputs of each layer
               len(list)=n_layers+1: 1st element = X itself
                                     last element = y_pred
               axis 0 = samples
               axis 1 = number of neurons of the layer
           -output2 = layers_activation_derivatives:
               list of 2D np-arrays of d_act/d_input of each layer
               len(list)=n_layers
               axis 0 = samples
               axis 1 = number of neurons of the layer
           '''
        #converting DataFrames, lists or lists of lists to nparray
        X = np.array(X)

        #deal with 1D inputs to forge a 1 * n_features 2D-array
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis = 0)

        #raise errors for unconsistant inputs
        if len(X.shape) > 2:
            raise ValueError('X vector dimension too high. Must be 2 max')
        if X.shape[1] != self.input_dim:
            raise ValueError(f'Unconsistent number of features. \
The network input_dim is {self.input_dim}')

        #compute the prediction
        layers_outputs = [X]
        layers_activation_derivatives = []
        for layer_index, activation_type in enumerate(self.activation_types):
            activation_input = np.dot(self.weights[layer_index], X.T)\
                               + np.expand_dims(self.bias[layer_index], axis=1)
            X = compute_activation(activation_input, activation_type).T

            layers_outputs.append(X)
            layers_activation_derivatives.append(\
                compute_activation_derivative(X, activation_type))

        if keep_hidden_layers:
            return layers_outputs, layers_activation_derivatives
        return X

    def score (self, X, y, metric):
        '''use predict method, then compute_metric function'''
        y_pred=self.predict(X)
        return compute_metric(y, y_pred, metric)

    def compute_backpropagation (self, X, y):
        '''This method :
            - executes self.predict(X) WITH keep_hidden_layers
                to keep all intermediate outputs
            - executes compute_metric (y, y_pred, loss) WITH loss_derivative
            - for each layer from last to first : computes loss
              derivatives (aka gradient) with respect to bias and weights

            output 1 : gradient with respect to weights
               (list of 2D arrays
               len(list) = n_layers
               axis 0 = number of neurons of the layer
               axis 1 = number of neurons of the previous layer (or features in the input)
            output 2 : gradient with respect to bias
               (list of 1D arrays)
               len(list) = n_layers
               axis 0 = number of neurons of the layer
            '''
        delta_weights=[]
        delta_bias=[]

        # compute the outputs and the derivatives of each layer
        layers_outputs, layers_activation_derivatives\
                = self.predict(X, keep_hidden_layers = True)
        # compute d_loss/d_ypred
        dloss_doutput = compute_metric (y,
                                        layers_outputs[-1],
                                        self.loss,
                                        loss_derivative = True)
        for layer_index in range(self.n_layers-1, -1, -1):
            # compute d_loss/d_input of the layer
            dloss_dinput = dloss_doutput * layers_activation_derivatives[layer_index]

            # compute gradient with respect to weights and bias
            delta_weights.append(np.dot(dloss_dinput.T, layers_outputs[layer_index]))
            delta_bias.append(np.sum(dloss_dinput, axis=0))

            # update dloss_doutput for next propagation
            if layer_index > 0:
                dloss_doutput = np.dot (dloss_dinput, self.weights[layer_index])

        delta_weights.reverse()
        delta_bias.reverse()

        return delta_weights, delta_bias



    def fit (self, X, y, loss=None, learning_rate=0.01,
             batch_size=1, n_epochs=10, verbose=1,
             optimizer_type='sgd',
             alpha_init=0.001, beta_1=0.9,
             beta_2=0.999, epsilon=1e-8):
        '''input X : 2D array or pd DataFrame
                axis 0 = samples
                axis 1 = features
        '''
        if loss:
            self.loss=loss

        if optimizer_type == 'adam':
            optimizer = adam_optimizer (self.weights, self.bias,
                                        alpha_init=alpha_init, beta_1=beta_1,
                                        beta_2=beta_2, epsilon=epsilon)

        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        n_minibatches_per_epoch = int(n_samples / batch_size)

        loss=self.score(X, y, self.loss)
        if self.loss_history == []:
            self.loss_history.append(loss)

        if verbose>0:
            print(f'initial loss: {self.score(X, y, self.loss)}')

        for epoch_index in range (n_epochs):
            if verbose>1:
                print(f'beginning epoch n°{epoch_index + 1}')

            #progress_batches = ProgressBar()
            #for mini_batch_index in progress_batches(range(n_minibatches_per_epoch)):
            for mini_batch_index in range(n_minibatches_per_epoch):
                gradient_weights, gradient_bias\
                    = self.compute_backpropagation(X[mini_batch_index * batch_size :\
                                                     (mini_batch_index +1) * batch_size],
                                                   y[mini_batch_index * batch_size :\
                                                     (mini_batch_index +1) * batch_size])
                if optimizer_type == 'sgd':
                    #compute the update directly
                    weights_update = [-learning_rate * grad for grad in gradient_weights]
                    bias_update = [-learning_rate * grad for grad in gradient_bias]

                elif optimizer_type == 'adam':
                    #compute the update with the optimizer
                    weights_update, bias_update = optimizer.get_update(gradient_weights,
                                                                       gradient_bias)

                else:
                    raise ValueError(f'unsupported optimizer type {optimizer_type}')

                # updating weights and bias
                self.weights = [w + w_update  for w, w_update in zip(self.weights, weights_update)]
                self.bias = [b + b_update for b, b_update in zip(self.bias, bias_update)]

            loss=self.score(X, y, self.loss)
            self.loss_history.append(loss)

            if verbose>1:
                print(f'end of epoch n°{epoch_index + 1}. loss: {self.score(X, y, self.loss)}')
        if verbose==1:
            print(f'final loss: {self.score(X, y, self.loss)}')

    def plot_loss_history(self):
        '''plots the complete loss history of the model since creation,
        including multiple .fit() calls'''
        graph=sns.lineplot(x=range(len(self.loss_history)),y=self.loss_history)
        graph.set(xlabel="epochs", ylabel = "loss")
