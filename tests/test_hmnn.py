import unittest

import sys
sys.path.append('../')

from hmnn.lib import *


#---------------------------------------------------------------------------
## tests for function compute_activation
#---------------------------------------------------------------------------
class Test_compute_activation(unittest.TestCase):

  def test_activation_relu(self):
    expected = np.array([[0,0], [0, 1], [1, 3]])
    actual = compute_activation(np.array([[-1,0], [0, 1], [1, 3]]), 'relu')
    self.assertTrue((actual == expected).all(),\
            msg="uncorrect relu function behaviour")

#---------------------------------------------------------------------------
  def test_activation_linear(self):
    expected = np.array([[-1,0], [0, 1], [1, 3]])
    actual = compute_activation(np.array([[-1,0], [0, 1], [1, 3]]), 'linear')

    self.assertTrue((actual == expected).all(),\
            msg="uncorrect linear function behaviour")

#---------------------------------------------------------------------------
  def test_activation_sigmoid(self):
    expected = np.array([[0.26894142, 0.5       ],
                         [0.5       , 0.73105858],
                         [0.73105858, 0.95257413]])
    actual = compute_activation(np.array([[-1,0], [0, 1], [1, 3]]),
                                        'sigmoid')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect sigmoid function behaviour")

#---------------------------------------------------------------------------
  def test_activation_tanh(self):
    expected = np.array([[-0.76159416,  0.        ],
                         [ 0.        ,  0.76159416],
                         [ 0.76159416,  0.99505475]])
    actual = compute_activation(np.array([[-1,0], [0, 1], [1, 3]]),
                                        'tanh')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect tanh function behaviour")

#---------------------------------------------------------------------------
  def test_activation_softmax(self):
    expected = np.array([[0.09003057, 0.04201007],
                         [0.24472847, 0.1141952 ],
                         [0.66524096, 0.84379473]])
    actual = compute_activation(np.array([[-1,0], [0, 1], [1, 3]]),
                                        'softmax')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect softmax function behaviour")

#---------------------------------------------------------------------------
  def test_unknown_activation_type_error(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_activation(0,'typo_error')
    self.assertTrue ('Unknown activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing an unknown activation_type\
         while calling compute_activation")


#---------------------------------------------------------------------------
## tests for function compute_activation_derivative
#---------------------------------------------------------------------------
class Test_compute_activation_derivative(unittest.TestCase):
    
  def test_activation_derivative_relu(self):
    expected = np.array([[0,0], [0, 1], [1, 1]])
    actual = compute_activation_derivative(np.array([[-1,0], [0, 1], [1, 3]]),
                                          'relu')
    self.assertTrue((actual == expected).all(),\
                    msg="uncorrect relu function derivative behaviour")

#---------------------------------------------------------------------------
  def test_activation_derivative_linear(self):
    expected = np.array([[1,1], [1, 1], [1, 1]])
    actual = compute_activation_derivative(np.array([[-1,0], [0, 1], [1, 3]]),
                                          'linear')
    self.assertTrue((actual == expected).all(),\
                    msg="uncorrect linear function derivative behaviour")

#---------------------------------------------------------------------------
  def test_activation_derivative_sigmoid(self):
    expected = np.array([[0.19661193, 0.25      ],
                         [0.25      , 0.19661193],
                         [0.19661193, 0.04517666]])
    actual = compute_activation_derivative\
                     (np.array([[0.26894142, 0.5       ],
                                [0.5       , 0.73105858],
                                [0.73105858, 0.95257413]]),
                      'sigmoid')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect sigmoid function derivative behaviour")

#---------------------------------------------------------------------------
  def test_activation_derivative_tanh(self):
    expected = np.array([[0.41997434, 1.        ],
                         [1.        , 0.41997434],
                         [0.41997434, 0.00986604]])
    actual = compute_activation_derivative\
                     (np.array([[-0.76159416,  0.        ],
                                [ 0.        ,  0.76159416],
                                [ 0.76159416,  0.99505475]]),
                      'tanh')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect tanh function derivative behaviour")

#---------------------------------------------------------------------------
  def test_activation_derivative_softmax(self):
    expected = np.array([[0.08192507, 0.04024522],
                         [0.18483645, 0.10115466],
                         [0.22269543, 0.13180518]])
    actual = compute_activation_derivative\
                     (np.array([[0.09003057, 0.04201007],
                                [0.24472847, 0.1141952 ],
                                [0.66524096, 0.84379473]]),
                      'softmax')
    self.assertTrue((np.round(actual,decimals=8) == expected).all(),\
            msg="uncorrect softmax function derivative behaviour")

#---------------------------------------------------------------------------
  def test_unknown_activation_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_activation_derivative(0,'typo_error')
    self.assertTrue ('Unknown activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing an unknown activation_type\
         while calling compute_activation_derivative")


#---------------------------------------------------------------------------
## tests for method add_dense_layer
#---------------------------------------------------------------------------
class Test_add_dense_layer(unittest.TestCase):

  def test_no_input_dim_exception(self):

    my_nn=handmade_nn()
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        my_nn.add_dense_layer(5,'relu')
    self.assertTrue ('Use set_input_dim before creating first layer'\
           in str(context.exception),\
        "no or wrong Exception raised when adding first layer\
         to a network without setting input_dim")

#---------------------------------------------------------------------------
  def test_unknown_activation_exception(self):
    my_nn=handmade_nn(5)
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        my_nn.add_dense_layer(10,'typo_error')
    self.assertTrue ('Unknown activation type' in str(context.exception),\
        "no or wrong Exception raised when inputing\
         an unknown activation_type while adding layer")


#---------------------------------------------------------------------------
## tests for method predict - normal mode (without intermediate states) ####
#---------------------------------------------------------------------------
class Test_predict_without_intermediate(unittest.TestCase):

  def test_no_input_dim_exception(self):
    my_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs

    self.assertTrue(my_nn.predict([2,3,2,3,4]).shape == (1,5),\
        "list not supported as an input for predict")
#---------------------------------------------------------------------------
  def test_list_as_input(self):
    my_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    
    self.assertTrue(my_nn.predict([[2,3,2,3,4],[-2,-1,1,3,4]]).shape == (2,5),\
        "list of list not supported as an input for predict")

#---------------------------------------------------------------------------
  def test_x_dimension_too_high_exception(self):
    my_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        my_nn.predict(np.array([[[1,1],[1,2],[1,3],[1,4],[1,5]],
                                    [[2,1],[2,2],[2,3],[3,4],[3,5]]]))
    self.assertTrue ('X vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array in predict method")

#---------------------------------------------------------------------------
  def test_unconsistent_n_features_exception(self):
    my_nn=handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        my_nn.predict(np.array([[1,1],[1,2],[1,3],[1,4],[1,5]]))
    self.assertTrue ('Unconsistent number of features' in str(context.exception),\
        "no or wrong Exception raised when inputing a X\
         with unconsistant size vs. network input_dim in predict method")

#---------------------------------------------------------------------------
  def test_oneD_array_as_input(self):
    my_nn=handmade_nn(5)
    my_nn.add_dense_layer(10, 'linear')
    my_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    self.assertTrue(my_nn.predict(np.array([-2,-1,2,3,4])).shape == (1,10),\
        "1-D array not supported as an input for predict method")

#---------------------------------------------------------------------------
  def test_output_dimension(self):
    my_nn=handmade_nn(5)
    my_nn.add_dense_layer(10, 'linear')
    my_nn.weights[0] = np.vstack((np.identity(5),np.zeros((5,5))))
    self.assertTrue (my_nn.predict(np.array([[-2,-1,2,3,4],
                                         [-12,-11,12,13,14]])).shape == (2,10),\
        "the shape of the prediction for a 2*5 X input\
         by a network having 10neurons on last layer should be 2*10")

#--General-test-of-predict-method-with-all-activation-types-----------------
  def test_general_predict_test_all_activations(self):

    my_nn=handmade_nn(5)

    my_nn.add_dense_layer(10, 'relu')
    my_nn.weights[-1] = np.concatenate([np.identity(5), np.zeros((5,5))], axis=0)
    my_nn.bias[-1] = np.array([0,0,0,0,1,1,1,0,0,0])

    my_nn.add_dense_layer(10, 'linear')
    my_nn.weights[-1] = np.flip(np.identity(10), 1)
    my_nn.bias[-1] = np.array([1,1,1,1,1,1,0,0,0,0])

    my_nn.add_dense_layer(10, 'tanh')
    my_nn.weights[-1] = np.identity(10)
    my_nn.bias[-1] = np.array([0,0,0,0,1,1,1,1,0,0])

    my_nn.add_dense_layer(10, 'softmax')
    my_nn.weights[-1] = np.flip(np.identity(10), 1)
    my_nn.bias[-1] = np.array([0,0,0,0,0,0,1,1,1,1])

    my_nn.add_dense_layer(1, 'sigmoid')
    my_nn.weights[-1] = np.expand_dims(np.arange(1,11,1), axis=0)
    my_nn.bias[-1] = np.array([0.5])


    self.assertTrue (np.round(my_nn.predict([-2,-1,2,3,4])[0,0], decimals=8) == 0.99939824,\
        "the general test of predict method on a network involving\
         all activation types and manually set bias and weights\
         did not return the correct value")


#---------------------------------------------------------------------------
## tests for method predict - backprop mode (with intermediate states) #####
#---------------------------------------------------------------------------
class Test_predict_backprop_mode(unittest.TestCase):

  def test_output_type_is_tuple(self):

    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    self.assertTrue (output_type == tuple,\
        f"predict method with keep_hidden_layers must return a tuple. type is {output_type}")

#---------------------------------------------------------------------------
  def test_number_of_outputs(self):
    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    self.assertTrue (n_outputs == 2,\
        f"predict method with keep_hidden_layers must return 2 output. here returns {n_outputs}")

#---------------------------------------------------------------------------
  def test_length_layers_outputs(self):
    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    layers_outputs, layers_derivatives = outputs
    
    self.assertTrue (len(layers_outputs) == 2,\
        "the list of outputs of layers has not correct length using predict with keep_hidden_layers")

#---------------------------------------------------------------------------
  def test_length_layers_derivatives(self):
    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    layers_outputs, layers_derivatives = outputs
    
    self.assertTrue (len(layers_derivatives) == 1,\
        "the list of derivatives of layers has not correct length using predict with keep_hidden_layers")

#---------------------------------------------------------------------------
  def test_layers_outputs_values(self):
    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    layers_outputs, layers_derivatives = outputs
    
    self.assertTrue ((layers_outputs[1] == np.array([[14., 14., 14.]])).all(),\
        "uncorrect layers_outputs of predict method with keep_hidden_layers")

#---------------------------------------------------------------------------
  def test_layers_derivatives_values(self):
    my_nn = handmade_nn(5)# Empty neural network : just a pass-through for 5-values inputs
    my_nn.add_dense_layer(3, 'linear', weights_initializer='ones')

    outputs = my_nn.predict([2,3,2,3,4], keep_hidden_layers=True)
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    layers_outputs, layers_derivatives = outputs
    
    self.assertTrue ((layers_derivatives[0] == np.array([[1., 1., 1.]])).all(),\
        "uncorrect layers_derivatives of predict method with keep_hidden_layers")


#---------------------------------------------------------------------------
### tests for function compute_metric - normal mode (not derivative)########
#---------------------------------------------------------------------------
class Test_compute_metric(unittest.TestCase):
    
  def test_y_dimension_too_high_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[[1,1],[1,2]],
                                 [[2,1],[2,2]]]),
                       np.array([[1,2],
                                 [3,4]]),
                       'mse')
    self.assertTrue ('y vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array as y\
         in compute_metric function")

#---------------------------------------------------------------------------
  def test_y_pred_dimension_too_high_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[1,2],
                                 [3,4]]),
                       np.array([[[1,1],[1,2]],
                                 [[2,1],[2,2]]]),
                       'mse')
    self.assertTrue ('y_pred vector dimension too high' in str(context.exception),\
        "no or wrong Exception raised when inputing a 3D-array as y_pred\
         in compute_metric function")

#---------------------------------------------------------------------------
  def test_unconsistent_dimensions_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric(np.array([[1,2,3],
                                 [4,5,6]]),
                       np.array([[1,2],
                                 [3,4]]),
                       'mse')
    self.assertTrue ('unconsistent vectors dimensions' in str(context.exception),\
        "no or wrong Exception raised when inputing unconsistent\
         y vs y_pred vectors shapes in compute_metric function")

#---------------------------------------------------------------------------
  def test_mse(self):
    self.assertTrue (compute_metric([1,0],[0.5,1],'mse') == 0.625,\
        "uncorrect mse metric behaviour")

#---------------------------------------------------------------------------
  def test_mse_multi_features(self):
    self.assertTrue (compute_metric([[1,0],[0,0]],[[0.5,1],[1,1]],'mse') == 0.8125,\
        "uncorrect mse metric behaviour for multi-features regressions\
         (2D y and y_pred vectors)")

#---------------------------------------------------------------------------
  def test_mae(self):
    self.assertTrue (compute_metric([1,0],[0.5,1],'mae') == 0.75,\
        "uncorrect mae metric behaviour")

#---------------------------------------------------------------------------
  def test_categorical_crossentropy(self):
    self.assertTrue (np.round(compute_metric([[1,0,0],[0,1,0]],[[0.8,0.1,0.1],[0.2,0.6,0.2]],
                                   'categorical_crossentropy'),
                    decimals=8) == 0.36698459,\
        "uncorrect categorical_crossentropy metric behaviour")

#---------------------------------------------------------------------------
  def test_binary_crossentropy(self):
    self.assertTrue (np.round(compute_metric([1,0],[0.9,0.1],'binary_crossentropy'),
                    decimals=8) == 0.10536052,\
        "uncorrect binary_crossentropy metric behaviour")

#---------------------------------------------------------------------------
  def test_y_dimension_too_high_with_binary_crossentropy_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric([[1,0,1],[0,0,0]],
                       [[0.5,0.9,0.1],
                        [0.9,0.9,0.1]],
                       'binary_crossentropy')
    self.assertTrue ('1 max for binary_crossentropy' in str(context.exception),\
        "no or wrong Exception raised when inputing 2D y/y_pred vectors\
         with binary_crossentropy selected in compute_metric function")

#---------------------------------------------------------------------------
  def test_unknown_metric_exception(self):
    test=unittest.TestCase()
    with test.assertRaises(ValueError) as context:
        compute_metric([0],[0],'typo_error')
    self.assertTrue ('Unknown metric' in str(context.exception),\
        "no or wrong Exception raised when inputing\
         unknown metric in compute_metric function")


#---------------------------------------------------------------------------
### tests for function compute_metric - derivative mode ####################
#---------------------------------------------------------------------------
class Test_metric_derivative(unittest.TestCase):
    
  def test_output_format(self): 
    self.assertTrue (len(compute_metric([1,0],[0.5,1],'mse', loss_derivative = True)\
               .shape) == 2,\
        "uncorrect output : compute_metric must return a 2D array in derivative mode")

#---------------------------------------------------------------------------
  def test_mse(self):
    self.assertTrue ((compute_metric([1,0],[0.5,1],'mse', loss_derivative = True)\
                == np.array([[-0.5],[1]])).all(),\
        "uncorrect mse metric behaviour in derivative mode")

#---------------------------------------------------------------------------
  def test_mse_multi_features(self):
    self.assertTrue ((compute_metric([[1,0],[0,0]],[[0.5,1],[1,1]],'mse',
                       loss_derivative = True)\
                == np.array([[-0.25, 0.5],[0.5, 0.5]])).all(),\
        "uncorrect mse metric behaviour for multi-features regressions\
         (2D y and y_pred vectors) in derivative mode")

#---------------------------------------------------------------------------
  def test_mae(self):
    self.assertTrue ((compute_metric([1,0],[0.5,1],'mae', loss_derivative = True)\
                == np.array([[-0.5],[0.5]])).all(),\
        "uncorrect mae metric behaviour in derivative mode")

#---------------------------------------------------------------------------
  def test_categorical_crossentropy(self):
    self.assertTrue ((np.round(compute_metric([[1,0,0],[0,1,0]],[[0.8,0.1,0.1],[0.2,0.6,0.2]],
                                   'categorical_crossentropy',
                                   loss_derivative = True),
                    decimals=8) == np.array([[-0.625, -0.        , -0.],
                                             [-0.   , -0.83333333, -0.]])).all(),\
        "uncorrect categorical_crossentropy metric behaviour in derivative mode")

#---------------------------------------------------------------------------
  def test_binary_crossentropy(self):
    self.assertTrue ((np.round(compute_metric([1,0],[0.9,0.1],
                                   'binary_crossentropy',
                                   loss_derivative = True),
                    decimals=8) == np.array([[-0.55555556],
                                             [ 0.55555556]])).all(),\
        "uncorrect binary_crossentropy metric behaviour in derivative mode")


#---------------------------------------------------------------------------
### tests for backpropagation method (computing the gradient) ##############
#---------------------------------------------------------------------------
class Test_backpropagation(unittest.TestCase):
    
  def test_output_type_is_tuple(self):

    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    self.assertTrue (output_type == tuple,\
        f"backpropagation method must return a tuple. type is {output_type}")

#---------------------------------------------------------------------------
  def test_number_of_outputs(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    self.assertTrue (n_outputs == 2,\
        f"backpropagation method must return 2 output. here returns {n_outputs}")

#---------------------------------------------------------------------------
  def test_weights_gradient_length(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    gradient_weights, gradient_bias = outputs

    self.assertTrue (len(gradient_weights) == 2,\
        "using backpropagation: the list of weights has not correct length")

#---------------------------------------------------------------------------
  def test_bias_gradient_length(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    gradient_weights, gradient_bias = outputs
    
    self.assertTrue (len(gradient_bias) == 2,\
        "using backpropagation: the list of bias has not correct length")

#---------------------------------------------------------------------------
  def test_weights_gradient_values(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    gradient_weights, gradient_bias = outputs
    
    self.assertTrue ((gradient_weights[0] == np.array([[24., 34.],
                                             [24., 34.]])).all(),\
        "using backpropagation: uncorrect gradient with respect to weights")

#---------------------------------------------------------------------------
  def test_bias_gradient_values(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(2, 'linear', weights_initializer='ones')
    my_nn.add_dense_layer(1, 'linear', weights_initializer='ones')
    my_nn.set_loss('mse')

    outputs = my_nn.compute_backpropagation(np.array([[1,2],[2,3],[3,4]]), np.array([4,5,6]))
    n_outputs = len(np.array(outputs))
    output_type = type(outputs)
    
    gradient_weights, gradient_bias = outputs
    
    self.assertTrue ((gradient_bias[0] == np.array([10., 10.])).all(),\
        "using backpropagation: uncorrect gradient with respect to bias")


#---------------------------------------------------------------------------
### tests for fit method
#---------------------------------------------------------------------------
class Test_fit_method(unittest.TestCase):
  
  def test_convergence_on_trivial_regression(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(1, 'linear',)
    X= np.ones((10_000, 2))
    y= np.zeros((10_000,1))
    my_nn.fit(X,y, loss='mse',optimizer_type='sgd', batch_size=7, n_epochs=2)

    self.assertTrue (my_nn.score(X,y,'mse') < 0.5,\
        "fit method has not converged with build-in sgd optimizer on a trivial regression")

#---------------------------------------------------------------------------
### tests for adam optimizer
#---------------------------------------------------------------------------
class Test_fit_method_with_adam(unittest.TestCase):
  
  def test_convergence_on_trivial_regression(self):
    my_nn=handmade_nn(input_dim = 2)
    my_nn.add_dense_layer(1, 'linear',)
    X= np.ones((10_000, 2))
    y= np.zeros((10_000,1))
    my_nn.fit(X,y, loss='mse',optimizer_type='adam', batch_size=7, n_epochs=2)

    self.assertTrue (my_nn.score(X,y,'mse') < 0.5,\
        "not converged with adam optimizer on a trivial regression")

#---------------------------------------------------------------------------


print ('all tests successfully passed')
