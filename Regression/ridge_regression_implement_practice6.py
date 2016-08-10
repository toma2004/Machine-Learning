import graphlab
import matplotlib.pyplot as plt
import numpy as np

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

def predict_output(feature_matrix, weights):
	predictions = np.dot(feature_matrix, weights)
	return predictions

# with ridge regression, we take into account the L2 penalty
# derivative = 2 * sum[errors *[feature_i]] + 2*l2_penalty*weights[i]
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
	if feature_is_constant == True:
		derivative = 2*np.dot(errors, feature)
	else:
		derivative = 2*np.dot(errors, feature) + 2*l2_penalty*weight
	return derivative

def get_numpy_data(data_sframe, features, output):
	data_sframe['constant'] = 1 # add a constant to SFrame

	features = ['constant'] + features

	# take out feature list from data_sframe
	features_sframe = data_sframe[features]

	# convert SFrame into numpy matrix
	feature_matrix = features_sframe.to_numpy()

	output_sarray = data_sframe[output]
	
	# Convert output sarray into numpy
	output_array = output_sarray.to_numpy()
	return (feature_matrix, output_array)

# Test feature_derivative_ridge function
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([1., 10.])
print "my example features = \n", example_features
print "my example output = \n", example_output
print "my weights = \n", my_weights
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
	weights = np.array(initial_weights)

	while max_iterations > 0:
		predictions = predict_output(feature_matrix, weights)
		# compute error
		errors = predictions - output

		for i in xrange(len(weights)): # loop over each weight
			if i == 0:
				derivative = feature_derivative_ridge(errors, feature_matrix[: , i], weights[i], l2_penalty, True)
			else:
				derivative = feature_derivative_ridge(errors, feature_matrix[: , i], weights[i], l2_penalty, False)

			weights[i] = weights[i] - step_size*derivative
		max_iterations -= 1

	return weights

# Visualize L2 penalty
simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0, 1000)
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 1e11, 1000)

print "coefficient with no regularization = \n", simple_weights_0_penalty
print "coefficient with high regularization = \n", simple_weights_high_penalty

plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
plt.show()

# Calculate Residual Sum of Squares (RSS) to see our error margin
# input: numpy arrays of predicted value and actual value
# output: return the RSS
def get_residual_sum_of_squares(predicted, output):
	error = predicted - output
	return np.sum(error*error)

weight_zero = get_residual_sum_of_squares(predict_output(simple_test_feature_matrix, initial_weights), test_output)
weight_no_regularization = get_residual_sum_of_squares(predict_output(simple_test_feature_matrix, simple_weights_0_penalty), test_output)
weight_with_regularization = get_residual_sum_of_squares(predict_output(simple_test_feature_matrix, simple_weights_high_penalty), test_output)

print "RSS of initial weights = %d\n" % weight_zero
print "RSS of weights with no regularization = %d\n" % weight_no_regularization
print "RSS of weights with high regularization = %d\n" % weight_with_regularization

print "smallest RSS = %d\n" % min(weight_zero, weight_no_regularization, weight_with_regularization)

########## Examine L2 penalty with multiple regression (multiple feature) #############
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0, max_iterations)
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)

print "multiple regression - coefficient with no regularization = \n", multiple_weights_0_penalty
print "multiple regression - coefficient with high regularization = \n", multiple_weights_high_penalty

plt.plot(feature_matrix,output,'k.',
         feature_matrix,predict_output(feature_matrix, multiple_weights_0_penalty),'b-',
        feature_matrix,predict_output(feature_matrix, multiple_weights_high_penalty),'r-')
plt.show()

multiple_regression_weight_zero = get_residual_sum_of_squares(predict_output(test_feature_matrix, initial_weights), test_output)
multiple_regression_weight_no_regularization = get_residual_sum_of_squares(predict_output(test_feature_matrix, multiple_weights_0_penalty), test_output)
multiple_regression_weight_with_regularization = get_residual_sum_of_squares(predict_output(test_feature_matrix, multiple_weights_high_penalty), test_output)

print "RSS of initial weights = %d\n" % multiple_regression_weight_zero
print "RSS of weights with no regularization = %d\n" % multiple_regression_weight_no_regularization
print "RSS of weights with high regularization = %d\n" % multiple_regression_weight_with_regularization

print "smallest RSS = %d\n" % min(multiple_regression_weight_zero, multiple_regression_weight_no_regularization, multiple_regression_weight_with_regularization)

no_regularization_predict = predict_output(test_feature_matrix, multiple_weights_0_penalty)
high_regularization_predict = predict_output(test_feature_matrix, multiple_weights_high_penalty)

print "no regularization predict house price for first house is %.2f\n" % no_regularization_predict[0]
print "high regularization predict house price for first house is %.2f\n" % high_regularization_predict[0]
print "actual house price for first house is %.2f\n" % test_output[0]