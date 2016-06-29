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