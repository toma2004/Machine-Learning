import graphlab
import numpy as np
from math import sqrt

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

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


# Test above function
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') # the [] around 'sqft_living' makes it a list
print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output

# predict output given regression weights
my_weights = np.array([1., 1.]) # the example weights
my_features = example_features[0,] # we'll use the first data point
predicted_value = np.dot(my_features, my_weights)
print predicted_value

# Calculate Residual Sum of Squares (RSS) to see our error margin
# input: numpy arrays of predicted value and actual value
# output: return the RSS
def get_residual_sum_of_squares(predicted, output):
	error = predicted - output
	return np.sum(error*error)

def predict_output(feature_matrix, weights):
	predictions = np.dot(feature_matrix, weights)
	return predictions

# test above function
test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0

def feature_derivative(errors, feature):
	derivative = 2*np.dot(errors, feature)
	return derivative

# test above function
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights) 
# just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print derivative
print -np.sum(example_output)*2 # should be the same as derivative

# implement gradient descent algorithm
# w(i+1) := w(i) - alpha * derivative(cost func) w.r.t feature[i] where cost func = RSS(predict - ouput)^2
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
	converged = False
	weights = np.array(initial_weights)

	while not converged:
		# compute predictions based on weights
		predictions = predict_output(feature_matrix, weights)

		# compute error
		error = predictions - output

		gradient_sum_squares = 0
		for i in range(len(weights)):
			# compute derivative of weight[i]
			derivative = feature_derivative(error, feature_matrix[: , i])

			gradient_sum_squares += np.sum(derivative*derivative)

			# update weight[i] for next round
			weights[i] = weights[i] - step_size*derivative

		# compute the gradient magnitude
		gradient_magnitude = sqrt(gradient_sum_squares)

		# check to see if we have converged
		if gradient_magnitude < tolerance:
			converged = True

	return weights

# Test our above function
# First is to set up train data and test data
train_data, test_data = sales.random_split(.8, seed=0)

simple_features = ['sqft_living']
my_output = 'price'
(simple_features_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

calculated_weights = regression_gradient_descent(simple_features_matrix, output, initial_weights, step_size, tolerance)
print calculated_weights

# now we have calculated weights, use test data to predict the price
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

test_predictions = predict_output(test_simple_feature_matrix, calculated_weights)
print test_predictions
print test_output

# calculate RSS for the predicted value
test_rss = get_residual_sum_of_squares(test_predictions, test_output)
print test_rss

# Running multiple regression
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

multiple_regression_model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print multiple_regression_model_weights

# prepare Test data
(test_multiple_feature_matrix, test_multiple_features_output) = get_numpy_data(test_data, model_features, my_output)

test_predictions_multiple_regression_model = predict_output(test_multiple_feature_matrix, multiple_regression_model_weights)
print test_predictions_multiple_regression_model
print test_output

# compute rs for multiple regression model
test_rss_multiple_regression_model = get_residual_sum_of_squares(test_predictions_multiple_regression_model, test_multiple_features_output)
print test_rss_multiple_regression_model