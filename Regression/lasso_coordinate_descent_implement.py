import graphlab
import matplotlib.pyplot as plt
import numpy as np
import math

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

# In dataset, floors was defined with type string
sales['floors'] = sales['floors'].astype(int)

def predict_output(feature_matrix, weights):
	predictions = np.dot(feature_matrix, weights)
	return predictions

# example of normalizing using numpy
X = np.array([[3.,5.,8.],[4.,12.,15.]])
print X

norms = np.linalg.norm(X, axis=0) # gives [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
print norms

print X / norms

# Function to normalize feature matrix
def normalize_features(feature_matrix):
	norms = np.linalg.norm(feature_matrix, axis=0)
	return (feature_matrix/norms, norms)

# test above function
features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
print norms

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

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
# Normalize feature matrix
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

# assign some random set of initial weights
weights = np.array([1., 4., 1.])
predictions = predict_output(simple_feature_matrix, weights)

print weights.size

# compute ro[i] for each i
# formula for coordinate descent (derivative): SUM [ [feature_i]*(output-predictions + w[i]*[feature_i]) ]
ro = []
for i in range(0, weights.size):
	ro.append( np.sum(simple_feature_matrix[:,i] * (output - predictions + weights[i]*simple_feature_matrix[:, i]) ) )

print ro

# Implement single coordinate descent step
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
	# compute predictions
	predictions = predict_output(feature_matrix, weights)
	# compute  ro[i] = SUM [ [feature_i]*(output-predictions + w[i]*[feature_i]) ]
	ro_i = np.sum(feature_matrix[:,i] * (output - predictions + weights[i]*feature_matrix[:, i]) )

	if i == 0: # we don't regularize intercept
		new_weight_i = ro_i
	elif ro_i < -l1_penalty/2:
		new_weight_i = ro_i + l1_penalty/2
	elif ro_i > l1_penalty/2:
		new_weight_i = ro_i - l1_penalty/2
	else:
		new_weight_i = 0

	return new_weight_i

# Test above function. should print 0.425558846691
print "\n test for lasso_coordinate_descent_step \n"
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), 
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
	weights = initial_weights
	done = 0
	max_change = 0
	while not done:
		for i in range(len(initial_weights)):
			# save the old weight
			old_weight_i = weights[i]
			weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)

			# compute change
			change = abs(old_weight_i - weights[i])
			if max_change == 0:
				max_change = change
			else:
				if change > max_change:
					max_change = change
		if max_change < tolerance:
			done = 1
		else:
			max_change = 0 # reset max_change for next round

	return weights

# Test
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

print "feature matrix and initial weights"
print normalized_simple_feature_matrix
print initial_weights

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output, initial_weights, l1_penalty, tolerance)
print weights

# split the sales dataset into training and test data
train_data,test_data = sales.random_split(.8,seed=0)

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

my_output = 'price'
(training_feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)

# Normalize training feature data
training_normalized, norm_training = normalize_features(training_feature_matrix)

print "\n experience different l1_penalty value \n"

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e7
tolerance = 1
weights1e7 = lasso_cyclical_coordinate_descent(training_normalized, output, initial_weights, l1_penalty, tolerance)
print weights1e7

normalized_weights1e7 = weights1e7/norm_training
print normalized_weights1e7[3]
print normalized_weights1e7

l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(training_normalized, output, initial_weights, l1_penalty, tolerance)
print weights1e8

normalized_weights1e8 = weights1e8/norm_training
print normalized_weights1e8

l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(training_normalized, output, initial_weights, l1_penalty, tolerance)
print weights1e4

normalized_weights1e4 = weights1e4/norm_training
print normalized_weights1e4



# Evaluate on test data
(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

def get_residual_sum_of_squares(predicted, output):
	# Find the difference between the predicted anc actual price
	diff = predicted - output

	# square and sum
	sqr = diff * diff
	return sqr.sum()

# Calculate RSS for each of the three normalized weights on the test data
weights1e7_predict = predict_output(test_feature_matrix, normalized_weights1e7)
print "RSS of weight 1e7 = %f\n" % (get_residual_sum_of_squares(weights1e7_predict, test_output))
print normalized_weights1e7

weights1e8_predict = predict_output(test_feature_matrix, normalized_weights1e8)
print "RSS of weight 1e8 = %f\n" % (get_residual_sum_of_squares(weights1e8_predict, test_output))
print normalized_weights1e8

weights1e4_predict = predict_output(test_feature_matrix, normalized_weights1e4)
print "RSS of weight 1e4 = %f\n" % (get_residual_sum_of_squares(weights1e4_predict, test_output))
print normalized_weights1e4