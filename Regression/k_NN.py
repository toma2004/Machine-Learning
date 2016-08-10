import graphlab
import matplotlib.pyplot as plt
import numpy as np
import math

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

# In dataset, floors was defined with type string
sales['floors'] = sales['floors'].astype(int)

# Function to normalize feature matrix
def normalize_features(feature_matrix):
	norms = np.linalg.norm(feature_matrix, axis=0)
	return (feature_matrix /norms, norms)

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

(train_and_validation, test) = sales.random_split(.8, seed=1) # initial train/test split
(train, validation) = train_and_validation.random_split(.8, seed=1) # split training set into training and validation sets

feature_list = ['bedrooms',  
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
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']

features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

print features_test[0]
print features_train[9]

test1 = np.array([1,2])
test2 = np.array([4,5])
def calculate_euclidean_distance (feature_matrix1, feature_matrix2):
	result = np.sqrt(np.sum((feature_matrix1 - feature_matrix2)**2))
	return result

print "euclidean distance = %f\n" % (calculate_euclidean_distance(test1,test2))

query_house = features_test[0]
print query_house
for i in range(10):
	print "distance of house %d to query house = %f\n" % (i, calculate_euclidean_distance(query_house, features_train[i]))

# define diff variable such that diff[i] gives the element-wise difference between the features of the query house and the i-th training house
diff = query_house - features_train[:]

distances = np.sqrt(np.sum(diff**2, axis=1)) # calculate Euclidean distance
print distances

def calculate_distance_using_vectorization(features_train, query_house):
	diff = query_house - features_train[:]
	distances = np.sqrt(np.sum(diff**2, axis=1)) # calculate Euclidean distance
	return distances

def find_1NN(features_train, query_house):
	diff = query_house - features_train[:]
	distances = np.sqrt(np.sum(diff**2, axis=1)) # calculate Euclidean distance
	return np.argmin(distances)

print find_1NN(features_train, features_test[2]) # 1_NN to the thrid house in test set

def find_kNN(k, features_train, query_house):
	# initialize the indices of the first k houses in training set
	distances = calculate_distance_using_vectorization(features_train, query_house)
	distances_k_houses = distances[0:k]
	# print distances_k_houses
	sorted_indices_k_houses = np.argsort(distances_k_houses)
	# print sorted_indices_k_houses
	# print sorted_indices_k_houses[k-1]
	# print distances_k_houses[sorted_indices_k_houses[k-1]]

	for i in range(k, len(features_train)):
		distance_i_query = calculate_euclidean_distance(features_train[i], query_house)
		# print distance_i_query
		# print distances[sorted_indices_k_houses[k-1]]
		if(distance_i_query < distances[sorted_indices_k_houses[k-1]]):
			# find the appropriate location to insert this new distance
			isSmallest = 1
			for j in range(k-2, -1, -1):
				# print distances[sorted_indices_k_houses[j]]
				if(distance_i_query > distances[sorted_indices_k_houses[j]]):
					isSmallest = 0
					sorted_indices_k_houses[j+1:k] = sorted_indices_k_houses[j:k-1] # remove the last distance before inserting the new distance
					sorted_indices_k_houses[j+1] = i
			if isSmallest: # if distance_i_query is the smallest among the initial k NN, put it in the front
				sorted_indices_k_houses[1:k] = sorted_indices_k_houses[0:k-1]
				sorted_indices_k_houses[0] = i
			# print sorted_indices_k_houses
	return sorted_indices_k_houses

print find_kNN(4, features_train, features_test[2])

# function to make single prediction by averaging k nearest neighbor outputs
def k_NN_predict(k, features_train, output_train, query_house):
	indices_k_NN = find_kNN(k, features_train, query_house)

	return (np.sum(output_train[indices_k_NN]) / len(indices_k_NN))
	# print test_output
	# print output_train[8698]
	# sum_test = np.sum(test_output)
	# print sum_test
	# avg = sum_test / len(indices_k_NN)
	# print avg

print k_NN_predict(4, features_train, output_train, features_test[2])

# function to predict the price for each house in test set or validation test
def k_NN_predict_multiple_houses(k, features_train, output_train, features_query):
	print features_query.shape[0]
	# .shape field returns a tuple consisting of (number of elements, number of features in each element)
	for i in range(features_query.shape[0]):
		print "Predicted price for house %dth is %d" % (i, k_NN_predict(k, features_train, output_train, features_query[i]))

k_NN_predict_multiple_houses(10, features_train, output_train, features_test[0:10])

# function to choose the best k.. Utilize the validation set to find the best k
def find_best_k(features_train, output_train, features_valid):
	# make prediction for each house in validation set
	house_price_validationSet = np.empty(features_valid.shape[0])
	rss_all = []
	# print len(house_price_validationSet)
	# print features_valid.shape
	# print house_price_validationSet
	for k in range(1,16):	
		for i in range(features_valid.shape[0]):
			temp_house_price = k_NN_predict(k, features_train, output_train, features_valid[i])
			#print "Predicted price for house %dth is %d" % (i, temp_house_price)
			house_price_validationSet[i] = temp_house_price
		# calculate RSS for each round of k
		rss_all.append(get_residual_sum_of_squares(house_price_validationSet, output_valid))
	print rss_all
	# find the smallest RSS and its corresponding k value
	smallest_index = rss_all.index(min(rss_all))
	print "the value k = %d produces the smallest RSS = %d" % (smallest_index+1, rss_all[smallest_index])

	# plot to visualize performance as a function of k
	kvals = range(1,16)
	plt.plot(kvals, rss_all, 'bo-')

def get_residual_sum_of_squares(predicted, output):
	error = predicted - output
	return np.sum(error*error)

find_best_k(features_train, output_train, features_valid)
