import graphlab
import matplotlib.pyplot as plt
import numpy as np
import sys

def polynomial_sframe(feature, degree):
	poly_frame = graphlab.SFrame()
	poly_frame['power_1'] = feature

	if degree > 1:
		for power in range(2, degree+1):
			name = 'power_' + str(power)
			tmp_pwr = feature.apply(lambda x: x**power)
			poly_frame[name] = tmp_pwr

	return poly_frame

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

l2_small_penalty = 1e-5
#l2_small_penalty = 1e5

# poly15_data = polynomial_sframe(sales['sqft_living'], 15)

# my_features = poly15_data.column_names() # get the name of the features

# poly15_data['price'] = sales['price']

# model15 = graphlab.linear_regression.create(poly15_data, target='price', features = my_features, l2_penalty=l2_small_penalty, validation_set=None, verbose=False)
# print model15.get("coefficients")
# plt.plot(poly15_data['power_1'],poly15_data['price'],'.', poly15_data['power_1'], model15.predict(poly15_data),'-')
# plt.show()

# # Split the original data into 4 different subsets
# tmp_set_1, tmp_set_2 = sales.random_split(.5, seed=0)
# set_1, set_2 = tmp_set_1.random_split(.5, seed=0)
# set_3, set_4 = tmp_set_2.random_split(.5, seed=0)

# # try 15th polynomial on set1
# poly15_set1_data = polynomial_sframe(set_1['sqft_living'], 15)
# my_features15_set1 = poly15_set1_data.column_names() # get the name of the features
# poly15_set1_data['price'] = set_1['price'] # add price to the data since it's the target
# model15_set1 = graphlab.linear_regression.create(poly15_set1_data, target = 'price', features = my_features15_set1, l2_penalty=l2_small_penalty, validation_set = None)
# print model15_set1.get("coefficients").print_rows(num_rows = 16)
# plt.plot(poly15_set1_data['power_1'],poly15_set1_data['price'],'.', poly15_set1_data['power_1'], model15_set1.predict(poly15_set1_data),'-')
# plt.show()

# # try 15th polynomial on set2
# poly15_set2_data = polynomial_sframe(set_2['sqft_living'], 15)
# my_features15_set2 = poly15_set2_data.column_names() # get the name of the features
# poly15_set2_data['price'] = set_2['price'] # add price to the data since it's the target
# model15_set2 = graphlab.linear_regression.create(poly15_set2_data, target = 'price', features = my_features15_set2, l2_penalty=l2_small_penalty, validation_set = None)
# print model15_set2.get("coefficients").print_rows(num_rows = 16)
# plt.plot(poly15_set2_data['power_1'],poly15_set2_data['price'],'.', poly15_set2_data['power_1'], model15_set2.predict(poly15_set2_data),'-')
# plt.show()

# # try 15th polynomial on set3
# poly15_set3_data = polynomial_sframe(set_3['sqft_living'], 15)
# my_features15_set3 = poly15_set3_data.column_names() # get the name of the features
# poly15_set3_data['price'] = set_3['price'] # add price to the data since it's the target
# model15_set3 = graphlab.linear_regression.create(poly15_set3_data, target = 'price', features = my_features15_set3, l2_penalty=l2_small_penalty, validation_set = None)
# print model15_set3.get("coefficients").print_rows(num_rows = 16)
# plt.plot(poly15_set3_data['power_1'],poly15_set3_data['price'],'.', poly15_set3_data['power_1'], model15_set3.predict(poly15_set3_data),'-')
# plt.show()

# # try 15th polynomial on set4
# poly15_set4_data = polynomial_sframe(set_4['sqft_living'], 15)
# my_features15_set4 = poly15_set4_data.column_names() # get the name of the features
# poly15_set4_data['price'] = set_4['price'] # add price to the data since it's the target
# model15_set4 = graphlab.linear_regression.create(poly15_set4_data, target = 'price', features = my_features15_set4, l2_penalty=l2_small_penalty, validation_set = None)
# print model15_set4.get("coefficients").print_rows(num_rows = 16)
# plt.plot(poly15_set4_data['power_1'],poly15_set4_data['price'],'.', poly15_set4_data['power_1'], model15_set4.predict(poly15_set4_data),'-')
# plt.show()

# train_valid, test = sales.random_split(0.9, seed=1)
# train_valid_shuffle = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

# n = len(train_valid_shuffle)
# k = 10 # 10-fold cross validation

# for i in range(k):
# 	start = (n*i)/k
# 	end = (n*(i+1))/k - 1
# 	print i, (start,end)

# # array slicing
# validation4 = train_valid_shuffle[(n*3)/k : (n*(3+1))/k]
# print int(round(validation4['price'].mean(), 0))

# # simple slice data to exclude the validation data
# temp4 = train_valid_shuffle[ 0 : (n*3)/k ]
# train4 = temp4.append(train_valid_shuffle[ (n*(3+1))/k : n ])

# print int(round(train4['price'].mean(), 0))

# Calculate Residual Sum of Squares (RSS) to see our error margin
# input: numpy arrays of predicted value and actual value
# output: return the RSS
def get_residual_sum_of_squares(predicted, output):
	error = predicted - output
	return np.sum(error*error)

# Function to compute average compute validation error (RSS)
# @input parameter: k-fold, l2_penalty, dataframe containing input features, column of output values (price), and list of features
def k_fold_cross_validation(k, l2_penalty, data, output, feature_list):
	n = len(data)
	sum_rss_error = 0
	for i in range(k):
		start = (n*i) / k
		end   = (n *(i+1)) /k - 1
		validation = data[start : end+1]
		temp_train = data[0 : start]
		train = temp_train.append(data[end+1 : n])

		model = graphlab.linear_regression.create(train, target = 'price', features = feature_list, l2_penalty=l2_penalty, validation_set = None)
		# plt.plot(validation['power_1'],validation['price'],'.', validation['power_1'], model.predict(validation),'-')
		# plt.show()
		predictions = model.predict(validation)
		numpy_predictions = predictions.to_numpy()
		numpy_validation_output = validation[output].to_numpy()
		rss_error = get_residual_sum_of_squares(numpy_predictions, numpy_validation_output)
		sum_rss_error += rss_error

	return sum_rss_error / k # return average compute validation error over k segments


poly15_data_k_fold_validation = polynomial_sframe(sales['sqft_living'], 15)
feature_list = poly15_data_k_fold_validation.column_names()
poly15_data_k_fold_validation['price'] = sales['price']
# k_fold_average_cv_error = k_fold_cross_validation(5, 1e5, poly15_data_k_fold_validation, 'price', feature_list)
# print "my average cn error = %f\n" % k_fold_average_cv_error

smallest = 0
l2_penalty_smallest = 0
mylist = []
for i in np.logspace(1, 7, num=13):
	k_fold_average_cv_error = k_fold_cross_validation(10, i, poly15_data_k_fold_validation, 'price', feature_list)
	if(smallest == 0):
		smallest = k_fold_average_cv_error
	else:
		if(k_fold_average_cv_error < smallest):
			smallest = k_fold_average_cv_error
			l2_penalty_smallest = i
	mylist.append(k_fold_average_cv_error)

print "my smallest average CV = %.2f with l2 penalty = %.2f\n" % (smallest, l2_penalty_smallest)

for i in mylist:
	print i