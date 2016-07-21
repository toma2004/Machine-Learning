import graphlab
import matplotlib.pyplot as plt
import numpy as np
from math import log, sqrt

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

# Create new features
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In dataset, floors was defined with type string
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

# all features

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features, validation_set=None, l2_penalty=0., l1_penalty=1e10)

print model_all.get("coefficients").print_rows(num_rows=18)

# Calculate Residual Sum of Squares (RSS) to see our error margin
# input: numpy arrays of predicted value and actual value
# output: return the RSS
def get_residual_sum_of_squares(predicted, output):
	error = predicted - output
	return np.sum(error*error)

# split data into training, validation, and test data
(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

smallest = 0;
l1 = 0;
for i in np.logspace(1, 7, num=13):
	train_model = graphlab.linear_regression.create(training, target='price', features=all_features, validation_set=None, l2_penalty=0., l1_penalty=i, verbose=False)
	# compute RSS on validation set
	rss_validation = get_residual_sum_of_squares(train_model.predict(validation).to_numpy(), validation['price'].to_numpy())
	if smallest == 0:
		smallest = rss_validation
		l1 = i
	else:
		if(smallest > rss_validation):
			smallest = rss_validation
			l1 = i
	print rss_validation

print "l1_penalty = %d produces the smallest RSS = %f\n" % (l1, smallest)

# limit number of nonzero weights
max_nonzero = 7

l1_penalty_values = np.logspace(8, 10, num=20)

# Find largest l1_penalty that has more non-zeros than max_nonzero
# Find smallest l1_penalty that has fewer non-zeros than max_nonzero
l1_penalty_min = -1
l1_penalty_max = -1
for i in l1_penalty_values:
	train_model = graphlab.linear_regression.create(training, target='price', features=all_features, validation_set=None, l2_penalty=0., l1_penalty=i, verbose=False)
	#print train_model['coefficients']['value']
	print train_model['coefficients']['value'].nnz(), i
	nonzero_cur_model = train_model['coefficients']['value'].nnz()
	if(nonzero_cur_model > max_nonzero):
		if l1_penalty_min == -1:
			l1_penalty_min = i
		else:
			if l1_penalty_min < i:
				l1_penalty_min = i
	elif nonzero_cur_model < max_nonzero:
		if l1_penalty_max == -1:
			l1_penalty_max = i
		else:
			if l1_penalty_max > i:
				l1_penalty_max = i

print "value of l1 penalty max = %d, value of l1 penalty min = %d\n" % (l1_penalty_max, l1_penalty_min)

smallest = 0;
l1 = 0;
best_model_fit_max_nonzero = None
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
for i in l1_penalty_values:
	train_model = graphlab.linear_regression.create(training, target='price', features=all_features, validation_set=None, l2_penalty=0., l1_penalty=i, verbose=False)
	# compute RSS on validation set
	rss_validation = get_residual_sum_of_squares(train_model.predict(validation).to_numpy(), validation['price'].to_numpy())
	# check to see if model has max_nonzero
	nonzero_cur_model = train_model['coefficients']['value'].nnz()
	if  nonzero_cur_model == max_nonzero:
		if smallest == 0:
			smallest = rss_validation
			l1 = i
			best_model_fit_max_nonzero = train_model
		else:
			if(smallest > rss_validation):
				smallest = rss_validation
				l1 = i
				best_model_fit_max_nonzero = train_model
	print rss_validation, nonzero_cur_model

print "l1_penalty = %d produces the smallest RSS = %f\n" % (l1, smallest)
print best_model_fit_max_nonzero.get("coefficients").print_rows(num_rows=18)