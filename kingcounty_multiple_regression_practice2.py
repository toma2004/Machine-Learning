import graphlab
import matplotlib.pyplot as plt
from math import log

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

# Split data into 80% training and 20% testing with same seed
train_data, test_data = sales.random_split(.8, seed=0)

# Make an example of multiple regression with 3 features
example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target='price', features=example_features, validation_set=None)

example_weight_summary = example_model.get("coefficients")
print example_weight_summary

example_predictions = example_model.predict(train_data)
print example_predictions[0] # should be 271789.505878

def get_residual_sum_of_squares(model, data, outcome):
	predictions = model.predict(data)

	residual = predictions - outcome

	# square and add up
	sqr = residual * residual

	return sqr.sum()


# Check error using RSS
rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print rss_example_train # should be 2.7376153833e+14

# Create new features
# bedrooms_squared = bedrooms*bedrooms
# bed_bath_rooms = bedrooms*bathrooms
# log_sqft_living = log(sqft_living)
# lat_plus_long = lat + long

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']

train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# Define models
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

# use pre-defined linear_regression in graphlab to learn the weights in each model
model1 = graphlab.linear_regression.create(train_data, target='price', features=model_1_features, validation_set=None)
print model1.get("coefficients")

model2 = graphlab.linear_regression.create(train_data, target='price', features=model_2_features, validation_set=None)
print model2.get("coefficients")

model3 = graphlab.linear_regression.create(train_data, target='price', features=model_3_features, validation_set=None)
print model3.get("coefficients")

# Compute RSS on train data
rss_model1_train_data = get_residual_sum_of_squares(model1, train_data, train_data['price'])
print rss_model1_train_data

rss_model2_train_data = get_residual_sum_of_squares(model2, train_data, train_data['price'])
print rss_model2_train_data

rss_model3_train_data = get_residual_sum_of_squares(model3, train_data, train_data['price'])
print rss_model3_train_data

# Compute RSS on test data
rss_model1_test_data = get_residual_sum_of_squares(model1, test_data, test_data['price'])
print rss_model1_test_data

rss_model2_test_data = get_residual_sum_of_squares(model2, test_data, test_data['price'])
print rss_model2_test_data

rss_model3_test_data = get_residual_sum_of_squares(model3, test_data, test_data['price'])
print rss_model3_test_data