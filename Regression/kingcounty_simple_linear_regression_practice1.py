import graphlab
import matplotlib.pyplot as plt

sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

# Split data into 80% training and 20% testing with same seed
train_data, test_data = sales.random_split(.8, seed=0)

# Function to calculate slope and intercept using closed form technique (set derivatives to zero to find min/max)
# input: input_feature and output are SArrays
# output: return tuple of slope and intercept
############################################
# Formula to calculate slop and intercept from derivative of (h(x) - y)^2
# intercept = avg house sale price - slope * avg(sqft) given sqft is the feature
# slope = (sum(house price * sqft) - (sum(house prince) * sum(sqft)) / N ) / (sum(sqft^2) - (sum(sqft)*sum(sqft)) / N)
############################################
def simple_linear_regression(input_feature, output):
	sum_input = input_feature.sum()
	sum_output = output.sum()
	input_times_output = input_feature * output
	sum_input_times_output = input_times_output.sum()
	sqrt_input = input_feature * input_feature
	sum_sqrt_input = sqrt_input.sum()
	num_house = input_feature.size()

	slope = (sum_input_times_output - (sum_output * sum_input) / num_house) / (sum_sqrt_input - (sum_input * sum_input) / num_house)

	intercept = (sum_output / num_house) - (slope * (sum_input / num_house))

	return (intercept, slope)


# Test to ensure we have correct function
test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)
(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)
print "Intercept: " + str(test_intercept)
print "Slope: " + str(test_slope)

# Get intercept and slope using training data
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)

# Function to calculate predicted prices given our model above
# input: sqrt of a house, slope, and intercept of our model
# output: return predicted price with 2 digits after decimal points
def get_regression_predictions(input, intercept, slope):
	return input * slope + intercept

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

# Plot the data and use standard linear regression function to double check our model
# linear regression model with 1 feature = sqft
# sqft_model = graphlab.linear_regression.create(train_data, target="price", features=['sqft_living'], validation_set=None, verbose=False)

# plt.plot(train_data['sqft_living'], train_data['price'], '.', train_data['sqft_living'], sqft_model.predict(train_data),'-')
# plt.show()

# Calculate Residual Sum of Squares (RSS) to see our error margin
# input: SArray of input feature and output. Then intercept and slope
# output: return the residual sum of squares
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
	# Get the predicted price from our model above
	predicted = get_regression_predictions(input_feature, intercept, slope)

	# Find the difference between the predicted anc actual price
	diff = predicted - output

	# square and sum
	sqr = diff * diff
	return sqr.sum()

# Test to check if our get residual sum of squares is working
print get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope) # should be 0.0

# Predict the error margin we have when use sqft as feature to predict sale price
rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)

# Function to predict the sqft of a house given sale price
# input: price as SArray, intercept, and slope
# output: return sqft
def inverse_regression_predictions(output, intercept, slope):
	return (output - intercept) / slope

# Predict sqft of a house with sale price is $800,000
my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)

########################
# Build new model to use number of bedroom as input feature
########################
# Get intercept and slope using training data
br_intercept, br_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])

print "br model intercept = " + str(br_intercept)
print "br model slope = " + str(br_slope)

my_house_br = 4
estimated_price_based_br = get_regression_predictions(my_house_br, br_intercept, br_slope)
print "The estimated price for a house with %d bredrooms is $%.2f" % (my_house_br, estimated_price_based_br)

# Plot the data and use standard linear regression function to double check our model
# linear regression model with 1 feature = bedroom
br_model = graphlab.linear_regression.create(train_data, target="price", features=['bedrooms'], validation_set=None, verbose=False)

plt.plot(train_data['bedrooms'], train_data['price'], '.', train_data['bedrooms'], br_model.predict(train_data),'-')
plt.show()

# Predict the error margin we have when use number of bedroom as feature to predict sale price
rss_prices_on_br = get_residual_sum_of_squares(train_data['bedrooms'], train_data['price'], br_intercept, br_slope)
print 'The RSS of predicting Prices based on number of bedroom is : ' + str(rss_prices_on_br)