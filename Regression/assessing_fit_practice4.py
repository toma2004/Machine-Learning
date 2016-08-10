import graphlab
import matplotlib.pyplot as plt

# warm-up examples
tmp = graphlab.SArray([1., 2., 3.])
tmp_cubed = tmp.apply(lambda x: x**3)
print tmp
print tmp_cubed

ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe

def polynomial_sframe(feature, degree):
	poly_frame = graphlab.SFrame()
	poly_frame['power_1'] = feature

	if degree > 1:
		for power in range(2, degree+1):
			name = 'power_' + str(power)
			tmp_pwr = feature.apply(lambda x: x**power)
			poly_frame[name] = tmp_pwr

	return poly_frame

print polynomial_sframe(tmp, 3)

# Visualizing polynomial regression
sales = graphlab.SFrame('kc_house_data_regression1/kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

model1 = graphlab.linear_regression.create(poly1_data, target='price', features = ['power_1'], validation_set=None)
print model1.get("coefficients")
plt.plot(poly1_data['power_1'], poly1_data['price'], '.', poly1_data['power_1'],model1.predict(poly1_data), '-')
plt.show()

# second polynomial
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
print model2.get("coefficients")
plt.plot(poly2_data['power_1'],poly2_data['price'],'.', poly2_data['power_1'], model2.predict(poly2_data),'-')
plt.show()

# third polynomial
poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features3 = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features3, validation_set = None)
print model3.get("coefficients")
plt.plot(poly3_data['power_1'],poly3_data['price'],'.', poly3_data['power_1'], model3.predict(poly3_data),'-')
plt.show()

# try 15th polynomial
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features15, validation_set = None)
print model15.get("coefficients")
plt.plot(poly15_data['power_1'],poly15_data['price'],'.', poly15_data['power_1'], model15.predict(poly15_data),'-')
plt.show()

# Split the original data into 4 different subsets
tmp_set_1, tmp_set_2 = sales.random_split(.5, seed=0)
set_1, set_2 = tmp_set_1.random_split(.5, seed=0)
set_3, set_4 = tmp_set_2.random_split(.5, seed=0)

# try 15th polynomial on set1
poly15_set1_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features15_set1 = poly15_set1_data.column_names() # get the name of the features
poly15_set1_data['price'] = set_1['price'] # add price to the data since it's the target
model15_set1 = graphlab.linear_regression.create(poly15_set1_data, target = 'price', features = my_features15_set1, validation_set = None)
print model15_set1.get("coefficients").print_rows(num_rows = 16)
plt.plot(poly15_set1_data['power_1'],poly15_set1_data['price'],'.', poly15_set1_data['power_1'], model15_set1.predict(poly15_set1_data),'-')
plt.show()

# try 15th polynomial on set2
poly15_set2_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features15_set2 = poly15_set2_data.column_names() # get the name of the features
poly15_set2_data['price'] = set_2['price'] # add price to the data since it's the target
model15_set2 = graphlab.linear_regression.create(poly15_set2_data, target = 'price', features = my_features15_set2, validation_set = None)
print model15_set2.get("coefficients").print_rows(num_rows = 16)
plt.plot(poly15_set2_data['power_1'],poly15_set2_data['price'],'.', poly15_set2_data['power_1'], model15_set2.predict(poly15_set2_data),'-')
plt.show()

# try 15th polynomial on set3
poly15_set3_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features15_set3 = poly15_set3_data.column_names() # get the name of the features
poly15_set3_data['price'] = set_3['price'] # add price to the data since it's the target
model15_set3 = graphlab.linear_regression.create(poly15_set3_data, target = 'price', features = my_features15_set3, validation_set = None)
print model15_set3.get("coefficients").print_rows(num_rows = 16)
plt.plot(poly15_set3_data['power_1'],poly15_set3_data['price'],'.', poly15_set3_data['power_1'], model15_set3.predict(poly15_set3_data),'-')
plt.show()

# try 15th polynomial on set4
poly15_set4_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features15_set4 = poly15_set4_data.column_names() # get the name of the features
poly15_set4_data['price'] = set_4['price'] # add price to the data since it's the target
model15_set4 = graphlab.linear_regression.create(poly15_set4_data, target = 'price', features = my_features15_set4, validation_set = None)
print model15_set4.get("coefficients").print_rows(num_rows = 16)
plt.plot(poly15_set4_data['power_1'],poly15_set4_data['price'],'.', poly15_set4_data['power_1'], model15_set4.predict(poly15_set4_data),'-')
plt.show()


def get_residual_sum_of_squares(model, data, outcome):
	predictions = model.predict(data)

	residual = predictions - outcome

	# square and add up
	sqr = residual * residual

	return sqr.sum()

# selecting a polynomial degree using "validation set"
training_and_validation, testing = sales.random_split(0.9, seed = 1)
training, validation = training_and_validation.random_split(0.5, seed = 1)
model = None
for i in range(1, 16):
	poly_degree = polynomial_sframe(training['sqft_living'], i)

	my_features = poly_degree.column_names()

	poly_degree['price'] = training['price']

	model = graphlab.linear_regression.create(poly_degree, target= 'price', features = my_features, validation_set = None, verbose = False)

	# computre RSS on validation data
	poly_degree_validation = polynomial_sframe(validation['sqft_living'], i)

	poly_degree_validation['price'] = validation['price']

	# predicted = model.predict(poly_degree_validation)

	# error = predicted - validation['price']

	rss = get_residual_sum_of_squares(model, poly_degree_validation, validation['price'])

	print "degree %d has rss = %.2f" % (i, rss)

# degree number 6 produces the lowest RSS. Let use degree 6 on Test data and compute RSS
poly_degree_test = polynomial_sframe(testing['sqft_living'], 6)
poly_degree_test['price'] = testing['price']
rss_test = get_residual_sum_of_squares(model, poly_degree_test, testing['price'])
print "rss_test = %.2f" % rss_test