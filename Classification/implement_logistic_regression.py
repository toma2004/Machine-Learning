import graphlab
import matplotlib.pyplot as plt
import numpy as np
import json
import string
from math import sqrt

products = graphlab.SFrame('amazon_baby_subset.gl')

# Explore the dataset
print products.head(10)['name']
print "# of positive reviews = ", len(products[products['sentiment']==1])
print "# of negative reviews = ", len(products[products['sentiment']==-1])

# read in 193 important words from json file
with open('important_words.json', 'r') as f:
	important_words = json.load(f)

important_words = [str(s) for s in important_words]
print important_words

def remove_punctuation(text):
	return text.translate(None, string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)

# count number of important words in reviews
for word in important_words:
	products[word] = products['review_clean'].apply(lambda s: s.split().count(word))

list_contain_perfect = []
# count number of perfect
for i in products['perfect']:
	if i >= 1:
		list_contain_perfect.append(1)
	else:
		list_contain_perfect.append(0)

products['contains_perfect'] = list_contain_perfect

print products['contains_perfect'], products['perfect']
print products['review_clean'][3]
print products['contains_perfect'].sum()
print products

# convert SFrame to numpy
def get_numpy_data(data_sframe, features, label):
	data_sframe['intercept'] = 1
	features = ['intercept'] + features
	features_sframe = data_sframe[features]
	feature_matrix = features_sframe.to_numpy()
	label_sarray = data_sframe[label]
	label_array = label_sarray.to_numpy()
	return (feature_matrix, label_array)

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
print "\n feature matrix \n"
print feature_matrix
print feature_matrix.shape
print sentiment

# estimate conditional probability with link (sigmoid) function
# P (y_i = +1 | x_i, w) = 1/(1+exp(-W transposed * H(xi)))
'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(features_matrix, coefficients):
	scores = np.dot(features_matrix, coefficients)

	predictions = 1/(1 + np.exp(-scores))

	return predictions

# testing above code
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients)

# compute derivative of log likelihood
def feature_derivative(error, feature):
	derivative = np.dot(error, feature)
	return derivative

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp

 # test above code
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_log_likelihood           =', correct_ll
print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)

# implement logistic regression using log likelihood function
def logistic_regression(feature_matrix, sentiment, initial_coefficient, step_size, max_iter):
	coefficients = np.array(initial_coefficient)
	for itr in xrange(max_iter):
		# predict P(y_i = +1 | x_i, w)
		predictions = predict_probability(feature_matrix, coefficients)

		# compute the indicator
		indicator = (sentiment ==+1)

		# compute error
		errors = indicator - predictions
		for j in xrange(len(coefficients)):
			derivative = feature_derivative(errors, feature_matrix[:,j])

			coefficients[j] = coefficients[j] + step_size*derivative

		if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
			lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
			print 'iteration %*d: log likelihood of observed labels = %.8f' % (int(np.ceil(np.log10(max_iter))), itr, lp)
	
	return coefficients

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficient=np.zeros(194), step_size=1e-7, max_iter=301)

scores = np.dot(feature_matrix, coefficients)

predict_result = []
positive_count = 0
for score in scores:
	if score > 0:
		predict_result.append(1)
		positive_count += 1
	else:
		predict_result.append(-1)

print "number of predicted positive sentiment = %d\n" % (positive_count)

# word contributes most to positive & negative sentiments
coefficients = list(coefficients[1:]) #exclude the intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

print "Top 10 positive words are %s\n" % (word_coefficient_tuples[0:10])
print "Top 10 negative words are %s\n" % (word_coefficient_tuples[-10:])