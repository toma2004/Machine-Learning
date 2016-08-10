from __future__ import division
import graphlab
import math
import string

products = graphlab.SFrame('amazon_baby.gl')

print products[269]

def remove_punctuation(text):
	return text.translate(None, string.punctuation)

review_without_punctuation = products['review'].apply(remove_punctuation)
# word_count is a dict where key is the word and value is the number of times the word occurs
products['word_count'] = graphlab.text_analytics.count_words(review_without_punctuation)
print products[269]['word_count']

# extract sentiments
# first, remove review with ratings = 3 since it is neutral
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
print products

# split data into training and test sets
train_data, test_data = products.random_split(.8, seed=1)

# train sentiment classifier with logistic regression
sentiment_model = graphlab.logistic_classifier.create(train_data, target='sentiment',  features=['word_count'], validation_set=None)
# extract weights
weights = sentiment_model.coefficients
print weights.column_names()

num_positive_weights = 0
for x in weights['value']:
	if x >= 0:
		num_positive_weights += 1

num_negative_weights = len(weights['value']) - num_positive_weights

# make predictions on test data set
sample_test_data = test_data[10:13]
print sample_test_data['rating']
print sample_test_data[0]['review']
print sample_test_data[1]['review']

scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores

print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data) # get 1 if scores > 0, else -1

# calculate the probability of sureness using logistic function 1 / (1 + e^-score)
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')

### find most positive (and negative) review ###
test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')
top20_prob_review = test_data.topk('probability', k=20)
lowest20_prob_negative_review = test_data.topk('probability', k=20, reverse=True)
print top20_prob_review
print lowest20_prob_negative_review

# Compute accuracy of classifier
def get_classification_accuracy(model, data, true_labels):
	predictions = model.predict(data)
	count_accuracy = 0
	total_sample = len(predictions)
	for i in range(0, total_sample):
		if predictions[i] == true_labels[i]:
			count_accuracy += 1
	print count_accuracy
	print total_sample
	return count_accuracy/total_sample

# Learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

len_significant_words = len(significant_words)

# Trim out the words which are not in significant set above from the original word_count set
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
print train_data[0]['review']
print train_data[0]['word_count']
print train_data[0]['word_count_subset']

# build model with subset of word count
simple_model = graphlab.logistic_classifier.create(train_data, target = 'sentiment', features=['word_count_subset'], validation_set=None)

print simple_model.coefficients

print simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)

# compare between simple model against the sentiment model
print "accuracy of sentiment model on train data = %f" % (get_classification_accuracy(sentiment_model, train_data, train_data['sentiment']))
print "accuracy of simple model on train data = %f" % (get_classification_accuracy(simple_model, train_data, train_data['sentiment']))
print "accuracy of sentiment model on train data = %f" % (get_classification_accuracy(sentiment_model, test_data, test_data['sentiment']))
print "accuracy of simple model on train data = %f" % (get_classification_accuracy(simple_model, test_data, test_data['sentiment']))