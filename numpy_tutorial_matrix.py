import numpy as np

# example1: make an array into numpy array
mylist = [1,2,3,4]
mynparray = np.array(mylist)
print mynparray

# initialize np array with ones() or zeroes()
one_vector = np.ones((2,4)) # 2 rows and 4 cols
print one_vector

zero_vector = np.zeros(4)
print zero_vector

# make a matrix
my_matrix = np.array([[1,2,3], [4,5,6]])
print my_matrix
print my_matrix[1, 2]

# select sequence of elements
print my_matrix[0:2, 2] # Get rows 0-1 and 2nd col elements

select_cols = np.array([1,2])
select_rows = np.array([1])
print my_matrix[select_rows, :]
print my_matrix[:, select_cols]

# Pass in a list of indices to select elements in array
fib_indices = np.array([1,1,2,3])
random_vector = np.random.random(10) # 10 random numbers between 0 and 1
print random_vector
print random_vector[fib_indices]

# Operations on Arrays
print "\nStart array operation example\n"
print mynparray*mynparray
print mynparray**2
print np.sum(mynparray)
print np.average(mynparray)

# dot product (multiply elementwise and sum them up)
print np.dot(mynparray, mynparray)
print np.sum(mynparray*mynparray)

# calculate Euclidean length or magnitude of a vector
print np.sqrt(np.sum(mynparray*mynparray))

# operation on 2D array (matrix)
my_features = np.array([[1.,2.], [3.,4.], [5.,6.], [7.,8.]])
print my_features

my_weights = np.array([0.4, 0.5])
print my_weights

# Perform dot product
print np.dot(my_features, my_weights)

# Multiply matrices
matrix_1 = np.array([[1., 2., 3.],[4., 5., 6.]])
print matrix_1

matrix_2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
print matrix_2

print np.dot(matrix_1, matrix_2)