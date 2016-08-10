import graphlab
import matplotlib.pyplot as plt
#%matplotlib inline

sales = graphlab.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv')

#sales.show(view="Scatter Plot", x="CrimeRate", y="HousePrice")

# linear regression model with 1 feature = CrimeRate
crime_model = graphlab.linear_regression.create(sales, target="HousePrice", features=['CrimeRate'], validation_set=None, verbose=False)

plt.plot(sales['CrimeRate'], sales['HousePrice'], '.', sales['CrimeRate'], crime_model.predict(sales),'-')
plt.show()

# Print out slope and intercept of the fitting line y= ax + b
print crime_model.get('coefficients')