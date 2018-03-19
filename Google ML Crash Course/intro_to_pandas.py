#importing pandas and check version
import pandas as pd
pd.__version__

#create a Series object (like a Column of a table)
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

#create a DataFrame object from Series (like a relational table)
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
pd.DataFrame({ 'City name': city_names, 'Population': population })

#create DataFrame object from CSV file
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

#summarise DataFrame and show the first few rows of data
california_housing_dataframe.describe()
california_housing_dataframe.head()

#quick histogram of a column
california_housing_dataframe.hist('housing_median_age')

#### ACCESS DATA ###
#accessing a column from DataFrame
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print type(cities['City name'])
cities['City name']

#accesing a particular cell in DataFrame
print type(cities['City name'][1])
cities['City name'][1]

#accessing a subset of rows in DataFrame
print type(cities[0:2])
cities[0:2]

### MANIPULATE DATA ###
#apply arithmetic operations to a Series
population / 1000

#apply operations from Numpy to a Series
import numpy as np
np.log(population)

#apply a custom function to each value of a Series
population.apply(lambda val: val > 1000000)

#adding more columns to a DataFrame
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
cities

### INDEXING ###
#index of a Series (index values are returned as a list)
city_names.index

#index of a DataFrame (note, indexes are like row numbers)
cities.index

#reindexing data manually
cities.reindex([2, 0, 1])

#reindexing data randomly
cities.reindex(np.random.permutation(cities.index))
