# data exploration exercise @ kaggle.com
import pandas as pd

##  import the model data
melbourne_file_path = './input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data.describe()
melbourne_data = melbourne_data.dropna(axis=0)

# prediction target, represented by variable y
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# the combined features and data, represented by the variable X
X = melbourne_data[melbourne_features]
# review the data defined by just the features
X.describe()
# review to couple of rows
X.head()

# import the DecisionTreeRegressor from sklearn
from sklearn.tree import DecisionTreeRegressor

# define the model. specify a number for random_state to ensure same results on each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# fit model
melbourne_model.fit(X, y)

# run predictions
predictions = melbourne_model.predict(X)
print(predictions)

# import the calculations module for mean_absolute_error
from sklearn.metrics import mean_absolute_error

mean_error = mean_absolute_error(y, predictions)
print(mean_error)


from sklearn.model_selection import train_test_split

# supplying value for random_state allows control over where split is performed
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor(random_state=0)
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predicitons = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predicitons))
