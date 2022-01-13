from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from modelbuilding import model
from exploration import X, y, data

# Calculating mean absolute error
predicted = model.predict(X)
# print(mean_absolute_error(y, predicted))

# This is the in-sample score.
# We used the same sample to build the model and evaluate it - which is a bad practice
# Your sample may be biased towards one or more features and thus may be innacurate
# when used in real-life.

# To resolve this, we exclude some data from the training process
# and call it Validation Data

# Splitting the data into train and test
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define model
better_model = DecisionTreeRegressor()
better_model.fit(train_X, train_y)

better_model_predicted = better_model.predict(val_X)
# print(mean_absolute_error(val_y, better_model_predicted))

# Print the data differences
# print(better_model_predicted[0:5])
# print(val_y.head().tolist())
