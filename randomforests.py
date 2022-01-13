# Intuition behind random forests
# Uses many trees, and it makes a prediction by averaging predictions of each component free
# Generally has much better predictive accuracy than a single decision tree
# Works well with default parameters

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from modelvalidation import train_X, train_y, val_X, val_y

# Fit random forest model to data - similar to Decision Trees
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

# Predict values
predictions = forest_model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
