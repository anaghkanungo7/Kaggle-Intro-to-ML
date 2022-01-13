from sklearn.tree import DecisionTreeRegressor
import exploration as exp
X = exp.X
y = exp.y


# Define model. Specify number for random_state to ensure same results in each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(X, y)

# Making predictions with the model (for first 5 rows)
# print("Making predictions for the following 5 houses: ")
# print(X.head())
# print("The prediction are: ")
# print(model.predict(X.head()))
