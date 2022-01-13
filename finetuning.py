from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from modelvalidation import train_X, train_y, val_X, val_y

# Underfitting and Overfitting data
# tree depth is an important factor which determines accuracy of model
# Overfitting -> Model matches training data almost perfectly, but does poorly in validation data
# Underfitting -> Fails to capture important patterns in data, does poorly even in training data
# Refer to finetuning.jpg for a visual representation

# MAE -> Mean absolute error

# Compare MAE scores from different depths (aka different max leaf nodes)


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_Y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    predicted_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, predicted_val)
    return mae


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: " + str(max_leaf_nodes), my_mae)


# From this, we can see that optimal number of leaves is 500
