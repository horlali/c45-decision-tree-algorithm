import numpy as np
import pandas as pd

from decision_tree_algorithm.directories import transformed_data


class Node:
    def __init__(self, feature=None, value=None, decision=None):
        self.feature = feature  # Feature to split on
        self.value = value  # Value of the feature
        self.decision = decision  # Decision if it's a leaf node
        self.children = {}  # Dictionary to store child nodes


class C45DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_instance(x, self.root) for _, x in X.iterrows()])

    def _build_tree(self, X, y):
        # Create a root node
        node = Node()

        # If all instances have the same decision, make it a leaf node
        if y.nunique() == 1:
            node.decision = y.iloc[0]
            return node

        # If no features left or maximum depth reached, make it a leaf node with the majority decision
        if len(X.columns) == 0 or X.shape[1] == 0:
            node.decision = y.mode().iloc[0]
            return node

        # Calculate information gain for each feature
        gains = [self._information_gain(X, y, feature) for feature in X.columns]

        # Select the feature with the highest information gain
        best_feature = X.columns[np.argmax(gains)]

        # Split on the best feature
        node.feature = best_feature
        for value in X[best_feature].unique():
            indices = X[best_feature] == value
            child_X = X.loc[indices].drop(best_feature, axis=1)
            child_y = y[indices]
            node.children[value] = self._build_tree(child_X, child_y)

        return node

    def _predict_instance(self, instance, node):
        if node.decision is not None:
            return node.decision

        value = instance[node.feature]
        if value in node.children:
            return self._predict_instance(instance, node.children[value])

        # If the feature value is not encountered during training, return the majority decision
        return max(node.children.values(), key=lambda x: x.decision).decision

    def _entropy(self, y):
        counts = y.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts))

    def _information_gain(self, X, y, feature):
        entropy_before = self._entropy(y)
        entropy_after = 0

        for value in X[feature].unique():
            indices = X[feature] == value
            entropy = self._entropy(y[indices])
            weight = sum(indices) / len(X)
            entropy_after += weight * entropy

        return entropy_before - entropy_after


# Load the transformed data from the CSV file
data = pd.read_csv(transformed_data)


# Split the data into 80% training and 20% testing
train_data = data.sample(frac=0.5, random_state=42)
test_data = data.drop(train_data.index)


# Separate the features (X) and the target variable (y)
X_train = train_data.drop("decision", axis=1)
y_train = train_data["decision"]

X_test = test_data.drop("decision", axis=1)
y_test = test_data["decision"]


# Create and train the C4.5 decision tree
tree = C45DecisionTree()
tree.fit(X_train, y_train)


decision = tree.predict(X_test)


print("Before the test Predictions")
# print(X_test)

X_test["predicted_decision"] = decision

print("After the test Predictions")
# print(X_test)

compare_y_test_and_predicted = pd.DataFrame(
    {"y_test": y_test, "predicted_decision": decision}
)
print(compare_y_test_and_predicted)
