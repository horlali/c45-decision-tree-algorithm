import numpy as np
import pandas as pd

from c45_decision_tree_algorithm.c45_algorithm import C45DecisionTree
from c45_decision_tree_algorithm.directories import transformed_data


def make_prediction(input_data, test_data=None):
    # Load the transformed data from the CSV file
    data = pd.read_csv(transformed_data)

    # Split the data into 80% training and 20% testing
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Separate the features (X) and the target variable (y)
    X_train = train_data.drop(["decision", "city"], axis=1)

    y_train = train_data["decision"]

    X_test = test_data.drop("decision", axis=1)
    y_test = test_data["decision"]

    # Create and train the C4.5 decision tree
    tree: C45DecisionTree = C45DecisionTree()
    tree.fit(X_train, y_train)

    decision = tree.predict(X_test)

    X_test["predicted_decision"] = decision

    accurancy = np.mean(decision == y_test)

    return train_data, X_test, f"Accurancy: {accurancy}"
