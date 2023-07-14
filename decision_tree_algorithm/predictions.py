import numpy as np
import pandas as pd

from decision_tree_algorithm.c45_algorithm import C45DecisionTree
from decision_tree_algorithm.directories import transformed_data


def make_prediction(input_data):
    # Load the transformed data from the CSV file
    data = pd.read_csv(transformed_data)

    # Split the data into 80% training and 20% testing
    train_data = data.sample(frac=0.1, random_state=42)
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

    # print("Before the test Predictions")
    # print(X_test)

    X_test["predicted_decision"] = decision

    # print("After the test Predictions")
    # print(X_test)

    compare_y_test_and_predicted = pd.DataFrame(
        {"y_test": y_test, "predicted_decision": decision}
    )

    accurancy = np.mean(decision == y_test)

    return train_data, X_test
