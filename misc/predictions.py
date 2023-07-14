import numpy as np
import pandas as pd

from decision_tree_algorithm.directories import transformed_data
from .c45 import build_tree, predict


def make_predictions(input_data):
    # Load the data
    data = pd.read_csv(input_data)

    # Preprocess the data
    data = data.dropna()  # Remove rows with missing values if any
    X = data.drop("decision", axis=1).values  # Features
    y = data["decision"].values  # Target variable

    # Convert categorical variables to numerical using one-hot encoding
    X_encoded = pd.get_dummies(data.drop("decision", axis=1)).values

    # Split the data into training and testing sets (you can adjust the split ratio as needed)
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X_encoded[:split_index], X_encoded[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the decision tree model
    tree = build_tree(X_train, y_train)

    # Make predictions on the test set
    predictions = []
    for sample in X_test:
        predicted_class = predict(tree, sample)
        predictions.append(predicted_class)

    # Evaluate the model
    accuracy = np.mean(predictions == y_test)

    print("Predictions:", predictions)


if __name__ == "__main__":
    make_predictions(transformed_data)
