import numpy as np
import pandas as pd


class DecisionNode:
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        true_branch=None,
        false_branch=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


def entropy(y):
    # print(np.unique(y, return_counts=True))
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def information_gain(X, y, feature, threshold):
    left_indices = X[:, feature] < threshold
    right_indices = X[:, feature] >= threshold

    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])

    left_weight = len(y[left_indices]) / len(y)
    right_weight = len(y[right_indices]) / len(y)

    gain = entropy(y) - (left_weight * left_entropy) - (right_weight * right_entropy)
    return gain


def find_best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        unique_values = np.unique(X[:, feature])
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            gain = information_gain(X, y, feature, threshold)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return DecisionNode(value=np.unique(y)[0])

    best_feature, best_threshold = find_best_split(X, y)

    if best_feature is None or best_threshold is None:
        return DecisionNode(value=np.bincount(y).argmax())

    left_indices = X[:, best_feature] < best_threshold
    right_indices = X[:, best_feature] >= best_threshold

    left_subtree = build_tree(X[left_indices], y[left_indices])
    right_subtree = build_tree(X[right_indices], y[right_indices])

    return DecisionNode(
        feature=best_feature,
        threshold=best_threshold,
        true_branch=left_subtree,
        false_branch=right_subtree,
    )


def predict(node, x):
    if node.value is not None:
        return node.value

    if x[node.feature] < node.threshold:
        return predict(node.true_branch, x)
    else:
        return predict(node.false_branch, x)


