import numpy as np
from collections import Counter

class KNNClassifier:
    """ Hand on KNN Classifier """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # lazy method: save the training data only
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        for x in X_test:
            # for simplicity, use L2
            # compute distance by line, i.e. axis=1
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # get the indice of the distance array
            k_indices = np.argsort(distances)[:self.k]

            # get its label from the y_train
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)