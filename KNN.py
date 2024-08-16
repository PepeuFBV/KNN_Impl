import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1/p)

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.method = euclidean_distance

    def fit(self, x, y, method=euclidean_distance):
        self.x_train = x
        self.y_train = y
        self.method = method

    def predict(self, x):
        predictions = [self._predict(i) for i in x] # calls secondary predict function
        return predictions

    # Brute force method
    def _predict(self, x):
        distances = [self.method(x, x_train) for x_train in self.x_train]

        # get the nearest neighbours, argsort returns the indexes that would sort the array without doing it
        k_indexes_found = np.argsort(distances)[:self.k] # gets only the k closest
        k_values = [self.y_train[i] for i in k_indexes_found]

        # return the most common class label
        most_commmon = np.bincount(k_values).argmax()
        return most_commmon

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self