import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2)) # Calculate the Euclidean distance between two points, handles multiple
    # dimensions as it calculates the distance between each dimension and sums them up, then takes the square root of
    # that sum

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

    def fit(self, x, y, method = euclidean_distance):
        self.x_train = x
        self.y_train = y
        self.method = method

    def predict(self, x):
        predictions = []
        for i in x:
            distances = [self.method(i, x_train) for x_train in self.x_train] # Calculate the distance between the
            # test point in comparison to all the training points in the dataset
            k_indexes_found = np.argsort(distances)[:self.k] # Get the indexes of the k nearest points
            k_values = [self.y_train[j] for j in k_indexes_found] # Get the values of the k nearest points
            most_common = np.bincount(k_values).argmax() # Get the most common value of the k nearest points
            predictions.append(most_common)
        return predictions

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self