from collections import Counter
class KNNClassifier:
    def __init__(self, k):
        self.k = k
        
    def distance(self, x1, x2):
        return ((x1[0] - x2[0])**2/9) + ((x1[1] - x2[1])**2)
        
    def fit(self, X, y):
        """
        In KNN, "fitting" can be as simple as storing the data, so this has been written for you.
        If you'd like to add some preprocessing here without changing the inputs, feel free,
        but this is completely optional.
        """
        self.X = X
        self.Y = y
        

    def predict(self, X_pred):
        """
        The code in this method should be removed and replaced! We included it
        just so that the distribution code is runnable and produces a
        (currently meaningless) visualization.
        
        Predict classes of points given feature values in X_pred
        
        :param X_pred: a 2D numpy array of (transformed) feature values. Shape is (n x 3)
        :return: a 1D numpy array of predicted classes (Dwarf=0, Giant=1, Supergiant=2).
                 Shape should be (n,)
        """
        
        y_pred = []
        for x_test in X_pred:
            # compute distances between x_test and all training points
            distances = []
            for x_train in self.X:
                dist = self.distance(x_test, x_train)
                distances.append(dist)
            
            # find k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.Y[nearest_indices]
            
            # choose the most common label among neighbors
            label_counts = Counter(nearest_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            y_pred.append(most_common_label)
        
        return np.array(y_pred)
     