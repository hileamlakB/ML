import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node():
    def __init__(self, feature_index=None, thresh_hold=None, left=None, right=None, info_gain=None, value=None):

        self.feature_index = feature_index
        self.thresh_hold = thresh_hold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf nodes
        self.value = value


class RegressionTree():
    def __init__(self, max_depth=5, min_leaf_size=5):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.tree = None

    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_leaf_size and curr_depth <= self.max_depth:
            best_split = self.get_best_split(
                dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                # if the info_gain is 0 then we can't split the data because it is already a pure devision
                # we can't split the data any further
                left_subtree = self.build_tree(
                    best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(
                    best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["thresh_hold"],  left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)  # the majority class
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresh_holds = np.unique(feature_values)
            for threshold in possible_thresh_holds:
                dataset_left, dataset_right = self.split(
                    dataset, feature_index, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -
                                                 1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(
                        y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["thresh_hold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset, feature_index, thresh_hold):
        dataset_left = []
        dataset_right = []
        for row in dataset:
            if row[feature_index] <= thresh_hold:
                dataset_left.append(row)
            else:
                dataset_right.append(row)
        return np.array(dataset_left), np.array(dataset_right)

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        print(feature_val, tree.feature_index, tree.thresh_hold)
        if feature_val <= tree.thresh_hold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)



# RegressionTree is a greedy alogirhtm
# it doesn't consider future splits and
# doesn't do backtracking

# some of the algorithms are like CART
# while other backtracking algorithms exhaustive CHAID
# another backtraking algorithm is Best-fit algorithm
# which uses heurisitic search


# col_names = ['sepal_length', 'sepal_width',
#              'petal_length', 'petal_width', 'type']
data = pd.read_csv('iris.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = RegressionTree(min_leaf_size=3, max_depth=3)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#r_tree_classifier = RegressionTree()
# print(r_tree_classifier.build_tree(data.values))
#print(data.values[:, 3:4])
