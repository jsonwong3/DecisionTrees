"""
This file contains implementation of Ensemble Decision Tree classifers.
This includes methodologies such as Bagging, Boosting and Random Forests
"""

import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    """ This class represents a Random Forest classifier.
    (Note: This class also represents a Bagging classifier when features_percent = 1)
    
    Args:
    - num_trees (int): The number of decision trees to use
    - features_percent (float): The percent of features to use to generate a subset
    - data_percent (float): The percent of data to use to generate a subset
    - num_classes (int): The number of class labels
    - max_depth (int): The maximum depth of every decision tree
    - min_leaf_data (int): The minimum number of data required to split
    - min_entropy (float): The minimum entropy required to determine a leaf node
    - max_gini (float): The maximum gini impurity required to determine a leaf node
    - num_split_retries (int): The number of retries if the split fails (i.e. when a split has 0 information gain)
    - method (str): The methodology used to determine splits
    - debug (bool): Will provide debugging information
    - rng (RandomState): The random number generator to generate random splits and permutation.
    """    
    
    def __init__(self,
                 num_trees=10,
                 features_percent=0.5,
                 data_percent=0.5,
                 num_classes=2,
                 max_depth=5,
                 min_leaf_data=10,
                 min_entropy=1e-3,
                 max_gini=1-1e-3,
                 num_split_retries=10,
                 method="entropy",
                 debug=False,
                 rng=np.random):

        self.num_classes = num_classes
        self.debug = debug
        self.rng = rng

        # Random Forest Parameters
        self.num_trees = num_trees
        self.features_percent = features_percent
        self.data_percent = data_percent

        self.forest = []
        self.feature_ids = []
        self.class_names = None
    
        # Decision Tree Parameters
        self.max_depth = max_depth
        self.min_leaf_data = min_leaf_data
        self.min_entropy = min_entropy
        self.max_gini = max_gini
        self.num_split_retries = num_split_retries
        self.method = method

    def build(self, X, y):
        """ This method creates the decision trees of the forest and stores them into a list.

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        """
        (N, D) = X.shape

        num_features_per_tree = int(np.ceil(self.features_percent * D))
        num_data_per_tree = int(np.ceil(self.data_percent * N))

        # For each decision tree, we sample subset of data and features.
        for tree_i in range(self.num_trees):
            if self.debug:
                print(f"Building Tree: {tree_i + 1}")
                
            keep_entries = self.rng.permutation(X.shape[0])[:num_data_per_tree]
            keep_features = self.rng.permutation(X.shape[1])[:num_features_per_tree]
            feat_ids = keep_features
    
            X_sub = np.take(X, keep_entries, 0)
            X_sub = np.take(X_sub, keep_features, 1)
            y_sub = np.take(y, keep_entries, 0)  

            model = DecisionTree(num_classes=self.num_classes,
                                 max_depth=self.max_depth,
                                 min_leaf_data=self.min_leaf_data,
                                 min_entropy=self.min_entropy,
                                 max_gini=self.max_gini,
                                 num_split_retries=self.num_split_retries,
                                 method=self.method,
                                 debug=self.debug,
                                 rng=self.rng)
            model.build(X_sub, y_sub)

            # Add tree to forest
            self.forest.append(model)
            self.feature_ids.append(feat_ids)

    def predict(self, X):
        """ This method predicts the probability of labels given X. 

        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting of N D-dimensional inputs.

        Output:
        - P (ndarray (shape: (N, C))): A NxC matrix consisting of N C-dimensional probabilities for each input using Random Forest.
        """
        forest_pred_y = np.zeros(shape=(self.num_trees, X.shape[0], self.num_classes))

        # Get predictions from all decision trees
        for ii, (current_tree, current_feature_ids) in enumerate(zip(self.forest, self.feature_ids)):
            forest_pred_y[ii] = current_tree.predict(X[:, current_feature_ids])

        average_pred_y = np.average(forest_pred_y, axis=0)
        return average_pred_y

class GradientBoosting:
    """ This class represents a Gradient Boosting classifier.
    
    
    """