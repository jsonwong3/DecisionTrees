"""
This file contains two implementations of the Decision Tree classifer.
One using entropy to determine the best split possible, the other using gini's impurity.
"""

import numpy as np
from node import Node

class DecisionTree:
    """ This class represents a Decision Tree classifier.

    Args:
    - num_classes (int): The number of class labels
    - max_depth (int): The maximum depth of the decision tree
    - min_leaf_data (int): The minimum number of data required to split
    - min_entropy (float): The minimum entropy required to determine a leaf node
    - max_gini (float): The maximum gini impurity required to determine a leaf node
    - num_split_retries (int): The number of retries if the split fails (i.e. when a split has 0 information gain)
    - method (str): The methodology used to determine splits
    - debug (bool): Will provide debugging information
    - rng (RandomState): The random number generator to generate random splits
    """

    def __init__(self,
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
            self.max_depth = max_depth
            self.min_leaf_data = min_leaf_data
            self.min_entropy = min_entropy
            self.max_gini = max_gini
            self.num_split_retries = num_split_retries
            self.method = method
            self.debug = debug
            self.rng = rng
    
            self.class_names = None
            self.tree = None
    
    def _entropy(self, y):
        """ This method computes the entropy of a categorical distribution given labels y.
        
        Args:
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N labels.
        
        Output:
        - entropy (float): The entropy of a categorical distribution given labels y.
        """
        # Count the number of data points per class
        (_, counts) = np.unique(y, return_counts=True)
        
        # Calculate entropy (measure of uncertainty)
        entropy = 0
        total_entries = sum(counts)
        
        if total_entries != 0:
            entropy = sum(-(counts/total_entries) * np.log2(counts/total_entries))

        return entropy

    def _giniImpurity(self, y):
        """ This method computes the gini impurity of a categorical distribution given labels y.
        
        Args:
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N labels.
        
        Output:
        - giniImpurity (float): The gini impurity of a categorical distribution given labels y.
        """
        # Count the number of data points per class
        (_, counts) = np.unique(y, return_counts=True)
        
        # Calculate gini impurity
        giniImpurity = 1
        total_entries = sum(counts)
        
        if total_entries != 0:
            giniImpurity = 1 - sum((counts/total_entries)*(counts/total_entries))

        return giniImpurity
    

    def _find_entropy_split(self, X, y, H_data):
        """ This method finds the optimal split over a random split dimension using entropy.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        - H_data (float): The entropy of the data before the split.
        
        Outputs:
        - split_dim (int): The split dimension of input features.
        - split_val (float): The value used to determine the left and right splits.
        - max_info_gain (float): The maximum information gain from all possible choices of a split value.
        """
        (N, D) = X.shape

        # Randomly choose the dimension for split
        split_dim = self.rng.randint(D)
        
        # Sort data based on column at split dimension
        sort_idx = np.argsort(X[:, split_dim])
        X = X[sort_idx]
        y = y[sort_idx]

        # This returns the unique values and their first indicies.
        # Since X is already sorted, we can split by looking at first_idxes.
        (unique_values, first_idxes) = np.unique(X[:, split_dim], return_index=True)

        split_val = 0
        max_info_gain = 0
        
        # Iterate over possible split values and find optimal split that maximizes the information gain.
        for i in range(unique_values.shape[0]-1):
            curr_split_idx = first_idxes[i+1]
            curr_split_val = unique_values[i]
    
            # Split data at the current split index
            y_left = y[:curr_split_idx]
            y_right = y[curr_split_idx:]
            
            # Calculate entropy for each subset
            H_left = self._entropy(y_left)
            H_right = self._entropy(y_right)
            
            # Calculate information gained
            curr_info_gain = (H_data -
                              (len(y_left)/len(y) * H_left) -
                              (len(y_right)/len(y) * H_right))

            # Update maximum information gain when applicable
            if curr_info_gain > max_info_gain:
                split_val = curr_split_val
                max_info_gain = curr_info_gain
                
        if self.debug:
            print(f"Finding Entropy Split: Selected Split Dimension: {split_dim}")
            print(f"Found Maximum Information Gain of {round(max_info_gain, 3)} at {round(split_val, 3)}")
                
        return split_dim, split_val, max_info_gain

    def _find_gini_split(self, X, y):
        """ This method finds the optimal split over a random split dimension using gini impruity.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        
        Outputs:
        - split_dim (int): The split dimension of input features.
        - split_val (float): The value used to determine the left and right splits.
        - min_gini_impurity (float): The minimum gini impurity from all possible choices of a split value.
        """
        (N, D) = X.shape

        # Randomly choose the dimension for split
        split_dim = self.rng.randint(D)
        
        # Sort data based on column at split dimension
        sort_idx = np.argsort(X[:, split_dim])
        X = X[sort_idx]
        y = y[sort_idx]

        # This returns the unique values and their first indicies.
        # Since X is already sorted, we can split by looking at first_idxes.
        (unique_values, first_idxes) = np.unique(X[:, split_dim], return_index=True)

        split_val = 0
        min_gini_impurity = 1
        
        # Iterate over possible split values and find optimal split that maximizes the information gain.
        for i in range(unique_values.shape[0]-1):
            curr_split_idx = first_idxes[i+1]
            curr_split_val = unique_values[i]
    
            # Split data at the current split index
            y_left = y[:curr_split_idx]
            y_right = y[curr_split_idx:]
            
            # Calculate gini impurity for each subset
            G_left = self._giniImpurity(y_left)
            G_right = self._giniImpurity(y_right)
            
            # Calculate weighted gini impurity
            curr_gini_impurity = (len(y_left)/len(y) * G_left) + (len(y_right)/len(y) * G_right)
        
            # Update minimum gini impurity when applicable
            if curr_gini_impurity < min_gini_impurity:
                split_val = curr_split_val
                min_gini_impurity = curr_gini_impurity
        
        if self.debug:
            print(f"Finding Gini Impurity Split: Selected Split Dimension: {split_dim}")
            print(f"Found Minimum Impurity of {round(min_gini_impurity, 3)} at {round(split_val, 3)}")
        
        return split_dim, split_val, min_gini_impurity
    
    def _build_tree(self, X, y, level):
        """ This method builds the decision tree from a specified level recursively.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        - level (int): The current level (depth) of the tree. NOTE: 0 <= level
        
        Output:
        - current_node (Node): The node at the specified level.
        
        NOTE: The Node class is the defined with the following attributes:
        - is_leaf
          - is_leaf == True -> probs
          - is_leaf == False -> split_dim, split_value, left, right
        """
        
        # During the first iteration, grab the names of each class
        if level == 0:
            self.class_names = np.unique(y)
        
        (N, D) = X.shape
        
        # Determine whether we should stop splitting the data and define a leaf node
        if (N < self.min_leaf_data or
            self.max_depth <= level or
            (self.method == "entropy" and self._entropy(y) < self.min_entropy) or
            (self.method == "gini" and self._giniImpurity(y) > self.max_gini)):

            # Count the number of labels per class and compute the probabilities.
            (_, counts) = np.unique(np.append(self.class_names, y), return_counts=True)
            counts -= 1
            probs = np.expand_dims(counts / N, axis=1)

            current_node = Node(is_leaf=True, probs=probs)

            if self.debug:
                print(f"Leaf Node Created with {N} Entries at Depth {level} with Probabilities {np.round(probs.T, 3)}")

            return current_node

        # Find an optimal split. If zero information is gained, repeat.
        for _ in range(self.num_split_retries + 1):
            if self.method == "entropy":
                H_data = self._entropy(y)
                split_dim, split_value, maximum_information_gain = self._find_entropy_split(X, y, H_data)
                
                if maximum_information_gain > 0:
                    break
            elif self.method == "gini":
                split_dim, split_value, minimum_gini_impurity = self._find_gini_split(X, y)
                
                if minimum_gini_impurity < 1:
                    break
                
        # Find indicies for left and right splits
        left_split = X[:, split_dim] <= split_value
        right_split = X[:, split_dim] > split_value

        if self.debug:
            print(f"Left Split Size: {left_split.sum()}")
            print(f"Right Split Size: {right_split.sum()}")

        # Build left and right sub-trees
        left_child = self._build_tree(X[left_split], y[left_split], level + 1)
        right_child = self._build_tree(X[right_split], y[right_split], level + 1)

        # Initalize decision tree node
        current_node = Node(split_dim=split_dim,
                            split_value=split_value,
                            left=left_child,
                            right=right_child,
                            is_leaf=False)
        return current_node  

    def build(self, X, y):
        """ Builds the decision tree from root level.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
        """
        self.tree = self._build_tree(X, y, 0)

    def _predict_tree(self, X, node):
        """ This method predicts the probability of labels given X from a specified node recursively.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        - node (Node): The starting node to determine the probability of labels.
        
        Output:
        - probs_data (ndarray (shape: (N, C))): A NxC matrix consisting N C-dimensional probabilities for each input.
        """
        (N, D) = X.shape
        
        # If testing set is empty, return empty probabilties 
        if N == 0:
            return np.empty(shape=(0, self.num_classes))

        if node.is_leaf:
            # node.probs is shape (C, 1)
            return np.repeat(node.probs.T, repeats=N, axis=0)

        left_split = X[:, node.split_dim] <= node.split_value
        right_split = X[:, node.split_dim] > node.split_value

        # Compute the probabilities following the left and right sub-trees
        probs_left = self._predict_tree(X[left_split], node.left)
        probs_right = self._predict_tree(X[right_split], node.right)

        # Combine the probabilities returned from left and right sub-trees
        probs_data = np.zeros(shape=(N, self.num_classes))
        probs_data[left_split] = probs_left
        probs_data[right_split] = probs_right
        return probs_data

    def predict(self, X):
        """ This method predict the probability of labels given X.
        
        Args:
        - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
        
        Output:
        - P (ndarray (shape: (N, C))): A NxC matrix consisting N C-dimensional probabilities for each input.
        """
        return self._predict_tree(X, self.tree)
