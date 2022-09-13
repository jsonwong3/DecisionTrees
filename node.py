"""
This file contains an implementation of a node for the Decision Tree classifer.
"""
import numpy as np

class Node:
    """ This class represents a node for the Decision Tree classifier.

    Args:
    - split_dim (int): The split dimension of the input features.
    - split_value (float): The value used to determine the left and right splits.
    - left (Node): The left sub-tree.
    - right (Node): The right sub-tree.
    - is_leaf (bool): Whether the node is a leaf node.
    - probs (ndarray (shape: (C, 1))): The C-column vector consisting the probabilities of classifying each class.
    """        
    def __init__(self,
                 split_dim=None,
                 split_value=None,
                 left=None,
                 right=None,
                 is_leaf=False,
                 probs=0.):

        self.is_leaf = is_leaf
        
        if self.is_leaf:
            self.probs = probs
        else:
            self.split_dim = split_dim
            self.split_value = split_value
            self.left = left
            self.right = right