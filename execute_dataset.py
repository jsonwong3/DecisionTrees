import numpy as np
import pandas as pd

from decision_tree import DecisionTree
from random_forest import RandomForest

def train_test_split(data, size=0.3, idx=0):
    """ This method computes the entropy of a categorical distribution given labels y.
    
    Args:
    - data (ndarray (shape: (N, D+1))): A NxD+1 matrix consisting N labels and D-dimensional inputs.
    - size (float): Proportion of the dataset reserved for testing
    - idx (int): Index of the testing variable
    
    Output:
    - trainX (ndarray (shape: (N*(1-size), D))): A NxD+1 matrix consisting N D-dimensional inputs.
    - trainy (ndarray (shape: (N*(1-size), 1))): A N-column vector consisting N labels.
    - testX (ndarray (shape: (N*size, D))): A NxD+1 matrix consisting N D-dimensional inputs.
    - testy (ndarray (shape: (N*size, 1))): A N-column vector consisting N labels.
    """
    
    (N, D_1) = data.shape
    
    # Randomly select data entries to be part of the testing set
    test_size = int(N*size)
    test_idx = np.random.choice(N, size=test_size, replace=False)
    
    # Select the remaining data entries to be part of the training set
    train_idx = np.arange(N) 
    condition = np.where(np.in1d(train_idx, test_idx), False, True)
    train_idx = train_idx[condition]
    
    # Define each set
    test_data = data[test_idx]
    train_data = data[train_idx]
    
    testy = test_data[:,idx] 
    testX = np.delete(test_data, idx, 1)
    
    trainy = train_data[:,idx]
    trainX = np.delete(train_data, idx, 1)

    return trainX, trainy, testX, testy

def execute_df_test(data, sets=1):
    avg = 0
    for i in range(0, sets):
        train_classes = 0
        test_classes = 1
        
        while train_classes != test_classes:
            trainX, trainy, testX, testy = train_test_split(data)
            
            train_classes = len(np.unique(trainy))
            test_classes = len(np.unique(testy))
            
        dt = DecisionTree(num_classes=2, min_entropy=1e-2, max_gini=1-1e-2, method="entropy")
        dt.build(trainX, trainy)
        results = dt.predict(testX)
        
        predicted_vals = np.argmax(results, axis=1)
        total_correct = np.sum(predicted_vals == testy)
        accuracy = total_correct/len(testy)*100
        avg += accuracy
        
    print(f"Average Accuracy: {avg/sets}")
    
def execute_rf_test(data, sets=1):
    avg = 0
    for i in range(0, sets):
        train_classes = 0
        test_classes = 1
        
        while train_classes != test_classes:
            trainX, trainy, testX, testy = train_test_split(data)
            
            train_classes = len(np.unique(trainy))
            test_classes = len(np.unique(testy))
            
        rf = RandomForest(num_trees=10, num_classes=2, min_entropy=1e-2, max_gini=1-1e-2, method="entropy")
        rf.build(trainX, trainy)
        results = rf.predict(testX)
        
        predicted_vals = np.argmax(results, axis=1)
        total_correct = np.sum(predicted_vals == testy)
        accuracy = total_correct/len(testy)*100
        avg += accuracy
        
    print(f"Average Accuracy: {avg/sets}")

if __name__ == "__main__":
    data = pd.read_csv("./data/winequality.csv")
    data = np.array(data)   
    
    print("Decision Tree:")
    execute_df_test(data, sets=10)
    print("Random Forest:")
    execute_rf_test(data, sets=10)