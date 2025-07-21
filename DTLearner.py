""""""  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			 	 	 		 		 	
or edited.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		 	 	 			  		 			 	 	 		 		 	
"""  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		 	 	 			  		 			 	 	 		 		 	
  		  	   		 	 	 			  		 			 	 	 		 		 	
  # DTLearner.py

import numpy as np

class DTLearner(object):
    """
    Regression Decision Tree learner following the pseudocode from the course slides.
    Nodes are stored in a single (n_rows × 4) numpy array.

    Returns a single tree, not an entire forest.
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        @param leaf_size: minimum number of samples required to split.
                          If a node has ≤ leaf_size samples, it becomes a leaf.
        @param verbose:   if True, print debug information during tree construction.
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "jlutkemuller3"  # replace with your Georgia Tech username

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 904051695  # replace with your GT ID number

    def study_group(self):
        """
        Returns
            A comma separated string of GT_Name of each member of your study group
            # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

        Return type
            str
        """
        return "jlutkemuller3"

    def add_evidence(self, Xtrain, Ytrain):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # Append Y as the last column so that each row is [x1, x2, …, xD, y].
        data = np.concatenate((Xtrain, Ytrain.reshape(-1, 1)), axis=1)
        self.tree = self._build_tree(data)

    def query(self, Xtest):
        """
        For each row in Xtest, traverse the tree and return predicted Y.
        Xtest: numpy array of features.
        Returns: numpy array of predictions.
        """
        if self.tree is None:
            raise ValueError("DTLearner: must call add_evidence before query.")

        Ypred = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            Ypred[i] = self._query_point(Xtest[i, :], 0)
        return Ypred

    def _build_tree(self, data):
        """
        Recursively build a numpy array representing a decision tree on 'data'.
        """
        # Number of samples (rows) and width (D+1)
        N, width = data.shape
        # Extract target column
        Ys = data[:, -1]

        # ----- Base case 1: too few samples → leaf -----
        if N <= self.leaf_size:
            leaf_val = Ys.mean()
            if self.verbose:
                print(f"Leaf (size ≤ leaf_size): N={N}, leaf_val={leaf_val:.2f}")
            # A leaf node is stored as [−1, leaf_value, nan, nan]
            return np.array([[-1, leaf_val, np.nan, np.nan]])

        # ----- Base case 2: all Y's are identical → leaf -----
        if np.all(Ys == Ys[0]):
            leaf_val = Ys[0]
            if self.verbose:
                print(f"Leaf (all Y identical): N={N}, leaf_val={leaf_val:.5f}")
            return np.array([[-1, leaf_val, np.nan, np.nan]])

        # ----- Otherwise: choose best feature to split on -----
        Xs = data[:, :-1]  # all columns except last
        D = Xs.shape[1]

        # Compute Pearson correlation between each feature and Y.
        corrs = np.zeros(D)
        for i in range(D):
            xi = Xs[:, i]
            if np.std(xi) == 0 or np.std(Ys) == 0:
                corrs[i] = 0.0
            else:
                corrs[i] = np.corrcoef(xi, Ys)[0, 1]

        # Pick feature with maximum absolute correlation
        best_feat = np.argmax(np.abs(corrs))
        split_val = np.median(Xs[:, best_feat])

        # If split does not partition (all ≤ or all >), make a leaf
        if np.all(Xs[:, best_feat] <= split_val) or np.all(Xs[:, best_feat] > split_val):
            leaf_val = Ys.mean()
            if self.verbose:
                print(
                    f"Leaf (no valid split): N={N}, feat={best_feat}, split={split_val:.5f}"
                )
            return np.array([[-1, leaf_val, np.nan, np.nan]])

        # Partition data into left and right subsets
        left_data_mask = Xs[:, best_feat] <= split_val
        right_data_mask = Xs[:, best_feat] > split_val
        left_data = data[left_data_mask, :]
        right_data = data[right_data_mask, :]

        # Build left and right subtrees recursively
        left_tree = self._build_tree(left_data)
        right_tree = self._build_tree(right_data)

        # Create root node:
        root_node = np.array(
            [[
                best_feat,
                split_val,
                1,
                1 + left_tree.shape[0]
            ]]
        )

        # Stack the root starting with the left tree and then the right tree
        return np.vstack((root_node, left_tree, right_tree))

    def _query_point(self, x, node_index):
        """
        Traverse the tree to predict for a single data point x.
        """
        feature = int(self.tree[node_index, 0])
        split_val = self.tree[node_index, 1]

        # If this is a leaf node (feat == -1), return split_val as prediction
        if feature == -1:
            return split_val

        # Otherwise, decide to go left or right
        if x[feature] <= split_val:
            # going left
            left_off = int(self.tree[node_index, 2])
            return self._query_point(x, node_index + left_off)
        else:
            # going right
            right_off = int(self.tree[node_index, 3])
            return self._query_point(x, node_index + right_off)
  		  	   		 	 	 			  		 			 	 	 		 		 	
if __name__ == "__main__":
    pass
    #print("the secret clue is 'zzyzx'")

