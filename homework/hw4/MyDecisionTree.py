import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node), -1 means the node is not a leaf node
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):

        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        
        for i in range(len(test_x)):

            cur_node = self.root
            
            while cur_node.right_child != None and cur_node.left_child != None:
                
                if test_x[i,cur_node.feature] == 0:
                    cur_node = cur_node.left_child
                    
                else:
                    cur_node = cur_node.right_child 

            prediction[i] = cur_node.label

        return prediction

    def generate_tree(self, data, label):
        
        # initialize the current tree node
        cur_node = Tree_node()
        
        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
        
        # ----------
        # Base case
        # ----------
        # determine if the current node is a leaf node based on minimum node entropy (ie if we've achieved less than that)
        # (if yes, find the corresponding class label with majority voting and exit the current recursion by returning
        # cur_node)
        if node_entropy < self.min_entropy:

            cur_node.label = np.argmax(np.bincount(label)) # majority vote as the label 

            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # -----------
        # Recursion
        # -----------
        # split the data based on the selected feature and start the next level of recursion

        feat_0 = data[np.where(data[:, selected_feature] == 0)]
        # feat_0 = np.delete(feat_0, selected_feature, axis = 1) # remove the feature used from consideration
        label_0 = label[np.where(data[:, selected_feature] == 0)]


        feat_1 = data[np.where(data[:, selected_feature] == 1)]
        # feat_1 = np.delete(feat_1, selected_feature, axis = 1) # remove the feature used from consideration
        label_1 = label[np.where(data[:, selected_feature] == 1)]
        
        # Recursively grow right and left sides of tree
        cur_node.left_child = self.generate_tree(feat_0, label_0)
        cur_node.right_child = self.generate_tree(feat_1, label_1)
        
        return cur_node


    def select_feature(self,data,label):
        
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        cur_entropy = []
        

        # Loop over columns (features)
        for i in range(len(data[0])):
            
            # Split y based on the ith feature
            # Feature x = 0 is the left branch, feature x = 1 is the right branch
            feat_entropies_0 = label[np.where(data[:, i] == 0)]
            feat_entropies_1 = label[np.where(data[:, i] == 1)]
            
            cur_entropy.append(self.compute_split_entropy(feat_entropies_0, feat_entropies_1))
    
    
        # The best feature is the one with the lowest entropy when the weighted branches are summed
        best_feat = np.argmin(cur_entropy)

        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        
        # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y are labels for the two branches

        p_left = len(left_y)/(len(left_y) + len(right_y))
        p_right = len(right_y)/(len(left_y) + len(right_y))
        
        # Weight each branch by the proportion of observations in that branch
        node_entropy_left = p_left * self.compute_node_entropy(left_y) 
        node_entropy_right = p_right * self.compute_node_entropy(right_y)

        # Sum the branches
        split_entropy = node_entropy_left + node_entropy_right 

        return split_entropy
    

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        predicted_classes = np.unique(label)

        p = []
        
        # Calculate probability for each class
        for i in predicted_classes:
            p.append(len(np.where(label == i)[0])/len(label))
            
        node_entropy = 0

        # Weight probabilities and sum over k - don't need to weight the branch as that happens in compute_split_entropy()
        for i in p:
            node_entropy += -i * np.log2(i + 1e-15)

        return node_entropy
