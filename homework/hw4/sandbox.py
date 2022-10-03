import numpy as np
os.chdir("/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/GitHub/csci5521/homework/hw4/hw4")

# read in data.
# training data
train_data = np.genfromtxt("optdigits_train.txt",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1].astype('int')

# validation data
valid_data = np.genfromtxt("optdigits_valid.txt",delimiter=",")
valid_x = valid_data[:,:-1]
valid_y = valid_data[:,-1].astype('int')

# test data
test_data = np.genfromtxt("optdigits_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1].astype('int')

candidate_min_entropy = [0.01,0.05,0.1,0.2,0.5,0.8,1,2.0]



predicted_classes = np.unique(train_y)

p = []

for i in predicted_classes:
    p.append(len(np.where(train_y == i)[0])/len(train_y))
    # p.append(len(train_y[np.where(train_y == i)])/len(train_y))
    
k_sum = 0

# Add 
for i in p:
    k_sum += -i * np.log2(i + 1e-15)
    
def compute_split_entropy(left_y, right_y):
    
    # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y are labels for the two branches

    weight_left = len(left_y)/(len(left_y) + len(right_y))                           # proportion of elements in the left and right arrays
    weight_right = len(right_y)/(len(left_y) + len(right_y))
    
    node_entropy_left = weight_left * compute_node_entropy(left_y)                   # individual node entropies 
    node_entropy_right = weight_right * compute_node_entropy(right_y)

    split_entropy = node_entropy_left + node_entropy_right 

    return split_entropy

def compute_node_entropy(label):
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

def select_feature(data,label):
    
    # iterate through all features and compute their corresponding entropy
    best_feat = 0
    cur_entropy = []
    
    for i in range(len(data[0])):
        left_y = []
        right_y = []
        
        for j in range(len(data[:,i])):
            
        # compute the entropy of splitting based on the selected features
            if data[j,i] != best_feat:
                right_y.append(label[j])
                
            else:
                left_y.append(label[j])
                
        cur_entropy.append(compute_split_entropy(left_y,right_y)) 

    # Select the feature (column) with minimum entropy
    best_feat = np.argmin(cur_entropy)

    return best_feat

compute_node_entropy(train_y)
compute_split_entropy


cur_node = Tree_node()
data_left, data_right, label_left, label_right = [], [], [], []
        
# compute the node entropy
node_entropy = compute_node_entropy(train_y)
        
        # ----------
        # Base case
        # ----------
        # determine if the current node is a leaf node based on minimum node entropy 
# (if yes, find the corresponding class label with majority voting and exit the current recursion)
if node_entropy < 0.01:
    
    class_counts = np.bincount(train_y)
    cur_node.label = np.argmax(class_counts) # majority vote as the label 
    
    
selected_feature = select_feature(data = train_x,label = train_y)

a = np.arange(12).reshape(3, 4)
np.delete(a, 2, axis=1)

feat_0 = train_x[np.where(train_x[:, selected_feature] == 0)]
feat_0 = np.delete(feat_0, selected_feature, axis=1)
label_0 = train_y[np.where(train_x[:, selected_feature] == 0)]


feat_1 = train_x[np.where(train_x[:, selected_feature] == 1)]
feat_1 = np.delete(feat_1, selected_feature, axis=1)
label_1 = train_y[np.where(train_x[:, selected_feature] == 1)]

# second index in train_x is col (feature) index
feat_entropies_0, feat_entropies_1 = [], []

cur_entropy = []

# Loop over columns (features)
for i in range(len(train_x[0])):
    
    # Split y based on the ith feature
    # x = 0 is the left branch, x = 1 is the right branch
    feat_entropies_0 = train_y[np.where(train_x[:, i] == 0)]
    feat_entropies_1 = train_y[np.where(train_x[:, i] == 1)]
    
    cur_entropy.append(compute_split_entropy(feat_entropies_0, feat_entropies_1))
    

    
