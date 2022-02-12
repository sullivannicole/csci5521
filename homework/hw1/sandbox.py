import numpy as np

df = np.genfromtxt("training_data.txt",delimiter=",")
dftest = np.genfromtxt("test_data.txt",delimiter=",")
Xtrain = df[:,0:8] # 9th column (pos = 8) is label
ytrain = df[:, 8]
k = 2
p = [1.0/k for i in range(k)]

#------------------------

# GOES IN FUNC 1, FIT

#------------------------

# compute the mean for each class
distinct_classes = np.unique(ytrain)

def class_data(y_class):
    return Xtrain[ytrain == y_class]

def class_mean(y_class):
    return list(map(np.mean, class_data(y_class).T))

class_means = np.array(list(map(class_mean, distinct_classes)))
class_data_matrix = list(map(class_data, distinct_classes))

def compute_covariance(data, means):
    
    diff_matrix = data - means
    n_subjects = len(data)

    cov_matrix = 1/(n_subjects - 1) * np.dot(diff_matrix.T, diff_matrix)
    
    return cov_matrix.tolist()

# class-independent covariance (S1=S2)
mean_vector = np.array(list(map(np.mean, Xtrain.T)))
cov_matrix = compute_covariance(Xtrain, mean_vector)
S = np.array(cov_matrix)

# compute the class-dependent covariance
cov_matrix_class = []

for i in range(k):
    cov_matrix_class.append(compute_covariance(class_data_matrix[i], class_means[i]))
    
S_class = np.array(cov_matrix_class)

#------------------------

# GOES IN FUNC 1, PREDICT

#------------------------

# class-dependent
S_class_inv =  np.linalg.inv(S_class[0])

W = -0.5 * S_class_inv
w = S_class_inv @ class_means[0]
w0 = -0.5 * (class_means[0].T @ S_class_inv * class_means[0]) - 0.5 * np.log(np.linalg.det(S_class[0])) + np.log(p[0])


# class-independent
S_inv =  np.linalg.inv(S)
w = S_inv * class_means[0]
w0 = -0.5 * class_means[0].T * S_inv * class_means[0] + np.log(p[0])
    
