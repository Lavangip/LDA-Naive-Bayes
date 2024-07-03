import csv
import numpy as np

def ComputeMeanDiff(X):
    # Compute mean vectors for each class
    mean_vector_0 = np.mean(X[X[:, 2] == 0][:, :2], axis=0)
    mean_vector_1 = np.mean(X[X[:, 2] == 1][:, :2], axis=0)
    return mean_vector_1 - mean_vector_0

def ComputeSW(X):
    # Compute within-class scatter matrix
    mean_vector_0 = np.mean(X[X[:, 2] == 0][:, :2], axis=0)
    mean_vector_1 = np.mean(X[X[:, 2] == 1][:, :2], axis=0)
    n_samples_class_0 = np.sum(X[:, 2] == 0)
    n_samples_class_1 = np.sum(X[:, 2] == 1)
    cov_0 = np.cov(X[X[:, 2] == 0][:, :2].T)
    cov_1 = np.cov(X[X[:, 2] == 1][:, :2].T)
    sw = (n_samples_class_0 * cov_0 + n_samples_class_1 * cov_1) / (n_samples_class_0 + n_samples_class_1)
    return sw

def ComputeSB(X):
    # Compute between-class scatter matrix
    mean_vector_0 = np.mean(X[X[:, 2] == 0][:, :2], axis=0)
    mean_vector_1 = np.mean(X[X[:, 2] == 1][:, :2], axis=0)
    mean_all = np.mean(X[:, :2], axis=0)
    sb = np.outer((mean_vector_1 - mean_vector_0), (mean_vector_1 - mean_vector_0))
    return sb

def GetLDAProjectionVector(X):
    # Compute LDA projection vector
    sw_inverse = np.linalg.inv(ComputeSW(X))
    sb = ComputeSB(X)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(sw_inverse, sb))
    lda_projection_vector = eigenvectors[:, np.argmax(eigenvalues)]
    return lda_projection_vector

def project(x, y, w):
    # Project a point onto the LDA projection vector
    point = np.array([x, y])
    return np.dot(point, w)

#########################################################
###################Helper Code###########################
#########################################################

X = np.empty((0, 3), dtype=float)  # Specify dtype=float to ensure numerical data

with open('data.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for sample in csvFile:
        sample = [float(i) for i in sample]  # Convert strings to float
        X = np.vstack((X, sample))

print(X)
print(X.shape)
# X Contains m samples each of format (x,y) and class label 0.0 or 1.0

opt = int(input("Input your option (1-5): "))

match opt:
    case 1:
        meanDiff = ComputeMeanDiff(X)
        print(meanDiff)
    case 2:
        SW = ComputeSW(X)
        print(SW)
    case 3:
        SB = ComputeSB(X)
        print(SB)
    case 4:
        w = GetLDAProjectionVector(X)
        print(w)
    case 5:
        x = int(input("Input x dimension of a 2-dimensional point: "))
        y = int(input("Input y dimension of a 2-dimensional point: "))
        w = GetLDAProjectionVector(X)
        print(project(x, y, w))