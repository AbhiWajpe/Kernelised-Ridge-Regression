import numpy as np

# Importing the training dataset
trng = np.loadtxt("C:\\Users\\wajpe\\OneDrive\\Desktop\\Lectures\\AESRP 597\\HW SET\\HW1\\steel_composition_train.csv", delimiter=',',
                encoding='utf8')

# Extracting the Inputs in a Matrix
X = trng[:,0:8]
# Normalizing the input matrix
X_max = np.max(X, axis=0, keepdims=True)
X_min = np.min(X, axis=0, keepdims=True)
X = (X - X_min)/(X_max - X_min)

N = len(X)
# Extracting the Output in a Matrix
Y = trng[:,8]

# Function for calculating the errors for different parts 
def error_from_kernels(m):
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if m == 5:
                K[i][j] = np.exp(-((X[i]-X[j]).T @ (X[i]-X[j]))/2) # Gaussian kernel
            else:
                K[i][j] =   (X[i].T @ X[j] + 1)**m # Polynomial Kernel for different parts 
    a = np.linalg.inv(K + 1*np.eye(len(K))) @ Y
    y_pred = K.T @ a
    error = np.sqrt(np.mean((y_pred-Y)**2))
    return error


# Calculating the error 
m = list(range(2, 6))
RMSE = np.zeros(4)
for i, j in enumerate(m):
    RMSE[i] = round(error_from_kernels(j),5)

print('The errors for Part a,b,c and d are {0},{1},{2} and {3} respectively.'.format(RMSE[0],RMSE[1] ,RMSE[2], RMSE[3]))