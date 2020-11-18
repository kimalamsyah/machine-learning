from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace


def sigmoid(z):
    g = 1 / (1 + (e ** (-z))); 
    return g
    
    from numpy import log

#costFunction with regularization
def costFunctionReg(theta, X, y, l):

    #initializing terms
    m = len(X)
    h = sigmoid(np.matmul(X, theta))
    
    #cost function
    J = -(1 / m) * (y.transpose().dot(np.log(h)) + (np.subtract(1, y)).transpose().dot(np.log(1-h))) + ( l / (2*m)) * np.sum(np.power((theta[1:]), 2))

    return(J[0])

def gradDescent(theta, X, y, l):
    m = len(X)
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m) * X.transpose().dot(h - y) + (np.sum(((1 / m) * theta)))

    return(grad.flatten())
    
#Set regularization parameter lambda to 1
l = 1

#initializing theta
initial_theta = np.zeros((X_train.shape[1], 1))

#adding intercept terms of ones to X and zeros to theta
theta_zeros = np.zeros((1 ,1))
initial_theta = np.vstack((theta_zeros, initial_theta))
    
intercept = np.ones((X_train.shape[0], 1))
X_train = np.hstack((intercept, X_train))

#finding optimum parameters with fmin_bfgs
import scipy.optimize

scipy.optimize.fmin_bfgs(costFunctionReg, x0 = initial_theta, args=(X_train, y_train, l), maxiter=400)
