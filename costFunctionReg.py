#costFunction with regularization
def costFunctionReg(theta, X, y, l):
    #initializing terms
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    
    #cost function
    J = -(1 / m) * (y.transpose().dot(log(h)) + (np.subtract(1, y)).transpose().dot(log(1-h))) + ( l / (2*m)) * np.sum(np.power((theta[1:, 0]), 2))

    #gradient descent
    delta = np.subtract(h, y)
    
    grad = (1/m) * ((X.transpose().dot(delta)) + ((1 / m) * theta))
    grad[0] = 0

    return J, grad.flatten()
