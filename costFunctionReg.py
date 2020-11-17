#costFunction with regularization
def costFunctionReg(theta, X, y, l):
    #initializing terms
    m = X.shape[1]
    h = sigmoid(X.dot(theta))

    thetaR = theta[1:, 0]
    
    #cost function
    J = -(1 / m) * (y.transpose().dot(log(h)) + (np.subtract(1, y)).transpose().dot(log(1-h))) + (1 / (2*m)) * np.matmul(thetaR.T, thetaR)
    
    
    #gradient descent
    delta = np.subtract(h, y)
    sumdelta = np.sum(delta)
    
    grad = (1 / m) * (sumdelta * X.transpose()) + (l / m) * thetaR
    grad[0] = 0

    return J, grad.flatten()
