def costFunctionReg(theta, X, y):
    #initializing terms
    m = X.shape[1]
    h = sigmoid(X.dot(theta))

    thetaR = theta[1:, 0]
    
    #cost function
    J = -(1 / m) * (y.transpose().dot(log(h)) + (np.subtract(1, y)).transpose().dot(log(1-h))) + (1 / (2*m)) * np.matmul(thetaR.T, thetaR)
    
    return J
    
