import sympy
import numpy as np

class ExtendedKalmanFilter:

    def __init__(self, function,  state, inputs, P, Q, h, R, initialStates):
        self.FUN = function
        self.X = state
        self.U = inputs
        self.size = len(self.X)
        self.h_fun = h
        self.symbs = [item for sublist in self.X.tolist() for item in sublist]
        self.X_symbs = [item for sublist in self.X.tolist() for item in sublist]
        self.symbs.extend([item for sublist in self.U.tolist() for item in sublist])

        self.fun = sympy.lambdify(self.symbs, self.FUN, 'numpy')
        self.h = sympy.lambdify(self.X_symbs, self.h_fun, 'numpy')
        self.F = self.FUN.jacobian(self.X)
        self.H_FUN = self.h_fun.jacobian(self.X)
        self.f = sympy.lambdify(self.symbs, self.F, 'numpy')
        self.H = sympy.lambdify(self.X_symbs, self.H_FUN, 'numpy')

        self.P = P
        self.Q = Q
        #self.H = H
        self.R = R
        self.X_values = initialStates

    def predict(self, controls):
        # print(self.X_values)
        self.inputs = []
        self.inputs.extend(self.X_values.T.reshape(-1,).tolist())
        self.inputs.extend(controls.T.reshape(-1,).tolist())
        #print(self.inputs)
        self.inputs = [item for sublist in self.inputs for item in sublist]
        #predict state readings
        self.X_values = self.fun(*self.inputs)
        self.F_values = self.f(*self.inputs)
        self.P = self.F_values*self.P*self.F_values.T + self.Q

    def update(self, measurements):
        X_inputs = []
        X_inputs.extend(self.X_values.T.reshape(-1,).tolist())
        X_inputs = [item for sublist in X_inputs for item in sublist]
        self.z = measurements
        self.y = self.z - self.h(*X_inputs)
        self.H_values = self.H(*X_inputs)
        #print(self.y)
        #print(self.H_values)
        #print(self.P)
        self.S = np.dot(self.H_values, self.P)*self.H_values.T + self.R #Variance of innovation
        #print(np.linalg.det(self.S))
        print(self.S)
        self.K = (self.P*self.H_values.T)*np.linalg.inv(self.S)
        #print(self.K)
        #print(np.dot(self.K,self.y))
        self.X_values = self.X_values + self.K*self.y
        #print(self.X_values)
        self.P = (np.eye(self.size) - self.K * self.H_values)*self.P
        #print(self.P)