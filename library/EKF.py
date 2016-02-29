import sympy
import numpy as np
import scipy.optimize

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

class ExtendedKalmanFilterParameter:

    def __init__(self, function, parameters, states, P, initialParams, Re = None, Rk = None, forget=0.5, method="Forget", constrained=False, constraint=None):
        self.FUN = function
        self.W = parameters
        self.X = states
        self.size = len(self.W)
        self.symbs = [item for sublist in self.W.tolist() for item in sublist]
        self.W_symbs = [item for sublist in self.W.tolist() for item in sublist]
        self.symbs.extend([item for sublist in self.X.tolist() for item in sublist])
        #print(self.symbs)

        self.fun = sympy.lambdify(self.symbs, self.FUN, 'numpy')
        self.C_w = self.FUN.T.jacobian(self.W)
        self.c_w = sympy.lambdify(self.symbs, self.C_w, 'numpy')

        self.P = P
        if Rk is None:
            self.Rk = np.ones((len(self.W_symbs),len(self.W_symbs)))
        else:
            self.Rk = Rk

        if Re is None:
            self.Re = np.ones((len(self.W_symbs),len(self.W_symbs)))
        else:
            self.Re = Re

        self.W_values = initialParams
        self.initial_estimates = initialParams
        self.forget = forget
        
        self.constrained = constrained
        if constraint is None:
            self.constraint = np.ones(self.size)
        else:
            self.constraint = constraint

    def Predict(self, inputs):
        self.inputs = []
        inputs = inputs.T.reshape(-1,).tolist()
        inputs = [item for sublist in inputs for item in sublist]
        self.inputs.extend(self.W_values.T.reshape(-1,).tolist())
        try:
            self.inputs = [item for sublist in self.inputs for item in sublist]
        except:
            self.inputs = self.inputs
        self.inputs.extend(inputs)
        #print(self.inputs)
        self.P = self.P/self.forget
        self.prediction = self.fun(*self.inputs)
        self.Cw_values = self.c_w(*self.inputs)

    def Update(self, measurements):
        self.z = measurements
        self.y = self.z - self.prediction
        #print(self.P)
        self.S = self.Cw_values*self.P*(self.Cw_values.T) + self.Re
        #print(self.Cw_values*self.P*(self.Cw_values.T))
        if self.S.shape[0] == 1:
            self.K = self.P*self.Cw_values.T/self.S
        else:
            #print(self.Cw_values.T)
            self.K = self.P*(self.Cw_values.T)*np.linalg.inv(self.S)
        #print(self.K)
        self.P *= (np.eye(self.size) - self.K * self.Cw_values)
        self.W_values += self.K*self.y
        
        if self.constrained is False:
            return
        
        if (self.W_values < 0).any() or (self.W_values > self.constraint).any():
            res = scipy.optimize.fmin_cobyla(self.target, self.initial_estimates, [self.constr, self.constr2])
            #print(res)
            self.W_values = res
            #print(self.W_values)
            #raise Exception('Cannot Optimize')
        else:
            #print(self.W_values)
            return

    def target(self, x):
        result = (x-self.W_values).T*np.linalg.inv(self.P)*(x-self.W_values)
        return result[0,0]

    def constr(self, x):
        return x

    def constr2(self, x):
        return self.constraint-x