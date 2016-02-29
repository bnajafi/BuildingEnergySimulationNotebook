import sympy
import numpy as np
import scipy.optimize
"""
This is an implementation of Dual Extended Kalman Filter used for combined state and parameter estimation
A tolerance is defined so that if the residual is small, parameter won't be updated
This allows for faster computation as well as more streamlined EKF tweaking
"""

class DualExtendedKalmanFilter:

    def __init__(self, function, states, parameters, inputs, measure,
                 state_values=None, parameter_values=None,
                 forget=0.2, P_x=None, R_v=None, P_w=None,
                 R_n=None, R_e=None, R_p=None,
                 tolerance = None, constraint=None, constrained=True):
        self.FUN = function
        self.X = states
        self.W = parameters
        self.U = inputs
        self.C = measure
        x_symbols = [item for sublist in self.X.tolist() for item in sublist]
        w_symbols = [item for sublist in self.W.tolist() for item in sublist]
        u_symbols = [item for sublist in self.U.tolist() for item in sublist]
        symbols = [item for sublist in self.X.tolist() for item in sublist]
        symbols.extend(w_symbols)
        symbols.extend(u_symbols)

        self.fun = sympy.lambdify(symbols, self.FUN, 'numpy')
        self.A = self.FUN.jacobian(self.X)
        self.a = sympy.lambdify(list(set(symbols)-set(x_symbols)), self.A, 'numpy')
        self.C_w = self.C*self.FUN.T.jacobian(self.W)
        self.c_w = sympy.lambdify(list(set(symbols)-set(w_symbols)), self.C_w, 'numpy')

        if P_x is None:
            self.Px = np.eye(len(x_symbols))
        else:
            self.Px = P_x

        self.lamb = forget

        if state_values is None:
            self.X_values = np.ones((len(x_symbols), 1)) * 0.0
        else:
            self.X_values = state_values
        if parameter_values is None:
            self.W_values = np.ones((len(w_symbols), 1))
        else:
            self.W_values = parameter_values
        self.W_initial = self.W_values

        if R_v is None:
            self.Rv = np.ones((len(x_symbols),len(x_symbols)))
        else:
            self.Rv = R_v

        if P_w is None:
            self.Pw = np.ones((len(w_symbols),len(w_symbols)))*0.01
        else:
            self.Pw = P_w

        if R_n is None:
            self.Rn = np.eye(len(x_symbols))*0.01
        else:
            self.Rn = R_n

        if R_e is None:
            self.Re = np.eye(len(w_symbols))*0.01
        else:
            self.Re = R_e
           
        if R_p is None:
            self.Rp = np.eye(len(w_symbols))*0.01
        else:
            self.Rp = R_p

        if tolerance is None:
            self.phi = np.ones((len(x_symbols),1))*0.1
        else:
            self.phi = tolerance
        
        self.constrained = constrained
        if constraint is None:
            self.constr = np.ones((len(w_symbols), 1))
        else:
            self.constr = constraint


    def Predict(self, inputs):
        u_values = [item for sublist in inputs.T.reshape(-1,).tolist() for item in sublist]
        x_values = [item for sublist in self.X_values.tolist() for item in sublist]
        
        
        #Parameter prediction
        self.W_values = self.W_values
        #self.Pw = self.Pw/self.lamb
        self.Pw = self.Pw + self.Rp
        #print(self.Pw)
        
        w_values = [item for sublist in self.W_values.tolist() for item in sublist]

        #State prediction
        inputs_list = [item for sublist in self.X_values.tolist() for item in sublist]
        inputs_list.extend(w_values)
        inputs_list.extend(u_values)
        self.X_values = self.fun(*inputs_list)
        A_values = self.a(*(w_values+u_values))
        #print(A_values)
        self.Px = A_values * self.Px * A_values.T + self.Rv
        
        u_values = [item for sublist in inputs.T.reshape(-1,).tolist() for item in sublist]
        x_values = [item for sublist in self.X_values.tolist() for item in sublist]
        self.Cw_values = self.c_w(*(x_values + u_values))
        #print(self.Cw_values)

    def Update(self, measurements):
        z = measurements - self.C*self.X_values
        #print(self.C)
        Sx = self.C*self.Px*self.C.T + self.Rn
        #print(np.dot(self.C, self.Px))
        if self.X.shape[0] == 1:
            Kx = self.Px*self.C.T/Sx
        else:
            Kx = self.Px*self.C.T*np.linalg.inv(Sx)
        #print(Kx)
        self.X_values = self.X_values + Kx * z
        self.Px = (np.eye(self.Px.shape[0]) - Kx*self.C)*self.Px
        #print(self.X_values)

        #if np.greater(z, self.phi).any():
        if True:
            Sw = np.dot(self.Cw_values, self.Pw) * self.Cw_values.T + self.Re
            #print(Sw)
            #print(np.linalg.det(Sw))
            if self.W.shape[0] == 1:
                self.Kw = self.Pw*self.Cw_values.T/Sw
            else:
                self.Kw = np.dot(self.Pw, self.Cw_values.T)*np.linalg.inv(Sw)
                #print(Kw)
            #print(self.Pw*self.Cw_values)
            #print(Kw)
            self.W_values = self.W_values + self.Kw*z
            #print(Kw*z)
            self.Pw = (np.eye(self.Pw.shape[0]) - self.Kw*self.Cw_values)*self.Pw
            #print(np.eye(self.Pw.shape[0]) - Kw*self.Cw_values)
            
        if (self.W_values < 0).any() or (self.W_values > self.constr).any() and self.constrained is True:
            res = scipy.optimize.fmin_cobyla(self.targetMin, self.W_initial, [self.constrLow, self.constrHigh])
            #print(res)
            self.W_values = res
            #print(self.W_values)
            #raise Exception('Cannot Optimize')
        else:
            #print(self.W_values)
            return

    def targetMin(self, w):
        result = (w-self.W_values).T*np.linalg.inv(self.Pw)*(w-self.W_values)
        return result[0,0]

    def constrLow(self, w):
        return w

    def constrHigh(self, w):
        return self.constr - w







if __name__ == "__main__":
    x1, x2, x3, u1, w1, w2, w3 = sympy.symbols('x1 x2 x3 u1 w1 w2 w3')
    states = sympy.Matrix([x1, x2, x3])
    parameters = sympy.Matrix([w1, w2, w3])
    inputs = sympy.Matrix([u1])
    fun = sympy.Matrix([[x1 + x2*w1],
                        [x2*w2 + u1*w3],
                        [u1]])
    mea = np.matrix([[1,0,0], [0,1,0],[0,0,0]])
    parameter_values = np.matrix([[1.0], [0.9], [1.8]])
    dekf = DualExtendedKalmanFilter(fun, states, parameters, inputs, mea)

    print(dekf.C_w)

    """
    speed = 1
    accleration = 1

    dekf.Predict(np.matrix([[accleration]]))
    print(dekf.X_values)

    distance = 1.5
    dekf.Update(np.matrix([[distance], [speed], [0]]))
    print(dekf.X_values)
    print(dekf.W_values)
    """


    iterations = 999
    distance = [0]
    speed = [0]
    np.random.seed(1234)
    prediction = [0]
    prediction_speed = [0]
    acceleration = np.random.normal(1, 1, iterations)
    W1 = 1.2
    W2 = 0.95
    W3 = 2.0
    for i in range(0, iterations):
        v = speed[i]*W2 + acceleration[i]*W3
        d = distance[i] + W1 * v
        distance.append(d)
        speed.append(v)


        dekf.Predict(np.matrix([[acceleration[i]]]))
        prediction.append(dekf.X_values[0,0])
        prediction_speed.append(dekf.X_values[1,0])

        dekf.Update(np.matrix([[d], [v], [0]]))


    print(dekf.W_values)
    print(dekf.Pw)
    print(dekf.X_values)


    import matplotlib.pyplot as plt
    plt.plot(distance,'r')
    plt.plot(prediction,'b')
    plt.show()

    plt.plot(speed, 'r')
    plt.plot(prediction_speed, 'x')
    plt.show()