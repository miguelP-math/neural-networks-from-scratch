import numpy as np
import random
import math

import activations as act

#We choose here the activation function for the process
f = act.f3
df = act.df3

###########################  ERROR FUNCTION AND GRADIENT ERROR FUNCTION #############################
def error(w,x,t):
    """Calculate CMV error with w bias and weights, x input variables and t target variable."""

    a1 = w[0] + w[1] * x[0] + w[2] * x[1]
    a2 = w[3] + w[4] * x[0] + w[5] * x[1]
    z1 = f(a1)
    z2 = f(a2)
    a = w[6] + w[7] * z1 + w[8] * z2
    y = f(a)
    return((y-t)**2/2)

def derror(w,x,t):
    """Calculate gradient of CMV error w bias and weights, x input variables and t target variable."""

    a1 = w[0] + w[1] * x[0] + w[2] * x[1]
    a2 = w[3] + w[4] * x[0] + w[5] * x[1]
    z1 = f(a1)
    z2 = f(a2)
    a = w[6] + w[7] * z1 + w[8] * z2
    g = f(a) - t
    d6 = g * df(a)
    d7 = d6 * z1
    d8 = d6 * z2
    d0 = d6 * w[7] * df(a1)
    d1 = d0 * x[0]
    d2 = d0 * x[1]
    d3 = d6 * w[8] * df(a2)
    d4 = d3 * x[0]
    d5 = d3 * x[1]

    sol = np.array([d0,d1,d2,d3,d4,d5,d6,d7,d8])
    return(sol)



###########################  TRAINING FUNCTIONS #############################
def singletrain(w0,lx,de,eta):
    """Trains with a list of data x, and optimise the initial witghts w0. The parameters are:
       -w0  -> Initial weights.
       -lx  -> Array with the trainning data, a matrix with elements of the form [x,t] with x the input and t the expected output.
       -e  -> The error function.
       -de -> Derivative of the error function.
       -eta -> The trainning rate."""

    w = w0
    for x,t in lx:
        w = w - eta * de(w,x,t)
    return(w)

def training(w0,lx,de,eta,epocs):

    w = np.array(w0)
    for i in range(epocs):
        w = singletrain(w,lx,de,eta)

    return(w)

###########################  EXAMPLE DATA #############################

#Simulation with OR
if __name__ == "__main__":
    w0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    trainingdata = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    eta = 0.3
    epocs = 800

    final_weights_or = training(w0, trainingdata, derror, eta, epocs)
    print("Pesos finales tras el entrenamiento:", final_weights_or)

###########################  EXAMPLE DATA #############################

#Simulation with XOR
if __name__ == "__main__":
    w0 = [random.uniform(-1.0, 1.0) for _ in range(9)]
    trainingdata = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    eta = 0.15
    epocs = 50000

    final_weights_xor = training(w0, trainingdata, derror, eta, epocs)
    print("Pesos finales tras el entrenamiento:", final_weights_xor)
