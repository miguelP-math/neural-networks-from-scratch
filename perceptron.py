import math

###########################  ACTIVATION FUNCTIONS AND DERIVATIONS #############################

#Identity(If we want to try with simplifications)
def f0(x):
    return(x)

def df0(x):
    return(1)


#Sigmoide
def f1(x):
    return(1/(1+math.exp(-x)))

def df1(x):
    a=f1(x)
    return(a*(1-a))

#tanh
def f2(x):
    return((math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)))

def df2(x):
    return(1-f2(x)**2)

#ReLu
def f3(x):
    if x<0:
        return(0)
    else:
        return(x)

def df3(x):
    if x<0:
        return(0)
    else:
        return(1)



#We choose here the f for the process
f = f1
df = df1

###########################  ERROR FUNCTION AND GRADIENT ERROR FUNCTION #############################
def error(w,x,t):
    """Calculate CMV error with w bias and weights, x input variables and t target variable."""

    a = w[0] + w[1] * x[0] + w[2] * x[1]
    y = f(a)
    return((y-t)**2/2)

def derror(w,x,t):
    """Calculate gradient of CMV error w bias and weights, x input variables and t target variable."""

    a = w[0] + w[1] * x[0] + w[2] * x[1]
    g = f(a) - t
    d0 = g * df(a)
    d1 = d0 * x[0]
    d2 = d0 * x[1]
    return(d0,d1,d2)

###########################  TRAINING FUNCTIONS #############################
def singletrain(w0,lx,de,eta):
    """Trains with a list of data x, and optimise the initial witghts w0. The parameters are:
       -w0  -> Initial weights.
       -lx  -> A list with the training data, each object is a tuple of the form (x,t).
       -e  -> The error function.
       -de -> Derivative of the error function.
       -eta -> The trainning rate."""

    w=w0
    for x,t in lx:
        grads = de(w,x,t)
        w = [wi - eta * gi for wi, gi in zip(w, grads)]
    return(w)

def training(w0,lx,de,eta,epocs):

    if epocs <= 0:
        return(w0)
    else:

        return(training(singletrain(w0,lx,de,eta),lx,de,eta,epocs - 1))

###########################  EXAMPLE DATA #############################

#Simulation with OR
if __name__ == "__main__":
    w0 = [0, 0, 0]
    trainingdata = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
    eta = 0.3
    epocs = 800

    final_weights_or = training(w0, trainingdata, derror, eta, epocs)
    print("Pesos finales tras el entrenamiento:", final_weights_or)

#Simulation with AND
if __name__ == "__main__":
    w0 = [0, 0, 0]
    trainingdata = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    eta = 0.3
    epocs = 800

    final_weights_and = training(w0, trainingdata, derror, eta, epocs)
    print("Pesos finales tras el entrenamiento:", final_weights_and)