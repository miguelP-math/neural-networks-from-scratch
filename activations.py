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