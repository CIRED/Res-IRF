from numpy import exp


def logistic(x, a=1, r=1, K=1):
    return K / (1 + a * exp(- r * x))

