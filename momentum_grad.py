# Boston housing
import numpy as np

X=[0.5, 2.5]
Y=[0.2, 0.9]

def f(w,b,x):
    return 1/ (1 + np.exp(-(w * x + b)))
def loss(w,b):
    err = 0
    for (x,y) in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y)**2
    return err
def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1-fx) * x

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx - y) * fx * (1-fx)

def do_grad_descent():
    w,b,eta,max_epochs = 2,2,1,1000
    prev_v_w,prev_v_b,gamma=0,0,0.9
    for i in range(max_epochs):
        dw,db=0,0
        for (x,y) in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)

        w -= dw * eta
        b -= db * eta
        print(w,b)
do_grad_descent()