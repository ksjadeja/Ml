# 4 neurons in i/p layer
#  3 neurons in hidden layer
#  1 neurons in o/p layer

# xor function of 3 input (min 8 neurons in hidden layer)  (see on taking 5 neurons)
# 6 cases in training , 2 in testing
# error func - cross entropy

import numpy as np
# def loss_func(y_hat,y):
#     return 0.5 * ((y_hat - y)**2)
def cross_entropy(y_hat, y):
    if y == 1:
      return -np.log10(y_hat)
    else:
      return -np.log10(1 - y_hat)
def hidden_layer_activation_func(x):
    return (np.exp(x) - np.exp(-x))/ (np.exp(x) + np.exp(-x))
def outer_layer_activation_func(x):
    return 1 / (1 + np.exp(-x))

Xa=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]]
X = np.matrix(Xa)
# Y = np.matrix([0,1,1,0,1,0,0,1])
Y = [0,1,1,0,1,0]
input_neurons = 3
neurons_in_hidden_layer = 8 #2^3
neurons_in_input_layer = 3
neurons_in_output_layer = 1

wh=np.random.uniform(size=(neurons_in_input_layer,neurons_in_hidden_layer))
wout = np.random.uniform(size=(neurons_in_hidden_layer,neurons_in_output_layer))
b1 = np.random.uniform(size=(1,neurons_in_hidden_layer))
# b2 = np.random(1)
# def grad_outer_weights():
#     if(y==1):
#         return out_n21 * (out_n31 - 1)
out_n2=np.array(float)
out_n2.resize(neurons_in_hidden_layer)
net_n2=np.array(float)
net_n2.resize(neurons_in_hidden_layer)
out_n3 = 0
net_n3 = 0

eta = 0.01
# dwh = np.array(int)
# dwh.resize(neurons_in_input_layer*neurons_in_hidden_layer)
dwh=np.random.uniform(size=(neurons_in_input_layer,neurons_in_hidden_layer))

dwo = np.array(float)
dwo.resize(neurons_in_hidden_layer*neurons_in_output_layer)
ouut_n2=np.array(float)
ouut_n2.resize(neurons_in_hidden_layer)
neet_n2=np.array(float)
neet_n2.resize(neurons_in_hidden_layer)
ouut_n3 = 0
neet_n3 = 0

def calc_val(x,w_h,w_out,b1):
    for j in range(neurons_in_hidden_layer):
        neet_n2[j] = np.dot(x, w_h[:, j]) + b1[j]
        ouut_n2[j] = hidden_layer_activation_func(neet_n2[j])
        # net_n3 += wout[j][1] * out_n2
    neet_n3 = np.dot(ouut_n2, w_out) + b1[neurons_in_hidden_layer]
    ouut_n3 = outer_layer_activation_func(neet_n3[0])
    print('i/p => ',x,'  o/p=>',ouut_n3)
def do_back_propagation():
    bh = 0
    bout = 0
    b = np.array(float)
    b.resize(neurons_in_hidden_layer + neurons_in_output_layer)
    b=np.random.uniform(size=(neurons_in_hidden_layer+neurons_in_output_layer))
    # b=0
    max_epochs = 100000
    for i in range(max_epochs):
        for (m,n) in zip(X,Y):
            # print(m,"   ",n)
            for j in range(neurons_in_hidden_layer):
                # print(type(b[j]))
                net_n2[j] = np.dot(m,wh[:,j]) + b[j]
                out_n2[j] = hidden_layer_activation_func(net_n2[j])
                # net_n3 += wout[j][1] * out_n2
            net_n3 = np.dot(out_n2,wout) + b[neurons_in_hidden_layer]
            out_n3 = outer_layer_activation_func(net_n3[0])

            Loss = cross_entropy(out_n3,n)
            # print("i ",i,"   ",Loss)
            for i in range(neurons_in_output_layer*neurons_in_hidden_layer):
                if n==1:
                    dwo[i] = (out_n3 - 1) * out_n2[i]
                else:
                    dwo[i] = out_n3 * out_n2[i]
                wout[i][0] -= dwo[i]*eta;
            for i in range(neurons_in_hidden_layer):
                for j in range(3):
                    # print(m[:,j])
                    if n==1:
                        dwh[j][i] = (out_n3 - 1) * (1 - (out_n2[i] ** 2)) * m[:, j]
                    else:
                        dwh[j][i] = out_n3 * (1 - (out_n2[i]**2)) * m[:,j]
                    wh[j][i] -= dwh[j][i]*eta
                if n==1:
                    # b[i] = b[i] -  ((out_n3 - 1)*eta)
                    b[i] = b[i] - ((out_n3 - 1) * (1 - (out_n2[0] ** 2)))
                else:
                    # bout -= out_n3 * eta
                    b[i] -= out_n3 * (1 - (out_n2[0]**2))
            if n==1:
                b[i] = b[i] - ((out_n3 - 1) * eta)
            else:
                b[i] -= out_n3 * eta
            # print(wout)
        # print("i ", i, "   ", Loss)
    print(wh)
    print(wout)
    for u in X:
        calc_val(u,wh,wout,b)

do_back_propagation()
# [[ 0.96051019 -1.75038234  1.68275771 -1.43114416  1.53141909  1.53512718
#    0.87052505 -1.76882658]
#  [ 0.96800315 -1.648634    1.5856299  -1.90402196  1.6137693   1.35622668
#    0.88101224 -1.37410629]
#  [-2.51549657  0.69379532  2.07581552  0.66520428  2.04627243  1.95010635
#   -2.34467351  0.61546678]]
# [[4.39338667]
#  [2.64655602]
#  [3.29172082]
#  [2.97375157]
#  [2.88157479]
#  [2.00704831]
#  [3.5939804 ]
#  [2.7515412 ]]