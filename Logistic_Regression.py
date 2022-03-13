#pip install idx2numpy

import os 
import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

if not os.path.isfile("./MNIST.tar.gz"):
    print("downloading MNIST zip data")
    os.system("wget www.di.ens.fr/~lelarge/MNIST.tar.gz")
if not os.path.exists("MNIST"):
    print("Unzipping files")
    os.system("tar -zxvf MNIST.tar.gz")
print("Done")

train_x = idx2numpy.convert_from_file("MNIST/raw/train-images-idx3-ubyte")
train_y = idx2numpy.convert_from_file("MNIST/raw/train-labels-idx1-ubyte")

test_x = idx2numpy.convert_from_file("MNIST/raw/t10k-images-idx3-ubyte")
test_y = idx2numpy.convert_from_file("MNIST/raw/t10k-labels-idx1-ubyte")

# preprocess the data

TRAIN_X, TRAIN_Y, TEST_X, TEST_Y = [], [], [], []
for idx, label in enumerate(train_y):
    TRAIN_X.append(train_x[idx].reshape(-1, )/255.0)
    if label == 0:
        TRAIN_Y.append(1)
    else:
        TRAIN_Y.append(0)
for idx, label in enumerate(test_y):
    TEST_X.append(test_x[idx].reshape(-1, )/255.0)
    if label == 0:
        TEST_Y.append(1)
    else:
        TEST_Y.append(0)

TRAIN_X = np.array(TRAIN_X)
TRAIN_Y = np.array(TRAIN_Y)
TEST_X = np.array(TEST_X)
TEST_Y = np.array(TEST_Y)



print(TRAIN_X.shape)
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y, y_hat):
    loss = np.mean(y*(np.log(y_hat)) + (1-y)*np.log(1-y_hat))
    return loss

def gradients(X, y, y_hat):
    
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).
    
    # m-> number of training examples.
    m = X.shape[0]
    
    # Gradient of loss w.r.t weights.
    dw = np.dot(X.T, (y-y_hat)) #(1/m)*
    
    return dw


def train(X, y, bs, epochs, lr):
    
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias to zeros.
    w = np.zeros((n,1))
    b = 0
    
    # Reshaping y.
    y = y.reshape(m,1)
    
    # Normalizing the inputs.
    
    # Empty list to store losses.
    losses = []
    train_acc = []
    test_acc = []
    # Training loop.
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, w) )
            
            # Calculate gradients 
            dw = gradients(xb, yb, y_hat)
            w += lr*dw

        l = loss(y, sigmoid(np.dot(X, w) + b))
        tr_acc = accuracy_score(TRAIN_Y, predict(TRAIN_X, w))
        te_acc = accuracy_score(TEST_Y, predict(TEST_X, w))
        losses.append(l)
        train_acc.append(tr_acc)
        test_acc.append(te_acc)
        print("training accuracy ", tr_acc)
        print("testing accuracy ", te_acc)
        
    # returning weights, bias and losses(List).
    return w, b, losses


def predict(X, w):
    
    # X --> Input.
    
    # Normalizing the inputs.
    #x = normalize(X)
    
    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w))
    
    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    
    return np.array(pred_class)

w, b, l = train(TRAIN_X, TRAIN_Y, bs=100, epochs=100, lr=0.1/len(TRAIN_X))

#print(TRAIN_X[1])
#print(TRAIN_X.shape)