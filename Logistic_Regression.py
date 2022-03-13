#pip install idx2numpy

import os 
import idx2numpy
import matplotlib.pyplot as plt

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
    TEST_X.append(train_x[idx].reshape(-1, )/255.0)
    if label == 0:
        TEST_Y.append(1)
    else:
        TEST_Y.append(0)

print(TRAIN_X[1])
print(len(TRAIN_X[1]))