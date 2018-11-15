"""
Prepare data for MNIST and FMNIST. The idea of this script is to read the databases and read them
in some "nice" format (since, for example, mnist is in ubyte and that is kinda hard to work with),
i.e. read them in the original format and save it in numpy.
"""

import numpy as np
import mnist_reader
from mnist import MNIST

#Use parser (First MNIST)
#The directory "./mnist_data" must contain MNIST data (in ubyte format).
mndata = MNIST('./mnist_data')

#Loads training set
images_train, labels_train = mndata.load_training()
#convert to numpy matrix and save
X_train_mnist = np.array(images_train)
np.save("./np_data/X_train_mnist.npy", X_train_mnist)
y_train_mnist = np.array(labels_train)
np.save("./np_data/y_train_mnist.npy", y_train_mnist)

#Load testing
images_test, labels_test = mndata.load_testing()
#convert to numpy matrix and save
X_test_mnist = np.array(images_test)
np.save("./np_data/X_test_mnist.npy", X_test_mnist)
y_test_mnist = np.array(labels_test)
np.save("./np_data/y_test_mnist.npy", y_test_mnist)

X_train_fmnist, y_train_fmnist = mnist_reader.load_mnist('./fmnist_data', kind='train')
X_test_fmnist, y_test_fmnist = mnist_reader.load_mnist('./fmnist_data', kind='t10k')
np.save("./np_data/X_train_fmnist.npy", X_train_fmnist)
np.save("./np_data/y_train_fmnist.npy", y_train_fmnist)
np.save("./np_data/X_test_fmnist.npy", X_test_fmnist)
np.save("./np_data/y_test_fmnist.npy", y_test_fmnist)
