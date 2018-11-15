# Classic Machine Learning Algorithms on MNIST and FMNIST

Usage:

1) Download FMNIST and MNIST. FMNIST is downloadable in https://github.com/zalandoresearch/fashion-mnist.
MNIST is Downloadable in http://yann.lecun.com/exdb/mnist/.

2) Put the four archives of each dataset in folders (inside this directory) called
'fmnist_data' and 'mnist_data' respectively.

3)run:

   python prepare_data.py

4)run the script "classic_algorithms.py" with the appropiate database. For example:

   python classic_algorithms.py './np_data' X_train_fmnist.npy y_train_fmnist.npy X_test_fmnist.npy y_test_fmnist.npy

5)the trained models/confusion matrices/scores are in the diagnostics folder