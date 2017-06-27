import argparse

import numpy as np

from Classifiers.LogisticRegression import LogisticRegressionClassifier

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--train-data",
                    help='Source images directory', required=True)

parser.add_argument("-t", "--iterations", type=int,
                    help='Train iterations', default=1000)

parser.add_argument("-i", "--plot-interactively", action='store_true', default=False,
                    help='Show and update plot during calculation')

parser.add_argument("-a", "--alpha", type=float,
                    help='Train alpha', default=10e-7)

parser.add_argument("-l", "--lambdaa", type=float,
                    help='Lambda value', default=3)

parser.add_argument("-e", "--plot-error-rate", action='store_true', default=False,
                    help='Plot error rate by examples count')

parser.add_argument("-c", "--plot-cost", action='store_true', default=False,
                    help='Plot cost by iterations')

parser.add_argument("-s", "--visualize-theta", action='store_true', default=False,
                    help='Show theta as an image')

parser.add_argument("-v", "--cv-size", type=float,
                    help='Cross validation samples part of the whole training set. E.g 0.5 = 50%', default=0.3)

def load_train_data(path, cv_percent):
    T = np.load(path)
    np.random.shuffle(T)
    X, y = T[:,:-1], T[:,-1].reshape(-1, 1)

    cv_start_index = X.shape[0] * (1 - cv_percent)

    train, train_y = X[:cv_start_index, :-1], X[:cv_start_index, -1].reshape(-1, 1)
    cv, cv_y = X[cv_start_index:, :-1], X[cv_start_index:, -1].reshape(-1, 1)

    return train, train_y, cv, cv_y

if __name__ == '__main__':

    args = parser.parse_args()

    classifier = LogisticRegressionClassifier(plot_interactively=args.plot_interactively, plot_cost=args.plot_cost,
                                              plot_error_rate=args.plot_error_rate, visualize_theta=args.visualize_theta)

    classifier.load_train_data(args.train_data, args.cv_size)
    #theta = classifier.train(args.alpha, args.lambdaa, args.iterations)#np.load('optimized_theta.npy')#
    optimal_params = classifier.optimize_parameters()
    print optimal_params
    #error_rate = classifier.test(theta)
    #np.save('optimized_theta', theta)
