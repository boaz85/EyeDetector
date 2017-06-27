import numpy as np
import matplotlib.pyplot as plt
import sys

from Classifiers.Base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):

    def __init__(self, *args, **kwargs):
        super(LogisticRegressionClassifier, self).__init__(*args, **kwargs)
        self._train_data = None
        self._train_data_y = None
        self._cv_data = None
        self._cv_data_y = None

    def _sigmoid(self, x):
        return float(1) / (1 + np.exp(-x))

    def _cost(self, theta, lambda_, X, y):
        m = X.shape[0]
        h_x = self._sigmoid(np.dot(X, theta))
        reg = np.multiply((lambda_ / (2 * m)), np.dot(theta.T, theta))

        return (1 / float(m)) * (np.dot(-y.T, np.log(h_x + 10e-10)) - np.dot((1 - y).T, np.log((1 - h_x) + 10e-10))) + reg

    def test(self, theta=None):

        if not self._is_train_data_loaded():
            raise Exception('No training data loaded')

        if theta is None is self.theta:
            raise Exception('Theta is missing')

        if theta is None:
            theta = self.theta

        predictions = []

        error_rate = lambda p, y: float(sum(((np.array(p) > 0.5).astype(int) != y))) / len(y)

        if self.plot_error_rate:
            error_rates = []
            plot_step = (len(self._cv_data) / 200) if len(self._cv_data) > 200 else 1

        for i, row in enumerate(self._cv_data):

            predictions.append(self.classify(row[1:], theta))

            if self.plot_error_rate:
                error_rates.append(error_rate(predictions, self._cv_data_y[:i + 1, 0]))

                if self.plot_interactively:
                    if i % plot_step == 0:
                        self.error_plot.plot(range(i + 1), error_rates, 'g-', linewidth=0.5)
                        plt.pause(0.001)

        if self.plot_error_rate and not self.plot_interactively:
            self.error_plot.plot(range(i + 1), error_rates, 'g-', linewidth=0.5)
            plt.show()

        positive_predictions = (np.array(predictions) > 0.5).astype(int)
        positive_actual = (self._cv_data_y[:, 0]).astype(int)

        precision = sum(positive_predictions & positive_actual) / float(sum(positive_predictions))
        recall = sum(positive_predictions & positive_actual) / float(sum(positive_actual))
        error = error_rate(predictions, self._cv_data_y[:, 0])

        if self.plot_error_rate:
            print 'Precision = ' + str(precision)
            print 'Recall = ' + str(recall)
            print 'FScore = ' + str(2 * ((precision * recall) / (precision + recall)))
            print 'Error = ' + str(error)

        return error

    def load_train_data(self, train_data_path, cv_size):

        T = np.load(train_data_path)
        np.random.shuffle(T)

        cv_start_index = int(T.shape[0] * (1 - cv_size))
        self.m, self.n = T.shape

        self._train_data = np.ones([cv_start_index, self.n])
        self._cv_data = np.ones([self.m - cv_start_index, self.n])
        self._train_data[:, 1:], self._train_data_y = T[:cv_start_index, :-1], T[:cv_start_index, -1].reshape(-1, 1)
        self._cv_data[:, 1:], self._cv_data_y = T[cv_start_index:, :-1], T[cv_start_index:, -1].reshape(-1, 1)

    def _is_train_data_loaded(self):
        return self._train_data is not None

    def _next_train_iteration(self, alpha, lambda_, iterations):

        m, n = self._train_data.shape
        theta = np.random.uniform(-1, 1, [n, 1])

        for i in range(iterations):

            h_x = self._sigmoid(np.dot(self._train_data, theta))
            reg = np.multiply(lambda_ / m, theta)
            theta -= (float(alpha) / m) * np.dot(self._train_data.T, (h_x - self._train_data_y)) + reg
            yield theta

    def _get_theta_square_dimensions(self):
        sqrt = np.ceil(np.sqrt(self.n - 1))
        while (self.n - 1) % sqrt != 0:
            sqrt += 1

        return int(sqrt), int((self.n - 1) / sqrt)

    def optimize_parameters(self):

        curved_space = lambda m, n, s: np.power(np.linspace(np.power(m, 1.0/4), np.power(n, 1.0/4), s), 4)
        alpha_range = curved_space(10e-10, 10, 100)
        lambda_range = curved_space(0.01, 10, 50)

        min_cost = np.inf
        min_params = (None, None)

        i = 0
        print ''

        for alpha in alpha_range:
            for lambda_ in lambda_range:
                sys.stdout.write('\rIteration: ' + str(i))
                i += 1
                theta = self.train(alpha, lambda_, 15000)
                cost = self.test(theta)

                if cost < min_cost:
                    print 'New min cost: ' + str(cost)
                    print 'Alpha: ' + str(alpha) + ', Lambda: ' + str(lambda_)
                    min_cost = cost
                    min_params = alpha, lambda_

        return min_params

    def train(self, alpha, lambda_, iterations):

        is_checkpoint = lambda i: i % ((iterations / 100) if iterations > 400 else 1) == 0
        text_args, text_kwargs = (0.05, 0.95, ''), {'fontsize': 14, 'verticalalignment': 'top',
                                                    'bbox': dict(boxstyle='round', facecolor='wheat', alpha=0.5)}

        if self.plot_cost:
            cost_history = []
            cost_textbox_text = self.cost_plot.text(*text_args, transform=self.cost_plot.transAxes, **text_kwargs)
            plot_args, plot_kwargs = ('g-',), {'linewidth': 0.5}

        if self.visualize_theta:
            theta_img = None
            theta_vis_shape = self._get_theta_square_dimensions()
            theta_textbox_text = self.theta_visualization.text(*text_args, transform=self.theta_visualization.transAxes,
                                                               **text_kwargs)

        for i, theta in enumerate(self._next_train_iteration(alpha, lambda_, iterations)):

            checkpoint = is_checkpoint(i)

            if checkpoint and self.plot_interactively:
                sys.stdout.write('\rIteration: ' + str(i))

            if self.plot_cost:
                cost_history.append(self._cost(theta, lambda_, self._train_data, self._train_data_y)[0][0])

                if self.plot_interactively and checkpoint:
                    self.cost_plot.plot(range(i + 1), cost_history, *plot_args, **plot_kwargs)
                    cost_textbox_text.set_text('Iteration = {x}\nCost = {y}'.format(x=i, y=cost_history[-1]))
                    plt.pause(0.0001)

            if self.visualize_theta and self.plot_interactively and checkpoint:
                if theta_img is None:
                    theta_img = self.theta_visualization.imshow(theta[1:].reshape(theta_vis_shape))
                else:
                    theta_textbox_text.set_text('Iteration: ' + str(i))
                    theta_img.set_data(theta[1:].reshape(theta_vis_shape))
                    theta_img.set_clim(theta[1:].min(), theta[1:].max())
                plt.pause(.0001)

        if not self.plot_interactively:

            if self.plot_cost:
                self.cost_plot.plot(range(iterations), cost_history, *plot_args, **plot_kwargs)

            if self.visualize_theta:
                self.theta_visualization.imshow(theta[1:].reshape(theta_vis_shape))

            plt.show()

        self.theta = theta
        # print ''
        return theta


    def classify(self, x, theta=None):

        if theta is None is self.theta:
            raise Exception('Theta is missing')

        if theta is None:
            theta = self.theta

        return self._sigmoid(np.dot(theta.T, np.append([1], x.reshape(-1, 1)))[0])