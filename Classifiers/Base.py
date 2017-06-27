import matplotlib.pyplot as plt

class BaseClassifier(object):

    def __init__(self, plot_interactively=False, plot_cost=False, plot_error_rate=False, visualize_theta=False):

        self.plot_interactively = plot_interactively
        self.plot_cost = plot_cost
        self.plot_error_rate = plot_error_rate
        self.visualize_theta = visualize_theta

        if plot_interactively:
            plt.ion()

        if self.plot_cost:

            self.cost_plot = plt.figure('Cost').add_subplot(111)
            self.cost_plot.set_title('Cost function')
            self.cost_plot.set_ylabel('Cost')
            self.cost_plot.set_xlabel('Iterations')

        if self.plot_error_rate:
            self.error_plot = plt.figure('Error').add_subplot(111)
            self.error_plot.set_title('Error rate')
            self.error_plot.set_ylabel('Error rate')
            self.error_plot.set_xlabel('Examples')

        if self.visualize_theta:
            self.theta_visualization = plt.figure('Theta').add_subplot(111)

    def train(self, X, y):
        pass

    def classify(self, x):
        pass

