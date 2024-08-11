import argparse
import os, csv
import sys
from enum import Enum, StrEnum

import numpy as np

FILE_NAME_SINGLE_INPUT = '/training_data/housing_prices.csv'
FILE_NAME_MULTIPLE_INPUTS = '/training_data/housing_prices_multiple_inputs.csv'


# move to utils GitHub repo?
def get_csv_file_contents(file_path, include_header=False):
    """
    Assumes file can fit in memory

    :returns: 2-D list with each index containing a single row of the file
    """
    rows = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for i, row in enumerate(csv_reader):
            if i == 0 and not include_header: continue
            rows.append(row)

    return rows


class RegressionType(StrEnum):
    SINGLE_INPUT = 'single_input'
    MULTIPLE_INPUT = 'multiple_input'


# Hacky way to create an interface that isn't enforced.
# TODO switch to a 'formal' instance and introduce a Meta class 
# https://realpython.com/python-interface/
class LinearRegressionInterface:
    def load_training_data(self, training_examples):
        pass

    def gradient_descent(self, x, y):
        pass

    def compute_cost(self, x, y, w, b):
        pass


class SingleLinearRegression(LinearRegressionInterface):
    def load_training_data(self, training_examples):
        # Load in our training data
        # by convention m = number of training examples in our data set
        # x = inputs or features
        # y = outputs or targets
        m = len(training_examples)
        x = np.zeros(m)
        y = np.zeros(m)

        for i, example in enumerate(training_examples):
            x[i] = example[0]
            y[i] = example[-1]

        print(f'x: {x}')
        print(f'y: {y}')
        return x, y

    def compute_cost(self, x, y, w, b):
        """
        Function to compute the cost of our linear regression function. "Cost" is 
        a measure of how close our models prediction line is to the target line. 

        :param x: list of features (inputs)
        :param y: list of targets (outputs)
        :param w: slope of linear regression function
        :param b: y-intercept of linear regression function
        :returns: the cost of our model with fixed w and b params. w and b are both scalars
        """
        assert x.shape[0] == y.shape[0], ('feature and target lists must be the '
                                          'same length')

        num_examples = x.shape[0]
        cost = 0

        for i in range(num_examples):
            f_wb = w * x[i] + b  # predicted value
            cost += (f_wb - y[i]) ** 2

        cost /= 2 * num_examples
        return cost

    def compute_gradient(self, x, y, w, b):
        """
        Function to compute derivatives of the cost function w.r.t. w and b
        for a fixed w and b value.

        :returns dj_dw: The gradient of the cost w.r.t. the parameters w (scalar)
        :returns dj_db: The gradient of the cost w.r.t. the parameter b (scalar)
        """

        assert x.shape[0] == y.shape[0], ('feature and target lists must be the '
                                          'same length')

        num_examples = x.shape[0]
        dj_dw, dj_db = 0, 0

        for i in range(num_examples):
            f_wb = w * x[i] + b  # predicted value
            target_diff = f_wb - y[i]
            dj_dw += target_diff * x[i]
            dj_db += target_diff

        dj_dw /= num_examples
        dj_db /= num_examples
        return dj_dw, dj_db

    def gradient_descent(self, x, y):
        """
        Function to run the gradient descent algorithm. The goal here is to find 
        values of w and b that minimize our cost function
        
        :param x: list of features (inputs)
        :param y: list of targets (outputs)
        :returns w and b values that minimize the cost function
        """
        w, b = 0, 0

        alpha = 1E-2
        for i in range(10000):
            dj_dw, dj_db = self.compute_gradient(x, y, w, b)
            w -= alpha * dj_dw
            b -= alpha * dj_db
            if i < 100:
                print(f'iteration {i}: w: {w} b:{b}')
            elif i % 100 == 0:
                print(f'iteration {i}: w: {w} b:{b}')

        return w, b


class MultipleLinearRegression(LinearRegressionInterface):

    def load_training_data(self, training_examples):
        # Load in our training data
        # by convention m = number of training examples in our data set
        # x = inputs or features
        # y = outputs or targets
        m = len(training_examples)
        # assumes all examples have the same dimensions
        ex_length = len(training_examples[0])

        x = np.zeros((m, ex_length - 1))
        y = np.zeros(m)

        for i, example in enumerate(training_examples):
            x[i] = example[0:ex_length - 1]
            y[i] = example[-1]

        print(f'x: {x}')
        print(f'y: {y}')
        return x, y

    def compute_cost(self, x, y, w, b):
        """
        Function to compute the cost of our linear regression function. "Cost" is
        a measure of how close our models prediction line is to the target line.

        :param x: list of features (inputs)
        :param y: list of targets (outputs)
        :param w: slope of linear regression function
        :param b: y-intercept of linear regression function
        :returns: the cost of our model with fixed w and b params
        """
        assert x.shape[0] == y.shape[0], ('feature and target lists must be the '
                                          'same length')

        num_examples = x.shape[0]
        cost = 0

        for i in range(num_examples):
            f_wb = np.dot(x[i], w) + b  # predicted value
            cost += (f_wb - y[i]) ** 2

        cost /= 2 * num_examples
        return cost

    def compute_gradient(self, x, y, w, b):
        """
        Function to compute derivatives of the cost function w.r.t. w and b
        for a fixed w and b value.

        :returns dj_dw: The gradient of the cost w.r.t. the parameters w (ndarray).
        :returns dj_db: The gradient of the cost w.r.t. the parameter b (scalar).
        """

        assert x.shape[0] == y.shape[0], ('feature and target lists must be the '
                                          'same length')

        num_examples, num_features = x.shape[0], x.shape[1]

        dj_dw = np.zeros((num_features,))
        dj_db = 0

        for i in range(num_examples):
            f_wb = np.dot(x[i], w) + b  # predicted value
            print(f_wb)
            target_diff = f_wb - y[i]
            print(f'err is: {target_diff}')
            for j in range(num_features):
                dj_dw[j] += target_diff * x[i, j]
            dj_db += target_diff

        dj_dw /= num_examples
        dj_db /= num_examples
        return dj_dw, dj_db

    # TODO if not changed moved back to base class
    def gradient_descent(self, x, y):
        """
        Function to run the gradient descent algorithm. The goal here is to find 
        values of w and b that minimize our cost function
        
        :param x: list of features (inputs)
        :param y: list of targets (outputs)
        :returns w and b values that minimize the cost function
        """
        # Harcoding demo numbers that work with the housing_prices_multiple_inputs.csv input
        #b = 785.1811367994083
        #w = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
        b = 0
        w = np.zeros((x.shape[1],))

        alpha = 5.0e-7
        for i in range(1000):
            print(f'calling gradient with {x} {y} {w} {b}')
            dj_dw, dj_db = self.compute_gradient(x, y, w, b)
            print(f'dj_dw {dj_dw}')
            print(f'dj_db {dj_db}')

            #print(f'w is.... {w}')
            #print(f'b is.... {b}')
            w -= alpha * dj_dw
            b -= alpha * dj_db
            #print(f'now w is.... {w}')
            #print(f'now b is.... {b}')

            if i < 100:
                print(f'iteration {i}: w: {w} b:{b}')
            elif i % 100 == 0:
                print(f'iteration {i}: w: {w} b:{b}')

        return w, b


def main():
    parser = argparse.ArgumentParser(description='Run linear regression over training data inputs')
    parser.add_argument('--mode', '-m', type=RegressionType, choices=[RegressionType.SINGLE_INPUT,
                                                                      RegressionType.MULTIPLE_INPUT], required=True)

    args = parser.parse_args()
    mode = args.mode

    # create linear regression instance and read raw training data from file
    if RegressionType.SINGLE_INPUT == mode:
        print('')
        linear_regression = SingleLinearRegression()
        training_examples = get_csv_file_contents(os.getcwd() + FILE_NAME_SINGLE_INPUT)
    else:
        linear_regression = MultipleLinearRegression()
        training_examples = get_csv_file_contents(os.getcwd() + FILE_NAME_MULTIPLE_INPUTS)

    x, y = linear_regression.load_training_data(training_examples)

    w, b = linear_regression.gradient_descent(x, y)
    cost = linear_regression.compute_cost(x, y, w, b)

    if RegressionType.SINGLE_INPUT == mode:
        print(f"(w,b) values found by gradient descent: ({w:8.4f},{b:8.4f}\n"
              f"cost for that w,b: {cost}")
    if RegressionType.MULTIPLE_INPUT == mode:
        print(f'b,w found by gradient descent: {b:0.2f},{w}')
        print(f'cost: {cost}')
        num_examples = x.shape[0]
        for i in range(num_examples):
            print(f"prediction: {np.dot(x[i], w) + b:0.2f}, target value: {y[i]}")


if __name__ == '__main__':
    main()
