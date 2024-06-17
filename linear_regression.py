import os, csv
import numpy as np


FILE_NAME = '/housing_prices.csv'

# move to utils github repo? 
def get_csv_file_contents(file_path, include_header=False):
    """
    Assumes file can fit in memory
    """
    rows = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file) 

        for i,row in enumerate(csv_reader):
            if i == 0 and not include_header: continue
            rows.append(row)

    return rows


def compute_cost(x,y,w,b):
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
        f_wb = w * x[i] + b # predicted value
        cost += (f_wb - y[i]) ** 2 

    cost /= 2 * num_examples 
    return cost


def compute_gradient(x,y,w,b):
    """
    Function to compute derivates of the cost function w.r.t. w and b 
    for a fixed w and b value
    """

    assert x.shape[0] == y.shape[0], ('feature and target lists must be the '
                                      'same length')
         
    num_examples = x.shape[0] 
    dj_dw, dj_db = 0, 0
    
    for i in range(num_examples):
        f_wb = w * x[i] + b #predicted value
        target_diff = f_wb - y[i] 
        dj_dw += target_diff * x[i]
        dj_db += target_diff
    
    dj_dw /= num_examples 
    dj_db /= num_examples 
    return dj_dw, dj_db

def gradient_descent(x,y):
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
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < 100:
            print(f'iteration {i}: w: {w} b:{b}') 
        elif i % 100 == 0:
            print(f'iteration {i}: w: {w} b:{b}') 

    return w,b


def main():
    training_examples = get_csv_file_contents(os.getcwd() + FILE_NAME)

    # Load in our training data
    # by convention m = number of training examples in our data set
    # x = inputs or features
    # y = ouputs or targets
    m = len(training_examples)
    x = np.zeros(m)
    y = np.zeros(m)

    for i,example in enumerate(training_examples):
       x[i] = example[0]
       y[i] = example[1]

    print(f'x: {x}')
    print(f'y: {y}')


    cost = compute_cost(x,y,200,-100)
    print(f'cost is {cost}')

    w,b = gradient_descent(x,y)
    cost = compute_cost(x,y,w,b)
    print(f"(w,b) values found by gradient descent: ({w:8.4f},{b:8.4f}\n"
          f"cost for that w,b: {cost}")


if __name__ == '__main__':
   main() 


