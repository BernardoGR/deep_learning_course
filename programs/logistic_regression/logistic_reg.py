from math import exp
from random import random
from csv import reader
import numpy as np
import matplotlib.pylab as plt


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    # normalize
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split data into train and test
def split_train_test(x, train_portion):
    train_num_rows = round(train_portion * (len(x)))

    # pick your indices for sample 1 and sample 2:
    s1 = np.random.choice(range(x.shape[0]), train_num_rows, replace=False)
    s2 = list(set(range(x.shape[0])) - set(s1))

    # extract your samples:
    train = x[s1]
    test = x[s2]

    return train, test


def initialize_params(num_params):
    params = []
    for i in range(0, num_params):
        params.append(random())
    return params


def predict(params, sample):
    y_hat = 0
    for i in range(len(params)):
        # evaluates h(x) = a+bx1+cx2+ ... nxn..
        y_hat += params[i] * sample[i]
    return 1.0 / (1.0 + exp(-y_hat))


# Estimate logistic regression coefficients using stochastic gradient descent
def train_model(l_rate, x_train_set, y_train_set, num_epochs):
    params = initialize_params(len(x_train_set[0]))
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        old_params = list(params)
        for j in range(len(x_train_set)):
            y_hat = predict(x_train_set[j], params)
            error = y_train_set[j] - y_hat
            for i in range(len(params)):
                params[i] = params[i] + l_rate * error * y_hat * (1.0 - y_hat) * x_train_set[j][i]
        if old_params == params:
            print("\nNo change in params, training finished")
            break
    print("\nepochs:  %f\nEnd of Training." % epoch)
    return params


# predict y values for a complete set
def predict_set(x_set, params):
    result = []
    for i in range(0, len(x_set)):
        y_hat = predict(params, x_set[i])
        result.append(round(y_hat))
    return result


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] ==  round(predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100.0


def plot_model(trained_params):
    x_model_set = np.linspace(0, 140, 1000)
    x_model_dim = []
    for x in x_model_set:
        x_model_dim.append([x, 1])
    plt.plot(x_model_set, predict_set(x_model_dim, trained_params))
    plt.show()


def main():
    x_set = []
    x_test_set = []
    y_set = []
    y_test_set = []
    train_size = 0.8
    l_rate = 0.1
    num_epochs = 100

    # load and prepare data
    filename = 'pima-indians-diabetes.csv'
    dataset = np.array(load_csv(filename))
    print(dataset.shape)

    train, test = split_train_test(dataset, train_size)
    print(train.shape)
    print(test.shape)

    # separate y and x set. Add bias column
    for row in train:
        row_li = row.tolist()
        y_set.append(row_li.pop())
        row_li.append(1.0)
        x_set.append(row_li)

    # separate test y and x set. Add bias column
    for row in test:
        row_li = row.tolist()
        y_test_set.append(row_li.pop())
        row_li.append(1.0)
        x_test_set.append(row_li)

    # train model
    trained_params = train_model(l_rate, x_set, y_set, num_epochs)
    print("\n-> Final params: ")
    print(trained_params)

    # test model: train data
    result_train = predict_set(x_set, trained_params)
    print("\n-> Predictions (train): ")
    print(result_train)
    print("-> Real Values (train): ")
    print(y_set)
    print("-> Accuracy (train): ")
    print(accuracy_metric(y_set, result_train))

    # test model with unseen data
    result_test = predict_set(x_test_set, trained_params)
    print("\n-> Test Predictions: ")
    print(result_test)
    print("-> Real Test Values: ")
    print(y_test_set)
    print("-> Accuracy (test): ")
    print(accuracy_metric(y_test_set, result_test))

    # test with single data instance
    '''
    print("\n-> Single instance prediction:")
    print(predict(trained_params, [45.419730144973755, 1]))
    print("-> Single instance real value:")
    print(55.165677145959123)
    '''

    # plot square mean error
    '''
    num_epochs_error = range(1, len(__errors__) + 1)
    plt.plot(num_epochs_error, __errors__)
    plt.show()
    '''

    # plot data
    '''
    plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
    plt.axis([0, 100, 0, 140])
    '''

    # plot model
    '''
    plot_model(trained_params)
    '''


if __name__ == "__main__":
    main()

