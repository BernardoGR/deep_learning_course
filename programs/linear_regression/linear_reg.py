from random import random
from csv import reader
import numpy as np
import matplotlib.pylab as plt

# global variable to store the errors/loss for visualisation
__errors__ = []


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
    return dataset


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


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def initialize_params(num_params):
    params = []
    for i in range(0, num_params):
        params.append(random())
    return params


def get_mean_error(params, x_set, y_set):
    error_acum = 0
    for i in range(len(x_set)):
        y_hat = predict(params, x_set[i])
        # print("x_set[i]:  %f,  y_set[i]:  %f,   y_hat:  %f" % (x_set[i][0], y_set[i], y_hat))
        # print( "y_hat  %f  y %f " % (y_hat,  y_set[i]))
        error = abs(y_hat - y_set[i])
        # this error is the original cost function, (the one used to make updates in GD is the
        # derived version of this formula)
        # error_acum += error**2
        error_acum += error**2
    mean_error_param = error_acum/len(x_set)
    return mean_error_param


def predict(params, sample):
    y_hat = 0
    for i in range(len(params)):
        # evaluates h(x) = a+bx1+cx2+ ... nxn..
        y_hat += params[i] * sample[i]
    return y_hat


def train_model(l_rate, x_train_set, y_train_set, num_epochs):
    global __errors__
    params = initialize_params(len(x_train_set[0]))
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        old_params = list(params)
        for j in range(len(params)):
            acum = 0
            for i in range(len(x_train_set)):
                # Sumatory part of the Gradient Descent formula for linear Regression.
                acum += (predict(params, x_train_set[i]) - y_train_set[i]) * x_train_set[i][j]
            # Subtraction of original parameter value with learning rate included.
            params[j] = params[j] - (l_rate * (1/len(x_train_set)) * acum)
        mean_error = get_mean_error(params, x_train_set, y_train_set)
        __errors__.append(mean_error)
        # print("epoch: %f, mean_error:  %f " % (epoch, mean_error))
        if epoch > 2:
            if __errors__[epoch-1] > (__errors__[epoch-2] + 1):
                print("\nDiverging")
                break
        if old_params == params:
            print("\nNo change in params, training finished")
            break
        if mean_error < 0.00001:
            print("\nMin error reached")
            break
    print("\nepochs:  %f, final mean_error:  %f \nEnd of Training." % (epoch, mean_error))
    return params

 
def test_model(x_test_set, params):
    result = []
    for i in range(0, len(x_test_set)):
        y_hat = predict(params, x_test_set[i])
        result.append(y_hat)
    return result


def plot_model(trained_params):
    x_model_set = np.linspace(0, 140, 1000)
    x_model_dim = []
    for x in x_model_set:
        x_model_dim.append([x, 1])
    plt.plot(x_model_set, test_model(x_model_dim, trained_params))
    plt.show()


def main():
    x_set = []
    x_test_set = []
    y_set = []
    y_test_set = []
    train_size = 0.9
    l_rate = 0.0001
    num_epochs = 100
    
    # load and prepare data
    filename = 'data.csv'
    dataset = np.array(load_csv(filename))
    print("\ndataset.shape:")
    print(dataset.shape)

    train, test = split_train_test(dataset, train_size)
    print("\ntrain.shape:")
    print(train.shape)
    print("\ntest.shape:")
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
    print("\ntraining params...")
    trained_params = train_model(l_rate, x_set, y_set, num_epochs)
    print("\n-> Final params: ")
    print(trained_params)

    # test model with unseen data
    result = test_model(x_test_set, trained_params)
    print("\n-> Test Error: ")
    print(get_mean_error(trained_params, x_test_set, y_test_set))

    # test with single data instance
    print("\n-> Single instance prediction:")
    print(predict(trained_params, [45.419730144973755,1]))
    print("-> Single instance real value:")
    print(55.165677145959123)

    # plot square mean error
    num_epochs_error = range(1, len(__errors__) + 1)
    plt.plot(num_epochs_error, __errors__)
    plt.show()

    # plot data and linear model
    plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
    plt.axis([0, 100, 0, 140])

    # plot model
    plot_model(trained_params)


if __name__ == "__main__":
    main()

