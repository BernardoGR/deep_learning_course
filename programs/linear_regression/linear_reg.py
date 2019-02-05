import fileinput
from random import random


def initialize_params(num_params):
    params = []
    for i in range(0, num_params):
        params.append(random())
    return params


def show_mean_error(params, x_set, y_set):
	global __errors__
	error_acum =0
	for i in range(len(x_set)):
		y_hat = eval_h(params, x_set[i])
		# print( "y_hat  %f  y %f " % (y_hat,  y_set[i]))   
		error = y_hat - y_set[i]
		error_acum += error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param = error_acum/len(x_set)
	print("mean_error_param  %f " % mean_error_param)
	return mean_error_param
	

def eval_h(params, sample):
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum;


def train_model(num_x_cols, l_rate, x_train_set, y_train_set, num_epochs):
    params = initialize_params(num_x_cols + 1)
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        old_params = list(params)
        for j in range(len(params)):
            acum =0
            for i in range(len(x_train_set)):
                acum += (eval_h(params,x_train_set[i]) - y_train_set[i]) * x_train_set[i][j] #Sumatory part of the Gradient Descent formula for linear Regression.
            params[j] = params[j] - l_rate*(1/len(x_train_set)) * acum  #Subtraction of original parameter value with learning rate included.
        mean_error_param = show_mean_error(params, x_train_set, y_train_set)
        if (old_params == params):
            print("No change in params, training finished")
            break
        if (mean_error_param < 0.00001):
            print("Error is 0")
            break
    print("epochs:  %f " % epoch)
    print("final params: ")
    print(params)
    return params

 
def test_model(x_test_set, params):
    result = []
    for i in range(0, len(x_test_set)):
        y_hat = eval_h(params, x_test_set[i])
        result.append(y_hat)
    return result


def main():
    file_input = fileinput.input()
    x_set = []
    y_set = []
    l_rate = 0.01
    num_epochs = 5000

    num_x_cols = int(file_input[0])
    data_size = int(file_input[1])
    for i in range(data_size):
        row_str = file_input[i + 2].replace(" ", "").replace("\t", "").replace("\n", "").split(',')
        row_float = [float(x) for x in row_str]
        y_set.append(row_float.pop())
        row_float.append(1.0)
        x_set.append(row_float)

    trained_params = train_model(num_x_cols, l_rate, x_set, y_set, num_epochs)
    result = test_model(x_set, trained_params)
    print("Predictions: ")
    print(result)


if __name__ == "__main__":
    main()

