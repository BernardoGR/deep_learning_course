import fileinput
from random import random
from csv import reader


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
	global __errors__
	error_acum =0
	for i in range(len(x_set)):
		y_hat = eval_h(params, x_set[i])
		# print( "y_hat  %f  y %f " % (y_hat,  y_set[i]))   
		error = y_hat - y_set[i]
		error_acum += error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param = error_acum/len(x_set)
	return mean_error_param
	

def eval_h(params, sample):
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum;


def train_model(l_rate, x_train_set, y_train_set, num_epochs):
    params = initialize_params(len(x_train_set[0]))
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        old_params = list(params)
        for j in range(len(params)):
            acum = 0
            for i in range(len(x_train_set)):
                acum += (eval_h(params,x_train_set[i]) - y_train_set[i]) * x_train_set[i][j] #Sumatory part of the Gradient Descent formula for linear Regression.
            params[j] = params[j] - l_rate*(1/len(x_train_set)) * acum  #Subtraction of original parameter value with learning rate included.
        mean_error = get_mean_error(params, x_train_set, y_train_set)
        print("epoch: %f, mean_error:  %f " % (epoch, mean_error))
        if (old_params == params):
            print("\nNo change in params, training finished")
            break
        if (mean_error < 0.00001):
            print("\nMin error reached")
            break
    print("epochs:  %f \nEnd of Training." % epoch)
    return params

 
def test_model(x_test_set, params):
    result = []
    for i in range(0, len(x_test_set)):
        y_hat = eval_h(params, x_test_set[i])
        result.append(y_hat)
    return result


def main():
    x_set = []
    y_set = []
    l_rate = 0.01
    num_epochs = 5000
    
    # load and prepare data
    filename = 'data.csv'
    dataset = load_csv(filename)
    
    # separate y and x set. Add bias column
    for row in dataset:
        y_set.append(row.pop())
        row.append(1.0)
        x_set.append(row)
    
    # train model
    trained_params = train_model(l_rate, x_set, y_set, num_epochs)
    print("\n-> Final params: ")
    print(trained_params)
    
    # test model
    result = test_model(x_set, trained_params)
    print("-> Predictions: ")
    print(result)
    print("-> Real Values: ")
    print(y_set)



if __name__ == "__main__":
    main()

