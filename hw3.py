import numpy as np


dataSet = np.genfromtxt('x06simple.csv', delimiter=',')

requiredData = dataSet[1:, 1:]

# randomizing data prior to division into training and testing sets / standardization

np.random.shuffle(requiredData)

trainingSamples = int(len(requiredData) * 2/3)

# isolating the x and y values in two separate matrices prior to standardization to make the whole thing easier
x_values_training = requiredData[:trainingSamples, :-1]
y_values_training = requiredData[:trainingSamples, -1]

x_values_testing = requiredData[trainingSamples:, :-1]
y_values_testing = requiredData[trainingSamples:, -1]


# standardization process

training_mean = np.mean(x_values_training, axis=0)
testing_mean = np.mean(x_values_testing, axis=0)

training_sd = np.std(x_values_training, axis=0)
testing_sd = np.std(x_values_testing, axis=0)

standardized_training = (x_values_training - training_mean) / training_sd
standardized_testing = (x_values_testing - testing_mean) / testing_sd

# padding a column of 1s to the standardized matrices
ones_training = np.ones((len(standardized_training), 1))
ones_testing = np.ones((len(standardized_testing), 1))

x_training = np.hstack((ones_training, standardized_training))
x_testing = np.hstack((ones_testing, standardized_testing))

# applying the closed form LSE rule
# the rule is theta = (X'X)^(-1) * X' * Y

x_training_transpose = np.transpose(x_training)


x_training_matrix = np.matrix(x_training)
x_training_transpose_matrix = np.matrix(x_training_transpose)



print(x_values_training.shape)

print(x_values_testing.shape)

print('Hello World!')
