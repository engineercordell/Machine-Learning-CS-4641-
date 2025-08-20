from typing import List, Tuple
import numpy as np


class Regression(object):

	def __init__(self):
		pass

	def rmse(self, pred: np.ndarray, label: np.ndarray) ->float:
		"""		
		Calculate the root mean square error.
		
		Args:
			pred: (N, 1) numpy array, the predicted labels
			label: (N, 1) numpy array, the ground truth labels
		Return:
			A float value
		"""
		return float(np.sqrt(sum((label - pred)**2) / pred.shape[0]))

	def construct_polynomial_feats(self, x: np.ndarray, degree: int
		) ->np.ndarray:
		"""		
		Given a feature matrix x, create a new feature matrix
		which is all the possible combinations of polynomials of the features
		up to the provided degree
		
		Args:
			x:
				1-dimensional case: (N,) numpy array
				D-dimensional case: (N, D) numpy array
				Here, N is the number of instances and D is the dimensionality of each instance.
			degree: the max polynomial degree
		Return:
			feat:
				For 1-D array, numpy array of shape Nx(degree+1), remember to include
				the bias term. feat is in the format of:
				[[1.0, x1, x1^2, x1^3, ....,],
				 [1.0, x2, x2^2, x2^3, ....,],
				 ......
				]
		Hints:
			- For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
			the bias term.
			- It is acceptable to loop over the degrees.
			- Example:
			For inputs x: (N = 3 x D = 2) and degree: 3,
			feat should be:
		 
			[[[ 1.0        1.0]
				[ x_{1,1}    x_{1,2}]
				[ x_{1,1}^2  x_{1,2}^2]
				[ x_{1,1}^3  x_{1,2}^3]]
		
				[[ 1.0        1.0]
				[ x_{2,1}    x_{2,2}]
				[ x_{2,1}^2  x_{2,2}^2]
				[ x_{2,1}^3  x_{2,2}^3]]
		
				[[ 1.0        1.0]
				[ x_{3,1}    x_{3,2}]
				[ x_{3,1}^2  x_{3,2}^2]
				[ x_{3,1}^3  x_{3,2}^3]]]
		"""
		if x.ndim == 1:
			polynomial_feats = np.ones([x.shape[0], degree + 1])

			for d in range(1, degree + 1): # From 1 inclusive to degree + 1 exclusive
				polynomial_feats[:, d] = x**d # x is 1D
	
			return polynomial_feats
		else:
			polynomial_feats = np.ones([x.shape[0], degree + 1, x.shape[1]])

			for d in range(1, degree + 1):
				polynomial_feats[:, d, :] = x**d # x is 2D

			return polynomial_feats

	def predict(self, xtest: np.ndarray, weight: np.ndarray) ->np.ndarray:
		"""		
		Using regression weights, predict the values for each data point in the xtest array
		
		Args:
			xtest: (N,1+D) numpy array, where N is the number
					of instances and D is the dimensionality
					of each instance with a bias term
			weight: (1+D,1) numpy array, the weights of linear regression model
		Return:
			prediction: (N,1) numpy array, the predicted labels
		"""
		# xtest_expand = self.construct_polynomial_feats(xtest, len(weight) - 1)

		# weight = weight.reshape(-1)

		# pred = xtest_expand @ weight

		# return pred.reshape(-1, 1)

		# # return np.reshape(weight, (np.newaxis, :, :))  * xtest_expand
		pred = xtest @ weight
		pred = pred.reshape(pred.shape[0], 1)
		return pred

	def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray
		) ->np.ndarray:
		"""		
		Fit a linear regression model using the closed form solution
		
		Args:
			xtrain: (N,1+D) numpy array, where N is number
					of instances and D is the dimensionality
					of each instance with a bias term
			ytrain: (N,1) numpy array, the true labels
		Return:
			weight: (1+D,1) numpy array, the weights of linear regression model
		Hints:
			- For pseudo inverse, you should use the numpy linear algebra function (np.linalg.pinv)
		"""
		# return np.linalg.pinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain # compute and return weight beta\hat, least-squares estimator
		X_pinv= np.linalg.pinv(xtrain)
		weight = X_pinv @ ytrain
		return weight

	def linear_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
		int=5, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
		"""		
		Fit a linear regression model using gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
			xtrain: (N,1+D) numpy array, where N is number
					of instances and D is the dimensionality
					of each instance with a bias term
			ytrain: (N,1) numpy array, the true labels
		Return:
			weight: (1+D,1) numpy array, the weights of linear regression model
			loss_per_epoch: (epochs,) list of floats, rmse of each epoch
		Hints:
			- RMSE loss should be recorded AFTER the gradient update in each iteration.
		"""
		N, D = xtrain.shape
		weights = np.zeros([D, 1])
		loss_per_epoch = []

		for _ in range(epochs):
			pred = self.predict(xtrain, weights) # both work
			# pred = xtrain @ weights # both work

			# error = np.sum((pred - ytrain)**2)
			# error = ytrain - x_curr_pred
			error = pred - ytrain # actual minus predicted or predicted minus actual?

			gradient = (xtrain.T @ error) / N # factor of 2 is a part of the learning rate
			weights -= learning_rate * gradient

			update_pred = xtrain @ weights

			rmse = self.rmse(update_pred, ytrain)
			loss_per_epoch.append(rmse)

		return (weights, loss_per_epoch)

	def linear_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
		int=100, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
		"""		
		Fit a linear regression model using stochastic gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
			xtrain: (N,1+D) numpy array, where N is number
					of instances and D is the dimensionality of each
					instance with a bias term
			ytrain: (N,1) numpy array, the true labels
			epochs: int, number of epochs
			learning_rate: float
		Return:
			weight: (1+D,1) numpy array, the weights of linear regression model
			loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
		Hints:
			- RMSE loss should be recorded AFTER the gradient update in each iteration.
			- Keep in mind that the number of epochs is the number of complete passes
			through the training dataset. SGD updates the weight for one datapoint at
			a time. For each epoch, you'll need to go through all of the points.
		
		NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
		"""
		N, D = xtrain.shape
		weights = np.zeros([D, 1])
		loss_per_step = []

		# for epoch in range(N*epochs):
		# 	pred = self.predict(xtrain, weights) # both work
		# 	# pred = xtrain @ weights # both work

		# 	# error = np.sum((pred - ytrain)**2)
		# 	# error = ytrain - x_curr_pred
		# 	error = pred - ytrain # actual minus predicted or predicted minus actual?

		# 	gradient = (xtrain.T @ error) / N # factor of 2 is a part of the learning rate
		# 	weights -= learning_rate * gradient

		# 	update_pred = xtrain @ weights

		# 	rmse = self.rmse(update_pred, ytrain)
		# 	loss_per_epoch.append(rmse)

		for _ in range(epochs):

			for i in range(N): # ith data point for entire dataset N
				x_i = xtrain[i].reshape(1, -1) # 1xD for ith data point
				y_i = ytrain[i].reshape(1, 1) # 1x1 actual label

				pred_i = x_i @ weights  # shape (1,1) dot prod

				error = pred_i - y_i  # shape (1,1) pred - actual

				gradient = x_i.T @ error  # shape (D, 1)

				weights -= learning_rate * gradient

				updated_preds = xtrain @ weights
				rmse = self.rmse(updated_preds, ytrain)
				loss_per_step.append(rmse)

		return (weights, loss_per_step)

	def ridge_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray,
		c_lambda: float) ->np.ndarray:
		"""		
		Fit a ridge regression model using the closed form solution
		
		Args:
			xtrain: (N,1+D) numpy array, where N is
					number of instances and D is the dimensionality
					of each instance with a bias term
			ytrain: (N,1) numpy array, the true labels
			c_lambda: float value, value of regularization constant
		Return:
			weight: (1+D,1) numpy array, the weights of ridge regression model
		Hints:
			- You should adjust your I matrix to handle the bias term differently than the rest of the terms
		"""
		# xtrain_no_bias = xtrain[:, 1:]
		_, D = xtrain.shape
		
		c_lambda_matrix = c_lambda * np.eye(D) # D is # cols + 1, # features + bias term
		c_lambda_matrix[0, 0] = 0  # only apply regularization to non-bias terms

		# return np.linalg.inv(xtrain_no_bias.T @ xtrain_no_bias + c_lambda * np.eye(xtrain_no_bias.shape[1])) @ xtrain_no_bias.T @ ytrain
		# this causes a dimension mismatch b/w computed weights and solution weights bc bias term shouldn't be removed at all
		# only regularization bias term should be modified, don't alter the original xtrain matrix
		
		return np.linalg.inv(xtrain.T @ xtrain + c_lambda_matrix) @ xtrain.T @ ytrain

	def ridge_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda:
		float, epochs: int=500, learning_rate: float=1e-07) ->Tuple[np.
		ndarray, List[float]]:
		"""		
		Fit a ridge regression model using gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
			xtrain: (N,1+D) numpy array, where N is number
					of instances and D is the dimensionality of each
					instance with a bias term
			ytrain: (N,1) numpy array, the true labels
			c_lambda: float value, value of regularization constant
			epochs: int, number of epochs
			learning_rate: float
		Return:
			weight: (1+D,1) numpy array, the weights of linear regression model
			loss_per_epoch: (epochs,) list of floats, rmse of each epoch
		Hints:
			- RMSE loss should be recorded AFTER the gradient update in each iteration.
			- You should avoid applying regularization to the bias term in the gradient update
		"""
		N, D = xtrain.shape
		weights = np.zeros([D, 1])
		loss_per_epoch = []
		c_lambda_matrix = c_lambda * np.eye(D)
		c_lambda_matrix[0, 0] = 0 

		for _ in range(epochs):
			pred = self.predict(xtrain, weights) # both work
			# pred = xtrain @ weights # both work

			# error = np.sum((pred - ytrain)**2)
			# error = ytrain - x_curr_pred
			error = pred - ytrain # actual minus predicted or predicted minus actual?

			gradient = (xtrain.T @ error) / N + (c_lambda_matrix @ weights) / N # factor of 2 is a part of the learning rate
			weights -= learning_rate * gradient

			update_pred = xtrain @ weights

			# regularization = (c_lambda_matrix / (2 * N)) * weights.T @ weights

			rmse = self.rmse(update_pred, ytrain)
			loss_per_epoch.append(rmse)

		return (weights, loss_per_epoch)

	def ridge_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray,
		c_lambda: float, epochs: int=100, learning_rate: float=0.001) ->Tuple[
		np.ndarray, List[float]]:
		"""		
		Fit a ridge regression model using stochastic gradient descent.
		Although there are many valid initializations, to pass the local tests
		initialize the weights with zeros.
		
		Args:
			xtrain: (N,1+D) numpy array, where N is number
					of instances and D is the dimensionality of each
					instance with a bias term
			ytrain: (N,1) numpy array, the true labels
			c_lambda: float, value of regularization constant
			epochs: int, number of epochs
			learning_rate: float
		Return:
			weight: (1+D,1) numpy array, the weights of linear regression model
			loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
		Hints:
			- RMSE loss should be recorded AFTER the gradient update in each iteration.
			- Keep in mind that the number of epochs is the number of complete passes
			through the training dataset. SGD updates the weight for one datapoint at
			a time. For each epoch, you'll need to go through all of the points.
			- You should avoid applying regularization to the bias term in the gradient update
		
		NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
		"""
		N, D = xtrain.shape
		weights = np.zeros([D, 1])
		loss_per_step = []
		c_lambda_matrix = (c_lambda / N) * np.eye(D)
		c_lambda_matrix[0, 0] = 0 

		for _ in range(epochs):
			for i in range(N):
				x_i = xtrain[i].reshape(1, -1) # 1xD for ith data point
				y_i = ytrain[i].reshape(1, 1) # 1x1 actual label

				pred_i = x_i @ weights  # shape (1,1) dot prod

				error = pred_i - y_i  # shape (1,1) pred - actual

				gradient = x_i.T @ error + (c_lambda_matrix @ weights) # shape (D, 1)
 
				weights -= learning_rate * gradient

				update_pred = xtrain @ weights

				rmse = self.rmse(update_pred, ytrain)
				loss_per_step.append(rmse)

		return (weights, loss_per_step)

	def ridge_cross_validation(self, X: np.ndarray, y: np.ndarray, kfold:
		int=5, c_lambda: float=100) ->List[float]:
		"""		
		For each of the k-folds of the provided X, y data, fit a ridge regression model
		and then evaluate the RMSE. Return the RMSE for each fold
		
		Args:
			X : (N,1+D) numpy array, where N is the number of instances
				and D is the dimensionality of each instance with a bias term
			y : (N,1) numpy array, true labels
			kfold: int, number of folds you should take while implementing cross validation.
			c_lambda: float, value of regularization constant
		Returns:
			loss_per_fold: list[float], RMSE loss for each kfold
		Hints:
			- np.concatenate might be helpful.
			- Use ridge_fit_closed for this function.
			- Look at 3.5 to see how this function is being used.
			- For kfold=5:
				split X and y into 5 equal-size folds
				use 80 percent for training and 20 percent for test
		"""
		# determine how many data points are in fold first, in case N is not divisible by kfold
		N, _ = X.shape # n data points
		size_fold = np.full(kfold, N // kfold, dtype=int) # if N=500 and kfold = 5, then create arr of 5 values, each indice containing N // kfold = 100 [100, 100, 100, 100, 100]
		size_fold[:N % kfold] += 1 # :0 doesn't select anything (indicating divisibility), but if N / kfold isn't divisible, for instance N % kfold = 2, then 0:2, which means
		# first 2 indices are incremented by 1
		# size_fold[:502 % 5] += 1 -> size_fold[:2] += 1 -> size_fold[0 and 1] += 1

		current = 0 # index 0
		folds = [] # stores indices of each fold
		indices = np.arange(N) # indices from 0 to N exclusive (N-1), which if N=500, then from i=0 to i=499, shape
		for fold in size_fold: # [101, 101, 101, 100, 100], where N = 503, kfold = 5
			start = current # start = i = 0
			stop = current + fold # stop = i = 0 + 101 = 101
			folds.append(indices[start:stop]) # folds = [[0,...,100], ...]
			current = stop # curr = i = 101

		loss_per_fold = []

		for k in range(kfold): # for every fold...
			train_i = np.concatenate([folds[i] for i in range(kfold) if i != k]) # k-1 indicecs are used to train
			test_i = folds[k] # kth indices used to test

			xtrain, ytrain = X[train_i], y[train_i]
			xtest, ytest = X[test_i], y[test_i]

			weights = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
			pred = self.predict(xtest, weights)

			rmse = self.rmse(pred, ytest)
			loss_per_fold.append(rmse)

		return loss_per_fold


	def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
		lambda_list: List[float], kfold: int) ->Tuple[float, float, List[float]
		]:
		"""
		FUNCTION PROVIDED TO STUDENTS

		Search over the given list of possible lambda values lambda_list
		for the one that gives the minimum average error from cross-validation

		Args:
			X : (N, 1+D) numpy array, where N is the number of instances and
				D is the dimensionality of each instance with a bias term
			y : (N,1) numpy array, true labels
			lambda_list: list of regularization constants (lambdas) to search from
			kfold: int, Number of folds you should take while implementing cross validation.
		Returns:
			best_lambda: (float) the best value for the regularization const giving the least RMSE error
			best_error: (float) the average RMSE error achieved using the best_lambda
			error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
		"""
		best_error = None
		best_lambda = None
		error_list = []
		for lm in lambda_list:
			err = self.ridge_cross_validation(X, y, kfold, lm)
			mean_err = np.mean(err)
			error_list.append(mean_err)
			if best_error is None or mean_err < best_error:
				best_error = mean_err
				best_lambda = lm
		return best_lambda, best_error, error_list
