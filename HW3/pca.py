import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


class PCA(object):

	def __init__(self):
		self.U = None
		self.S = None
		self.V = None

	def fit(self, X: np.ndarray) ->None:
		"""		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
			  Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
			X: (N,D) numpy array corresponding to a dataset
		
		Return:
			None
		
		Set:
			self.U: (N, min(N,D)) numpy array
			self.S: (min(N,D), ) numpy array
			self.V: (min(N,D), D) numpy array
		"""
		means = np.mean(X, axis=0)[np.newaxis, :] # shape (1, d)
		X_centered = X - means # (n, d) - (1, d) -> (n, d) subtract the mean from observation

		U, S, V = np.linalg.svd(X_centered, full_matrices=False)
		# self.U = U[:, :np.min(U.shape)]
		# self.S = S[:np.min(S.shape)]
		# self.V = V[:np.min(V.shape), :]
		self.U = U
		self.S = S
		self.V = V

	def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
			data: (N,D) numpy array corresponding to a dataset
			K: int value for number of columns to be kept
		
		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		means = np.mean(data, axis=0, keepdims=True) # shape (1, d)
		data_centered = data - means # shape (n, d)

		# V_dot_S_squared = self.V * (self.S**2)
		# X_transpose_X = V_dot_S_squared * self.V.T

		# diag = np.diagonal(X_transpose_X)
		# k_features = diag[:K]

		# self.V already contains principal directions, so no need to do any of that ^

		X_new = data_centered @ self.V.T[:, :K] # shape (N, K)
		
		return X_new


	def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
		) ->np.ndarray:
		"""		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
			data: (N,D) numpy array corresponding to a dataset
			retained_variance: float value for amount of variance to be retained
		
		Return:
			X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
				   to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
		means = np.mean(data, axis=0, keepdims=True) # shape (1, d)
		data_centered = data - means # shape (n, d)

		tot_variance = np.sum(self.S**2) # sum of variance across all principal components. overall variance. shape (scalar)
		cum_variance = np.cumsum(self.S**2) # sum of variance so far across K principal components. shape (min(N, D), )
		desired_variance = retained_variance * tot_variance # keep some percentage (retained_variance) of the variance (total_variance)
		k = np.searchsorted(cum_variance, desired_variance) + 1 # find the smallest index where the desired variance falls within range of cum_variance
		# k starts at 1, so we add 1 at end

		X_new = data_centered @ self.V.T[:, :k] # reminder that it excludes index k

		return X_new

	def get_V(self) ->np.ndarray:
		"""		
		Getter function for value of V
		"""
		return self.V

	def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
		"""		
		You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
		For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
		You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
		Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
		Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.
		
		Args:
			xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
			ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
		self.fit(X)

		X_2d = self.transform(X, K=2) # 2d = Top 2 principal components
		X_3d = self.transform(X, K=3) # 3d = Top 3 principal components

		# 2D Plot
		table_2d = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
		table_2d['class'] = y
		figure_2d = px.scatter(table_2d, x='PC1', y='PC2', color='class', title=f"{fig_title} - 2d plot (Top 2 PCs)")
		figure_2d.show()

		# 3D Plot
		table_3d = pd.DataFrame(X_3d, columns=['PC1', 'PC2', 'PC3'])  
		table_3d['class'] = y
		figure_3d = px.scatter_3d(table_3d, x='PC1', y='PC2', z='PC3', color='class', title=f"{fig_title} - 3d plot (Top 3 PCs)")
		figure_3d.show()

		# 2D Random Features Plot
		random_idx = np.random.choice(X.shape[1], 2, replace=False) # 2 random features w/o replacement, from 0 to X.shape[1] exclusive
		X_random_2d = X[:, random_idx] # two random features with N data points
		table_random_2d = pd.DataFrame(X_random_2d, columns=['Random Feature 1', 'Random Feature 2'])
		table_random_2d['class'] = y
		fig_random_2d = px.scatter(table_random_2d, x='Random Feature 1', y='Random Feature 2', color='class', title=f"{fig_title} - 2d plot (Random Features)")
		fig_random_2d.show()