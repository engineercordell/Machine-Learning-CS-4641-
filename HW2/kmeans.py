"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""
import numpy as np

class KMeans(object):

	def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-05
		):
		"""
		Args:
			points: NxD numpy array, where N is # points and D is the dimensionality
			K: number of clusters
			init : how to initial the centers
			max_iters: maximum number of iterations (Hint: You could change it when debugging)
			rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
		Return:
			none
		"""
		self.points = points
		self.K = k
		if init == 'random':
			self.centers = self.init_centers()
		else:
			self.centers = self.kmpp_init()
		self.assignments = None
		self.loss = 0.0
		self.rel_tol = rel_tol
		self.max_iters = max_iters

	def init_centers(self):
		"""		
			Initialize the centers randomly
		Return:
			self.centers : K x D numpy array, the centers.
		Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
		"""
		indices = np.random.choice(self.points.shape[0], self.K, replace=False) # select random indices

		return self.points[indices] # self.points[indices, :] also works

	def kmpp_init(self):
		"""		
			Use the intuition that points further away from each other will probably be better initial centers.
			To complete this method, refer to the steps outlined below:.
			1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
			This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
			2. From the above sample, select only one random point to be the first cluster center.
			3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
			4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
			In other words, we will choose the next center based on the maximum of the minimum calculated distance
			instead of sampling randomly like in step 2. You may break ties arbitrarily.
			5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
		Return:
			self.centers : K x D numpy array, the centers.
		Hint:
			You could use functions like np.vstack() here.
		"""
		sample = self.points[np.random.choice(self.points.shape[0], int(np.ceil(0.01 * self.points.shape[0])), replace=False)] # 1% of randomly sampled points

		firstClusterCenter = sample[np.random.choice(sample.shape[0], 1)] # randomly sample first center
		
		newClusterCenters = firstClusterCenter # intialize the growing K x D array of cluster centers to be updated within loop

		for _ in range(1, self.K): # K-1 times because we already found the first center

			distance = np.min(pairwise_dist(sample, newClusterCenters), axis=1) ** 2 # calculate distance from points to *nearest* cluster centers, where taking the minimum
			# along a column gives min distance b/w x1 and clusters, x2 and clusters, etc.

			newClusterCenters = np.vstack([newClusterCenters, sample[np.argmax(distance)]]) # we need to get indice of maximum from distance. this is also now a 1D
			# array and can therefore simply take the argmax with default params.
			# also np.vstack([a,b]) only takes one param inside either () or [] to stack multiple arrays, in contrast to np.vstack(a,b)

		return newClusterCenters

	def update_assignment(self):
		"""		
			Update the membership of each point based on the closest center
		Return:
			self.assignments : numpy array of length N, the cluster assignment for each point
		Hint: Do not use loops for the update_assignment function
		Hint: You could call pairwise_dist() function
		Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
		"""
		if self.points.size == 0:
			return None
		
		return np.argmin(pairwise_dist(self.points, self.centers) ** 2, axis=1) # plus one, since argmin returns 0-based indicecs, meaning we need to offset by 1

	def update_centers(self):
		"""		
			update the cluster centers
		Return:
			self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.
		
		HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
		HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
		"""
		updated_clusters = []
		# updated_clusters = np.empty((self.K, self.points.shape[1]))

		for k in range(self.K):
			# have to sum together all points in the cluster...
			# we need to obtain indices from self.points
			mask = self.assignments == k

			data_k = self.points[mask] # we mask along the first dimension of point data N x D

			if data_k.size == 0: # avoid the case in which the cluster has no assigned points to prevent division by zero
				# updated_clusters[k] = np.nan
				if self.points.shape[0] > 0:
				# Reinitialize to a random point from the dataset
					mean = self.points[np.random.choice(self.points.shape[0])]
				else:
				# If dataset is empty, set to zeros
					mean = np.zeros(self.centers.shape[1], dtype=self.centers.dtype)
			else:
			# number_points = self.assignments[mask].size # size of cluster k
			# mean = np.divide(sum(data_k, axis=0), number_points) # sum along the rows, since there are N data points from 1 <= i <= N
				mean = np.mean(data_k, axis=0) # more concise, thanks numpy

			# updated_clusters = np.vstack([updated_clusters, mean]) avoid using np.vstack in a loop due to repeated allocations
			updated_clusters.append(mean)
			# updated_clusters[k] = np.mean(data_k, axis=0)

		return np.array(updated_clusters)
		# return updated_clusters


	def get_loss(self):
		"""		
			The loss will be defined as the sum of the squared distances between each point and it's respective center.
		Return:
			self.loss: a single float number, which is the objective function of KMeans.
		"""
		total_loss = 0.0 # calculate the loss for this iteration, so we don't assign it the value of self.loss

		for k in range(self.K):

			mask = self.assignments == k
			data_k = self.points[mask]

			# total_loss += sum(pairwise_dist(data_k, self.centers[k]) ** 2) let's be more efficient
			if data_k.size > 0:
				dist_squared = np.sum((data_k - self.centers[k]) ** 2, axis=1) # simply get rid of the square root in the euclid dist calculation formula d(x1, x2)
				total_loss += np.sum(dist_squared)

		return total_loss

	def train(self):
		"""		
			Train KMeans to cluster the data:
				0. Recall that centers have already been initialized in __init__
				1. Update the cluster assignment for each point
				2. Update the cluster centers based on the new assignments from Step 1
				3. Check to make sure there is no mean without a cluster,
				   i.e. no cluster center without any points assigned to it.
				   - In the event of a cluster with no points assigned,
					 pick a random point in the dataset to be the new center and
					 update your cluster assignment accordingly.
				4. Calculate the loss and check if the model has converged to break the loop early.
				   - The convergence criteria is measured by whether the percentage difference
					 in loss compared to the previous iteration is less than the given
					 relative tolerance threshold (self.rel_tol).
				   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
				5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
		
		Return:
			self.centers: K x D numpy array, the centers
			self.assignments: Nx1 int numpy array
			self.loss: final loss value of the objective function of KMeans.
		
		HINT: Do not loop over all the points in every iteration. This may result in time out errors
		HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
		"""
		self.loss = float('inf') # intialize to inf for first iteration to keep everything consistent for subsequent loops and avoid division by zero

		if self.points.size == 0:
			self.assignments = None
			return self.centers, self.assignments, self.loss

		for _ in range(self.max_iters):

			self.assignments = self.update_assignment()
			self.centers = self.update_centers()

			for k in range(self.K):
				if self.points[self.assignments == k].size == 0: # check if a cluster has 0 data points assigned to it
					# self.centers[k] = self.points[np.random.choice(self.points.shape[0])] # assign random point in all of data to be new center
					if self.points.size > 0:
						self.centers[k] = self.points[np.random.choice(self.points.shape[0])]
					else:
						self.centers[k] = np.zeros_like(self.centers[k])

			new_loss = self.get_loss()
			if abs((new_loss - self.loss) / self.loss) < self.rel_tol:
				break

			self.loss = new_loss # update for next iteration

		return self.centers, self.assignments, self.loss

def pairwise_dist(x, y):
	"""	
	Args:
		x: N x D numpy array
		y: M x D numpy array
	Return:
			dist: N x M array, where dist2[i, j] is the euclidean distance between
			x[i, :] and y[j, :]
	
	HINT: Do not use loops for the pairwise_distance function
	"""
	# arr1 = np.array(x)
	# arr2 = np.array(y)
	# return np.sqrt(np.sum(np.square(np.abs(arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :])), 2))

	# Understanding:
	# Suppose then that X \exists R^(5x3) and Y \exists R^(3x3). 
	# Given that (X-Y)^2 = X^2 - 2XY + Y^2:
	# 	the resultant array of 2XY will have dimensions 5x3
	# 	X^2 will have dimensions 5x1 since the sum returns a column vector
	# 	Y^2 will have dimensions 3x1 since the sum returns a column vector, later reshaped to 1x3 since we need to broadcast. 
	# Each term X^2, Y^2, 2XY can be written independently as a term and combined together based on the simple summation property.

	input_X = np.array(x)
	cluster_Y = np.array(y)

	X_squared = np.sum(np.square(input_X), axis=1)[:, np.newaxis]
	Y_squared = np.sum(np.square(cluster_Y), axis=1)[np.newaxis, :]

	twoXY = 2 * np.dot(input_X, np.transpose(cluster_Y))

	return np.sqrt(np.maximum(X_squared + Y_squared - twoXY, 0))


def fowlkes_mallow(xGroundTruth, xPredicted):
	"""	
	Args:
		xPredicted : list of length N where N = no. of test samples
		xGroundTruth: list of length N where N = no. of test samples
	Return:
		fowlkes-mallow value: final coefficient value as a float
	
	HINT: You can use loops for this function.
	HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
		1. Choose a pair of points from the Prediction.
		2. Compare the prediction pair pattern with the ground truth pair.
		3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
		4. Then calculate fowlkes-mallow value
	"""
	
	total_N = np.array([0, 0, 0, 0]) # where the format is TP/FN/FP/TN

	# assuming 0 = separated, 1 = together

	for i in range(len(xGroundTruth)): # n*(n-1)/2 computations
		for j in range(i + 1, len(xGroundTruth)):
			pred = xPredicted[i] == xPredicted[j]
			truth = xGroundTruth[i] == xGroundTruth[j]

			if pred and truth: # TP
				total_N[0] += 1
			elif not pred and truth: # FN
				total_N[1] += 1
			elif pred and not truth: # FP
				total_N[2] += 1
			else: # TN
				total_N[3] += 1
			
	# for pred, truth in zip(xGroundTruth, xPredicted):
	# 	if pred == 1 and truth == 1: # true positive
	# 		total_N[0] += 1
	# 	elif pred == 1 and truth == 0: # false negative
	# 		total_N[2] += 1
	# 	elif pred == 0 and truth == 1: # false positive
	# 		total_N[1] += 1
	# 	else: # true negative, where both pred and truth == 0
	# 		total_N[3] += 1

	numerator = total_N[0]
	denom = float(np.sqrt((total_N[0] + total_N[1]) * (total_N[0] + total_N[2])))

	if denom == 0:
		return 0.0

	return float(numerator / denom)