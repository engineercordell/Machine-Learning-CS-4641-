import numpy as np
from kmeans import pairwise_dist


class DBSCAN(object):

	def __init__(self, eps, minPts, dataset):
		self.eps = eps
		self.minPts = minPts
		self.dataset = dataset

	def fit(self):
		"""		
		Fits DBSCAN to dataset and hyperparameters defined in init().
		Args:
			None
		Return:
			cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
		Hint: Using sets for visitedIndices may be helpful here.
		Iterate through the dataset sequentially and keep track of your points' cluster assignments.
		See in what order the clusters are being expanded and new points are being checked, recommended to check neighbors of a point then decide whether to expand the cluster or not.
		If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
		Set the first cluster as C = 0
		"""
		c = 0
		cluster_idx = np.full(len(self.dataset), -1) # initialize everything as -1, to indicate unvisited/noise as instructed

		visitedIndices = set()
		for unvisitedPointIdx in range(len(self.dataset)):
			if unvisitedPointIdx in visitedIndices:
				continue

			visitedIndices.add(unvisitedPointIdx) # add to visited indices
			neighborPts = self.regionQuery(unvisitedPointIdx) # get indices of neighboring points

			if neighborPts.size < self.minPts: # if this point isn't immediately surrounded by the minimum # points needed for algorithm
				cluster_idx[unvisitedPointIdx] = -1 # mark as noise
			else:
				self.expandCluster(unvisitedPointIdx, neighborPts, c, cluster_idx, visitedIndices)
				c += 1

		return cluster_idx

	def expandCluster(self, index, neighborIndices, C, cluster_idx,
		visitedIndices):
		"""		
		Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
		   HINT: regionQuery could be used in your implementation
		Args:
			index: index of point P in dataset (self.dataset)
			neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
			C: current cluster as an int
			cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
			visitedIndices: set of indices in dataset visited so far
		Return:
			None
		Hints:
			1. np.concatenate(), and np.sort() may be helpful here. A while loop may be better than a for loop.
			2. Use, np.unique(), np.take() to ensure that you don't re-explore the same Indices. This way we avoid redundancy.
		"""
		cluster_idx[index] = C # add P to cluster C

		i = 0
		while i < len(neighborIndices): # for pointIdx in neighborIndices: # for each point P' in point P's eps-neighborhood
			# a while loop ensures that as new neighbors are being added, they are still being processed. 
			# the for loop cannot work here since it operates on the original sequence of neighborIndices used when the loop first began
			# - modifying neighborIndices within the loop doesn't change the loop's copy of neighborIndices
			# - only the original neighborIndices would be considered
			# neighborIndices is being expanded dynamically
			pointIdx = neighborIndices[i]

			if pointIdx not in visitedIndices: # if P' is not visited
				visitedIndices.add(pointIdx) # mark P' as visited
				newNeighborIndices = self.regionQuery(pointIdx) # NeighborPts' = regionQuery(P', eps)

				if newNeighborIndices.size >= self.minPts: # if sizeof(NeighborPts') >= MinPts
					# neighborIndices = np.sort(np.unique(np.concatenate([neighborIndices, newNeighborIndices]))) # NeighborPts = NeighborPts joined with NeighborPts'
					# for neighborIdx in newNeighborIndices:
					# 	if neighborIdx not in visitedIndices:
					# 		neighborIndices = np.append(newNeighborIndices, neighborIdx)
					new_points = set(newNeighborIndices) - set(neighborIndices) # get difference b/w sets, elements in the new neighbor indices, but not the already 
					# existing set of neighbor indices
					if new_points: # if this set is not null/empty
						neighborIndices = np.concatenate((neighborIndices, np.array(list(new_points)))) # new_points conversion set -> list -> np array
						# concatentate to preserve order 

			if cluster_idx[pointIdx] == -1: # if P' is not yet of member of any cluster
				cluster_idx[pointIdx] = C # add P' to cluster C

			i += 1


	def regionQuery(self, pointIndex):
		"""		
		Returns all points within P's eps-neighborhood (including P)
		
		Args:
			pointIndex: index of point P in dataset (self.dataset)
		Return:
			indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
		Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
		"""
		point = self.dataset[pointIndex] # retrieve actual point itself which has dimensionality D
		
		dist = pairwise_dist(point[np.newaxis, :], self.dataset) # expects 2D arrays not a single point, so we broadcast to 1 x D, instead of (D, ), so it's now
		# [[x, y]], 1 row, 2 dimensions of x and y, and dataset has dimensions N x D
		# result dist will be 1D array of shape (N, )

		return np.argwhere(dist <= self.eps)[:, 1] # np.argwhere will return an array of (I, 1), but because the shape needs to be returned as (I, ), we apply
		# .flatten()
		# return np.where(dist <= self.eps)[1]
