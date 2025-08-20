import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):

	def __init__(self, X, K, max_iters=100):
		"""
		Args:
			X: the observations/datapoints, N x D numpy array
			K: number of clusters/components
			max_iters: maximum number of iterations (used in EM implementation)
		"""
		self.points = X
		self.max_iters = max_iters
		self.N = self.points.shape[0]
		self.D = self.points.shape[1]
		self.K = K

	def softmax(self, logit):
		"""		
		Args:
			logit: N x D numpy array
		Return:
			prob: N x D numpy array. See the above function.
		Hint:
			Add keepdims=True in your np.sum() function to avoid broadcast error.
		"""
		# Used to compute responsibilities

		# It is possible that $logit_{i, j}$ is very large, making $\exp(\cdot)$ of it to explode. 
		# To make sure it is numerically stable, for each row of $logits$ subtract the maximum of that row.
		# data is normalized across each row prob_{i,j} 
		adjusted_logit = logit - np.max(logit, axis=1)[:, np.newaxis] # must do z-c and not c-z where c is the max in each row to preserve invariance to constant shifts
		# property of softmax: softmax(z+c)=softmax(z).
		# softmax(c-z) is equivalent to flipping the sign and shifting, changing the relative relationship b/w logits
		# softmax(c-z) = softmax(-(z-c)) \neq softmax(z-c)
		exp_logit = np.exp(adjusted_logit)
		denom_sum = np.sum(exp_logit, axis=1, keepdims=True) # sum across the dimension D, so from left to right in a row. i (row) remains fixed, and d is iterated
		# across to sum the entire row
		return np.divide(exp_logit, denom_sum) # divide and get prob

	def logsumexp(self, logit):
		"""		
		Args:
			logit: N x D numpy array
		Return:
			s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
		Hint:
			The keepdims parameter could be handy
		"""
		# Used to calculate log-likelihoods 

		adjusted_logit = logit - np.max(logit, axis=1)[:, np.newaxis]
		exp_logit = np.exp(adjusted_logit)
		s = np.log(np.sum(exp_logit, axis=1, keepdims=True)) + np.max(logit, axis=1)[:, np.newaxis]
		
		return s

	def normalPDF(self, points, mu_i, sigma_i):
		"""		
		Args:
			points: N x D numpy array
			mu_i: (D,) numpy array, the center for the ith gaussian.
			sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
			pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
			np.diagonal() should be handy.
		"""
		n, d = points.shape
		main_diag_sigma = sigma_i.diagonal().reshape(1, d) # 1xD
		mu = mu_i.reshape(1, d) # 1xD

		outside_coeff = (1 / np.sqrt(2 * np.pi * main_diag_sigma))
		x_minus_mu = (points - mu) ** 2 # NxD - (1xD -> NxD) = NxD 
		inside_exp = -0.5 * x_minus_mu / main_diag_sigma

		exp_term = np.exp(inside_exp)
		pdf = np.prod(outside_coeff * exp_term, axis=1)

		return pdf

	def multinormalPDF(self, points, mu_i, sigma_i):
		"""		
		Args:
			points: N x D numpy array
			mu_i: (D,) numpy array, the center for the ith gaussian.
			sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
			normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
			1. np.linalg.det() and np.linalg.inv() should be handy.
			2. Note the value in self.D may be outdated and not correspond to the current dataset.
			3. You may wanna check if the matrix is singular before implementing calculation process.
		"""
		raise NotImplementedError

	def create_pi(self):
		"""		
		Initialize the prior probabilities
		Args:
		Return:
		pi: numpy array of length K, prior
		"""
		return np.full(self.K, 1/(self.K))

	def create_mu(self):
		"""		
		Intialize random centers for each gaussian
		Args:
		Return:
		mu: KxD numpy array, the center for each gaussian.
		"""
		indices = np.random.choice(self.points.shape[0], self.K, replace=True)

		return self.points[indices]

	def create_sigma(self):
		"""		
		Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
		by K diagonal matrices.
		Args:
		Return:
		sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
			You will have KxDxD numpy array for full covariance matrix case
		"""
		arr = np.eye(self.points.shape[1])
		return np.repeat(arr[np.newaxis, :, :], self.K, axis=0)

	def _init_components(self, **kwargs):
		"""		
		Args:
			kwargs: any other arguments you want
		Return:
			pi: numpy array of length K, prior
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
				You will have KxDxD numpy array for full covariance matrix case
		
			Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
		"""
		np.random.seed(5)
		return self.create_pi(), self.create_mu(), self.create_sigma()

	def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
			array for full covariance matrix case
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		
		Return:
			ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
		"""
		log_pi_arr = np.repeat(np.log(pi + 1e-32)[np.newaxis, :], self.points.shape[0], axis=0) # already numpy array

		ll = [] # use to maintain and append to growing matrix

		for cluster in range(self.K): # k times for each Gaussian component
			log_gaussian_col = np.log(self.normalPDF(self.points, mu[cluster, :], sigma[cluster, :, :]).reshape(self.points.shape[0], 1) + 1e-32) # returns a (N, ), so reshape to (N, 1)

			ll.append(log_gaussian_col) # Append to python list

		ll = np.column_stack(ll) # convert to numpy array and stack

		ll_joint = log_pi_arr + ll # element-wise addition to compute log-likelihood[i, k]

		return ll_joint

	def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
			array for full covariance matrix case
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
		
		Hint:
			You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
		"""
		return self.softmax(self._ll_joint(pi, mu, sigma))

	def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
		"""		
		Args:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
			full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
			pi: np array of length K, the prior of each component
			mu: KxD numpy array, the center for each gaussian.
			sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
			array for full covariance matrix case
		
		Hint:
			There are formulas in the slides and in the Jupyter Notebook.
			Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
		"""
		N_per_k = np.sum(tau, axis=0) # shape (K, )
		N_num_points = self.points.shape[0] # scalar

		pi_new = N_per_k / N_num_points # shape (K, )
		sum_tau_xn = np.sum(tau[:, :, np.newaxis] * self.points[:, np.newaxis, :], axis=0) # shape (K, D)

		mu_new = sum_tau_xn / N_per_k[:, np.newaxis] # shape (K, D)

		# N x D - K x D ...

		diff_squared = (self.points[:, np.newaxis, :] - mu_new[np.newaxis, :, :])  ** 2  # (N, D) - (K, D) -> (N, K, D)
		sum_tau_diff_squared = np.sum(tau[:, :, np.newaxis] * diff_squared, axis=0)  # shape (K, D)
		sigma_new = sum_tau_diff_squared / N_per_k[:, np.newaxis]  # shape (K, D)
		sigma_new = np.array([np.diag(sigma_new[k]) for k in range(self.K)])  # shape (K, D, D), converts to appropriate dimension

		return pi_new, mu_new, sigma_new

	def __call__(
		self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs
	):  # No need to change
		"""
		Args:
			abs_tol: convergence criteria w.r.t absolute change of loss
			rel_tol: convergence criteria w.r.t relative change of loss
			kwargs: any additional arguments you want

		Return:
			tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
			(pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

		Hint:
			You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
		"""
		pi, mu, sigma = self._init_components(**kwargs)
		pbar = tqdm(range(self.max_iters))

		prev_loss = None
		for it in pbar:
			# E-step
			tau = self._E_step(pi, mu, sigma, full_matrix)

			# M-step
			pi, mu, sigma = self._M_step(tau, full_matrix)

			# calculate the negative log-likelihood of observation
			joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
			loss = -np.sum(self.logsumexp(joint_ll))
			if it:
				diff = np.abs(prev_loss - loss)
				if diff < abs_tol and diff / prev_loss < rel_tol:
					break
			prev_loss = loss
			pbar.set_description("iter %d, loss: %.4f" % (it, loss))
		return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
	"""	
	Clusters pixels in the input image
	
	Each pixel can be considered as a separate data point (of length 3),
	which you can then cluster using GMM. Then, process the outputs into
	the shape of the original image, where each pixel is its most likely value.
	
	Args:
		image: input image of shape(H, W, 3)
		K: number of components
		max_iters: maximum number of iterations in GMM. Default is 10
		full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
	Return:
		clustered_img: image of shape(H, W, 3) after pixel clustering
	
	Hints:
		What do mu and tau represent?
	"""
	# If a pixel is a data point, that means there are H * W = N data points in an image, each with 3 colors RGB, so that means D = 3
	# Therefore, the image array should be reshape to an N x 3 matrix so GMM can process it.

	height, width, _ = image.shape

	dataset = image.reshape(-1, 3) # H * W gets resized to fit within the first dimension, hence the -1. Dimensionality D = 3 remains unchanged

	gmm_model = GMM(dataset, K, max_iters)
	tau, (_, mu, _) = gmm_model.__call__(full_matrix) # luckily didn't have to implement
	
	
	cluster_assignments = np.argmax(tau, axis=1)  # collect tau of shape (N, K) and get the probability that a pixel belongs to a particular cluster, with shape (N,) being returned
	
	# the cluster with the highest responsibility will have a particular mean value
	# we need to find the most probable cluster for which each pixel in the image belongs, and that particular cluster has a certain mean value
	# once that's determined, we extract its mean pixel value and replace the pixel color with the mean value
	# i.e. replace each pixel's RGB value with the corresponding cluster's mean RGB value, and this reduces the number of colors in the image to K, because there's K mean values
	clustered_dataset = mu[cluster_assignments]  # shape (N, 3)
	
	clustered_dataset = np.clip(clustered_dataset, 0, 255).astype(np.uint8)  # ensures values are strcitly within the bounds of 0 and 255 inclusive, and conver them to uint8 for image processing
	# purposes

	return clustered_dataset.reshape(height, width, 3)
	
	
