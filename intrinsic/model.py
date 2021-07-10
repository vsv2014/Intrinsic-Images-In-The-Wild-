from sklearn.cluster import MiniBatchKMeans
import numpy as np
import scipy

class Model():
	def __init__(self, input, params, cost_func, DenseCRF):
		self.input = input
		self.cost_func = cost_func
		self.params = params
		self.dense_crf = DenseCRF
		self.iter_num = 0
		self.prev_s_layer = None

	def solve(self):
		self.initialize()
		for i in range(self.params.n_iters):
			print(i+1)
			self.iter_num = i
			# STAGE 1
			self.optimize_reflectance()
			self.remove_intensities()
			self.prev_s_layer = self.get_r_s()[1].copy()
			# STAGE 2
			self.smooth_shading()
			self.prev_s_layer = self.get_r_s()[1].copy()
		return self.get_r_s()

	def remove_intensities(self):
		labels = self.labels_nz
		intensities = self.intensities
		chromaticities = self.chromaticities
		nlabels = intensities.shape[0]

		new_to_old = np.nonzero(np.bincount(labels, minlength=nlabels))[0]
		old_to_new = np.empty(nlabels, dtype=np.int32)
		old_to_new.fill(-1)
		for new, old in enumerate(new_to_old):
			old_to_new[old] = new

		self.labels_nz = old_to_new[labels]
		self.intensities = intensities[new_to_old]
		self.chromaticities = chromaticities[new_to_old]

	def smooth_shading(self):
		median_intensity = np.median(self.intensities)
		log_intensities = np.log(self.intensities)
		A_data, A_rows, A_cols, A_shape, b = self.smooth_system(log_intensities)
		delta_intensities = self.minimize_l2(A_data, A_rows, A_cols, A_shape, b, 1e-8)
		intensities = np.exp(log_intensities + delta_intensities)
		intensities *= median_intensity / np.median(intensities)
		self.intensities = intensities

	def smooth_system(self, log_intensities):
		rows, cols = self.input.img.shape[0:2]
		labels = self.get_labels()
		log_image_gray = self.input.log_image_gray
		A_rows = []
		A_cols = []
		A_data = []
		b = []
		for i in xrange(rows - 1):
			for j in xrange(cols - 1):
				l0 = labels[i, j]
				l1 = labels[i + 1, j]
				if l0 != l1:
					A_rows.append(len(b))
					A_cols.append(l0)
					A_data.append(1)
					A_rows.append(len(b))
					A_cols.append(l1)
					A_data.append(-1)
					bval = log_image_gray[i, j] - log_image_gray[i + 1, j]
					bval+= log_intensities[l1] - log_intensities[l0]
					b.append(bval)
				l0 = labels[i, j]
				l1 = labels[i, j + 1]
				if l0 != l1:
					A_rows.append(len(b))
					A_cols.append(l0)
					A_data.append(1)
					A_rows.append(len(b))
					A_cols.append(l1)
					A_data.append(-1)
					bval = log_image_gray[i, j] - log_image_gray[i, j + 1]
					bval+= log_intensities[l1] - log_intensities[l0]
					b.append(bval)

		A_shape = (len(b), log_intensities.shape[0])
		return (
		    np.array(A_data),
		    np.array(A_rows),
		    np.array(A_cols),
		    A_shape,
		    np.array(b, dtype=np.float)
		)

	def initialize(self):
		img_irg = self.input.img_irg
		mask_nz = self.input.mask_nz
		rnd_state = None
		if self.params.fixed_seed:
			rnd_state = np.random.RandomState(seed=59173)
		samples = img_irg[mask_nz[0], mask_nz[1], :]

		# Handling large images
		if samples.shape[0] > self.params.kmeans_max_samples:
			samples = sklearn.utils.shuffle(samples)[:self.params.kmeans_max_samples, :]  

		samples[:, 0] *= self.params.kmeans_intensity_scale
		kmeans = MiniBatchKMeans(n_clusters=self.params.kmeans_n_clusters,
								compute_labels=False, random_state=rnd_state)
		kmeans.fit(samples)
		self.intensities = kmeans.cluster_centers_[:, 0] / self.params.kmeans_intensity_scale
		self.chromaticities = kmeans.cluster_centers_[:, 1:3]

	def optimize_reflectance(self):
		nlabels = self.intensities.shape[0]
		npixels = self.input.mask_nz[0].size
		dcrf = self.dense_crf(npixels, nlabels)
		u_cost = self.cost_func.compute_unary_costs(self.intensities, self.chromaticities, self.iter_num, self.prev_s_layer)
		dcrf.set_unary_energy(u_cost)
		p_cost = self.cost_func.compute_pairwise_costs(self.intensities, self.chromaticities, self.get_reflectances_rgb())
		p_cost = (self.params.pairwise_weight * p_cost).astype(np.float32)
		dcrf.add_pairwise_energy(pairwise_costs=p_cost, features=self.cost_func.features.copy())
		self.labels_nz = dcrf.map(self.params.n_crf_iters)

	def get_r_s(self):
		s_nz = self.input.image_gray_nz / self.intensities[self.labels_nz]
		r_nz = self.input.image_rgb_nz / np.clip(s_nz, 1e-4, 1e5)[:, np.newaxis]
		r = np.zeros((self.input.rows, self.input.cols, 3), dtype=r_nz.dtype)
		s = np.zeros((self.input.rows, self.input.cols), dtype=s_nz.dtype)
		r[self.input.mask_nz] = r_nz
		s[self.input.mask_nz] = s_nz
		return r, s

	def get_reflectances_rgb(self):
		nlabels = self.intensities.shape[0]
		rgb = np.zeros((nlabels, 3))
		s = 3.0 * self.intensities
		r = self.chromaticities[:, 0]
		g = self.chromaticities[:, 1]
		b = 1.0 - r - g
		rgb[:, 0] = s * r
		rgb[:, 1] = s * g
		rgb[:, 2] = s * b
		return rgb

	def get_labels(self):
		labels = np.empty((self.input.img.shape[0], self.input.img.shape[1]), dtype=np.int32)
		labels.fill(-1)
		labels[self.input.mask_nz] = self.labels_nz
		return labels

	def minimize_l2(self, A_data, A_rows, A_cols, A_shape, b, damp=1e-8):
		A = scipy.sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=A_shape)
		return scipy.sparse.linalg.lsmr(A, b, damp=damp)[0]