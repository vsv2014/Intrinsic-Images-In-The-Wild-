import numpy as np

class CostFunction():
	def __init__(self, input, params):
		self.input = input
		self.params = params
		self.features = self.get_features()

	def cost_r(self, rm):
		return 0

	def cost_s(self, sm):
		return self.params.abs_shading_weight * np.abs(np.log(sm) - np.log(self.params.abs_shading_gray_point))

	def compute_unary_costs(self, intensities, chromaticities, iter_num, prev_s_layer):
		nlabels = intensities.shape[0]
		unary_costs = np.zeros((self.input.mask_nnz, nlabels), dtype=np.float32)
		for i in range(nlabels):
			# reflectance part
			s_layer = self.input.image_gray_nz / intensities[i]
			r_layer = self.input.image_rgb_nz / np.clip(s_layer, 1e-4, 1e5)[:, np.newaxis]
			unary_costs[:, i] += self.cost_s(s_layer) + self.cost_r(r_layer)
			chroma_val = np.sum(np.square(self.input.image_irg_nz[:, 1:3] - chromaticities[i, :]), axis=1)
			unary_costs[:, i] += self.params.chromaticity_weight * chroma_val
			# shading part
			if prev_s_layer is not None:
				blur_inp = np.log(prev_s_layer)
				sigma_sp = 0.1 * self.input.diag / (1 + iter_num)
				log_s_target_layer = self.input.apply_blur(blur_inp, sigma_sp).flatten()
				log_s_layer = np.log(s_layer)
				unary_costs[:, i] += self.params.shading_target_weight * np.square(log_s_layer - log_s_target_layer)
		return unary_costs

	def compute_pairwise_costs(self, intensities, chromaticities, reflectances):
		nlabels = intensities.shape[0]
		reflectances = np.log(np.clip(reflectances, 1e-5, np.inf))
		binary_costs = np.zeros((nlabels, nlabels), dtype=np.float32)
		for i in xrange(nlabels):
			for j in xrange(i):
				cost = np.sum(np.abs(reflectances[i, :] - reflectances[j, :]))
				binary_costs[i, j] = cost
				binary_costs[j, i] = cost
		return binary_costs

	def get_features(self):
		mask_nz = self.input.mask_nz
		mask_nnz = self.input.mask_nnz
		features = np.zeros((mask_nnz, 5), dtype=np.float32)
		# intensity
		features[:, 0] = self.input.img_irg[mask_nz[0], mask_nz[1], 0] / self.params.theta_l
		# chromaticity
		features[:, 1] = self.input.img_irg[mask_nz[0], mask_nz[1], 1] / self.params.theta_c
		features[:, 2] = self.input.img_irg[mask_nz[0], mask_nz[1], 2] / self.params.theta_c
		# pixel location
		features[:, 3] = mask_nz[0] / (self.params.theta_p * self.input.diag)
		features[:, 4] = mask_nz[1] / (self.params.theta_p * self.input.diag)
		return features