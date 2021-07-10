import copy
import json
import random
import hashlib
import numpy as np


class HyperParameters():

    def __init__(self):

        # Iteration parameters
        self.n_iters = 10
        self.n_crf_iters = 10

        # Kmeans clustering parameters
        self.fixed_seed = False
        self.split_clusters = True
        self.kmeans_intensity_scale = 0.5
        self.kmeans_n_clusters = 20
        self.kmeans_max_samples = 2000000

        # Reflectance and shading unary loss parameters
        self.abs_reflectance_weight = 0
        self.abs_shading_weight = 500.0
        self.abs_shading_gray_point = 0.5
        self.abs_shading_log = True
        self.chromaticity_weight = 0
        self.shading_norm = "L2"
        self.shading_target_weight = 20000.0
        self.shading_chromaticity = False

        # Pairwise costs parameters
        self.chromaticity_norm = "L2"
        self.pairwise_intensity_log = True
        self.pairwise_intensity_chromaticity = True
        self.pairwise_weight = 10000.0

        # Standard deviation: pairwise pixels, intensity, chromaticity
        # Required while extracting features to send to DenseCRF.
        self.theta_p = 0.1
        self.theta_l = 0.1
        self.theta_c = 0.025