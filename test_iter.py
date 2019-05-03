from vnetg import Vnet
from data_generation import *

ITER_TH = 11  # may be tuned
POINT_STEP = 3
EPSILON = 0.00001
INTER_DSC_TH = 0.99

def arg_parser():
	parser = argparse.ArgumentParser(description='EBP')
	parser.add_argument('--data_path', type=str, default=None)
	parser.add_argument('--organ_id', type=int, default=1)
	parser.add_argument('--organ_name', type=str, default='sp') # MSD spleen
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--slices', type=int, default=5)
	parser.add_argument('-e', '--epoch', type=int, default=1, help='number of epochs to train')
	parser.add_argument('-t', '--timestamp', type=str, default=None, help='snapshot model')
	parser.add_argument('--folds', type=int, default=1)
	parser.add_argument('-f', '--current_fold', type=int, default=0)
	parser.add_argument('--train_cases', type=int, default=21)
	return parser.parse_args()

if __name__ == '__main__':
	args = arg_parser()
	os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)
	model = Vnet(slices=args.slices, inchans=[16,64,128], outchans=[64,128,256], down_layers=[3,3,3], up_layers=[3,3,3], dilations=[2,2,2])
	model = model.cuda()
	dataset_name = 'dataset_organ' + str(args.organ_id)
	model_name = 'vnetg_e' + str(args.epoch) + 'S' + str(args.slices) + 'FD' + str(args.folds) + str(args.current_fold) + '_' + args.timestamp
	model_path = os.path.join(args.data_path, 'models', dataset_name, model_name + '.pkl')
	model.load_state_dict(torch.load(model_path))
	model.eval()

class TEST():
	def __init__(self):
		self.organ_id = args.organ_id
		self.slices = args.slices
		self.images_path = os.path.join(args.data_path, 'test_images')
		self.labels_path = os.path.join(args.data_path, 'test_labels')
		self.results_path = os.path.join(args.data_path, 'results', args.organ_name, 
			'train' + str(args.train_cases) + '_' + model_name)
		os.makedirs(self.results_path, exist_ok=True)

		self.valid_cases_list = os.listdir(self.images_path)
		self.valid_cases_list.sort()
		self.direction_x, self.direction_y, self.direction_z = self.set_sphere_projection()

		self.DSC_valid_lists = []
		self.DSC99_P0, self.DSC99_P1, self.DSC99_P2, self.DSC99_P3 = [], [], [], []
		self.D_DSC_text = open(os.path.join(self.results_path, 'D_DSC.txt'), 'a+')
		
		start_all = time.time()
		print('Now starts test: ' + args.organ_name + ' id ' + str(args.organ_id) + ' has ' + str(len(self.valid_cases_list)) + ' cases ...')
		print('Using training model ' + model_name + ' from ' + str(args.train_cases) + ' cases')
		for case_idx in self.valid_cases_list:
			start = time.time()
			self.test_case(case_idx)
			print('Test %s end: time elapsed %ds'%(case_idx, time.time() - start))
		self.summary()
		print('total time costs: %ds'%(time.time() - start_all))
		
	def test_case(self, case_idx):
		'''Entrance'''
		self.case_idx = case_idx
		self.image = np.load(os.path.join(self.images_path, self.case_idx)).astype(np.float32)
		self.label = np.load(os.path.join(self.labels_path, self.case_idx)).astype(np.float32)
		self.label[self.label != self.organ_id] = 0
		self.label[self.label == self.organ_id] = 1
		self.relabel = np.load(os.path.join(args.data_path, 'relabel', dataset_name, self.case_idx))
		(self.height, self.width, self.depth) = self.image.shape

		print('Test Stage: %s begin!'%(self.case_idx))
		print('shape', self.image.shape, self.label.shape, self.relabel.shape)
		self.bbox = self.get_bbox()
		print(self.case_idx, 'bounding box', self.bbox)

		self.relabel_D = []
		self.model_D = []
		self.pivot_list = []
		self.PIVOT_POINTS = 0
		self.P0, self.P1, self.P2, self.P3 = [], [], [], []
		self.iterate()

		self.model_D = np.asarray(self.model_D)
		self.model_D = self.model_D.reshape(-1, ITER_TH, HEIGHT, WIDTH)
		self.model_D = self.model_D.astype(np.float32)
		self.relabel_D = np.asarray(self.relabel_D)
		self.relabel_D = self.relabel_D.astype(np.float32)
		self.relabel_D = self.relabel_D.reshape(-1, HEIGHT, WIDTH)
		self.pivot_list = np.asarray(self.pivot_list)
		self.selected_model_D = self.inter_DSC()
		self.binary_matrix = self.get_binary_matrix(self.selected_model_D)
		self.save_iteration_results()

	def summary(self):
		DSC_lists = np.asarray(self.DSC_valid_lists).mean(axis = 0)
		print('D_DSC Summary: for all pivots')
		print('all mean DSC', DSC_lists)
		print('P0 Inter_DSC 0.99:', np.mean(self.DSC99_P0))
		print('P1 Inter_DSC 0.99:', np.mean(self.DSC99_P1))
		print('P2 Inter_DSC 0.99:', np.mean(self.DSC99_P2))
		print('P3 Inter_DSC 0.99:', np.mean(self.DSC99_P3))
		self.D_DSC_text.write('D_DSC SUMMARY: \n')
		self.D_DSC_text.write('P0 Inter_DSC 0.99: ' + str(np.mean(self.DSC99_P0)) + '.\n')
		self.D_DSC_text.write('P1 Inter_DSC 0.99: ' + str(np.mean(self.DSC99_P1)) + '.\n')
		self.D_DSC_text.write('P2 Inter_DSC 0.99: ' + str(np.mean(self.DSC99_P2)) + '.\n')
		self.D_DSC_text.write('P3 Inter_DSC 0.99: ' + str(np.mean(self.DSC99_P3)) + '.\n')

	def set_sphere_projection(self):
		'''initialize the (x,y,z) unit sphere coordinate'''
		direction_x = np.zeros((HEIGHT,WIDTH))
		direction_y = np.zeros((HEIGHT,WIDTH))
		direction_z = np.zeros((HEIGHT,WIDTH))
		p = np.arccos((2 * np.arange(1, HEIGHT+1) / (HEIGHT+1)) -1) 
		q = 2 * np.pi * np.arange(WIDTH) / WIDTH
		direction_x = np.outer(np.sin(p), np.cos(q)) # col vector * row vector
		direction_y = np.outer(np.sin(p), np.sin(q))
		direction_z += np.cos(p)[:, np.newaxis] # col vector, horizontal broadcast
		self.direction = np.concatenate((direction_x[np.newaxis, :], direction_y[np.newaxis, :], direction_z[np.newaxis, :]))
		self.direction = torch.from_numpy(self.direction.astype(np.float32))
		return direction_x, direction_y, direction_z

	def get_bbox(self):
		'''get 3D bounding box with padding'''
		non_zero = np.nonzero(self.label)
		return [max(0, non_zero[0].min() - PAD), min(self.height, non_zero[0].max() + PAD), 
				max(0, non_zero[1].min() - PAD), min(self.width, non_zero[1].max() + PAD), 
				max(0, non_zero[2].min() - PAD), min(self.depth, non_zero[2].max() + PAD)]

	def iterate(self):
		for i in range(self.bbox[0], self.bbox[1], STEP):
			for j in range(self.bbox[2], self.bbox[3], STEP):
				for k in range(self.bbox[4], self.bbox[5], STEP_Z):
					print('relabel[%d,%d,%d] = %.2f'%(i, j, k, self.relabel[i,j,k]))
					self.get_surface_by_relabel(a=i, b=j, c=k)
					self.get_surface_by_model(a=i, b=j, c=k)
					self.pivot_list.append([i, j, k])
					self.update_P_stat(a=i, b=j, c=k)
					self.PIVOT_POINTS += 1

	def get_surface_by_relabel(self, a, b, c):
		'''ground-truth iteration'''
		D = INIT_D * np.ones((HEIGHT,WIDTH), dtype=np.float32)
		Y = np.zeros((HEIGHT,WIDTH), dtype=np.float32)

		iter = 0
		while True:
			start_iter = time.time()
			Y = self.interp3(self.relabel, a + D * self.direction_x, b + D * self.direction_y, c + D * self.direction_z)
			np.clip(Y, a_min=-Y_SCALE, a_max=Y_SCALE, out=Y) 

			D += Y
			np.clip(D, a_min=0, a_max=None, out=D) # avoid negative
			norm_Y = np.linalg.norm(Y)
			if self.PIVOT_POINTS % 50 == 0 or self.relabel[a,b,c] > -INIT_D:
				print('RELABEL', "Pivot point (%d,%d,%d), Iteration:%02d, normY is %.2f, min/mean/max/non0 D is %.2f/%.2f/%.2f/%d," \
					"time elapsed %.4fs"% (a, b, c, iter, norm_Y, D.min(), D.mean(), D.max(), np.count_nonzero(D), time.time()-start_iter))

			iter += 1
			if iter >= ITER_TH:  # we need the last D.
				self.relabel_D.append(D.copy())
				break
	
	def get_surface_by_model(self, a, b, c):
		'''model predicted iteration'''
		D = INIT_D * np.ones((HEIGHT, WIDTH), dtype=np.float32)
		X = np.zeros((self.slices + IN_SLICE, HEIGHT, WIDTH), dtype=np.float32)

		iter = 0
		while True:
			start_iter = time.time()
			for i in range(self.slices):
				U = i - ((self.slices - 1) // 2)
				X[i] = self.interp3(self.image,
									a + np.maximum(D + U, 0) * self.direction_x,
									b + np.maximum(D + U, 0) * self.direction_y,
									c + np.maximum(D + U, 0) * self.direction_z)
			for i in range(IN_SLICE):
				U = (i + 1) * D / (IN_SLICE + 1)
				X[self.slices + i] = self.interp3(self.image,
												  a + U * self.direction_x,
												  b + U * self.direction_y,
												  c + U * self.direction_z)
			np.clip(X, a_min=CT_INF, a_max=CT_SUP, out=X) # first interp, then normalize
			X += (-CT_INF)
			X /= (CT_SUP - CT_INF)
			Y = model((torch.cat((torch.from_numpy(X[:self.slices]), self.direction,
								  torch.from_numpy(X[self.slices:]), self.direction)) \
					   .view(1,self.slices + IN_SLICE + 6,HEIGHT,WIDTH)).cuda().float())
			Y = Y.view(HEIGHT,WIDTH).data.cpu().numpy()
			self.model_D.append(D.copy()) # pay attention
	
			D += Y
			np.clip(D, a_min=0, a_max=None, out=D) # avoid negative
			norm_Y = np.linalg.norm(Y) # model's prediction of Y, not ground truth
			if self.PIVOT_POINTS % 50 == 0 or self.relabel[a,b,c] > -INIT_D:
				print('MODEL', "Pivot point (%d,%d,%d), Iteration:%02d, normY is %.2f, min/mean/max/non0 D is %.2f/%.2f/%.2f/%d," \
					"time elapsed %.4fs"% (a, b, c, iter, norm_Y, D.min(), D.mean(), D.max(), np.count_nonzero(D), time.time()-start_iter))
			iter += 1
			if iter >= ITER_TH:
				break

	def interp3(self, Data, a, b, c): 
		'''3D interpolation'''
		floor_a = np.floor(a)
		floor_b = np.floor(b)
		floor_c = np.floor(c)
		np.clip(floor_a, a_min=0, a_max=self.height-2, out=floor_a)
		np.clip(floor_b, a_min=0, a_max=self.width-2, out=floor_b)
		np.clip(floor_c, a_min=0, a_max=self.depth-2, out=floor_c)
		floor_a = floor_a.astype(np.uint16)
		floor_b = floor_b.astype(np.uint16)
		floor_c = floor_c.astype(np.uint16)
		da = a - floor_a
		db = b - floor_b
		dc = c - floor_c
		return (1-da) * ((1-db) * (Data[floor_a,floor_b,floor_c] * (1-dc) + \
								Data[floor_a,floor_b,floor_c+1] * (dc)) + \
						(db)   * (Data[floor_a,floor_b+1,floor_c] * (1-dc) + \
								Data[floor_a,floor_b+1,floor_c+1] * (dc))) + \
				(da)  * ((1-db) * (Data[floor_a+1,floor_b,floor_c] * (1-dc) + \
								Data[floor_a+1,floor_b,floor_c+1] * (dc)) + \
						(db)   * (Data[floor_a+1,floor_b+1,floor_c] * (1-dc) + \
								Data[floor_a+1,floor_b+1,floor_c+1] * (dc)))

	def D_DSC(self, D1, D2):
		''' DSC metric applied to D, a 3D shell '''
		return (2 * (np.minimum(D1, D2) ** 3).sum() + EPSILON) \
										/ ((D1 ** 3 + D2 ** 3).sum() + EPSILON)
	def update_P_stat(self, a, b, c):
		value = self.relabel[a,b,c]
		if value >= INIT_D:
			self.P0.append(self.PIVOT_POINTS)
		elif value > 0:
			self.P1.append(self.PIVOT_POINTS)
		elif value > - INIT_D:
			self.P2.append(self.PIVOT_POINTS)
		else:
			self.P3.append(self.PIVOT_POINTS)
			
	def inter_DSC(self):
		''' STAT: 
		(1) DSC between groud-truth D and i-th predicted D in each pivot
		(2) DSC between (i-1)-th predicted D and i-th predicted D in each pivot
		'''
		DSC_list = np.zeros((self.PIVOT_POINTS, ITER_TH))
		selected_model_D = np.zeros((self.PIVOT_POINTS, HEIGHT, WIDTH))
		INTER_D_DSC = np.zeros((6, self.PIVOT_POINTS)) 
		# first dimension means inter_DSC in [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]
		for p in range(self.PIVOT_POINTS):
			for i in range(ITER_TH):
				DSC_list[p,i] = self.D_DSC(self.relabel_D[p], self.model_D[p,i])
				if i == 0:
					inter_DSC = 0
				else:
					inter_DSC = self.D_DSC(self.model_D[p,i-1], self.model_D[p,i])
				
				for idx, thres in enumerate([0.90, 0.93, 0.95, 0.97, 0.98, 0.99]):
					if INTER_D_DSC[idx, p] == 0 and (i == ITER_TH - 1 or inter_DSC >= thres):
						INTER_D_DSC[idx, p] = DSC_list[p,i]

				if not selected_model_D[p].any() and (i == ITER_TH - 1 or inter_DSC >= INTER_DSC_TH):
					selected_model_D[p] = self.model_D[p,i]

		print(self.case_idx, 'all pivots')
		print('mean_D_DSC', DSC_list.mean(axis=0))
		self.D_DSC_text.write('*' * 10 + ' ' + self.case_idx + '\n')
		for idx, thres in enumerate([0.90, 0.93, 0.95, 0.97, 0.98, 0.99]):
			print('Inter_D_DSC ' + str(thres), INTER_D_DSC[idx].mean())
			self.D_DSC_text.write('Inter_D_DSC ' + str(thres) + ' ' + str(INTER_D_DSC[idx].mean()) + '\n')
		self.DSC_valid_lists.append(DSC_list.mean(axis=0))

		# P0, P1, P2, P3. D DSC 0.99 stats:
		print('*' * 10, 'P0,P1,P2,P3 summary of Inter DSC 0.99')
		print('P0: Inter_D_DSC 0.99:', INTER_D_DSC[-1, self.P0].mean())
		print('P1: Inter_D_DSC 0.99:', INTER_D_DSC[-1, self.P1].mean())
		print('P2: Inter_D_DSC 0.99:', INTER_D_DSC[-1, self.P2].mean())
		print('P3: Inter_D_DSC 0.99:', INTER_D_DSC[-1, self.P3].mean())
		self.DSC99_P0.append(INTER_D_DSC[-1, self.P0].mean())
		self.DSC99_P1.append(INTER_D_DSC[-1, self.P1].mean())
		self.DSC99_P2.append(INTER_D_DSC[-1, self.P2].mean())
		self.DSC99_P3.append(INTER_D_DSC[-1, self.P3].mean())
		self.D_DSC_text.write('*' * 10 + ' P0,P1,P2,P3 summary of Inter DSC\n')
		self.D_DSC_text.write('P0: Inter_D_DSC 0.99: ' + str(INTER_D_DSC[-1, self.P0].mean()) + '\n')
		self.D_DSC_text.write('P1: Inter_D_DSC 0.99: ' + str(INTER_D_DSC[-1, self.P1].mean()) + '\n')
		self.D_DSC_text.write('P2: Inter_D_DSC 0.99: ' + str(INTER_D_DSC[-1, self.P2].mean()) + '\n')
		self.D_DSC_text.write('P3: Inter_D_DSC 0.99: ' + str(INTER_D_DSC[-1, self.P3].mean()) + '\n\n\n')
		self.D_DSC_text.flush()
		
		return selected_model_D
			
	def save_iteration_results(self):
		np.save(os.path.join(self.results_path, 
			self.case_idx[:-4] + '_selected_model_D.npy'), self.selected_model_D.astype(np.float32))
		np.save(os.path.join(self.results_path, 
			self.case_idx[:-4] + '_relabel_D.npy'), self.relabel_D)
		np.save(os.path.join(self.results_path, 
			self.case_idx[:-4] + '_pivot_list.npy'), self.pivot_list)
		np.save(os.path.join(self.results_path, 
			self.case_idx[:-4] + '_binary_matrix.npy'), self.binary_matrix)
		print('save iteration results successfully!')

	def set_DSC(self, s1, s2):
		return 2 * len(s1 & s2) / (len(s1) + len(s2) + EPSILON) # DSC([], []) is 0

	def nearest_spherical_dir(self, vec):
		if not vec.any():
			return -1, -1
		inner_prod = self.direction_x * vec[0] + self.direction_y * vec[1] + self.direction_z * vec[2] 
		max_dir = np.argmax(inner_prod.reshape(-1)) # no abs
		return max_dir // WIDTH, max_dir % WIDTH  # h, w

	def get_binary_matrix(self, D):
		'''prepare a relation matrix between the shells of any pivot pair.
		We sample a dense set of points in ROI and then count how many points are within the shells
		to build a set of inner points for each pivot.
		Then the relation is defined as DSC (IoU) between the sets of any pivot pair.
		'''
		# sample points
		start = time.time()
		point_list = []
		for i in range(self.bbox[0], self.bbox[1], POINT_STEP):
			for j in range(self.bbox[2], self.bbox[3], POINT_STEP):
				for k in range(self.bbox[4], self.bbox[5], POINT_STEP):
					point_list.append([i, j, k])
		point_list = np.asarray(point_list)

		# count inner points for each pivot
		in_shell_list = []
		start = time.time()
		for p in range(self.PIVOT_POINTS):
			in_pivot_set = set()
			rad = min(30, D[p].max()) # diameter 60 pixels. I think that's enough
			if rad == 0:
				in_shell_list.append(in_pivot_set)
				continue
			near_pivot_point_list = np.where(np.linalg.norm(point_list - self.pivot_list[p], axis=1) < rad)[0]
			for id in near_pivot_point_list:
				vec = point_list[id] - self.pivot_list[p]
				h, w = self.nearest_spherical_dir(vec)
				dist = np.linalg.norm(vec)
				if h == -1 or D[p, h, w] > dist:
					in_pivot_set.add(id)
			in_shell_list.append(in_pivot_set)
		print('in_shell_list done', time.time() - start)

		# compute set DSC for any two pivots
		start = time.time()
		assert self.PIVOT_POINTS == len(in_shell_list)
		binary_matrix = np.zeros((self.PIVOT_POINTS, self.PIVOT_POINTS), dtype=np.float32)
		for p in range(self.PIVOT_POINTS):
			for q in range(p + 1, self.PIVOT_POINTS):
				binary_matrix[p, q] = self.set_DSC(in_shell_list[p], in_shell_list[q])
		binary_matrix += np.swapaxes(binary_matrix, 0, 1) # copy upper tri to lower
		print('binary matrix done', time.time() - start)
		return binary_matrix

if __name__ == '__main__':
	test = TEST()
