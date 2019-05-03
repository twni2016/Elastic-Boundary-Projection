import os
import time
import numpy as np
from scipy.spatial import cKDTree
import argparse

PAD = 10
MAX_EUCLID = 10
MAX_MANHAT = int(np.ceil(np.sqrt(3) * MAX_EUCLID)) + 2
MAX_VALUE = MAX_MANHAT + 2
ITER_TH = 10
HEIGHT = 120
WIDTH = 120
STEP = 6 # for x,y axis
STEP_Z = 3 # due to small z_len
INIT_D = 5.0
CT_INF = -125
CT_SUP = 275
Y_SCALE = 2
IN_SLICE = 5

def arg_parser():
	parser = argparse.ArgumentParser(description='EBP')
	parser.add_argument('--data_path', type=str, default=None, help='data path')
	parser.add_argument('--organ_id', type=int, default=1, help='organ id')
	parser.add_argument('--slices', type=int, default=5, help='neighboring slices')
	parser.add_argument('--folds', type=int, default=1)
	args = parser.parse_args()
	print(args.data_path, args.organ_id, args.slices, args.folds)
	return args

class DATA_GENERATION():
	def __init__(self):
		''' prepare the dataset list (including training and testing cases) '''
		self.organ_id = args.organ_id
		self.slices = args.slices
		if self.slices % 2 == 0:
			raise ValueError('slices should be an odd number')
		self.images_path = os.path.join(args.data_path, 'images')
		self.labels_path = os.path.join(args.data_path, 'labels')
		self.dataset_name = 'dataset_organ' + str(self.organ_id)
		self.relabels_path = os.path.join(args.data_path, 'relabel', self.dataset_name)
		self.output_dataset_path = os.path.join(args.data_path, self.dataset_name)
		os.makedirs(self.relabels_path, exist_ok=True)
		os.makedirs(self.output_dataset_path, exist_ok=True)		
		self.images_list = os.listdir(self.images_path)
		self.labels_list = os.listdir(self.labels_path)
		self.images_list.sort()
		self.labels_list.sort()
		if len(self.images_list) != len(self.labels_list):
			raise RuntimeError('len of images should be equal to that of labels')
		self.direction_x, self.direction_y, self.direction_z = self.set_sphere_projection()
		self.pivot_stat = np.zeros((4),dtype=np.int64)

		start_all = time.time()
		for case_num in self.images_list:
			self.pivot_stat += self.generate(case_num)
		print('summary of pivot stat:', self.pivot_stat)
		print('#' * 20, 'all time costs %ds'%(time.time() - start_all))

	def generate(self, case_num):
		'''generate dataset per case
		'''
		self.case_num = case_num
		print('#' * 20, 'processing', self.case_num)
		self.image = np.load(os.path.join(self.images_path, case_num))
		self.label = np.load(os.path.join(self.labels_path, case_num))
		self.label[self.label != self.organ_id] = 0 # 0/1 label
		self.label[self.label == self.organ_id] = 1
		(self.height, self.width, self.depth) = self.label.shape
		self.case_path = os.path.join(self.output_dataset_path, self.case_num[:-4])
		os.makedirs(self.case_path, exist_ok=True)

		self.bbox = self.get_bbox()
		print(self.case_num, 'bounding box', self.bbox)
		self.man_dist = self.Manhattan_Dist()
		self.relabel = self.Euclidean_Dist()
		# self.relabel = np.load(os.path.join(self.relabels_path, self.case_num)) if you have generated relabel

		self.pivot_num = 0
		self.pivot_list = []
		self.iterate()
		return self.pivot_stats()

	def get_bbox(self):
		'''get 3D bounding box with padding'''
		non_zero = np.nonzero(self.label)
		return [max(0, non_zero[0].min() - PAD), min(self.height, non_zero[0].max() + PAD),
				max(0, non_zero[1].min() - PAD), min(self.width, non_zero[1].max() + PAD),
				max(0, non_zero[2].min() - PAD), min(self.depth, non_zero[2].max() + PAD)]

	def Manhattan_Dist(self):
		'''Step 1: Compute Manhattan Distance.'''
		start = time.time()
		(h, w, d) = self.label.shape
		# inside voxels
		dist1 = np.zeros(self.label.shape, dtype=np.int8)
		graph = - MAX_VALUE * self.label 
		mask = MAX_VALUE * (np.ones(self.label.shape, dtype=np.int8) * (graph < 0))
		for K in range(MAX_MANHAT):
			tmp1 = graph.copy()
			tmp1[0:h-1,:,:] = np.maximum(tmp1[0:h-1,:,:], tmp1[1:h,:,:])
			tmp1[1:h,:,:]   = np.maximum(tmp1[0:h-1,:,:], tmp1[1:h,:,:])
			tmp2 = graph.copy()
			tmp2[:,0:w-1,:] = np.maximum(tmp2[:,0:w-1,:], tmp2[:,1:w,:])
			tmp2[:,1:w,:]   = np.maximum(tmp2[:,0:w-1,:], tmp2[:,1:w,:]) 
			tmp3 = graph.copy()
			tmp3[:,:,0:d-1] = np.maximum(tmp3[:,:,0:d-1], tmp3[:,:,1:d])
			tmp3[:,:,1:d]   = np.maximum(tmp3[:,:,0:d-1], tmp3[:,:,1:d])
			graph = np.maximum.reduce([tmp1, tmp2, tmp3])
			graph = np.minimum(graph + 1, mask)
			dist1 = np.maximum(dist1, graph)
			mask = MAX_VALUE * (np.ones(self.label.shape, dtype=np.int8) * (graph < 0))

		# outside voxels
		dist2 = np.zeros(self.label.shape, dtype=np.int8)
		graph = - MAX_VALUE * (1 - self.label)
		mask = MAX_VALUE * (np.ones(self.label.shape, dtype=np.int8) * (graph < 0))
		for K in range(MAX_MANHAT):
			tmp1 = graph.copy()
			tmp1[0:h-1,:,:] = np.maximum(tmp1[0:h-1,:,:], tmp1[1:h,:,:])
			tmp1[1:h,:,:]   = np.maximum(tmp1[0:h-1,:,:], tmp1[1:h,:,:])
			tmp2 = graph.copy()
			tmp2[:,0:w-1,:] = np.maximum(tmp2[:,0:w-1,:], tmp2[:,1:w,:])
			tmp2[:,1:w,:]   = np.maximum(tmp2[:,0:w-1,:], tmp2[:,1:w,:]) 
			tmp3 = graph.copy()
			tmp3[:,:,0:d-1] = np.maximum(tmp3[:,:,0:d-1], tmp3[:,:,1:d])
			tmp3[:,:,1:d]   = np.maximum(tmp3[:,:,0:d-1], tmp3[:,:,1:d])
			graph = np.maximum.reduce([tmp1, tmp2, tmp3])
			graph = np.minimum(graph + 1, mask)
			dist2 = np.maximum(dist2, graph)
			mask = MAX_VALUE * (np.ones(self.label.shape, dtype=np.int8) * (graph < 0))

		man_dist = np.maximum(dist1, dist2) # merge dist
		man_dist = man_dist.astype(np.int8)
		print(self.case_num, 'Step 1: Compute Manhattan Distance. time elapsed %.2fs'%(time.time()-start))
		return man_dist

	def Euclidean_Dist(self):
		'''Step 2: use selected Manhantan voxels to compute Euclidean dist.'''
		start = time.time()
		label_sum = self.label 					 		  \
				  + np.roll(self.label, shift=1, axis=0)  \
				  + np.roll(self.label, shift=-1, axis=0) \
				  + np.roll(self.label, shift=1, axis=1)  \
				  + np.roll(self.label, shift=-1, axis=1) \
				  + np.roll(self.label, shift=1, axis=2)  \
				  + np.roll(self.label, shift=-1, axis=2)

		inner_surface = np.where((label_sum > 0) & (label_sum < 7) & (self.label == 1))
		outer_surface = np.where((label_sum > 0) & (label_sum < 7) & (self.label == 0))
		outer_selected = np.where((self.man_dist > 0) & (self.label == 0))
		inner_selected = np.where((self.man_dist > 0) & (self.label == 1))

		relabel = np.zeros(self.label.shape, dtype=np.float32)
		relabel[self.label == 1] = +MAX_EUCLID
		relabel[self.label == 0] = -MAX_EUCLID

		osel = np.asarray(outer_selected)
		osel = osel.transpose()
		osur = np.asarray(outer_surface)
		osur = osur.transpose()
		tree = cKDTree(osur)
		mindist, minid = tree.query(osel)
		relabel[outer_selected] = -mindist

		isel = np.asarray(inner_selected)
		isel = isel.transpose()
		isur = np.asarray(inner_surface)
		isur = isur.transpose()
		tree = cKDTree(isur)
		mindist, minid = tree.query(isel)
		relabel[inner_selected] = mindist

		relabel[outer_surface] = -1 / 2
		relabel[inner_surface] = +1 / 2
		relabel[relabel > MAX_EUCLID] = +MAX_EUCLID
		relabel[relabel < -MAX_EUCLID] = -MAX_EUCLID
		print(self.case_num, "Step 2: use selected Manhantan voxels to compute Euclidean dist. time elapsed %.2fs"%(time.time()-start))
		np.save(os.path.join(self.relabels_path, self.case_num), relabel)
		return relabel

	def iterate(self):
		'''Iteration entrance'''
		start = time.time()
		for i in range(self.bbox[0], self.bbox[1], STEP):
			for j in range(self.bbox[2], self.bbox[3], STEP):
				for k in range(self.bbox[4], self.bbox[5], STEP_Z):
					if self.relabel[i, j, k] > -MAX_EUCLID: # constraint
						self.store(*self.get_surface(a=i, b=j, c=k))
						self.pivot_num += 1
						self.pivot_list.append([i, j, k])
		print(self.case_num, '***** total iteration time elapsed %.4fs *****'%(time.time()-start))

	def get_surface(self, a, b, c):
		'''Given pivot (a,b,c), to get its final shell iteratively'''
		start_pivot = time.time()
		D = INIT_D * np.ones((ITER_TH, HEIGHT, WIDTH), dtype=np.float32)
		X = np.zeros((ITER_TH, self.slices + IN_SLICE, HEIGHT, WIDTH), dtype=np.float32)
		Y = np.zeros((ITER_TH, HEIGHT, WIDTH), dtype=np.float32)

		for iter in range(ITER_TH):
			start = time.time()
			for i in range(self.slices):
				U = i - ((self.slices - 1) // 2)
				X[iter, i] = self.interp3(self.image,
										  a + np.maximum(D[iter] + U, 0) * self.direction_x,
										  b + np.maximum(D[iter] + U, 0) * self.direction_y, 
										  c + np.maximum(D[iter] + U, 0) * self.direction_z)
			for i in range(0, IN_SLICE):
				U = (i + 1) * D[iter] / (IN_SLICE + 1)
				X[iter, self.slices + i] = self.interp3(self.image,
														a + U * self.direction_x,
														b + U * self.direction_y,
														c + U * self.direction_z)
			Y[iter] = self.interp3(self.relabel,
								   a + D[iter] * self.direction_x, 
								   b + D[iter] * self.direction_y, 
								   c + D[iter] * self.direction_z)                         
			np.clip(Y[iter], a_min=-Y_SCALE, a_max=Y_SCALE, out=Y[iter]) 

			norm_Y = np.linalg.norm(Y[iter])
			norm_D = np.linalg.norm(D[iter])
			print("Case %s Pivot point (%d,%d,%d), Iteration:%02d, norm_Y: %.2f, norm_D: %.2f, min/max D: %.2f/%.2f, time elapsed %.4fs" \
				%(self.case_num, a, b, c, iter, norm_Y, norm_D, D[iter].min(), D[iter].max(), time.time()-start))
			if iter + 1 >= ITER_TH:
				break
			D[iter + 1] = D[iter] + Y[iter]
			np.clip(D[iter + 1], a_min=0, a_max=None, out=D[iter + 1]) # for outer pivots

		print('----------- Case%s Pivot point (%d,%d,%d), relabel is %.4f, total time elapsed %.4fs ----------' \
			%(self.case_num, a, b, c, self.relabel[a,b,c], time.time() - start_pivot))
		return X, Y, D, self.relabel[a,b,c], a, b, c

	def store(self, X, Y, D, key, a, b, c):
		'''prepare dataset. normalization and reshape for training stage.
		must save per pivot, otherwise > 100G MEM. 
		(ITER_TH, slices + in_slice + 2, HEIGHT, WIDTH)
		'''
		np.clip(X, a_min=CT_INF, a_max=CT_SUP, out=X)
		X -= CT_INF
		X /= (CT_SUP - CT_INF)
		XYD_cat = np.concatenate((X, np.expand_dims(Y, 1), np.expand_dims(D, 1)), axis=1)
		XYD_cat = XYD_cat.astype(np.float32)
		if key >= INIT_D:
			kind = 'p0' # disjoint inner pivots
		elif key >= 0:
			kind = 'p1' # joint inner pivots
		elif key > - INIT_D:
			kind = 'p2' # joint outer pivots
		else:
			kind = 'p3' # disjoint outer pivots
		np.save(os.path.join(self.case_path, 'XYD' + str(a).zfill(3) + str(b).zfill(3) + str(c).zfill(3) \
			+ kind + '.npy'), XYD_cat)

	def pivot_stats(self):
		npy_list = os.listdir(self.case_path)
		npy_list.sort()
		pivot_stat = np.zeros((4), dtype=np.int64)
		for i in range(len(npy_list)):
			pivot_stat[int(npy_list[i][-5])] += 1
		print('pivot_stats', pivot_stat)
		return pivot_stat

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
		return direction_x, direction_y, direction_z

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

def save_lists():
	dataset_path = data_generation.output_dataset_path
	total_cases_list = os.listdir(dataset_path)
	total_cases_list.sort()
	case_len = len(total_cases_list)
	lists_path = os.path.join(args.data_path, 'lists', data_generation.dataset_name)
	os.makedirs(lists_path, exist_ok=True)
	current_fold = 0 # for MSD spleen dataset, only one fold.
	print('FD %d/%d:'%(current_fold, args.folds))

	file_name = os.path.join(lists_path, \
		'S' + str(args.slices) + 'FD' + str(args.folds) + str(current_fold) + 'valid.txt')
	output = open(file_name, 'a+')
	valid_cases_list = total_cases_list[:(case_len // 2)]
	print('valid cases list:', valid_cases_list)
	for case_num in valid_cases_list:
		npy_list = os.listdir(os.path.join(dataset_path, case_num))
		npy_list.sort()
		for npy in npy_list:
			output.write(os.path.join(case_num, npy) + '\n')
	output.close()

	file_name = os.path.join(lists_path, \
		'S' + str(args.slices) + 'FD' + str(args.folds) + str(current_fold) + 'train.txt')
	output = open(file_name, 'a+')
	train_cases_list = total_cases_list[(case_len // 2):]
	print('train cases list:', train_cases_list)
	for case_num in train_cases_list:
		npy_list = os.listdir(os.path.join(dataset_path, case_num))
		npy_list.sort()
		for npy in npy_list:
			output.write(os.path.join(case_num, npy) + '\n')
	output.close()

def mk_dir():
	os.makedirs(os.path.join(args.data_path, 'logs', data_generation.dataset_name), exist_ok=True)
	os.makedirs(os.path.join(args.data_path, 'models', data_generation.dataset_name), exist_ok=True)
	os.makedirs(os.path.join(args.data_path, 'results', data_generation.dataset_name), exist_ok=True)

if __name__ == '__main__':
	args = arg_parser()
	data_generation = DATA_GENERATION()
	mk_dir()
	save_lists()
