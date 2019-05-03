import torch
import torch.utils.data as data
from data_generation import *

# npy shape (ITER_TH, SLICES + IN_SLICE + 2, HEIGHT, WIDTH) 
# total samples ITER_TH * len(npy_list)
class EBP(data.Dataset):
	def __init__(self, train, data_path, folds, current_fold, organ_id, slices):
		self.train = train
		self.data_path = data_path
		self.folds = folds
		self.current_fold = current_fold
		self.slices = slices
		self.X_slices = self.slices + IN_SLICE
		self.dataset_name = 'dataset_organ' + str(organ_id)
		self.dataset_path = os.path.join(data_path, self.dataset_name)
		self.set_sphere_projection()

		if self.train:
			self.npy_list = open(os.path.join(data_path, 'lists', self.dataset_name, \
				'S' + str(slices) + 'FD' + str(folds) + str(current_fold) + 'train.txt'), 'r').read().splitlines()
		else:
			self.npy_list = open(os.path.join(data_path, 'lists', self.dataset_name, \
				'S' + str(slices) + 'FD' + str(folds) + str(current_fold) + 'valid.txt'), 'r').read().splitlines()	
		self.npy_list_len = len(self.npy_list)

	def __getitem__(self, index):
		''' return (slices + 3 + IN_SLICE + 3, H, W), (1, H, W), (p), (i) '''
		sample_pivot = index // ITER_TH
		sample_id = index % ITER_TH
		XYD_cat = np.load(os.path.join(self.dataset_path, \
			self.npy_list[sample_pivot]), mmap_mode='r')
		X = np.array(XYD_cat[sample_id, :self.slices])
		X_in = np.array(XYD_cat[sample_id, self.slices:self.X_slices])
		Y = np.array(XYD_cat[sample_id, self.X_slices])[np.newaxis, :]
		pivot_kind = torch.tensor([int(self.npy_list[sample_pivot][-5])])
		iter_kind = torch.tensor([int(sample_id)])
		return torch.cat((torch.from_numpy(X), self.direction, \
						  torch.from_numpy(X_in), self.direction)), \
			   torch.from_numpy(Y), pivot_kind, iter_kind

	def __len__(self):
		return ITER_TH * self.npy_list_len

	def set_sphere_projection(self):
		'''initialize the (x,y,z) unit sphere coordinate'''
		self.x = np.zeros((HEIGHT,WIDTH))
		self.y = np.zeros((HEIGHT,WIDTH))
		self.z = np.zeros((HEIGHT,WIDTH))
		self.p = np.arccos((2 * np.arange(1, HEIGHT+1) / (HEIGHT+1)) -1) 
		self.q = 2 * np.pi * np.arange(WIDTH) / WIDTH
		self.x = np.outer(np.sin(self.p), np.cos(self.q)) # col vector * row vector
		self.y = np.outer(np.sin(self.p), np.sin(self.q))
		self.z += np.cos(self.p)[:, np.newaxis] # col vector, horizontal broadcast
		self.direction_np = np.concatenate((self.x[np.newaxis, :], self.y[np.newaxis, :], self.z[np.newaxis, :]))
		self.direction_np = self.direction_np.astype(np.float32)
		self.direction = torch.from_numpy(self.direction_np)
