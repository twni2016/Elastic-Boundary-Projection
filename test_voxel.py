from queue import *
from test_iter import *
from test_util import *

# for find_target_component(). may be tuned
DSC_TH = 0.60
TARGET_RATIO = 0.10
# for remove_by_D_value()
NONZERO_TH = 10000 
MEAN_TH = 2.0

def set_sphere_projection():
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

def bbox_info():
	'''get 3D bounding box with padding, compute lower bound of target volume, and border pivots'''
	non_zero = np.nonzero(label)
	bbox = [max(0, non_zero[0].min() - PAD), min(height, non_zero[0].max() + PAD),
			max(0, non_zero[1].min() - PAD), min(width, non_zero[1].max() + PAD),
			max(0, non_zero[2].min() - PAD), min(depth, non_zero[2].max() + PAD)]
	
	bbox_volume = 1
	bbox_border_pivots = set()
	for i in range(3):
		bbox_volume *= (bbox[2 * i + 1] - bbox[2 * i])
		bbox_border_pivots |= set(np.where(pivot_list[:, i] == bbox[2 * i])[0])
		if i < 2:
			oppo_border = ((bbox[2 * i + 1] - bbox[2 * i]) // STEP) * STEP + bbox[2 * i]
		else:
			oppo_border = ((bbox[2 * i + 1] - bbox[2 * i]) // STEP_Z) * STEP_Z + bbox[2 * i]
		bbox_border_pivots |= set(np.where(pivot_list[:, i] == oppo_border)[0])	

	return bbox, (bbox_volume / (STEP * STEP * STEP_Z)) * TARGET_RATIO, bbox_border_pivots

def find_idx(pivot):
	idx = np.nonzero((pivot_list == np.array(pivot)).all(axis=1))[0]
	if len(idx) == 0:
		return -1
	return idx[0]

def BFS_by_neighbor(binary_matrix, DSC_TH):
	'''BFS on pivot node graph (simple 6-neighbor Euclidean graph) 
	where edge weights are given by binary matrix,
	to find the all valid components where weights are over DSC_TH. '''
	neigh = np.array([[0,0,0,0,STEP,-STEP], 
					  [0,0,STEP,-STEP,0,0], 
					  [STEP_Z,-STEP_Z,0,0,0,0]]).T
	marked = np.ones((PIVOT_POINTS), dtype=np.bool)
	marked[binary_matrix.sum(axis=0) > DSC_TH] = False
	components = []
	head = np.argmax(marked == False)

	while True:
		q = Queue()
		q.put(head)
		connection = set()
		connection.add(head)
		marked[head] = True
		while (not q.empty()):
			p = q.get()
			for k in range(6):
				pair = pivot_list[p] + neigh[k]
				pair_id = find_idx(pair)
				if pair_id >= 0 and \
					binary_matrix[p, pair_id] > DSC_TH and marked[pair_id] == False:
					connection.add(pair_id)
					q.put(pair_id)
					marked[pair_id] = True
		components.append(connection)
		rest = np.where(marked == False)[0]
		if rest.shape[0] == 0:
			break
		head = rest[0]
	print('BFS by 6 neignbor to get components done')
	return components

def find_target_component(binary_matrix, DSC_TH):
	''' return the component candidate, which satisfies:
	(1) size is over a lower bound (2) has the minimal border pivots. '''
	components = BFS_by_neighbor(binary_matrix, DSC_TH)
	id_list = []
	border_num_list = []
	candidates = 0
	for id, comp in enumerate(components):
		if len(comp) > PROPER_SIZE / 3:
			candidates += 1
			id_list.append(id)
			border_num_list.append(len(comp & bbox_border_pivots))
	if candidates == 0:
		print('DSC TH is too large')
		return -1, None
	border_num_array = np.asarray(border_num_list)
	target_id = id_list[np.argmin(border_num_array)]
	return components[target_id]

def remove_by_D_value(pred_pivot, D):
	'''trim the component by D. A minor step. '''
	nonzero_cnt = np.count_nonzero(D, axis=(1,2))
	nonzero_list = [id for id in range(PIVOT_POINTS) if nonzero_cnt[id] >= NONZERO_TH]
	mean_cnt = np.mean(D, axis=(1,2))
	mean_list = [id for id in range(PIVOT_POINTS) if mean_cnt[id] >= MEAN_TH]
	intersect_list = list(set(nonzero_list) & set(mean_list) & pred_pivot)
	intersect_list.sort()
	print('*'*10, 'total pivots:', PIVOT_POINTS, 'after filtering')
	print('non_zero', len(nonzero_list), 'mean', len(mean_list), 'pred_pivot_by_parts', len(pred_pivot))
	print('intersect', len(intersect_list))
	return intersect_list

def voxelize(pivot_id_list, D):
	''' Voxelize the predicted point cloud. Call test_util.post_processing '''
	pred_point_cloud = [[], [], []]
	for p in pivot_id_list:
		pred_point_cloud[0].append(pivot_list[p,0] + (D[p] * direction_x).reshape(-1))
		pred_point_cloud[1].append(pivot_list[p,1] + (D[p] * direction_y).reshape(-1))
		pred_point_cloud[2].append(pivot_list[p,2] + (D[p] * direction_z).reshape(-1))
	for i in range(3):
		pred_point_cloud[i] = np.asarray(pred_point_cloud[i]).reshape(-1)
	pred_point_cloud = np.asarray(pred_point_cloud) # 3 * n
	np.save(os.path.join(voxel_path, case_idx[:-4] + '_pred_point_cloud.npy'), pred_point_cloud)
	return post_processing(pred_point_cloud, label, bbox, voxel_path, case_idx, final_DSC_text)

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
	direction_x, direction_y, direction_z = set_sphere_projection()

	model_name = 'vnetg_e' + str(args.epoch) + 'S' + str(args.slices) + 'FD' + str(args.folds) + str(args.current_fold) + '_' + args.timestamp
	results_path = os.path.join(args.data_path, 'results', args.organ_name, 'train' + str(args.train_cases) + '_' + model_name)
	test_cases_path = os.path.join(args.data_path, 'test_labels')
	test_cases_list = os.listdir(test_cases_path)
	test_cases_list.sort()

	voxel_path = os.path.join(results_path, 
		'components_NONZERO' + str(NONZERO_TH) + ',MEAN' + str(MEAN_TH))
	os.makedirs(voxel_path, exist_ok=True)
	final_DSC_text = open(os.path.join(voxel_path, 'results.txt'), 'a+')
	final_DSC_array = np.zeros((len(test_cases_list)))

	start_all = time.time()
	for id, case_idx in enumerate(test_cases_list):
		start = time.time()
		print('#' * 10, 'EBP final part:', 'organ id', args.organ_id, 'based on', results_path)
		print('#' * 10, case_idx, 'starts!')
		label = np.load(os.path.join(test_cases_path, case_idx))
		label[label != args.organ_id] = 0
		label[label == args.organ_id] = 1
		label = label.astype(np.bool)
		(height, width, depth) = label.shape

		relabel = np.load(os.path.join(args.data_path,
			'relabel', 'dataset_organ' + str(args.organ_id), case_idx))
		pivot_list = np.load(os.path.join(results_path,
			case_idx[:-4] + '_pivot_list.npy'))
		selected_model_D = np.load(os.path.join(results_path,
			case_idx[:-4] + '_selected_model_D.npy'))
		binary_matrix = np.load(os.path.join(results_path,
			case_idx[:-4] + '_binary_matrix.npy'))

		PIVOT_POINTS = len(pivot_list)
		bbox, PROPER_SIZE, bbox_border_pivots = bbox_info()
		pred_pivot = find_target_component(binary_matrix, DSC_TH)
		intersect_list = remove_by_D_value(pred_pivot, selected_model_D)
		final_DSC_array[id] = voxelize(intersect_list, selected_model_D)
		print(case_idx, 'ends. time costs: %dmin\n'%((time.time() - start) / 60))

	np.save(os.path.join(voxel_path, 'FINAL_DSC_ARRAY.npy'), final_DSC_array)
	DSC_mean, DSC_std = final_DSC_array.mean(), final_DSC_array.std()
	print('\n\n', '!' * 10, 'FINAL DSC RESULT FOR EBP PROJECT')
	print('DSC_mean:', DSC_mean, 'DSC_std', DSC_std)
	final_DSC_text.write('DSC mean = ' + str(DSC_mean) + 'and std = ' + str(DSC_std) + '\n')
	print('TOTAL TIME COSTS %dmin' % ((time.time() - start_all) / 60))
