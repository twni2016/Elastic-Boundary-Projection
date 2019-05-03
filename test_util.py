import os
import time
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.spatial import Delaunay, distance
from scipy.ndimage.morphology import binary_fill_holes
import trimesh
import fast_functions as ff

# for KDE_tri(). may be tuned
KDE_bandwidth = 1.0
KDE_log_prob_th = -14
# for mesh3d(). may be tuned
ALPHA = 5

def KDE_tri(pred_point_cloud, bbox, text):
	''' use KDE to filter outliers in predict point cloud '''
	(h0, h1, w0, w1, d0, d1) = bbox
	X,Y,Z = np.mgrid[h0:h1, w0:w1, d0:d1]
	positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

	# 1. KDE
	start = time.time()
	kde = KernelDensity(kernel='epanechnikov', bandwidth=KDE_bandwidth).fit(pred_point_cloud.T)
	score = kde.score_samples(positions.T)
	score = score.reshape(h1-h0, w1-w0, d1-d0)
	filtered_pred_point_cloud = np.where(score > KDE_log_prob_th)
	points_list = [filtered_pred_point_cloud[0] + h0, 
				   filtered_pred_point_cloud[1] + w0, 
				   filtered_pred_point_cloud[2] + d0]
	print('KDE filter done', time.time() - start)
	print('filtered_pred_point_cloud (', filtered_pred_point_cloud[0].shape[0], '* 3 )')
	text.write('filtered_pred_point_cloud: ' + str(filtered_pred_point_cloud[0].shape[0]) + ' * 3 \n')
	text.flush()

	# 2. Delaunay triangulation
	start = time.time()
	points = np.asarray(points_list).T
	tri = Delaunay(points)
	print('Delaunay triangulation done', time.time() - start)
	return points, tri

def mesh3d(points, tri, alpha, label, text):
	'''Obtain alpha shape, voxelize and fill holes '''
	# 3. alpha shape
	start = time.time()
	alpha_complex = np.asarray(list(
		filter(lambda simplex: 0 < circumsphere(points, simplex) < alpha, tri.simplices)))
	print(alpha, 'alpha shape done')

	# 4. voxelize the mesh. most time consuming and compute DSC
	tri_faces = [[tetra[[0,1,2]], tetra[[0,1,3]], tetra[[0,2,3]], tetra[[1,2,3]]] 
					for tetra in alpha_complex]
	tri_faces = np.asarray(tri_faces).reshape(-1, 3)
	mesh = trimesh.base.Trimesh(vertices = points, faces = tri_faces)
	voxel_mesh = mesh.voxelized(pitch = 1) # multi-CPU and huge CPU memory
	print('voxel_mesh.matrix_solid.shape', voxel_mesh.matrix_solid.shape)
	print('voxel_mesh.origin', voxel_mesh.origin)

	pred = np.zeros(label.shape, dtype = np.bool)
	pred[voxel_mesh.origin[0]:voxel_mesh.origin[0]+voxel_mesh.shape[0], 
		 voxel_mesh.origin[1]:voxel_mesh.origin[1]+voxel_mesh.shape[1], 
		 voxel_mesh.origin[2]:voxel_mesh.origin[2]+voxel_mesh.shape[2]] = voxel_mesh.matrix_solid
	DSC, inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
	print('DSC', DSC, inter_sum, pred_sum, label_sum)
	text.write('Initial DSC: ' + \
		str(DSC) + ' = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + str(label_sum) + ')\n')
	
	# 5. fill the holes
	filled_pred = binary_fill_holes(pred)
	DSC, inter_sum, pred_sum, label_sum = DSC_computation(label, filled_pred)
	print('After fill holes', DSC, inter_sum, pred_sum, label_sum)
	print('time cost', time.time() - start)
	text.write('After filling holes, DSC: ' + \
		str(DSC) + ' = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + str(label_sum) + ')\n')
	return filled_pred

def post_processing(pred_point_cloud, label, bbox, voxel_path, case_idx, text):
	''' Entrance for test_util.py. Return final DSC.
	Given the predicted volume, find the largest component and trim it.'''
	text.write('#' * 10 + ' ' + case_idx + ' TESTING \n')
	points, tri = KDE_tri(pred_point_cloud, bbox, text)
	pred = mesh3d(points, tri, ALPHA, label, text)

	# 6. find the largest component
	pred = largest_component(pred)
	DSC, inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
	print('After keeping the largest component', DSC, inter_sum, pred_sum, label_sum)
	text.write('After keeping the largest component, DSC : ' + \
		str(DSC) + ' = 2 * ' + str(inter_sum) + ' / ' + str(pred_sum) + ' + ' + str(label_sum) + '\n')
	np.save(os.path.join(voxel_path, case_idx[:-4] + '_pred.npy'), pred.astype(np.uint8))
	
	# 7. delete one piece of boundary
	delete_boundary(pred) # it can be done multiple times.
	DSC, inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
	print('Delete boundary', DSC, inter_sum, pred_sum, label_sum)
	text.write('Delete boundary, DSC: ' + \
		str(DSC) + ' = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + str(label_sum) + ')\n')

	return DSC

def circumsphere(points, simplex):
	''' return the circumradius of a tetrahedron. 
	The beautiful formula: Area of (aa1, bb1, cc1) = 6 * V * R
	Proof: https://cms.math.ca/crux/v27/n4/page246-247.pdf'''
	coord_matrix = np.ones((4,4))
	for i in range(4):
		coord_matrix[i, :3] = points[simplex[i]]
	Volume_6 = np.abs(np.linalg.det(coord_matrix))
	if Volume_6 == 0:
		return 0
	dist_matrix = distance.cdist(coord_matrix[:, :3], coord_matrix[:, :3])
	aa1 = dist_matrix[0,1] * dist_matrix[2,3]
	bb1 = dist_matrix[0,2] * dist_matrix[1,3]
	cc1 = dist_matrix[0,3] * dist_matrix[1,2]
	tri_peri = (aa1 + bb1 + cc1) / 2
	if tri_peri < max(aa1, bb1, cc1):
		return 0
	tri_area = np.sqrt(tri_peri * (tri_peri - aa1) * (tri_peri - bb1) * (tri_peri - cc1))
	return tri_area / Volume_6

def DSC_computation(label, pred):
	pred_sum = pred.sum()
	label_sum = label.sum()
	inter_sum = np.logical_and(pred, label).sum()
	return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum

def delete_boundary(voxel):
	voxel_sum = np.zeros(voxel.shape, dtype = np.int32)
	voxel_sum = voxel \ 				 		
			  + np.roll(voxel, shift=1, axis=0) \
			  + np.roll(voxel, shift=-1, axis=0) \
			  + np.roll(voxel, shift=1, axis=1) \
			  + np.roll(voxel, shift=-1, axis=1) \
			  + np.roll(voxel, shift=1, axis=2) \
			  + np.roll(voxel, shift=-1, axis=2)
	boundary_list = np.where((voxel_sum > 0) & (voxel_sum < 7) & (voxel == 1))
	voxel[boundary_list] = 0

def largest_component(voxel):
	''' Call ff '''
	voxel = voxel.astype(np.uint8)
	ff.post_processing(voxel, voxel, 1, False)
	return voxel.astype(np.int32)
