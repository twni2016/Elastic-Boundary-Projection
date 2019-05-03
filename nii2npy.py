import os
import numpy as np
import nibabel as nib
import shutil
import argparse

parser = argparse.ArgumentParser(description='EBP')
parser.add_argument('--data_path', type=str, default=None, help='data path')
data_path = args.data_path
images_list = os.listdir(os.path.join(data_path, 'imagesTr'))
labels_list = os.listdir(os.path.join(data_path, 'labelsTr'))
assert len(images_list) == len(labels_list) == 41
images_list.sort()
labels_list.sort()
if not os.path.exists(os.path.join(data_path, 'images')):
	os.makedirs(os.path.join(data_path, 'images'))
if not os.path.exists(os.path.join(data_path, 'labels')):
	os.makedirs(os.path.join(data_path, 'labels'))

for img_name in images_list:
	print(img_name)
	img = nib.load(os.path.join(data_path, 'imagesTr', img_name))
	img_array = img.get_fdata()
	img_id = img_name[7:9]
	if img_id[1] == '.':
		img_id = img_id[0]
	np.save(os.path.join(data_path, 'images', 'MD' + img_id.zfill(2) + '.npy'), img_array.astype(np.float32))

for lbl_name in labels_list:
	print(lbl_name)
	lbl = nib.load(os.path.join(data_path, 'labelsTr', lbl_name))
	lbl_array = lbl.get_data()
	lbl_id = lbl_name[7:9]
	if lbl_id[1] == '.':
		lbl_id = lbl_id[0]
	np.save(os.path.join(data_path, 'labels', 'MD' + lbl_id.zfill(2) + '.npy'), lbl_array.astype(np.uint8))

# half split the dataset.
image_list = os.listdir(os.path.join(data_path, 'images'))
image_list.sort()
os.makedirs(os.path.join(data_path, 'test_images'))
os.makedirs(os.path.join(data_path, 'test_labels'))
os.makedirs(os.path.join(data_path, 'train_images'))
os.makedirs(os.path.join(data_path, 'train_labels'))

for i, npy in enumerate(image_list):
	print(i, npy)
	if i < 20:
		shutil.copy2(os.path.join(data_path, 'images', npy), os.path.join(data_path, 'test_images', npy))
		shutil.copy2(os.path.join(data_path, 'labels', npy), os.path.join(data_path, 'test_labels', npy))
	else:
		shutil.copy2(os.path.join(data_path, 'images', npy), os.path.join(data_path, 'train_images', npy))
		shutil.copy2(os.path.join(data_path, 'labels', npy), os.path.join(data_path, 'train_labels', npy))
