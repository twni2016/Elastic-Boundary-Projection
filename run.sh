# Elastic Boundary Projection for 3D Medical Image Segmentation, CVPR 2019 
# Author: Tianwei Ni.

# turn on these switches to execute each module
ENABLE_DATA_DOWNLOAD=0
ENABLE_DATA_GENERATION=0
ENABLE_TRAINING=0
ENABLE_TESTING=0

DATA_PATH='/mnt/data0/tianwei/EBP_MD_Spleen/'
GPU_ID=0
CURRENT_FOLD=0
FOLDS=1
ORGAN_ID=1
SLICES=5

if [ "$ENABLE_DATA_DOWNLOAD" = "1" ]; then
	python nii2npy.py --data_path $DATA_PATH
fi

# data generation
if [ "$ENABLE_DATA_GENERATION" = "1" ]; then
	GENERATION_TIMESTAMP=$(date +'%Y%m%d')
	GENERATION_LOG=${DATA_PATH}logs/GENERATION_FD${FOLDS}_${GENERATION_TIMESTAMP}.txt
	python -u data_generation.py \
		--data_path $DATA_PATH --organ_id $ORGAN_ID --slices $SLICES --folds $FOLDS \
		2>&1 | tee $GENERATION_LOG
fi

# training stage
BATCH=32
EPOCH=5
LR=0.001

if [ "$ENABLE_TRAINING" = "1" ]; then
	TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
	TRAINING_LOG=${DATA_PATH}logs/S${SLICES}FD${FOLDS}${CURRENT_FOLD}_vnetg_${TRAINING_TIMESTAMP}.txt
	python -u vnetg.py \
		--data_path $DATA_PATH --slices $SLICES --organ_id $ORGAN_ID --folds $FOLDS -f $CURRENT_FOLD \
		-b $BATCH -e $EPOCH --lr $LR --gpu_id $GPU_ID -t $TRAINING_TIMESTAMP \
		2>&1 | tee $TRAINING_LOG
else
	TRAINING_TIMESTAMP=_
fi

# testing stage
if [ "$ENABLE_TESTING" = "1" ]; then
	python test_iter.py \
		--data_path $DATA_PATH --slices $SLICES --organ_id $ORGAN_ID --folds $FOLDS -f $CURRENT_FOLD \
		-e $EPOCH --gpu_id $GPU_ID -t $TRAINING_TIMESTAMP
	python test_voxel.py \
		--data_path $DATA_PATH --slices $SLICES --organ_id $ORGAN_ID --folds $FOLDS -f $CURRENT_FOLD \
		-e $EPOCH --gpu_id $GPU_ID -t $TRAINING_TIMESTAMP
fi
