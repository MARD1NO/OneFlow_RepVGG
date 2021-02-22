rm -rf core.*
rm -rf ./output/snapshots/*

# NUM_EPOCH=120
NUM_EPOCH=160
echo NUM_EPOCH=$NUM_EPOCH

# You should CHECK THIS!!!
MODEL_SAVE_DIR=./repvggA0/snapshots/model_save/
echo MODEL_SAVE_DIR=$MODEL_SAVE_DIR

LOG_FOLDER=../logs
mkdir -p $LOG_FOLDER
# You should CHECK THIS!!!
LOGFILE=$LOG_FOLDER/repvggA0_trainingv2.log

# training with imagenet
if [ -n "$2" ]; then
    DATA_ROOT=$2
else
    DATA_ROOT=/DATA/disk1/ImageNet/ofrecord
fi
echo DATA_ROOT=$DATA_ROOT

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE

python3 of_cnn_train_val.py \
     --train_data_dir=$DATA_ROOT/train \
     --train_data_part_num=256 \
     --val_data_dir=$DATA_ROOT/validation \
     --val_data_part_num=256 \
     --num_nodes=1 \
     --gpu_num_per_node=4 \
     --optimizer="sgd" \
     --momentum=0.9 \
     --label_smoothing=0.1 \
     --lr_decay='cosine'\
     --learning_rate=0.1 \
     --loss_print_every_n_iter=100 \
     --batch_size_per_device=64 \
     --val_batch_size_per_device=64 \
     --channel_last=False \
     --fuse_bn_relu=True \
     --fuse_bn_add_relu=True \
     --nccl_fusion_threshold_mb=16 \
     --nccl_fusion_max_ops=24 \
     --gpu_image_decoder=True \
     --num_epoch=$NUM_EPOCH \
     --warmup_epochs=5 \
     --mixup=False \
     --model_save_dir=$MODEL_SAVE_DIR \
     --model="repvggA0" 2>&1 | tee ${LOGFILE}

echo "Writting log to ${LOGFILE}"
